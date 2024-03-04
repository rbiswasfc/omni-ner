import json
import os
import time
from collections import OrderedDict

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from datasets import load_from_disk
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from r_arc.omni_dataset import OmniNERDataset
    from r_arc.omni_loader import OmniNERCollator, show_batch
    from r_arc.omni_model import OnmiNERModel
    from r_arc.omni_optimizer import get_optimizer
    from utils.train_utils import (AverageMeter, as_minutes,
                                   get_custom_cosine_schedule_with_warmup,
                                   get_lr, setup_training_run)

except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)

# -------- Evaluation -------------------------------------------------------------------#


def run_evaluation(accelerator, model, valid_dl):
    model.eval()

    all_losses = []
    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for _, batch in enumerate(valid_dl):
        with torch.no_grad():
            loss = model(**batch)

        batch_losses = accelerator.gather_for_metrics(loss)
        batch_losses = batch_losses.cpu().numpy().tolist()

        all_losses.extend(batch_losses)
        progress_bar.update(1)
    progress_bar.close()

    # compute metric
    eval_dict = dict()
    eval_dict['valid_loss'] = np.mean(all_losses)
    eval_dict['lb'] = np.mean(all_losses)

    return eval_dict


# -------- Main Function ----------------------------------------------------------------#


@hydra.main(version_base=None, config_path="../conf/r_arc", config_name="conf_r_arc")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit*50 + suffix)

    # ------- load data -----------------------------------------------------------------#
    print_line()

    with open(cfg.input_path, "r") as f:
        examples = json.load(f)
    accelerator.print(f"# examples: {len(examples)}")

    # ------- Datasets ------------------------------------------------------------------#
    print_line()
    accelerator.print("loading the train and test dataset...")
    train_ds = load_from_disk(os.path.join(cfg.data.output_dir, 'train'))
    valid_ds = load_from_disk(os.path.join(cfg.data.output_dir, 'test'))

    accelerator.print(f"# examples in train data: {len(train_ds)}")
    accelerator.print(f"# examples in valid data: {len(valid_ds)}")

    # tokenizer ---
    with accelerator.main_process_first():
        dataset_creator = OmniNERDataset(cfg)
    tokenizer = dataset_creator.tokenizer

    # add arcface labels ---
    entity_types = set()
    for ex in examples:
        for anno in ex["annotations"]:
            entity_types.add(anno["entity_type"])
    entity_types = sorted(list(entity_types))

    cfg.model.arcface.n_groups = len(entity_types)
    label2id = {k: i for i, k in enumerate(entity_types)}
    id2label = {v: k for k, v in label2id.items()}
    accelerator.print(f"# entity types: {len(entity_types)}")

    # convert labels to ids in the datasets ---
    print_line()

    def convert_to_ids(examples):
        labels = []
        for ex_labels in examples['labels']:
            ex_labels = [label2id[l] for l in ex_labels]
            labels.append(ex_labels)
        return {"labels": labels}

    # print(train_ds[0]['labels'])

    train_ds = train_ds.map(convert_to_ids, batched=True)
    valid_ds = valid_ds.map(convert_to_ids, batched=True)

    print_line()

    # ------- data loaders --------------------------------------------------------------#
    data_collector = OmniNERCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collector,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collector,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch --------------------------------------------------------------------#
    for batch_idx, b in enumerate(train_dl):
        show_batch(b, tokenizer, id2label, task='training', print_fn=accelerator.print)
        if batch_idx >= 4:
            break
    print_line()

    # ------- Config --------------------------------------------------------------------#
    accelerator.print("config for the current run:")
    accelerator.print(json.dumps(cfg_dict, indent=4))
    print_line()

    # ------- Model ---------------------------------------------------------------------#
    print_line()
    accelerator.print("creating the OnmiNERModel...")
    model = OnmiNERModel(cfg)
    print_line()

    # # ------- Optimizer -----------------------------------------------------------------#
    print_line()
    accelerator.print("creating the optimizer...")
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)
    print_line()

    # # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ------- training setup ------------------------------------------------------------#
    best_lb = 1e4  # lower is better
    patience_tracker = 0
    current_iteration = 0

    # # ------- training  -----------------------------------------------------------------#
    start_time = time.time()
    accelerator.wait_for_everyone()

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # training ------
        model.train()
        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                loss = model(**batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # ------- evaluation  -------------------------------------------------------#
            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                # set model in eval mode
                model.eval()
                eval_dict = run_evaluation(accelerator, model, valid_dl)
                lb = eval_dict["lb"]

                print_line()
                et = as_minutes(time.time()-start_time)
                accelerator.print(
                    f""">>> Epoch {epoch+1} | Step {current_iteration} | Time: {et} | Training Loss: {loss_meter.avg:.4f}"""
                )
                accelerator.print(f">>> Current LB (valid loss) = {round(lb, 4)}")
                print_line()

                is_best = False
                if lb <= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                    # -----
                    best_dict = dict()
                    for k, v in eval_dict.items():
                        best_dict[f"{k}_at_best"] = v
                else:
                    patience_tracker += 1

                if is_best:
                    pass
                else:
                    accelerator.print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
                    accelerator.print(f">>> current best valid loss: {round(best_lb, 4)}")

                # saving last & best checkpoints -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                if cfg.save_model:
                    # save only the backbone ---
                    unwrapped_model.backbone.save_pretrained(
                        f"{cfg.outputs.model_dir}/last",
                        # state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                # logging ----
                if cfg.use_wandb:
                    accelerator.log({"lb": lb}, step=current_iteration)
                    accelerator.log({"best_lb": best_lb}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg.train_params.patience:
                    print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return


if __name__ == "__main__":
    run_training()
