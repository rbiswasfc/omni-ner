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
    from utils.train_utils import AverageMeter, get_lr, setup_training_run

except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)

# -------- Evaluation -------------------------------------------------------------------#


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

    cfg.model.n_groups = len(entity_types)
    label2id = {k: i for i, k in enumerate(entity_types)}
    accelerator.print(f"# entity types: {len(entity_types)}")

    # convert labels to ids in the datasets ---
    def convert_to_ids(examples):
        labels = []
        for ex in examples:
            ex_labels = [label2id[l] for l in ex["labels"]]
            labels.append(ex_labels)
        return {"labels": labels}

    train_ds = train_ds.map(convert_to_ids, batched=True)
    valid_ds = valid_ds.map(convert_to_ids, batched=True)

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
        show_batch(b, tokenizer, task='training', print_fn=accelerator.print)
        if batch_idx >= 4:
            break
    print_line()

    # ------- Config --------------------------------------------------------------------#
    accelerator.print("config for the current run:")
    accelerator.print(json.dumps(cfg_dict, indent=4))
    print_line()

    # # ------- Model ---------------------------------------------------------------------#
    # print_line()
    # accelerator.print("creating the PII Data Detection model...")
    # num_labels = len(label2id)
    # model = DebertaForPII.from_pretrained(
    #     cfg.model.backbone_path,
    #     num_labels=num_labels,
    #     id2label=id2label,
    #     label2id=label2id,
    # )
    # print_line()

    # # ------- Optimizer -----------------------------------------------------------------#
    # print_line()
    # accelerator.print("creating the optimizer...")
    # optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)
    # # ------- Prepare -------------------------------------------------------------------#

    # model, optimizer, train_dl, valid_dl = accelerator.prepare(
    #     model, optimizer, train_dl, valid_dl
    # )

    # # ------- Scheduler -----------------------------------------------------------------#
    # print_line()
    # num_epochs = cfg.train_params.num_train_epochs
    # grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    # warmup_pct = cfg.train_params.warmup_pct

    # num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    # num_training_steps = num_epochs * num_update_steps_per_epoch
    # num_warmup_steps = int(warmup_pct*num_training_steps)

    # accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    # accelerator.print(f"# training steps: {num_training_steps}")
    # accelerator.print(f"# warmup steps: {num_warmup_steps}")

    # scheduler = get_custom_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )

    # # ------- training setup ------------------------------------------------------------#
    # best_lb = -1.  # higher is better
    # patience_tracker = 0
    # current_iteration = 0

    # # ------- training  -----------------------------------------------------------------#
    # start_time = time.time()
    # accelerator.wait_for_everyone()

    # for epoch in range(num_epochs):
    #     # close and reset progress bar
    #     if epoch != 0:
    #         progress_bar.close()

    #     progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
    #     loss_meter = AverageMeter()

    #     # Training ------
    #     model.train()
    #     for step, batch in enumerate(train_dl):
    #         with accelerator.accumulate(model):
    #             outputs = model(**batch)
    #             loss = outputs.loss
    #             accelerator.backward(loss)

    #             if accelerator.sync_gradients:
    #                 accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
    #                 optimizer.step()
    #                 scheduler.step()
    #                 optimizer.zero_grad()

    #             loss_meter.update(loss.item())

    #         if accelerator.sync_gradients:
    #             progress_bar.set_description(
    #                 f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
    #                 f"LR: {get_lr(optimizer):.4f}. "
    #                 f"Loss: {loss_meter.avg:.4f}. "
    #             )

    #             progress_bar.update(1)
    #             current_iteration += 1

    #             if cfg.use_wandb:
    #                 accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)
    #                 accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

    #         # ------- evaluation  -------------------------------------------------------#
    #         if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
    #             # set model in eval mode
    #             model.eval()
    #             eval_response = run_evaluation(cfg, accelerator, model, valid_dl, id2label, reference_df)

    #             scores_dict = eval_response["scores"]
    #             oof_df = eval_response["oof_df"]
    #             pred_df = eval_response["pred_df"]
    #             optim_outside_threshold = round(eval_response["best_threshold"], 2)
    #             optim_f5 = eval_response["best_f5"]
    #             lb = scores_dict["lb"]

    #             print_line()
    #             et = as_minutes(time.time()-start_time)
    #             accelerator.print(
    #                 f""">>> Epoch {epoch+1} | Step {current_iteration} | Time: {et} | Training Loss: {loss_meter.avg:.4f}"""
    #             )
    #             accelerator.print(f">>> Current LB (F5) = {round(lb, 4)}")
    #             accelerator.print(f">>> Optim F5@th=({optim_outside_threshold}) = {round(optim_f5, 4)}")

    #             print_line()

    #             granular_scores = scores_dict["ents_per_type"]

    #             accelerator.print(f">>> Granular Evaluation:")
    #             granular_scores = OrderedDict(sorted(granular_scores.items()))

    #             for k, v in granular_scores.items():
    #                 accelerator.print(
    #                     f"> [{k:<24}] P: {round(v['p'], 3):<8} | R: {round(v['r'], 3):<8} | F5: {round(v['f5'], 3):<8} |"
    #                 )

    #             print_line()

    #             accelerator.print(f">>> Threshold Curve Points:")
    #             for th_i, f5_i in eval_response["th_curve_points"]:
    #                 accelerator.print(f"> F5@{th_i} = {f5_i}")

    #             print_line()

    #             is_best = False
    #             if lb >= best_lb:
    #                 best_lb = lb
    #                 is_best = True
    #                 patience_tracker = 0

    #                 # -----
    #                 best_dict = dict()
    #                 for k, v in scores_dict.items():
    #                     best_dict[f"{k}_at_best"] = v
    #             else:
    #                 patience_tracker += 1

    #             if is_best:
    #                 oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_best.csv"), index=False)
    #                 pred_df.to_parquet(os.path.join(cfg.outputs.model_dir, f"pred_df_best.parquet"), index=False)
    #             else:
    #                 accelerator.print(f">>> patience reached {patience_tracker}/{cfg_dict['train_params']['patience']}")
    #                 accelerator.print(f">>> current best score: {round(best_lb, 4)}")

    #             oof_df.to_csv(os.path.join(cfg.outputs.model_dir, f"oof_df_last.csv"), index=False)
    #             pred_df.to_parquet(os.path.join(cfg.outputs.model_dir, f"pred_df_last.parquet"), index=False)

    #             # saving last & best checkpoints -----
    #             accelerator.wait_for_everyone()
    #             unwrapped_model = accelerator.unwrap_model(model)

    #             if cfg.save_model:
    #                 unwrapped_model.save_pretrained(
    #                     f"{cfg.outputs.model_dir}/last",
    #                     state_dict=accelerator.get_state_dict(model),
    #                     save_function=accelerator.save,
    #                 )

    #                 if accelerator.is_main_process:
    #                     tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

    #             # if is_best:
    #             #     unwrapped_model.save_pretrained(
    #             #         f"{cfg.outputs.model_dir}/best",
    #             #         state_dict=accelerator.get_state_dict(model),
    #             #         save_function=accelerator.save,
    #             #     )
    #             #     if accelerator.is_main_process:
    #             #         tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/best")

    #             # logging ----
    #             if cfg.use_wandb:
    #                 accelerator.log({"lb": lb}, step=current_iteration)
    #                 accelerator.log({"best_lb": best_lb}, step=current_iteration)

    #                 # -- log scores dict
    #                 accelerator.log(scores_dict, step=current_iteration)

    #             # -- post eval
    #             model.train()
    #             torch.cuda.empty_cache()
    #             print_line()

    #             # early stopping ----
    #             if patience_tracker >= cfg.train_params.patience:
    #                 print("stopping early")
    #                 model.eval()
    #                 accelerator.end_training()
    #                 return


if __name__ == "__main__":
    run_training()
