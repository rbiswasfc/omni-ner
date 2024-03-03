import argparse
import json
import os
import random
import uuid
from copy import deepcopy

from datasets import concatenate_datasets, save_to_disk
from omegaconf import OmegaConf

try:
    from r_arc.omni_dataset import OmniNERDataset
except Exception as e:
    print(e)
    raise ImportError


def generate_random_string():
    return str(uuid.uuid4())


def get_meta_examples(cfg, examples):
    examples = deepcopy(examples)

    # create meta dataset ---
    entity_dict = dict()

    for ex in examples:
        entities = ex['annotations']
        for ent in entities:
            ent_t = ent['entity_type']
            ent_d = ent['entity_description']

            if ent_t not in entity_dict:
                entity_dict[ent_t] = set()
            entity_dict[ent_t].add(ent_d)

    for k, v in entity_dict.items():
        entity_dict[k] = list(v)

    # downsample ---
    n_keep = cfg.data.n_descriptions_per_entity

    processed_dict = dict()
    for k, v in entity_dict.items():
        if len(v) >= n_keep:
            v = random.sample(v, k=n_keep)
        processed_dict[k] = v

    # ---
    entity_meta_examples = []

    for k, v in processed_dict.items():
        for vi in v:
            text = f"{k} | {vi}"
            annotations = [
                {
                    'char_start': 0,
                    'char_end': len(k),
                    'entity_name': k,
                    'entity_type': k,
                    'entity_description': vi,
                }
            ]

            ex = {'id': generate_random_string(), 'text': text, 'annotations': annotations}

            entity_meta_examples.append(ex)
    return entity_meta_examples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="input file")
    ap.add_argument("--config_path", required=True, help="config path")
    ap.add_argument("--output_dir", required=True, help="output file")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_path, "r") as f:
        examples = json.load(f)
    n_examples = len(examples)

    print(f"# of examples in omni-ner dataset: {len(examples)}")

    cfg = OmegaConf.load(args.config_path)

    # train - test split ---
    test_frac = cfg.data.test_fraction
    num_test = int(n_examples * test_frac)
    random.shuffle(examples)
    test_examples = examples[:num_test]
    train_examples = examples[num_test:]

    # create train dataset ---
    dataset_creator = OmniNERDataset(cfg)

    train_ds = dataset_creator.get_dataset(train_examples)
    train_meta_examples = get_meta_examples(cfg, train_examples)
    train_ds_meta = dataset_creator.get_dataset(train_meta_examples)
    final_train_ds = [train_ds] * cfg.data.content_multiplier + [train_ds_meta]
    final_train_ds = concatenate_datasets(final_train_ds).shuffle(seed=42)

    # create test dataset ---
    test_ds = dataset_creator.get_dataset(test_examples)
    test_meta_examples = get_meta_examples(cfg, test_examples)
    test_ds_meta = dataset_creator.get_dataset(test_meta_examples)
    final_test_ds = [test_ds, test_ds_meta]
    final_test_ds = concatenate_datasets(final_test_ds).shuffle(seed=42)

    # save datasets ---
    save_to_disk(final_train_ds, os.path.join(args.output_dir, "train"))
    save_to_disk(final_test_ds, os.path.join(args.output_dir, "test"))

    print(f"Saved Omni-NER train & test Datasets to {args.output_dir}")
    print(f"# of examples in final train dataset: {len(final_train_ds)}")
    print(f"# of examples in final test dataset: {len(final_test_ds)}")

    print(f"Example: {final_test_ds[0]}")
    print("Done!")
