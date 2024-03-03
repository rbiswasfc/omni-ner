from copy import deepcopy

from datasets import Dataset
from transformers import AutoTokenizer


def detect_token_annotation_overlap(
    token_char_start,
    token_char_end,
    annotation_char_start,
    annotation_char_end,
):
    if token_char_end <= annotation_char_start:
        return 'before_entity'

    elif token_char_start >= annotation_char_end:
        return 'after_entity'

    elif (token_char_start <= annotation_char_start) and (token_char_end <= annotation_char_end):
        return 'start_entity'

    elif (token_char_start > annotation_char_start) and (token_char_end <= annotation_char_end):
        return 'in_entity'

    elif (token_char_start < annotation_char_end) and (token_char_end > annotation_char_end):
        return 'end_entity'

    else:
        raise ValueError


def process_annotations(offsets, annotations):
    num_tokens, num_annotations = len(offsets), len(annotations)
    sorted_annotations = sorted(annotations, key=lambda x: x['char_start'])

    # pointers ---
    token_idx = 0
    annotation_idx = 0

    # fields ---
    heads = []
    tails = []
    labels = []

    # span buffer ---
    current_span = []

    # loop ---
    while (token_idx < num_tokens) and (annotation_idx < num_annotations):
        token_char_start, token_char_end = offsets[token_idx]

        # skip special tokens ---
        if token_char_start == token_char_end:
            token_idx += 1
            continue

        # current entity
        this_annotation = sorted_annotations[annotation_idx]
        annotation_char_start = this_annotation['char_start']
        annotation_char_end = this_annotation['char_end']

        # detect current token and current annotation alignment ---
        pos_type = detect_token_annotation_overlap(
            token_char_start,
            token_char_end,
            annotation_char_start,
            annotation_char_end,
        )

        if pos_type in ['start_entity', 'in_entity', 'end_entity']:
            current_span.append(token_idx)

        if (pos_type == 'end_entity') or (pos_type == 'after_entity'):
            if current_span:
                heads.append(current_span[0])
                tails.append(current_span[-1] + 1)
                labels.append(this_annotation['entity_type'])

                # reset buffer ---
                current_span = []

            # move to next annotation ---
            annotation_idx += 1

        if pos_type != 'after_entity':
            token_idx += 1

    # handle any remaining tokens or annotations after exiting the loop
    if current_span:
        if annotation_idx < num_annotations:
            heads.append(current_span[0])
            tails.append(current_span[-1] + 1)
            labels.append(annotations[annotation_idx]['entity_type'])

    return heads, tails, labels


class OmniNERDataset:
    """
    Dataset class for omni-ner
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone_path, use_fast=True)

    def process_inputs(self, examples):
        to_return = dict()

        tx = self.tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
            return_offsets_mapping=True,
            is_split_into_words=False,
        )

        # ------
        span_head_idxs = []
        span_tail_idxs = []
        labels = []

        # ---
        for eidx in range(len(tx['input_ids'])):
            offsets = tx['offset_mapping'][eidx]
            annotations = examples['annotations'][eidx]
            ex_heads, ex_tails, ex_labels = process_annotations(offsets, annotations)
            span_head_idxs.append(ex_heads)
            span_tail_idxs.append(ex_tails)
            labels.append(ex_labels)

        # ---
        tx['span_head_idxs'] = span_head_idxs
        tx['span_tail_idxs'] = span_tail_idxs
        tx['labels'] = labels

        return tx

    def get_dataset(self, examples):

        dataset_dict = {
            "id": [x["id"] for x in examples],
            "text": [x["text"] for x in examples],
            "annotations": [x["annotations"] for x in examples],
        }

        task_dataset = Dataset.from_dict(dataset_dict)

        task_dataset = task_dataset.map(
            self.process_inputs,
            batched=True,
            batch_size=512,
            num_proc=self.cfg.model.num_proc,
            remove_columns=task_dataset.column_names,
        )

        return task_dataset
