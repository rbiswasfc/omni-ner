from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.tokenization_utils_base import (PaddingStrategy,
                                                  PreTrainedTokenizerBase)


@dataclass
class OmniNERCollator:
    """
    Data collator for omni-ner task
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        # labels ---
        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        span_head_idxs = [feature["span_head_idxs"] for feature in features]
        span_tail_idxs = [feature["span_tail_idxs"] for feature in features]

        features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            } for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,  # "pt",
        )

        batch['span_head_idxs'] = span_head_idxs
        batch['span_tail_idxs'] = span_tail_idxs

        # padding of spans ---
        max_num_spans = max([len(ex_spans) for ex_spans in span_head_idxs])
        max_seq_length = len(batch["input_ids"][0])
        default_head_idx = max_seq_length - 1  # for padding
        default_tail_idx = max_seq_length  # for padding

        span_head_idxs = [idxs + [default_head_idx] * (max_num_spans - len(idxs)) for idxs in span_head_idxs]
        span_tail_idxs = [idxs + [default_tail_idx] * (max_num_spans - len(idxs)) for idxs in span_tail_idxs]
        batch['span_head_idxs'] = span_head_idxs
        batch['span_tail_idxs'] = span_tail_idxs

        if labels is not None:
            labels = [ex_labels + [-1] * (max_num_spans - len(ex_labels)) for ex_labels in labels]  # -1 for padding
            batch["labels"] = labels

        # cast to tensor ---
        tensor_keys = [
            "input_ids",
            "attention_mask",
            "span_head_idxs",
            "span_tail_idxs",
        ]

        for key in tensor_keys:
            batch[key] = torch.tensor(batch[key], dtype=torch.int64)

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int64)

        return batch


# -----


def show_batch(batch, tokenizer, id2label, num_examples=8, task="train", print_fn=print):

    print_fn('=='*40)
    num_examples = min(num_examples, len(batch['input_ids']))
    print_fn(f"Showing {num_examples} from a {task} batch...")

    for i in range(num_examples):
        print_fn('#---' + f" Example: {i+1}" + '---' * 40 + '#')

        text = tokenizer.decode(batch['input_ids'][i], skip_special_tokens=False)
        print_fn("INPUT:\n")
        print_fn(text)

        print_fn("--"*20)
        num_spans = len(batch['span_head_idxs'][i])

        if "labels" in batch:
            labels = batch['labels'][i]
            labels = [id2label.get(label, 'NA') for label in labels]

        for span_idx in range(num_spans):
            start, end = batch['span_head_idxs'][i][span_idx], batch['span_tail_idxs'][i][span_idx]
            span = tokenizer.decode(batch['input_ids'][i][start:end])
            if "infer" not in task.lower():
                print_fn(f"[Entity {span_idx+1}]: {span} -> {labels[span_idx]}")
            else:
                print_fn(f"[Entity {span_idx+1}]: {'-'*len(span)}")

        print_fn('=='*40)
