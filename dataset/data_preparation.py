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


def process_example(example, tokenizer):
    # assumption: no overlapping annotations ---
    num_annotations = len(example['annotations'])
    sorted_annotations = sorted(example['annotations'], key=lambda x: x['char_start'])

    tx = tokenizer(
        example['text'],
        return_offsets_mapping=True,
        is_split_into_words=False,
        add_special_tokens=True,
    )

    num_tokens = len(tx['input_ids'])

    # initialize two pointers
    token_idx = 0
    annotation_idx = 0
    current_span = []

    while token_idx < num_tokens and annotation_idx < num_annotations:
        token_char_start, token_char_end = tx['offset_mapping'][token_idx]

        # skip special tokens
        if token_char_start == token_char_end:
            token_idx += 1
            continue

        annotation = sorted_annotations[annotation_idx]
        annotation_char_start = annotation['char_start']
        annotation_char_end = annotation['char_end']

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
                annotation['token_idxs'] = current_span
                current_span = []
            annotation_idx += 1

        if pos_type != 'after_entity':
            token_idx += 1

    # handle any remaining tokens or annotations after exiting the loop
    if current_span:
        if annotation_idx < num_annotations:
            sorted_annotations[annotation_idx]['token_idxs'] = current_span

    example['annotations'] = sorted_annotations
    return example


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    example = {
        'text': "I am a happy",
        'annotations': [
            {'char_start': 0, 'char_end': 1, 'label': 'A'},
            {'char_start': 2, 'char_end': 3, 'label': 'B'},
            {'char_start': 4, 'char_end': 5, 'label': 'C'},
        ],
    }
    process_example(example, tokenizer)
