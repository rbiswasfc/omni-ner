import argparse
import glob
import json
import uuid

import pandas as pd
from tqdm.auto import tqdm

KEEP_TH = 0.70  # keep the examples with at least this fraction of annotations found in the text
MIN_ANNOTATIONS = 1  # minimum number of annotations to keep the example
MAX_ANNOTATIONS = 64  # maximum number of annotations to keep the example


def generate_random_string():
    return str(uuid.uuid4())


def find_entity_positions(text, entity_text, lowercase=True):
    """
    Find and return all positions of an entity within a given text.
    """
    # avoid infinite loop for empty entity_text ---
    if not entity_text:
        return []

    entity_positions = []
    start_pos = 0

    if lowercase:
        text = text.lower()
        entity_text = entity_text.lower()

    while start_pos != -1:
        start_pos = text.find(entity_text, start_pos)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            entity_positions.append((start_pos, end_pos))
            start_pos = end_pos  # move past the last found occurrence

    return entity_positions


def deduplicate(entity_list):
    """
    Remove duplicate or fully contained entities from the list.
    """
    to_remove = set()

    for i, current_entity in enumerate(entity_list):
        if current_entity in to_remove:
            continue

        for other_entity in entity_list[i + 1:]:
            if (current_entity[0] >= other_entity[0]) and (current_entity[1] <= other_entity[1]):
                to_remove.add(current_entity)
                break

    entity_list = [ent for ent in entity_list if ent not in to_remove]

    return entity_list


def process_entities(text, entities, min_char_in_entity=2):
    """
    Process entities to find their positions in text, deduplicate, and format the results.
    """
    if not isinstance(text, str) or not isinstance(entities, list):
        raise ValueError("Invalid input types for text or entities")

    entity_list = []

    for entity_name, entity_type, entity_description in entities:
        if len(entity_name) < min_char_in_entity:
            continue

        entity_positions = find_entity_positions(text, entity_name)

        entity_list.extend(
            [(start, end, text[start:end], entity_type, entity_description) for start, end in entity_positions]
        )

    entity_list = deduplicate(entity_list)

    # format
    formatted_entities = [{
        "char_start": start,
        "char_end": end,
        "entity_name": entity_name,
        "entity_type": entity_type,
        "entity_description": entity_description
    } for start, end, entity_name, entity_type, entity_description in entity_list]

    return sorted(formatted_entities, key=lambda x: x["char_start"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    # read
    paths = glob.glob(f"{args.input_dir}/*/*.parquet")

    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        dfs.append(df)
    omni_df = pd.concat(dfs).reset_index(drop=True)

    print(f"# of texts annotated by LLM: {len(omni_df)}")

    # process --
    processed_examples = []
    processed_texts = set()

    pbar = tqdm(total=omni_df.shape[0])
    for idx, row in omni_df.iterrows():
        text = row["text"]
        annotations = row["entity_list"]

        if text in processed_texts:
            continue

        # max & min annotation number check ---
        num_annotations = len(annotations)
        if (num_annotations > MAX_ANNOTATIONS) or (num_annotations < MIN_ANNOTATIONS):
            # print(f"Skipping {idx} due to {num_annotations} annotations")
            continue

        # check validity of the label ---
        updated_annotations = []
        cnt_found = 0

        for ent, ent_t, ent_d in annotations:
            if ent.lower() in text.lower():
                cnt_found += 1

                updated_annotations.append([ent, ent_t, ent_d])

        found_fraction = cnt_found / num_annotations if num_annotations > 0 else 0

        if found_fraction >= KEEP_TH:
            updated_annotations = process_entities(text, updated_annotations)

            processed_examples.append(
                {"id": generate_random_string(), "text": text, "annotations": updated_annotations}
            )
            processed_texts.add(text)

        pbar.update(1)
    pbar.close()

    print(f"# of annotated texts after processing: {len(processed_texts)}")

    with open(f"{args.output_dir}/omni_ner_dataset.json", "w") as f:
        json.dump(processed_examples, f)

    print(f"Saved the processed dataset to {args.output_dir}/omni_ner_dataset.json")
    print("Done!")
