# Omni-Ner

Entity recognition is a widely used information extraction task, yet publicly available foundation models are not well suited for it. We leverage modern LLMs to create a small-yet-powerful foundation model for this task. This BERT-size model can be used to create custom entity recognizers with typically 5x less annotated data than before. This model is powering NuMind and we open-source it with an MIT license for everyone to use. Spread the word!

Building a foundation model for NER tasks


# Omni-NER Dataset

# Omni-NER Model


# Create Arc Dataset
```python
python create_arc_dataset.py \
--input_path ../data/omni_ner_dataset.json \
--config_path ../conf/r_arc/conf_r_arc.yaml \
--output_dir ../data/arc
```