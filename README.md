# Omni-Ner

Entity recognition is a widely used information extraction task, yet publicly available foundation models are not well suited for it. We leverage modern LLMs to create a small-yet-powerful foundation model for this task. This BERT-size model can be used to create custom entity recognizers with typically 5x less annotated data than before. This model is powering NuMind and we open-source it with an MIT license for everyone to use. Spread the word!

Building a foundation model for NER tasks


# Omni-NER Dataset

# Omni-NER Model


# Create Arc Dataset
```bash
python ./omni-ner/code/create_arc_dataset.py --config_path ./omni-ner/conf/r_arc/conf_r_arc.yaml
```

# Train Arc Model
```bash
accelerate launch ./omni-ner/code/train_arc.py \
--config-name conf_r_arc \
seed=42 \
train_params.eval_frequency=100 \
use_wandb=false
```