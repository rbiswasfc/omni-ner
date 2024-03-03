hdir=$(pwd)
cd ..

mkdir datasets

cd datasets

kaggle datasets download -d conjuring92/omni-ner-dataset-v1
unzip omni-ner-dataset-v1.zip -d ./omni_ner_dataset_v1
rm omni-ner-dataset-v1.zip

cd $hdir