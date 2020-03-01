# Description

A solution for [GramEval2020](https://github.com/dialogue-evaluation/GramEval2020) competition.

## Preparations
Firstly, clone the repo:
```bash
git clone https://github.com/DanAnastasyev/GramEval2020.git
cd GramEval2020/
```

Then, install the dependencies. It's reasonable to use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) or a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). E.g., run:
```bash
python3 -m venv solution-env
source solution-env/bin/activate
```

To install the dependencies run:
```bash
pip install -r requirements.txt
pip install git+git://github.com/DanAnastasyev/allennlp.git
```

The later command will install a very specific version of the allennlp library with a simple patch (that allows to pass arguments to the tokenizer).

After that, download the data. You can simply run `./download_data.sh` on linux/mac. Otherwise, download the archive manually from Google Drive [here](https://drive.google.com/open?id=1bSZW3D7M1Gyv7W5Rl4ajiqDvyyK19iny).

It contains labeled data from the [official repository](https://github.com/dialogue-evaluation/GramEval2020) and [DeepPavlov's RuBERT](http://docs.deeppavlov.ai/en/master/features/models/bert.html) converted to the pytorch format using [transformers script](https://huggingface.co/transformers/converting_tensorflow_models.html#bert).

## Train
To train a model, run the following command:
```bash
cd solution
python -m train.main
```
It will use the BERT-based model config and train it on the data from `data/data_train` folder.

## Apply
To apply a trained model from the previous step on the test data, use `train.applier` script.

You can also apply it on already trained model from my final submission: download the model weights: `./download_model.sh` (or using this [link](https://drive.google.com/open?id=1XiCj0OXZtBfxKyhMEPodt6wp2Jpc3RI9)).

And run:
```bash
cd solution
python -m train.applier --model-name ru_bert_final_model --batch-size 8
```

The applier script allows to specify the following parameters:
```
  --model-name MODEL_NAME   Model's name (the name of directory with the trained model)
  --pretrained-models-dir PRETRAINED_MODELS_DIR   Path to directory with pretrained models (e.g., RuBERT)
  --models-dir MODELS_DIR   Path to directory where the models are stored
  --data-dir DATA_DIR   Path to directory with files to apply the model to
  --predictions-dir PREDICTIONS_DIR   Path to directory to store the predictions
  --batch-size BATCH_SIZE
  --checkpoint-name CHECKPOINT_NAME   Name of the checkpoint to use
```

Which means that if you trained a new model with `python -m train.main`, you will need to pass `--model-name ru_bert_parser_with_tagger` because the model in the default config has this name.
