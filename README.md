# Description

A solution for [GramEval2020](https://github.com/dialogue-evaluation/GramEval2020) competition.

## About the Model
The model is based on [DeepPavlov's RuBERT](http://docs.deeppavlov.ai/en/master/features/models/bert.html) and AllenNLP dependencies parser implementation (which is based on [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)](https://arxiv.org/abs/1611.01734)).

This is an end-to-end parser: the predictions for grammar values, lemmas and dependencies are made by a single model trained in a multi-task mode and they are not conditioned on each other.
It uses BERT embedder with a single layer LSTM encoder, simple feedforward predictors for grammar values and lemmas and biaffine attention predictors for dependencies and their labels.

## Try it!
Simply open it here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10aQbeZqibllgfpZkqBlIddDY426c4sNq?authuser=1#forceEdit=true&sandboxMode=true)

## Available Models
In the paper, lots of different models were trained and evaluated. Here they are:

| Model                         | POS            | MorphoFeats   | Lemma         | LAS            | Overall        |
|-------------------------------|----------------|---------------|---------------|----------------|----------------|
| chars                         | 94,7% / 91,7%  | 92,2% / 90,9% | 95,5% / 93,6% | 41,1% / 38,1%  | 80,9% / 78,6%  |
| chars_lstm                    | 97,2% / 94,1%  | 96,9% / 94,6% | 97,3% / 95,0% | 87,2% / 77,8%  | 94,7% / 90,4%  |
| chars_morph_lstm              | 97,5% / 94,4%  | 97,7% / 95,2% | 98,1% / 95,6% | 89,6% / 77,0%  | 95,7% / 90,6%  |
| frozen_elmo                | 97,4% / 95,4%  | 96,2% / 95,8% | 93,1% / 92,8% | 80,3% / 74,1%  | 91,8% / 89,5%  |
| frozen_elmo_lstm           | 97,9% / **95,9%**  | 97,5% / 95,9% | 97,0% / 95,3% | 88,9% / 80,3%  | 95,3% / 91,9%  |
| frozen_elmo_morph_lstm     | 97,8% / **95,7%**  | 97,7% / 96,1% | 97,3% / 95,3% | 89,5% / 79,6%  | 95,6% / 91,7%  |
| trainable_elmo                | 98,2% / 95,5%  | 97,8% / 95,8% | 98,2% / 95,9% | 91,5% / 79,7%  | 96,4% / 91,7%  |
| trainable_elmo_lstm           | 98,3% / **95,7%**  | 97,9% / 95,8% | 98,3% / 95,8% | 92,2% / 81,2%  | 96,7% / 92,1%  |
| frozen_bert                   | 96,0% / 94,0%  | 95,5% / 94,3% | 86,6% / 86,6% | 81,7% / 76,7%  | 89,9% / 87,9%  |
| frozen_bert_lstm              | 97,1% / 95,3%  | 96,6% / 95,1% | 92,3% / 91,0% | 86,8% / 82,0%  | 93,2% / 90,9%  |
| trainable_bert                | **98,4%** / 96,2%  | **98,3%** / **96,4%** | **98,6%** / **96,5%** | **93,1%** / **84,6%**  | **97,1%** / **93,4%**  |
| trainable_bert_lstm           | **98,6%** / **95,8%**  | **98,4%** / **96,3%** | **98,5%** / 96,2% | **93,2%** / 83,5%  | **97,2%** / 92,9%  |
| trainable_bert_morph_lstm     | **98,4%** / **95,9%**  | **98,4%** / **96,4%** | **98,5%** / **96,4%** | **93,3%** / 84,1%  | **97,2%** / **93,2%**  |

The qualities were calculated on the dev/test sets.

You can download any model running `./download_model.sh <model_name>`, where model_name is taken from the table.

**Attention** Those models were trained on all data excluding 17th century. To obtain the model that can work on that dataset too, use `ru_bert_final_model` model. This is the model that was actually used in the competition.

## Preparations
In order to reproduce it locally, you will need to follow these steps.

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

The later command will install a very specific version of the AllenNLP library with a simple patch (that allows to pass arguments to the tokenizer).

After that, download the data. You can simply run `./download_data.sh` on linux/mac. Otherwise, download the archive manually from Google Drive [here](https://drive.google.com/open?id=1bSZW3D7M1Gyv7W5Rl4ajiqDvyyK19iny).

It contains labeled data from the [official repository](https://github.com/dialogue-evaluation/GramEval2020) and [DeepPavlov's RuBERT](http://docs.deeppavlov.ai/en/master/features/models/bert.html) converted to the pytorch format using [transformers script](https://huggingface.co/transformers/converting_tensorflow_models.html#bert).

## Train
To train a model, run the following command:
```bash
cd solution
python -m train.main
```
It will use the [BERT-based model config](solution/train/configs/bert.json) and train it on the data from `data/data_train` folder.

## Apply
To apply a trained model from the previous step on the test data, use `train.applier` script.

You can also apply it on already trained model from my final submission: download the model weights: `./download_model.sh ru_bert_final_model` (or using this [link](https://drive.google.com/file/d/1RpWcC8PGkSO7eduW5DBCS4ygHKjpEC4P/view?usp=sharing)).

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
