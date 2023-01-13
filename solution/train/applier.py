# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import torch

from tqdm import tqdm

from allennlp.data.vocabulary import Vocabulary

from train.main import Config, _build_model, _get_reader
from train.lemmatize_helper import LemmatizeHelper
from train.morpho_vectorizer import MorphoVectorizer

from joblib import parallel_backend

logger = logging.getLogger(__name__)

BERT_MAX_LENGTH = 512


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model\'s name (the name of directory with the trained model)')
    parser.add_argument(
        '--pretrained-models-dir', default=None, help='Path to directory with pretrained models (e.g., RuBERT)'
    )
    parser.add_argument('--models-dir', default='../models', help='Path to directory where the models are stored')
    parser.add_argument(
        '--data-dir', default='../data/test_private_data', help='Path to directory with files to apply the model to'
    )
    parser.add_argument(
        '--predictions-dir', default='../predictions/private', help='Path to directory to store the predictions'
    )
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--checkpoint-name', default='best.th', help='Name of the checkpoint to use')
    args = parser.parse_args()

    model_dir = os.path.join(args.models_dir, args.model_name)
    result_data_dir = os.path.join(args.predictions_dir, args.model_name)

    if not os.path.isdir(result_data_dir):
        os.makedirs(result_data_dir)

    config = Config.load(os.path.join(model_dir, 'config.json'))

    if args.models_dir:
        config.data.models_dir = args.models_dir
    if args.pretrained_models_dir:
        config.data.pretrained_models_dir = args.pretrained_models_dir

    logger.info('Config: %s', config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    vocab = Vocabulary.from_files(os.path.join(model_dir, 'vocab'))
    lemmatize_helper = LemmatizeHelper.load(model_dir)
    morpho_vectorizer = MorphoVectorizer() if config.embedder.use_pymorphy else None

    model = _build_model(config, vocab, lemmatize_helper, morpho_vectorizer, bert_max_length=BERT_MAX_LENGTH)
    model.to(device)

    model.load_state_dict(torch.load(os.path.join(model_dir, args.checkpoint_name), map_location=device))
    model.eval()

    reader = _get_reader(config, skip_labels=True, bert_max_length=BERT_MAX_LENGTH, reader_max_length=None)

    for path in os.listdir(args.data_dir):
        if not path.endswith('.conllu'):
            continue

        data = reader.read(os.path.join(args.data_dir, path))

        if morpho_vectorizer is not None:
            morpho_vectorizer.apply_to_instances(data)

        with open(os.path.join(result_data_dir, path), 'w') as f_out:
            # Use parallel threading to speed up saving process
            with parallel_backend('threading', n_jobs=30):
                i = 0
                for begin_index in tqdm(range(0, len(data), args.batch_size)):
                    end_index = min(len(data), begin_index + args.batch_size)
                    predictions_list = model.forward_on_instances(data[begin_index: end_index])
                    for predictions in predictions_list:
                        # Save sentence's id to file
                        print(f"# sent_id = s{i}", file=f_out)
                        # Save full text in one line
                        full_text = " ".join([word for word in predictions['words']])
                        print(f"# text = {full_text}", file=f_out)
                        for token_index in range(len(predictions['words'])):
                            word = predictions['words'][token_index]
                            lemma = predictions['predicted_lemmas'][token_index]
                            upos, feats = predictions['predicted_gram_vals'][token_index].split('|', 1)
                            head_tag = predictions['predicted_dependencies'][token_index]
                            head_index = predictions['predicted_heads'][token_index]

                            print(token_index + 1, word, lemma, upos, '_', feats,
                                head_index, head_tag, '_', '_', sep='\t', file=f_out)
                        print(file=f_out)
                        i += 1


if __name__ == "__main__":
    main()
