# -*- coding: utf-8 -*-

import argparse
import logging
import os
import stanfordnlp

from tqdm import tqdm
from train.corpus_iterator import CorpusIterator


def _parse_file(input_file, output_file, parser):
    with CorpusIterator(path=input_file, lemma_col_index=None, grammar_val_col_indices=None) as iterator:
        with open(output_file, 'w') as f_out:
            for sentence in tqdm(iterator, desc=input_file):
                tokens = [token.text for token in sentence]
                result = parser([tokens])

                assert len(result.sentences) == 1
                assert len(result.sentences[0].words) == len(tokens)

                for word in result.sentences[0].words:
                    # TODO: check Nones
                    # ru_taiga-ud-dev-social.conllu: Подпишись на нас чтоб не пропустить новый рецепт 👉 @screened-208 _ @screened-209 ________________________ #кухня #рецепты #еда #дети #лайфхак #посуда #работа #праздничноеменю #мебель #кулинария #мама #питер #nl #осетия #сервировка #вкусно #новыйгод #краснодар #хендмейд #крым #мамскийблог #ростов #детскоеменю #москва #сочи #подмосковье #сервировка #своимируками #соленье #заготовки
                    # token 10
                    print(word.index, word.text, word.lemma, word.upos, '_', word.feats,
                          word.governor, word.dependency_relation, '_', '_', sep='\t', file=f_out)
                print(file=f_out)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)

    args = parser.parse_args()

    parser = stanfordnlp.Pipeline(lang='ru', models_dir='/home/dan-anastasev/Documents/stanfordnlp_resources',
                                  tokenize_pretokenized=True)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    for file_name in os.listdir(args.input_dir):
        _parse_file(os.path.join(args.input_dir, file_name), os.path.join(args.output_dir, file_name), parser)


if __name__ == "__main__":
    main()
