# -*- coding: utf-8 -*-

import argparse
import os
import pymorphy2

from collections import defaultdict


_LEFT_PUNCT = {
    '«', '('
}

_RIGHT_PUNCT = {
    '»', ')', ',', '.'
}

_MORPH = pymorphy2.MorphAnalyzer()



def get_pos(token):
    return {parse.tag.POS for parse in _MORPH.parse(token)}


def fix_verb(token):
    if len(get_pos(token) & {'GRND', 'PRTF'}) == 0:
        return token

    for parse in _MORPH.parse(token):
        for parse in parse.lexeme:
            if parse.tag.POS == 'INFN':
                return parse.word

    return token


def read_predictions(input_path, prediction_path):
    predictions = defaultdict(lambda: defaultdict(list))
    with open(input_path) as f1, open(prediction_path) as f2:
        for line1, line2 in zip(f1, f2):
            line1, line2 = line1[:-1], line2[:-1]

            if not line1:
                continue

            index1, word1, lemma_id, _, _, suffix = line1.split('\t', 5)
            assert suffix.strip() == ''
            suffix = ' ' if suffix else ''

            index2, word2, lemma = line2.split('\t')[:3]
            assert index1 == index2
            assert word1 == word2

            if lemma_id:
                data_type, file_id, span_id = lemma_id.split(',')[:3]
                predictions[(data_type, file_id)][int(span_id)].append((lemma, suffix, index1))

    return predictions


def get_span_text(lemmas, data_type):
    tokens = []
    for i in range(len(lemmas)):
        lemma, suffix, token_index = lemmas[i]
        next_token_index = int(lemmas[i + 1][2]) if i + 1 < len(lemmas) else None
        if int(token_index) + 1 != next_token_index:
            suffix = ' '

        if lemma in _LEFT_PUNCT:
            suffix = ''

        if lemma in _RIGHT_PUNCT and tokens:
            tokens[-1] = ''

        tokens.append(lemma)
        tokens.append(suffix)

    result = ''.join(tokens).strip()

    if len(result.split()) == 1 and 'generic' in data_type:
        result = fix_verb(result)

    return result


def write_predictions(predictions, prediction_path):
    output_dir = os.path.dirname(prediction_path)
    for (data_type, file_id), spans in predictions.items():
        dir_path = os.path.join(output_dir, data_type.replace('texts_and_ann', ''))
        os.makedirs(dir_path, exist_ok=True)
        with open(f'{dir_path}/{file_id}.norm', 'w') as f:
            for _, lemmas in sorted(spans.items(), key=lambda kv: kv[0]):
                f.write(get_span_text(lemmas, data_type) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path')
    parser.add_argument('--prediction-path')
    args = parser.parse_args()

    predictions = read_predictions(args.input_path, args.prediction_path)
    write_predictions(predictions, args.prediction_path)


if __name__ == "__main__":
    main()
