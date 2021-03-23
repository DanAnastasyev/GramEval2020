# -*- coding: utf-8 -*-

import argparse
import os

GRAMEVAL_DIR = '/home/dan-anastasev/Documents/junk/GramEval2020'


def read_results(model_name):
    with open(os.path.join(GRAMEVAL_DIR, 'data/span_normalization/data_open_test/valid.conllu')) as f:
        data = [[]]
        for line in f:
            line = line[:-1]
            if not line:
                data.append([])
                continue
            word, lemma, label = line.split('\t')[1:4]
            data[-1].append((word, lemma, label))

    with open(os.path.join(GRAMEVAL_DIR, f'predictions/span_normalization/{model_name}/valid.conllu')) as f:
        predictions = [[]]
        for line in f:
            line = line[:-1]
            if not line:
                predictions.append([])
                continue
            word, lemma = line.split('\t')[1:3]
            predictions[-1].append((word, lemma))

    return data, predictions


def evaluate(data, predictions):
    correct_count = 0
    for sample, predicted_sample in zip(data, predictions):
        is_correct = True
        for (word, lemma, _), (_, predicted_lemma) in zip(sample, predicted_sample):
            if lemma != '_' and lemma.replace('ё', 'е') != predicted_lemma.replace('ё', 'е'):
                is_correct = False
        correct_count += int(is_correct)

        if not is_correct:
            for (word, lemma, label), (_, predicted_lemma) in zip(sample, predicted_sample):
                print(word, lemma, predicted_lemma,
                     label + (' !!!' if lemma != '_' and lemma != predicted_lemma else ''), sep='\t')
            print()

    print('Accuracy: {:.2%}'.format(correct_count / len(data)))


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-name', default='model')

    args = parser.parse_args()

    data, predictions = read_results(args.model_name)
    evaluate(data, predictions)


if __name__ == '__main__':
    main()
