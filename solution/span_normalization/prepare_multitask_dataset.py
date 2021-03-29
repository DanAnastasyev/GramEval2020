# -*- coding: utf-8 -*-

import os


def prepare_dataset(data_type):
    if data_type == 'train':
        raw_path = '../data/span_normalization/data_train/train.conllu'
        out_path = '../data/span_normalization/multitask/data_train/train.conllu'
        os.makedirs('../data/span_normalization/multitask/data_train/', exist_ok=True)
    else:
        raw_path = '../data/span_normalization/data_open_test/valid.conllu'
        out_path = '../data/span_normalization/multitask/data_open_test/valid.conllu'
        os.makedirs('../data/span_normalization/multitask/data_open_test/', exist_ok=True)

    parsed_path = f'../predictions/span_normalization/multitask_train/multitask_bert_2/{data_type}.conllu'

    with open(raw_path) as f1, open(parsed_path) as f2, open(out_path, 'w') as f_out:
        for line1, line2 in zip(f1, f2):
            line1, line2 = line1.strip(), line2.strip()
            if not line1:
                print(file=f_out)
                continue

            fields1, fields2 = line1.split('\t'), line2.split('\t')
            index, word1, lemma1, span_tag, dataset_tag = fields1[:5]
            _, word2, lemma2, pos, _, feats, head_index, head_tag = fields2[:8]

            assert word1 == word2

            lemma = lemma1 if lemma1 != '_' else lemma2
            if word1 != '_':
                assert lemma1 == '_' and span_tag == 'O' or lemma1 != '_' and span_tag in ('B', 'I'), line1

            print(index, word1, lemma1, span_tag, dataset_tag, pos, feats, head_index, head_tag, sep='\t', file=f_out)


def main():
    prepare_dataset('train')
    prepare_dataset('valid')


if __name__ == '__main__':
    main()
