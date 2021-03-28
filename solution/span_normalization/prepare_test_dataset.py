# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from prepare_dataset import sentenize_text, collect_span_tokens, remove_intersections


def read_file(data_type, file_id):
    with open(f'../data/span_normalization/raw/{data_type}/{file_id}.txt') as f:
        text = f.readlines()
        assert len(text) == 1
        text = text[0].rstrip()

    with open(f'../data/span_normalization/raw/{data_type}/{file_id}.ann') as f:
        spans = [list(map(int, line.strip().split())) for line in f]

    spans = [
        [(span[i], span[i + 1]) for i in range(0, len(span), 2)]
        for span in spans
    ]
    for span in spans:
        assert sorted(span) == span, span

    spans = [sorted(set(span)) for span in spans]

    return text, spans


def process_file(data_type, file_id):
    text, spans = read_file(data_type, file_id)
    sentence_tokens = sentenize_text(text)

    spans = list(zip(range(len(spans)), spans))

    for spans in remove_intersections(spans):
        spans_token_ids = []
        for _, span in spans:
            spans_token_ids.append(collect_span_tokens(sentence_tokens, span, file_id))

        for sent_index, sentence in enumerate(sentence_tokens):
            tokens = [token.text for token in sentence]
            labels = ['O' for _ in sentence]
            indices = ['' for _ in sentence]
            spaces = []
            for token_index in range(len(sentence)):
                if token_index + 1 != len(sentence):
                    spaces.append(text[sentence[token_index].stop: sentence[token_index + 1].start])
                else:
                    spaces.append('')

            for span_token_ids, (span_idx, _) in zip(spans_token_ids, spans):
                is_first = True
                for token_idx, (span_sent_index, span_token_index) in enumerate(span_token_ids):
                    if span_sent_index != sent_index:
                        continue
                    if is_first:
                        labels[span_token_index] = 'B'
                        is_first = False
                    else:
                        labels[span_token_index] = 'I'
                    indices[span_token_index] = ','.join(map(str, [data_type, file_id, span_idx, token_idx]))

            if all(label == 'O' for label in labels):
                continue
            yield tokens, labels, spaces, indices


def collect_dataset(data_split):
    dataset = []
    for data_tag in ['generic', 'named']:
        data_type = f'{data_split}/{data_tag}'
        if data_split == 'valid':
            data_type += '/texts_and_ann'
        file_ids = [
            path.replace('.txt', '')
            for path in os.listdir(f'../data/span_normalization/raw/{data_type}')
            if path.endswith('.txt')
        ]

        for file_id in tqdm(file_ids):
            for row in process_file(data_type, file_id):
                row = row + ([data_tag] * len(row[0]),)
                dataset.append(row)

    return dataset


def write_dataset(dataset, data_split):
    with open(f'../data/span_normalization/{data_split}/{data_split}.conllu', 'w') as f:
        for row in dataset:
            assert all(len(element) == len(row[0]) for element in row)
            for i, (token, label, space, index, data_tag) in enumerate(zip(*row)):
                print(f'{i + 1}\t{token}\t{index}\t{label}\t{data_tag}\t{space}', file=f)
            print(file=f)


def main():
    os.makedirs('../data/span_normalization/valid', exist_ok=True)
    os.makedirs('../data/span_normalization/test', exist_ok=True)

    for data_split in ['valid', 'test']:
        dataset = collect_dataset(data_split)
        write_dataset(dataset, data_split)


if __name__ == '__main__':
    main()
