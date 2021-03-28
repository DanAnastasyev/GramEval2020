# -*- coding: utf-8 -*-

import os
import pymorphy2
import random
import re

from difflib import SequenceMatcher
from razdel import sentenize
from razdel.segmenters.tokenize import TokenSegmenter, DebugTokenSegmenter, Rule2112, RULES, DASHES, PUNCT, JOIN
from string import punctuation
from tqdm import tqdm

MORPH = pymorphy2.MorphAnalyzer()


_ALLOWED_ALIGNMENTS = {
    ('й', 'я'),
    ('меньше', 'малый'),
    ('реализуются', 'реализовываться'),
    ('утвержденной', 'утверждать'),
    ('реализуется', 'реализовываться'),
    ('направлены', 'направлять'),
    ('го', 'й'),
    ('MC', 'МС'),
    ('Олы', 'Ола'),
    ('организует', 'организовывать'),
    ('ведется', 'вести'),
    ('снизился', 'снизить'),
    ('реализуется', 'реализовать'),
    ('возросло', 'возрастать'),
    ('формируются', 'формировать'),
    ('более', 'больший'),
    ('м', 'й'),
    ('му', 'й'),
    ('Оле', 'Ола'),
    ('создаются', 'создавать'),
    ('установка', 'установление'),
    ('образованная', 'образовывать'),
    ('отражены', 'отражать'),
    ('усилилась', 'усиливаться'),
    ('сформироваться', 'сформирует'),
    ('устанавливаемых', 'установить'),
    ('Произведено', 'Производить'),
    ('увеличились', 'увеличиваться'),
    ('увеличилось', 'увеличиваться'),
    ('возросла', 'возрастать'),
    ('возросли', 'возрастать'),
    ('получили', 'получать'),
    ('повышенными', 'повышать'),
    ('решен', 'решать'),
    ('выверенно', 'выверять'),
    ('реализованных', 'реализовывать'),
    ('созданная', 'создавать'),
    ('снижен', 'снижать'),
    ('запущен', 'запускать'),
    ('реализуемых', 'реализовывать'),
    ('Планируется', 'Планировать'),
    ('производимых', 'произвести'),
    ('реализующим', 'реализовывать'),
    ('планируется', 'планировать'),
    ('намерены', 'намериться'),
    ('имеющихся', 'иметь'),
    ('создаваемых', 'создать'),
    ('связующего', 'связывать'),
    ('имеющиеся', 'иметь'),
    ('отраженная', 'отражать'),
    ('меняющимся', 'менять'),
    ('развивающейся', 'развивать'),
    ('Установлены', 'Устанавливать'),
    ('реализует', 'реализовывать'),
    ('получены', 'получать'),
    ('возникших', 'возникать'),
    ('строящегося', 'строить'),
    ('строящееся', 'строить'),
    ('рассовой', 'расовая'),
    ('Лешиному', 'Лёша'),
    ('Гру', 'Гра'),
    ('Гры', 'Гра'),
    ('BBC', 'ВВС'),
    ('определены', 'определять'),
    ('признается', 'признавать'),
    ('Совершенствуется', 'совершенствовать'),
    ('открылся', 'открыть'),
    ('превышает', 'превысить'),
    ('установленными', 'устанавливать'),
}

_PUNCT_FIXES = [
    ('–', '-'),
    ('-', '–'),
    ('»', '"'),
    ('«', '"'),
    ('’', "'"),
    ('"', '«'),
]

_BE = ('будет', 'будут', 'был', 'была', 'было', 'были')

_IS_PUNCT = re.compile(f'[{punctuation}]+')


class MyDashRule(Rule2112):
    name = 'my_dash'

    def delimiter(self, delimiter):
        return delimiter in DASHES

    def rule(self, left, right):
        if left.type == PUNCT or right.type == PUNCT:
            return
        if left.text in ('кое',):
            return JOIN
        if right.text in ('то', 'либо', 'нибудь'):
            return JOIN


tokenize = TokenSegmenter(rules=[MyDashRule()] + RULES[1:])


def read_file(data_type, file_id):
    with open(f'../data/span_normalization/raw/{data_type}/norm/{file_id}.norm') as f:
        normalizations = [line.rstrip() for line in f]

    with open(f'../data/span_normalization/raw/{data_type}/texts_and_ann/{file_id}.txt') as f:
        text = f.readlines()
        assert len(text) == 1
        text = text[0].rstrip()

    with open(f'../data/span_normalization/raw/{data_type}/texts_and_ann/{file_id}.ann') as f:
        spans = [list(map(int, line.strip().split())) for line in f]

    spans = [
        [(span[i], span[i + 1]) for i in range(0, len(span), 2)]
        for span in spans
    ]
    for span in spans:
        assert sorted(span) == span, span

    spans = [sorted(set(span)) for span in spans]

    return text, normalizations, spans


def sentenize_text(text):
    position = 0
    sentence_tokens = []
    for line in text.split('\u2028'):
        for sentence in sentenize(line):
            sentence.start + position
            cur_sentence_tokens = []
            for token in tokenize(sentence.text):
                token.start += sentence.start + position
                token.stop += sentence.start + position
                cur_sentence_tokens.append(token)
            sentence_tokens.append(cur_sentence_tokens)

        position += len(line) + 1

    return sentence_tokens


def collect_span_tokens(sentence_tokens, spans, file_id):
    i, j = 0, 0
    span_tokens = []
    while i < len(sentence_tokens):
        while j < len(sentence_tokens[i]):
            for span in spans:
                if ((span[0] > sentence_tokens[i][j].start or sentence_tokens[i][j].stop > span[1])
                    and max(span[0], sentence_tokens[i][j].start) < min(span[1], sentence_tokens[i][j].stop)
                ):
                    print(file_id, span, sentence_tokens[i][j])
                if span[0] <= sentence_tokens[i][j].start and sentence_tokens[i][j].stop <= span[1]:
                    span_tokens.append((i, j))
            j += 1
        i += 1
        j = 0
    return span_tokens


def normalize(text):
    return text.lower().replace('ё', 'е')


def can_align_tokens(token_1, token_2):
    if (token_1, token_2) in _ALLOWED_ALIGNMENTS:
        return True
    if SequenceMatcher(None, token_1, token_2).ratio() > 0.7:
        return True

    parses_1 = MORPH.parse(token_1)
    parses_2 = MORPH.parse(token_2)
    lemmas_1 = [normalize(token_1)] + [normalize(parse.normal_form) for parse in parses_1]
    lemmas_2 = [normalize(token_2)] + [normalize(parse.normal_form) for parse in parses_2]

    if len(set(lemmas_1) & set(lemmas_2)) == 0:
        return False
    return True


def can_align_normalization(span_tokens, normalization_tokens):
    if len(span_tokens) != len(normalization_tokens):
        return False

    has_error = False
    for token_1, token_2 in zip(span_tokens, normalization_tokens):
        has_error = can_align_tokens(token_1.text, token_2) or has_error

    return has_error


def try_fix_normalization_tokens(span_tokens, norm_tokens):
    # if len(span_tokens) == 2 and len(norm_tokens) == 1 and span_tokens[0].text in _BE:
    #     return ['~'] + norm_tokens

    norm_token_index = 0
    span_token_index = 0
    updated_norm_tokens = []
    while span_token_index < len(span_tokens) and norm_token_index < len(norm_tokens):
        was_found = False
        for from_punct, to_punct in _PUNCT_FIXES:
            if span_tokens[span_token_index].text == from_punct and norm_tokens[norm_token_index] == to_punct:
                span_token_index += 1
                norm_token_index += 1
                updated_norm_tokens.append(from_punct)
                was_found = True
                break
        if was_found:
            continue

        if can_align_tokens(span_tokens[span_token_index].text, norm_tokens[norm_token_index]):
            updated_norm_tokens.append(norm_tokens[norm_token_index])
            span_token_index += 1
            norm_token_index += 1
            continue

        if norm_tokens[norm_token_index].lower().startswith(span_tokens[span_token_index].text.lower()):
            norm_suffix = norm_tokens[norm_token_index][len(span_tokens[span_token_index].text):]
            if (span_token_index + 1 < len(span_tokens)
                and can_align_tokens(span_tokens[span_token_index + 1].text, norm_suffix)
            ):
                updated_norm_tokens.append(span_tokens[span_token_index].text)
                updated_norm_tokens.append(norm_suffix)
                span_token_index += 2
                norm_token_index += 1
                continue

        if (norm_tokens[norm_token_index] == '-'
            and norm_token_index + 1 < len(norm_tokens)
            and can_align_tokens(span_tokens[span_token_index].text, norm_tokens[norm_token_index + 1])
        ):
            updated_norm_tokens[-1] += '-'
            updated_norm_tokens.append(norm_tokens[norm_token_index + 1])
            span_token_index += 1
            norm_token_index += 2
            continue

        if _IS_PUNCT.match(span_tokens[span_token_index].text) and not _IS_PUNCT.match(norm_tokens[norm_token_index]):
            updated_norm_tokens.append(span_tokens[span_token_index].text)
            span_token_index += 1
            continue

        if not _IS_PUNCT.match(span_tokens[span_token_index].text) and _IS_PUNCT.match(norm_tokens[norm_token_index]):
            updated_norm_tokens[-1] += norm_tokens[norm_token_index]
            norm_token_index += 1
            continue

        return

    if norm_token_index == len(norm_tokens) and span_token_index == len(span_tokens):
        assert len(updated_norm_tokens) == len(span_tokens)
        return updated_norm_tokens


def should_remove(spans, i, spans_to_remove):
    for j in range(i):
        if j in spans_to_remove:
            continue

        span1 = sorted(spans[i][1])
        span2 = sorted(spans[j][1])

        max_begin = max(span1[0][0], span2[0][0])
        min_end = min(span1[-1][1], span2[-1][1])
        if max_begin < min_end:
            return True
    return False


def remove_intersections(spans):
    if not spans:
        return []

    spans_to_remove = set()
    for i in range(len(spans)):
        if should_remove(spans, i, spans_to_remove):
            spans_to_remove.add(i)

    non_intersecting_spans = [spans[i] for i in range(len(spans)) if i not in spans_to_remove]
    intersecting_spans = [spans[i] for i in range(len(spans)) if i in spans_to_remove]

    return [non_intersecting_spans] + remove_intersections(intersecting_spans)


def process_file(data_type, file_id):
    text, normalizations, spans = read_file(data_type, file_id)
    sentence_tokens = sentenize_text(text)
    normalizations = sorted(zip(normalizations, spans), key=lambda x: x[1][0])

    for normalizations in remove_intersections(normalizations):
        spans_token_ids, normalizations_tokens = [], []
        for normalization, span in normalizations:
            span_token_ids = collect_span_tokens(sentence_tokens, span, file_id)
            if not span_token_ids:
                print(f'Skipping: {normalization} -> {span}')
                continue

            span_tokens = [sentence_tokens[span[0]][span[1]] for span in span_token_ids]
            normalization_tokens = [token.text for token in tokenize(normalization)]
            if can_align_normalization(span_tokens, normalization_tokens):
                spans_token_ids.append(span_token_ids)
                normalizations_tokens.append(normalization_tokens)
            else:
                normalization_tokens = try_fix_normalization_tokens(span_tokens, normalization_tokens)
                if normalization_tokens and can_align_normalization(span_tokens, normalization_tokens):
                    spans_token_ids.append(span_token_ids)
                    normalizations_tokens.append(normalization_tokens)
                else:
                    print(f'Cannot align {span_tokens} with {normalization}')

        for sent_index, sentence in enumerate(sentence_tokens):
            tokens = [token.text for token in sentence]
            lemmas = ['_' for _ in sentence]
            labels = ['O' for _ in sentence]
            spaces = []
            for token_index in range(len(sentence)):
                if token_index + 1 != len(sentence):
                    spaces.append(text[sentence[token_index].stop: sentence[token_index + 1].start])
                else:
                    spaces.append('')

            for span_token_ids, normalization_tokens in zip(spans_token_ids, normalizations_tokens):
                for token_idx, (span_token_id, normalization) in enumerate(zip(span_token_ids, normalization_tokens)):
                    span_sent_index, span_token_index = span_token_id
                    if span_sent_index == sent_index:
                        lemmas[span_token_index] = normalization
                        labels[span_token_index] = 'I' if token_idx > 0 else 'B'

            if all(lemma == '_' for lemma in lemmas):
                continue
            yield tokens, lemmas, spaces, labels


def collect_dataset(data_split):
    dataset = []
    for data_tag in ['generic', 'named']:
        data_type = f'{data_split}/{data_tag}'
        file_ids = [path.replace('.norm', '') for path in os.listdir(f'../data/span_normalization/raw/{data_type}/norm')]

        for file_id in tqdm(file_ids):
            for tokens, lemmas, spaces, labels in process_file(data_type, file_id):
                data_tags = [data_tag] * len(labels)
                dataset.append((tokens, lemmas, spaces, labels, data_tags))

    return dataset


def write_dataset(dataset, data_split):
    path = 'data_train/train.conllu' if data_split == 'train' else 'data_open_test/valid.conllu'
    with open(f'../data/span_normalization/{path}', 'w') as f:
        for row in dataset:
            assert all(len(element) == len(row[0]) for element in row)
            for i, (token, lemma, space, label, data_tag) in enumerate(zip(*row)):
                print(f'{i + 1}\t{token}\t{lemma}\t{label}\t{data_tag}\t{space}', file=f)
            print(file=f)


def main():
    os.makedirs('../data/span_normalization/data_train', exist_ok=True)
    os.makedirs('../data/span_normalization/data_open_test', exist_ok=True)

    for data_split in ['train', 'valid']:
        dataset = collect_dataset(data_split)
        write_dataset(dataset, data_split)


if __name__ == '__main__':
    main()
