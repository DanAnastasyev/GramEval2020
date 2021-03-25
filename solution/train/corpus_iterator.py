# -*- coding: utf-8 -*-

import attr
from itertools import chain
from typing import Tuple, List


@attr.s(frozen=True)
class Token:
    text: str = attr.ib()
    lemma: str = attr.ib()
    pos_tag: str = attr.ib()
    grammar_value: str = attr.ib()
    head: int = attr.ib()
    head_tag: str = attr.ib()
    span_tag: str = attr.ib()
    dataset_tag: str = attr.ib()


class Sentence(object):
    def __init__(self, tokens: List[Token]):
        self._tokens = tokens

    @property
    def words(self):
        return [token.text for token in self._tokens]

    @property
    def pos_tags(self):
        if not self._tokens or self._tokens[0].pos_tag is None:
            return None
        return [token.pos_tag for token in self._tokens]

    @property
    def lemmas(self):
        if not self._tokens or self._tokens[0].lemma is None:
            return None
        return [token.lemma for token in self._tokens]

    @property
    def grammar_values(self):
        if not self._tokens or self._tokens[0].grammar_value is None:
            return None
        return [token.grammar_value for token in self._tokens]

    @property
    def heads(self):
        if not self._tokens or self._tokens[0].head is None:
            return None
        return [token.head for token in self._tokens]

    @property
    def head_tags(self):
        if not self._tokens or self._tokens[0].head_tag is None:
            return None
        return [token.head_tag for token in self._tokens]

    @property
    def span_tags(self):
        if not self._tokens or self._tokens[0].span_tag is None:
            return None
        return [token.span_tag for token in self._tokens]

    @property
    def dataset_tags(self):
        if not self._tokens or self._tokens[0].dataset_tag is None:
            return None
        return [token.dataset_tag for token in self._tokens]

    def __len__(self):
        return len(self._tokens)


class CorpusIterator:
    def __init__(self, path: str, separator: str='\t', token_col_index: int=1, lemma_col_index: int=2,
                 grammar_val_col_indices: Tuple=(3, 5), grammemes_separator: str='|',
                 head_col_index: int=6, head_tag_col_index: int=7, span_tag_index: int=None,
                 dataset_tag_index: int=None, column_count: int=None, skip_line_prefix: str='#', encoding: str='utf8'):
        """
        Creates iterator over the corpus in conll-like format:
        - each line contains token and its annotations (lemma and grammar value info) separated by ``separator``
        - sentences are separated by empty line
        :param path: path to corpus
        :param separator: separator between fields
        :param token_col_index: index of token field
        :param lemma_col_index: index of lemma field
        :param grammar_val_col_indices: indices of grammar value fields (e.g. POS and morphological tags)
        :param grammemes_separator: separator between grammemes (as in 'Case=Nom|Definite=Def|Gender=Com|Number=Sing')
        :param head_col_index: index of head field
        :param head_tag_col_index: index of head_tag field
        :param skip_line_prefix: prefix for comment lines
        :param encoding: encoding of the corpus file
        """
        self._path = path
        self._separator = separator
        self._token_col_index = token_col_index
        self._lemma_col_index = lemma_col_index
        self._grammar_val_col_indices = grammar_val_col_indices
        self._grammemes_separator = grammemes_separator
        self._head_col_index = head_col_index
        self._head_tag_col_index = head_tag_col_index
        self._skip_line_prefix = skip_line_prefix
        self._span_tag_index = span_tag_index
        self._dataset_tag_index = dataset_tag_index
        self._column_count = column_count
        self._encoding = encoding

    def __enter__(self):
        self._file = open(self._path, encoding=self._encoding)
        return self

    def __exit__(self, type, value, traceback):
        self._file.close()

    def __iter__(self):
        return self

    def _read_token(self, line):
        if self._column_count is not None:
            fields = line.split(self._separator, self._column_count)
        else:
            fields = line.split(self._separator)

        token_text = fields[self._token_col_index]
        lemma, pos_tag, grammar_value, head, head_tag = None, None, None, None, None
        span_tag, dataset_tag = None, None

        if self._lemma_col_index is not None and self._lemma_col_index < len(fields):
            lemma = fields[self._lemma_col_index]

        if (self._grammar_val_col_indices is not None
            and all(index < len(fields) for index in self._grammar_val_col_indices)
        ):
            grammar_value = '|'.join(chain(*(sorted(fields[col_index].split(self._grammemes_separator))
                                                for col_index in self._grammar_val_col_indices)))

        if self._grammar_val_col_indices and self._grammar_val_col_indices[0] < len(fields):
            pos_tag = fields[self._grammar_val_col_indices[0]]

        if self._head_col_index is not None and self._head_col_index < len(fields):
            head = int(fields[self._head_col_index])

        if self._head_tag_col_index is not None and self._head_tag_col_index < len(fields):
            head_tag = fields[self._head_tag_col_index]

        if self._span_tag_index is not None and self._span_tag_index < len(fields):
            span_tag = fields[self._span_tag_index]
            assert span_tag in ('O', 'B', 'I'), fields

        if self._dataset_tag_index is not None and self._dataset_tag_index < len(fields):
            dataset_tag = fields[self._dataset_tag_index]
            assert dataset_tag in ('generic', 'named'), fields

        return Token(
            text=token_text,
            lemma=lemma,
            pos_tag=pos_tag,
            grammar_value=grammar_value,
            head=head,
            head_tag=head_tag,
            span_tag=span_tag,
            dataset_tag=dataset_tag,
        )

    def __next__(self) -> Sentence:
        while True:
            sentence = []
            for line in self._file:
                line = line[:-1]
                if line.startswith(self._skip_line_prefix):
                    continue
                if len(line) == 0:
                    break

                sentence.append(self._read_token(line))
            else:
                if not sentence:
                    raise StopIteration

            if sentence:
                return Sentence(sentence)
