# -*- coding: utf-8 -*-

import logging

from typing import Dict, Tuple, List

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

from train.corpus_iterator import CorpusIterator, Sentence as CorpusSentence

logger = logging.getLogger(__name__)


@DatasetReader.register('ud')
class UDDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, skip_labels=False,
                 max_length=None, read_first=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._skip_labels = skip_labels
        self._max_length = max_length
        self._read_first = read_first

    @overrides
    def _read(self, file_path: str):
        with CorpusIterator(file_path, span_tag_index=3, dataset_tag_index=4, grammar_val_col_indices=None,
                            head_col_index=None, head_tag_col_index=None) as corpus:
            for sent_index, sentence in enumerate(corpus):
                if self._max_length is not None and len(sentence) > self._max_length:
                    logger.info(
                        'Filtering out %s as too long in tokens (%s > %s)',
                        sentence.words, len(sentence), self._max_length
                    )
                    continue

                yield self.text_to_instance(sentence)

                if self._read_first is not None and sent_index + 1 == self._read_first:
                    return

    @overrides
    def text_to_instance(self, sentence: CorpusSentence) -> Instance:
        fields: Dict[str, Field] = {}
        metadata = {}

        text_field = TextField(list(map(Token, sentence.words)), self._token_indexers)

        fields['words'] = text_field
        metadata['words'] = sentence.words

        if sentence.lemmas:
            metadata['lemmas'] = sentence.lemmas

        if sentence.pos_tags:
            if not self._skip_labels:
                fields['pos_tags'] = SequenceLabelField(sentence.pos_tags, text_field, 'pos')
            metadata['pos'] = sentence.pos_tags

        if sentence.grammar_values and not self._skip_labels:
            fields['grammar_values'] = SequenceLabelField(sentence.grammar_values, text_field, 'grammar_value_tags')

        if sentence.heads and not self._skip_labels:
            fields['head_indices'] = SequenceLabelField(sentence.heads, text_field, 'head_index_tags')

        if sentence.head_tags and not self._skip_labels:
            fields['head_tags'] = SequenceLabelField(sentence.head_tags, text_field, 'head_tags')

        if sentence.span_tags:
            fields['span_tags'] = SequenceLabelField(sentence.span_tags, text_field, 'span_tags')

        if sentence.dataset_tags:
            fields['dataset_tags'] = SequenceLabelField(sentence.dataset_tags, text_field, 'dataset_tags')

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
