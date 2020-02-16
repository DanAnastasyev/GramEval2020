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
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with CorpusIterator(file_path) as corpus:
            for sentence in corpus:
                yield self.text_to_instance(sentence)

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
            fields['pos_tags'] = SequenceLabelField(sentence.pos_tags, text_field, 'pos')
            metadata['pos'] = sentence.pos_tags

        if sentence.grammar_values:
            fields['grammar_values'] = SequenceLabelField(sentence.grammar_values, text_field, 'grammar_value_tags')

        if sentence.heads:
            fields['head_indices'] = SequenceLabelField(sentence.heads, text_field, 'head_index_tags')

        if sentence.head_tags:
            fields['head_tags'] = SequenceLabelField(sentence.head_tags, text_field, 'head_tags')

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
