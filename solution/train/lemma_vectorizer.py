# -*- coding: utf-8 -*-

import numpy as np
import pymorphy2
import logging

from tqdm import tqdm
from allennlp.data.fields import ArrayField
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class LemmaVectorizer(object):
    def __init__(self):
        self._morph = None
        self._lemmatize_helper = None

    def load(self, lemmatize_helper):
        self._morph = pymorphy2.MorphAnalyzer()
        self._lemmatize_helper = lemmatize_helper

    def vectorize_word(self, word):
        lemma_vector = np.zeros(self._lemmatize_helper.lemmatize_rule_count())
        for parse in self._morph.parse(word):
            for form in parse.lexeme:
                lemma_id, _ = self._lemmatize_helper.get_rule_index(word, form.word)
                lemma_vector[lemma_id] = 1.
        return lemma_vector

    def __call__(self, inputs):
        index, instance = inputs
        meta = instance.fields['metadata']
        return np.array([self.vectorize_word(word) for word in meta['words']]), index


_LEMMA_VECTORIZER = LemmaVectorizer()


def _load(lemmatize_helper):
    _LEMMA_VECTORIZER.load(lemmatize_helper)


def _apply(inputs):
    return _LEMMA_VECTORIZER(inputs)


def apply_to_instances(lemmatize_helper, instances):
    with Pool(32, initializer=_load, initargs=(lemmatize_helper,)) as pool:
        output_stream = pool.imap(_apply, enumerate(instances), chunksize=100)
        for matrix, index in tqdm(output_stream, desc='LemmaVectorizer apply_to_instances', total=len(instances)):
            instances[index].add_field('lemma_embedding', ArrayField(matrix))
