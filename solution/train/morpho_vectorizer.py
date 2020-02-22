# -*- coding: utf-8 -*-

import numpy as np
import pymorphy2
import logging

from tqdm import tqdm
from allennlp.data.fields import ArrayField

logger = logging.getLogger(__name__)


class MorphoVectorizer(object):
    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer()
        self._grammeme_to_index = self._build_grammeme_to_index()
        self._morpho_vector_dim = max(self._grammeme_to_index.values()) + 1

    @property
    def morpho_vector_dim(self):
        return self._morpho_vector_dim

    def _build_grammeme_to_index(self):
        grammar_categories = [
            self._morph.TagClass.PARTS_OF_SPEECH,
            self._morph.TagClass.ANIMACY,
            self._morph.TagClass.ASPECTS,
            self._morph.TagClass.CASES,
            self._morph.TagClass.GENDERS,
            self._morph.TagClass.INVOLVEMENT,
            self._morph.TagClass.MOODS,
            self._morph.TagClass.NUMBERS,
            self._morph.TagClass.PERSONS,
            self._morph.TagClass.TENSES,
            self._morph.TagClass.TRANSITIVITY,
            self._morph.TagClass.VOICES
        ]

        grammeme_to_index = {}
        shift = 0
        for category in grammar_categories:
            # TODO: Save grammeme_to_index
            for grammeme_index, grammeme in enumerate(sorted(category)):
                grammeme_to_index[grammeme] = grammeme_index + shift
            shift += len(category) + 1  # +1 to address lack of the category in a parse

        return grammeme_to_index

    def vectorize_word(self, word):
        grammar_vector = np.zeros(self._morpho_vector_dim)
        sum_parses_score = 0.
        for parse in self._morph.parse(word):
            sum_parses_score += parse.score
            for grammeme in parse.tag.grammemes:
                grammeme_index = self._grammeme_to_index.get(grammeme)
                if grammeme_index:
                    grammar_vector[grammeme_index] += parse.score

        if sum_parses_score != 0.:
            grammar_vector /= sum_parses_score

        assert np.all(grammar_vector < 1.01) and np.all(grammar_vector >= 0.)

        # TODO: check why it doesn't work
        # pos_sum_prob = np.sum(grammar_vector[:len(self._morph.TagClass.PARTS_OF_SPEECH)])
        # assert pos_sum_prob == 0. or 0.99 < pos_sum_prob < 1.01, pos_sum_prob

        return grammar_vector

    def vectorize_instance(self, instance):
        meta = instance.fields['metadata']
        return np.array([self.vectorize_word(word) for word in meta['words']])

    def apply_to_instances(self, instances):
        for instance in tqdm(instances, 'MorphoVectorizer apply_to_instances'):
            grammar_matrix = self.vectorize_instance(instance)
            instance.add_field('morpho_embedding', ArrayField(grammar_matrix))
