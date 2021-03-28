# -*- coding: utf-8 -*-

import attr
import json
import logging
import numbers
import os

from os.path import commonprefix
from collections import Counter, defaultdict
from tqdm import tqdm

from allennlp.data.fields import SequenceLabelField

logger = logging.getLogger(__name__)


@attr.s(frozen=True)
class LemmatizeRule(object):
    cut_prefix = attr.ib(default=0)
    cut_suffix = attr.ib(default=0)
    append_suffix = attr.ib(default='')


@attr.s(frozen=True)
class CapitalizeRule(object):
    lower = attr.ib(default=False)
    capitalize = attr.ib(default=False)
    upper = attr.ib(default=False)


class LemmatizeHelper(object):
    UNKNOWN_RULE_INDEX = 0
    _UNKNOWN_RULE_PLACEHOLDER = LemmatizeRule(cut_prefix=100, cut_suffix=100, append_suffix='-' * 90)
    _OUTPUT_FILE_NAME = 'lemmatizer_info.json'

    def __init__(self, lemmatize_rules=None):
        self._lemmatize_rules = lemmatize_rules
        self._capitalize_rules = {
            CapitalizeRule(lower=True, capitalize=True, upper=True): 0,  # Padding (maybe redundant)
            CapitalizeRule(): 1,
            CapitalizeRule(lower=True): 2,
            CapitalizeRule(capitalize=True): 3,
            CapitalizeRule(upper=True): 4,
        }
        self._index_to_lemmatize_rule = self._get_index_to_rule(self._lemmatize_rules)
        self._index_to_capitalize_rule = self._get_index_to_rule(self._capitalize_rules)

    def fit(self, data, min_freq=3):
        rules_counter = Counter()

        for instance in tqdm(data):
            meta = instance.fields['metadata']
            words, lemmas = meta['words'], meta['lemmas']
            for word, lemma in zip(words, lemmas):
                if lemma == '_' and word != '_':
                    continue

                lemmatize_rule, capitalize_rule = self.predict_lemmatize_rule(word, lemma)
                rules_counter[lemmatize_rule] += 1

                assert self.lemmatize(word, lemmatize_rule, capitalize_rule).replace('ё', 'е') == lemma.replace('ё', 'е')

        self._lemmatize_rules = {
            self._UNKNOWN_RULE_PLACEHOLDER: self.UNKNOWN_RULE_INDEX,
            LemmatizeRule(): self.UNKNOWN_RULE_INDEX + 1
        }
        skipped_count, total_count = 0., 0
        for rule, count in rules_counter.most_common():
            total_count += count
            if count < min_freq:
                skipped_count += count
                continue

            if rule not in self._lemmatize_rules:
                self._lemmatize_rules[rule] = len(self._lemmatize_rules)

        self._index_to_lemmatize_rule = self._get_index_to_rule(self._lemmatize_rules)

        logger.info('Lemmatize rules count = {}, did not cover {:.2%} of words'.format(
            len(self._lemmatize_rules), skipped_count / total_count))

    def _get_index_to_rule(self, rules):
        if not rules:
            return []
        return [rule for rule, _ in sorted(rules.items(), key=lambda pair: pair[1])]

    def get_rule_index(self, word, lemma):
        if lemma == '_' and word != '_':
            return self.UNKNOWN_RULE_INDEX, self.UNKNOWN_RULE_INDEX

        lemmatize_rule, capitalize_rule = self.predict_lemmatize_rule(word, lemma)
        lemmatize_rule_index = self._lemmatize_rules.get(lemmatize_rule, self.UNKNOWN_RULE_INDEX)
        capitalize_rule_index = self._capitalize_rules.get(capitalize_rule, self.UNKNOWN_RULE_INDEX)

        return lemmatize_rule_index, capitalize_rule_index

    def get_rule_indices(self, instance):
        meta = instance.fields['metadata']
        words, lemmas = meta['words'], meta['lemmas']

        lemmatize_rule_indices, capitalize_rule_indices = [], []
        for word, lemma in zip(words, lemmas):
            lemmatize_rule_index, capitalize_rule_index = self.get_rule_index(word, lemma)
            lemmatize_rule_indices.append(lemmatize_rule_index)
            capitalize_rule_indices.append(capitalize_rule_index)

        return lemmatize_rule_indices, capitalize_rule_indices

    def get_lemmatize_rule(self, rule_index):
        return self._index_to_lemmatize_rule[rule_index]

    def get_capitalize_rule(self, rule_index):
        return self._index_to_capitalize_rule[rule_index]

    def lemmatize_rule_count(self):
        return len(self._lemmatize_rules)

    def capitalize_rule_count(self):
        return len(self._capitalize_rules)

    @staticmethod
    def predict_lemmatize_rule(word: str, lemma: str):
        def _predict_lemmatize_rule(word: str, lemma: str, **kwargs):
            if len(word) == 0:
                return LemmatizeRule(append_suffix=lemma), CapitalizeRule(**kwargs)

            common_prefix = commonprefix([word, lemma])
            if len(common_prefix) == 0:
                lemmatize_rule, capitalize_rule = _predict_lemmatize_rule(word[1:], lemma, **kwargs)
                return attr.evolve(lemmatize_rule, cut_prefix=lemmatize_rule.cut_prefix + 1), capitalize_rule

            lemmatize_rule = LemmatizeRule(
                cut_suffix=len(word) - len(common_prefix), append_suffix=lemma[len(common_prefix):]
            )
            capitalize_rule = CapitalizeRule(**kwargs)

            return lemmatize_rule, capitalize_rule

        word, lemma = word.replace('ё', 'е'), lemma.replace('ё', 'е')
        return min([
            _predict_lemmatize_rule(word, lemma),
            _predict_lemmatize_rule(word.lower(), lemma, lower=True),
            _predict_lemmatize_rule(word.capitalize(), lemma, capitalize=True),
            _predict_lemmatize_rule(word.upper(), lemma, upper=True)
        ], key=lambda rule: rule[0].cut_prefix + rule[0].cut_suffix)

    def lemmatize(self, word, lemmatize_rule, capitalize_rule):
        if isinstance(lemmatize_rule, numbers.Integral):
            lemmatize_rule = self.get_lemmatize_rule(lemmatize_rule)

        if isinstance(capitalize_rule, numbers.Integral):
            capitalize_rule = self.get_capitalize_rule(capitalize_rule)

        assert isinstance(lemmatize_rule, LemmatizeRule)
        assert isinstance(capitalize_rule, CapitalizeRule)

        if capitalize_rule.lower:
            word = word.lower()
        if capitalize_rule.capitalize:
            word = word.capitalize()
        if capitalize_rule.upper:
            word = word.upper()

        if lemmatize_rule.cut_suffix != 0:
            lemma = word[lemmatize_rule.cut_prefix: -lemmatize_rule.cut_suffix]
        else:
            lemma = word[lemmatize_rule.cut_prefix:]
        lemma += lemmatize_rule.append_suffix

        return lemma

    def apply_to_instances(self, instances):
        for instance in tqdm(instances, 'Lemmatizer apply_to_instances'):
            lemmatize_rule_indices, capitalize_rule_indices = self.get_rule_indices(instance)

            field = SequenceLabelField(lemmatize_rule_indices, instance.fields['words'], 'lemma_index_tags')
            instance.add_field('lemma_indices', field)

            field = SequenceLabelField(capitalize_rule_indices, instance.fields['words'], 'capitalize_index_tags')
            instance.add_field('capitalize_indices', field)

    def save(self, dir_path):
        with open(os.path.join(dir_path, self._OUTPUT_FILE_NAME), 'w') as f:
            index_to_rule = [attr.asdict(rule) for rule in self._index_to_lemmatize_rule]
            json.dump(index_to_rule, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_path):
        with open(os.path.join(dir_path, cls._OUTPUT_FILE_NAME)) as f:
            index_to_rule = json.load(f)

        lemmatize_rules = {
            LemmatizeRule(**rule_dict): index
            for index, rule_dict in enumerate(index_to_rule)
        }

        return cls(lemmatize_rules)
