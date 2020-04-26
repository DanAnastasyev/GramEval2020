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
    lower = attr.ib(default=False)
    capitalize = attr.ib(default=False)
    upper = attr.ib(default=False)


class LemmatizeHelper(object):
    UNKNOWN_RULE_INDEX = 0
    _UNKNOWN_RULE_PLACEHOLDER = LemmatizeRule(cut_prefix=100, cut_suffix=100, append_suffix='-' * 90)
    _OUTPUT_FILE_NAME = 'lemmatizer_info.json'

    def __init__(self, lemmatize_rules=None):
        self._lemmatize_rules = lemmatize_rules
        self._index_to_rule = self._get_index_to_rule()

    def fit(self, data, min_freq=3):
        rules_counter = Counter()

        for instance in tqdm(data):
            meta = instance.fields['metadata']
            words, lemmas = meta['words'], meta['lemmas']
            for word, lemma in zip(words, lemmas):
                if lemma == '_' and word != '_':
                    continue

                rule = self.predict_lemmatize_rule(word, lemma)
                rules_counter[rule] += 1

                assert self.lemmatize(word, rule).replace('ё', 'е') == lemma.replace('ё', 'е')

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

        self._index_to_rule = self._get_index_to_rule()

        logger.info('Lemmatize rules count = {}, did not cover {:.2%} of words'.format(
            len(self._lemmatize_rules), skipped_count / total_count))

    def _get_index_to_rule(self):
        if not self._lemmatize_rules:
            return []
        return [rule for rule, _ in sorted(self._lemmatize_rules.items(), key=lambda pair: pair[1])]

    def get_rule_index(self, word, lemma):
        if lemma == '_' and word != '_':
            return self.UNKNOWN_RULE_INDEX

        rule = self.predict_lemmatize_rule(word, lemma)
        return self._lemmatize_rules.get(rule, self.UNKNOWN_RULE_INDEX)

    def get_rule_indices(self, instance):
        meta = instance.fields['metadata']
        words, lemmas = meta['words'], meta['lemmas']

        return [self.get_rule_index(word, lemma) for word, lemma in zip(words, lemmas)]

    def get_rule(self, rule_index):
        return self._index_to_rule[rule_index]

    def __len__(self):
        return len(self._lemmatize_rules)

    @staticmethod
    def predict_lemmatize_rule(word: str, lemma: str):
        def _predict_lemmatize_rule(word: str, lemma: str, **kwargs):
            if len(word) == 0:
                return LemmatizeRule(append_suffix=lemma, **kwargs)

            common_prefix = commonprefix([word, lemma])
            if len(common_prefix) == 0:
                rule = _predict_lemmatize_rule(word[1:], lemma, **kwargs)
                return attr.evolve(rule, cut_prefix=rule.cut_prefix + 1)

            return LemmatizeRule(cut_suffix=len(word) - len(common_prefix),
                                 append_suffix=lemma[len(common_prefix):], **kwargs)

        word, lemma = word.replace('ё', 'е'), lemma.replace('ё', 'е')
        return min([
            _predict_lemmatize_rule(word, lemma),
            _predict_lemmatize_rule(word.lower(), lemma, lower=True),
            _predict_lemmatize_rule(word.capitalize(), lemma, capitalize=True),
            _predict_lemmatize_rule(word.upper(), lemma, upper=True)
        ], key=lambda rule: rule.cut_prefix + rule.cut_suffix)

    def lemmatize(self, word, rule):
        if isinstance(rule, numbers.Integral):
            rule = self.get_rule(rule)

        assert isinstance(rule, LemmatizeRule)

        if rule.lower:
            word = word.lower()
        if rule.capitalize:
            word = word.capitalize()
        if rule.upper:
            word = word.upper()

        if rule.cut_suffix != 0:
            lemma = word[rule.cut_prefix: -rule.cut_suffix]
        else:
            lemma = word[rule.cut_prefix:]
        lemma += rule.append_suffix

        return lemma

    def apply_to_instances(self, instances):
        for instance in tqdm(instances, 'Lemmatizer apply_to_instances'):
            lemmatize_rule_indices = self.get_rule_indices(instance)
            field = SequenceLabelField(lemmatize_rule_indices, instance.fields['words'], 'lemma_index_tags')
            instance.add_field('lemma_indices', field)

    def save(self, dir_path):
        with open(os.path.join(dir_path, self._OUTPUT_FILE_NAME), 'w') as f:
            index_to_rule = [attr.asdict(rule) for rule in self._index_to_rule]
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
