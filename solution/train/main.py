# -*- coding: utf-8 -*-

import attr
import json
import logging
import os
import torch
import torch.optim as optim

from itertools import chain

from allennlp.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.data.fields import SequenceLabelField
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer

from train.dataset_reader import UDDatasetReader
from train.lemmatize_helper import LemmatizeHelper
from train.model import DependencyParser

logger = logging.getLogger(__name__)


@attr.s
class Config(object):
    model_name = attr.ib(default='parser_with_tagger')
    embedder_name = attr.ib(default='elmo')
    batch_size = attr.ib(default=128)
    num_epochs = attr.ib(default=15)
    patience = attr.ib(default=5)
    data_dir = attr.ib(default='../data')

    train_data_all_except = attr.ib(default=['GramEval2020-SynTagRus-train.conllu',
                                             'GramEval2020-17cent-train.conllu'])
    # train_data_all_except = attr.ib(default=None)

    # train_data = attr.ib(default=['GramEval2020-SynTagRus-train.conllu'])
    train_data = attr.ib(default=['GramEval2020-GSD-train.conllu'])

    valid_data = attr.ib(default='all')
    # valid_data = attr.ib(default=['GramEval2020-SynTagRus-dev.conllu'])
    # valid_data = attr.ib(default=['GramEval2020-GSD-wiki-dev.conllu'])

    pretrained_models_dir = attr.ib(default='../pretrained_models')
    models_dir = attr.ib(default='../models')


def _get_reader(config):
    indexer = ELMoTokenCharactersIndexer()
    return UDDatasetReader({config.embedder_name: indexer})


def _load_train_data(config):
    reader = _get_reader(config)

    train_data, valid_data = [], []

    if config.train_data_all_except:
        for path in os.listdir(os.path.join(config.data_dir, 'data_train')):
            if path not in config.train_data_all_except:
                if not path.endswith('.conllu'):
                    continue
                logger.info('Loading train file %s', path)
                train_data.extend(reader.read(os.path.join(config.data_dir, 'data_train', path)))
    else:
        for path in config.train_data:
            logger.info('Loading train file %s', path)
            train_data.extend(reader.read(os.path.join(config.data_dir, 'data_train', path)))

    if config.valid_data == 'all':
        for path in os.listdir(os.path.join(config.data_dir, 'data_open_test')):
            if not path.endswith('.conllu'):
                continue
            logger.info('Loading valid file %s', path)
            valid_data.extend(reader.read(os.path.join(config.data_dir, 'data_open_test', path)))
    else:
        for path in config.valid_data:
            logger.info('Loading valid file %s', path)
            valid_data.extend(reader.read(os.path.join(config.data_dir, 'data_open_test', path)))

    return train_data, valid_data


def _load_embedder(config):
    embedder = ElmoTokenEmbedder(
        options_file=os.path.join(config.pretrained_models_dir, 'elmo/options.json'),
        weight_file=os.path.join(config.pretrained_models_dir, 'elmo/model.hdf5'),
        dropout=0.3
    )
    embedder.eval()

    return BasicTextFieldEmbedder({config.embedder_name: embedder})


def _build_model(config, vocab, lemmatize_helper):
    embedder = _load_embedder(config)

    # TODO: AWD-LSTM Dropout
    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(
            embedder.get_output_dim(), 256, num_layers=2, dropout=0.3,
            bidirectional=True, batch_first=True
        )
    )

    return DependencyParser(
        vocab=vocab,
        text_field_embedder=embedder,
        encoder=encoder,
        lemmatize_helper=lemmatize_helper,
        tag_representation_dim=128,
        arc_representation_dim=128,
        dropout=0.2
    )


def _build_trainer(config, model, vocab, train_data, valid_data):
    optimizer = optim.AdamW(model.parameters())
    iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[('words', 'num_tokens')])
    iterator.index_with(vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
        logger.info('Using cuda')
    else:
        cuda_device = -1
        logger.info('Using cpu')

    logger.info('Example batch:')
    batch = next(iterator(train_data))
    for key, value in batch.items():
        if isinstance(value, dict):
            for inner_key, tensor in value.items():
                if isinstance(tensor, torch.Tensor):
                    logger.info('%s -> %s', inner_key, tensor.shape)
        elif isinstance(value, torch.Tensor):
            logger.info('%s -> %s', key, value.shape)

    return Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_data,
        validation_dataset=valid_data,
        patience=config.patience,
        num_epochs=config.num_epochs,
        cuda_device=cuda_device,
        grad_clipping=5.,
        serialization_dir=os.path.join(config.models_dir, config.model_name)
    )


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    config = Config()

    train_data, valid_data = _load_train_data(config)
    logger.info('Train data size = %s, valid data size = %s', len(train_data), len(valid_data))

    lemmatize_helper = LemmatizeHelper()
    lemmatize_helper.fit(train_data)

    for instance in chain(train_data, valid_data):
        lemmatize_rule_indices = lemmatize_helper.get_rule_indices(instance)
        field = SequenceLabelField(lemmatize_rule_indices, instance.fields['words'], 'lemma_index_tags')
        instance.add_field('lemma_indices', field)

    vocab = Vocabulary.from_instances(chain(train_data, valid_data))
    logger.info('Vocab = %s', vocab)
    vocab.print_statistics()

    model = _build_model(config, vocab, lemmatize_helper)
    logger.info('Model:\n%s', model)

    trainer = _build_trainer(config, model, vocab, train_data, valid_data)
    trainer.train()

    model_dir = os.path.join(config.models_dir, config.model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    vocab.save_to_files(os.path.join(model_dir, 'vocab'))
    lemmatize_helper.save(model_dir)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(attr.asdict(config), f)


if __name__ == "__main__":
    main()
