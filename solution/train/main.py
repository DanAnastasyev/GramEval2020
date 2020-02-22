# -*- coding: utf-8 -*-

import attr
import json
import logging
import os
import torch
import torch.optim as optim

from itertools import chain
from tqdm import tqdm

from allennlp.data.dataset_readers.universal_dependencies import UniversalDependenciesDatasetReader
from allennlp.data.fields import ArrayField, SequenceLabelField
from allennlp.data.iterators import BasicIterator, BucketIterator
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers.pretrained_transformer_mismatched_indexer import PretrainedTransformerMismatchedIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper, PassThroughEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
from allennlp.training.learning_rate_schedulers.noam import NoamLR
from allennlp.training.trainer import Trainer

from train.dataset_reader import UDDatasetReader
from train.lemmatize_helper import LemmatizeHelper
from train.model import DependencyParser
from train.morpho_vectorizer import MorphoVectorizer

logger = logging.getLogger(__name__)


@attr.s
class Config(object):
    model_name = attr.ib(default='ru_bert_parser_with_tagger')
    embedder_name = attr.ib(default='ru_bert')
    use_pymorphy = attr.ib(default=False)
    batch_size = attr.ib(default=32)
    num_epochs = attr.ib(default=15)
    patience = attr.ib(default=10)
    data_dir = attr.ib(default='../data')
    train_data_all_except = attr.ib(default=None)
    train_data = attr.ib(default=['GramEval2020-GSD-train.conllu'])
    valid_data = attr.ib(default=['GramEval2020-GSD-wiki-dev.conllu'])
    pretrained_models_dir = attr.ib(default='../pretrained_models')
    models_dir = attr.ib(default='../models')


_SAMPLE_DATA_CONFIG = Config()

_FULL_DATA_CONFIG = Config(
    train_data_all_except=['GramEval2020-SynTagRus-train.conllu', 'GramEval2020-17cent-train.conllu'],
    train_data=None,
    valid_data='all'
)


def _get_reader(config):
    indexer = None
    if config.embedder_name == 'elmo':
        indexer = ELMoTokenCharactersIndexer()
    elif config.embedder_name.endswith('bert'):
        bert_path = os.path.join(config.pretrained_models_dir, config.embedder_name)
        indexer = PretrainedTransformerMismatchedIndexer(
            model_name=bert_path, tokenizer_kwargs={'do_lower_case': False}
        )
    else:
        assert False, 'Unknown embedder {}'.format(config.embedder_name)

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


def _build_lemmatizer(train_data):
    lemmatize_helper = LemmatizeHelper()
    lemmatize_helper.fit(train_data)

    return lemmatize_helper


def _apply_lemmatizer(lemmatize_helper, data):
    for instance in data:
        lemmatize_rule_indices = lemmatize_helper.get_rule_indices(instance)
        field = SequenceLabelField(lemmatize_rule_indices, instance.fields['words'], 'lemma_index_tags')
        instance.add_field('lemma_indices', field)


def _apply_morpho_vectorizer(morpho_vectorizer, data):
    for instance in data:
        grammar_matrix = morpho_vectorizer.vectorize_instance(instance)
        instance.add_field('morpho_embedding', ArrayField(grammar_matrix))

    return morpho_vectorizer


def _load_embedder(config):
    if config.embedder_name == 'elmo':
        embedder = ElmoTokenEmbedder(
            options_file=os.path.join(config.pretrained_models_dir, 'elmo/options.json'),
            weight_file=os.path.join(config.pretrained_models_dir, 'elmo/model.hdf5'),
            dropout=0.3
        )
        embedder.eval()
    else:
        embedder = PretrainedTransformerMismatchedEmbedder(
            model_name=os.path.join(config.pretrained_models_dir, config.embedder_name)
        )

    return BasicTextFieldEmbedder({config.embedder_name: embedder})


def _build_model(config, vocab, lemmatize_helper, morpho_vectorizer):
    embedder = _load_embedder(config)

    # TODO: AWD-LSTM Dropout
    input_dim = embedder.get_output_dim()
    if config.use_pymorphy:
        input_dim += morpho_vectorizer.morpho_vector_dim

    encoder = None
    if config.embedder_name == 'elmo':
        encoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(input_dim, 256, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        )
    elif config.embedder_name.endswith('bert'):
        encoder = PassThroughEncoder(input_dim=768)
    else:
        assert False

    return DependencyParser(
        vocab=vocab,
        text_field_embedder=embedder,
        encoder=encoder,
        lemmatize_helper=lemmatize_helper,
        morpho_vector_dim=morpho_vectorizer.morpho_vector_dim if config.use_pymorphy else 0,
        tag_representation_dim=128,
        arc_representation_dim=128,
        dropout=0.,
        input_dropout=0.3
    )


def _log_batch(batch):
    if isinstance(batch, tuple):
        key, value = batch

        if isinstance(value, torch.Tensor):
            logger.info('%s -> %s', key, value.shape)
            return
        if isinstance(value, list):
            logger.info('%s -> %s', key, len(value))
            return
        batch = value

    if not isinstance(batch, dict):
        return

    for key, value in batch.items():
        _log_batch((key, value))


def _filter_data(data, vocab):
    def _is_correct_instance(batch):
        assert len(batch['words']['ru_bert']['offsets']) == 1
        return all(begin <= end for begin, end in batch['words']['ru_bert']['offsets'][0])

    iterator = BasicIterator(batch_size=1)
    iterator.index_with(vocab)

    result_data = []
    for instance in tqdm(data):
        batch = next(iterator([instance]))
        if _is_correct_instance(batch):
            result_data.append(instance)
        else:
            logger.info('Filtering out %s', batch['metadata'][0]['words'])

    logger.info('Removed %s samples', len(data) - len(result_data))
    return result_data


def _build_trainer(config, model, vocab, train_data, valid_data):
    optimizer = optim.AdamW(model.parameters())
    scheduler = None

    if config.embedder_name.endswith('bert'):
        non_bert_params = (
            param for name, param in model.named_parameters() if not name.startswith('text_field_embedder')
        )

        optimizer = optim.AdamW([
            {'params': model.text_field_embedder.parameters(), 'lr': 1e-5},
            {'params': non_bert_params, 'lr': 1e-3}
        ])

    iterator = BucketIterator(batch_size=config.batch_size)
    iterator.index_with(vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
        logger.info('Using cuda')
    else:
        cuda_device = -1
        logger.info('Using cpu')

    logger.info('Example batch:')
    _log_batch(next(iterator(train_data)))

    if config.embedder_name.endswith('bert'):
        train_data = _filter_data(train_data, vocab)
        valid_data = _filter_data(valid_data, vocab)

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
        learning_rate_scheduler=scheduler,
        serialization_dir=os.path.join(config.models_dir, config.model_name),
        should_log_parameter_statistics=False,
        should_log_learning_rate=True
    )


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    config = _FULL_DATA_CONFIG

    train_data, valid_data = _load_train_data(config)
    logger.info('Train data size = %s, valid data size = %s', len(train_data), len(valid_data))

    lemmatize_helper = _build_lemmatizer(train_data)
    _apply_lemmatizer(lemmatize_helper, chain(train_data, valid_data))

    morpho_vectorizer = None
    if config.use_pymorphy:
        morpho_vectorizer = MorphoVectorizer()
        _apply_morpho_vectorizer(morpho_vectorizer, chain(train_data, valid_data))

    vocab = Vocabulary.from_instances(chain(train_data, valid_data))
    logger.info('Vocab = %s', vocab)
    vocab.print_statistics()

    model = _build_model(config, vocab, lemmatize_helper, morpho_vectorizer)
    logger.info('Model:\n%s', model)

    trainer = _build_trainer(config, model, vocab, train_data, valid_data)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info('Early stopping was triggered...')

    model_dir = os.path.join(config.models_dir, config.model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    vocab.save_to_files(os.path.join(model_dir, 'vocab'))
    lemmatize_helper.save(model_dir)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(attr.asdict(config), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
