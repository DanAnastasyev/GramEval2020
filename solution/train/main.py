# -*- coding: utf-8 -*-

import argparse
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
from allennlp.data.token_indexers.token_characters_indexer import TokenCharactersIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2seq_encoders import PassThroughEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import PretrainedTransformerMismatchedEmbedder
from allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from allennlp.training.learning_rate_schedulers.slanted_triangular import SlantedTriangular
from allennlp.training.trainer import Trainer

from train.dataset_reader import UDDatasetReader
from train.lemma_vectorizer import apply_to_instances as apply_lemma_vectorizer_to_instances
from train.lemmatize_helper import LemmatizeHelper
from train.model import DependencyParser, LstmWeightDropSeq2SeqEncoder, TaskConfig
from train.morpho_vectorizer import MorphoVectorizer

logger = logging.getLogger(__name__)


@attr.s
class EmbedderModelConfig(object):
    name = attr.ib(default=['ru_bert'])
    params = attr.ib(default=None)


@attr.s
class EmbedderConfig(object):
    models = attr.ib(default=[EmbedderModelConfig()])
    dropout = attr.ib(default=0.4)
    use_pymorphy = attr.ib(default=False)
    use_lemmas = attr.ib(default=False)


@attr.s
class EncoderConfig(object):
    encoder_type = attr.ib(default='lstm', validator=attr.validators.in_(['lstm', 'none']))
    hidden_dim = attr.ib(default=256)
    num_layers = attr.ib(default=2)
    dropout = attr.ib(default=0.3)
    variational_dropout = attr.ib(default=0.3)
    use_weight_drop = attr.ib(default=False)


@attr.s
class ParserConfig(object):
    dropout = attr.ib(default=0.1)
    tag_representation_dim = attr.ib(default=128)
    arc_representation_dim = attr.ib(default=512)
    gram_val_representation_dim = attr.ib(default=-1)
    lemma_representation_dim = attr.ib(default=-1)


@attr.s
class TrainerConfig(object):
    batch_size = attr.ib(default=128)
    num_epochs = attr.ib(default=15)
    patience = attr.ib(default=10)
    lr = attr.ib(default=1e-3)
    bert_lr = attr.ib(default=1e-4)
    cut_frac = attr.ib(default=0.1)
    gradual_unfreezing = attr.ib(default=True)
    discriminative_fine_tuning = attr.ib(default=True)
    num_gradient_accumulation_steps = attr.ib(default=1)
    loss_dropper_config = attr.ib(default=None)


@attr.s
class DataConfig(object):
    data_dir = attr.ib(default='../data')
    pretrained_models_dir = attr.ib(default='../pretrained_models')
    models_dir = attr.ib(default='../models')
    train_data_all_except = attr.ib(default=None)
    train_data = attr.ib(default=['GramEval2020-GSD-train.conllu'])
    valid_data = attr.ib(default=['GramEval2020-GSD-wiki-dev.conllu'])


@attr.s
class Config(object):
    model_name = attr.ib(default='ru_bert_parser_with_tagger_pymorphy')
    embedder = attr.ib(default=EmbedderConfig())
    encoder = attr.ib(default=EncoderConfig())
    parser = attr.ib(default=ParserConfig())
    trainer = attr.ib(default=TrainerConfig())
    data = attr.ib(default=DataConfig())
    task = attr.ib(default=TaskConfig())

    @classmethod
    def load(cls, path):
        with open(path) as f:
            json_config = json.load(f)

        model_name = json_config['model_name']
        encoder = EncoderConfig(**json_config['encoder'])
        parser = ParserConfig(**json_config['parser'])
        trainer = TrainerConfig(**json_config['trainer'])
        data = DataConfig(**json_config['data'])
        task = TaskConfig(**json_config['task'])

        embedder_config = json_config['embedder']
        embedder_models = [EmbedderModelConfig(**model_config) for model_config in embedder_config['models']]
        embedder = EmbedderConfig(
            models=embedder_models,
            dropout=embedder_config['dropout'],
            use_pymorphy=embedder_config['use_pymorphy'],
            use_lemmas=embedder_config.get('use_lemmas', False),
        )

        return cls(
            model_name=model_name,
            embedder=embedder,
            encoder=encoder,
            parser=parser,
            trainer=trainer,
            data=data,
            task=task
        )


def build_config(config_dir, model, full_data, pretrained_models_dir=None, models_dir=None):
    config = Config.load(os.path.join(config_dir, model + '.json'))

    if full_data:
        config.data.train_data_all_except = ['GramEval2020-SynTagRus-train.conllu', 'GramEval2020-17cent-train.conllu']
        config.data.train_data = None,
        config.data.valid_data = 'all'

    if pretrained_models_dir:
        config.data.pretrained_models_dir = pretrained_models_dir

    if models_dir:
        config.data.models_dir = models_dir

    return config


def _get_reader(config, skip_labels=False, bert_max_length=None, reader_max_length=200, read_first=None):
    indexers = {}
    for embedder_config in config.embedder.models:
        if embedder_config.name == 'elmo':
            indexers[embedder_config.name] = ELMoTokenCharactersIndexer()
        elif embedder_config.name.endswith('bert'):
            bert_path = os.path.join(config.data.pretrained_models_dir, embedder_config.name)
            indexers[embedder_config.name] = PretrainedTransformerMismatchedIndexer(
                model_name=bert_path, tokenizer_kwargs={'do_lower_case': False},
                max_length=bert_max_length
            )
        elif embedder_config.name == 'char_bilstm':
            indexers[embedder_config.name] = TokenCharactersIndexer()
        else:
            assert False, 'Unknown embedder {}'.format(embedder_config.name)

    return UDDatasetReader(indexers, skip_labels=skip_labels, max_length=reader_max_length, read_first=read_first)


def _load_train_data(config):
    reader = _get_reader(config)

    train_data, valid_data = [], []

    if config.data.train_data_all_except:
        for path in os.listdir(os.path.join(config.data.data_dir, 'data_train')):
            if path not in config.data.train_data_all_except:
                if not path.endswith('.conllu'):
                    continue
                logger.info('Loading train file %s', path)
                train_data.extend(reader.read(os.path.join(config.data.data_dir, 'data_train', path)))
    else:
        for path in config.data.train_data:
            logger.info('Loading train file %s', path)
            train_data.extend(reader.read(os.path.join(config.data.data_dir, 'data_train', path)))

    if config.data.valid_data == 'all':
        for path in os.listdir(os.path.join(config.data.data_dir, 'data_open_test')):
            if not path.endswith('.conllu'):
                continue
            logger.info('Loading valid file %s', path)
            valid_data.extend(reader.read(os.path.join(config.data.data_dir, 'data_open_test', path)))
    else:
        for path in config.data.valid_data:
            logger.info('Loading valid file %s', path)
            valid_data.extend(reader.read(os.path.join(config.data.data_dir, 'data_open_test', path)))

    return train_data, valid_data


def _build_lemmatizer(train_data):
    lemmatize_helper = LemmatizeHelper()
    lemmatize_helper.fit(train_data)

    return lemmatize_helper


def _load_embedder(config, vocab, bert_max_length):
    embedders = {}
    for embedder_config in config.embedder.models:
        if embedder_config.name == 'elmo':
            embedders[embedder_config.name] = ElmoTokenEmbedder(
                options_file=os.path.join(config.data.pretrained_models_dir, 'elmo/options.json'),
                weight_file=os.path.join(config.data.pretrained_models_dir, 'elmo/model.hdf5'),
                requires_grad=embedder_config.params['requires_grad'],
                dropout=0.
            )
            embedders[embedder_config.name].eval()
        elif embedder_config.name.endswith('bert'):
            embedders[embedder_config.name] = PretrainedTransformerMismatchedEmbedder(
                model_name=os.path.join(config.data.pretrained_models_dir, embedder_config.name),
                max_length=bert_max_length,
                requires_grad=embedder_config.params['requires_grad']
            )
        elif embedder_config.name == 'char_bilstm':
            embedders[embedder_config.name] = TokenCharactersEncoder(
                embedding=Embedding(
                    num_embeddings=vocab.get_vocab_size('token_characters'),
                    embedding_dim=embedder_config.params['char_embedding_dim']
                ),
                encoder=PytorchSeq2VecWrapper(torch.nn.LSTM(
                    embedder_config.params['char_embedding_dim'],
                    embedder_config.params['lstm_dim'],
                    num_layers=embedder_config.params['lstm_num_layers'],
                    dropout=embedder_config.params['lstm_dropout'],
                    bidirectional=True, batch_first=True
                )),
                dropout=embedder_config.params['dropout']
            )
        else:
            assert False, 'Unknown embedder {}'.format(embedder_config.name)

    return BasicTextFieldEmbedder(embedders)


def _build_model(config, vocab, lemmatize_helper, morpho_vectorizer, bert_max_length=None):
    embedder = _load_embedder(config, vocab, bert_max_length)

    input_dim = embedder.get_output_dim()
    if config.embedder.use_pymorphy:
        input_dim += morpho_vectorizer.morpho_vector_dim
    if config.embedder.use_lemmas:
        input_dim += lemmatize_helper.lemmatize_rule_count()

    pos_tag_embedding = None
    if config.task.task_type == 'single' and config.task.params['use_pos_tag']:
        pos_tag_embedding = Embedding(
            num_embeddings=vocab.get_vocab_size('grammar_value_tags'),
            embedding_dim=config.task.params['pos_embedding_dim']
        )
        input_dim += config.task.params['pos_embedding_dim']

    encoder = None
    if config.encoder.encoder_type != 'lstm':
        encoder = PassThroughEncoder(input_dim=input_dim)
    elif config.encoder.use_weight_drop:
        encoder = LstmWeightDropSeq2SeqEncoder(
            input_dim, config.encoder.hidden_dim, num_layers=config.encoder.num_layers, bidirectional=True,
            dropout=config.encoder.dropout, variational_dropout=config.encoder.variational_dropout
        )
    else:
        encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(
            input_dim + 16, config.encoder.hidden_dim, num_layers=config.encoder.num_layers,
            dropout=config.encoder.dropout, bidirectional=True, batch_first=True
        ))

    return DependencyParser(
        vocab=vocab,
        text_field_embedder=embedder,
        encoder=encoder,
        lemmatize_helper=lemmatize_helper,
        task_config=config.task,
        pos_tag_embedding=pos_tag_embedding,
        morpho_vector_dim=morpho_vectorizer.morpho_vector_dim if config.embedder.use_pymorphy else 0,
        lemma_vector_dim=lemmatize_helper.lemmatize_rule_count() if config.embedder.use_lemmas else 0,
        tag_representation_dim=config.parser.tag_representation_dim,
        arc_representation_dim=config.parser.arc_representation_dim,
        dropout=config.parser.dropout,
        input_dropout=config.embedder.dropout,
        gram_val_representation_dim=config.parser.gram_val_representation_dim,
        lemma_representation_dim=config.parser.lemma_representation_dim,
        loss_dropper_config=config.trainer.loss_dropper_config,
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

        if batch['words']['ru_bert']['token_ids'].shape[1] > 256:
            return False

        return all(
            begin <= end < batch['words']['ru_bert']['token_ids'].shape[1]
            for begin, end in batch['words']['ru_bert']['offsets'][0]
        )

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
    optimizer = optim.AdamW(model.parameters(), lr=config.trainer.lr)
    scheduler = None

    is_bert_based = any(model.name.endswith('bert') for model in config.embedder.models)
    is_trainable_elmo_based = any(
        model.name == 'elmo' and model.params['requires_grad']
        for model in config.embedder.models
    )

    if is_bert_based or is_trainable_elmo_based:
        def _is_pretrained_param(name):
            return 'transformer_model' in name or '_elmo_lstm' in name

        pretrained_params, non_pretrained_params = [], []
        for name, param in model.named_parameters():
            if _is_pretrained_param(name):
                logger.info('Pretrained param: %s', name)
                pretrained_params.append(param)
            else:
                logger.info('Non-pretrained param: %s', name)
                non_pretrained_params.append(param)

        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': config.trainer.bert_lr},
            {'params': non_pretrained_params, 'lr': config.trainer.lr},
            {'params': []}
        ])

        scheduler = SlantedTriangular(
            optimizer=optimizer,
            num_epochs=config.trainer.num_epochs,
            num_steps_per_epoch=len(train_data) / config.trainer.batch_size,
            cut_frac=config.trainer.cut_frac,
            gradual_unfreezing=config.trainer.gradual_unfreezing,
            discriminative_fine_tuning=config.trainer.discriminative_fine_tuning
        )

    logger.info('Trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info('\t' + name)

    iterator = BucketIterator(batch_size=config.trainer.batch_size)
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

    if is_bert_based:
        train_data = _filter_data(train_data, vocab)
        valid_data = _filter_data(valid_data, vocab)

    return Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_data,
        validation_dataset=valid_data,
        validation_metric='+MeanAcc',
        patience=config.trainer.patience,
        num_epochs=config.trainer.num_epochs,
        cuda_device=cuda_device,
        grad_clipping=5.,
        learning_rate_scheduler=scheduler,
        serialization_dir=os.path.join(config.data.models_dir, config.model_name),
        should_log_parameter_statistics=False,
        should_log_learning_rate=False,
        num_gradient_accumulation_steps=config.trainer.num_gradient_accumulation_steps
    )


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config-dir', default='train/configs', help='Path to the directory with configs')
    parser.add_argument('--model', default='bert', help='Name of the config to use')
    parser.add_argument(
        '--pretrained-models-dir', default=None, help='Path to directory with pretrained models (e.g., RuBERT)'
    )
    parser.add_argument('--models-dir', default=None, help='Path to directory to save the model')
    parser.add_argument(
        '--full-data', default=False, action='store_true',
        help='Whether to train the model on the full train data'
             ' (it\'s useful when you\'re too lazy to specify all files)'
    )
    parser.add_argument('--finetune-from', default=None)
    args = parser.parse_args()

    config = build_config(args.config_dir, args.model, args.full_data, args.pretrained_models_dir, args.models_dir)
    logger.info('Config: %s', config)

    model_dir = os.path.join(config.data.models_dir, config.model_name)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(attr.asdict(config), f, indent=2, ensure_ascii=False)

    train_data, valid_data = _load_train_data(config)
    logger.info('Train data size = %s, valid data size = %s', len(train_data), len(valid_data))

    lemmatize_helper = _build_lemmatizer(train_data)
    lemmatize_helper.apply_to_instances(chain(train_data, valid_data))

    morpho_vectorizer = None
    if config.embedder.use_pymorphy:
        morpho_vectorizer = MorphoVectorizer()
        morpho_vectorizer.apply_to_instances(chain(train_data, valid_data))

    if config.embedder.use_lemmas:
        apply_lemma_vectorizer_to_instances(lemmatize_helper, train_data)
        apply_lemma_vectorizer_to_instances(lemmatize_helper, valid_data)

    vocab = Vocabulary.from_files('../models/span_normalization/vocab')
    model = _build_model(config, vocab, lemmatize_helper, morpho_vectorizer)

    if args.finetune_from:
        logger.info('Loading model from %s', args.finetune_from)
        skip_weights = {
            '_head_sentinel',
            'head_arc_feedforward._linear_layers.0.weight',
            'child_arc_feedforward._linear_layers.0.weight',
            'head_tag_feedforward._linear_layers.0.weight',
            'child_tag_feedforward._linear_layers.0.weight',
            '_gram_val_output.weight',
            '_gram_val_output.bias',
            '_lemma_output.weight',
            '_lemma_output.bias',
        }
        state_dict = torch.load(os.path.join(args.finetune_from, 'best.th'), map_location='cpu')
        state_dict = {key: val for key, val in state_dict.items() if key not in skip_weights}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        logger.info('Missing keys:')
        logger.info('\n'.join('\t' + key for key in missing_keys))
        logger.info('\n'.join('\t' + key for key in unexpected_keys))

    vocab.save_to_files(os.path.join(model_dir, 'vocab'))
    lemmatize_helper.save(model_dir)

    trainer = _build_trainer(config, model, vocab, train_data, valid_data)
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info('Early stopping was triggered...')


if __name__ == "__main__":
    main()
