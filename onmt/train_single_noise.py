#!/usr/bin/env python
"""Training on a single process."""
import os

import torch

from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer_noise import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.noise_generator import Noise


def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

def _new_check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    g_model_save_dir = os.path.join(model_dirname, 'g_model')
    m_model_save_dir = os.path.join(model_dirname, 'm_model')
    n_model_save_dir = os.path.join(model_dirname, 'n_model')
    if not os.path.exists(g_model_save_dir):
        os.makedirs(g_model_save_dir)
    if not os.path.exists(m_model_save_dir):
        os.makedirs(m_model_save_dir)
    if not os.path.exists(n_model_save_dir):
        os.makedirs(n_model_save_dir)
    return g_model_save_dir, m_model_save_dir, n_model_save_dir, model_dirname

def _tally_parameters(model):
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        else:
            dec += param.nelement()
    return enc + dec, enc, dec


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id, batch_queue=None, semaphore=None):
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    g_model_save_dir, m_model_save_dir, n_model_save_dir, model_dirname = _new_check_save_model_path(opt)
    # opt.log_file = os.path.join(model_dirname, 'train.log')
    init_logger(opt.log_file, maxBytes=0, backupCount=0)
    assert len(opt.accum_count) == len(opt.accum_steps), \
        'Number of accum_count values must match number of accum_steps'
    # Load checkpoint if we resume from a previous training.
    if opt.g_train_from:
        logger.info('Loading checkpoint from %s' % opt.g_train_from)
        g_checkpoint = torch.load(opt.g_train_from, map_location=lambda storage, loc: storage)
        #g_checkpoint = torch.load(opt.g_train_from)
        # g_model_opt = ArgumentParser.ckpt_model_opts(g_checkpoint["opt"])
        # ArgumentParser.update_model_opts(g_model_opt)
        # ArgumentParser.validate_model_opts(g_model_opt)
        # logger.info('Loading vocab from checkpoint at %s.' % opt.g_train_from)
        # g_vocab = g_checkpoint['vocab']
    else:
        g_checkpoint = None
    if opt.m_train_from:
        logger.info('Loading checkpoint from %s' % opt.m_train_from)
        m_checkpoint = torch.load(opt.m_train_from, map_location=lambda storage, loc: storage)
        # m_model_opt = ArgumentParser.ckpt_model_opts(m_checkpoint["opt"])
        # ArgumentParser.update_model_opts(m_model_opt)
        # ArgumentParser.validate_model_opts(m_model_opt)
        # logger.info('Loading vocab from checkpoint at %s.' % opt.m_train_from)
        # m_vocab = m_checkpoint['vocab']
    else:
        m_checkpoint = None
    g_model_opt = opt
    m_model_opt = opt
    m_vocab = torch.load(opt.data + '.vocab.pt')
    g_vocab = m_vocab
    g_mask_id = g_vocab['src'].fields[0][1].vocab.stoi['<mask>']
    m_mask_id = m_vocab['src'].fields[0][1].vocab.stoi['<mask>']
    # m_pad_id = m_vocab['src'].fields[0][1].vocab.stoi[m_vocab['src'].fields[0][1].pad_token]
    m_pad_id = m_vocab['src'].fields[0][1].vocab.stoi['<blank>']
    m_bos_id = m_vocab['src'].fields[0][1].vocab.stoi['<s>']
    m_eos_id = m_vocab['src'].fields[0][1].vocab.stoi['</s>']
    logger.info('=' * 40)
    logger.info('g_mask_id:{}, m_mask_id:{}, pad_id:{}, bos_id:{}, eos_id: {}'.format(g_mask_id, m_mask_id, m_pad_id, m_bos_id, m_eos_id))
    logger.info('=' * 40)
    # check for code where vocab is saved instead of fields
    # (in the future this will be done in a smarter way)
    if old_style_vocab(g_vocab):
        g_fields = load_old_vocab(g_vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        g_fields = g_vocab
    if old_style_vocab(m_vocab):
        m_fields = load_old_vocab(m_vocab, opt.model_type, dynamic_dict=opt.copy_attn)
    else:
        m_fields = m_vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = g_fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    # Build model.
    g_model = build_model(g_model_opt, opt, g_fields, g_checkpoint)
    m_model = build_model(m_model_opt, opt, m_fields, m_checkpoint)
    logger.info('=' * 80)
    logger.info('G Model')
    n_params, enc, dec = _tally_parameters(g_model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    logger.info('=' * 80)
    logger.info('M Model')
    n_params, enc, dec = _tally_parameters(m_model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    # Build optimizer.
    g_optim = Optimizer.from_opt(g_model, opt, checkpoint=g_checkpoint)
    m_optim = Optimizer.from_opt(m_model, opt, checkpoint=m_checkpoint)
    n_checkpoint = None
    n_model = Noise(opt.enc_rnn_size)
    n_model = n_model.cuda()
    n_optim = Optimizer.from_opt(n_model, opt, checkpoint=n_checkpoint)

    # Build model saver
    g_save_model = os.path.join(g_model_save_dir, 'md')
    opt.save_model = g_save_model
    g_model_saver = build_model_saver(g_model_opt, opt, g_model, g_fields, g_optim)
    m_save_model = os.path.join(m_model_save_dir, 'md')
    opt.save_model = m_save_model
    m_model_saver = build_model_saver(m_model_opt, opt, m_model, m_fields, m_optim)
    n_model_save_path = os.path.join(n_model_save_dir, 'md-{}.pt')

    trainer = build_trainer(opt, device_id, g_model, m_model, n_model, g_optim, m_optim, n_optim,
                            m_mask_id, m_pad_id, m_bos_id, m_eos_id,
                            g_fields, m_fields, g_model_saver=g_model_saver, m_model_saver=m_model_saver, n_model_save_path=n_model_save_path)
    m_train_iter, m_valid_iter = _get_data_iter(batch_queue, opt, m_fields, semaphore, use_shuffle=True)
    g_valid_iter = _get_step1_valid_iter(opt, m_fields)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.new_train(
        m_train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        g_valid_iter=g_valid_iter,
        m_valid_iter=m_valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()


def _get_data_iter(batch_queue, opt, fields, semaphore, use_shuffle=True):
    if batch_queue is None:
        if len(opt.data_ids) > 1:
            train_shards = []
            for train_id in opt.data_ids:
                shard_base = "train_" + train_id
                train_shards.append(shard_base)
            train_iter = build_dataset_iter_multiple(train_shards, fields, opt)
        else:
            if opt.data_ids[0] is not None:
                shard_base = "train_" + opt.data_ids[0]
            else:
                shard_base = "train"
            train_iter = build_dataset_iter(shard_base, fields, opt, is_train=use_shuffle)
    else:
        assert semaphore is not None, "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                yield batch

        train_iter = _train_iter()

    valid_iter = build_dataset_iter(
        "valid", fields, opt, is_train=False)
    return train_iter, valid_iter


def _get_step1_valid_iter(opt, fields):
    valid_iter = build_dataset_iter(
        "g_valid", fields, opt, is_train=False)
    return valid_iter

