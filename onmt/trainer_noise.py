"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import traceback
import numpy as np
import onmt.utils
from onmt.utils.logging import logger
from torch import nn
from torch.nn.functional import sigmoid
from torch import autograd
from torch.autograd import Variable


class EasyLoss:
    def __init__(self, pad_id):
        self.criterion = nn.NLLLoss(ignore_index=pad_id, reduction='none')
        self.pad_id = pad_id
    # def cal(self, pred, ground_truth):
    #     sizes = list(pred.size())
    #     print('sizes: {}'.format(sizes))
    #     print('gt size: {}'.format(ground_truth.size()))
    #     pat_pred = pred.contiguous().view(-1, sizes[-1])
    #     pat_gt = ground_truth.contiguous().view(-1)
    #     loss = self.criterion(pat_pred, pat_gt)
    #     loss = loss.contiguous().view(sizes[0], sizes[1])
    #     loss = torch.sum(loss, 1)
    #     return loss
    def cal(self, pred, ground_truth):
        sizes = list(pred.size())
        pat_pred = pred.contiguous().view(-1, sizes[-1])
        pat_gt = ground_truth.contiguous().view(-1)
        loss = self.criterion(pat_pred, pat_gt)
        loss = loss.contiguous().view(sizes[0], sizes[1])
        gt_lens = torch.sum(ground_truth.ne(self.pad_id).float(), 1)
        loss = torch.sum(loss, 1) / gt_lens
        return loss

def build_trainer(opt, device_id,
                  g_model, m_model, n_model, g_optim, m_optim, n_optim,
                  m_mask_id, m_pad_id, m_bos_id, m_eos_id,
                  g_fields, m_fields, g_model_saver=None, m_model_saver=None, n_model_save_path=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    g_tgt_field = dict(g_fields)["tgt"].base_field
    m_tgt_field = dict(m_fields)["tgt"].base_field
    g_train_loss = onmt.utils.loss.build_loss_compute(g_model, g_tgt_field, opt)
    g_valid_loss = onmt.utils.loss.build_loss_compute(g_model, g_tgt_field, opt, train=False)
    m_train_loss = onmt.utils.loss.build_loss_compute(m_model, m_tgt_field, opt)
    m_valid_loss = onmt.utils.loss.build_loss_compute(m_model, m_tgt_field, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = opt.world_size
    average_decay = opt.average_decay
    average_every = opt.average_every
    dropout = opt.dropout
    dropout_steps = opt.dropout_steps
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level
    wh_sample_num = opt.wh_sample_num
    wh_disturb_ratio = opt.wh_disturb_ratio
    fgsm_alpha = opt.fgsm_alpha
    gm_iter_count = opt.gm_iter_count

    earlystopper = onmt.utils.EarlyStopping(opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = Trainer(g_model, m_model, n_model,
                           g_train_loss, m_train_loss, g_valid_loss, m_valid_loss,
                           g_optim, m_optim, n_optim,
                           m_mask_id, m_pad_id, m_bos_id, m_eos_id,
                           trunc_size,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           wh_sample_num=wh_sample_num, wh_disturb_ratio=wh_disturb_ratio, fgsm_alpha=fgsm_alpha,
                           with_align=True if opt.lambda_align > 0 else False,
                           g_model_saver=g_model_saver if gpu_rank == 0 else None,
                           m_model_saver=m_model_saver if gpu_rank == 0 else None,
                           n_model_save_path=n_model_save_path,
                           average_decay=average_decay,
                           average_every=average_every,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           dropout=dropout,
                           dropout_steps=dropout_steps, gm_iter_count=gm_iter_count, train_config=opt)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, g_model, m_model, n_model,
                 g_train_loss, m_train_loss, g_valid_loss, m_valid_loss,
                 g_optim, m_optim, n_optim,
                 m_mask_id, m_pad_id, m_bos_id, m_eos_id,
                 trunc_size=0, shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0,
                 report_manager=None,
                 wh_sample_num=5, wh_disturb_ratio=0.1, fgsm_alpha=1.0,
                 with_align=False, g_model_saver=None, m_model_saver=None, n_model_save_path=None,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0], gm_iter_count=8,
                 train_config=None):
        # Basic attributes.
        # TODO new here
        self.g_model = g_model
        self.m_model = m_model
        self.g_optim = g_optim
        self.m_optim = m_optim
        self.g_train_loss = g_train_loss
        self.m_train_loss = m_train_loss
        self.g_valid_loss = g_valid_loss
        self.m_valid_loss = m_valid_loss
        self.g_model_saver = g_model_saver
        self.m_model_saver = m_model_saver
        self.m_pad_id = m_pad_id
        self.m_mask_id = m_mask_id
        self.m_bos_id = m_bos_id
        self.m_eos_id = m_eos_id
        self.data_device = None
        self.m_accum_count = 1  # fix, do not change, since we do not accumulate gradients over multiple batches
        self.g_accum_count = 1  # fix, do not change, since we do not accumulate gradients over multiple batches
        # self.gm_iter_count = 8  # interactively train forward net and backward net with gm_iter_count batches
        self.gm_iter_count = gm_iter_count  # interactively train forward net and backward net with gm_iter_count batches
        self.easy_loss_fn = EasyLoss(m_pad_id)
        self.wh_sample_num = wh_sample_num
        self.wh_disturb_ratio = wh_disturb_ratio
        self.fgsm_alpha = fgsm_alpha

        self.n_model = n_model
        self.n_optim = n_optim
        self.n_model_save_path = n_model_save_path
        self.n_loss_accu = 0.0
        self.n_loss_g_accu = 0.0
        self.n_loss_m_accu = 0.0
        self.n_sigma_norm = 0.0
        self.n_step_accu = 0.0

        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.with_align = with_align
        self.average_decay = average_decay
        self.moving_average = None
        self.average_every = average_every
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.dropout = dropout
        self.dropout_steps = dropout_steps
        self.report_every = 50
        try:
            self.report_every = train_config.report_every
        except Exception as e:
            pass

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.g_model.train()
        self.m_model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    # def _g_accum_batches(self, iterator):
    #     batches = []
    #     normalization = 0
    #     self.g_accum_count = self._accum_count(self.g_optim.training_step)
    #     for batch in iterator:
    #         batches.append(batch)
    #         if self.norm_method == "tokens":
    #             num_tokens = batch.tgt[1:, :, 0].ne(self.g_train_loss.padding_idx).sum()
    #             normalization += num_tokens.item()
    #         else:
    #             normalization += batch.batch_size
    #         if len(batches) == self.g_accum_count:
    #             yield batches, normalization
    #             self.g_accum_count = self._accum_count(self.g_optim.training_step)
    #             batches = []
    #             normalization = 0
    #     if batches:
    #         yield batches, normalization
    def _new_accum_batches(self, iterator):
        """ extract B2 and A3, use the M dataset """
        batches = []
        other_datas = []
        normalization = 0
        for batch in iterator:
            old_src = batch.src
            old_tgt = batch.tgt
            if self.data_device is None:
                self.data_device = old_tgt.device
            o_src_ids = old_src[0][:, :, 0].transpose(0, 1).tolist()  # [len, N, 1] -- > [N, len]
            o_src_lens = old_src[1].tolist()  # [N]
            o_tgt_ids = old_tgt[:, :, 0].transpose(0, 1).tolist()  # [len, N, 1] -- > [N, len]
            o_tgt_lens = batch.tgt[:, :, 0].ne(self.m_pad_id).transpose(0, 1).sum(1).tolist()
            a1b1a2_ids, b2_ids, a3_ids = [], [], []
            for osids, oslen, otids, otlen in zip(o_src_ids, o_src_lens, o_tgt_ids, o_tgt_lens):
                midx = osids.index(self.m_mask_id)
                a1b1a2_ids.append(osids[:midx])
                b2_ids.append(otids[1: otlen])
                a3_ids.append(osids[midx+1: oslen])
            other_datas.append((a1b1a2_ids, b2_ids, a3_ids))
            batches.append(batch)

            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(self.g_train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.g_accum_count:
                yield batches, normalization, other_datas
                batches = []
                other_datas = []
                normalization = 0
        if batches:
            yield batches, normalization, other_datas

    def _wrap_g_datas(self, batch_datas):
        a1b1a2_ids, b2_ids, a3_ids = batch_datas
        src_ids = [p1 + p2[:-1] + [self.m_mask_id] for p1, p2 in zip(a1b1a2_ids, b2_ids)]
        tgt_ids = [[self.m_bos_id] + p1[1:] + [self.m_eos_id] for p1 in a3_ids]
        src_lens = [len(x) for x in src_ids]
        tgt_lens = [len(x) for x in tgt_ids]
        max_src_len = int(max(src_lens))
        max_tgt_len = int(max(tgt_lens))
        batch_size = len(src_ids)
        src_ids_tensor = torch.full([batch_size, max_src_len], self.m_pad_id).long()
        tgt_ids_tensor = torch.full([batch_size, max_tgt_len], self.m_pad_id).long()
        for idx in range(batch_size):
            src_ids_tensor[idx, :src_lens[idx]] = torch.LongTensor(src_ids[idx])[:]
            tgt_ids_tensor[idx, : tgt_lens[idx]] = torch.LongTensor(tgt_ids[idx])[:]
        src_lengths = torch.LongTensor(src_lens)
        src_ids_tensor = src_ids_tensor.to(self.data_device)
        tgt_ids_tensor = tgt_ids_tensor.to(self.data_device)
        src_lengths = src_lengths.to(self.data_device)
        return (src_ids_tensor.transpose(0, 1).unsqueeze(2), src_lengths, tgt_ids_tensor.transpose(0, 1).unsqueeze(2))
    def _repack_m_datas_from_g(self, a1b1a2_ids, a3_ids):
        src_ids = [p1 + [self.m_mask_id] + p2[:-1] for p1, p2 in zip(a1b1a2_ids, a3_ids)]
        src_lens = [len(x) for x in src_ids]
        max_src_len = int(max(src_lens))
        batch_size = len(src_ids)
        src_ids_tensor = torch.full([batch_size, max_src_len], self.m_pad_id).long()
        for idx in range(batch_size):
            src_ids_tensor[idx, :src_lens[idx]] = torch.LongTensor(src_ids[idx])[:]
        src_lengths = torch.LongTensor(src_lens)
        src_ids_tensor = src_ids_tensor.to(self.data_device)
        src_lengths = src_lengths.to(self.data_device)
        return src_ids_tensor.transpose(0, 1).unsqueeze(2), src_lengths

    # TODO new here
    def new_train(self,
              m_train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              g_valid_iter=None,
              m_valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if g_valid_iter is None and m_valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',  valid_steps)

        g_total_stats = onmt.utils.Statistics()
        g_report_stats = onmt.utils.Statistics()
        m_total_stats = onmt.utils.Statistics()
        m_report_stats = onmt.utils.Statistics()
        m_g_total_stats = onmt.utils.Statistics()
        m_g_report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=g_total_stats.start_time)

        for i, (m_batches, m_normalization, other_datas) in enumerate(self._new_accum_batches(m_train_iter)):
            g_step = self.g_optim.training_step
            m_step = self.m_optim.training_step
            # UPDATE DROPOUT
            self._g_maybe_update_dropout(g_step)
            self._m_maybe_update_dropout(m_step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d n_minibatch %d" % (self.gpu_rank, i + 1, len(m_batches)))

            if self.n_gpu > 1:
                raise ValueError('Error: we do not support multi-gpu anymore!')

            # INFO: train noise model here
            g_batches = []
            m_tgts = []
            g_normalization = 0
            for m_batch, g_batch_data in zip(m_batches, other_datas):
                m_tgts.append(m_batch.tgt)
                g_src_ids, g_src_lens, g_tgt_ids = self._wrap_g_datas(g_batch_data)
                m_batch.src = (g_src_ids, g_src_lens)
                m_batch.tgt = g_tgt_ids
                g_batches.append(m_batch)
                if self.norm_method == "tokens":
                    num_tokens = g_tgt_ids[1:, :, 0].ne(self.m_pad_id).sum()
                    g_normalization += num_tokens.item()
                else:
                    g_normalization += g_src_ids.size(1)
            self.train_noise_model(g_batches, g_normalization, other_datas, m_tgts, g_total_stats, g_report_stats)

            if self.average_decay > 0 and i % self.average_every == 0:
                self._g_update_average(g_step)
                self._m_update_average(m_step)

            g_report_stats = self._maybe_report_training(
                g_step, train_steps,
                self.g_optim.learning_rate(),
                g_report_stats, reporter_name='G')
            m_report_stats = self._maybe_report_training(
                m_step, train_steps,
                self.m_optim.learning_rate(),
                m_report_stats, reporter_name='M')
            m_g_report_stats = self._maybe_report_training(
                m_step, train_steps,
                self.m_optim.learning_rate(),
                m_g_report_stats, reporter_name='M_G')

            if g_valid_iter is not None and g_step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d' % (self.gpu_rank, g_step))
                valid_stats = self.g_validate(g_valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat step %d' % (self.gpu_rank, g_step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d' % (self.gpu_rank, g_step))
                self._report_step(self.g_optim.learning_rate(), g_step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, g_step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break
                    pass
                pass
            if m_valid_iter is not None and m_step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d' % (self.gpu_rank, m_step))
                valid_stats = self.m_validate(m_valid_iter, moving_average=self.moving_average)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat step %d' % (self.gpu_rank, m_step))
                valid_stats = self._maybe_gather_stats(valid_stats)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d' % (self.gpu_rank, m_step))
                self._report_step(self.m_optim.learning_rate(), m_step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, m_step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        logger.info('=' * 40)
                        logger.info('early stop now. ')
                        break
                    pass
                pass
            if (self.g_model_saver is not None and (save_checkpoint_steps != 0 and g_step % save_checkpoint_steps == 0)):
                self.g_model_saver.save(g_step, moving_average=self.moving_average)
            if (self.m_model_saver is not None and (save_checkpoint_steps != 0 and m_step % save_checkpoint_steps == 0)):
                self.m_model_saver.save(m_step, moving_average=self.moving_average)
            n_step = int(self.n_step_accu)
            if (self.n_model_save_path is not None and (save_checkpoint_steps !=0 and n_step % save_checkpoint_steps == 0)):
                save_path = self.n_model_save_path.format(int(self.n_step_accu))
                logger.info('saving model to : {}'.format(save_path))
                torch.save(self.n_model, save_path)
            if train_steps > 0 and g_step >= train_steps and m_step >= train_steps:
                logger.info('=' * 40)
                logger.info('stop now with g_step:{}, m_step:{}. '.format(g_step, m_step))
                break

        if self.g_model_saver is not None:
            self.g_model_saver.save(g_step, moving_average=self.moving_average)
        if self.m_model_saver is not None:
            self.m_model_saver.save(m_step, moving_average=self.moving_average)
        if self.n_model_save_path is not None:
            save_path = self.n_model_save_path.format(int(self.n_step_accu))
            logger.info('saving model to : {}'.format(save_path))
            torch.save(self.n_model, save_path)
        return g_total_stats, m_total_stats

    def _g_maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.g_model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d" % (self.dropout[i], step))
            pass
        pass
    def _m_maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.m_model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d" % (self.dropout[i], step))
            pass
        pass

    def _train_noise_model(self, src, tgt, src_lengths, other_data, m_tgt, bptt=False):
        disturb_steps = self.wh_sample_num
        assert disturb_steps >= 1, 'disturb_steps requires bigger than 0'

        src = src.contiguous()
        src_lengths = src_lengths.contiguous()

        dec_in = tgt[:-1]  # exclude last target token from inputs
        enc_state, memory_bank, lengths = self.g_model.encoder(src, src_lengths)
        d_model = memory_bank.size(2)
        n0, n1 = memory_bank.size(0), memory_bank.size(1)
        m_tgt_cal_loss = m_tgt[1:, :, 0].transpose(0, 1)
        g_tgt_cal_loss = tgt[1:, :, 0].transpose(0, 1)

        h = memory_bank
        h_disturbs = self.n_model(h, n_sample=disturb_steps)

        cur_g_loss = 0.0
        cur_m_loss = 0.0
        cur_gaussian_disturb_strength = 0.0
        loss = None
        for noise in h_disturbs:
            memory_bank = h + noise
            gaussian_strength_raw = torch.norm(noise, 2, 1)
            # print('gaussian_strength_raw: {}'.format(gaussian_strength_raw.tolist()))
            gaussian_strength = torch.mean(gaussian_strength_raw)
            cur_gaussian_disturb_strength += gaussian_strength.item()
            if bptt is False:
                self.g_model.decoder.init_state(src, memory_bank, enc_state)
            dec_out, attns = self.g_model.decoder(dec_in, memory_bank, memory_lengths=lengths,
                                                  with_align=self.with_align)
            vocab_logits = self.g_model.generator(dec_out).transpose(0, 1)
            # the g_loss here is just for analysis, meaningless for optimizing Noise Model
            g_loss = self.easy_loss_fn.cal(vocab_logits, g_tgt_cal_loss)
            g_loss = torch.mean(g_loss)

            pred_ids = torch.argmax(vocab_logits, 2).tolist()
            a3_ids = []
            for sent_ids in pred_ids:
                if self.m_eos_id in sent_ids:
                    idx = sent_ids.index(self.m_eos_id)
                    a3_ids.append(sent_ids[: idx + 1])
                else:
                    a3_ids.append(sent_ids + [self.m_eos_id])
            fake_a3_src_ids, fake_a3_src_lens = self._repack_m_datas_from_g(other_data[0], a3_ids)
            fake_a3_src_ids = fake_a3_src_ids.detach()
            fake_a3_src_lens = fake_a3_src_lens.detach()
            fake_a3_outputs, fake_a3_attns = self.m_model(
                fake_a3_src_ids, m_tgt, fake_a3_src_lens, bptt=False, with_align=self.with_align)
            vocab_logits = self.m_model.generator(fake_a3_outputs).transpose(0, 1)
            m_loss = self.easy_loss_fn.cal(vocab_logits, m_tgt_cal_loss)
            m_loss = torch.mean(m_loss)
            if loss is None:
                loss = m_loss - self.wh_disturb_ratio * gaussian_strength
            else:
                loss = loss + m_loss - self.wh_disturb_ratio * gaussian_strength

            cur_g_loss += g_loss.item()
            cur_m_loss += m_loss.item()
        f_loss = loss / disturb_steps
        self.n_loss_accu += f_loss.item()
        self.n_loss_g_accu += cur_g_loss / disturb_steps
        self.n_loss_m_accu += cur_m_loss / disturb_steps
        self.n_sigma_norm += cur_gaussian_disturb_strength / disturb_steps
        self.n_optim.zero_grad()
        f_loss.backward()
        self.n_optim.step()

        log_steps = self.report_every
        if 0 == self.n_step_accu % log_steps:
            logger.info('Noise step: {}, loss: {}, g_loss: {}, m_loss: {}, lr: {}, sigma: {}'.format(
                self.n_step_accu, self.n_loss_accu / log_steps,
                self.n_loss_g_accu / log_steps, self.n_loss_m_accu / log_steps,
                self.n_optim.learning_rate(), self.n_sigma_norm / log_steps))
            self.n_loss_accu = 0.0
            self.n_loss_g_accu = 0.0
            self.n_loss_m_accu = 0.0
            self.n_sigma_norm = 0.0

    def train_noise_model(self, true_batches, normalization,
                          other_datas, m_tgts, total_stats, report_stats):
        for k in range(len(true_batches)):
            self.n_step_accu += 1

            batch = true_batches[k]
            other_data = other_datas[k]
            m_tgt = m_tgts[k]
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt = batch.tgt
            # 1. Create truncated target.
            # 2. F-prop all but generator.
            self._train_noise_model(src, tgt, src_lengths, other_data, m_tgt, bptt=False)

    def _g_update_average(self, step):
        if self.g_moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.g_model.parameters()]
            self.g_moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.g_moving_average), self.g_model.parameters()):
                self.g_moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay
        pass
    def _m_update_average(self, step):
        if self.m_moving_average is None:
            copy_params = [params.detach().float()
                           for params in self.m_model.parameters()]
            self.m_moving_average = copy_params
        else:
            average_decay = max(self.average_decay, 1 - (step + 1)/(step + 10))
            for (i, avg), cpt in zip(enumerate(self.m_moving_average), self.m_model.parameters()):
                self.m_moving_average[i] = \
                    (1 - average_decay) * avg + \
                    cpt.detach().float() * average_decay
        pass
    def g_validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        logger.info('Validation of G')
        valid_model = self.g_model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.g_optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt
                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths, with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.g_valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.g_model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats
    def m_validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        logger.info('Validation of M')
        valid_model = self.m_model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average, valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.m_optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
                tgt = batch.tgt
                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths, with_align=self.with_align)

                # Compute loss.
                _, batch_stats = self.m_valid_loss(batch, outputs, attns)

                # Update statistics.
                stats.update(batch_stats)
        if moving_average:
            for param_data, param in zip(model_params_data, self.m_model.parameters()):
                param.data = param_data

        # Set model back to training mode.
        valid_model.train()

        return stats
    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats, reporter_name=''):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1, reporter_name=reporter_name)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)


