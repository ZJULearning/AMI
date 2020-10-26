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

import onmt.utils
from onmt.utils.logging import logger
from torch import nn
from torch.nn.functional import sigmoid
from torch.nn.utils import clip_grad_norm_


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


def build_trainer(
        opt, device_id,
        g_model, m_model, n_model, g_optim, m_optim, n_optim,
        fields, g_model_saver=None, m_model_saver=None, n_model_save_path=None,
        bos_id=2, pad_id=1, eos_id=3, wh_sample_num=5, wh_disturb_ratio=0.1, only_step3=False):
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

    tgt_field = dict(fields)["tgt"].base_field
    g_train_loss = onmt.utils.loss.build_loss_compute(g_model, tgt_field, opt)
    g_valid_loss = onmt.utils.loss.build_loss_compute(g_model, tgt_field, opt, train=False)
    m_train_loss = onmt.utils.loss.build_loss_compute(m_model, tgt_field, opt)
    m_valid_loss = onmt.utils.loss.build_loss_compute(m_model, tgt_field, opt, train=False)

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
    gm_iter_count = opt.gm_iter_count
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level
    noise_lambda = opt.noise_lambda
    noise_l2 = opt.noise_l2
    bw_adv_w_clip = opt.bw_adv_w_clip
    fw_adv_w_clip = opt.fw_adv_w_clip

    earlystopper = onmt.utils.EarlyStopping(opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) if opt.early_stopping > 0 else None

    report_manager = onmt.utils.build_report_manager(opt, gpu_rank)
    trainer = Trainer(
        g_model, g_train_loss, g_valid_loss, g_optim,
        m_model, m_train_loss, m_valid_loss, m_optim,
        n_model, n_optim,
        trunc_size, shard_size, norm_method, accum_count, accum_steps,
        n_gpu, gpu_rank, gpu_verbose_level, report_manager,
        with_align=True if opt.lambda_align > 0 else False,
        g_model_saver=g_model_saver, m_model_saver=m_model_saver, n_model_save_path=n_model_save_path,
        pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, wh_sample_num=wh_sample_num, wh_disturb_ratio=wh_disturb_ratio,
        noise_lambda=noise_lambda, noise_l2=noise_l2, bw_adv_w_clip=bw_adv_w_clip, fw_adv_w_clip=fw_adv_w_clip,
        only_step3=only_step3,
        average_decay=average_decay, average_every=average_every, model_dtype=opt.model_dtype,
        earlystopper=earlystopper, dropout=dropout, dropout_steps=dropout_steps, gm_iter_count=gm_iter_count,
        train_config=opt
    )
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

    def __init__(self,
                 g_model, g_train_loss, g_valid_loss, g_optim,
                 m_model, m_train_loss, m_valid_loss, m_optim,
                 n_model, n_optim,
                 trunc_size=0, shard_size=32, norm_method="sents", accum_count=[1], accum_steps=[0],
                 n_gpu=1, gpu_rank=1, gpu_verbose_level=0, report_manager=None,
                 with_align=False,
                 g_model_saver=None, m_model_saver=None, n_model_save_path=None,
                 pad_id=1, bos_id=2, eos_id=3, wh_sample_num=5, wh_disturb_ratio=0.1,
                 noise_lambda=1.0, noise_l2=1.0, bw_adv_w_clip=1.0, fw_adv_w_clip=1.0,
                 only_step3=False,
                 average_decay=0, average_every=1, model_dtype='fp32',
                 earlystopper=None, dropout=[0.3], dropout_steps=[0], gm_iter_count=8, train_config=None
                 ):
        # Basic attributes.
        self.g_model = g_model
        self.g_train_loss = g_train_loss
        self.g_valid_loss = g_valid_loss
        self.g_optim = g_optim
        self.m_model = m_model
        self.m_train_loss = m_train_loss
        self.m_valid_loss = m_valid_loss
        self.m_optim = m_optim
        self.n_model = n_model
        self.n_optim = n_optim

        self.g_model_saver = g_model_saver
        self.m_model_saver = m_model_saver
        self.n_model_save_path = n_model_save_path
        self.n_loss_accu = 0.0
        self.n_l2_accu = 0.0
        self.n_loss_g_accu = 0.0
        self.n_loss_m_accu = 0.0
        self.n_step_accu = 0.0
        self.disturb_type = 'Noise'
        self.only_step3 = only_step3

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.m_eos_id = eos_id

        self.data_device = None
        self.easy_loss_fn = EasyLoss(pad_id)
        self.g_adv_train_flag = False
        self.m_adv_train_flag = False
        self.wh_sample_num = wh_sample_num
        self.wh_disturb_ratio = wh_disturb_ratio
        self.noise_lambda = noise_lambda
        self.noise_l2 = noise_l2
        self.bw_adv_w_clip = bw_adv_w_clip
        self.fw_adv_w_clip = fw_adv_w_clip
        # self.gm_iter_count = 8  # interactively train forward net and backward net with gm_iter_count batches
        self.gm_iter_count = gm_iter_count  # interactively train forward net and backward net with gm_iter_count batches

        self.trunc_size = trunc_size
        self.shard_size = 0
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
        self.g_adv_step = 0.0
        self.g_adv_loss = 0.0
        self.g_adv_raw_loss = 0.0
        self.g_adv_cos = 0.0
        self.g_adv_weight = 0.0
        self.m_adv_cos = 0.0
        self.m_adv_step = 0.0

        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0
            if self.accum_count_l[i] > 1:
                assert self.trunc_size == 0, \
                    """To enable accumulated gradients,
                       you must disable target sequence truncating."""

        # Set model in training mode.
        self.g_model.train()
        self.m_model.train()
        self.n_model.train()

    def _g_maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.g_model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d" % (self.dropout[i], step))
    def _m_maybe_update_dropout(self, step):
        for i in range(len(self.dropout_steps)):
            if step > 1 and step == self.dropout_steps[i] + 1:
                self.m_model.update_dropout(self.dropout[i])
                logger.info("Updated dropout to %f from step %d" % (self.dropout[i], step))

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = 1
        for batch in iterator:
            batches.append(batch)
            if self.data_device is None:
                self.data_device = batch.tgt.device
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.g_train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization

    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None, m_valid_iter=None,
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
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        g_total_stats = onmt.utils.Statistics()
        g_report_stats = onmt.utils.Statistics()
        m_total_stats = onmt.utils.Statistics()
        m_report_stats = onmt.utils.Statistics()
        m_g_total_stats = onmt.utils.Statistics()
        m_g_report_stats = onmt.utils.Statistics()

        self._start_report_manager(start_time=g_total_stats.start_time)
        temp_counter = 0
        for i, (batches, normalization) in enumerate(self._accum_batches(train_iter)):
            g_step = self.g_optim.training_step
            m_step = self.m_optim.training_step
            # UPDATE DROPOUT
            self._g_maybe_update_dropout(g_step)
            self._m_maybe_update_dropout(m_step)

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d n_minibatch %d" % (self.gpu_rank, i + 1, len(batches)))

            if self.n_gpu > 1:
                # new here
                raise ValueError('Error: we do not support multi-gpu anymore!')
            if self.only_step3:
                # train N here
                self.train_noise_model(batches)
            else:
                try:
                    if temp_counter < self.gm_iter_count:
                        # train G here
                        self.train_g(batches, normalization, g_total_stats, g_report_stats)
                        if self.g_adv_train_flag:
                            self.train_g_adversarial(batches, normalization, g_total_stats, g_report_stats)
                        self.g_adv_train_flag = not self.g_adv_train_flag
                    elif temp_counter < 2 * self.gm_iter_count:
                        # train M here
                        self.train_m(
                            batches, normalization, m_total_stats, m_report_stats, m_g_total_stats, m_g_report_stats
                        )
                        if self.m_adv_train_flag:
                            self.train_m_adv(
                                batches, normalization, m_total_stats, m_report_stats, m_g_total_stats, m_g_report_stats
                            )
                        self.m_adv_train_flag = not self.m_adv_train_flag
                    else:
                        # train N here
                        self.train_noise_model(batches)
                    temp_counter += 1
                except Exception as e:
                    logger.info('+' * 100)
                    logger.info('Error')
                    logger.info(e)
                    continue
                if temp_counter >= 2.5 * self.gm_iter_count:
                    temp_counter = 0

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

                if valid_iter is not None and g_step % valid_steps == 0:
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: validate step %d' % (self.gpu_rank, g_step))
                    valid_stats = self.g_validate(valid_iter, moving_average=self.moving_average)
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
                if (self.g_model_saver is not None and (
                        save_checkpoint_steps != 0 and g_step % save_checkpoint_steps == 0)):
                    self.g_model_saver.save(g_step, moving_average=self.moving_average)
                if (self.m_model_saver is not None and (
                        save_checkpoint_steps != 0 and m_step % save_checkpoint_steps == 0)):
                    self.m_model_saver.save(m_step, moving_average=self.moving_average)
                if train_steps > 0 and g_step >= train_steps and m_step >= train_steps:
                    logger.info('=' * 40)
                    logger.info('stop now with g_step:{}, m_step:{}. '.format(g_step, m_step))
                    break
            n_step = int(self.n_step_accu)
            if (self.n_model_save_path is not None and (save_checkpoint_steps != 0)
                    and n_step > 0 and (n_step % save_checkpoint_steps == 0)):
                save_path = self.n_model_save_path.format(int(self.n_step_accu))
                logger.info('saving model to : {}'.format(save_path))
                torch.save(self.n_model, save_path)

        n_step = int(self.n_step_accu)
        if (self.n_model_save_path is not None and (save_checkpoint_steps != 0)
                and n_step > 0 and (n_step % save_checkpoint_steps == 0)):
            save_path = self.n_model_save_path.format(int(self.n_step_accu))
            logger.info('saving model to : {}'.format(save_path))
            torch.save(self.n_model, save_path)
        if not self.only_step3:
            if self.g_model_saver is not None:
                self.g_model_saver.save(g_step, moving_average=self.moving_average)
            if self.m_model_saver is not None:
                self.m_model_saver.save(m_step, moving_average=self.moving_average)
        return g_total_stats, m_total_stats

    def _repack_m_datas_from_g(self, a3_ids):
        src_ids = a3_ids
        src_lens = [len(x) for x in src_ids]
        max_src_len = int(max(src_lens))
        batch_size = len(src_ids)
        src_ids_tensor = torch.full([batch_size, max_src_len], self.pad_id).long()
        for idx in range(batch_size):
            src_ids_tensor[idx, :src_lens[idx]] = torch.LongTensor(src_ids[idx])[:]
        src_lengths = torch.LongTensor(src_lens)
        src_ids_tensor = src_ids_tensor.to(self.data_device)
        src_lengths = src_lengths.to(self.data_device)
        return src_ids_tensor.transpose(0, 1).unsqueeze(2), src_lengths

    def train_g(self, true_batches, normalization, total_stats, report_stats):
        if self.accum_count > 1:
            self.g_optim.zero_grad()

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            if self.trunc_size:
                trunc_size = self.trunc_size
            else:
                trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            bptt = False
            for j in range(0, target_size-1, trunc_size):
                # 1. Create truncated target.
                tgt = tgt_outer[j: j + trunc_size]

                # 2. F-prop all but generator.
                if self.accum_count == 1:
                    self.g_optim.zero_grad()

                outputs, attns = self.g_model(src, tgt, src_lengths, bptt=bptt, with_align=self.with_align)
                bptt = True

                # 3. Compute loss.
                try:
                    loss, batch_stats = self.g_train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=j,
                        trunc_size=trunc_size)

                    if loss is not None:
                        self.g_optim.backward(loss)

                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)

                except Exception:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d", self.g_optim.training_step, k)

                # 4. Update the parameters and statistics.
                if self.accum_count == 1:
                    # Multi GPU gradient gather
                    if self.n_gpu > 1:
                        grads = [p.grad.data for p in self.g_model.parameters()
                                 if p.requires_grad and p.grad is not None]
                        onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
                    self.g_optim.step()

                # If truncated, don't backprop fully.
                # TO CHECK
                # if dec_state is not None:
                #    dec_state.detach()
                if self.g_model.decoder.state is not None:
                    self.g_model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.g_model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.g_optim.step()

    def _cal_emb_vec(self, sent_ids):
        """ length first here
            :param sent_ids: [length, N, 1]
            :return [N, dim]
        """
        s_emb = self.g_model.encoder.embeddings(sent_ids)
        s_mask_for_lens = sent_ids.transpose(0, 1).squeeze(2).data.ne(self.pad_id)
        s_lens = torch.sum(s_mask_for_lens, 1)
        # [s_len, N, 1]
        s_mask = sent_ids.data.eq(self.pad_id)
        s_emb.masked_fill_(s_mask, 0.0)
        emb_vec = torch.sum(s_emb, 0) / s_lens.float().unsqueeze(1)
        return emb_vec

    def _pack_data_g_adv(self, g_pred_ids):
        """
        :param g_pred_ids: [N, x_len]
        :return:
        """
        pred_ids = g_pred_ids.tolist()
        g_x_len = g_pred_ids.size(1)

        g_tgt = []
        m_src = []
        m_src_lengths = []
        for sent_ids in pred_ids:
            if self.eos_id in sent_ids:
                idx = sent_ids.index(self.eos_id)
                base = sent_ids[: idx]
            else:
                base = sent_ids
            m_src.append(base)
            # m_src_lengths.append(len(m_src[-1]))
            m_src_lengths.append(len(m_src[-1]) if len(m_src[-1]) > 0 else 1)
            g_sent = base + [self.eos_id]
            g_tgt.append(g_sent[:g_x_len])
        m_max_len = max(m_src_lengths)
        for i in range(len(g_tgt)):
            while len(g_tgt[i]) < g_x_len:
                g_tgt[i].append(self.pad_id)
            while len(m_src[i]) < m_max_len:
                m_src[i].append(self.pad_id)
        m_src = torch.LongTensor(m_src).type_as(g_pred_ids)
        m_src_lengths = torch.LongTensor(m_src_lengths).type_as(g_pred_ids)
        g_tgt = torch.LongTensor(g_tgt).type_as(g_pred_ids)

        m_src = m_src.transpose(0, 1).unsqueeze(2)  # [len, N, 1]
        return g_tgt, m_src, m_src_lengths

    def _pack_m_tgt_data(self, src, src_lengths, tgt):
        with torch.no_grad():
            n_tgt = src.squeeze(2).transpose(0, 1)
            bz = n_tgt.size(0)

            boss = tgt[:1].squeeze(2).transpose(0, 1)
            pads = boss.clone().fill_(self.pad_id)
            n_tgt = torch.cat((boss, n_tgt, pads), 1)
            old_s_lens = src_lengths.tolist()
            for i in range(bz):
                n_tgt[i, old_s_lens[i] + 1] = self.eos_id
            n_tgt = n_tgt.transpose(0, 1).unsqueeze(2).detach()

        return n_tgt

    def train_g_adversarial(self, true_batches, normalization, total_stats, report_stats):
        if self.accum_count > 1:
            self.g_optim.zero_grad()
        disturb_steps = self.wh_sample_num

        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            trunc_size = target_size
            self.g_adv_step += 1

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            tgt_outer = batch.tgt

            # 1. Create truncated target.
            tgt = tgt_outer
            dec_in = tgt[:-1]  # exclude last target token from inputs
            m_tgt = self._pack_m_tgt_data(src, src_lengths, batch.tgt)
            m_tgt_cal_loss = m_tgt[1:, :, 0].transpose(0, 1)

            # 2. F-prop all but generator.
            if self.accum_count == 1:
                self.g_optim.zero_grad()

            enc_state, memory_bank, lengths = self.g_model.encoder(src, src_lengths)
            h = memory_bank
            h_disturbs = self.n_model(h, n_sample=disturb_steps)
            # print('checking length first: g tgt: {}'.format(tgt.size()))

            bptt = False
            f_g_loss = None
            f_g_loss_raw = 0.0
            f_g_loss_cos = 0.0
            f_g_loss_weight = 0.0
            for noise in h_disturbs:
                memory_bank = h + noise * self.noise_lambda
                if bptt is False:
                    self.g_model.decoder.init_state(src, memory_bank, enc_state)
                dec_out, attns = self.g_model.decoder(
                    dec_in, memory_bank, memory_lengths=lengths,
                    with_align=self.with_align
                )
                g_vocab_logits = self.g_model.generator(dec_out).transpose(0, 1)
                pred_ids_raw = torch.argmax(g_vocab_logits, 2)
                pred_ids_raw, fake_a3_src_ids, fake_a3_src_lens = self._pack_data_g_adv(pred_ids_raw)

                g_loss = self.easy_loss_fn.cal(g_vocab_logits, pred_ids_raw)

                fake_a3_src_ids = fake_a3_src_ids.detach()
                fake_a3_src_lens = fake_a3_src_lens.detach()
                fake_a3_outputs, fake_a3_attns = self.m_model(
                    fake_a3_src_ids, m_tgt, fake_a3_src_lens,
                    bptt=False, with_align=self.with_align
                )
                vocab_logits = self.m_model.generator(fake_a3_outputs).transpose(0, 1)
                m_loss = self.easy_loss_fn.cal(vocab_logits, m_tgt_cal_loss)
                m_loss = 0 - m_loss

                if f_g_loss is None:
                    f_g_loss = g_loss * m_loss
                else:
                    f_g_loss = f_g_loss + g_loss * m_loss

                f_g_loss_raw += torch.mean(g_loss).item()
                f_g_loss_weight += torch.mean(m_loss).item()

            f_g_loss = torch.mean(f_g_loss / disturb_steps)
            f_g_loss_val = f_g_loss.item()
            self.g_adv_loss += f_g_loss_val
            self.g_adv_raw_loss += f_g_loss_raw / disturb_steps
            self.g_adv_weight += f_g_loss_weight / disturb_steps

            f_g_loss.backward()

            # 4. Update the parameters and statistics.
            # Multi GPU gradient gather
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.g_model.parameters()
                         if p.requires_grad and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))

            # 4. Update the parameters and statistics.
            if self.accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.g_model.parameters()
                             if p.requires_grad and p.grad is not None]
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
                clip_norm = clip_grad_norm_(self.g_model.parameters(), self.fw_adv_w_clip)
                self.g_optim.step()

            # If truncated, don't backprop fully.
            # TO CHECK
            # if dec_state is not None:
            #    dec_state.detach()
            if self.g_model.decoder.state is not None:
                self.g_model.decoder.detach_state()

            log_steps = self.report_every
            if 0 == self.g_adv_step % log_steps:
                logger.info('G_Adv step: {}, loss: {}, row_loss: {}, weight: {}'.format(
                    self.g_adv_step,
                    self.g_adv_loss / log_steps, self.g_adv_raw_loss / log_steps,
                    self.g_adv_weight / log_steps))
                self.g_adv_loss = 0.0
                self.g_adv_raw_loss = 0.0
                self.g_adv_cos = 0.0
                self.g_adv_weight = 0.0
        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.g_model.parameters()
                         if p.requires_grad and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
            clip_norm = clip_grad_norm_(self.g_model.parameters(), self.fw_adv_w_clip)
            self.g_optim.step()

    def handle_m_data(self, src, src_lengths, tgt):

        with torch.no_grad():
            n_tgt = src.squeeze(2).transpose(0, 1)
            n_src = tgt[1:].squeeze(2).transpose(0, 1)
            bz = n_src.size(0)
            n_src_len = n_src.size(1)
            n_src_lens = [1] * bz
            for i in range(bz):
                for j in range(n_src_len):
                    cur_idx = -1 -j
                    if n_src[i, cur_idx].item() == self.eos_id:
                        n_src[i, cur_idx] = self.pad_id
                        n_src_lens[i] = n_src_len - j - 1
                        break
            n_src = n_src[:, :-1]

            boss = tgt[:1].squeeze(2).transpose(0, 1)
            pads = boss.clone().fill_(self.pad_id)
            n_tgt = torch.cat((boss, n_tgt, pads), 1)
            old_s_lens = src_lengths.tolist()
            for i in range(bz):
                n_tgt[i, old_s_lens[i] + 1] = self.eos_id
            n_src = n_src.transpose(0, 1).unsqueeze(2).detach()
            n_tgt = n_tgt.transpose(0, 1).unsqueeze(2).detach()
            n_src_lens = torch.LongTensor(n_src_lens).type_as(n_src)

        return n_src, n_src_lens, n_tgt

    def _gen_fake_T_with_noise(self, src, src_lengths, tgt):
        """
        :param src: fw src
        :param src_lengths: fw src_lengths
        :param tgt: fw tgt
        :return:
        """
        disturb_steps = self.wh_sample_num
        bptt = False

        noise_Ts = []
        with torch.no_grad():
            enc_state, memory_bank, lengths = self.g_model.encoder(src, src_lengths)
            h = memory_bank
            h_disturbs = self.n_model(h, n_sample=disturb_steps)
            # T' embedding
            g_tgt_emb = self._cal_emb_vec(tgt)
            dec_in = tgt[:-1]

            for noise in h_disturbs:
                memory_bank = h + noise * self.noise_lambda
                if bptt is False:
                    self.g_model.decoder.init_state(src, memory_bank, enc_state)
                dec_out, attns = self.g_model.decoder(
                    dec_in, memory_bank, memory_lengths=lengths,
                    with_align=self.with_align
                )
                g_vocab_logits = self.g_model.generator(dec_out).transpose(0, 1)
                pred_ids_raw = torch.argmax(g_vocab_logits, 2)
                pred_ids_raw, fake_a3_src_ids, fake_a3_src_lens = self._pack_data_g_adv(pred_ids_raw)

                g_pred_emb = self._cal_emb_vec(pred_ids_raw.transpose(0, 1).unsqueeze(2))
                cos_sim = torch.cosine_similarity(g_tgt_emb, g_pred_emb, dim=1)
                noise_Ts.append((fake_a3_src_ids, fake_a3_src_lens, cos_sim))
        return noise_Ts

    def train_m_adv(self, true_batches, normalization, total_stats, report_stats, m_g_total_stats, m_g_report_stats):
        if self.accum_count > 1:
            self.m_optim.zero_grad()
        disturb_steps = self.wh_sample_num

        for k, batch in enumerate(true_batches):
            self.m_adv_step += 1

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            old_src = src
            old_src_lens = src_lengths
            old_tgt = batch.tgt

            src, src_lengths, tgt = self.handle_m_data(src, src_lengths, old_tgt)

            batch.src = (src, src_lengths)
            batch.tgt = tgt
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()
            trunc_size = batch.tgt.size(0)

            fake_Ts = self._gen_fake_T_with_noise(old_src, old_src_lens, old_tgt)
            bp_losses = []
            f_m_loss_cos = 0
            for (fake_a3_src_ids, fake_a3_src_lens, cos_sim) in fake_Ts:
                fake_a3_outputs, fake_a3_attns = self.m_model(
                    fake_a3_src_ids, tgt, fake_a3_src_lens, bptt=False, with_align=self.with_align
                )
                f_m_loss_cos += torch.mean(cos_sim).item()
                # 3. Compute loss.
                try:
                    fake_a3_loss, fake_a3_batch_stats = self.m_train_loss(
                        batch=batch,
                        output=fake_a3_outputs,
                        attns=fake_a3_attns,
                        reward_weights=1 - cos_sim,
                        normalization=normalization,
                        shard_size=self.shard_size,
                        trunc_start=0,
                        trunc_size=trunc_size)
                    bp_losses.append(-fake_a3_loss)

                    m_g_total_stats.update(fake_a3_batch_stats)
                    m_g_report_stats.update(fake_a3_batch_stats)
                except Exception as e:
                    traceback.print_exc()
                    logger.info("At step %d, we removed a batch - accum %d", self.m_optim.training_step, k)
            # 3. BP.
            try:
                bp_loss = sum(bp_losses) / disturb_steps
                self.m_optim.backward(bp_loss)
            except Exception as e:
                traceback.print_exc()
                logger.info("At step %d, we removed a batch - accum %d", self.m_optim.training_step, k)
            self.m_adv_cos += f_m_loss_cos / disturb_steps

            # 4. Update the parameters and statistics.
            if self.accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.m_model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(grads, float(1))
                clip_norm = clip_grad_norm_(self.m_model.parameters(), self.bw_adv_w_clip)
                self.m_optim.step()

            # If truncated, don't backprop fully.
            # TO CHECK
            # if dec_state is not None:
            #    dec_state.detach()
            if self.m_model.decoder.state is not None:
                self.m_model.decoder.detach_state()

            log_steps = self.report_every
            if 0 == self.m_adv_step % log_steps:
                logger.info('M_Adv step: {}, cos: {}'.format(
                    self.g_adv_step,
                    self.m_adv_cos / log_steps)
                )
                self.m_adv_cos = 0.0

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.m_model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            clip_norm = clip_grad_norm_(self.m_model.parameters(), self.bw_adv_w_clip)
            self.m_optim.step()

    def train_m(self, true_batches, normalization, total_stats, report_stats, m_g_total_stats, m_g_report_stats):
        if self.accum_count > 1:
            self.m_optim.zero_grad()

        for k, batch in enumerate(true_batches):
            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)
            old_src = src
            old_src_lens = src_lengths
            old_tgt = batch.tgt

            src, src_lengths, tgt = self.handle_m_data(src, src_lengths, old_tgt)

            batch.src = (src, src_lengths)
            batch.tgt = tgt
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()
            trunc_size = batch.tgt.size(0)

            outputs, attns = self.m_model(src, tgt, src_lengths, bptt=False, with_align=self.with_align)

            # 3. Compute loss.
            try:
                loss, batch_stats = self.m_train_loss(
                    batch,
                    outputs,
                    attns,
                    normalization=normalization,
                    shard_size=self.shard_size,
                    trunc_start=0,
                    trunc_size=trunc_size)
                bp_loss = loss
                self.m_optim.backward(bp_loss)

                total_stats.update(batch_stats)
                report_stats.update(batch_stats)
            except Exception:
                traceback.print_exc()
                logger.info("At step %d, we removed a batch - accum %d", self.m_optim.training_step, k)

            # 4. Update the parameters and statistics.
            if self.accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.m_model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    onmt.utils.distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.m_optim.step()

            # If truncated, don't backprop fully.
            # TO CHECK
            # if dec_state is not None:
            #    dec_state.detach()
            if self.m_model.decoder.state is not None:
                self.m_model.decoder.detach_state()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.m_model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                onmt.utils.distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.m_optim.step()

    def _pack_data_n_adv(self, g_pred_ids):
        """
        :param g_pred_ids: [N, x_len]
        :return:
        """
        pred_ids = g_pred_ids.tolist()
        m_src = []
        m_src_lengths = []
        for sent_ids in pred_ids:
            if self.eos_id in sent_ids:
                idx = sent_ids.index(self.eos_id)
                base = sent_ids[: idx]
            else:
                base = sent_ids
            m_src.append(base)
            # m_src_lengths.append(len(m_src[-1]))
            m_src_lengths.append(len(m_src[-1]) if len(m_src[-1]) > 0 else 1)
        m_max_len = max(m_src_lengths)
        for i in range(len(m_src)):
            while len(m_src[i]) < m_max_len:
                m_src[i].append(self.pad_id)
        m_src = torch.LongTensor(m_src).type_as(g_pred_ids)
        m_src_lengths = torch.LongTensor(m_src_lengths).type_as(g_pred_ids)

        m_src = m_src.transpose(0, 1).unsqueeze(2)  # [len, N, 1]
        return m_src, m_src_lengths

    def _train_noise_model(self, src, tgt, src_lengths, bptt=False):
        disturb_steps = self.wh_sample_num
        src = src.contiguous()
        src_lengths = src_lengths.contiguous()
        dec_in = tgt[:-1]  # exclude last target token from inputs
        enc_state, memory_bank, lengths = self.g_model.encoder(src, src_lengths)
        g_tgt_cal_loss = tgt[1:, :, 0].transpose(0, 1)

        # m_tgt = torch.cat((tgt[:1], src), 0)
        m_tgt = self._pack_m_tgt_data(src, src_lengths, tgt)
        m_tgt_cal_loss = m_tgt[1:, :, 0].transpose(0, 1)

        h = memory_bank
        h_disturbs = self.n_model(h, n_sample=disturb_steps)

        cur_g_loss = 0.0
        cur_m_loss = 0.0
        cur_gaussian_disturb_strength = 0.0
        loss = None
        for noise in h_disturbs:
            memory_bank = h + noise
            gaussian_strength = torch.mean(torch.norm(noise, 2, 1))
            cur_gaussian_disturb_strength += gaussian_strength.item()
            if bptt is False:
                self.g_model.decoder.init_state(src, memory_bank, enc_state)
            dec_out, attns = self.g_model.decoder(
                dec_in, memory_bank, memory_lengths=lengths,
                with_align=self.with_align
            )
            vocab_logits = self.g_model.generator(dec_out).transpose(0, 1)
            # the g_loss here is just for analysis, meaningless for optimizing Noise Model
            g_loss = self.easy_loss_fn.cal(vocab_logits, g_tgt_cal_loss)
            g_loss = torch.mean(g_loss)

            fake_a3_src_ids, fake_a3_src_lens = self._pack_data_n_adv(torch.argmax(vocab_logits, 2))
            fake_a3_src_ids = fake_a3_src_ids.detach()
            fake_a3_src_lens = fake_a3_src_lens.detach()

            fake_a3_outputs, fake_a3_attns = self.m_model(
                fake_a3_src_ids, m_tgt, fake_a3_src_lens, bptt=False,
                with_align=self.with_align
            )
            vocab_logits = self.m_model.generator(fake_a3_outputs).transpose(0, 1)
            m_loss = self.easy_loss_fn.cal(vocab_logits, m_tgt_cal_loss)
            m_loss = torch.mean(m_loss)
            gaussian_strength_loss = gaussian_strength if gaussian_strength.item() <= self.noise_l2 else 0.0
            if loss is None:
                # loss = m_loss - self.wh_disturb_ratio * gaussian_strength
                loss = m_loss - self.wh_disturb_ratio * gaussian_strength_loss
            else:
                # loss = loss + m_loss - self.wh_disturb_ratio * gaussian_strength
                loss = loss + m_loss - self.wh_disturb_ratio * gaussian_strength_loss

            cur_g_loss += g_loss.item()
            cur_m_loss += m_loss.item()
        f_loss = loss / disturb_steps
        self.n_loss_accu += f_loss.item()
        self.n_l2_accu += cur_gaussian_disturb_strength / disturb_steps
        self.n_loss_g_accu += cur_g_loss / disturb_steps
        self.n_loss_m_accu += cur_m_loss / disturb_steps
        self.n_optim.zero_grad()
        f_loss.backward()
        self.n_optim.step()

        log_steps = self.report_every
        if 0 == self.n_step_accu % log_steps:
            logger.info('Noise step: {}, loss: {}, g_loss: {}, m_loss: {}, l2: {}, lr: {}'.format(
                self.n_step_accu, self.n_loss_accu / log_steps,
                self.n_loss_g_accu / log_steps, self.n_loss_m_accu / log_steps,
                self.n_l2_accu / log_steps,
                self.n_optim.learning_rate()))
            self.n_loss_accu = 0.0
            self.n_l2_accu = 0.0
            self.n_loss_g_accu = 0.0
            self.n_loss_m_accu = 0.0

    def train_noise_model(self, true_batches):
        for k in range(len(true_batches)):
            self.n_step_accu += 1

            batch = true_batches[k]
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            trunc_size = target_size

            src, src_lengths = batch.src if isinstance(batch.src, tuple) else (batch.src, None)

            tgt = batch.tgt
            # 1. Create truncated target.
            # 2. F-prop all but generator.
            self._train_noise_model(src, tgt, src_lengths, bptt=False)

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

    def g_validate(self, valid_iter, moving_average=None):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.g_model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.g_optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
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
        valid_model = self.m_model
        if moving_average:
            # swap model params w/ moving average
            # (and keep the original parameters)
            model_params_data = []
            for avg, param in zip(self.moving_average,
                                  valid_model.parameters()):
                model_params_data.append(param.data)
                param.data = avg.data.half() if self.m_optim._fp16 == "legacy" else avg.data

        # Set model in validating mode.
        valid_model.eval()

        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
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

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats, reporter_name=''):
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
