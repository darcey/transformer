# TODO(darcey): compute dev BLEU

# TODO(darcey): improve how the support mask is handled, so that the logic around it, and the connection to the label smoothing counts, is less confusing (see tests for loss)
# TODO(darcey): right now prep_batch is run identically every time we evaluate on the dev data; do it just once for dev data
# TODO(darcey): consider moving the label smoothing initialization stuff into its own function for modularity
# TODO(darcey): improve torch efficiency throughout codebase (switch from reshape to view? bmm vs. matmul? stop using one-hots where possible?)
# TODO(darcey): implement classifier learning also

import math
import torch
from configuration import *

class Trainer():

    def __init__(self, model, vocab, config, device):
        self.vocab = vocab
        self.model = model

        self.lr_config = config.train.lr
        self.lr_config.d_model = config.arch.d_model
        self.lr = self.get_initial_learning_rate()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.max_epochs = config.train.max_epochs
        self.epoch_size = config.train.epoch_size
        self.num_toks = 0
        self.num_steps = 0
        self.num_epochs = 0
        self.dev_ppls = []

        self.support_mask = vocab.get_tgt_support_mask()

        self.label_smoothing = config.train.label_smoothing
        ls_mask = vocab.get_tgt_support_mask().type(torch.float)
        if not config.train.label_smooth_eos:
            ls_mask[vocab.eos_idx()] = 0.0
        if not config.train.label_smooth_unk:
            ls_mask[vocab.unk_idx()] = 0.0
        ls_counts = ls_mask / torch.sum(ls_mask)
        self.label_smoothing_counts = ls_counts

        self.word_dropout_prob = config.train.word_dropout

        self.device = device
        self.model.to(device)
        self.support_mask = self.support_mask.to(device)
        self.label_smoothing_counts = self.label_smoothing_counts.to(device)

        return

    def train(self, train, dev):
        dev_ppl = self.perplexity(dev)
        print("DEV PPL: " + str(dev_ppl))
        print("--------------------------------")
        for epoch in range(self.max_epochs):
            self.num_epochs += 1
            # early stopping
            if self.early_stopping():
                break

            # train for one epoch
            for step in range(self.epoch_size):
                self.num_steps += 1
                # prepare the batch
                batch = train.get_batch()
                self.num_toks += batch["num_tgt_toks"]
                src, tgt_in, tgt_out = self.prep_batch(batch, do_dropout=True)
                self.num_toks += batch
                # do one step of training
                self.train_one_step(src, tgt_in, tgt_out)
                # adjust learning rate
                self.lr = self.adjust_learning_rate_step()

            # evaluate dev perplexity
            # TODO(darcey): evaluate dev BLEU
            dev_ppl = self.perplexity(dev)
            self.dev_ppls.append(dev_ppl)
            print("DEV PPL: " + str(dev_ppl))
            print("--------------------------------")
            # save checkpoints as needed
            # adjust learning rate
            self.lr = self.adjust_learning_rate_epoch()

    def train_one_step(self, src, tgt_in, tgt_out):
        self.optimizer.zero_grad()
        log_probs = self.model(src, tgt_in)
        loss = self.loss(log_probs, tgt_out)
        print(loss)
        loss.backward()
        self.optimizer.step()

    def perplexity(self, data):
        self.model.eval()
        with torch.no_grad():
            cross_ent_total = 0.0
            num_toks_total = 0
            for batch_num in range(len(data)):
                # prepare the batch
                batch = data.get_batch()
                src, tgt_in, tgt_out = self.prep_batch(batch, do_dropout=False)

                # get the log probability of the batch
                log_probs = self.model(src, tgt_in)
                cross_ent, num_toks = self.cross_ent(log_probs, tgt_out)
                cross_ent_total += cross_ent.cpu()
                num_toks_total += num_toks.cpu()

            # compute perplexity based on all the log probs
            perplexity = torch.exp(cross_ent_total / num_toks_total).item()
        self.model.train()
        return perplexity

    def get_initial_learning_rate(self):
        conf = self.lr_config
        # For warmup-based strategies, start at the low rate, then warm up
        if (conf.lr_strategy == LearningRateStrategy.WARMUP_INV_SQRT_DECAY or\
            conf.lr_strategy == LearningRateStrategy.WARMUP_VAL_DECAY):
            return conf.lr_scale * (conf.d_model ** -0.5) * (conf.warmup_steps ** -1.5)
        # For non-warmup-based strategies, start at user-specified rate
        else:
            return conf.start_lr

    def adjust_learning_rate_step(self):
        conf = self.lr_config
        steps = self.num_steps + 1
        # For warmup-based strategies, if during warmup, do warmup
        if (conf.lr_strategy == LearningRateStrategy.WARMUP_INV_SQRT_DECAY or\
            conf.lr_strategy == LearningRateStrategy.WARMUP_VAL_DECAY) and\
           (steps < conf.warmup_steps):
            self.lr = conf.lr_scale * (conf.d_model ** -0.5) * steps * (conf.warmup_steps ** -1.5)
        # For inverse square root decay, if not during warmup, do decay
        elif conf.lr_strategy == LearningRateStrategy.WARMUP_INV_SQRT_DECAY:
            self.lr = conf.lr_scale * (conf.d_model ** -0.5) * (steps ** -0.5)
        else:
            return

        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

    def adjust_learning_rate_epoch(self):
        conf = self.lr_config
        steps = self.num_steps + 1
        # For validation-based decay, if not during warmup, do decay
        if (conf.lr_strategy == LearningRateStrategy.NO_WARMUP_VAL_DECAY) or\
           (conf.lr_strategy == LearningRateStrategy.WARMUP_VAL_DECAY and\
            steps >= conf.warmup_steps):
            if (self.num_epochs > conf.patience) and\
               (self.dev_ppls[-1] > max(self.dev_ppls[-1-conf.patience:-1])):
                self.lr = self.lr * conf.lr_decay
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr

    def early_stopping(self):
        conf = self.lr_config
        steps = self.num_steps + 1
        # For validation-based decay, if not during warmup, check whether to stop
        if (conf.lr_strategy == LearningRateStrategy.NO_WARMUP_VAL_DECAY) or\
           (conf.lr_strategy == LearningRateStrategy.WARMUP_VAL_DECAY and\
            steps >= conf.warmup_steps):
            return self.lr < conf.stop_lr
        else:
            return False

    def prep_batch(self, batch, do_dropout=False):
        # get relevant info from batch
        src = batch["src"].to(self.device)
        tgt_in = batch["tgt_in"].to(self.device)
        tgt_out = batch["tgt_out"].to(self.device)

        # word dropout
        if do_dropout:
            src = self.word_dropout(src, self.word_dropout_prob)
            tgt_in = self.word_dropout(tgt_in, self.word_dropout_prob)

        # convert to one-hots
        vocab_size = len(self.vocab)
        src = torch.nn.functional.one_hot(src, vocab_size).type(torch.float)
        tgt_in = torch.nn.functional.one_hot(tgt_in, vocab_size).type(torch.float)
        tgt_out = torch.nn.functional.one_hot(tgt_out, vocab_size).type(torch.float)

        return src, tgt_in, tgt_out

    # data: [batch, seq]
    def word_dropout(self, data, dropout):
        unk_idx = self.vocab.unk_idx()
        unk_tensor = torch.full_like(data, unk_idx)
        prob = torch.full_like(data, dropout, dtype=torch.double)
        unk_mask = torch.bernoulli(prob).type(torch.long)
        return data * (1 - unk_mask) + unk_tensor * unk_mask

    # predicted: [batch, tgt_seq, vocab_size] <-- log probs output by model
    # gold:    [batch, tgt_seq, vocab_size] <-- one-hot of correct answer
    # ret:       scalar
    def loss(self, predicted, gold):
        cross_ent, num_toks = self.cross_ent(predicted, gold, smooth=True)
        return cross_ent / num_toks

    # cross-entropy as estimated from an empirical sample
    # predicted: [batch, tgt_seq, vocab_size] <-- log probs output by model
    # gold:    [batch, tgt_seq, vocab_size] <-- one-hot of correct answer
    # ret:       scalar
    def cross_ent(self, predicted, gold, smooth=False):
        # filter out positions which are PAD
        # [batch*tgt_seq, vocab_size]
        vocab_size   = predicted.size(-1)
        predicted    = predicted.reshape(-1, vocab_size)
        gold         = gold.reshape(-1, vocab_size)
        pad_idx      = self.vocab.pad_idx()
        non_pad_mask = (gold[:, pad_idx] != 1)
        predicted    = predicted[non_pad_mask]
        gold       = gold[non_pad_mask]
        num_toks     = torch.sum(non_pad_mask)

        # compute the smoothed true counts according to label smoothing
        # [batch*tgt_seq, vocab_size]
        ls = self.label_smoothing if smooth else 0.0
        gold_smoothed = (1 - ls) * gold + ls * self.label_smoothing_counts

        # remove positions not in the support of the model/true distribution
        # (because they are src vocab words or special tokens that can never
        #  appear on the target side, and should have probability -inf)
        predicted = predicted[:, self.support_mask]
        gold_smoothed = gold_smoothed[:, self.support_mask]

        # compute the cross entropy
        cross_ent = - gold_smoothed * predicted
        return torch.sum(cross_ent), num_toks
