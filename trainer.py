# TODO(darcey): finish reading Toan's paper for additional insights
# TODO(darcey): compute dev BLEU

# TODO(darcey): improve how the support mask is handled, so that the logic around it, and the connection to the label smoothing counts, is less confusing (see tests for loss)
# TODO(darcey): right now prep_batch is run identically every time we evaluate on the dev data; do it just once for dev data
# TODO(darcey): consider moving the label smoothing initialization stuff into its own function for modularity
# TODO(darcey): improve code structure of the functions shared between train and perplexity
# TODO(darcey): improve torch efficiency throughout codebase (switch from reshape to view? bmm vs. matmul? stop using one-hots where possible?)
# TODO(darcey): implement classifier learning also

import torch

class Trainer():

    def __init__(self, model, vocab, config_train):
        self.vocab = vocab
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        self.max_epochs = config_train.max_epochs
        self.epoch_size = config_train.epoch_size
        self.num_epochs = 0
        self.num_steps = 0
        self.num_toks = 0

        self.support_mask = vocab.get_tgt_support_mask()

        self.label_smoothing = config_train.label_smoothing
        ls_mask = vocab.get_tgt_support_mask().type(torch.float)
        if not config_train.label_smooth_eos:
            ls_mask[vocab.eos_idx()] = 0.0
        if not config_train.label_smooth_unk:
            ls_mask[vocab.unk_idx()] = 0.0
        ls_counts = ls_mask / torch.sum(ls_mask)
        self.label_smoothing_counts = ls_counts

        self.word_dropout_prob = config_train.word_dropout

        return

    def train(self, train, dev):
        for epoch in range(self.max_epochs):
            for step in range(self.epoch_size):
                # prepare the batch
                batch = train.get_batch()
                src, tgt_in, tgt_out = self.prep_batch(batch, do_dropout=True)

                # do one step of training
                self.train_one_step(src, tgt_in, tgt_out)

                # update training statistics
                self.num_steps += 1
                self.num_toks += batch["num_tgt_toks"]

            # evaluate dev perplexity
            dev_ppl = self.perplexity(dev)
            print("DEV PPL: " + str(dev_ppl))
            print("--------------------------------")
            # TODO(darcey): evaluate dev BLEU
            # save checkpoints as needed
            # adjust learning rate / early stopping
            self.num_epochs += 1

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
                cross_ent_total += cross_ent
                num_toks_total += num_toks

            # compute perplexity based on all the log probs
            perplexity = torch.exp(cross_ent_total / num_toks_total).item()
        self.model.train()
        return perplexity

    def prep_batch(self, batch, do_dropout=False):
        # get relevant info from batch
        src = batch["src"]
        tgt_in = batch["tgt_in"]
        tgt_out = batch["tgt_out"]

        # word dropout
        if do_dropout:
            src = self.word_dropout(src, self.word_dropout_prob)
            tgt_in = self.word_dropout(tgt_in, self.word_dropout_prob)

        # convert to one-hots
        vocab_size = len(self.vocab)
        src = torch.nn.functional.one_hot(src, vocab_size).type(torch.FloatTensor)
        tgt_in = torch.nn.functional.one_hot(tgt_in, vocab_size).type(torch.FloatTensor)
        tgt_out = torch.nn.functional.one_hot(tgt_out, vocab_size).type(torch.FloatTensor)

        return src, tgt_in, tgt_out

    # data: [batch, seq]
    def word_dropout(self, data, dropout):
        unk_idx = self.vocab.unk_idx()
        unk_tensor = torch.full_like(data, unk_idx)
        prob = torch.full_like(data, dropout, dtype=torch.double)
        unk_mask = torch.bernoulli(prob).type(torch.long)
        return data * (1 - unk_mask) + unk_tensor * unk_mask

    # predicted: [batch, tgt_seq, vocab_size] <-- log probs output by model
    # actual:    [batch, tgt_seq, vocab_size] <-- one-hot of correct answer
    # ret:       scalar
    def loss(self, predicted, actual):
        cross_ent, num_toks = self.cross_ent(predicted, actual, smooth=True)
        return cross_ent / num_toks

    # cross-entropy as estimated from an empirical sample
    # predicted: [batch, tgt_seq, vocab_size] <-- log probs output by model
    # actual:    [batch, tgt_seq, vocab_size] <-- one-hot of correct answer
    # ret:       scalar
    def cross_ent(self, predicted, actual, smooth=False):
        # filter out positions which are PAD
        # [batch*tgt_seq, vocab_size]
        vocab_size   = predicted.size(-1)
        predicted    = predicted.reshape(-1, vocab_size)
        actual       = actual.reshape(-1, vocab_size)
        pad_idx      = self.vocab.pad_idx()
        non_pad_mask = (actual[:, pad_idx] != 1)
        predicted    = predicted[non_pad_mask]
        actual       = actual[non_pad_mask]
        num_toks     = torch.sum(non_pad_mask)

        # compute the smoothed true counts according to label smoothing
        # [batch*tgt_seq, vocab_size]
        ls = self.label_smoothing if smooth else 0.0
        actual_smoothed = (1 - ls) * actual + ls * self.label_smoothing_counts

        # remove positions not in the support of the model/true distribution
        # (because they are src vocab words or special tokens that can never
        #  appear on the target side)
        predicted = predicted[:, self.support_mask]
        actual_smoothed = actual_smoothed[:, self.support_mask]

        # compute the cross entropy
        cross_ent = actual_smoothed * predicted
        return torch.sum(cross_ent), num_toks
