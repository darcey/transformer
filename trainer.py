# TODO(darcey): finish reading Toan's paper for additional insights
# TODO(darcey): add the label smoothing settings / label smoothing masking to train_config
# TODO(darcey): standardize which argument the config file is in the __init__ function across files in this project
# TODO(darcey): improve torch efficiency throughout codebase (switch from reshape to view? bmm vs. matmul? stop using one-hots where possible?)
# TODO(darcey): implement classifier learning also

import torch
from vocabulary import SpecialTokens

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

        # TODO(darcey): fix this (see TODO note above)
        self.label_smoothing = 0.1
        ls_counts = torch.tensor([1.0]*len(self.vocab))
        pad_idx = self.vocab.tok_to_idx(SpecialTokens.PAD)
        ls_counts[pad_idx] = 0
        ls_counts = ls_counts / torch.sum(ls_counts)
        self.label_smoothing_counts = ls_counts

        self.word_dropout_prob = config_train.word_dropout

        return

    def train(self, train, dev):
        for epoch in range(self.max_epochs):
            for step in range(self.epoch_size):
                # get the batch
                batch = train.get_batch()
                src = batch["src"]
                tgt_in = batch["tgt_in"]
                tgt_out = batch["tgt_out"]

                # word dropout
                src = self.word_dropout(src, self.word_dropout_prob)
                tgt_in = self.word_dropout(tgt_in, self.word_dropout_prob)

                # convert to one-hots
                vocab_size = len(self.vocab)
                src = torch.nn.functional.one_hot(src, vocab_size)
                tgt_in = torch.nn.functional.one_hot(tgt_in, vocab_size)
                tgt_out = torch.nn.functional.one_hot(tgt_out, vocab_size)

                # do one step of training
                self.train_one_step(src, tgt_in, tgt_out)

                # compute train perplexity?

                # update training statistics
                self.num_steps += 1
                self.num_toks += batch["num_tgt_toks"]
            # evaluate dev perplexity
            # evaluate dev BLEU
            # save checkpoints as needed
            # adjust learning rate / early stopping
            self.num_epochs += 1

    def train_one_step(self, src, tgt_in, tgt_out):
        self.optimizer.zero_grad()
        log_probs = self.model(src, tgt_in)
        loss = self.loss(log_probs, tgt_out)
        loss.backward()
        self.optimizer.step()

    # predicted: [batch, tgt_seq, vocab_size] <-- log probs output by model
    # actual:    [batch, tgt_seq, vocab_size] <-- one-hot of correct answer
    # ret:       scalar
    def loss(self, predicted, actual):
        # filter out positions which are PAD
        # [batch*tgt_seq, vocab_size]
        vocab_size   = predicted.size(-1)
        predicted    = predicted.reshape(-1, vocab_size)
        actual       = actual.reshape(-1, vocab_size)
        pad_idx      = self.vocab.tok_to_idx(SpecialTokens.PAD)
        non_pad_mask = (actual[:, pad_idx] != 1)
        predicted    = predicted[non_pad_mask]
        actual       = actual[non_pad_mask]
        num_toks     = torch.sum(non_pad_mask)

        # compute the smoothed counts according to label smoothing
        # [batch*tgt_seq, vocab_size]
        ls = self.label_smoothing
        actual_smoothed = (1 - ls) * actual + ls * self.label_smoothing_counts

        # compute the loss
        loss = actual_smoothed * predicted
        loss = torch.sum(loss) / num_toks        
        return loss

    # data: [batch, seq]
    def word_dropout(self, data, dropout):
        unk_idx = self.vocab.tok_to_idx(SpecialTokens.UNK)
        unk_tensor = torch.full_like(data, unk_idx)
        prob = torch.full_like(data, dropout)
        unk_mask = torch.bernoulli(prob)
        return data * (1 - unk_mask) + unk_tensor * unk_mask
