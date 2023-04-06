# TODO(darcey): look at original Vaswani et al. paper
# TODO(darcey): finish reading Toan's paper for additional insights
# TODO(darcey): word dropout? either here or in the transformer file
# TODO(darcey): make a learning config, add label smoothing settings to it (allow option to include EOS in label smoothing mask); also decide what else to exclude from the label smoothing mask (BOS, EOS, etc.); probably the best way to do this is to have the Vocabulary class include functions like do_not_generate and do_not_smooth; I would like this file to be as agnostic as possible about what special tokens the vocabulary contains. I don't know how I can write the loss function without awareness of the PAD token, though, and it seems task agnostic, so I am allowing this file to know about that.
# TODO(darcey): improve torch efficiency throughout codebase (switch from reshape to view? bmm vs. matmul? stop using one-hots where possible?)
# TODO(darcey): make a learning config, add label smoothing settings to it (allow option to include e.g. EOS in label smoothing mask)
# TODO(darcey): implement classifier learning also

import torch
import vocabulary

class Trainer():

    def __init__(self, model, vocab_size):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        # TODO(darcey): fix this (see TODO note above)
        self.label_smoothing = 0.1
        ls_counts = torch.tensor([1.0]*vocab_size)
        pad_idx = vocabulary.SpecialTokens.PAD.value
        ls_counts[pad_idx] = 0
        ls_counts = ls_counts / torch.sum(ls_counts)
        self.label_smoothing_counts = ls_counts

        return

#    def train(self, train, dev):
#        for epoch in range(self.max_epochs):
#            for _ in range(self.epoch_size):
#                (src, tgt_in, tgt_out) = train.next_batch()
#                train_one_step
#            [evaluate perplexity]
#            [evaluate bleu]
#            [save checkpoints as needed]
#            [adjust learning rate / early stopping]

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
        pad_idx      = vocabulary.SpecialTokens.PAD.value
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
