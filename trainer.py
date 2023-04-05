# TODO(darcey): look at original Vaswani et al. paper
# TODO(darcey): finish reading Toan's paper for additional insights
# TODO(darcey): word dropout? either here or in the transformer file
# TODO(darcey): improve torch efficiency throughout codebase (switch from reshape to view? bmm vs. matmul? stop using one-hots where possible?)
# TODO(darcey): make a learning config, add label smoothing settings to it (allow option to include e.g. EOS in label smoothing mask)

import torch
import vocab

class Trainer():

    def __init__(self):
        return
        # TODO(darcey): make config, incorporate label smoothing + EOS masking into config
        #self.label_smoothing = 0.1
        #self.label_smoothing_counts = torch.tensor([1.0]*vocab_size) / vocab_size

#    def train(self, train, dev):
#        for epoch in range(self.max_epochs):
#            for _ in range(self.epoch_size):
#                (src, tgt) = train.next_batch()
#                probs = model(src, tgt)
#                loss = compute_loss(probs, tgt)
#                [do gradient update]
#                [evaluate perplexity]
#                [evaluate bleu]
#                [save checkpoints as needed]
#                [adjust learning rate / early stopping]

    # predicted: [batch, tgt_seq, vocab_size] <-- log probs output by model
    # actual:    [batch, tgt_seq, vocab_size] <-- one-hot of correct answer
    # ret:       scalar
    def loss(self, predicted, actual):
        # filter out positions which are PAD
        # [batch*tgt_seq, vocab_size]
        vocab_size   = predicted.size(-1)
        predicted    = predicted.reshape(-1, vocab_size)
        actual       = actual.reshape(-1, vocab_size)
        pad_idx      = vocab.SpecialTokens.PAD.value
        non_pad_mask = (actual[:, pad_idx] == 0)
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
