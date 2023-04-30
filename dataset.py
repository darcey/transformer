# TODO(darcey): write the seq2seq test dataset code

# TODO(darcey): convert batch from dict to object so it can be worked with more easily
# TODO(darcey): write dataset classes for classification tasks

import random
import numpy as np
import torch
from vocabulary import SpecialTokens



class Seq2SeqTrainDataset():

    # assumes src, tgt are lists of lists of token indices
    # sorts by tgt length because that determines number of training steps
    def __init__(self, src, tgt, vocab, batch_size, randomize=False):
        self.vocab = vocab
        sorted_src, sorted_tgt = self.sort_by_tgt_len(src, tgt)
        self.batches = self.make_batches(sorted_src, sorted_tgt, batch_size)
        self.randomize = randomize
        self.num_iters = 0
        self.batch_iter = self.get_batch_iter()

    def __len__(self):
        return len(self.batches)

    def sort_by_tgt_len(self, src, tgt):
        lens = [len(tgt_sent) for tgt_sent in tgt]
        sorted_idxs = np.argsort(lens)
        sorted_src = [src[i] for i in sorted_idxs]
        sorted_tgt = [tgt[i] for i in sorted_idxs]
        return sorted_src, sorted_tgt

    def make_batches(self, src, tgt, batch_size):
        batches = []

        curr_batch = []
        curr_num_toks = 0
        for (src_sent, tgt_sent) in zip(src, tgt):
            curr_batch.append((src_sent, tgt_sent))
            curr_num_toks += len(tgt_sent) + 1
            if curr_num_toks >= batch_size:
                batches.append(self.make_one_batch(curr_batch))
                curr_batch = []
                curr_num_toks = 0
        if len(curr_batch) > 0:
            batches.append(self.make_one_batch(curr_batch))

        return batches

    def make_one_batch(self, sent_list):
        num_sents   = len(sent_list)
        max_src_len = max([len(src_sent) for (src_sent, _) in sent_list])
        max_tgt_len = max([len(tgt_sent) for (_, tgt_sent) in sent_list])

        src_tensor     = torch.full((num_sents, max_src_len+1), 0)
        tgt_in_tensor  = torch.full((num_sents, max_tgt_len+1), 0)
        tgt_out_tensor = torch.full((num_sents, max_tgt_len+1), 0)

        PAD = [self.vocab.tok_to_idx(SpecialTokens.PAD)]
        BOS = [self.vocab.tok_to_idx(SpecialTokens.BOS)]
        EOS = [self.vocab.tok_to_idx(SpecialTokens.EOS)]
        for i, (src_sent, tgt_sent) in enumerate(sent_list):
            src_tensor[i]     = torch.tensor(src_sent + EOS + PAD*(max_src_len - len(src_sent)))
            tgt_in_tensor[i]  = torch.tensor(BOS + tgt_sent + PAD*(max_tgt_len - len(tgt_sent)))
            tgt_out_tensor[i] = torch.tensor(tgt_sent + EOS + PAD*(max_tgt_len - len(tgt_sent)))

        pad_idx = self.vocab.tok_to_idx(SpecialTokens.PAD)
        num_src_toks = torch.sum(src_tensor != pad_idx)
        num_tgt_toks = torch.sum(tgt_in_tensor != pad_idx)

        return {
            "src": src_tensor,
            "tgt_in": tgt_in_tensor,
            "tgt_out": tgt_out_tensor,
            "num_src_toks": num_src_toks,
            "num_tgt_toks": num_tgt_toks,
        }

    def get_batch(self):
        try:
            batch = next(self.batch_iter)
        except StopIteration:
            self.num_iters += 1
            self.batch_iter = self.get_batch_iter()
            batch = next(self.batch_iter)
        return batch

    def get_batch_iter(self):
        order = list(range(len(self.batches)))
        if self.randomize:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]



#class Seq2SeqTranslateDataset():
#    def __init__(self, src):
        # sort src by length
        # need to keep track of original idxs for organizing the translations
        # (maybe include an option of whether it should be length batched or not? might want to do translation in dataset order so we can print intermediate results / clear the buffer, for cases with large # of samples)
        # divide up into batches by number of concurrent beams
        # need to consider very large (larger than one batch) numbers of samples; that should be dealt with during batching
        # need iterator that gives the next batch
        # need function that takes in translations and adds them to the ordered list of translations
        # need to think about how this will interact with writing the translations to file
