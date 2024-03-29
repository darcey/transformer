# TODO(darcey): is sort_by_tgt_only actually necessary?
# TODO(darcey): somewhere (maybe here?) the code should enforce that data is less than context window length
# TODO(darcey): improve code sharing between dataset classes
# TODO(darcey): think about adding code for Toan and Kenton's data augmentation strategy that glues multiple sentences together before training
# TODO(darcey): write dataset classes for classification tasks

import random
import numpy as np
import torch



class Seq2SeqTrainBatch:

    def __init__(self, src, tgt_in, tgt_out, num_src_toks, num_tgt_toks):
        self.src          = src
        self.tgt_in       = tgt_in
        self.tgt_out      = tgt_out
        self.num_src_toks = num_src_toks
        self.num_tgt_toks = num_tgt_toks

class Seq2SeqTrainDataset:

    # assumes src, tgt are lists of lists of token indices
    def __init__(self, src, tgt, toks_per_batch, pad_idx, bos_idx, eos_idx, sort_by_tgt_only=False, randomize=False):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        if sort_by_tgt_only:
            sorted_src, sorted_tgt = self.sort_by_tgt_len(src, tgt)
        else:
            sorted_src, sorted_tgt = self.sort_by_both_lens(src, tgt)
        self.batches = self.make_batches(sorted_src, sorted_tgt, toks_per_batch)
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

    # radix sort on src and tgt lengths
    def sort_by_both_lens(self, src, tgt):
        src_lens = [len(src_sent) for src_sent in src]
        sorted1_idxs = np.argsort(src_lens, kind="stable")
        sorted1_src = [src[i] for i in sorted1_idxs]
        sorted1_tgt = [tgt[i] for i in sorted1_idxs]

        tgt_lens = [len(tgt_sent) for tgt_sent in sorted1_tgt]
        sorted2_idxs = np.argsort(tgt_lens, kind="stable")
        sorted2_src = [sorted1_src[i] for i in sorted2_idxs]
        sorted2_tgt = [sorted1_tgt[i] for i in sorted2_idxs]

        return sorted2_src, sorted2_tgt

    def make_batches(self, src, tgt, toks_per_batch):
        batches = []

        curr_batch = []
        curr_src_toks = 0
        curr_tgt_toks = 0
        for (src_sent, tgt_sent) in zip(src, tgt):
            curr_batch.append((src_sent, tgt_sent))
            curr_src_toks += len(src_sent) + 1  # the +1 is for BOS/EOS
            curr_tgt_toks += len(tgt_sent) + 1
            if (curr_src_toks >= toks_per_batch) or (curr_tgt_toks >= toks_per_batch):
                batches.append(self.make_one_batch(curr_batch))
                curr_batch = []
                curr_src_toks = 0
                curr_tgt_toks = 0
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

        PAD = self.pad_idx
        BOS = self.bos_idx
        EOS = self.eos_idx
        for i, (src_sent, tgt_sent) in enumerate(sent_list):
            src_tensor[i]     = torch.tensor(src_sent + [EOS] + [PAD]*(max_src_len - len(src_sent)))
            tgt_in_tensor[i]  = torch.tensor([BOS] + tgt_sent + [PAD]*(max_tgt_len - len(tgt_sent)))
            tgt_out_tensor[i] = torch.tensor(tgt_sent + [EOS] + [PAD]*(max_tgt_len - len(tgt_sent)))

        num_src_toks = torch.sum(src_tensor != PAD)
        num_tgt_toks = torch.sum(tgt_in_tensor != PAD)

        return Seq2SeqTrainBatch(src_tensor, tgt_in_tensor, tgt_out_tensor, num_src_toks, num_tgt_toks)

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



class Seq2SeqTranslateSubdataset:

    def __init__(self, src, sents_per_batch, pad_idx, bos_idx, eos_idx):
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        
        src, self.orig_idxs = self.sort_by_len(src)

        self.batches = []
        for start in range(0, len(src), sents_per_batch):
            batch = self.make_one_batch(src[start:start+sents_per_batch])
            self.batches.append(batch)

    def __len__(self):
        return len(self.batches)

    def sort_by_len(self, src):
        lens = [len(src_sent) for src_sent in src]
        sorted_idxs = np.argsort(lens)
        sorted_src = [src[i] for i in sorted_idxs]
        orig_idxs = np.argsort(sorted_idxs)
        return sorted_src, orig_idxs

    def make_one_batch(self, src_list):
        num_sents = len(src_list)
        max_len = max([len(src_sent) for src_sent in src_list])

        PAD = self.pad_idx
        EOS = self.eos_idx
        src_tensor = torch.full((num_sents, max_len+1), 0)
        for i, src_sent in enumerate(src_list):
            src_tensor[i] = torch.tensor(src_sent + [EOS] + [PAD]*(max_len - len(src_sent)))

        return src_tensor

    # unpad a list of idxs
    def unpad(self, gen):
        tok_list = []
        for tok in gen:
            if tok == self.eos_idx:
                tok_list.append(tok)
                break
            elif tok == self.pad_idx:
                break
            else:
                tok_list.append(tok)
        return tok_list

    # takes target stuff output by the generator; untensors it
    # tgt_final: [batch, seq_len]
    # tgt_all:   [batch, beam, seq_len]
    # probs_all: [batch, beam]
    # return values:
    #   tgt_final: list of lists of idxs
    #   tgt_all:   list of lists of lists of idxs
    #   probs_all: list of lists of probabilities
    def unbatch(self, tgt_final, tgt_all, probs_all):
        tgt_final = [self.unpad(tgt_sent) for tgt_sent in tgt_final.tolist()]
        tgt_all = [[self.unpad(gen) for gen in tgt_sent] for tgt_sent in tgt_all.tolist()]
        probs_all = [[prob for prob in tgt_sent] for tgt_sent in probs_all.tolist()] 
        return tgt_final, tgt_all, probs_all

    def restore_order(self, tgt_list):
        order = np.argsort(np.argsort(self.orig_idxs))
        return [tgt_list[i] for i in order]

class Seq2SeqTranslateDataset:

    # assumes src is a list of lists of token indices
    def __init__(self, src, sents_per_batch, sents_per_subdataset, pad_idx, bos_idx, eos_idx):
        self.subdatasets = []
        for start in range(0, len(src), sents_per_subdataset):
            subdataset = Seq2SeqTranslateSubdataset(src[start:start+sents_per_subdataset], sents_per_batch, pad_idx, bos_idx, eos_idx)
            self.subdatasets.append(subdataset)

    def __len__(self):
        return len(self.subdatasets)
