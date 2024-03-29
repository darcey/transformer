# TODO(darcey): gracefully handle the appearance of the special token symbols in the data instead of just crashing
# TODO(darcey): code currently assumes that it's a seq2seq task (includes tokens BOS and EOS but not CLS); make an option for which type of task it is

from enum import Enum
from collections import Counter
import torch

# Guiding principle:
#   In places where a token would normally be referred to
#   using its string value, special tokens are just referred to
#   as SpecialToken.[tok]. The string values are only used during
#   printing. However, we do make sure there is no overlap between
#   the non-special tokens and the special tokens' string values.
class SpecialTokens(Enum):
    PAD = "<<PAD>>"
    UNK = "<<UNK>>"
    BOS = "<<BOS>>"
    EOS = "<<EOS>>"

class Vocabulary:

    def __init__(self):
        self.size = 0

    # assumes src, tgt are lists of lists of tokens
    def initialize_from_data(self, src_data, tgt_data):
        # build the token vocabulary from the data
        src_toks = Counter()
        tgt_toks = Counter()
        for src_sent in src_data:
            src_toks.update(src_sent)
        for tgt_sent in tgt_data:
            tgt_toks.update(tgt_sent)
        src_tgt_toks = src_toks + tgt_toks

        # build the set of special tokens
        # (stored as enum values, not strings like the other tokens)
        special_toks = [SpecialTokens.PAD, SpecialTokens.UNK, SpecialTokens.BOS, SpecialTokens.EOS]

        # make sure the data does not contain the special tokens' string values
        for tok in special_toks:
            if tok.value in src_tgt_toks.keys():
                raise ValueError(f"Vocabulary: data contains token {tok.value} which is reserved")

        # store vocab size
        l = len(special_toks)
        self.size = len(src_tgt_toks) + l

        # build the tok-to-idx and idx-to-tok mappings
        self.t_to_i  = {tok:idx for idx, tok in enumerate(special_toks)}
        self.i_to_t  = {idx:tok for idx, tok in enumerate(special_toks)}
        self.t_to_i |= {tok:(idx+l) for idx, (tok, _) in enumerate(src_tgt_toks.most_common())}
        self.i_to_t |= {(idx+l):tok for idx, (tok, _) in enumerate(src_tgt_toks.most_common())}

        # store the sets of tokens
        self.src_toks = set(src_toks.keys())
        self.tgt_toks = set(tgt_toks.keys())
        self.src_tgt_toks = set(src_tgt_toks.keys())
        self.special_toks = set(special_toks)

    def read_from_file(self, filename):
        src_toks = set()
        tgt_toks = set()
        special_toks = set()
        t_to_i = dict()
        i_to_t = dict()
        with open(filename) as f:
            for line in f:
                data = line.split()
                idx = int(data[0])
                tok = data[1]
                src = data[2]
                tgt = data[3]

                if src == "True":
                    src_toks.add(tok)
                if tgt == "True":
                    tgt_toks.add(tok)
                if src == "False" and tgt == "False":
                    tok = SpecialTokens(tok)
                    special_toks.add(tok)

                t_to_i[tok] = idx
                i_to_t[idx] = tok

        self.t_to_i = t_to_i
        self.i_to_t = i_to_t
        self.src_toks = src_toks
        self.tgt_toks = tgt_toks
        self.src_tgt_toks = src_toks | tgt_toks
        self.special_toks = special_toks
        self.size = len(self.t_to_i.keys())

    def write_to_file(self, filename):
        # pretty printing stuff
        num_len = len(str(len(self)))
        max_tok_len = 0
        for tok in self.src_tgt_toks:
            max_tok_len = max(max_tok_len, len(tok))
        for tok in self.special_toks:
            max_tok_len = max(max_tok_len, len(tok.value))

        with open(filename, "w") as f:
            for idx in range(len(self)):
                tok = self.idx_to_tok(idx)
                num_pad = num_len - len(str(idx))
                if tok in self.special_toks:
                    tok = tok.value
                tok_pad = max_tok_len - len(tok)
                src = "True " if tok in self.src_toks else "False"
                tgt = "True " if tok in self.tgt_toks else "False"
                f.write(f"{idx:>{num_len}}  {tok:>{max_tok_len}}  {src}  {tgt}\n")

    def __len__(self):
        return self.size

    def pad_idx(self):
        return self.tok_to_idx(SpecialTokens.PAD)
    def unk_idx(self):
        return self.tok_to_idx(SpecialTokens.UNK)
    def bos_idx(self):
        return self.tok_to_idx(SpecialTokens.BOS)
    def eos_idx(self):
        return self.tok_to_idx(SpecialTokens.EOS)

    # assumes data is a list of list of tokens
    def tok_to_idx_data(self, data, nesting=2):
        if nesting == 2:
            return [[self.tok_to_idx(tok) for tok in sent] for sent in data]
        elif nesting > 2:
            return [self.tok_to_idx_data(lists, nesting-1) for lists in data]
        else:
            raise ValueError("Nesting must be at least 2")
    # assumes data is a list of list of indices
    def idx_to_tok_data(self, data, nesting=2):
        if nesting == 2:
            return [[self.idx_to_tok(idx) for idx in sent] for sent in data]
        elif nesting > 2:
            return [self.idx_to_tok_data(lists, nesting-1) for lists in data]
        else:
            raise ValueError("Nesting must be at least 2")
    # assumes t is a single token
    def tok_to_idx(self, t):
        return self.t_to_i[t]
    # assumes i is a single index
    def idx_to_tok(self, i):
        return self.i_to_t[i]

    # assumes data is a list of list of tokens
    def remove_bos_eos_data(self, data, nesting=2):
        if nesting == 2:
            return [self.remove_bos_eos_sent(sent) for sent in data]
        elif nesting > 2:
            return [self.remove_bos_eos_data(lists, nesting-1) for lists in data]
        else:
            raise ValueError("Nesting must be at least 2")
    # assumes sent is a list of tokens
    def remove_bos_eos_sent(self, sent):
        start = 1 if (sent[0] == SpecialTokens.BOS) else 0
        end = -1 if (sent[-1] == SpecialTokens.EOS) else len(sent)
        return sent[start:end]

    # assumes data is a list of lists of tokens
    def unk_data(self, data, src=True):
        tok_set = self.src_toks if src else self.tgt_toks
        return [[self.unk_tok(tok, tok_set) for tok in sent] for sent in data]
    # assumes tok is a single token
    def unk_tok(self, tok, tok_set):
        return tok if tok in tok_set else SpecialTokens.UNK

    def get_tgt_support_mask(self):
        tgt_mask = [self.idx_to_tok(i) in self.tgt_toks for i in range(len(self))]
        tgt_mask[self.tok_to_idx(SpecialTokens.UNK)] = True
        tgt_mask[self.tok_to_idx(SpecialTokens.EOS)] = True
        return torch.tensor(tgt_mask)
