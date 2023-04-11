# TODO(darcey): might be good to have a way to add data one sentence at a time instead of all at once, but, not important right now
# TODO(darcey): gracefully handle the appearance of the special token symbols in the data instead of just crashing
# TODO(darcey): code currently assumes that it's a seq2seq task (includes tokens BOS and EOS but not CLS); make an option for which type of task it is
# TODO(darcey): add functionality for reading/writing to a file

from enum import Enum
from collections import Counter
import torch

# Guiding principle:
#   These tokens are always referred to using either their enum name (e.g. SpecialTokens.PAD)
#   or their index in a particular vocabulary. The string values are only used when printing.
#   However, we make sure the vocabulary doesn't have any tokens which match the string values,
#   to avoid bugs and confusion.
class SpecialTokens(Enum):
    PAD = "<<PAD>>"
    UNK = "<<UNK>>"
    BOS = "<<BOS>>"
    EOS = "<<EOS>>"
    CLS = "<<CLS>>"

class Vocabulary():

    # assumes src, tgt are lists of lists of tokens
    def __init__(self, src_data, tgt_data):
        # build the token vocabulary from the data
        self.src_toks = Counter()
        self.tgt_toks = Counter()
        for src_sent in src_data:
            self.src_toks.update(src_sent)
        for tgt_sent in tgt_data:
            self.tgt_toks.update(tgt_sent)
        self.all_toks = self.src_toks + self.tgt_toks

        # build the set of special tokens
        # (stored as enum values, not strings like the other tokens)
        self.special_toks = [SpecialTokens.PAD, SpecialTokens.UNK, SpecialTokens.BOS, SpecialTokens.EOS]

        # store vocab size
        l = len(self.special_toks)
        self.size = len(self.all_toks.keys()) + l

        # make sure the data does not contain the special tokens' string values
        for tok in self.special_toks:
            if tok.value in self.all_toks.keys():
                raise ValueError(f"Vocabulary: data contains token {tok.value} which is reserved")

        # build the tok-to-idx and idx-to-tok mappings
        self.t_to_i  = {tok:idx for idx, tok in enumerate(self.special_toks)}
        self.i_to_t  = {idx:tok for idx, tok in enumerate(self.special_toks)}
        self.t_to_i |= {tok:(idx+l) for idx, (tok, _) in enumerate(self.all_toks.most_common())}
        self.i_to_t |= {(idx+l):tok for idx, (tok, _) in enumerate(self.all_toks.most_common())}

    def __len__(self):
        return self.size

    def has_tok(self, tok):
        return (tok in self.special_toks) or (tok in self.all_toks.keys())

    def has_src_tok(self, tok):
        return tok in self.src_toks.keys()

    def has_tgt_tok(self, tok):
        return tok in self.tgt_toks.keys()

    def get_toks(self):
        return set(self.special_toks) | set(self.all_toks.keys())

    def get_src_toks(self):
        return set(self.src_toks.keys())

    def get_tgt_toks(self):
        return set(self.tgt_toks.keys())

    def get_special_toks(self):
        return set(self.special_toks)

    # assumes t is either a single token or a list of tokens
    def tok_to_idx(self, t):
        if isinstance(t, str) or t in self.special_toks:
            return self.t_to_i[t]
        else:
            return [self.t_to_i[tok] for tok in t]

    # assumes i is either a single index or a list of indices
    def idx_to_tok(self, i):
        if isinstance(i, int):
            return self.i_to_t[i]
        else:
            return [self.i_to_t[idx] for idx in i]

    # assumes sent is a list of tokens
    def unk_src(self, sent):
        return [tok if self.has_src_tok(tok) else SpecialTokens.UNK for tok in sent]

    # assumes sent is a list of tokens
    def unk_tgt(self, sent):
        return [tok if self.has_tgt_tok(tok) else SpecialTokens.UNK for tok in sent]
