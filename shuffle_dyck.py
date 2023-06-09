# TODO(darcey): implement the Shuffle-Dyck recognizer using only Transformer components

import math
import torch
from configuration import *
import transformer



# Transformer for Shuffle-Dyck languages, following
# https://aclanthology.org/2020.emnlp-main.576.pdf
class ShuffleDyckRecognizer():

    def __init__(self, k):

        # vocab is [0, ..., [k-1, ]0, ..., ]k-1
        vocab_size = 2*k

        config = read_config("configuration.toml")
        config.arch.transformer_type = TransformerType.CUSTOM_ONE_SEQ
        config.arch.output_probs = False
        config.arch.use_masked_att_decoder = True
        config.arch.num_decoder_layers = 1
        config.arch.num_attention_heads = 1
        config.arch.d_model = 2*k
        config.arch.d_ff = 2*k
        config.arch.use_resid_connection = False
        config.arch.pos_enc_type = PositionalEncodingType.NONE
        config.arch.norm_type = NormType.NONE

        t = transformer.get_transformer(config, vocab_size, pad_idx=-1)

        emb = torch.zeros(2*k, 2*k)
        for j in range(k):
            emb[j, 2*j] = 1
            emb[j, 2*j+1] = -1
            emb[k+j, 2*j] = -1
            emb[k+j, 2*j+1] = 1
        emb = emb / math.sqrt(2*k)
        embedding = torch.nn.Parameter(emb)
        t.input.embedding.embedding = embedding

        proj_k = torch.zeros(2*k, 2*k)
        proj_v = torch.eye(2*k, 2*k)
        proj_out = torch.eye(2*k, 2*k)
        t.xxcoder.layers[0].self_attention.proj_k.weight = torch.nn.Parameter(proj_k)
        t.xxcoder.layers[0].self_attention.proj_v.weight = torch.nn.Parameter(proj_v)
        t.xxcoder.layers[0].self_attention.proj_out.weight = torch.nn.Parameter(proj_out)

        t.xxcoder.layers[0].feed_forward.layer1 = torch.nn.Linear(2*k, 2*k, bias=False)
        t.xxcoder.layers[0].feed_forward.layer2 = torch.nn.Linear(2*k, 2*k, bias=False)
        t.xxcoder.layers[0].feed_forward.layer1.weight = torch.nn.Parameter(torch.eye(2*k, 2*k))
        t.xxcoder.layers[0].feed_forward.layer2.weight = torch.nn.Parameter(torch.eye(2*k, 2*k))

        self.k = k
        self.vocab_size = vocab_size
        self.transformer = t

    # x:   [batch, seq_len, vocab_size]
    # ret: boolean
    def recognize(self, x):
        zeros = torch.zeros(x.size(0), self.k)
        encoding = self.transformer(x)
        enc_sum_odd = torch.sum(encoding, dim=1)[:, 1::2]
        enc_final_even = encoding[:, -1, 0::2]
        return torch.equal(enc_sum_odd, zeros) and torch.equal(enc_final_even, zeros)
