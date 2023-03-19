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

        config = get_config_arch()
        config.transformer_type = TransformerType.CUSTOM_ONE_SEQ
        config.output_probs = False
        config.use_masked_att_decoder = True
        config.num_decoder_layers = 1
        config.num_attention_heads = 1
        config.d_model = 2*k
        config.d_ff = 2*k
        config.use_resid_connection = False
        config.pos_enc_type = PositionalEncodingType.NONE
        config.norm_type = NormType.NONE
        
        t = transformer.get_transformer(config, vocab_size)
        
        emb = torch.zeros(2*k, 2*k)
        for j in range(k):
            emb[j, 2*j] = 1
            emb[j, 2*j+1] = -1
            emb[k+j, 2*j] = -1
            emb[k+j, 2*j+1] = 1
        emb = emb / math.sqrt(2*k)
        t.embedding.embedding = torch.nn.Parameter(emb)
        
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
