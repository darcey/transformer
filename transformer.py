# TODO(darcey): look into methods of initializing the parameters (see Toan's paper, section 2.2)
# TODO(darcey): remove dependence on max sentence len (in positional encoding)
# TODO(darcey): maybe switch the input to just be indices and not one-hots to save memory?
# TODO(darcey): consider switching to Brian's clever strategy for src/tgt masking

import math
import torch
from configuration import *



def get_embedding(config, vocab_size):
    return Embedding(vocab_size, config.arch.d_model, config.arch.fix_norm)

class Embedding(torch.nn.Module):

    def __init__(self, vocab_size, embed_dim, fix_norm):
        super().__init__()

        # FixNorm is from https://aclanthology.org/2019.iwslt-1.17.pdf
        self.fix_norm       = fix_norm

        self.embed_dim      = embed_dim
        self.embed_dim_sqrt = math.sqrt(embed_dim)
        self.embedding      = torch.nn.Parameter(torch.rand(vocab_size, embed_dim))

    # seq:  [batch, seq, vocab_size]
    # ret:  [batch, seq, d_model]
    def forward(self, seq, reverse=False):
        if self.fix_norm:
            emb_mat = torch.nn.functional.normalize(self.embedding, dim=-1)
        else:
            emb_mat = self.embedding

        if not reverse:
            return torch.matmul(seq, emb_mat) * self.embed_dim_sqrt
        else:
            return torch.matmul(seq, torch.t(emb_mat))



def get_positional_encoding(config):
    match config.arch.pos_enc_type:
        case PositionalEncodingType.NONE:
            return NullPositionalEncoding()
        case PositionalEncodingType.SINUSOIDAL:
            return SinusoidalPositionalEncoding(config.arch.context_window_length, config.arch.d_model)

class NullPositionalEncoding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    # x:   [batch, seq, d_model]
    # ret: [batch, seq, d_model]
    def forward(self, x):
        return torch.zeros_like(x)

class SinusoidalPositionalEncoding(torch.nn.Module):

    # Based on Toan Nguyen's implementation
    def __init__(self, max_len, dim, denom_base=10000):
        super().__init__()

        pos = torch.arange(max_len).unsqueeze(1)       # [seq, 1]
        idx = torch.floor_divide(torch.arange(dim), 2) # [d_model]
        div = torch.pow(denom_base, 2*idx/dim)         # [d_model]
        pe = pos / div                                 # [seq, d_model]
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        self.register_buffer('pe', pe)

    # x:   [batch, seq, d_model]
    # ret:        [seq, d_model]
    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len, :]



def get_normalization(config):
    match config.arch.norm_type:
        case NormType.NONE:
            return torch.nn.Identity()
        case NormType.LAYER_NORM:
            return LayerNorm(config.arch.d_model, config.arch.layer_norm_epsilon)
        case NormType.SCALE_NORM:
            return ScaleNorm()

class LayerNorm(torch.nn.Module):

    # Layer Normalization: https://arxiv.org/pdf/1607.06450.pdf
    def __init__(self, dim, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.gamma   = torch.nn.Parameter(torch.rand(dim))
        self.beta    = torch.nn.Parameter(torch.rand(dim))

    # x:   [batch, seq, d_model]
    # ret: [batch, seq, d_model]
    def forward(self, x):
        mu       = torch.mean(x, dim=-1, keepdim=True)                 # [batch, seq, 1]
        sigma_sq = torch.var(x, dim=-1, unbiased=False, keepdim=True)  # [batch, seq, 1]
        return self.gamma * (x - mu) / torch.sqrt(sigma_sq + self.epsilon) + self.beta

class ScaleNorm(torch.nn.Module):

    # Scale Norm: https://aclanthology.org/2019.iwslt-1.17.pdf
    def __init__(self):
        super().__init__()
        self.g = torch.nn.Parameter(torch.rand(()))

    # x:   [batch, seq, d_model]
    # ret: [batch, seq, d_model]
    def forward(self, x):
        return self.g * torch.nn.functional.normalize(x, dim=-1)



def get_feed_forward(config):
    return FeedForward(config.arch.d_model, config.arch.d_ff, config.train.ff_dropout)

class FeedForward(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.layer1  = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer2  = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout)

    # x:   [batch, seq, d_model]
    # ret: [batch, seq, d_model]
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x



def get_attention(config):
    return MultiHeadAttention(config.arch.d_model, config.arch.num_attention_heads, dropout=config.train.att_dropout)

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, input_dim, num_heads, qk_dim=None, v_dim=None, dropout=0.3):
        super().__init__()

        if qk_dim == None or v_dim == None:
            if input_dim % num_heads > 0:
                raise ValueError("MultiHeadAttention: input_dim should be divisible by num_heads")

        self.num_heads = num_heads
        self.qk_dim    = qk_dim if qk_dim else int(input_dim / num_heads)
        self.v_dim     = v_dim  if v_dim  else int(input_dim / num_heads)

        self.proj_q   = torch.nn.Linear(input_dim, num_heads*self.qk_dim, bias=False)
        self.proj_k   = torch.nn.Linear(input_dim, num_heads*self.qk_dim, bias=False)
        self.proj_v   = torch.nn.Linear(input_dim, num_heads*self.v_dim, bias=False)
        self.proj_out = torch.nn.Linear(num_heads*self.v_dim, input_dim, bias=False)

        self.dropout = torch.nn.Dropout(p=dropout)

    # q:    [batch, seq1, d_input]
    # k:    [batch, seq2, d_input]
    # v:    [batch, seq2, d_input]
    # mask: [seq1, seq2]
    # ret:  [batch, seq1, d_input]
    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        seq1 = q.size(1)
        seq2 = k.size(1)

        # project to heads
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # reshape to heads, permute for matrix multiplication
        q = q.reshape(batch, seq1, self.num_heads, self.qk_dim).permute((0,2,1,3))
        k = k.reshape(batch, seq2, self.num_heads, self.qk_dim).permute((0,2,3,1))
        v = v.reshape(batch, seq2, self.num_heads, self.v_dim).permute((0,2,1,3))

        # do multihead attention
        key_queries = torch.matmul(q,k)/math.sqrt(self.qk_dim)
        if mask is not None:
            key_queries += mask
        probs = torch.softmax(key_queries, dim=-1)
        probs = self.dropout(probs)
        ret = torch.matmul(probs, v)

        # reshape back
        ret = ret.reshape(batch, seq1, self.num_heads*self.v_dim)

        # project back
        ret = self.proj_out(ret)

        return ret



class SublayerConnection(torch.nn.Module):

    def __init__(self, sublayer_func, sublayer, config):
        super().__init__()
        self.use_resid     = config.arch.use_resid_connection

        # PreNorm is from https://aclanthology.org/2019.iwslt-1.17.pdf
        self.pre_norm      = config.arch.pre_norm
        self.norm          = get_normalization(config)
        self.dropout       = torch.nn.Dropout(p=config.train.dropout)

        self.sublayer      = sublayer
        self.sublayer_func = sublayer_func

    # prev_seq: [batch, prev_seq, d_model]
    # this_seq: [batch, this_seq, d_model]
    # mask:     [this_seq, this_seq]
    # ret:      [batch, this_seq, d_model]
    def forward(self, this_seq, *other_inputs):
        ret = this_seq
        ret = self.norm(ret) if self.pre_norm else ret
        ret = self.sublayer_func(self.sublayer, ret, *other_inputs)
        ret = self.dropout(ret)
        ret = ret + this_seq if self.use_resid else ret
        ret = self.norm(ret) if not self.pre_norm else ret
        return ret



class Layer(torch.nn.Module):

    def __init__(self, config, take_two_seqs, use_mask):
        super().__init__()
        self.use_resid     = config.arch.use_resid_connection
        self.use_mask      = use_mask
        self.take_two_seqs = take_two_seqs

        self_attention         = get_attention(config)
        self_att_func          = lambda s, y, m: s(y, y, y, m)
        self.self_att_sublayer = SublayerConnection(self_att_func, self_attention, config)

        if take_two_seqs:
            cross_attention         = get_attention(config)
            cross_att_func          = lambda s, y, x: s(y, x, x)
            self.cross_att_sublayer = SublayerConnection(cross_att_func, cross_attention, config)

        feed_forward      = get_feed_forward(config)
        feed_forward_func = lambda s, y: s(y)
        self.ff_sublayer  = SublayerConnection(feed_forward_func, feed_forward, config)

    # prev_seq: [batch, prev_seq, d_model]
    # this_seq: [batch, this_seq, d_model]
    # mask:     [this_seq, this_seq]
    # ret:      [batch, this_seq, d_model]
    def forward(self, this_seq, prev_seq=None, mask=None):
        if ((prev_seq == None) != (not self.take_two_seqs)) or\
           ((mask == None) != (not self.use_mask)):
            raise ValueError("Layer: bad combination of arguments")

        ret = this_seq
        ret = self.self_att_sublayer(ret, mask)
        ret = self.cross_att_sublayer(ret, prev_seq) if self.take_two_seqs else ret
        ret = self.ff_sublayer(ret) 
        return ret



# Implements an Encoder, Decoder, or anything in between.
#   Encoder:       takes one sequence,   no masked self-attention
#   Decoder:       takes two sequences, has masked self-attention
#   Decoder only:  takes one sequence,  has masked self-attention
#   ???:           takes two sequences,  no masked self-attention
class EncoderOrDecoder(torch.nn.Module):

    def __init__(self, config, num_layers, take_two_seqs, use_mask):
        super().__init__()
        self.use_mask      = use_mask
        self.pre_norm      = config.arch.pre_norm
        if self.pre_norm:
            self.norm      = get_normalization(config)
        self.layers        = torch.nn.ModuleList([Layer(config, take_two_seqs, use_mask) for _ in range(num_layers)])

    # prev_seq: [batch, prev_seq, d_model]
    # this_seq: [batch, this_seq, d_model]
    # ret:      [batch, this_seq, d_model]
    def forward(self, this_seq, prev_seq=None):
        if self.use_mask:
            l = this_seq.size(1)
            mask = torch.triu(torch.full((l, l), float('-inf')), diagonal=1)
            mask = mask.type(this_seq.type())
        else:
            mask = None        

        ret = this_seq
        for layer in self.layers:
            ret = layer(ret, prev_seq, mask)
        ret = self.norm(ret) if self.pre_norm else ret
        return ret



# Transformer model, https://arxiv.org/pdf/1706.03762.pdf
# For maximum flexibility, I have implemented two versions, one which
# takes two sequences as input and another which takes one sequence.
# The logic in get_transformer configures these two basic templates
# into the three standard EncoderDecoder, EncoderOnly, and DecoderOnly
# models, as well as custom options.
def get_transformer(config, vocab_size, tgt_support_mask=None):
    match config.arch.transformer_type:
        case TransformerType.ENCODER_DECODER:
            return TransformerTwoSeq(config,
                                     num_enc_layers=config.arch.num_encoder_layers,
                                     use_mask_enc=False,
                                     num_dec_layers=config.arch.num_decoder_layers,
                                     use_mask_dec=True,
                                     output_probs=True,
                                     vocab_size=vocab_size,
                                     tgt_support_mask=tgt_support_mask)
        case TransformerType.ENCODER_ONLY:
            return TransformerOneSeq(config,
                                     num_layers=config.arch.num_encoder_layers,
                                     use_mask=False,
                                     output_probs=False,
                                     vocab_size=vocab_size,
                                     support_mask=tgt_support_mask)
        case TransformerType.DECODER_ONLY:
            return TransformerOneSeq(config,
                                     num_layers=config.arch.num_decoder_layers,
                                     use_mask=True,
                                     output_probs=True,
                                     vocab_size=vocab_size,
                                     support_mask=tgt_support_mask)
        case TransformerType.CUSTOM_TWO_SEQ:
            return TransformerTwoSeq(config,
                                     num_enc_layers=config.arch.num_encoder_layers,
                                     use_mask_enc=config.arch.use_masked_att_encoder,
                                     num_dec_layers=config.arch.num_decoder_layers,
                                     use_mask_dec=config.arch.use_masked_att_decoder,
                                     output_probs=config.arch.output_probs,
                                     vocab_size=vocab_size,
                                     tgt_support_mask=tgt_support_mask)
        case TransformerType.CUSTOM_ONE_SEQ:
            return TransformerOneSeq(config,
                                     num_layers=config.arch.num_decoder_layers,
                                     use_mask=config.arch.use_masked_att_decoder,
                                     output_probs=config.arch.output_probs,
                                     vocab_size=vocab_size,
                                     support_mask=tgt_support_mask)

class TransformerTwoSeq(torch.nn.Module):

    def __init__(self, config, num_enc_layers, use_mask_enc, num_dec_layers, use_mask_dec, output_probs, vocab_size, tgt_support_mask=None):
        super().__init__()
        self.output_probs = output_probs
        if self.output_probs:
            if tgt_support_mask == None:
                tgt_support_mask = torch.tensor([0.0]*vocab_size)
            else:
                tgt_support_mask = torch.log(tgt_support_mask.type(torch.float))
            self.register_buffer('tgt_support_mask', tgt_support_mask)

        self.embedding  = get_embedding(config, vocab_size)
        self.positional = get_positional_encoding(config)
        self.dropout    = torch.nn.Dropout(p=config.train.dropout)
        self.encoder    = EncoderOrDecoder(config,
                                           num_layers=num_enc_layers,
                                           take_two_seqs=False,
                                           use_mask=use_mask_enc)
        self.decoder    = EncoderOrDecoder(config,
                                           num_layers=num_dec_layers,
                                           take_two_seqs=True,
                                           use_mask=use_mask_dec)

    # src: [batch, src_seq, vocab_size]
    # tgt: [batch, tgt_seq, vocab_size]
    # ret: [batch, tgt_seq, vocab_size]
    def forward(self, src, tgt):
        src_embed  = self.embedding(src)
        src_embed += self.positional(src)
        src_input  = self.dropout(src_embed)
        src_output = self.encoder(src_input)

        tgt_embed  = self.embedding(tgt)
        tgt_embed += self.positional(tgt)
        tgt_input  = self.dropout(tgt_embed)
        tgt_output = self.decoder(tgt_input, src_output)
        if not self.output_probs:
            return tgt_output
        else:
            tgt_logits = self.embedding(tgt_output, reverse=True)
            tgt_logits_masked = tgt_logits + self.tgt_support_mask
            tgt_probs = torch.nn.functional.log_softmax(tgt_logits_masked, dim=-1)
            return tgt_probs

class TransformerOneSeq(torch.nn.Module):

    def __init__(self, config, num_layers, use_mask, output_probs, vocab_size, support_mask=None):
        super().__init__()
        self.output_probs = output_probs
        if self.output_probs:
            if support_mask == None:
                support_mask = torch.tensor([0.0]*vocab_size)
            else:
                support_mask = torch.log(support_mask.type(torch.float))
            self.register_buffer('support_mask', support_mask)

        self.embedding  = get_embedding(config, vocab_size)
        self.positional = get_positional_encoding(config)
        self.dropout    = torch.nn.Dropout(p=config.train.dropout)
        self.xxcoder    = EncoderOrDecoder(config,
                                           num_layers=num_layers,
                                           take_two_seqs=False,
                                           use_mask=use_mask)

    # seq: [batch, seq, vocab_size]
    # ret: [batch, seq, vocab_size]
    def forward(self, seq):
        seq_embed  = self.embedding(seq)
        seq_embed += self.positional(seq)
        seq_input  = self.dropout(seq_embed)
        seq_output = self.xxcoder(seq_input)
        if not self.output_probs:
            return seq_output
        else:
            seq_logits = self.embedding(seq_output, reverse=True)
            seq_logits_masked = seq_logits + self.support_mask
            seq_probs = torch.nn.functional.log_softmax(seq_logits_masked, dim=-1)
            return seq_probs
