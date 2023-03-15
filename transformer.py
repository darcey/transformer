# TODO(darcey): implement alternatives to LayerNorm like BatchNorm and ScaleNorm (see Toan's paper)
# TODO(darcey): figure out if I need to do all the to_device stuff here
# TODO(darcey): figure out if I need to include dtype information here
# TODO(darcey): add dropout
# TODO(darcey): look into better ways of initializing the parameters
# TODO(darcey): remove dependence on max sentence len (in positional encoding)
# TODO(darcey): consider switching to Brian's clever strategy for src/tgt masking

import math
import torch
from configuration import *



def get_embedding(config, vocab_size):
    return Embedding(vocab_size, config.d_model)

class Embedding(torch.nn.Module):
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Parameter(torch.rand(vocab_size, embed_dim))

    # seq:  [batch, seq, vocab_size]
    # ret:  [batch, seq, d_model]
    def forward(self, seq, reverse=False):
        if not reverse:
            return torch.matmul(seq, self.embedding) * math.sqrt(self.embed_dim)
        else:
            return torch.matmul(seq, torch.t(self.embedding))



def get_positional_encoding(config):
    return SinusoidalPositionalEncoding(config.context_window_length, config.d_model)

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
    match config.norm_type:
        case NormType.NONE:
            return torch.nn.Identity()
        case NormType.LAYER_NORM:
            return LayerNorm(config.d_model, config.layer_norm_epsilon)

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



def get_feed_forward(config):
    return FeedForward(config.d_model, config.d_ff)

class FeedForward(torch.nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.layer2 = torch.nn.Linear(hidden_dim, input_dim, bias=True)
    
    # x:   [batch, seq, d_model]
    # ret: [batch, seq, d_model]
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        return x



def get_attention(config):
    return MultiHeadAttention(config.d_model, config.num_attention_heads)

class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, input_dim, num_heads, qk_dim=None, v_dim=None):
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
        ret = torch.matmul(probs, v)
        
        # reshape back
        ret = ret.reshape(batch, seq1, self.num_heads*self.v_dim)
        
        # project back
        ret = self.proj_out(ret)
        
        return ret



class EncoderLayer(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = get_attention(config)
        self.norm1          = get_normalization(config)
        self.feed_forward   = get_feed_forward(config)
        self.norm2          = get_normalization(config)

    # src: [batch, src_seq, d_model]
    # ret: [batch, src_seq, d_model]
    def forward(self, src):
        resid  = src
        resid += self.self_attention(resid, resid, resid)
        resid  = self.norm1(resid)
        resid += self.feed_forward(resid)
        resid  = self.norm2(resid)
        return resid

class Encoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])

    # src: [batch, src_seq, d_model]
    # ret: [batch, src_seq, d_model]
    def forward(self, src):
        resid = src
        for layer in self.layers:
            resid = layer(resid)
        return resid



class DecoderLayer(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention  = get_attention(config)
        self.norm1           = get_normalization(config)
        self.cross_attention = get_attention(config)
        self.norm2           = get_normalization(config)
        self.feed_forward    = get_feed_forward(config)
        self.norm3           = get_normalization(config)

    # src_enc: [batch, src_seq, d_model]
    # tgt:     [batch, tgt_seq, d_model]
    # mask:    [tgt_seq, tgt_seq]
    # ret:     [batch, tgt_seq, d_model]
    def forward(self, tgt, src_enc, mask):
        resid  = tgt
        resid += self.self_attention(resid, resid, resid, mask)
        resid  = self.norm1(resid)
        resid += self.cross_attention(resid, src_enc, src_enc)
        resid  = self.norm2(resid)
        resid += self.feed_forward(resid)
        resid  = self.norm3(resid)
        return resid

class Decoder(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
    
    # src_enc: [batch, src_seq, d_model]
    # tgt:     [batch, tgt_seq, d_model]
    # ret:     [batch, tgt_seq, d_model]
    def forward(self, tgt, src_enc):
        mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf')), diagonal=1)
        resid = tgt
        for layer in self.layers:
            resid = layer(resid, src_enc, mask)
        return resid



class DecoderOnlyLayer(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = get_attention(config)
        self.norm1          = get_normalization(config)
        self.feed_forward   = get_feed_forward(config)
        self.norm2          = get_normalization(config)

    # tgt:  [batch, tgt_seq, d_model]
    # mask: [tgt_seq, tgt_seq]
    # ret:  [batch, tgt_seq, d_model]
    def forward(self, tgt, mask):
        resid  = tgt
        resid += self.self_attention(resid, resid, resid, mask)
        resid  = self.norm1(resid)
        resid += self.feed_forward(resid)
        resid  = self.norm2(resid)
        return resid

class DecoderOnly(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderOnlyLayer(config) for _ in range(config.num_decoder_layers)])
    
    # tgt: [batch, tgt_seq, d_model]
    # ret: [batch, tgt_seq, d_model]
    def forward(self, tgt):
        mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf')), diagonal=1)
        resid = tgt
        for layer in self.layers:
            resid = layer(resid, mask)
        return resid
        


# Transformer model, https://arxiv.org/pdf/1706.03762.pdf
class TransformerEncoderDecoder(torch.nn.Module):

    def __init__(self, config, vocab_size, tgt_mask):
        super().__init__()
        self.tgt_mask   = tgt_mask
        self.embedding  = get_embedding(config, vocab_size)
        self.positional = get_positional_encoding(config)
        self.encoder    = Encoder(config)
        self.decoder    = Decoder(config)
        
    # src: [batch, src_seq, vocab_size]
    # tgt: [batch, tgt_seq, vocab_size]
    # ret: [batch, tgt_seq, vocab_size]
    def forward(self, src, tgt):
        src_embed  = self.embedding(src)
        src_embed += self.positional(src)
        src_output = self.encoder(src_embed)
        
        tgt_embed  = self.embedding(tgt)
        tgt_embed += self.positional(tgt)
        tgt_output = self.decoder(tgt_embed, src_output)
        tgt_logits = self.embedding(tgt_output, reverse=True)
        tgt_probs  = torch.nn.functional.log_softmax(tgt_logits[:,:,self.tgt_mask], dim=-1)
        return tgt_probs

class TransformerEncoderOnly(torch.nn.Module):

    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding  = get_embedding(config, vocab_size)
        self.positional = get_positional_encoding(config)
        self.encoder    = Encoder(config)
        
    # src: [batch, src_seq, vocab_size]
    # ret: [batch, src_seq, vocab_size]
    def forward(self, src):
        src_embed  = self.embedding(src)
        src_embed += self.positional(src)
        src_output = self.encoder(src_embed)
        return src_output

class TransformerDecoderOnly(torch.nn.Module):

    def __init__(self, config, vocab_size, tgt_mask):
        super().__init__()
        self.tgt_mask   = tgt_mask
        self.embedding  = get_embedding(config, vocab_size)
        self.positional = get_positional_encoding(config)
        self.decoder    = DecoderOnly(config)
        
    # src: [batch, src_seq, vocab_size]
    # tgt: [batch, tgt_seq, vocab_size]
    # ret: [batch, tgt_seq, vocab_size]
    def forward(self, tgt):
        tgt_embed  = self.embedding(tgt)
        tgt_embed += self.positional(tgt)
        tgt_output = self.decoder(tgt_embed)
        tgt_logits = self.embedding(tgt_output, reverse=True)
        tgt_probs  = torch.nn.functional.log_softmax(tgt_logits[:,:,self.tgt_mask], dim=-1)
        return tgt_probs
