# TODO: adjust the finished mask strategy in light of the need to update the cache

import torch

# All functions assume the data is in [batch*beam, ...] format.
# They can switch it into [batch, beam, ...] format as necessary.
class BeamCache:

    def __init__(self, batch_size, beam_size):
        self.batch_size = batch_size
        self.beam_size = beam_size

        self.src_mask = None
        self.k = dict()
        self.v = dict()

        self.finished_mask = None

    # Store the source mask.
    # src_mask: [batch_size * beam_size, 1, src_seq]
    def cache_src_mask(self, src_mask):
        self.src_mask = src_mask

    # Retrieve the source mask.
    # src_mask: [(batch_size * beam_size) - |finished|, 1, src_seq]
    def get_src_mask(self):
        mask = self.src_mask
        if self.finished_mask is not None:
            mask = mask[~self.finished_mask]
        return mask

    # Store the k and v projections, for either src or tgt.
    # k, v: [batch_size * beam_size, seq, d_attention]
    def cache_k(self, layer_id, k):
        self.k[layer_id] = k
    def cache_v(self, layer_id, v):
        self.v[layer_id] = v

    # Retrieve the k and v projections, for either src or tgt.
    # k, v: [(batch_size * beam_size) - |finished|, seq, d_attention]
    def get_k(self, layer_id):
        k = self.k[layer_id]
        if self.finished_mask is not None:
            k = k[~self.finished_mask]
        return k
    def get_v(self, layer_id):
        v = self.v[layer_id]
        if self.finished_mask is not None:
            v = v[~self.finished_mask]
        return v

    # Reshape the cache to accommodate beam_size beams or samples per sentence.
    def expand_to_beam_size(self, beam_size):
        if self.src_mask is not None:
            self.src_mask = self.src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, 1, -1)
        for layer in self.k:
            self.k[layer] = self.k[layer].unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, -1, self.k[layer].size(-1))
        for layer in self.v:
            self.v[layer] = self.v[layer].unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, -1, self.k[layer].size(-1))
        self.beam_size = beam_size

    # Trim finished sentences from the cache.
    # This is for sentences where the whole beam is finished,
    # so it can be removed entirely.
    # finished_mask: [batch_size]
    def trim_finished_sents(self, finished_mask):
        new_batch_size = (~finished_mask).sum()
        if self.src_mask is not None:
            src_mask = self.src_mask
            src_len = src_mask.size(-1)
            src_mask = src_mask.reshape(self.batch_size, self.beam_size, 1, src_len)[~finished_mask]
            self.src_mask = src_mask.reshape(new_batch_size * self.beam_size, 1, src_len)
        for layer in self.k:
            k = self.k[layer]
            k_len = k.size(-2)
            d_att = k.size(-1)
            k = k.reshape(self.batch_size, self.beam_size, k_len, d_att)[~finished_mask]
            k = k.reshape(new_batch_size * self.beam_size, k_len, d_att)
            self.k[layer] = k
        for layer in self.v:
            v = self.v[layer]
            v_len = v.size(-2)
            d_att = v.size(-1)
            v = v.reshape(self.batch_size, self.beam_size, v_len, d_att)[~finished_mask]
            v = v.reshape(new_batch_size * self.beam_size, v_len, d_att)
            self.v[layer] = v
        self.batch_size = new_batch_size

    # This is for sentences where only some items in the beam
    # are finished, so they should be filtered out before
    # passing through the transformer.
    # finished_mask: [batch_size * beam_size]
    def register_finished_mask(self, finished_mask):
        self.finished_mask = finished_mask
