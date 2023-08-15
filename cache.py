import torch

# All functions assume the data is in [batch*beam, ...] format.
# They can switch it into [batch, beam, ...] format as necessary.
class BeamCache:

    def __init__(self, batch_size, beam_size):
        self.batch_size = batch_size
        self.beam_size = beam_size

        self.finished_mask = None
        self.src_mask = None
        self.tgt_mask = None
        self.k = dict()
        self.v = dict()

    # Store the masks.
    # src_mask: [batch_size * beam_size, 1, src_seq]
    # tgt_mask: [batch_size * beam_size, 1, tgt_seq]
    def cache_src_mask(self, src_mask):
        self.src_mask = src_mask
    def cache_tgt_mask(self, tgt_mask):
        self.tgt_mask = tgt_mask

    # Retrieve the masks.
    # src_mask: [(batch_size * beam_size) - |finished|, 1, src_seq]
    # tgt_mask: [(batch_size * beam_size) - |finished|, 1, tgt_seq]
    def get_src_mask(self):
        mask = self.src_mask
        if self.finished_mask is not None:
            mask = mask[~self.finished_mask]
        return mask
    def get_tgt_mask(self):
        mask = self.tgt_mask
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
        self.tgt_mask = self.tgt_mask.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, 1, -1)
        for layer in self.k:
            self.k[layer] = self.k[layer].unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, -1, self.k[layer].size(-1))
        for layer in self.v:
            self.v[layer] = self.v[layer].unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, -1, self.k[layer].size(-1))
        self.beam_size = beam_size

    # Trim finished sentences from the cache.
    # This is for sentences where the whole beam is finished,
    # meaning that it (the whole beam) can be removed entirely.
    # finished_mask: [batch_size]
    def trim_finished_sents(self, finished_mask):
        new_batch_size = (~finished_mask).sum()
        if self.src_mask is not None:
            src_mask = self.src_mask.reshape(self.batch_size, self.beam_size, 1, -1)[~finished_mask]
            self.src_mask = src_mask.reshape(new_batch_size * self.beam_size, 1, -1)
        tgt_mask = self.tgt_mask.reshape(self.batch_size, self.beam_size, 1, -1)[~finished_mask]
        self.tgt_mask = tgt_mask.reshape(new_batch_size * self.beam_size, 1, -1)
        for layer in self.k:
            k = self.k[layer]
            d_att = k.size(-1)
            k = k.reshape(self.batch_size, self.beam_size, -1, d_att)[~finished_mask]
            k = k.reshape(new_batch_size * self.beam_size, -1, d_att)
            self.k[layer] = k
        for layer in self.v:
            v = self.v[layer]
            d_att = v.size(-1)
            v = v.reshape(self.batch_size, self.beam_size, -1, d_att)[~finished_mask]
            v = v.reshape(new_batch_size * self.beam_size, -1, d_att)
            self.v[layer] = v
        self.batch_size = new_batch_size

    # This is for sentences where only some items in the beam
    # are finished, so they should be filtered out before
    # passing through the transformer, but the beam should
    # stay in the cache so the generation algorithm can use it.
    # finished_mask: [batch_size * beam_size]
    def register_finished_mask(self, finished_mask):
        self.finished_mask = finished_mask
