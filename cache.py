import torch

# All functions assume the data is in [batch*beam, ...] format.
# They can switch it into [batch, beam, ...] format as necessary.
class BeamCache:

    def __init__(self, batch_size, beam_size, src_len):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.src_len = src_len
        self.tgt_len = 0

        self.finished_mask = None
        self.src_mask = None
        self.src_k = dict()
        self.src_v = dict()

    # Store the source mask.
    # src_mask: [batch_size * beam_size, 1, src_seq]
    def cache_src_mask(self, src_mask):
        self.src_mask   = src_mask

    # Retrieve the source mask.
    # src_mask: [(batch_size * beam_size) - |finished|, 1, src_seq]
    def get_src_mask(self):
        mask = self.src_mask
        if self.finished_mask is not None:
            mask = mask[~self.finished_mask]
        return mask

    # Store the k and v projections of the source embedding.
    # k, v: [batch_size * beam_size, src_seq, d_attention]
    def cache_src_k(self, layer_id, k):
        self.src_k[layer_id] = k
    def cache_src_v(self, layer_id, v):
        self.src_v[layer_id] = v

    # Retrieve the k and v projections.
    # k, v: [(batch_size * beam_size) - |finished|, src_seq, d_attention]
    def get_k(self, layer_id):
        k = self.src_k[layer_id]
        if self.finished_mask is not None:
            k = k[~self.finished_mask]
        return k
    def get_v(self, layer_id):
        v = self.src_v[layer_id]
        if self.finished_mask is not None:
            v = v[~self.finished_mask]
        return v

    # Reshape the cache to accommodate beam_size beams or samples per sentence.
    def expand_to_beam_size(self, beam_size):
        self.src_mask = self.src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, 1, self.src_len)
        for layer in self.src_k:
            self.src_k[layer] = self.src_k[layer].unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, self.src_len, -1)
        for layer in self.src_v:
            self.src_v[layer] = self.src_v[layer].unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, self.src_len, -1)
        self.beam_size = beam_size

    # Trim finished sentences from the cache.
    # This is for sentences where the whole beam is finished,
    # so it can be removed entirely.
    # finished_mask: [batch_size]
    def trim_finished_sents(self, finished_mask):
        src_mask = self.src_mask.reshape(self.batch_size, self.beam_size, 1, self.src_len)[~finished_mask]
        self.src_mask = src_mask.reshape(-1, 1, self.src_len)
        for layer in self.src_k:
            src_k = self.src_k[layer]
            d_att = src_k.size(-1)
            src_k = src_k.reshape(self.batch_size, self.beam_size, self.src_len, d_att)[~finished_mask]
            src_k = src_k.reshape(-1, self.src_len, d_att)
            self.src_k[layer] = src_k
        for layer in self.src_v:
            src_v = self.src_v[layer]
            d_att = src_v.size(-1)
            src_v = src_v.reshape(self.batch_size, self.beam_size, self.src_len, d_att)[~finished_mask]
            src_v = src_v.reshape(-1, self.src_len, d_att)
            self.src_v[layer] = src_v
        self.batch_size = (~finished_mask).sum()

    # This is for sentences where only some items in the beam
    # are finished, so they should be filtered out before
    # passing through the transformer.
    # finished_mask: [batch_size * beam_size]
    def register_finished_mask(self, finished_mask):
        self.finished_mask = finished_mask
