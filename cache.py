import torch

class BeamCache:

    def __init__(self):
        self.batch_size = 1
        self.beam_size = 1
        self.src_len = 0
        self.tgt_len = 0

    # Store the source embeddings and source mask
    # src_embed: [batch_size, src_seq, d_model]
    # src_mask:  [batch_size, 1, src_seq]
    def cache_src(self, src_embed, src_mask):
        self.batch_size = src_embed.size(0)
        self.src_len    = src_embed.size(1)
        self.d_model    = src_embed.size(2)
        self.src_embed  = src_embed
        self.src_mask   = src_mask

    # Retrieve the source embeddings and source mask,
    # filtering out finished sentences in the process
    # finished_mask: [batch_size * beam_size]
    # src_embed:     [(batch_size * beam_size) - |finished|, src_seq, d_model]
    # src_mask:      [(batch_size * beam_size) - |finished|, 1, src_seq]
    def get_src(self, finished_mask):
        src_embed = self.src_embed[~finished_mask]
        src_mask  = self.src_mask[~finished_mask]
        return src_embed, src_mask

    # Reshape the cache to accommodate beam_size beams or samples per sentence.
    def expand_to_beam_size(self, beam_size):
        self.src_embed = self.src_embed.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, self.src_len, -1)
        self.src_mask = self.src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, 1, self.src_len)
        self.beam_size = beam_size

    # Trim finished sentences from the cache.
    # finished_mask: [batch_size]
    def trim_finished_sents(self, finished_mask):
        src_embed = self.src_embed.reshape(self.batch_size, self.beam_size, self.src_len, self.d_model)[~finished_mask]
        src_mask  = self.src_mask.reshape(self.batch_size, self.beam_size, 1, self.src_len)[~finished_mask]
        self.batch_size = src_embed.size(0)
        self.src_embed  = src_embed.reshape(-1, self.src_len, self.d_model)
        self.src_mask   = src_mask.reshape(-1, 1, self.src_len)
