# TODO(darcey): should there be a separate SampleCache vs. BeamCache?

import torch

class BeamCache:

    def __init__(self):
        self.batch_size = 1

    # Store the source embeddings and source mask
    # src_embed: [batch_size, src_seq, d_model]
    # src_mask:  [batch_size, 1, src_seq]
    def cache_src(self, src_embed, src_mask):
        self.batch_size = src_embed.size(0)
        self.src_len    = src_embed.size(1)
        self.src_embed  = src_embed
        self.src_mask   = src_mask

    # Retrieve the source embeddings and source mask
    # src_embed: [batch_size * beam_size, src_seq, d_model]
    # src_mask:  [batch_size * beam_size, 1, src_seq]
    def get_src(self):
        return self.src_embed, self.src_mask

    # Reshape the cache to accommodate beam_size beams or samples per sentence.
    # num_samples: int
    def expand_to_num_samples(self, num_samples):
        self.src_embed = self.src_embed.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(self.batch_size * num_samples, self.src_len, -1)
        self.src_mask = self.src_mask.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(self.batch_size * num_samples, 1, self.src_len)
        self.batch_size *= num_samples

    # Trim finished sentences from the cache.
    # finished_mask: [batch_size]
    def trim_finished_sents(self, finished_mask):
        self.src_embed  = self.src_embed[~finished_mask]
        self.src_mask   = self.src_mask[~finished_mask]
        self.batch_size = self.src_embed.size(0)
