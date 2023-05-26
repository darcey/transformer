import torch
import unittest
from cache import *



class TestBeamCache(unittest.TestCase):

    def testExpandToNumSamples(self):
        ni = float("-inf")
        src_embed = torch.tensor([[[0.1, 0.2, 0.3],
                                   [0.4, 0.5, 0.6]],
                                  [[0.7, 0.8, 0.9],
                                   [1.0, 1.1, 1.2]]])
        src_mask = torch.tensor([[[0, ni]],
                                 [[0, 0]]])

        src_embed_correct = torch.tensor([[[0.1, 0.2, 0.3],
                                           [0.4, 0.5, 0.6]],
                                          [[0.1, 0.2, 0.3],
                                           [0.4, 0.5, 0.6]],
                                          [[0.1, 0.2, 0.3],
                                           [0.4, 0.5, 0.6]],
                                          [[0.7, 0.8, 0.9],
                                           [1.0, 1.1, 1.2]],
                                          [[0.7, 0.8, 0.9],
                                           [1.0, 1.1, 1.2]],
                                          [[0.7, 0.8, 0.9],
                                           [1.0, 1.1, 1.2]]])
        src_mask_correct = torch.tensor([[[0, ni]],
                                         [[0, ni]],
                                         [[0, ni]],
                                         [[0, 0]],
                                         [[0, 0]],
                                         [[0, 0]]])

        cache = BeamCache()
        cache.cache_src(src_embed, src_mask)
        cache.expand_to_num_samples(3)
        src_embed_out, src_mask_out = cache.get_src()

        self.assertTrue(torch.equal(src_embed_out, src_embed_correct))
        self.assertTrue(torch.equal(src_mask_out, src_mask_correct))

    def testTrimFinishedSents(self):
        ni = float("-inf")
        src_embed = torch.tensor([[[0.1, 0.2, 0.3],
                                   [0.4, 0.5, 0.6]],
                                  [[0.2, 0.2, 0.3],
                                   [0.4, 0.5, 0.6]],
                                  [[0.3, 0.2, 0.3],
                                   [0.4, 0.5, 0.6]],
                                  [[0.1, 0.8, 0.9],
                                   [1.0, 1.1, 1.2]],
                                  [[0.2, 0.8, 0.9],
                                   [1.0, 1.1, 1.2]],
                                  [[0.3, 0.8, 0.9],
                                   [1.0, 1.1, 1.2]]])
        src_mask = torch.tensor([[[0, ni]],
                                 [[0, ni]],
                                 [[0, ni]],
                                 [[0, 0]],
                                 [[0, 0]],
                                 [[0, 0]]])
        finished_mask = torch.tensor([False, False, True, True, False, True])
        src_embed_correct = torch.tensor([[[0.1, 0.2, 0.3],
                                           [0.4, 0.5, 0.6]],
                                          [[0.2, 0.2, 0.3],
                                           [0.4, 0.5, 0.6]],
                                          [[0.2, 0.8, 0.9],
                                           [1.0, 1.1, 1.2]]])
        src_mask_correct = torch.tensor([[[0, ni]],
                                         [[0, ni]],
                                         [[0, 0]]])

        cache = BeamCache()
        cache.cache_src(src_embed, src_mask)
        cache.trim_finished_sents(finished_mask)
        src_embed_out, src_mask_out = cache.get_src()

        self.assertTrue(torch.equal(src_embed_out, src_embed_correct))
        self.assertTrue(torch.equal(src_mask_out, src_mask_correct))
