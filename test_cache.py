import torch
import unittest
from cache import *



class TestBeamCache(unittest.TestCase):

    def testGetCacheAndRetrieve(self):
        ni = float("-inf")
        src_mask = torch.tensor([[[0, ni]],
                                 [[0, 0]]])
        tgt_mask = torch.tensor([[[ni, 0]],
                                 [[0, ni]]])
        k1 = torch.tensor([[[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]],
                           [[0.7, 0.8, 0.9],
                            [1.0, 1.1, 1.2]]])
        k2 = torch.tensor([[[1.1, 1.2, 1.3],
                            [1.4, 1.5, 1.6]],
                           [[1.7, 1.8, 1.9],
                            [2.0, 2.1, 2.2]]])
        v1 = torch.tensor([[[2.1, 2.2, 2.3],
                            [2.4, 2.5, 2.6]],
                           [[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]]])
        v2 = torch.tensor([[[3.1, 3.2, 3.3],
                            [3.4, 3.5, 3.6]],
                           [[3.7, 3.8, 3.9],
                            [4.0, 4.1, 4.2]]])

        cache = BeamCache(2,1)
        cache.cache_src_mask(src_mask)
        cache.cache_tgt_mask(tgt_mask)
        cache.cache_k(1, k1)
        cache.cache_k(2, k2)
        cache.cache_v(1, v1)
        cache.cache_v(2, v2)

        # No finished mask
        src_mask_out = cache.get_src_mask()
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertTrue(torch.equal(src_mask_out, src_mask))
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask))
        self.assertTrue(torch.equal(k1_out, k1))
        self.assertTrue(torch.equal(k2_out, k2))
        self.assertTrue(torch.equal(v1_out, v1))
        self.assertTrue(torch.equal(v2_out, v2))

        # Finished mask set to False
        cache.register_finished_mask(torch.tensor([False,False]))
        src_mask_out = cache.get_src_mask()
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertTrue(torch.equal(src_mask_out, src_mask))
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask))
        self.assertTrue(torch.equal(k1_out, k1))
        self.assertTrue(torch.equal(k2_out, k2))
        self.assertTrue(torch.equal(v1_out, v1))
        self.assertTrue(torch.equal(v2_out, v2))

        # More interesting finished mask
        src_mask_correct = torch.tensor([[[0, ni]]])
        tgt_mask_correct = torch.tensor([[[ni, 0]]])
        k1_correct = torch.tensor([[[0.1, 0.2, 0.3],
                                    [0.4, 0.5, 0.6]]])
        k2_correct = torch.tensor([[[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]]])
        v1_correct = torch.tensor([[[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]]])
        v2_correct = torch.tensor([[[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]]])
        cache.register_finished_mask(torch.tensor([False,True]))
        src_mask_out = cache.get_src_mask()
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertTrue(torch.equal(src_mask_out, src_mask_correct))
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask_correct))
        self.assertTrue(torch.equal(k1_out, k1_correct))
        self.assertTrue(torch.equal(k2_out, k2_correct))
        self.assertTrue(torch.equal(v1_out, v1_correct))
        self.assertTrue(torch.equal(v2_out, v2_correct))

    def testExpandToBeamSize(self):
        ni = float("-inf")
        src_mask = torch.tensor([[[0, ni]],
                                 [[0, 0]]])
        tgt_mask = torch.tensor([[[ni, 0]],
                                 [[0, ni]]])
        k1 = torch.tensor([[[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]],
                           [[0.7, 0.8, 0.9],
                            [1.0, 1.1, 1.2]]])
        k2 = torch.tensor([[[1.1, 1.2, 1.3],
                            [1.4, 1.5, 1.6]],
                           [[1.7, 1.8, 1.9],
                            [2.0, 2.1, 2.2]]])
        v1 = torch.tensor([[[2.1, 2.2, 2.3],
                            [2.4, 2.5, 2.6]],
                           [[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]]])
        v2 = torch.tensor([[[3.1, 3.2, 3.3],
                            [3.4, 3.5, 3.6]],
                           [[3.7, 3.8, 3.9],
                            [4.0, 4.1, 4.2]]])

        src_mask_correct = torch.tensor([[[0, ni]],
                                         [[0, ni]],
                                         [[0, ni]],
                                         [[0, 0]],
                                         [[0, 0]],
                                         [[0, 0]]])
        tgt_mask_correct = torch.tensor([[[ni, 0]],
                                         [[ni, 0]],
                                         [[ni, 0]],
                                         [[0, ni]],
                                         [[0, ni]],
                                         [[0, ni]]])
        k1_correct = torch.tensor([[[0.1, 0.2, 0.3],
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
        k2_correct = torch.tensor([[[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]],
                                   [[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]],
                                   [[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]],
                                   [[1.7, 1.8, 1.9],
                                    [2.0, 2.1, 2.2]],
                                   [[1.7, 1.8, 1.9],
                                    [2.0, 2.1, 2.2]],
                                   [[1.7, 1.8, 1.9],
                                    [2.0, 2.1, 2.2]]])
        v1_correct = torch.tensor([[[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]],
                                   [[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]],
                                   [[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]],
                                   [[2.7, 2.8, 2.9],
                                    [3.0, 3.1, 3.2]],
                                   [[2.7, 2.8, 2.9],
                                    [3.0, 3.1, 3.2]],
                                   [[2.7, 2.8, 2.9],
                                    [3.0, 3.1, 3.2]]])
        v2_correct = torch.tensor([[[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]],
                                   [[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]],
                                   [[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]],
                                   [[3.7, 3.8, 3.9],
                                    [4.0, 4.1, 4.2]],
                                   [[3.7, 3.8, 3.9],
                                    [4.0, 4.1, 4.2]],
                                   [[3.7, 3.8, 3.9],
                                    [4.0, 4.1, 4.2]]])

        # With src mask
        cache = BeamCache(2,1)
        cache.cache_src_mask(src_mask)
        cache.cache_tgt_mask(tgt_mask)
        cache.cache_k(1, k1)
        cache.cache_k(2, k2)
        cache.cache_v(1, v1)
        cache.cache_v(2, v2)

        cache.expand_to_beam_size(3)
        src_mask_out = cache.get_src_mask()
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertTrue(torch.equal(src_mask_out, src_mask_correct))
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask_correct))
        self.assertTrue(torch.equal(k1_out, k1_correct))
        self.assertTrue(torch.equal(k2_out, k2_correct))
        self.assertTrue(torch.equal(v1_out, v1_correct))
        self.assertTrue(torch.equal(v2_out, v2_correct))

        # Without src mask
        cache = BeamCache(2,1)
        cache.cache_tgt_mask(tgt_mask)
        cache.cache_k(1, k1)
        cache.cache_k(2, k2)
        cache.cache_v(1, v1)
        cache.cache_v(2, v2)

        cache.expand_to_beam_size(3)
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask_correct))
        self.assertTrue(torch.equal(k1_out, k1_correct))
        self.assertTrue(torch.equal(k2_out, k2_correct))
        self.assertTrue(torch.equal(v1_out, v1_correct))
        self.assertTrue(torch.equal(v2_out, v2_correct))

    def testTrimFinishedSents(self):
        ni = float("-inf")
        src_mask = torch.tensor([[[0, ni]],
                                 [[0, ni]],
                                 [[0, ni]],
                                 [[0, 0]],
                                 [[0, 0]],
                                 [[0, 0]]])
        tgt_mask = torch.tensor([[[ni, 0]],
                                 [[ni, 0]],
                                 [[ni, 0]],
                                 [[0, ni]],
                                 [[0, ni]],
                                 [[0, ni]]])
        k1 = torch.tensor([[[0.1, 0.2, 0.3],
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
        k2 = torch.tensor([[[1.1, 1.2, 1.3],
                            [1.4, 1.5, 1.6]],
                           [[1.1, 1.2, 1.3],
                            [1.4, 1.5, 1.6]],
                           [[1.1, 1.2, 1.3],
                            [1.4, 1.5, 1.6]],
                           [[1.7, 1.8, 1.9],
                            [2.0, 2.1, 2.2]],
                           [[1.7, 1.8, 1.9],
                            [2.0, 2.1, 2.2]],
                           [[1.7, 1.8, 1.9],
                            [2.0, 2.1, 2.2]]])
        v1 = torch.tensor([[[2.1, 2.2, 2.3],
                            [2.4, 2.5, 2.6]],
                           [[2.1, 2.2, 2.3],
                            [2.4, 2.5, 2.6]],
                           [[2.1, 2.2, 2.3],
                            [2.4, 2.5, 2.6]],
                           [[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]],
                           [[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]],
                           [[2.7, 2.8, 2.9],
                            [3.0, 3.1, 3.2]]])
        v2 = torch.tensor([[[3.1, 3.2, 3.3],
                            [3.4, 3.5, 3.6]],
                           [[3.1, 3.2, 3.3],
                            [3.4, 3.5, 3.6]],
                           [[3.1, 3.2, 3.3],
                            [3.4, 3.5, 3.6]],
                           [[3.7, 3.8, 3.9],
                            [4.0, 4.1, 4.2]],
                           [[3.7, 3.8, 3.9],
                            [4.0, 4.1, 4.2]],
                           [[3.7, 3.8, 3.9],
                            [4.0, 4.1, 4.2]]])

        finished_mask = torch.tensor([False, True])
        src_mask_correct = torch.tensor([[[0, ni]],
                                         [[0, ni]],
                                         [[0, ni]]])
        tgt_mask_correct = torch.tensor([[[ni, 0]],
                                         [[ni, 0]],
                                         [[ni, 0]]])
        k1_correct = torch.tensor([[[0.1, 0.2, 0.3],
                                    [0.4, 0.5, 0.6]],
                                   [[0.1, 0.2, 0.3],
                                    [0.4, 0.5, 0.6]],
                                   [[0.1, 0.2, 0.3],
                                    [0.4, 0.5, 0.6]]])
        k2_correct = torch.tensor([[[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]],
                                   [[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]],
                                   [[1.1, 1.2, 1.3],
                                    [1.4, 1.5, 1.6]]])
        v1_correct = torch.tensor([[[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]],
                                   [[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]],
                                   [[2.1, 2.2, 2.3],
                                    [2.4, 2.5, 2.6]]])
        v2_correct = torch.tensor([[[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]],
                                   [[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]],
                                   [[3.1, 3.2, 3.3],
                                    [3.4, 3.5, 3.6]]])

        # With src mask
        cache = BeamCache(2,3)
        cache.cache_src_mask(src_mask)
        cache.cache_tgt_mask(tgt_mask)
        cache.cache_k(1, k1)
        cache.cache_k(2, k2)
        cache.cache_v(1, v1)
        cache.cache_v(2, v2)

        cache.trim_finished_sents(finished_mask)
        src_mask_out = cache.get_src_mask()
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertEqual(cache.batch_size, 1)
        self.assertTrue(torch.equal(src_mask_out, src_mask_correct))
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask_correct))
        self.assertTrue(torch.equal(k1_out, k1_correct))
        self.assertTrue(torch.equal(k2_out, k2_correct))
        self.assertTrue(torch.equal(v1_out, v1_correct))
        self.assertTrue(torch.equal(v2_out, v2_correct))

        # Without src mask
        cache = BeamCache(2,3)
        cache.cache_tgt_mask(tgt_mask)
        cache.cache_k(1, k1)
        cache.cache_k(2, k2)
        cache.cache_v(1, v1)
        cache.cache_v(2, v2)

        cache.trim_finished_sents(finished_mask)
        tgt_mask_out = cache.get_tgt_mask()
        k1_out = cache.get_k(1)
        k2_out = cache.get_k(2)
        v1_out = cache.get_v(1)
        v2_out = cache.get_v(2)
        self.assertEqual(cache.batch_size, 1)
        self.assertTrue(torch.equal(tgt_mask_out, tgt_mask_correct))
        self.assertTrue(torch.equal(k1_out, k1_correct))
        self.assertTrue(torch.equal(k2_out, k2_correct))
        self.assertTrue(torch.equal(v1_out, v1_correct))
        self.assertTrue(torch.equal(v2_out, v2_correct))
