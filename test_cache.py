import torch
import unittest
from cache import *

class TestBeamCache(unittest.TestCase):

    def setUp(self):
        self.ni = float("-inf")
        self.src_mask_beam_1 = torch.tensor([[[0, self.ni]],
                                             [[0, 0]]])
        self.src_k1_beam_1 = torch.tensor([[[0.1, 0.2, 0.3],
                                            [0.4, 0.5, 0.6]],
                                           [[0.7, 0.8, 0.9],
                                            [1.0, 1.1, 1.2]]])
        self.src_k2_beam_1 = torch.tensor([[[1.1, 1.2, 1.3],
                                            [1.4, 1.5, 1.6]],
                                           [[1.7, 1.8, 1.9],
                                            [2.0, 2.1, 2.2]]])
        self.src_v1_beam_1 = torch.tensor([[[2.1, 2.2, 2.3],
                                            [2.4, 2.5, 2.6]],
                                           [[2.7, 2.8, 2.9],
                                            [3.0, 3.1, 3.2]]])
        self.src_v2_beam_1 = torch.tensor([[[3.1, 3.2, 3.3],
                                            [3.4, 3.5, 3.6]],
                                           [[3.7, 3.8, 3.9],
                                            [4.0, 4.1, 4.2]]])

        self.src_mask_beam_3 = torch.tensor([[[0, self.ni]],
                                             [[0, self.ni]],
                                             [[0, self.ni]],
                                             [[0, 0]],
                                             [[0, 0]],
                                             [[0, 0]]])
        self.src_k1_beam_3 = torch.tensor([[[0.1, 0.2, 0.3],
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
        self.src_k2_beam_3 = torch.tensor([[[1.1, 1.2, 1.3],
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
        self.src_v1_beam_3 = torch.tensor([[[2.1, 2.2, 2.3],
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
        self.src_v2_beam_3 = torch.tensor([[[3.1, 3.2, 3.3],
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

        self.src_mask_beam_3_trim = torch.tensor([[[0, self.ni]],
                                                  [[0, 0]],
                                                  [[0, 0]]])
        self.src_k1_beam_3_trim = torch.tensor([[[0.1, 0.2, 0.3],
                                                 [0.4, 0.5, 0.6]],
                                                [[0.7, 0.8, 0.9],
                                                 [1.0, 1.1, 1.2]],
                                                [[0.7, 0.8, 0.9],
                                                 [1.0, 1.1, 1.2]]])
        self.src_k2_beam_3_trim = torch.tensor([[[1.1, 1.2, 1.3],
                                                 [1.4, 1.5, 1.6]],
                                                [[1.7, 1.8, 1.9],
                                                 [2.0, 2.1, 2.2]],
                                                [[1.7, 1.8, 1.9],
                                                 [2.0, 2.1, 2.2]]])
        self.src_v1_beam_3_trim = torch.tensor([[[2.1, 2.2, 2.3],
                                                 [2.4, 2.5, 2.6]],
                                                [[2.7, 2.8, 2.9],
                                                 [3.0, 3.1, 3.2]],
                                                [[2.7, 2.8, 2.9],
                                                 [3.0, 3.1, 3.2]]])
        self.src_v2_beam_3_trim = torch.tensor([[[3.1, 3.2, 3.3],
                                                 [3.4, 3.5, 3.6]],
                                                [[3.7, 3.8, 3.9],
                                                 [4.0, 4.1, 4.2]],
                                                [[3.7, 3.8, 3.9],
                                                 [4.0, 4.1, 4.2]]])

        self.src_mask_beam_3_unique = torch.tensor([[[0, self.ni]],
                                                    [[0, self.ni]],
                                                    [[self.ni, self.ni]],
                                                    [[self.ni, 0]],
                                                    [[self.ni, 0]],
                                                    [[0, 0]]])
        self.src_k1_beam_3_unique = torch.arange(36).reshape(6,2,3)
        self.src_k2_beam_3_unique = torch.arange(36,72).reshape(6,2,3)
        self.src_v1_beam_3_unique = torch.arange(72,108).reshape(6,2,3)
        self.src_v2_beam_3_unique = torch.arange(108,144).reshape(6,2,3)

    def setUpCacheBeam1(self):
        cache = BeamCache(batch_size=2,
                          beam_size=1,
                          num_layers=2,
                          device="cpu")
        cache.cache_src_mask(self.src_mask_beam_1)
        cache.cache_src_k(34, 0, self.src_k1_beam_1)
        cache.cache_src_k(56, 1, self.src_k2_beam_1)
        cache.cache_src_v(34, 0, self.src_v1_beam_1)
        cache.cache_src_v(56, 1, self.src_v2_beam_1)
        return cache        

    def setUpCacheBeam3(self):
        cache = BeamCache(batch_size=2,
                          beam_size=3,
                          num_layers=2,
                          device="cpu")
        cache.cache_src_mask(self.src_mask_beam_3)
        cache.cache_src_k(34, 0, self.src_k1_beam_3)
        cache.cache_src_k(56, 1, self.src_k2_beam_3)
        cache.cache_src_v(34, 0, self.src_v1_beam_3)
        cache.cache_src_v(56, 1, self.src_v2_beam_3)
        return cache

    def setUpCacheBeam3Unique(self):
        cache = BeamCache(batch_size=2,
                          beam_size=3,
                          num_layers=2,
                          device="cpu")
        cache.cache_src_mask(self.src_mask_beam_3_unique)
        cache.cache_src_k(34, 0, self.src_k1_beam_3_unique)
        cache.cache_src_k(56, 1, self.src_k2_beam_3_unique)
        cache.cache_src_v(34, 0, self.src_v1_beam_3_unique)
        cache.cache_src_v(56, 1, self.src_v2_beam_3_unique)
        return cache

    def evalCacheBeam1(self, cache):
        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, self.src_mask_beam_1))
        self.assertTrue(torch.equal(src_k1_out, self.src_k1_beam_1))
        self.assertTrue(torch.equal(src_k2_out, self.src_k2_beam_1))
        self.assertTrue(torch.equal(src_v1_out, self.src_v1_beam_1))
        self.assertTrue(torch.equal(src_v2_out, self.src_v2_beam_1))

    def evalCacheBeam3(self, cache):
        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, self.src_mask_beam_3))
        self.assertTrue(torch.equal(src_k1_out, self.src_k1_beam_3))
        self.assertTrue(torch.equal(src_k2_out, self.src_k2_beam_3))
        self.assertTrue(torch.equal(src_v1_out, self.src_v1_beam_3))
        self.assertTrue(torch.equal(src_v2_out, self.src_v2_beam_3))

    def evalCacheBeam3Trim(self, cache):
        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, self.src_mask_beam_3_trim))
        self.assertTrue(torch.equal(src_k1_out, self.src_k1_beam_3_trim))
        self.assertTrue(torch.equal(src_k2_out, self.src_k2_beam_3_trim))
        self.assertTrue(torch.equal(src_v1_out, self.src_v1_beam_3_trim))
        self.assertTrue(torch.equal(src_v2_out, self.src_v2_beam_3_trim))

    def testGetSrc(self):
        cache = self.setUpCacheBeam1()
        self.evalCacheBeam1(cache)

    def testExpandToBeamSize(self):
        cache = self.setUpCacheBeam1()
        cache.expand_to_beam_size(3)
        self.evalCacheBeam3(cache)

    def testRegisterFinishedSentsOneIterAllFalseBeam1(self):
        cache = self.setUpCacheBeam1()
        mask = torch.tensor([False,False])
        cache.register_finished_sents(mask)
        self.evalCacheBeam1(cache)
        self.assertTrue(torch.equal(cache.finished_mask, mask))

    def testRegisterFinishedSentsOneIterAllFalseBeamLarger(self):
        cache = self.setUpCacheBeam3()
        mask = torch.tensor([False,False,False,False,False,False])
        cache.register_finished_sents(mask)
        self.evalCacheBeam3(cache)
        self.assertTrue(torch.equal(cache.finished_mask, mask))

    def testRegisterFinishedSentsOneIterAllFalseBeamExpanded(self):
        cache = self.setUpCacheBeam1()
        cache.expand_to_beam_size(3)
        mask = torch.tensor([False,False,False,False,False,False])
        cache.register_finished_sents(mask)
        self.evalCacheBeam3(cache)
        self.assertTrue(torch.equal(cache.finished_mask, mask))

    def testRegisterFinishedSentsOneIterSomeTrueBeamLarger(self):
        cache = self.setUpCacheBeam3()
        mask = torch.tensor([False,True,True,True,False,False])
        cache.register_finished_sents(mask)
        self.evalCacheBeam3Trim(cache)
        self.assertTrue(torch.equal(cache.finished_mask, mask))

    def testRegisterFinishedSentsOneIterSomeTrueBeamExpanded(self):
        cache = self.setUpCacheBeam1()
        cache.expand_to_beam_size(3)
        mask = torch.tensor([False,True,True,True,False,False])
        cache.register_finished_sents(mask)
        self.evalCacheBeam3Trim(cache)
        self.assertTrue(torch.equal(cache.finished_mask, mask))

    def testRegisterFinishedSentsMultiIter(self):
        cache = self.setUpCacheBeam3()
        mask = torch.tensor([False,True,False,True,False,False])
        cache.register_finished_sents(mask)
        mask = torch.tensor([False,True,True,True,False,False])
        cache.register_finished_sents(mask)
        self.evalCacheBeam3Trim(cache)
        self.assertTrue(torch.equal(cache.finished_mask, mask))

    def testRegisterFinishedBeams(self):
        cache = self.setUpCacheBeam3()
        mask = torch.tensor([False,True])
        cache.register_finished_beams(mask)

        src_mask_beam_3_trim = torch.tensor([[[0, self.ni]],
                                             [[0, self.ni]],
                                             [[0, self.ni]]])
        src_k1_beam_3_trim = torch.tensor([[[0.1, 0.2, 0.3],
                                            [0.4, 0.5, 0.6]],
                                           [[0.1, 0.2, 0.3],
                                            [0.4, 0.5, 0.6]],
                                           [[0.1, 0.2, 0.3],
                                            [0.4, 0.5, 0.6]]])
        src_k2_beam_3_trim = torch.tensor([[[1.1, 1.2, 1.3],
                                            [1.4, 1.5, 1.6]],
                                           [[1.1, 1.2, 1.3],
                                            [1.4, 1.5, 1.6]],
                                           [[1.1, 1.2, 1.3],
                                            [1.4, 1.5, 1.6]]])
        src_v1_beam_3_trim = torch.tensor([[[2.1, 2.2, 2.3],
                                            [2.4, 2.5, 2.6]],
                                           [[2.1, 2.2, 2.3],
                                            [2.4, 2.5, 2.6]],
                                           [[2.1, 2.2, 2.3],
                                            [2.4, 2.5, 2.6]]])
        src_v2_beam_3_trim = torch.tensor([[[3.1, 3.2, 3.3],
                                            [3.4, 3.5, 3.6]],
                                           [[3.1, 3.2, 3.3],
                                            [3.4, 3.5, 3.6]],
                                           [[3.1, 3.2, 3.3],
                                            [3.4, 3.5, 3.6]]])

        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, src_mask_beam_3_trim))
        self.assertTrue(torch.equal(src_k1_out, src_k1_beam_3_trim))
        self.assertTrue(torch.equal(src_k2_out, src_k2_beam_3_trim))
        self.assertTrue(torch.equal(src_v1_out, src_v1_beam_3_trim))
        self.assertTrue(torch.equal(src_v2_out, src_v2_beam_3_trim))
        self.assertTrue(torch.equal(cache.finished_mask, torch.tensor([False,False,False])))

    def testRegisterFinishedSentsAndBeams(self):
        cache = self.setUpCacheBeam3()
        mask = torch.tensor([False,True,False,True,False,False])
        cache.register_finished_sents(mask)
        mask = torch.tensor([False,True])
        cache.register_finished_beams(mask)

        src_mask_beam_3_trim = torch.tensor([[[0, self.ni]],
                                             [[0, self.ni]]])
        src_k1_beam_3_trim = torch.tensor([[[0.1, 0.2, 0.3],
                                            [0.4, 0.5, 0.6]],
                                           [[0.1, 0.2, 0.3],
                                            [0.4, 0.5, 0.6]]])
        src_k2_beam_3_trim = torch.tensor([[[1.1, 1.2, 1.3],
                                            [1.4, 1.5, 1.6]],
                                           [[1.1, 1.2, 1.3],
                                            [1.4, 1.5, 1.6]]])
        src_v1_beam_3_trim = torch.tensor([[[2.1, 2.2, 2.3],
                                            [2.4, 2.5, 2.6]],
                                           [[2.1, 2.2, 2.3],
                                            [2.4, 2.5, 2.6]]])
        src_v2_beam_3_trim = torch.tensor([[[3.1, 3.2, 3.3],
                                            [3.4, 3.5, 3.6]],
                                           [[3.1, 3.2, 3.3],
                                            [3.4, 3.5, 3.6]]])

        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, src_mask_beam_3_trim))
        self.assertTrue(torch.equal(src_k1_out, src_k1_beam_3_trim))
        self.assertTrue(torch.equal(src_k2_out, src_k2_beam_3_trim))
        self.assertTrue(torch.equal(src_v1_out, src_v1_beam_3_trim))
        self.assertTrue(torch.equal(src_v2_out, src_v2_beam_3_trim))
        self.assertTrue(torch.equal(cache.finished_mask, torch.tensor([False,True,False])))

    def testSelectIdxsNoFinished(self):
        cache = self.setUpCacheBeam3Unique()
        chosen_idxs = torch.tensor([[0,1,1],[2,1,2]])
        cache.select_idxs(chosen_idxs)

        src_mask_beam_3_rearrange = torch.tensor([[[0, self.ni]],
                                                  [[0, self.ni]],
                                                  [[0, self.ni]],
                                                  [[0, 0]],
                                                  [[self.ni, 0]],
                                                  [[0, 0]]])
        src_k1_beam_3_rearrange = torch.tensor([[[0,1,2], [3,4,5]],
                                                [[6,7,8], [9,10,11]],
                                                [[6,7,8], [9,10,11]],
                                                [[30,31,32], [33,34,35]],
                                                [[24,25,26], [27,28,29]],
                                                [[30,31,32], [33,34,35]]])
        src_k2_beam_3_rearrange = torch.tensor([[[36,37,38], [39,40,41]],
                                                [[42,43,44], [45,46,47]],
                                                [[42,43,44], [45,46,47]],
                                                [[66,67,68], [69,70,71]],
                                                [[60,61,62], [63,64,65]],
                                                [[66,67,68], [69,70,71]]])
        src_v1_beam_3_rearrange = torch.tensor([[[72,73,74], [75,76,77]],
                                                [[78,79,80], [81,82,83]],
                                                [[78,79,80], [81,82,83]],
                                                [[102,103,104], [105,106,107]],
                                                [[96,97,98], [99,100,101]],
                                                [[102,103,104], [105,106,107]]])
        src_v2_beam_3_rearrange = torch.tensor([[[108,109,110], [111,112,113]],
                                                [[114,115,116], [117,118,119]],
                                                [[114,115,116], [117,118,119]],
                                                [[138,139,140], [141,142,143]],
                                                [[132,133,134], [135,136,137]],
                                                [[138,139,140], [141,142,143]]])

        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, src_mask_beam_3_rearrange))
        self.assertTrue(torch.equal(src_k1_out, src_k1_beam_3_rearrange))
        self.assertTrue(torch.equal(src_k2_out, src_k2_beam_3_rearrange))
        self.assertTrue(torch.equal(src_v1_out, src_v1_beam_3_rearrange))
        self.assertTrue(torch.equal(src_v2_out, src_v2_beam_3_rearrange))
        self.assertTrue(torch.equal(cache.finished_mask, torch.tensor([False,False,False,False,False,False])))

    def testSelectIdxsSomeFinished(self):
        cache = self.setUpCacheBeam3Unique()
        mask = torch.tensor([False,True,False,True,True,False])
        cache.register_finished_sents(mask)
        chosen_idxs = torch.tensor([[0,1,1],[2,1,2]])
        cache.select_idxs(chosen_idxs)

        src_mask_beam_3_rearrange = torch.tensor([[[0, self.ni]],
                                                  [[0, 0]],
                                                  [[0, 0]]])
        src_k1_beam_3_rearrange = torch.tensor([[[0,1,2], [3,4,5]],
                                                [[30,31,32], [33,34,35]],
                                                [[30,31,32], [33,34,35]]])
        src_k2_beam_3_rearrange = torch.tensor([[[36,37,38], [39,40,41]],
                                                [[66,67,68], [69,70,71]],
                                                [[66,67,68], [69,70,71]]])
        src_v1_beam_3_rearrange = torch.tensor([[[72,73,74], [75,76,77]],
                                                [[102,103,104], [105,106,107]],
                                                [[102,103,104], [105,106,107]]])
        src_v2_beam_3_rearrange = torch.tensor([[[108,109,110], [111,112,113]],
                                                [[138,139,140], [141,142,143]],
                                                [[138,139,140], [141,142,143]]])

        src_mask_out = cache.get_src_mask()
        src_k1_out = cache.get_k(34)
        src_k2_out = cache.get_k(56)
        src_v1_out = cache.get_v(34)
        src_v2_out = cache.get_v(56)
        self.assertTrue(torch.equal(src_mask_out, src_mask_beam_3_rearrange))
        self.assertTrue(torch.equal(src_k1_out, src_k1_beam_3_rearrange))
        self.assertTrue(torch.equal(src_k2_out, src_k2_beam_3_rearrange))
        self.assertTrue(torch.equal(src_v1_out, src_v1_beam_3_rearrange))
        self.assertTrue(torch.equal(src_v2_out, src_v2_beam_3_rearrange))
        self.assertTrue(torch.equal(cache.finished_mask, torch.tensor([False,True,True,False,True,False])))
