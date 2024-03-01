import math
import torch
import unittest
from configuration import read_config, DecodingMethod, LengthNormalization
from distances import Identity
from cache import BeamCache
from generator import *



class MockModelDoesNothing:
    def get_autoregressive_one_step_fn(self, src, cache):
        return 5, 6

class MockCacheDoesNothing:
    def expand_to_beam_size(self, beam_size):
        return
    def register_finished_sents(self, mask):
        return
    def register_finished_beams(self, mask):
        return
    def select_idxs(self, chosen_idxs):
        return



class TestGenerate(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2

        device = "cpu"
        config = read_config("configuration.toml")
        model = MockModelDoesNothing()

        self.gen = Generator(model, config, device, 5, self.pad_idx, self.bos_idx, self.eos_idx)
        self.config = config.gen

    def testMaxLengthsRelative(self):
        self.config.use_rel_max_len = True
        self.config.rel_max_len = 5
        self.config.abs_max_len = 6
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([8,9,7])

        max_lengths, max_possible_length = self.gen.get_max_lengths(self.config, src)
        self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
        self.assertEqual(max_possible_length, 9)

    def testMaxLengthsAbsolute(self):
        self.config.use_rel_max_len = False
        self.config.rel_max_len = 5
        self.config.abs_max_len = 6
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([6,6,6])

        max_lengths, max_possible_length = self.gen.get_max_lengths(self.config, src)
        self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
        self.assertEqual(max_possible_length, 6)

    def testMaxLengthsContextWindow(self):
        self.config.use_rel_max_len = True
        self.config.rel_max_len = 7
        self.gen.window = 5
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([5,5,5])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            max_lengths, max_possible_length = self.gen.get_max_lengths(self.config, src)
            self.assertEqual(len(w), 1)
            self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
            self.assertEqual(max_possible_length, 5)



class TestSampling(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.a_idx   = 3
        self.b_idx   = 4

        device = "cpu"
        config = read_config("configuration.toml")
        model = MockModelDoesNothing()

        self.gen = Generator(model, config, device, 5, self.pad_idx, self.bos_idx, self.eos_idx)
        self.cache = MockCacheDoesNothing()
        self.config = config.gen

    def testSampleOuterLoopOneIter(self):
        iteration = 0
        def mock_sample_function(config, batch, sample, w, x, y, z):
            nonlocal iteration
            iteration += 1
            all_symbols = torch.full(size=(batch, sample, iteration*2), fill_value=iteration)
            all_probs = torch.full(size=(batch, sample), fill_value=float(iteration))
            return all_symbols, all_probs
        self.gen.sample = mock_sample_function
        self.config.max_parallel_sentences = 12
        self.config.num_beams_or_samples = 5

        samples_correct = torch.tensor([[[1,1],
                                         [1,1],
                                         [1,1],
                                         [1,1],
                                         [1,1]],
                                        [[1,1],
                                         [1,1],
                                         [1,1],
                                         [1,1],
                                         [1,1]]])
        samples_final_correct = torch.tensor([[1,1],[1,1]])
        probs_correct = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0, 1.0]])

        samples_final_out, samples_out, probs_out = self.gen.sample_outer_loop(self.config, 2, None, None, None, None)

        self.assertTrue(torch.equal(samples_out, samples_correct))
        self.assertTrue(torch.equal(samples_final_out, samples_final_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSampleOuterLoopManyIter(self):
        iteration = 0
        def mock_sample_function(config, batch, sample, w, x, y, z):
            nonlocal iteration
            iteration += 1
            all_symbols = torch.full(size=(batch, sample, iteration*2), fill_value=iteration)
            all_probs = torch.full(size=(batch, sample), fill_value=float(iteration))
            return all_symbols, all_probs
        self.gen.sample = mock_sample_function
        self.config.max_parallel_sentences = 5
        self.config.num_beams_or_samples = 13

        samples_correct = torch.tensor([[[1,1,0,0,0,0],
                                         [1,1,0,0,0,0],
                                         [1,1,0,0,0,0],
                                         [1,1,0,0,0,0],
                                         [1,1,0,0,0,0],
                                         [2,2,2,2,0,0],
                                         [2,2,2,2,0,0],
                                         [2,2,2,2,0,0],
                                         [2,2,2,2,0,0],
                                         [2,2,2,2,0,0],
                                         [3,3,3,3,3,3],
                                         [3,3,3,3,3,3],
                                         [3,3,3,3,3,3]]])
        samples_final_correct = torch.tensor([[1,1,0,0,0,0]])
        probs_correct = torch.tensor([[1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0]])

        samples_final_out, samples_out, probs_out = self.gen.sample_outer_loop(self.config, 1, torch.rand(5), None, None, None)

        self.assertTrue(torch.equal(samples_out, samples_correct))
        self.assertTrue(torch.equal(samples_final_out, samples_final_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSampleSimpleDistribution1(self):
        # At beginning, splits off into two possibilities: all as or all bs.
        def mock_autoregressive_fn(symbols, timestep, cache):
            lh       = math.log(0.5)
            ni       = float("-inf")
            a_dist   = torch.tensor([[[ni, ni, lh, lh, ni]]])
            b_dist   = torch.tensor([[[ni, ni, lh, ni, lh]]])
            ab_dist  = torch.tensor([[[ni, ni, ni, lh, lh]]])
            all_dist = torch.empty((symbols.size(0), 5))
            all_dist[(symbols[:,-1] == 1)] = ab_dist
            all_dist[(symbols[:,-1] == 3)] = a_dist
            all_dist[(symbols[:,-1] == 4)] = b_dist
            all_dist = all_dist.unsqueeze(1)
            return all_dist, torch.log_softmax(all_dist, dim=-1)

        max_lengths = torch.tensor([40]*1000)
        symbols_out, probs_out = self.gen.sample(self.config, 1000, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        # Every sample should have just a or just b
        a_samples = torch.eq(symbols_out, 3).sum(dim=-1).type(torch.bool)
        b_samples = torch.eq(symbols_out, 4).sum(dim=-1).type(torch.bool)
        self.assertFalse(torch.logical_and(a_samples, b_samples).any())

        # Should be roughly half a, half b
        self.assertAlmostEqual(a_samples.sum()/5000, 0.5, delta=0.02)
        self.assertAlmostEqual(b_samples.sum()/5000, 0.5, delta=0.02)

    def testSampleSimpleDistribution2(self):
        # Generates strings which alternate between as and bs
        # Always starts with an a, ends with a b
        def mock_autoregressive_fn(symbols, timestep, cache):
            lh       = math.log(0.5)
            ni       = float("-inf")
            a_dist   = torch.tensor([[[ni, ni, lh, lh, ni]]])
            b_dist   = torch.tensor([[[ni, ni, ni, ni, 0.0]]])
            all_dist = torch.empty((symbols.size(0), 5))
            all_dist[(symbols[:,-1] == 1)] = a_dist
            all_dist[(symbols[:,-1] == 3)] = b_dist
            all_dist[(symbols[:,-1] == 4)] = a_dist
            all_dist = all_dist.unsqueeze(1)
            return all_dist, torch.log_softmax(all_dist, dim=-1)

        max_lengths = torch.tensor([40]*1000)
        symbols_out, probs_out = self.gen.sample(self.config, 1000, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        # Every sample should have equal numbers of as and bs.
        num_as = torch.eq(symbols_out, 3).sum(dim=-1)
        num_bs = torch.eq(symbols_out, 4).sum(dim=-1)
        self.assertTrue(torch.equal(num_as, num_bs))

        # Should be roughly half length 2, half length 4, etc...
        self.assertAlmostEqual((num_as == 0).sum()/5000, 0.5, delta=0.02)
        self.assertAlmostEqual((num_as == 1).sum()/5000, 0.25, delta=0.02)

    def testTemperature(self):
        lh   = math.log(0.5)
        l1   = math.log(1.0)
        ni   = float("-inf")
        dist = torch.tensor([[[ni, ni, 2*lh, 2*lh, 2*l1]]])
        dist_correct = torch.tensor([[[0.0, 0.0, 0.25, 0.25, 0.5]]])
        self.config.sampling_temp = 2.0
        dist_out = self.gen.process_logits(self.config, dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))

    def testTopK(self):
        dist = torch.tensor([0.01, 0.03, 0.09, 0.07, 0.55, 0.05, 0.02, 0.04, 0.06, 0.08])
        dist = dist.unsqueeze(0).unsqueeze(0)

        self.config.decoding_method = DecodingMethod.SAMPLING
        self.config.sampling_k = 4
        dist_correct = torch.tensor([0.0, 0.0, 0.09, 0.07, 0.55, 0.0, 0.0, 0.0, 0.0, 0.08])
        dist_correct = dist_correct.unsqueeze(0).unsqueeze(0)

        dist_out = self.gen.truncate_probs(self.config, dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))

    def testTopKLargerThanVocab(self):
        dist = torch.softmax(torch.rand(4,5,10), dim=-1)

        self.config.decoding_method = DecodingMethod.SAMPLING
        self.config.sampling_k = 15

        dist_out = self.gen.truncate_probs(self.config, dist)
        self.assertTrue(torch.equal(dist_out, dist))

    def testTopP(self):
        self.config.decoding_method = DecodingMethod.SAMPLING

        # No matter what the distribution is, p = 1.0 should just return it.
        self.config.sampling_p = 1.0
        dist = torch.nn.functional.softmax(torch.rand(3,5,8), dim=-1)
        dist_out = self.gen.truncate_probs(self.config, dist)
        self.assertTrue(torch.equal(dist_out, dist))

        # Test for p < 1.0
        self.config.sampling_p = 0.6
        dist = torch.tensor([[[0.26, 0.1, 0.2,  0.03, 0.05, 0.02, 0.3,  0.04],
                              [0.08, 0.5, 0.02, 0.25, 0.05, 0.04, 0.03, 0.03]],
                             [[0.3,  0.0, 0.11, 0.02, 0.3,  0.03, 0.2,  0.04],
                              [0.1,  0.3, 0.2,  0.15, 0.05, 0.25, 0.02, 0.03]]])
        dist_correct = torch.tensor([[[0.26, 0.0, 0.2, 0.0,  0.0, 0.0,  0.3, 0.0],
                                      [0.0 , 0.5, 0.0, 0.25, 0.0, 0.0,  0.0, 0.0]],
                                     [[0.3,  0.0, 0.0, 0.0,  0.3, 0.0,  0.0, 0.0],
                                      [0.0,  0.3, 0.2, 0.0,  0.0, 0.25, 0.0, 0.0]]])
        dist_out = self.gen.truncate_probs(self.config, dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))

    def testTopPAndK(self):
        self.config.decoding_method = DecodingMethod.SAMPLING

        self.config.sampling_p = 0.6
        self.config.sampling_k = 3
        dist = torch.tensor([[[0.26, 0.1, 0.2,  0.03, 0.05, 0.02, 0.3,  0.04],
                              [0.08, 0.5, 0.02, 0.25, 0.05, 0.04, 0.03, 0.03]],
                             [[0.3,  0.0, 0.11, 0.02, 0.3,  0.03, 0.2,  0.04],
                              [0.02, 0.3, 0.15, 0.12, 0.11, 0.1,  0.1,  0.1]]])
        dist_correct = torch.tensor([[[0.26, 0.0, 0.2,  0.0,  0.0, 0.0, 0.3, 0.0],
                                      [0.0 , 0.5, 0.0,  0.25, 0.0, 0.0, 0.0, 0.0]],
                                     [[0.3,  0.0, 0.0,  0.0,  0.3, 0.0, 0.0, 0.0],
                                      [0.0,  0.3, 0.15, 0.12, 0.0, 0.0, 0.0, 0.0]]])
        dist_out = self.gen.truncate_probs(self.config, dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))



class TestBeamSearch(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.a_idx   = 3
        self.b_idx   = 4

        device = "cpu"
        config = read_config("configuration.toml")
        model = MockModelDoesNothing()

        self.gen = Generator(model, config, device, 5, self.pad_idx, self.bos_idx, self.eos_idx)
        self.cache = MockCacheDoesNothing()
        self.config = config.gen

    def testBeamSearch(self):
        # Generates strings which are either all as or all bs
        def mock_autoregressive_fn(cumul_symbols, timestep, cache):
            dist1    = torch.tensor([[[0.0, 0.0, 0.0,  0.5,  0.5]]])
            dist2    = torch.tensor([[[0.0, 0.0, 0.55, 0.45, 0.0]]])
            dist3    = torch.tensor([[[0.0, 0.0, 0.6,  0.0,  0.4]]])
            all_dist = (cumul_symbols == 1).unsqueeze(-1).type(torch.float) * dist1 + \
                       (cumul_symbols == 3).unsqueeze(-1).type(torch.float) * dist2 + \
                       (cumul_symbols == 4).unsqueeze(-1).type(torch.float) * dist3
            return torch.rand(all_dist.size()), torch.log(all_dist)

        max_lengths = torch.tensor([40]*2)
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(self.config, 2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        symbols_final_correct = torch.tensor([[1,4,2,0,0],
                                              [1,4,2,0,0]])
        symbols_all_correct = torch.tensor([[[1,4,2,0,0],
                                             [1,3,2,0,0],
                                             [1,3,3,2,0],
                                             [1,4,4,2,0],
                                             [1,3,3,3,2]],
                                            [[1,4,2,0,0],
                                             [1,3,2,0,0],
                                             [1,3,3,2,0],
                                             [1,4,4,2,0],
                                             [1,3,3,3,2]]])
        probs_all_correct = torch.tensor([[0.3, 0.275, 0.12375, 0.12, 0.0556875],
                                          [0.3, 0.275, 0.12375, 0.12, 0.0556875]])

        self.assertTrue(torch.equal(symbols_final_out, symbols_final_correct))
        self.assertTrue(torch.equal(symbols_all_out, symbols_all_correct))
        self.assertTrue(torch.equal(probs_all_out, torch.log(probs_all_correct)))

    # Test the case where beam size is k and the number of strings
    # in the language is n < k.
    def testBeamSearchBeamTooBig(self):
        # A distribution that only generates two strings
        def mock_autoregressive_fn(cumul_symbols, timestep, cache):
            dist1    = torch.tensor([[[0.0, 0.0, 0.0, 0.4, 0.6]]])
            dist2    = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]])
            dist3    = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 0.0]]])
            all_dist = (cumul_symbols == 1).unsqueeze(-1).type(torch.float) * dist1 + \
                       (cumul_symbols == 3).unsqueeze(-1).type(torch.float) * dist2 + \
                       (cumul_symbols == 4).unsqueeze(-1).type(torch.float) * dist3
            return torch.rand(all_dist.size()), torch.log(all_dist)

        max_lengths = torch.tensor([40]*2)
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(self.config, 2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        symbols_final_correct = torch.tensor([[1,4,2],
                                              [1,4,2]])
        symbols_all_correct = torch.tensor([[[1,4,2],
                                             [1,3,2],
                                             [0,0,0],
                                             [0,0,0],
                                             [0,0,0]],
                                            [[1,4,2],
                                             [1,3,2],
                                             [0,0,0],
                                             [0,0,0],
                                             [0,0,0]]])
        probs_all_correct = torch.tensor([[0.6, 0.4, 0.0, 0.0, 0.0],
                                          [0.6, 0.4, 0.0, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_final_out, symbols_final_correct))
        self.assertTrue(torch.equal(symbols_all_out, symbols_all_correct))
        self.assertTrue(torch.equal(probs_all_out, torch.log(probs_all_correct)))

    def testBeamSearchNoEmptyString(self):
        # A distribution that places its highest probability on the empty string
        def mock_autoregressive_fn(cumul_symbols, timestep, cache):
            dist1    = torch.tensor([[[0.0, 0.0, 0.6, 0.25, 0.15]]])
            dist2    = torch.tensor([[[0.0, 0.0, 1.0, 0.0,  0.0]]])
            dist3    = torch.tensor([[[0.0, 0.0, 1.0, 0.0,  0.0]]])
            all_dist = (cumul_symbols == 1).unsqueeze(-1).type(torch.float) * dist1 + \
                       (cumul_symbols == 3).unsqueeze(-1).type(torch.float) * dist2 + \
                       (cumul_symbols == 4).unsqueeze(-1).type(torch.float) * dist3
            return torch.rand(all_dist.size()), torch.log(all_dist)

        max_lengths = torch.tensor([40]*2)

        self.config.allow_empty_string = True
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(self.config, 2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        symbols_final_correct = torch.tensor([[1,2,0],
                                              [1,2,0]])
        symbols_all_correct = torch.tensor([[[1,2,0],
                                             [1,3,2],
                                             [1,4,2],
                                             [0,0,0],
                                             [0,0,0]],
                                            [[1,2,0],
                                             [1,3,2],
                                             [1,4,2],
                                             [0,0,0],
                                             [0,0,0]]])
        probs_all_correct = torch.tensor([[0.6, 0.25, 0.15, 0.0, 0.0],
                                          [0.6, 0.25, 0.15, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_final_out, symbols_final_correct))
        self.assertTrue(torch.equal(symbols_all_out, symbols_all_correct))
        self.assertTrue(torch.equal(probs_all_out, torch.log(probs_all_correct)))

        self.config.allow_empty_string = False
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(self.config, 2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        symbols_final_correct = torch.tensor([[1,3,2],
                                              [1,3,2]])
        symbols_all_correct = torch.tensor([[[1,3,2],
                                             [1,4,2],
                                             [0,0,0],
                                             [0,0,0],
                                             [0,0,0]],
                                            [[1,3,2],
                                             [1,4,2],
                                             [0,0,0],
                                             [0,0,0],
                                             [0,0,0]]])
        probs_all_correct = torch.tensor([[0.25, 0.15, 0.0, 0.0, 0.0],
                                          [0.25, 0.15, 0.0, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_final_out, symbols_final_correct))
        self.assertTrue(torch.equal(symbols_all_out, symbols_all_correct))
        self.assertTrue(torch.equal(probs_all_out, torch.log(probs_all_correct)))

    def testLengthNormalization(self):
        # Not used, but the lengths and probs are based on this,
        # so I include it for reference.
        symbols = torch.tensor([[[1,3,4,2],
                                 [1,4,2,0],
                                 [1,4,4,3]],
                                [[1,4,3,3],
                                 [1,3,3,3],
                                 [1,3,3,2]],
                                [[1,2,0,0],
                                 [1,3,2,0],
                                 [1,4,3,2]],
                                [[1,4,4,2],
                                 [1,2,0,0],
                                 [1,3,3,4]]])

        ni = float("-inf")
        log_probs = torch.tensor([[[-6.0, ni, ni, ni, ni],
                                   [-3.0, ni, ni, ni, ni],
                                   [ni, ni, -1.0, -2.0, -3.0]],
                                  [[ni, ni, -4.0, -5.0, -6.0],
                                   [ni, ni, -7.0, -8.0, -9.0],
                                   [-1.0, ni, ni, ni, ni]],
                                  [[-4.0, ni, ni, ni, ni],
                                   [-2.0, ni, ni, ni, ni],
                                   [-5.0, ni, ni, ni, ni]],
                                  [[-7.0, ni, ni, ni, ni],
                                   [-8.0, ni, ni, ni, ni],
                                   [ni, ni, -10.0, -11.0, -12.0]]])
        lengths = torch.tensor([[[3,3,3,3,3],
                                 [2,2,2,2,2],
                                 [4,5,4,5,5]],
                                [[4,5,4,5,5],
                                 [4,5,4,5,5],
                                 [3,3,3,3,3]],
                                [[1,1,1,1,1],
                                 [2,2,2,2,2],
                                 [3,3,3,3,3]],
                                [[3,3,3,3,3],
                                 [1,1,1,1,1],
                                 [4,5,4,5,5]]])
        class MockBeamManager:
            def get_all_choices_lengths(self):
                return lengths
        bm = MockBeamManager()

        self.config.length_normalization = LengthNormalization.NONE
        self.config.length_reward_gamma = 5.0
        self.config.length_norm_alpha = 2.0
        log_probs_out = self.gen.length_normalize(self.config, bm, log_probs)
        self.assertTrue(torch.equal(log_probs_out, log_probs))

        self.config.length_normalization = LengthNormalization.LENGTH_REWARD
        self.config.length_reward_gamma = 3.0
        self.config.length_norm_alpha = 2.0
        log_probs_correct = torch.tensor([[[-6.0+3*3.0, ni, ni, ni, ni],
                                           [-3.0+2*3.0, ni, ni, ni, ni],
                                           [ni, ni, -1.0+4*3.0, -2.0+5*3.0, -3.0+5*3.0]],
                                          [[ni, ni, -4.0+4*3.0, -5.0+5*3.0, -6.0+5*3.0],
                                           [ni, ni, -7.0+4*3.0, -8.0+5*3.0, -9.0+5*3.0],
                                           [-1.0+3*3.0, ni, ni, ni, ni]],
                                          [[-4.0+1*3.0, ni, ni, ni, ni],
                                           [-2.0+2*3.0, ni, ni, ni, ni],
                                           [-5.0+3*3.0, ni, ni, ni, ni]],
                                          [[-7.0+3*3.0, ni, ni, ni, ni],
                                           [-8.0+1*3.0, ni, ni, ni, ni],
                                           [ni, ni, -10.0+4*3.0, -11.0+5*3.0, -12.0+5*3.0]]])
        log_probs_out = self.gen.length_normalize(self.config, bm, log_probs)
        self.assertTrue(torch.equal(log_probs_out, log_probs_correct))

        self.config.length_normalization = LengthNormalization.LENGTH_NORM
        self.config.length_reward_gamma = 3.0
        self.config.length_norm_alpha = 2.0
        log_probs_correct = torch.tensor([[[-6.0/3, ni, ni, ni, ni],
                                           [-3.0/2, ni, ni, ni, ni],
                                           [ni, ni, -1.0/4, -2.0/5, -3.0/5]],
                                          [[ni, ni, -4.0/4, -5.0/5, -6.0/5],
                                           [ni, ni, -7.0/4, -8.0/5, -9.0/5],
                                           [-1.0/3, ni, ni, ni, ni]],
                                          [[-4.0/1, ni, ni, ni, ni],
                                           [-2.0/2, ni, ni, ni, ni],
                                           [-5.0/3, ni, ni, ni, ni]],
                                          [[-7.0/3, ni, ni, ni, ni],
                                           [-8.0/1, ni, ni, ni, ni],
                                           [ni, ni, -10.0/4, -11.0/5, -12.0/5]]])
        log_probs_out = self.gen.length_normalize(self.config, bm, log_probs)
        self.assertTrue(torch.equal(log_probs_out, log_probs_correct))

        # not doing the calculation on this one, just making sure it runs
        self.config.length_normalization = LengthNormalization.GOOGLE_METHOD
        self.config.length_reward_gamma = 3.0
        self.config.length_norm_alpha = 2.0
        log_probs_out = self.gen.length_normalize(self.config, bm, log_probs)
        self.assertEqual(log_probs_out.size(), log_probs.size())



class TestClusterBeamSearch(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        vocab_size = 10

        device = "cpu"
        config = read_config("configuration.toml")
        model = MockModelDoesNothing()

        self.gen = Generator(model, config, device, vocab_size, self.pad_idx, self.bos_idx, self.eos_idx)
        self.config = config.gen
        self.config.cluster_beam_search_alpha = 1

    def testClusterWithIdentityDistanceReducesToBeam(self):

        class MockCacheStoresFinished:
            def __init__(self, is_cluster):
                self.is_cluster = is_cluster
                self.finished_beams = torch.tensor([False]*4)
            def expand_to_beam_size(self, beam_size):
                return
            def register_finished_sents(self, mask):
                self.finished_mask = mask
            def register_finished_beams(self, mask):
                self.finished_beams = mask
            def select_idxs(self, chosen_idxs):
                return

        # Create a random distribution.
        # Should give the same answers for the beam items in both
        # beam search and cluster beam search.
        logits = torch.rand(4,5,6+1,8,10)
        logits_modified = logits.clone()
        logits_modified[:,:,:,:,0] = float("-inf")
        logits_modified[:,:,:,:,1] = float("-inf")
        dist = torch.nn.functional.log_softmax(torch.rand(4,5,6+1,8,10), dim=-1)

        def auto_fn(symbols, timestep, cache):
            if cache.is_cluster:
                dist_reshape = dist.clone()[~cache.finished_beams]
                logits_reshape = logits.clone()[~cache.finished_beams]
            else:
                dist_reshape = dist.clone()[:,:,0,:,:][~cache.finished_beams]
                logits_reshape = logits.clone()[:,:,0,:,:][~cache.finished_beams]
            dist_ret = dist_reshape.reshape(-1,8,10)[:,timestep:timestep+symbols.size(1),:][~cache.finished_mask]
            logits_ret = logits_reshape.reshape(-1,8,10)[:,timestep:timestep+symbols.size(1),:][~cache.finished_mask]
            return logits_ret, dist_ret

        batch_size = 4
        beam_size = 5
        ball_size = 6
        vocab_size = 10
        max_lengths = torch.tensor([8]*batch_size)
        max_poss_length = 8

        cache_beam = MockCacheStoresFinished(False)
        symbols_final_beam, symbols_all_beam, probs_all_beam = self.gen.beam_search(self.config, batch_size, beam_size, max_lengths, max_poss_length, auto_fn, cache_beam)

        cache_cluster = MockCacheStoresFinished(True)
        symbols_final_cluster, symbols_all_cluster, probs_all_cluster= self.gen.cluster_beam_search(self.config, batch_size, beam_size, ball_size, max_lengths, max_poss_length, Identity, auto_fn, cache_cluster)

        self.assertTrue(torch.equal(symbols_final_beam, symbols_final_cluster))
        self.assertTrue(torch.equal(symbols_all_beam, symbols_all_cluster[:,:,0,:]))
        self.assertTrue(torch.equal(probs_all_beam, probs_all_cluster[:,:,0]))
