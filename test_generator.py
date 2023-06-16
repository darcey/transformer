# TODO(darcey): figure out how to suppress warnings during testing

import math
import torch
import unittest
from configuration import read_config, DecodingMethod, SamplingMethod
from generator import *



class MockModelDoesNothing:
    def get_autoregressive_one_step_fn(self, src, cache):
        return 5

class MockCacheDoesNothing:
    def expand_to_beam_size(self, beam_size):
        return
    def trim_finished_sents(self, finished):
        return



class TestGenerate(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2

        device = "cpu"
        config = read_config("configuration.toml")
        model = MockModelDoesNothing()

        self.gen = Generator(model, config, device, self.pad_idx, self.bos_idx, self.eos_idx)

    def testMaxLengthsRelative(self):
        self.gen.config.use_rel_max_len = True
        self.gen.config.rel_max_len = 5
        self.gen.config.abs_max_len = 6
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([8,9,7])

        max_lengths, max_possible_length = self.gen.get_max_lengths(src)
        self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
        self.assertEqual(max_possible_length, 9)

    def testMaxLengthsAbsolute(self):
        self.gen.config.use_rel_max_len = False
        self.gen.config.rel_max_len = 5
        self.gen.config.abs_max_len = 6
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([6,6,6])

        max_lengths, max_possible_length = self.gen.get_max_lengths(src)
        self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
        self.assertEqual(max_possible_length, 6)

    def testMaxLengthsContextWindow(self):
        self.gen.config.use_rel_max_len = True
        self.gen.config.rel_max_len = 7
        self.gen.window = 5
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([5,5,5])

        max_lengths, max_possible_length = self.gen.get_max_lengths(src)
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

        self.gen = Generator(model, config, device, self.pad_idx, self.bos_idx, self.eos_idx)
        self.cache = MockCacheDoesNothing()

    def testSampleOuterLoopOneIter(self):
        iteration = 0
        def mock_sample_function(batch, sample, w, x, y, z):
            nonlocal iteration
            iteration += 1
            all_symbols = torch.full(size=(batch, sample, iteration*2), fill_value=iteration)
            all_probs = torch.full(size=(batch, sample), fill_value=float(iteration))
            return all_symbols, all_probs
        self.gen.sample = mock_sample_function
        self.gen.config.max_parallel_sentences = 12
        self.gen.config.num_beams_or_samples = 5

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

        samples_final_out, samples_out, probs_out = self.gen.sample_outer_loop(2, None, None, None, None)

        self.assertTrue(torch.equal(samples_out, samples_correct))
        self.assertTrue(torch.equal(samples_final_out, samples_final_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSampleOuterLoopManyIter(self):
        iteration = 0
        def mock_sample_function(batch, sample, w, x, y, z):
            nonlocal iteration
            iteration += 1
            all_symbols = torch.full(size=(batch, sample, iteration*2), fill_value=iteration)
            all_probs = torch.full(size=(batch, sample), fill_value=float(iteration))
            return all_symbols, all_probs
        self.gen.sample = mock_sample_function
        self.gen.config.max_parallel_sentences = 5
        self.gen.config.num_beams_or_samples = 13

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

        samples_final_out, samples_out, probs_out = self.gen.sample_outer_loop(1, torch.rand(5), None, None, None)

        self.assertTrue(torch.equal(samples_out, samples_correct))
        self.assertTrue(torch.equal(samples_final_out, samples_final_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSampleSimpleDistribution1(self):
        # At beginning, splits off into two possibilities: all as or all bs.
        def mock_autoregressive_fn(cumul_symbols, cache, mask):
            a_dist   = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0])
            b_dist   = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.5])
            ab_dist  = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5])
            all_dist = (cumul_symbols[:,-1] == 1).unsqueeze(1).type(torch.float) * ab_dist + \
                       (cumul_symbols[:,-1] == 3).unsqueeze(1).type(torch.float) * a_dist + \
                       (cumul_symbols[:,-1] == 4).unsqueeze(1).type(torch.float) * b_dist
            return torch.log(all_dist)

        self.gen.model.vocab_size = 5
        max_lengths = torch.tensor([40]*1000)
        symbols_out, probs_out = self.gen.sample(1000, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

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
        def mock_autoregressive_fn(cumul_symbols, cache, mask):
            a_dist   = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0])
            b_dist   = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
            all_dist = (cumul_symbols[:,-1] == 1).unsqueeze(1).type(torch.float) * a_dist + \
                       (cumul_symbols[:,-1] == 3).unsqueeze(1).type(torch.float) * b_dist + \
                       (cumul_symbols[:,-1] == 4).unsqueeze(1).type(torch.float) * a_dist
            return torch.log(all_dist)

        self.gen.model.vocab_size = 5
        max_lengths = torch.tensor([40]*1000)
        symbols_out, probs_out = self.gen.sample(1000, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        # Every sample should have equal numbers of as and bs.
        num_as = torch.eq(symbols_out, 3).sum(dim=-1)
        num_bs = torch.eq(symbols_out, 4).sum(dim=-1)
        self.assertTrue(torch.equal(num_as, num_bs))

        # Should be roughly half length 2, half length 4, etc...
        self.assertAlmostEqual((num_as == 0).sum()/5000, 0.5, delta=0.02)
        self.assertAlmostEqual((num_as == 1).sum()/5000, 0.25, delta=0.02)

    def testTopK(self):
        dist = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 0.5, 2.0, 4.0, 6.0, 8.0])
        dist = dist.unsqueeze(0).unsqueeze(0)

        self.gen.config.decoding_method = DecodingMethod.SAMPLING
        self.gen.config.sampling_method = SamplingMethod.TOP_K
        self.gen.config.sampling_k = 5
        dist_correct = torch.tensor([0.0, 0.0, 5.0, 7.0, 9.0, 0.0, 0.0, 0.0, 6.0, 8.0])
        dist_correct = dist_correct.unsqueeze(0).unsqueeze(0)

        dist_out = self.gen.truncate_probs(dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))

    def testTopP(self):
        self.gen.config.decoding_method = DecodingMethod.SAMPLING
        self.gen.config.sampling_method = SamplingMethod.TOP_P

        # No matter what the distribution is, p = 1.0 should just return it.
        self.gen.config.sampling_p = 1.0
        dist = torch.nn.functional.softmax(torch.rand(3,5,8), dim=-1)
        dist_out = self.gen.truncate_probs(dist)
        self.assertTrue(torch.equal(dist_out, dist))

        # Test for p < 1.0
        self.gen.config.sampling_p = 0.6
        dist = torch.tensor([[[0.26, 0.1, 0.2,  0.03, 0.05, 0.02, 0.3,  0.04],
                              [0.08, 0.5, 0.02, 0.25, 0.05, 0.04, 0.03, 0.03]],
                             [[0.3,  0.0, 0.01, 0.02, 0.4,  0.03, 0.2,  0.04],
                              [0.1,  0.3, 0.2,  0.15, 0.05, 0.25, 0.02, 0.03]]])
        dist_correct = torch.tensor([[[0.26, 0.0, 0.2, 0.0,  0.0, 0.0,  0.3, 0.0],
                                      [0.0 , 0.5, 0.0, 0.25, 0.0, 0.0,  0.0, 0.0]],
                                     [[0.3,  0.0, 0.0, 0.0,  0.4, 0.0,  0.0, 0.0],
                                      [0.0,  0.3, 0.2, 0.0,  0.0, 0.25, 0.0, 0.0]]])
        dist_out = self.gen.truncate_probs(dist)
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

        self.gen = Generator(model, config, device, self.pad_idx, self.bos_idx, self.eos_idx)
        self.cache = MockCacheDoesNothing()

    def testBeamSearch(self):
        # Generates strings which are either all as or all bs
        def mock_autoregressive_fn(cumul_symbols, cache, mask):
            dist1    = torch.tensor([0.0, 0.0, 0.0,  0.5,  0.5])
            dist2    = torch.tensor([0.0, 0.0, 0.55, 0.45, 0.0])
            dist3    = torch.tensor([0.0, 0.0, 0.6,  0.0,  0.4])
            all_dist = (cumul_symbols[:,-1] == 1).unsqueeze(1).type(torch.float) * dist1 + \
                       (cumul_symbols[:,-1] == 3).unsqueeze(1).type(torch.float) * dist2 + \
                       (cumul_symbols[:,-1] == 4).unsqueeze(1).type(torch.float) * dist3
            return torch.log(all_dist)

        self.gen.model.vocab_size = 5
        max_lengths = torch.tensor([40]*2)
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

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
        def mock_autoregressive_fn(cumul_symbols, cache, mask):
            dist1    = torch.tensor([0.0, 0.0, 0.0, 0.4, 0.6])
            dist2    = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
            dist3    = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
            all_dist = (cumul_symbols[:,-1] == 1).unsqueeze(1).type(torch.float) * dist1 + \
                       (cumul_symbols[:,-1] == 3).unsqueeze(1).type(torch.float) * dist2 + \
                       (cumul_symbols[:,-1] == 4).unsqueeze(1).type(torch.float) * dist3
            return torch.log(all_dist)

        self.gen.model.vocab_size = 5
        max_lengths = torch.tensor([40]*2)
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

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
        def mock_autoregressive_fn(cumul_symbols, cache, mask):
            dist1    = torch.tensor([0.0, 0.0, 0.6, 0.25, 0.15])
            dist2    = torch.tensor([0.0, 0.0, 1.0, 0.0,  0.0])
            dist3    = torch.tensor([0.0, 0.0, 1.0, 0.0,  0.0])
            all_dist = (cumul_symbols[:,-1] == 1).unsqueeze(1).type(torch.float) * dist1 + \
                       (cumul_symbols[:,-1] == 3).unsqueeze(1).type(torch.float) * dist2 + \
                       (cumul_symbols[:,-1] == 4).unsqueeze(1).type(torch.float) * dist3
            return torch.log(all_dist)

        self.gen.model.vocab_size = 5
        max_lengths = torch.tensor([40]*2)

        self.gen.config.allow_empty_string = True
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

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

        self.gen.config.allow_empty_string = False
        symbols_final_out, symbols_all_out, probs_all_out = self.gen.beam_search(2, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

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
