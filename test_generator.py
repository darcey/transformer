import math
import torch
import unittest
from configuration import read_config, DecodingMethod, SamplingMethod
from generator import *



class MockModelDoesNothing:
    def get_autoregressive_one_step_fn(self, src, cache):
        return 5

class MockCacheDoesNothing:
    def expand(self, num_samples):
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

        def mock_sample_outer_loop(b, max_lengths, max_possible_length, fn, cache):
            self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
            self.assertEqual(max_possible_length, 9)
        self.gen.sample_outer_loop = mock_sample_outer_loop
        self.gen.generate(src)

    def testMaxLengthsAbsolute(self):
        self.gen.config.use_rel_max_len = False
        self.gen.config.rel_max_len = 5
        self.gen.config.abs_max_len = 6
        src = torch.tensor([[1,2,3,0,0],
                            [1,2,3,4,0],
                            [1,2,0,0,0]])
        max_lengths_correct = torch.tensor([6,6,6])

        def mock_sample_outer_loop(b, max_lengths, max_possible_length, fn, cache):
            self.assertTrue(torch.equal(max_lengths, max_lengths_correct))
            self.assertEqual(max_possible_length, 6)
        self.gen.sample_outer_loop = mock_sample_outer_loop
        self.gen.generate(src)



class TestSampling(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.a_idx   = 3
        self.b_idx   = 4

        device = "cpu"
        config = read_config("configuration.toml")
        model = None

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
        probs_correct = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0],
                                      [1.0, 1.0, 1.0, 1.0, 1.0]])

        samples_out, probs_out = self.gen.sample_outer_loop(2, None, None, None, None)

        self.assertTrue(torch.equal(samples_out, samples_correct))
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
        probs_correct = torch.tensor([[1.0,1.0,1.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,3.0,3.0,3.0]])

        samples_out, probs_out = self.gen.sample_outer_loop(1, torch.rand(5), None, None, None)

        self.assertTrue(torch.equal(samples_out, samples_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSamplePrunesEOS(self):
        orig_size = 10
        timestep = 0
        def mock_autoregressive_fn(cumul_symbols, cache):
            # No sentence should end w/ EOS because
            # all sentences w/ EOS should be pruned
            eos_count = (cumul_symbols == self.eos_idx).sum()
            self.assertEqual(eos_count, 0)

            # Number of generations should decrease by one at each timestep
            nonlocal timestep
            self.assertTrue(timestep < orig_size)
            self.assertEqual(cumul_symbols.size(), (orig_size - timestep, timestep+1))
            timestep += 1

            # Mock distribution to return;
            # ends first sentence and continues rest
            ni = float("-inf")
            eos_dist = [ni, ni, 0.0, ni]
            a_dist   = [ni, ni, ni, 0.0]
            size     = cumul_symbols.size(0)
            all_dist = torch.tensor([eos_dist] + [list(a_dist) for i in range(size-1)])
            return all_dist

        # Max lengths should not play a role in this test
        max_lengths = torch.tensor([30]*2)
        symbols_out, probs_out = self.gen.sample(2, int(orig_size/2), max_lengths, 30, mock_autoregressive_fn, self.cache)

        symbols_correct = torch.tensor([[[1,2,0,0,0,0,0,0,0,0,0],
                                         [1,3,2,0,0,0,0,0,0,0,0],
                                         [1,3,3,2,0,0,0,0,0,0,0],
                                         [1,3,3,3,2,0,0,0,0,0,0],
                                         [1,3,3,3,3,2,0,0,0,0,0]],
                                        [[1,3,3,3,3,3,2,0,0,0,0],
                                         [1,3,3,3,3,3,3,2,0,0,0],
                                         [1,3,3,3,3,3,3,3,2,0,0],
                                         [1,3,3,3,3,3,3,3,3,2,0],
                                         [1,3,3,3,3,3,3,3,3,3,2]]])
        probs_correct = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_out, symbols_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSamplePrunesEOSReverseOrder(self):
        # This is the same as the previous test but tests reordering
        # by finishing the sentences in reverse order.
        def mock_autoregressive_fn(cumul_symbols, cache):
            # Mock distribution to return;
            # ends last sentence and continues rest
            ni = float("-inf")
            eos_dist = [ni, ni, 0.0, ni]
            a_dist   = [ni, ni, ni, 0.0]
            size     = cumul_symbols.size(0)
            all_dist = torch.tensor([list(a_dist) for i in range(size-1)] + [eos_dist])
            return all_dist

        # Max lengths do not play a role in this test
        max_lengths = torch.tensor([30]*2)
        symbols_out, probs_out = self.gen.sample(2, 5, max_lengths, 30, mock_autoregressive_fn, self.cache)

        symbols_correct = torch.tensor([[[1,3,3,3,3,3,3,3,3,3,2],
                                         [1,3,3,3,3,3,3,3,3,2,0],
                                         [1,3,3,3,3,3,3,3,2,0,0],
                                         [1,3,3,3,3,3,3,2,0,0,0],
                                         [1,3,3,3,3,3,2,0,0,0,0]],
                                        [[1,3,3,3,3,2,0,0,0,0,0],
                                         [1,3,3,3,2,0,0,0,0,0,0],
                                         [1,3,3,2,0,0,0,0,0,0,0],
                                         [1,3,2,0,0,0,0,0,0,0,0],
                                         [1,2,0,0,0,0,0,0,0,0,0]]])
        probs_correct = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_out, symbols_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSamplePrunesMaxLengthRel(self):
        def mock_autoregressive_fn(cumul_symbols, cache):
            # Mock distribution to return;
            # always just assigns probabiliy 1 to a.
            ni = float("-inf")
            a_dist   = [ni, ni, ni, 0.0]
            size     = cumul_symbols.size(0)
            all_dist = torch.tensor([list(a_dist) for i in range(size)])
            return all_dist

        # This time the max lengths should determine when the sentence stops
        max_lengths = torch.tensor([1,3])
        symbols_out, probs_out = self.gen.sample(2, 3, max_lengths, 30, mock_autoregressive_fn, self.cache)

        symbols_correct = torch.tensor([[[1,3,0,0],
                                         [1,3,0,0],
                                         [1,3,0,0]],
                                        [[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,3]]])
        probs_correct = torch.tensor([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_out, symbols_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSampleStopsAtMaxLengthAbs(self):
        def mock_autoregressive_fn(cumul_symbols, cache):
            # Mock distribution to return;
            # always just assigns probabiliy 1 to a.
            ni = float("-inf")
            a_dist   = [ni, ni, ni, 0.0]
            size     = cumul_symbols.size(0)
            all_dist = torch.tensor([list(a_dist) for i in range(size)])
            return all_dist

        # This time the max lengths should determine when the sentence stops
        max_lengths = torch.tensor([40,40])
        symbols_out, probs_out = self.gen.sample(2, 3, max_lengths, 3, mock_autoregressive_fn, self.cache)

        symbols_correct = torch.tensor([[[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,3]],
                                        [[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,3]]])
        probs_correct = torch.tensor([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0]])

        self.assertTrue(torch.equal(symbols_out, symbols_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSampleSimpleDistribution(self):
        def mock_autoregressive_fn(cumul_symbols, cache):
            a_dist   = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0])
            b_dist   = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.5])
            ab_dist  = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.5])
            all_dist = (cumul_symbols[:,-1] == 1).unsqueeze(1).type(torch.float) * ab_dist + \
                       (cumul_symbols[:,-1] == 3).unsqueeze(1).type(torch.float) * a_dist + \
                       (cumul_symbols[:,-1] == 4).unsqueeze(1).type(torch.float) * b_dist
            return torch.log(all_dist)

        max_lengths = torch.tensor([40]*1000)
        symbols_out, probs_out = self.gen.sample(1000, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        # Every sample should have just a or just b
        a_samples = torch.eq(symbols_out, 3).sum(dim=-1).type(torch.bool)
        b_samples = torch.eq(symbols_out, 4).sum(dim=-1).type(torch.bool)
        self.assertFalse(torch.logical_and(a_samples, b_samples).any())

        # Should be roughly half a, half b
        self.assertAlmostEqual(a_samples.sum()/5000, 0.5, delta=0.02)
        self.assertAlmostEqual(b_samples.sum()/5000, 0.5, delta=0.02)

    def testTopK(self):
        dist = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 0.5, 2.0, 4.0, 6.0, 8.0])
        dist = dist.unsqueeze(0)

        self.gen.config.decoding_method = DecodingMethod.SAMPLING
        self.gen.config.sampling_method = SamplingMethod.TOP_K
        self.gen.config.sampling_k = 5
        dist_correct = torch.tensor([0.0, 0.0, 5.0, 7.0, 9.0, 0.0, 0.0, 0.0, 6.0, 8.0])
        dist_correct = dist_correct.unsqueeze(0)

        dist_out = self.gen.adjust_or_truncate_probs(dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))

    def testTopP(self):
        self.gen.config.decoding_method = DecodingMethod.SAMPLING
        self.gen.config.sampling_method = SamplingMethod.TOP_P

        # No matter what the distribution is, p = 1.0 should just return it.
        self.gen.config.sampling_p = 1.0
        dist = torch.nn.functional.softmax(torch.rand(5,8), dim=-1)
        dist_out = self.gen.adjust_or_truncate_probs(dist)
        self.assertTrue(torch.equal(dist_out, dist))

        # Test for p < 1.0
        self.gen.config.sampling_p = 0.6
        dist = torch.tensor([[0.26, 0.1, 0.2,  0.03, 0.05, 0.02, 0.3,  0.04],
                             [0.08, 0.5, 0.02, 0.25, 0.05, 0.04, 0.03, 0.03]])
        dist_correct = torch.tensor([[0.26, 0.0, 0.2, 0.0,  0.0, 0.0, 0.3, 0.0],
                                     [0.0 , 0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0]])
        dist_out = self.gen.adjust_or_truncate_probs(dist)
        self.assertTrue(torch.equal(dist_out, dist_correct))
