import torch
import unittest
from beam_manager import *



def mock_auto_fn_does_nothing(symbols, timestep, cache):
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
    


class TestBeamManager(unittest.TestCase):

    def testInit(self):
        bm = BeamManager(batch_size=2,
                         beam_size=3,
                         vocab_size=4,
                         max_lengths=torch.tensor([3,5,7,4,6]),
                         max_possible_length=6,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")
        symbols_correct = torch.tensor([[[1],
                                         [1],
                                         [1]],
                                        [[1],
                                         [1],
                                         [1]]])
        probs_correct = torch.tensor([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0]])
        self.assertTrue(torch.equal(bm.symbols, symbols_correct))
        self.assertTrue(torch.equal(bm.probs, probs_correct))

    # Check that it correctly prunes sentences where everything
    # in the beam ends with EOS or PAD, but if only some items
    # in the beam end with EOS or PAD, it leaves that sentence alone.
    def testPruneFinishedEOSorPAD(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=4,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        bm.symbols = torch.tensor([[[1,3,3,3],
                                    [1,3,3,3],
                                    [1,3,3,3]],
                                   [[1,3,3,0],
                                    [1,3,3,2],
                                    [1,3,3,2]],
                                   [[1,3,3,3],
                                    [1,3,3,0],
                                    [1,3,3,2]],
                                   [[1,3,3,2],
                                    [1,3,3,0],
                                    [1,3,3,0]]])
        symbols_correct = torch.tensor([[[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,3]],
                                        [[1,3,3,3],
                                         [1,3,3,0],
                                         [1,3,3,2]]])
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, symbols_correct))

    def testPruneFinishedMaxLengths(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([4,5,4,7]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        bm.seq_len = 4
        bm.symbols = torch.tensor([[[1,3,3,3],
                                    [1,3,3,3],
                                    [1,3,3,3]],
                                   [[1,3,3,3],
                                    [1,3,3,3],
                                    [1,3,3,4]],
                                   [[1,3,3,3],
                                    [1,3,3,4],
                                    [1,3,3,4]],
                                   [[1,3,3,4],
                                    [1,3,3,4],
                                    [1,3,3,4]]])
        symbols_correct = torch.tensor([[[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,4]],
                                        [[1,3,3,4],
                                         [1,3,3,4],
                                         [1,3,3,4]]])
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, symbols_correct))
    
    def testPruneFinishedMaxPossibleLength(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=4,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        bm.seq_len = 4
        bm.symbols = torch.tensor([[[1,3,3,3],
                                    [1,3,3,3],
                                    [1,3,3,3]],
                                   [[1,3,3,3],
                                    [1,3,3,3],
                                    [1,3,3,4]],
                                   [[1,3,3,3],
                                    [1,3,3,4],
                                    [1,3,3,4]],
                                   [[1,3,3,4],
                                    [1,3,3,4],
                                    [1,3,3,4]]])
        symbols_correct = torch.empty(0,3,4)
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, symbols_correct))

    def testPruneFinishedIterated(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([3,6,4,5]),
                         max_possible_length=6,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        # This is what it will be generating
        symbols_correct = torch.tensor([[[1,3,2,0,0,0],
                                         [1,3,3,0,0,0],
                                         [1,3,2,0,0,0]],
                                        [[1,3,3,3,3,3],
                                         [1,3,3,3,3,2],
                                         [1,3,3,3,2,0]],
                                        [[1,3,3,2,0,0],
                                         [1,3,3,3,0,0],
                                         [1,3,2,0,0,0]],
                                        [[1,3,3,3,3,0],
                                         [1,3,3,2,0,0],
                                         [1,2,0,0,0,0]]])
        probs_correct = torch.tensor([[0,1,2,],[3,4,5],[6,7,8],[9,10,11]])
        bm.probs = probs_correct

        # Iteration 1
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, symbols_correct[:,:,:1]))
        self.assertFalse(bm.all_done())

        # Iteration 2
        bm.symbols = torch.cat((bm.symbols, symbols_correct[:,:,1].unsqueeze(-1)), dim=-1)
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, symbols_correct[:,:,:2]))
        self.assertFalse(bm.all_done())

        # Iteration 3
        bm.symbols = torch.cat((bm.symbols, symbols_correct[:,:,2].unsqueeze(-1)), dim=-1)
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.symbols
        pruned_correct = torch.tensor([True, False, False, False])
        self.assertTrue(torch.equal(symbols_out, symbols_correct[~pruned_correct][:,:,:3]))
        self.assertFalse(bm.all_done())

        # Iteration 4
        bm.symbols = torch.cat((bm.symbols, symbols_correct[~pruned_correct][:,:,3].unsqueeze(-1)), dim=-1)
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.symbols
        pruned_correct = torch.tensor([True, False, True, False])
        self.assertTrue(torch.equal(symbols_out, symbols_correct[~pruned_correct][:,:,:4]))
        self.assertFalse(bm.all_done())

        # Iteration 5
        bm.symbols = torch.cat((bm.symbols, symbols_correct[~pruned_correct][:,:,4].unsqueeze(-1)), dim=-1)
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.symbols
        pruned_correct = torch.tensor([True, False, True, True])
        self.assertTrue(torch.equal(symbols_out, symbols_correct[~pruned_correct][:,:,:5]))
        self.assertFalse(bm.all_done())

        # Iteration 5
        bm.symbols = torch.cat((bm.symbols, symbols_correct[~pruned_correct][:,:,5].unsqueeze(-1)), dim=-1)
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, torch.empty(0,3,6)))
        self.assertTrue(bm.all_done())

        symbols_out, probs_out = bm.get_final()
        self.assertTrue(torch.equal(symbols_out, symbols_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testComputeNextTokenProbsNoFinished(self):
        symbols = torch.tensor([[[1,3,3,3],
                                 [1,3,3,3],
                                 [1,3,3,3]],
                                [[1,3,3,3],
                                 [1,3,3,3],
                                 [1,3,3,4]],
                                [[1,3,3,3],
                                 [1,3,3,4],
                                 [1,3,3,4]],
                                [[1,3,3,4],
                                 [1,3,3,4],
                                 [1,3,3,4]]])

        def mock_auto_fn(active_symbols, timestep, cache):
            self.assertTrue(torch.equal(active_symbols, symbols.reshape(4*3,4)[:,timestep:timestep+1]))

            batch_beam = active_symbols.size(0)
            probs3 = torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4]]]).expand(batch_beam, -1, -1)
            probs4 = torch.tensor([[[0.5, 0.6, 0.7, 0.8, 0.9]]]).expand(batch_beam, -1, -1)
            is3 = (active_symbols == 3).type(torch.int).unsqueeze(-1)
            is4 = (active_symbols == 4).type(torch.int).unsqueeze(-1)
            probs = is3 * probs3 + is4 * probs4
            return torch.rand(probs.size()), probs

        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")
        bm.symbols = symbols
        bm.probs = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        bm.seq_len = 4

        next_token_probs_correct = torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4],
                                                  [0.0, 0.1, 0.2, 0.3, 0.4],
                                                  [0.0, 0.1, 0.2, 0.3, 0.4]],
                                                 [[0.0, 0.1, 0.2, 0.3, 0.4],
                                                  [0.0, 0.1, 0.2, 0.3, 0.4],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9]],
                                                 [[0.0, 0.1, 0.2, 0.3, 0.4],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9]],
                                                 [[0.5, 0.6, 0.7, 0.8, 0.9],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9]]])
        all_choices_probs_correct = torch.tensor([[[1.0, 1.1, 1.2, 1.3, 1.4],
                                                   [2.0, 2.1, 2.2, 2.3, 2.4],
                                                   [3.0, 3.1, 3.2, 3.3, 3.4]],
                                                  [[4.0, 4.1, 4.2, 4.3, 4.4],
                                                   [5.0, 5.1, 5.2, 5.3, 5.4],
                                                   [6.5, 6.6, 6.7, 6.8, 6.9]],
                                                  [[7.0, 7.1, 7.2, 7.3, 7.4],
                                                   [8.5, 8.6, 8.7, 8.8, 8.9],
                                                   [9.5, 9.6, 9.7, 9.8, 9.9]],
                                                  [[10.5, 10.6, 10.7, 10.8, 10.9],
                                                   [11.5, 11.6, 11.7, 11.8, 11.9],
                                                   [12.5, 12.6, 12.7, 12.8, 12.9]]])
        _, next_token_probs_out, all_choices_probs_out = bm.compute_next_token_probs()
        self.assertTrue(torch.equal(next_token_probs_out, next_token_probs_correct))
        self.assertTrue(torch.equal(all_choices_probs_out, all_choices_probs_correct))

    def testComputeNextTokenProbsSomeFinished(self):
        symbols = torch.tensor([[[1,3,3,2],
                                 [1,3,2,0],
                                 [1,3,3,3]],
                                [[1,3,3,3],
                                 [1,3,3,2],
                                 [1,3,3,4]],
                                [[1,3,2,0],
                                 [1,3,3,2],
                                 [1,3,3,4]],
                                [[1,3,3,4],
                                 [1,3,3,4],
                                 [1,3,3,2]]])
        active_correct = torch.tensor([[1,3,3,3],
                                       [1,3,3,3],
                                       [1,3,3,4],
                                       [1,3,3,4],
                                       [1,3,3,4],
                                       [1,3,3,4]])
    
        def mock_auto_fn(active_symbols, timestep, cache):
            self.assertTrue(torch.equal(active_symbols, active_correct[:,timestep:timestep+1]))

            batch_beam = active_symbols.size(0)
            probs3 = torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4]]]).expand(batch_beam, -1, -1)
            probs4 = torch.tensor([[[0.5, 0.6, 0.7, 0.8, 0.9]]]).expand(batch_beam, -1, -1)
            is3 = (active_symbols == 3).type(torch.int).unsqueeze(-1)
            is4 = (active_symbols == 4).type(torch.int).unsqueeze(-1)
            probs = is3 * probs3 + is4 * probs4
            return torch.rand(probs.size()), probs

        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")
        bm.symbols = symbols
        bm.probs = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
        bm.seq_len = 4

        ni = float("-inf")
        next_token_probs_correct = torch.tensor([[[0.0, ni, ni, ni, ni],
                                                  [0.0, ni, ni, ni, ni],
                                                  [0.0, 0.1, 0.2, 0.3, 0.4]],
                                                 [[0.0, 0.1, 0.2, 0.3, 0.4],
                                                  [0.0, ni, ni, ni, ni],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9]],
                                                 [[0.0, ni, ni, ni, ni],
                                                  [0.0, ni, ni, ni, ni],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9]],
                                                 [[0.5, 0.6, 0.7, 0.8, 0.9],
                                                  [0.5, 0.6, 0.7, 0.8, 0.9],
                                                  [0.0, ni, ni, ni, ni],]])
        all_choices_probs_correct = torch.tensor([[[1.0, ni, ni, ni, ni],
                                                   [2.0, ni, ni, ni, ni],
                                                   [3.0, 3.1, 3.2, 3.3, 3.4]],
                                                  [[4.0, 4.1, 4.2, 4.3, 4.4],
                                                   [5.0, ni, ni, ni, ni],
                                                   [6.5, 6.6, 6.7, 6.8, 6.9]],
                                                  [[7.0, ni, ni, ni, ni],
                                                   [8.0, ni, ni, ni, ni],
                                                   [9.5, 9.6, 9.7, 9.8, 9.9]],
                                                  [[10.5, 10.6, 10.7, 10.8, 10.9],
                                                   [11.5, 11.6, 11.7, 11.8, 11.9],
                                                   [12.0, ni, ni, ni, ni]]])
        _, next_token_probs_out, all_choices_probs_out = bm.compute_next_token_probs()
        self.assertTrue(torch.equal(next_token_probs_out, next_token_probs_correct))
        self.assertTrue(torch.equal(all_choices_probs_out, all_choices_probs_correct))

    def testSelectIdxsIndependent(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        all_choices_probs = torch.tensor([[[1.0, 1.1, 1.2, 1.3, 1.4],
                                           [2.0, 2.1, 2.2, 2.3, 2.4],
                                           [3.0, 3.1, 3.2, 3.3, 3.4]],
                                          [[4.0, 4.1, 4.2, 4.3, 4.4],
                                           [5.0, 5.1, 5.2, 5.3, 5.4],
                                           [6.0, 6.1, 6.2, 6.3, 6.4]],
                                          [[7.0, 7.1, 7.2, 7.3, 7.4],
                                           [8.0, 8.1, 8.2, 8.3, 8.4],
                                           [9.0, 9.1, 9.2, 9.3, 9.4]],
                                          [[10.0, 10.1, 10.2, 10.3, 10.4],
                                           [11.0, 11.1, 11.2, 11.3, 11.4],
                                           [12.0, 12.1, 12.2, 12.3, 12.4]]])
        choices = torch.tensor([[2,4,1],[0,3,3],[2,3,1],[4,4,2]])
        probs_correct = torch.tensor([[1.2,2.4,3.1],
                                      [4.0,5.3,6.3],
                                      [7.2,8.3,9.1],
                                      [10.4,11.4,12.2]])
        
        bm.all_choices_probs = all_choices_probs
        bm.select_idxs_independent(choices)
        probs_out = bm.probs
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSelectIdxsDependent(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         vocab_size=5,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        all_choices_probs = torch.tensor([[[1.0, 1.1, 1.2, 1.3, 1.4],
                                           [2.0, 2.1, 2.2, 2.3, 2.4],
                                           [3.0, 3.1, 3.2, 3.3, 3.4]],
                                          [[4.0, 4.1, 4.2, 4.3, 4.4],
                                           [5.0, 5.1, 5.2, 5.3, 5.4],
                                           [6.0, 6.1, 6.2, 6.3, 6.4]],
                                          [[7.0, 7.1, 7.2, 7.3, 7.4],
                                           [8.0, 8.1, 8.2, 8.3, 8.4],
                                           [9.0, 9.1, 9.2, 9.3, 9.4]],
                                          [[10.0, 10.1, 10.2, 10.3, 10.4],
                                           [11.0, 11.1, 11.2, 11.3, 11.4],
                                           [12.0, 12.1, 12.2, 12.3, 12.4]]])
        choices = torch.tensor([[0*5+1,0*5+3,2*5+2],
                                [1*5+0,2*5+3,0*5+3],
                                [2*5+1,2*5+2,2*5+0],
                                [0*5+1,1*5+2,2*5+3]])
        probs_correct = torch.tensor([[1.1,1.3,3.2],
                                      [5.0,6.3,4.3],
                                      [9.1,9.2,9.0],
                                      [10.1,11.2,12.3]])
        
        bm.all_choices_probs = all_choices_probs
        bm.select_idxs_dependent(choices)
        probs_out = bm.probs
        self.assertTrue(torch.equal(probs_out, probs_correct))
