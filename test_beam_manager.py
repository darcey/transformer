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
                         ball_size=4,
                         vocab_size=4,
                         max_lengths=torch.tensor([3,7]),
                         max_possible_length=6,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")
        symbols_correct = torch.tensor([[[[1],[1],[1],[1],[1]],
                                         [[1],[1],[1],[1],[1]],
                                         [[1],[1],[1],[1],[1]]],
                                        [[[1],[1],[1],[1],[1]],
                                         [[1],[1],[1],[1],[1]],
                                         [[1],[1],[1],[1],[1]]]])
        probs_correct = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]],
                                      [[0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, 0.0]]])
        self.assertTrue(torch.equal(bm.symbols, symbols_correct))
        self.assertTrue(torch.equal(bm.probs, probs_correct))

    # For a given batch item, if all beam items end in EOS or PAD<
    # batch item should be pruned, regardless of what the ball items are.
    def testPruneFinishedEOSorPAD(self):
        bm = BeamManager(batch_size=4,
                         beam_size=3,
                         ball_size=3,
                         vocab_size=4,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        symbols = torch.tensor([[[[1,3,3,3], # example 1: nothing finished in beam or ball
                                  [1,3,3,3],
                                  [1,3,3,3]],
                                 [[1,3,3,3],
                                  [1,3,3,3],
                                  [1,3,3,3]],
                                 [[1,3,3,3],
                                  [1,3,3,3],
                                  [1,3,3,3]]],
                                [[[1,3,2,0], # example 2: some beam/ball items finished
                                  [1,3,3,2],
                                  [1,3,2,0]],
                                 [[1,3,3,2],
                                  [1,3,3,3],
                                  [1,3,2,0]],
                                 [[1,3,3,3],
                                  [1,3,2,0],
                                  [1,3,3,2]]],
                                [[[1,3,2,0], # example 3: all beam items finished, some ball items
                                  [1,3,3,2],
                                  [1,3,2,0]],
                                 [[1,3,3,2],
                                  [1,3,3,3],
                                  [1,3,2,0]],
                                 [[1,3,3,2],
                                  [1,3,3,3],
                                  [1,2,0,0]]],
                                [[[1,3,2,0], # example 4: everything finished
                                  [1,2,0,0],
                                  [1,3,2,0]],
                                 [[1,3,3,2],
                                  [1,3,2,0],
                                  [1,3,2,0]],
                                 [[1,3,3,2],
                                  [1,3,2,0],
                                  [1,2,0,0]]]])
        bm.symbols = symbols
        symbols_correct = torch.tensor([[[[1,3,3,3], # example 1: nothing finished in beam or ball
                                          [1,3,3,3],
                                          [1,3,3,3]],
                                         [[1,3,3,3],
                                          [1,3,3,3],
                                          [1,3,3,3]],
                                         [[1,3,3,3],
                                          [1,3,3,3],
                                          [1,3,3,3]]],
                                        [[[1,3,2,0], # example 2: some beam/ball items finished
                                          [1,3,3,2],
                                          [1,3,2,0]],
                                         [[1,3,3,2],
                                          [1,3,3,3],
                                          [1,3,2,0]],
                                         [[1,3,3,3],
                                          [1,3,2,0],
                                          [1,3,3,2]]]])
        bm.prune_finished()
        symbols_out = bm.symbols
        self.assertTrue(torch.equal(symbols_out, symbols_correct))

    def testSelectIdxCluster(self):
        bm = BeamManager(batch_size=2,
                         beam_size=3,
                         ball_size=2,
                         vocab_size=6,
                         max_lengths=torch.tensor([10,10,10,10]),
                         max_possible_length=10,
                         pad=0,
                         bos=1,
                         eos=2,
                         autoregressive_fn = mock_auto_fn_does_nothing,
                         cache = MockCacheDoesNothing(),
                         device = "cpu")

        symbols = torch.tensor([[[[1,3,3,3],
                                  [1,3,3,4],
                                  [1,3,3,5]],
                                 [[1,3,4,3],
                                  [1,3,4,4],
                                  [1,3,4,5]],
                                 [[1,3,5,3],
                                  [1,3,5,4],
                                  [1,3,5,5]]],
                                [[[1,4,3,3],
                                  [1,4,3,4],
                                  [1,4,3,5]],
                                 [[1,4,4,3],
                                  [1,4,4,4],
                                  [1,4,4,5]],
                                 [[1,4,5,3],
                                  [1,4,5,4],
                                  [1,4,5,5]]]])
        probs = torch.tensor([[[[  0,   1,   2,   3,   4,   5],
                                [  6,   7,   8,   9,  10,  11],
                                [ 12,  13,  14,  15,  16,  17]],
                               [[ 18,  19,  20,  21,  22,  23],
                                [ 24,  25,  26,  27,  28,  29],
                                [ 30,  31,  32,  33,  34,  35]],
                               [[ 36,  37,  38,  39,  40,  41],
                                [ 42,  43,  44,  45,  46,  47],
                                [ 48,  49,  50,  51,  52,  53]]],
                              [[[ 54,  55,  56,  57,  58,  59],
                                [ 60,  61,  62,  63,  64,  65],
                                [ 66,  67,  68,  69,  70,  71]],
                               [[ 72,  73,  74,  75,  76,  77],
                                [ 78,  79,  80,  81,  82,  83],
                                [ 84,  85,  86,  87,  88,  89]],
                               [[ 90,  91,  92,  93,  94,  95],
                                [ 96,  97,  98,  99, 100, 101],
                                [102, 103, 104, 105, 106, 107]]]])
        prev_finished = torch.tensor([[[True, False, True],
                                       [False, False, True],
                                       [False, False, False]],
                                      [[False, True, False],
                                       [False, False, True],
                                       [True, True, True]]])

        chosen_idxs_ball = torch.tensor([[[[0*6+0,0*6+3,1*6+4],    # ball for [1,3,3,3,0]: [[1,3,3,3,0], [1,3,3,3,3], [1,3,3,4,4]] [0, 3, 10]     [True, True, False]
                                           [0*6+1,2*6+4,1*6+5],    # ball for [1,3,3,3,1]: [[1,3,3,3,1], [1,3,3,5,4], [1,3,3,4,5]] [1, 16, 11]    [True, True, False]
                                           [0*6+2,1*6+2,2*6+2],    # ball for [1,3,3,3,2]: [[1,3,3,3,2], [1,3,3,4,2], [1,3,3,5,2]] [2, 8, 14]     [True, False, True]
                                           [0*6+3,1*6+3,2*6+3],    # ball for [1,3,3,3,3]: [[1,3,3,3,3], [1,3,3,4,3], [1,3,3,5,3]] [3, 9, 15]     [True, False, True]
                                           [0*6+4,2*6+4,1*6+4],    # ball for [1,3,3,3,4]: [[1,3,3,3,4], [1,3,3,5,4], [1,3,3,4,4]] [4, 16, 10]    [True, True, False]
                                           [0*6+5,1*6+5,2*6+3]],   # ball for [1,3,3,3,5]: [[1,3,3,3,5], [1,3,3,4,5], [1,3,3,5,3]] [5, 11, 17]    [True, False, True]
                                          [[0*6+0,2*6+2,1*6+5],    # ball for [1,3,4,3,0]: [[1,3,4,3,0], [1,3,4,5,2], [1,3,4,4,5]] [18, 32, 29]   [False, True, True]
                                           [0*6+1,1*6+3,0*6+2],    # ball for [1,3,4,3,1]: [[1,3,4,3,1], [1,3,4,4,3], [1,3,4,3,2]] [19, 27, 20]   [False, False, False]
                                           [0*6+2,0*6+3,0*6+5],    # ball for [1,3,4,3,2]: [[1,3,4,3,2], [1,3,4,3,3], [1,3,4,3,5]] [20, 21, 23]   [False, False, False]
                                           [0*6+3,0*6+5,2*6+2],    # ball for [1,3,4,3,3]: [[1,3,4,3,3], [1,3,4,3,5], [1,3,4,5,2]] [21, 23, 32]   [False, False, True]
                                           [0*6+4,1*6+3,1*6+4],    # ball for [1,3,4,3,4]: [[1,3,4,3,4], [1,3,4,4,3], [1,3,4,4,4]] [22, 27, 28]   [False, False, False]
                                           [0*6+5,1*6+5,2*6+5]],   # ball for [1,3,4,3,5]: [[1,3,4,3,5], [1,3,4,4,5], [1,3,4,5,5]] [23, 29, 35]   [False, False, True]
                                          [[0*6+0,0*6+5,1*6+4],    # ball for [1,3,5,3,0]: [[1,3,5,3,0], [1,3,5,3,5], [1,3,5,4,4]] [36, 41, 46]   [False, False, False]
                                           [0*6+1,1*6+2,1*6+3],    # ball for [1,3,5,3,1]: [[1,3,5,3,1], [1,3,5,4,2], [1,3,5,4,3]] [37, 44, 45]   [False, False, False]
                                           [0*6+2,2*6+5,2*6+2],    # ball for [1,3,5,3,2]: [[1,3,5,3,2], [1,3,5,5,5], [1,3,5,5,2]] [38, 53, 50]   [False, False, False]
                                           [0*6+3,2*6+4,1*6+3],    # ball for [1,3,5,3,3]: [[1,3,5,3,3], [1,3,5,5,4], [1,3,5,4,3]] [39, 52, 45]   [False, False, False]
                                           [0*6+4,0*6+2,0*6+3],    # ball for [1,3,5,3,4]: [[1,3,5,3,4], [1,3,5,3,2], [1,3,5,3,3]] [40, 38, 39]   [False, False, False]
                                           [0*6+5,2*6+3,0*6+4]]],  # ball for [1,3,5,3,5]: [[1,3,5,3,5], [1,3,5,5,3], [1,3,5,3,4]] [41, 51, 40]   [False, False, False]
                                         [[[0*6+0,2*6+2,1*6+2],    # ball for [1,4,3,3,0]: [[1,4,3,3,0], [1,4,3,5,2], [1,4,3,4,2]] [54, 68, 62]   [False, False, True]
                                           [0*6+1,2*6+3,0*6+2],    # ball for [1,4,3,3,1]: [[1,4,3,3,1], [1,4,3,5,3], [1,4,3,3,2]] [55, 69, 56]   [False, False, False]
                                           [0*6+2,0*6+5,1*6+2],    # ball for [1,4,3,3,2]: [[1,4,3,3,2], [1,4,3,3,5], [1,4,3,4,2]] [56, 59, 62]   [False, False, True]
                                           [0*6+3,2*6+4,0*6+5],    # ball for [1,4,3,3,3]: [[1,4,3,3,3], [1,4,3,5,4], [1,4,3,3,5]] [57, 70, 59]   [False, False, False]
                                           [0*6+4,0*6+2,0*6+3],    # ball for [1,4,3,3,4]: [[1,4,3,3,4], [1,4,3,3,2], [1,4,3,3,3]] [58, 56, 57]   [False, False, False]
                                           [0*6+5,2*6+5,1*6+4]],   # ball for [1,4,3,3,5]: [[1,4,3,3,5], [1,4,3,5,5], [1,4,3,4,4]] [59, 71, 64]   [False, False, True]
                                          [[0*6+0,1*6+3,2*6+3],    # ball for [1,4,4,3,0]: [[1,4,4,3,0], [1,4,4,4,3], [1,4,4,5,3]] [72, 81, 87]   [False, False, True]
                                           [0*6+1,0*6+2,1*6+3],    # ball for [1,4,4,3,1]: [[1,4,4,3,1], [1,4,4,3,2], [1,4,4,4,3]] [73, 74, 81]   [False, False, False]
                                           [0*6+2,2*6+4,1*6+4],    # ball for [1,4,4,3,2]: [[1,4,4,3,2], [1,4,4,5,4], [1,4,4,4,4]] [74, 88, 82]   [False, True, False]
                                           [0*6+3,1*6+4,1*6+3],    # ball for [1,4,4,3,3]: [[1,4,4,3,3], [1,4,4,4,4], [1,4,4,4,3]] [75, 82, 81]   [False, False, False]
                                           [0*6+4,2*6+2,0*6+3],    # ball for [1,4,4,3,4]: [[1,4,4,3,4], [1,4,4,5,2], [1,4,4,3,3]] [76, 86, 75]   [False, True, False]
                                           [0*6+5,0*6+4,1*6+3]],   # ball for [1,4,4,3,5]: [[1,4,4,3,5], [1,4,4,3,4], [1,4,4,4,3]] [77, 76, 81]   [False, False, False]
                                          [[0*6+0,2*6+3,0*6+3],    # ball for [1,4,5,3,0]: [[1,4,5,3,0], [1,4,5,5,3], [1,4,5,3,3]] [90, 105, 93]  [True, True, True]
                                           [0*6+1,1*6+4,2*6+2],    # ball for [1,4,5,3,1]: [[1,4,5,3,1], [1,4,5,4,4], [1,4,5,5,2]] [91, 100, 104] [True, True, True]
                                           [0*6+2,0*6+5,1*6+4],    # ball for [1,4,5,3,2]: [[1,4,5,3,2], [1,4,5,3,5], [1,4,5,4,4]] [92, 95, 100]  [True, True, True]
                                           [0*6+3,0*6+4,2*6+3],    # ball for [1,4,5,3,3]: [[1,4,5,3,3], [1,4,5,3,4], [1,4,5,5,3]] [93, 94, 105]  [True, True, True]
                                           [0*6+4,2*6+4,2*6+3],    # ball for [1,4,5,3,4]: [[1,4,5,3,4], [1,4,5,5,4], [1,4,5,5,3]] [94, 106, 105] [True, True, True]
                                           [0*6+5,1*6+4,1*6+2]]]]) # ball for [1,4,5,3,5]: [[1,4,5,3,5], [1,4,5,4,4], [1,4,5,4,2]] [95, 100, 98]  [True, True, True]
        chosen_idxs_beam = torch.tensor([[0*6+2,2*6+5,1*6+3],
                                         [1*6+3,1*6+5,2*6+4]])

        symbols_correct = torch.tensor([[[[1,3,3,3,2], [1,3,3,4,2], [1,3,3,5,2]],
                                         [[1,3,5,3,5], [1,3,5,5,3], [1,3,5,3,4]],
                                         [[1,3,4,3,3], [1,3,4,3,5], [1,3,4,5,2]]],
                                        [[[1,4,4,3,3], [1,4,4,4,4], [1,4,4,4,3]],
                                         [[1,4,4,3,5], [1,4,4,3,4], [1,4,4,4,3]],
                                         [[1,4,5,3,4], [1,4,5,5,4], [1,4,5,5,3]]]])
        probs_correct = torch.tensor([[[2, 8, 14], [41, 51, 40], [21, 23, 32]],
                                      [[75, 82, 81], [77, 76, 81], [94, 106, 105]]])
        prev_finished_correct = torch.tensor([[[True, False, True],
                                               [False, False, False],
                                               [False, False, True]],
                                              [[False, False, False],
                                               [False, False, False],
                                               [True, True, True]]])

        bm.seq_len = 4
        bm.symbols = symbols
        bm.all_choices_probs = probs
        bm.prev_finished = prev_finished
        bm.select_idxs_cluster(chosen_idxs_ball, chosen_idxs_beam)
        #print(bm.symbols)
        self.assertTrue(torch.equal(bm.symbols, symbols_correct))
        self.assertTrue(torch.equal(bm.probs, probs_correct))
        self.assertTrue(torch.equal(bm.prev_finished, prev_finished_correct))



class TestBeamManagerNoBall(unittest.TestCase):

    def testInit(self):
        bm = BeamManagerNoBall(batch_size=2,
                               beam_size=3,
                               vocab_size=4,
                               max_lengths=torch.tensor([3,7]),
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
        self.assertTrue(torch.equal(bm.get_symbols(), symbols_correct))
        self.assertTrue(torch.equal(bm.get_probs(), probs_correct))

    # Check that it correctly prunes sentences where everything
    # in the beam ends with EOS or PAD, but if only some items
    # in the beam end with EOS or PAD, it leaves that sentence alone.
    def testPruneFinishedEOSorPAD(self):
        bm = BeamManagerNoBall(batch_size=4,
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

        symbols = torch.tensor([[[1,3,3,3],
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
        bm.set_symbols(symbols)
        symbols_correct = torch.tensor([[[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,3]],
                                        [[1,3,3,3],
                                         [1,3,3,0],
                                         [1,3,3,2]]])
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, symbols_correct))

    def testPruneFinishedMaxLengths(self):
        bm = BeamManagerNoBall(batch_size=4,
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
        my_symbols = torch.tensor([[[1,3,3,3],
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
        bm.set_symbols(my_symbols)
        symbols_correct = torch.tensor([[[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,4]],
                                        [[1,3,3,4],
                                         [1,3,3,4],
                                         [1,3,3,4]]])
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, symbols_correct))
    
    def testPruneFinishedMaxPossibleLength(self):
        bm = BeamManagerNoBall(batch_size=4,
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
        my_symbols = torch.tensor([[[1,3,3,3],
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
        bm.set_symbols(my_symbols)
        symbols_correct = torch.empty(0,3,4)
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, symbols_correct))

    def testPruneFinishedPrevFinished(self):
        bm = BeamManagerNoBall(batch_size=4,
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

        prev_finished = torch.tensor([[[False],[False],[False]],
                                      [[True],[False],[True]],
                                      [[True],[False],[False]],
                                      [[True],[True],[True]]])

        my_symbols = torch.tensor([[[1,3,3,2],  # ex1: some finished, no prev finished
                                    [1,3,2,0],
                                    [1,3,3,3]],
                                   [[1,3,3,2],  # ex2: some finished, same ones prev finished
                                    [1,3,3,3],
                                    [1,3,2,0]],
                                   [[1,3,3,3],  # ex3: some finished, rest prev_finished
                                    [1,3,2,0],
                                    [1,2,0,0]],
                                   [[1,3,3,3],  # ex4: none finished, all prev_finished
                                    [1,3,2,3],
                                    [1,2,3,3]]])

        symbols_correct = torch.tensor([[[1,3,3,2],  # ex1: some finished, no prev finished
                                         [1,3,2,0],
                                         [1,3,3,3]],
                                        [[1,3,3,2],  # ex2: some finished, same ones prev finished
                                         [1,3,3,3],
                                         [1,3,2,0]]])

        bm.prev_finished = prev_finished
        bm.set_symbols(my_symbols)
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, symbols_correct))

    def testPruneFinishedIterated(self):
        bm = BeamManagerNoBall(batch_size=4,
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
        bm.set_probs(probs_correct)

        # Iteration 1
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, symbols_correct[:,:,:1]))
        self.assertFalse(bm.all_done())

        # Iteration 2
        bm.set_symbols(torch.cat((bm.get_symbols(), symbols_correct[:,:,1].unsqueeze(-1)), dim=-1))
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, symbols_correct[:,:,:2]))
        self.assertFalse(bm.all_done())

        # Iteration 3
        bm.set_symbols(torch.cat((bm.get_symbols(), symbols_correct[:,:,2].unsqueeze(-1)), dim=-1))
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        pruned_correct = torch.tensor([True, False, False, False])
        self.assertTrue(torch.equal(symbols_out, symbols_correct[~pruned_correct][:,:,:3]))
        self.assertFalse(bm.all_done())

        # Iteration 4
        bm.set_symbols(torch.cat((bm.get_symbols(), symbols_correct[~pruned_correct][:,:,3].unsqueeze(-1)), dim=-1))
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        pruned_correct = torch.tensor([True, False, True, False])
        self.assertTrue(torch.equal(symbols_out, symbols_correct[~pruned_correct][:,:,:4]))
        self.assertFalse(bm.all_done())

        # Iteration 5
        bm.set_symbols(torch.cat((bm.get_symbols(), symbols_correct[~pruned_correct][:,:,4].unsqueeze(-1)), dim=-1))
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        pruned_correct = torch.tensor([True, False, True, True])
        self.assertTrue(torch.equal(symbols_out, symbols_correct[~pruned_correct][:,:,:5]))
        self.assertFalse(bm.all_done())

        # Iteration 5
        bm.set_symbols(torch.cat((bm.get_symbols(), symbols_correct[~pruned_correct][:,:,5].unsqueeze(-1)), dim=-1))
        bm.seq_len += 1
        bm.prune_finished()
        symbols_out = bm.get_symbols()
        self.assertTrue(torch.equal(symbols_out, torch.empty(0,3,6)))
        self.assertTrue(bm.all_done())

        symbols_out, probs_out = bm.get_final()
        self.assertTrue(torch.equal(symbols_out, symbols_correct))
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testGetAllChoicesLengths(self):
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

        bm = BeamManagerNoBall(batch_size=4,
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
        bm.set_symbols(symbols)
        bm.seq_len = 4

        lengths_correct = torch.tensor([[[3,3,3,3,3],
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
        lengths_out = bm.get_all_choices_lengths()
        self.assertTrue(torch.equal(lengths_out, lengths_correct))

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

        bm = BeamManagerNoBall(batch_size=4,
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
        bm.set_symbols(symbols)
        bm.set_probs(torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
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

        bm = BeamManagerNoBall(batch_size=4,
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
        bm.set_symbols(symbols)
        bm.set_probs(torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
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

    # This test is to ensure that prev_finished prevents the code
    # from crashing if EOS or PAD is for some reason followed by a
    # normal token.
    def testComputeNextTokenProbsPrevFinished(self):
        symbols = torch.tensor([[[1,3,3,2,0],
                                 [1,3,2,0,3],
                                 [1,3,3,3,3]],
                                [[1,3,3,3,2],
                                 [1,3,3,2,0],
                                 [1,3,3,4,4]],
                                [[1,3,2,0,3],
                                 [1,3,3,2,4],
                                 [1,3,3,4,0]],
                                [[1,3,3,4,2],
                                 [1,3,3,4,3],
                                 [1,3,3,2,0]]])
        active_correct_1 = torch.tensor([[1,3,3,3],
                                         [1,3,3,3],
                                         [1,3,3,4],
                                         [1,3,3,4],
                                         [1,3,3,4],
                                         [1,3,3,4]])
        active_correct_2 = torch.tensor([[1,3,3,3,3],
                                         [1,3,3,4,4],
                                         [1,3,3,4,3]])

        def mock_auto_fn(active_symbols, timestep, cache):
            if timestep == 3:
                self.assertTrue(torch.equal(active_symbols, active_correct_1[:,3:3+1]))
            else:
                self.assertTrue(torch.equal(active_symbols, active_correct_2[:,4:4+1]))

            batch_beam = active_symbols.size(0)
            return torch.rand(size=[batch_beam,1,5]), torch.rand(size=[batch_beam,1,5])

        bm = BeamManagerNoBall(batch_size=4,
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

        bm.set_symbols(symbols[:,:,0:4])
        bm.seq_len = 4
        _, next_token_probs_out, all_choices_probs_out = bm.compute_next_token_probs()

        bm.set_symbols(symbols)
        bm.seq_len = 5
        _, next_token_probs_out, all_choices_probs_out = bm.compute_next_token_probs()

    def testSelectIdxSampling(self):
        bm = BeamManagerNoBall(batch_size=4,
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
        
        bm.all_choices_probs = all_choices_probs.unsqueeze(2)
        bm.select_idxs_sampling(choices)
        probs_out = bm.get_probs()
        self.assertTrue(torch.equal(probs_out, probs_correct))

    def testSelectIdxsBeam(self):
        bm = BeamManagerNoBall(batch_size=4,
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
        
        bm.all_choices_probs = all_choices_probs.unsqueeze(2)
        bm.select_idxs_beam(choices)
        probs_out = bm.get_probs()
        self.assertTrue(torch.equal(probs_out, probs_correct))
