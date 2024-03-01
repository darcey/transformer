import torch
import unittest
from distances import *

        

class TestIdentity(unittest.TestCase):

    def testGetDistanceEstimates(self):
        batch_size = 2
        beam_size = 3
        ball_size = 2
        vocab_size = 5
        seq_len = 2

        class MockBeamManager:
            def __init__(self):
                self.symbols = torch.tensor([[[[0, 1],
                                               [2, 3],
                                               [0, 1]],
                                              [[1, 2],
                                               [2, 3],
                                               [0, 1]],
                                              [[1, 2],
                                               [1, 2],
                                               [1, 2]]],
                                             [[[3, 2],
                                               [2, 2],
                                               [1, 2]],
                                              [[1, 1],
                                               [2, 1],
                                               [1, 1]],
                                              [[3, 0],
                                               [0, 3],
                                               [0, 3]]]])
        beam_manager = MockBeamManager()

        same_symbols = torch.tensor([[[[[True, False, False, False, False],    # [0,1,0] vs. [0, 1, ?]
                                        [False, False, False, False, False],   # [0,1,0] vs. [2, 3, ?]
                                        [True, False, False, False, False]],   # [0,1,0] vs. [0, 1, ?]
                                       [[False, True, False, False, False],    # [0,1,1] vs, [0, 1, ?]
                                        [False, False, False, False, False],   # [0,1,1] vs. [2, 3, ?]
                                        [False, True, False, False, False]],   # [0,1,1] vs. [0, 1, ?]
                                       [[False, False, True, False, False],
                                        [False, False, False, False, False],
                                        [False, False, True, False, False]],
                                       [[False, False, False, True, False],
                                        [False, False, False, False, False],
                                        [False, False, False, True, False]],
                                       [[False, False, False, False, True],
                                        [False, False, False, False, False],
                                        [False, False, False, False, True]]],
                                      [[[True, False, False, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, True, False, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, True, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, False, True, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, False, False, True],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]]],
                                      [[[True, False, False, False, False],
                                        [True, False, False, False, False],
                                        [True, False, False, False, False]],
                                       [[False, True, False, False, False],
                                        [False, True, False, False, False],
                                        [False, True, False, False, False]],
                                       [[False, False, True, False, False],
                                        [False, False, True, False, False],
                                        [False, False, True, False, False]],
                                       [[False, False, False, True, False],
                                        [False, False, False, True, False],
                                        [False, False, False, True, False]],
                                       [[False, False, False, False, True],
                                        [False, False, False, False, True],
                                        [False, False, False, False, True]]]],
                                     [[[[True, False, False, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, True, False, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, True, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, False, True, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, False, False, True],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]]],
                                      [[[True, False, False, False, False],
                                        [False, False, False, False, False],
                                        [True, False, False, False, False]],
                                       [[False, True, False, False, False],
                                        [False, False, False, False, False],
                                        [False, True, False, False, False]],
                                       [[False, False, True, False, False],
                                        [False, False, False, False, False],
                                        [False, False, True, False, False]],
                                       [[False, False, False, True, False],
                                        [False, False, False, False, False],
                                        [False, False, False, True, False]],
                                       [[False, False, False, False, True],
                                        [False, False, False, False, False],
                                        [False, False, False, False, True]]],
                                      [[[True, False, False, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, True, False, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, True, False, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, False, True, False],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]],
                                       [[False, False, False, False, True],
                                        [False, False, False, False, False],
                                        [False, False, False, False, False]]]]])
        dists_correct = torch.full((batch_size, beam_size, vocab_size, ball_size+1, vocab_size), float("inf"))
        dists_correct[same_symbols] = 0.0

        identity = Identity(batch_size, beam_size, ball_size, vocab_size, beam_manager)
        dists_out = identity.get_distance_estimates()
        self.assertTrue(torch.equal(dists_out, dists_correct))
