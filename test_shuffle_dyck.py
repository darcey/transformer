# Integration tests involving the Shuffle-Dyck languages

import torch
import unittest
from shuffle_dyck import *



class TestShuffleDyckHardCoded(unittest.TestCase):

    def setUp(self):
        self.sdr = ShuffleDyckRecognizer(3)

    def construct_input(self, seq):
        return torch.nn.functional.one_hot(seq, num_classes=6).float().unsqueeze(0)

    def testInLanguage(self):
        seq = torch.tensor([0,3,1,4])
        self.assertTrue(self.sdr.recognize(self.construct_input(seq)))
        seq = torch.tensor([0,1,3,4])
        self.assertTrue(self.sdr.recognize(self.construct_input(seq)))

    def testOpenParenNotClosed(self):
        seq = torch.tensor([0,0,3,1,4])
        self.assertFalse(self.sdr.recognize(self.construct_input(seq)))

    def testUnmatchedCloseParen(self):
        seq = torch.tensor([0,3,1,5,4])
        self.assertFalse(self.sdr.recognize(self.construct_input(seq)))

    def testOutOfOrder(self):
        seq = torch.tensor([0,3,1,4,4,1])
        self.assertFalse(self.sdr.recognize(self.construct_input(seq)))

    def testManyProblemsAtOnce(self):
        seq = torch.tensor([0,0,3,1,4,4,5,1])
        self.assertFalse(self.sdr.recognize(self.construct_input(seq)))
