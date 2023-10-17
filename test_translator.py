import random
import torch
import unittest
from configuration import *
from dataset import Seq2SeqTranslateDataset
from translator import *



class MockGenerator:

    def __init__(self, model, config):
        return

    def generate(self, src):
        tgt_final = src.clone()
        tgt_all = src.clone().unsqueeze(1)
        probs_all = torch.rand(src.size(0), 1)
        return tgt_final, tgt_all, probs_all

class TestTranslator(unittest.TestCase):

    def testTranslate(self):
        config = read_config("configuration.toml")
        model = torch.nn.Linear(5,6)
        generator = MockGenerator(model, config)
        translator = Translator(model, generator, "cpu")

        src_sents = []
        for i in range(56):
            src_len = random.randint(6,15)
            src_sent = [random.randint(5,9) for _ in range(src_len)]
            src_sents.append(src_sent)
        ds = Seq2SeqTranslateDataset(src_sents, 5, 15, pad_idx=0, bos_idx=1, eos_idx=2)

        finals = []
        for tgt_final, tgt_all, probs_all in translator.translate(ds):
            finals.extend(tgt_final)

        self.assertEqual(len(finals), 56)
        for src, tgt in zip(src_sents, finals):
            self.assertEqual(src, tgt)
