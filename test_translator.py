import torch
import unittest
from configuration import *
from dataset import Seq2SeqTranslateBatch
from translator import *



class MockGenerator:

    def __init__(self, model, config):
        return

    def generate(self, src):
        tgt_all = src.clone().unsqueeze(1)
        return tgt_all

class MockDataset:

    def __init__(self):
        orig_idxs = list(range(10 * 30))
        self.batches = []
        for i in range(10):
            src = torch.randint(low=1, high=100, size=(30, 8))
            batch = Seq2SeqTranslateBatch(src, orig_idxs[10*i : 10*(i+1)])
            self.batches.append(batch)



class TestTranslator(unittest.TestCase):

    def setUp(self):
        config = read_config("configuration.toml")
        model = torch.nn.Linear(5,6)
        generator = MockGenerator(model, config)
        self.translator = Translator(model, generator, "cpu")

        self.data = MockDataset()

    def testPrintAtEnd(self):
        outputs = []
        for translations in self.translator.translate(self.data, print_every=0):
            outputs.append(translations)

        self.assertEqual(len(outputs), 1)
        for src_batch, tgt_batch in zip(self.data.batches, outputs[0]):
            self.assertTrue(torch.equal(src_batch.src.unsqueeze(1), tgt_batch.tgt_all))

    def testIntermittentPrinting(self):
        outputs = []
        flat_outputs = []
        for translations in self.translator.translate(self.data, print_every=40):
            outputs.append(translations)
            flat_outputs.extend(translations)

        self.assertEqual(len(outputs), 5)
        for src_batch, tgt_batch in zip(self.data.batches, flat_outputs):
            self.assertTrue(torch.equal(src_batch.src.unsqueeze(1), tgt_batch.tgt_all))
