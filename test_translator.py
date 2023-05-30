import torch
import unittest
from configuration import *
from dataset import Seq2SeqTranslateBatch
from translator import *



class MockGenerator:

    def __init__(self, model, config):
        return

    def generate(self, src):
        tgt_final = src.clone()
        tgt_all = src.clone().unsqueeze(1)
        probs_all = torch.rand(src.size(0), 1)
        return tgt_final, tgt_all, probs_all

class MockDataset:

    def __init__(self):
        orig_idxs = list(range(10 * 30))
        self.batches = []
        for i in range(10):
            src = torch.randint(low=1, high=100, size=(30, 8))
            batch = Seq2SeqTranslateBatch(src, orig_idxs[10*i : 10*(i+1)])
            self.batches.append(batch)

    def __len__(self):
        return len(self.batches)

    def get_empty_tgt_dataset(self):
        empty = MockDataset()
        empty.batches = []
        return empty

    def add_batch(self, batch):
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
        for translations in self.translator.translate(self.data, yield_interval=0):
            outputs.append(translations)

        self.assertEqual(len(outputs), 1)
        for src_batch, tgt_batch in zip(self.data.batches, outputs[0].batches):
            self.assertTrue(torch.equal(src_batch.src, tgt_batch.tgt_final))
            self.assertTrue(torch.equal(src_batch.src.unsqueeze(1), tgt_batch.tgt_all))

    def testIntermittentPrinting(self):
        outputs = []
        for translations in self.translator.translate(self.data, yield_interval=40):
            outputs.append(translations)
        flat_outputs = [batch for dataset in outputs for batch in dataset.batches]

        self.assertEqual(len(outputs), 5)
        for src_batch, tgt_batch in zip(self.data.batches, flat_outputs):
            self.assertTrue(torch.equal(src_batch.src, tgt_batch.tgt_final))
            self.assertTrue(torch.equal(src_batch.src.unsqueeze(1), tgt_batch.tgt_all))
