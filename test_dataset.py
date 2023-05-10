import torch
import unittest
from dataset import *
from vocabulary import *


class TestSeq2SeqTrainDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data([['4','5','6','7','8','9']],[['10','11','12','13','14','15']])

    def testSortByTgtLen(self):
        src = [[1,2,3,4],[1,2,3,4,5],[1,2],[1,2,3]]
        tgt = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15]]
        ds = Seq2SeqTrainDataset(src, tgt, self.vocab, 100)
        
        sorted_src_correct = [[1,2],[1,2,3,4,5],[1,2,3,4],[1,2,3]]
        sorted_tgt_correct = [[10,11,12],[10,11,12,13],[10,11,12,13,14],[10,11,12,13,14,15]]
        sorted_src_actual, sorted_tgt_actual = ds.sort_by_tgt_len(src, tgt)
        self.assertEqual(sorted_src_actual, sorted_src_correct)
        self.assertEqual(sorted_tgt_actual, sorted_tgt_correct)

    def testMakeOneBatch(self):
        src_sents = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7]]
        tgt_sents = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15]]
        sent_list = list(zip(src_sents, tgt_sents))
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, self.vocab, 100)
        
        PAD = self.vocab.tok_to_idx(SpecialTokens.PAD)
        BOS = self.vocab.tok_to_idx(SpecialTokens.BOS)
        EOS = self.vocab.tok_to_idx(SpecialTokens.EOS)
        src_correct = torch.tensor([[5,6,7,8,EOS,PAD],
                                    [5,6,7,8,9,EOS],
                                    [5,6,EOS,PAD,PAD,PAD],
                                    [5,6,7,EOS,PAD,PAD]])
        tgt_in_correct = torch.tensor([[BOS,10,11,12,13,14,PAD],
                                       [BOS,10,11,12,13,PAD,PAD],
                                       [BOS,10,11,12,PAD,PAD,PAD],
                                       [BOS,10,11,12,13,14,15]])
        tgt_out_correct = torch.tensor([[10,11,12,13,14,EOS,PAD],
                                        [10,11,12,13,EOS,PAD,PAD],
                                        [10,11,12,EOS,PAD,PAD,PAD],
                                        [10,11,12,13,14,15,EOS]])
        actual = ds.make_one_batch(sent_list)
        self.assertTrue(torch.equal(actual["src"], src_correct))
        self.assertTrue(torch.equal(actual["tgt_in"], tgt_in_correct))
        self.assertTrue(torch.equal(actual["tgt_out"], tgt_out_correct))
        self.assertEqual(actual["num_src_toks"], 18)
        self.assertEqual(actual["num_tgt_toks"], 22)

    def testMakeBatches(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7,8,9],[5,6,7,8],[5,6,7]]
        tgt_sents = [[10,11,12],[10,11,12,13],[10,11,12,13],[10,11,12,13,14],[10,11,12,13,14,15]]
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, self.vocab, 8)

        PAD = self.vocab.tok_to_idx(SpecialTokens.PAD)
        BOS = self.vocab.tok_to_idx(SpecialTokens.BOS)
        EOS = self.vocab.tok_to_idx(SpecialTokens.EOS)
        src_correct_1 = torch.tensor([[5,6,EOS,PAD,PAD],[5,6,8,9,EOS]])
        src_correct_2 = torch.tensor([[5,6,7,8,9,EOS],[5,6,7,8,EOS,PAD]])
        src_correct_3 = torch.tensor([[5,6,7,EOS]])
        tgt_in_correct_1 = torch.tensor([[BOS,10,11,12,PAD],[BOS,10,11,12,13]])
        tgt_in_correct_2 = torch.tensor([[BOS,10,11,12,13,PAD],[BOS,10,11,12,13,14]])
        tgt_in_correct_3 = torch.tensor([[BOS,10,11,12,13,14,15]])
        tgt_out_correct_1 = torch.tensor([[10,11,12,EOS,PAD],[10,11,12,13,EOS]])
        tgt_out_correct_2 = torch.tensor([[10,11,12,13,EOS,PAD],[10,11,12,13,14,EOS]])
        tgt_out_correct_3 = torch.tensor([[10,11,12,13,14,15,EOS]])

        actual = ds.make_batches(src_sents, tgt_sents, 8)

        self.assertEqual(len(actual), 3)
        self.assertTrue(torch.equal(actual[0]["src"], src_correct_1))
        self.assertTrue(torch.equal(actual[0]["tgt_in"], tgt_in_correct_1))
        self.assertTrue(torch.equal(actual[0]["tgt_out"], tgt_out_correct_1))
        self.assertEqual(actual[0]["num_src_toks"], 8)
        self.assertEqual(actual[0]["num_tgt_toks"], 9)
        self.assertTrue(torch.equal(actual[1]["src"], src_correct_2))
        self.assertTrue(torch.equal(actual[1]["tgt_in"], tgt_in_correct_2))
        self.assertTrue(torch.equal(actual[1]["tgt_out"], tgt_out_correct_2))
        self.assertEqual(actual[1]["num_src_toks"], 11)
        self.assertEqual(actual[1]["num_tgt_toks"], 11)
        self.assertTrue(torch.equal(actual[2]["src"], src_correct_3))
        self.assertTrue(torch.equal(actual[2]["tgt_in"], tgt_in_correct_3))
        self.assertTrue(torch.equal(actual[2]["tgt_out"], tgt_out_correct_3))
        self.assertEqual(actual[2]["num_src_toks"], 4)
        self.assertEqual(actual[2]["num_tgt_toks"], 7)

    def testInit(self):
        src_sents = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7],[5,6,8,9]]
        tgt_sents = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15],[10,11,12,13]]
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, self.vocab, 8)
        
        PAD = self.vocab.tok_to_idx(SpecialTokens.PAD)
        BOS = self.vocab.tok_to_idx(SpecialTokens.BOS)
        EOS = self.vocab.tok_to_idx(SpecialTokens.EOS)
        src_correct_1 = torch.tensor([[5,6,EOS,PAD,PAD,PAD],[5,6,7,8,9,EOS]])
        src_correct_2 = torch.tensor([[5,6,8,9,EOS],[5,6,7,8,EOS]])
        src_correct_3 = torch.tensor([[5,6,7,EOS]])
        tgt_in_correct_1 = torch.tensor([[BOS,10,11,12,PAD],[BOS,10,11,12,13]])
        tgt_in_correct_2 = torch.tensor([[BOS,10,11,12,13,PAD],[BOS,10,11,12,13,14]])
        tgt_in_correct_3 = torch.tensor([[BOS,10,11,12,13,14,15]])
        tgt_out_correct_1 = torch.tensor([[10,11,12,EOS,PAD],[10,11,12,13,EOS]])
        tgt_out_correct_2 = torch.tensor([[10,11,12,13,EOS,PAD],[10,11,12,13,14,EOS]])
        tgt_out_correct_3 = torch.tensor([[10,11,12,13,14,15,EOS]])

        actual = ds.batches

        self.assertEqual(len(actual), 3)
        self.assertTrue(torch.equal(actual[0]["src"], src_correct_1))
        self.assertTrue(torch.equal(actual[0]["tgt_in"], tgt_in_correct_1))
        self.assertTrue(torch.equal(actual[0]["tgt_out"], tgt_out_correct_1))
        self.assertEqual(actual[0]["num_src_toks"], 9)
        self.assertEqual(actual[0]["num_tgt_toks"], 9)
        self.assertTrue(torch.equal(actual[1]["src"], src_correct_2))
        self.assertTrue(torch.equal(actual[1]["tgt_in"], tgt_in_correct_2))
        self.assertTrue(torch.equal(actual[1]["tgt_out"], tgt_out_correct_2))
        self.assertEqual(actual[1]["num_src_toks"], 10)
        self.assertEqual(actual[1]["num_tgt_toks"], 11)
        self.assertTrue(torch.equal(actual[2]["src"], src_correct_3))
        self.assertTrue(torch.equal(actual[2]["tgt_in"], tgt_in_correct_3))
        self.assertTrue(torch.equal(actual[2]["tgt_out"], tgt_out_correct_3))
        self.assertEqual(actual[2]["num_src_toks"], 4)
        self.assertEqual(actual[2]["num_tgt_toks"], 7)

    def testIterRandomize(self):
        src_sents = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7],[5,6,8,9]]
        tgt_sents = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15],[10,11,12,13]]
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, self.vocab, 8, randomize=True)
        ds.batches = [1, 2, 3, 4]
        
        # test one iteration
        correct_batches = set([1, 2, 3, 4])
        actual_batches = set()
        for i in range(4):
            actual_batches.add(ds.get_batch())
        self.assertTrue(actual_batches, correct_batches)
        
        # test multiple iteration at a time
        correct_batches = set([1, 2, 3, 4])
        actual_batches = set()
        for i in range(20):
            b = ds.get_batch()
            if i > 16:
                actual_batches.add(b)
        self.assertTrue(actual_batches, correct_batches)
        
        # test iteration counting
        self.assertEqual(ds.num_iters, 5)