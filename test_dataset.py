# TODO(darcey): replace real vocab class with mock

import random
import torch
import unittest
from dataset import *
from vocabulary import *



class TestSeq2SeqTrainDataset(unittest.TestCase):

    def testSortByTgtLenToyExample(self):
        src = [[1,2,3,4],[1,2,3,4,5],[1,2],[1,2,3]]
        tgt = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15]]
        ds = Seq2SeqTrainDataset(src, tgt, toks_per_batch=100, pad_idx=0, bos_idx=1, eos_idx=2)
        
        sorted_src_correct = [[1,2],[1,2,3,4,5],[1,2,3,4],[1,2,3]]
        sorted_tgt_correct = [[10,11,12],[10,11,12,13],[10,11,12,13,14],[10,11,12,13,14,15]]
        sorted_src_actual, sorted_tgt_actual = ds.sort_by_tgt_len(src, tgt)
        self.assertEqual(sorted_src_actual, sorted_src_correct)
        self.assertEqual(sorted_tgt_actual, sorted_tgt_correct)

    def testSortByBothLensToyExample(self):
        src = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7],[5,6,7,9]]
        tgt = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15],[10,11,12,14]]
        ds = Seq2SeqTrainDataset(src, tgt, toks_per_batch=100, pad_idx=0, bos_idx=1, eos_idx=2)

        sorted_src_correct = [[5,6],[5,6,7,9],[5,6,7,8,9],[5,6,7,8],[5,6,7]]
        sorted_tgt_correct = [[10,11,12],[10,11,12,14],[10,11,12,13],[10,11,12,13,14],[10,11,12,13,14,15]]
        sorted_src_actual, sorted_tgt_actual = ds.sort_by_both_lens(src, tgt)
        self.assertEqual(sorted_src_actual, sorted_src_correct)
        self.assertEqual(sorted_tgt_actual, sorted_tgt_correct)

    def testSortByBothLensLargeExample(self):
        src_data = []
        tgt_data = []
        for i in range(100):
            src_length = random.randint(6,15)
            tgt_length = random.randint(6,15)
            src_data.append([1]*src_length)
            tgt_data.append([1]*tgt_length)

        ds = Seq2SeqTrainDataset(src_data, tgt_data, toks_per_batch=100, pad_idx=0, bos_idx=1, eos_idx=2)
        sorted_src, sorted_tgt = ds.sort_by_both_lens(src_data, tgt_data)

        tgt_lens = [len(sent) for sent in sorted_tgt]
        self.assertEqual(tgt_lens, sorted(tgt_lens))

        for l in range(6,16):
            src_lens_for_tgt_len_l = [len(sorted_src[i]) for i in range(100) if len(sorted_tgt[i]) == l]
            self.assertEqual(src_lens_for_tgt_len_l, sorted(src_lens_for_tgt_len_l))

    def testMakeOneBatch(self):
        src_sents = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7]]
        tgt_sents = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15]]
        sent_list = list(zip(src_sents, tgt_sents))
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, toks_per_batch=100, pad_idx=0, bos_idx=1, eos_idx=2)
        
        PAD, BOS, EOS = 0, 1, 2
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
        self.assertTrue(torch.equal(actual.src, src_correct))
        self.assertTrue(torch.equal(actual.tgt_in, tgt_in_correct))
        self.assertTrue(torch.equal(actual.tgt_out, tgt_out_correct))
        self.assertEqual(actual.num_src_toks, 18)
        self.assertEqual(actual.num_tgt_toks, 22)

    def testMakeBatches(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        tgt_sents = [[10,11,12],[10,11,12,13],[10,11,12,14],[10,11,12,13,14],[10,11],[10,12,13],[10,11,12,13,14,15]]
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, toks_per_batch=8, pad_idx=0, bos_idx=1, eos_idx=2)

        PAD, BOS, EOS = 0, 1, 2
        src_correct_1 = torch.tensor([[5,6,EOS,PAD,PAD],[5,6,8,9,EOS]])
        src_correct_2 = torch.tensor([[5,6,7,EOS],[7,8,EOS,PAD]])
        src_correct_3 = torch.tensor([[5,6,7,8,9,EOS],[5,6,7,8,EOS,PAD]])
        src_correct_4 = torch.tensor([[5,6,9,EOS]])
        tgt_in_correct_1 = torch.tensor([[BOS,10,11,12,PAD],[BOS,10,11,12,13]])
        tgt_in_correct_2 = torch.tensor([[BOS,10,11,12,14,PAD],[BOS,10,11,12,13,14]])
        tgt_in_correct_3 = torch.tensor([[BOS,10,11,PAD],[BOS,10,12,13]])
        tgt_in_correct_4 = torch.tensor([[BOS,10,11,12,13,14,15]])
        tgt_out_correct_1 = torch.tensor([[10,11,12,EOS,PAD],[10,11,12,13,EOS]])
        tgt_out_correct_2 = torch.tensor([[10,11,12,14,EOS,PAD],[10,11,12,13,14,EOS]])
        tgt_out_correct_3 = torch.tensor([[10,11,EOS,PAD],[10,12,13,EOS]])
        tgt_out_correct_4 = torch.tensor([[10,11,12,13,14,15,EOS]])

        actual = ds.make_batches(src_sents, tgt_sents, toks_per_batch=8)

        self.assertEqual(len(actual), 4)
        # both src and tgt hit batch size
        self.assertTrue(torch.equal(actual[0].src, src_correct_1))
        self.assertTrue(torch.equal(actual[0].tgt_in, tgt_in_correct_1))
        self.assertTrue(torch.equal(actual[0].tgt_out, tgt_out_correct_1))
        self.assertEqual(actual[0].num_src_toks, 8)
        self.assertEqual(actual[0].num_tgt_toks, 9)
        # tgt hits batch size, src doesn't
        self.assertTrue(torch.equal(actual[1].src, src_correct_2))
        self.assertTrue(torch.equal(actual[1].tgt_in, tgt_in_correct_2))
        self.assertTrue(torch.equal(actual[1].tgt_out, tgt_out_correct_2))
        self.assertEqual(actual[1].num_src_toks, 7)
        self.assertEqual(actual[1].num_tgt_toks, 11)
        # src hits batch size, tgt doesn't
        self.assertTrue(torch.equal(actual[2].src, src_correct_3))
        self.assertTrue(torch.equal(actual[2].tgt_in, tgt_in_correct_3))
        self.assertTrue(torch.equal(actual[2].tgt_out, tgt_out_correct_3))
        self.assertEqual(actual[2].num_src_toks, 11)
        self.assertEqual(actual[2].num_tgt_toks, 7)
        # neither hits batch size
        self.assertTrue(torch.equal(actual[3].src, src_correct_4))
        self.assertTrue(torch.equal(actual[3].tgt_in, tgt_in_correct_4))
        self.assertTrue(torch.equal(actual[3].tgt_out, tgt_out_correct_4))
        self.assertEqual(actual[3].num_src_toks, 4)
        self.assertEqual(actual[3].num_tgt_toks, 7)

    def testInit(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        tgt_sents = [[10,11,12],[10,11,12,13],[10,11,12,14],[10,11,12,13,14],[10,11],[10,12,13],[10,11,12,13,14,15]]
        # sorted src: [[5,6,7,8,9],[5,6],[5,6,7,8],[5,6,7],[5,6,8,9],[7,8],[5,6,9]]
        # sorted tgt: [[10,11],[10,11,12],[10,12,13],[10,11,12,14],[10,11,12,13],[10,11,12,13,14],[10,11,12,13,14,15]]
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, toks_per_batch=8, pad_idx=0, bos_idx=1, eos_idx=2)
        
        PAD, BOS, EOS = 0, 1, 2
        src_correct_1 = torch.tensor([[5,6,7,8,9,EOS],[5,6,EOS,PAD,PAD,PAD]])
        src_correct_2 = torch.tensor([[5,6,7,8,EOS],[5,6,7,EOS,PAD]])
        src_correct_3 = torch.tensor([[5,6,8,9,EOS],[7,8,EOS,PAD,PAD]])
        src_correct_4 = torch.tensor([[5,6,9,EOS]])
        tgt_in_correct_1 = torch.tensor([[BOS,10,11,PAD],[BOS,10,11,12]])
        tgt_in_correct_2 = torch.tensor([[BOS,10,12,13,PAD],[BOS,10,11,12,14]])
        tgt_in_correct_3 = torch.tensor([[BOS,10,11,12,13,PAD],[BOS,10,11,12,13,14]])
        tgt_in_correct_4 = torch.tensor([[BOS,10,11,12,13,14,15]])
        tgt_out_correct_1 = torch.tensor([[10,11,EOS,PAD],[10,11,12,EOS]])
        tgt_out_correct_2 = torch.tensor([[10,12,13,EOS,PAD],[10,11,12,14,EOS]])
        tgt_out_correct_3 = torch.tensor([[10,11,12,13,EOS,PAD],[10,11,12,13,14,EOS]])
        tgt_out_correct_4 = torch.tensor([[10,11,12,13,14,15,EOS]])

        actual = ds.batches

        self.assertEqual(len(actual), 4)
        self.assertTrue(torch.equal(actual[0].src, src_correct_1))
        self.assertTrue(torch.equal(actual[0].tgt_in, tgt_in_correct_1))
        self.assertTrue(torch.equal(actual[0].tgt_out, tgt_out_correct_1))
        self.assertEqual(actual[0].num_src_toks, 9)
        self.assertEqual(actual[0].num_tgt_toks, 7)
        self.assertTrue(torch.equal(actual[1].src, src_correct_2))
        self.assertTrue(torch.equal(actual[1].tgt_in, tgt_in_correct_2))
        self.assertTrue(torch.equal(actual[1].tgt_out, tgt_out_correct_2))
        self.assertEqual(actual[1].num_src_toks, 9)
        self.assertEqual(actual[1].num_tgt_toks, 9)
        self.assertTrue(torch.equal(actual[2].src, src_correct_3))
        self.assertTrue(torch.equal(actual[2].tgt_in, tgt_in_correct_3))
        self.assertTrue(torch.equal(actual[2].tgt_out, tgt_out_correct_3))
        self.assertEqual(actual[2].num_src_toks, 8)
        self.assertEqual(actual[2].num_tgt_toks, 11)
        self.assertTrue(torch.equal(actual[3].src, src_correct_4))
        self.assertTrue(torch.equal(actual[3].tgt_in, tgt_in_correct_4))
        self.assertTrue(torch.equal(actual[3].tgt_out, tgt_out_correct_4))
        self.assertEqual(actual[3].num_src_toks, 4)
        self.assertEqual(actual[3].num_tgt_toks, 7)

    def testIterRandomize(self):
        src_sents = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7],[5,6,8,9]]
        tgt_sents = [[10,11,12,13,14],[10,11,12,13],[10,11,12],[10,11,12,13,14,15],[10,11,12,13]]
        ds = Seq2SeqTrainDataset(src_sents, tgt_sents, toks_per_batch=8, pad_idx=0, bos_idx=1, eos_idx=2, randomize=True)
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



class TestSeq2SeqTranslateDataset(unittest.TestCase):

    def testInit(self):
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        self.assertEqual(len(ds), 0)

    def testSortByLen(self):
        src = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7],[5,6,7,9]]
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)

        sorted_src_correct = [[5,6],[5,6,7],[5,6,7,8],[5,6,7,9],[5,6,7,8,9]]
        orig_idxs_correct = [2,4,0,1,3]
        sorted_src_actual, orig_idxs_actual = ds.sort_by_len(src)
        self.assertEqual(sorted_src_actual, sorted_src_correct)
        self.assertEqual(list(orig_idxs_actual), orig_idxs_correct)

    def testMakeOneBatch(self):
        src_sents = [[5,6,7,8],[5,6,7,8,9],[5,6],[5,6,7]]
        orig_idxs = [0,1,2,3]
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)

        PAD, BOS, EOS = 0, 1, 2
        src_correct = torch.tensor([[5,6,7,8,EOS,PAD],
                                    [5,6,7,8,9,EOS],
                                    [5,6,EOS,PAD,PAD,PAD],
                                    [5,6,7,EOS,PAD,PAD]])
        actual = ds.make_one_batch(src_sents, orig_idxs)
        self.assertTrue(torch.equal(actual.src, src_correct))
        self.assertEqual(actual.orig_idxs, orig_idxs)

    def testMakeBatches(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        orig_idxs = [0,1,2,3,4,5,6]
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)

        PAD, BOS, EOS = 0, 1, 2
        src_correct_1 = torch.tensor([[5,6,EOS,PAD,PAD],[5,6,8,9,EOS]])
        src_correct_2 = torch.tensor([[5,6,7,EOS],[7,8,EOS,PAD]])
        src_correct_3 = torch.tensor([[5,6,7,8,9,EOS],[5,6,7,8,EOS,PAD]])
        src_correct_4 = torch.tensor([[5,6,9,EOS]])

        actual = ds.make_batches(src_sents, orig_idxs, sents_per_batch=2)

        self.assertEqual(len(actual), 4)

        self.assertTrue(torch.equal(actual[0].src, src_correct_1))
        self.assertEqual(list(actual[0].orig_idxs), [0,1])

        self.assertTrue(torch.equal(actual[1].src, src_correct_2))
        self.assertEqual(list(actual[1].orig_idxs), [2,3])

        self.assertTrue(torch.equal(actual[2].src, src_correct_3))
        self.assertEqual(list(actual[2].orig_idxs), [4,5])

        self.assertTrue(torch.equal(actual[3].src, src_correct_4))
        self.assertEqual(list(actual[3].orig_idxs), [6])

    def testInitFromSrcDataInOrder(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        ds.initialize_from_src_data(src_sents, sents_per_batch=2, in_order=True)

        PAD, BOS, EOS = 0, 1, 2
        src_correct_1 = torch.tensor([[5,6,EOS,PAD,PAD],[5,6,8,9,EOS]])
        src_correct_2 = torch.tensor([[5,6,7,EOS],[7,8,EOS,PAD]])
        src_correct_3 = torch.tensor([[5,6,7,8,9,EOS],[5,6,7,8,EOS,PAD]])
        src_correct_4 = torch.tensor([[5,6,9,EOS]])

        actual = ds.batches

        self.assertEqual(len(actual), 4)

        self.assertTrue(torch.equal(actual[0].src, src_correct_1))
        self.assertEqual(list(actual[0].orig_idxs), [0,1])

        self.assertTrue(torch.equal(actual[1].src, src_correct_2))
        self.assertEqual(list(actual[1].orig_idxs), [2,3])

        self.assertTrue(torch.equal(actual[2].src, src_correct_3))
        self.assertEqual(list(actual[2].orig_idxs), [4,5])

        self.assertTrue(torch.equal(actual[3].src, src_correct_4))
        self.assertEqual(list(actual[3].orig_idxs), [6])

    def testInitFromSrcDataSorted(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        # sorted src: [[5,6],[7,8],[5,6,7],[5,6,9],[5,6,8,9],[5,6,7,8],[5,6,7,8,9]]
        # orig idxs: [0,4,2,1,6,5,3]
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        ds.initialize_from_src_data(src_sents, sents_per_batch=2, in_order=False)

        PAD, BOS, EOS = 0, 1, 2
        src_correct_1 = torch.tensor([[5,6,EOS],[7,8,EOS]])
        src_correct_2 = torch.tensor([[5,6,7,EOS],[5,6,9,EOS]])
        src_correct_3 = torch.tensor([[5,6,8,9,EOS],[5,6,7,8,EOS]])
        src_correct_4 = torch.tensor([[5,6,7,8,9,EOS]])

        actual = ds.batches

        self.assertEqual(len(actual), 4)

        self.assertTrue(torch.equal(actual[0].src, src_correct_1))
        self.assertEqual(list(actual[0].orig_idxs), [0,4])

        self.assertTrue(torch.equal(actual[1].src, src_correct_2))
        self.assertEqual(list(actual[1].orig_idxs), [2,1])

        self.assertTrue(torch.equal(actual[2].src, src_correct_3))
        self.assertEqual(list(actual[2].orig_idxs), [6,5])

        self.assertTrue(torch.equal(actual[3].src, src_correct_4))
        self.assertEqual(list(actual[3].orig_idxs), [3])

    def testGetEmptyTgtDataset(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        ds.initialize_from_src_data(src_sents, sents_per_batch=2, in_order=False)

        new_ds = ds.get_empty_tgt_dataset()
        self.assertEqual(len(new_ds), 0)
        self.assertEqual(new_ds.pad_idx, ds.pad_idx)
        self.assertEqual(new_ds.bos_idx, ds.bos_idx)
        self.assertEqual(new_ds.eos_idx, ds.eos_idx)

    def testAddBatch(self):
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        self.assertEqual(len(ds), 0)
        b1 = Seq2SeqTranslateBatch([], [])
        ds.add_batch(b1)
        self.assertEqual(len(ds), 1)
        b2 = Seq2SeqTranslateBatch([], [])
        ds.add_batch(b2)
        self.assertEqual(len(ds), 2)

    def testUnpad(self):
        src_sents = []
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)

        gen_with_eos = [1, 3, 4, 5, 6, 7, 2, 0, 0, 0]
        gen_no_eos = [1, 3, 4, 5, 6, 7, 0, 0, 0, 0]
        
        gen_unpad_correct = [3, 4, 5, 6, 7]
        gen_with_eos_unpad_actual = ds.unpad(gen_with_eos)
        gen_no_eos_unpad_actual = ds.unpad(gen_no_eos)
        self.assertEqual(gen_with_eos_unpad_actual, gen_unpad_correct)
        self.assertEqual(gen_no_eos_unpad_actual, gen_unpad_correct)

        gen_with_eos_unpad_correct = [1, 3, 4, 5, 6, 7, 2]
        gen_no_eos_unpad_correct = [1, 3, 4, 5, 6, 7]
        gen_with_eos_unpad_actual = ds.unpad(gen_with_eos, keep_bos_eos=True)
        gen_no_eos_unpad_actual = ds.unpad(gen_no_eos, keep_bos_eos=True)
        self.assertEqual(gen_with_eos_unpad_actual, gen_with_eos_unpad_correct)
        self.assertEqual(gen_no_eos_unpad_actual, gen_no_eos_unpad_correct)

    def testRestoreOrder(self):
        src_sents = []
        ds = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)

        tgt_orig = [[10,11,12],[10,11,12,13],[10,11,12,14],[10,11,12,13,14],[10,11],[10,12,13],[10,11,12,13,14,15]]
        tgt_sorted = [[10,11],[10,11,12],[10,12,13],[10,11,12,13],[10,11,12,14],[10,11,12,13,14],[10,11,12,13,14,15]]
        orig_idxs = [1,3,4,5,0,2,6]
        self.assertEqual(ds.restore_order(tgt_sorted, orig_idxs), tgt_orig)

    def testRoundTripInOrder(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        ds_src = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        ds_src.initialize_from_src_data(src_sents, sents_per_batch=2, in_order=True)

        ds_tgt = ds_src.get_empty_tgt_dataset()
        for src_batch in ds_src.batches:
           tgt_final = src_batch.src.clone()
           tgt_all = src_batch.src.clone().unsqueeze(1).expand(-1,3,-1)
           tgt_probs = torch.rand(size=tgt_all.size())
           tgt_batch = src_batch.with_translation(tgt_final, tgt_all, tgt_probs)
           ds_tgt.add_batch(tgt_batch)

        tgt_final_correct = src_sents
        tgt_all_correct = [[sent]*3 for sent in src_sents]
        tgt_final_actual, tgt_all_actual, probs_all_actual = ds_tgt.unbatch()
        self.assertEqual(tgt_final_actual, tgt_final_correct)
        self.assertEqual(tgt_all_actual, tgt_all_correct)

    def testRoundTripSorted(self):
        src_sents = [[5,6],[5,6,8,9],[5,6,7],[7,8],[5,6,7,8,9],[5,6,7,8],[5,6,9]]
        ds_src = Seq2SeqTranslateDataset(pad_idx=0, bos_idx=1, eos_idx=2)
        ds_src.initialize_from_src_data(src_sents, sents_per_batch=2, in_order=False)

        ds_tgt = ds_src.get_empty_tgt_dataset()
        for src_batch in ds_src.batches:
           tgt_final = src_batch.src.clone()
           tgt_all = src_batch.src.clone().unsqueeze(1).expand(-1,3,-1)
           tgt_probs = torch.rand(size=tgt_all.size())
           tgt_batch = src_batch.with_translation(tgt_final, tgt_all, tgt_probs)
           ds_tgt.add_batch(tgt_batch)

        tgt_final_correct = src_sents
        tgt_all_correct = [[sent]*3 for sent in src_sents]
        tgt_final_actual, tgt_all_actual, probs_all_actual = ds_tgt.unbatch()
        self.assertEqual(tgt_final_actual, tgt_final_correct)
        self.assertEqual(tgt_all_actual, tgt_all_correct)
