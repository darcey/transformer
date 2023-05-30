import unittest
from tempfile import NamedTemporaryFile
from vocabulary import *


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.fake_src = [["the", "dog", "walked", "to", "the", "park"],
                         ["I", "saw", "the", "rock", "in", "the", "park"]]
        self.fake_tgt = [["the", "ogday", "alkedway", "to", "the", "arkpay"],
                         ["I", "awsay", "the", "ockray", "in", "the", "arkpay"]]

    def testInitFromData(self):
        vocab = Vocabulary()
        vocab.initialize_from_data(self.fake_src, self.fake_tgt)
        self.assertEqual(len(vocab), 14+4)

        src_toks_correct = set(["the", "dog", "walked", "to", "park", "I", "saw", "rock", "in"])
        tgt_toks_correct = set(["the", "ogday", "alkedway", "to", "arkpay", "I", "awsay", "ockray", "in"])
        spc_toks_correct = set([SpecialTokens.PAD, SpecialTokens.UNK, SpecialTokens.BOS, SpecialTokens.EOS])
        all_toks_correct = set(["the", "dog", "ogday", "walked", "alkedway", "to", "park", "arkpay", "I", "saw", "awsay", "rock", "ockray", "in", SpecialTokens.PAD, SpecialTokens.UNK, SpecialTokens.BOS, SpecialTokens.EOS])
        src_toks_actual = vocab.src_toks
        tgt_toks_actual = vocab.tgt_toks
        spc_toks_actual = vocab.special_toks
        all_toks_actual = vocab.special_toks | vocab.src_tgt_toks
        self.assertEqual(src_toks_actual, src_toks_correct)
        self.assertEqual(tgt_toks_actual, tgt_toks_correct)
        self.assertEqual(spc_toks_actual, spc_toks_correct)
        self.assertEqual(all_toks_actual, all_toks_correct)

    def testReadWriteFile(self):
        vocab = Vocabulary()
        vocab.initialize_from_data(self.fake_src, self.fake_tgt)

        f = NamedTemporaryFile()
        filename = f.name

        vocab.write_to_file(filename)
        vocab_copy = Vocabulary()
        vocab_copy.read_from_file(filename)

        self.assertEqual(len(vocab), len(vocab_copy))
        self.assertEqual(vocab.src_toks, vocab_copy.src_toks)
        self.assertEqual(vocab.tgt_toks, vocab_copy.tgt_toks)
        self.assertEqual(vocab.src_tgt_toks, vocab_copy.src_tgt_toks)
        self.assertEqual(vocab.special_toks, vocab_copy.special_toks)

        self.assertEqual(vocab.t_to_i, vocab_copy.t_to_i)
        self.assertEqual(vocab.i_to_t, vocab_copy.i_to_t)

    def testSpecialTokensInInputData(self):
        self.fake_src.append(["<<PAD>>"])
        with self.assertRaises(ValueError):
            vocab = Vocabulary()
            vocab.initialize_from_data(self.fake_src, self.fake_tgt)

    def testTokIdxMappings(self):
        vocab = Vocabulary()
        vocab.initialize_from_data(self.fake_src, self.fake_tgt)

        for tok in vocab.special_toks | vocab.src_tgt_toks:
            self.assertEqual(vocab.idx_to_tok(vocab.tok_to_idx(tok)), tok)
        for idx in range(len(vocab)):
            self.assertEqual(vocab.tok_to_idx(vocab.idx_to_tok(idx)), idx)

        with self.assertRaises(ValueError):
            vocab.tok_to_idx_data(["the"], nesting=1)
        with self.assertRaises(ValueError):
            vocab.idx_to_tok_data([5], nesting=1)
        self.assertEqual(vocab.idx_to_tok_data(vocab.tok_to_idx_data(self.fake_src)), self.fake_src)
        self.assertEqual(vocab.idx_to_tok_data(vocab.tok_to_idx_data(self.fake_tgt)), self.fake_tgt)
        fake_src_3 = [self.fake_src.copy(), self.fake_src.copy(), self.fake_src.copy()]
        fake_tgt_3 = [self.fake_tgt.copy(), self.fake_tgt.copy(), self.fake_tgt.copy()]
        self.assertEqual(vocab.idx_to_tok_data(vocab.tok_to_idx_data(fake_src_3, nesting=3), nesting=3), fake_src_3)
        self.assertEqual(vocab.idx_to_tok_data(vocab.tok_to_idx_data(fake_tgt_3, nesting=3), nesting=3), fake_tgt_3)

    def testUnk(self):
        vocab = Vocabulary()
        vocab.initialize_from_data(self.fake_src, self.fake_tgt)
        unk = SpecialTokens.UNK

        src_data = [["the", "dog", "saw", "a", "rock", "in", "the", "arkpay"],
                    ["my", "dog", "barks"]]
        src_correct = [["the", "dog", "saw", unk, "rock", "in", "the", unk],
                       [unk, "dog", unk]]
        src_actual = vocab.unk_data(src_data, src=True)
        self.assertEqual(src_actual, src_correct)

        tgt_data = [["the", "ogday", "awsay", "a", "ockray", "in", "the", "park"],
                    ["ymay", "ogday", "arksbay"]]
        tgt_correct = [["the", "ogday", "awsay", unk, "ockray", "in", "the", unk],
                       [unk, "ogday", unk]]
        tgt_actual = vocab.unk_data(tgt_data, src=False)
        self.assertEqual(tgt_actual, tgt_correct)

    def testTgtOutputMask(self):
        vocab = Vocabulary()
        vocab.initialize_from_data(self.fake_src, self.fake_tgt)

        tgt_support_mask = vocab.get_tgt_support_mask()
        self.assertTrue(tgt_support_mask[vocab.tok_to_idx("the")])
        self.assertFalse(tgt_support_mask[vocab.tok_to_idx("dog")])
        self.assertTrue(tgt_support_mask[vocab.tok_to_idx("ogday")])
        self.assertFalse(tgt_support_mask[vocab.bos_idx()])
        self.assertTrue(tgt_support_mask[vocab.eos_idx()])
