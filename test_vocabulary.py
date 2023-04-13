import unittest
from vocabulary import *


class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.fake_src = [["the", "dog", "walked", "to", "the", "park"],
                         ["I", "saw", "the", "rock", "in", "the", "park"]]
        self.fake_tgt = [["the", "ogday", "alkedway", "to", "the", "arkpay"],
                         ["I", "awsay", "the", "ockray", "in", "the", "arkpay"]]

    def testReadInVocab(self):
        vocab = Vocabulary(self.fake_src, self.fake_tgt)
        self.assertEqual(len(vocab), 14+4)

        src_toks_correct = set(["the", "dog", "walked", "to", "park", "I", "saw", "rock", "in"])
        tgt_toks_correct = set(["the", "ogday", "alkedway", "to", "arkpay", "I", "awsay", "ockray", "in"])
        spc_toks_correct = set([SpecialTokens.PAD, SpecialTokens.UNK, SpecialTokens.BOS, SpecialTokens.EOS])
        all_toks_correct = set(["the", "dog", "ogday", "walked", "alkedway", "to", "park", "arkpay", "I", "saw", "awsay", "rock", "ockray", "in", SpecialTokens.PAD, SpecialTokens.UNK, SpecialTokens.BOS, SpecialTokens.EOS])
        src_toks_actual = vocab.get_src_toks()
        tgt_toks_actual = vocab.get_tgt_toks()
        spc_toks_actual = vocab.get_special_toks()
        all_toks_actual = vocab.get_toks()
        self.assertEqual(src_toks_actual, src_toks_correct)
        self.assertEqual(tgt_toks_actual, tgt_toks_correct)
        self.assertEqual(spc_toks_actual, spc_toks_correct)
        self.assertEqual(all_toks_actual, all_toks_correct)

        for tok in all_toks_correct:
            self.assertTrue(vocab.has_tok(tok))
        self.assertFalse(vocab.has_tok("Thursday"))
        self.assertFalse(vocab.has_tok(SpecialTokens.CLS))
        for tok in src_toks_correct:
            self.assertTrue(vocab.has_src_tok(tok))
        self.assertFalse(vocab.has_src_tok("arkpay"))
        for tok in tgt_toks_correct:
            self.assertTrue(vocab.has_tgt_tok(tok))
        self.assertFalse(vocab.has_tgt_tok("park"))

    def testSpecialTokensInInputData(self):
        self.fake_src.append(["<<PAD>>"])
        with self.assertRaises(ValueError):
            vocab = Vocabulary(self.fake_src, self.fake_tgt)

    def testTokIdxMappings(self):
        vocab = Vocabulary(self.fake_src, self.fake_tgt)
        for tok in vocab.get_toks():
            self.assertEqual(vocab.idx_to_tok(vocab.tok_to_idx(tok)), tok)
        for idx in range(len(vocab)):
            self.assertEqual(vocab.tok_to_idx(vocab.idx_to_tok(idx)), idx)

    def testUnk(self):
        vocab = Vocabulary(self.fake_src, self.fake_tgt)
        unk = SpecialTokens.UNK

        src_sent = ["the", "dog", "saw", "a", "rock", "in", "the", "arkpay"]
        src_correct = ["the", "dog", "saw", unk, "rock", "in", "the", unk]
        src_actual = vocab.unk_src(src_sent)
        self.assertEqual(src_actual, src_correct)

        tgt_sent = ["the", "ogday", "awsay", "a", "ockray", "in", "the", "park"]
        tgt_correct = ["the", "ogday", "awsay", unk, "ockray", "in", "the", unk]
        tgt_actual = vocab.unk_tgt(tgt_sent)
        self.assertEqual(tgt_actual, tgt_correct)

    def testTgtOutputMask(self):
        vocab = Vocabulary(self.fake_src, self.fake_tgt)

        tgt_output_mask = vocab.get_tgt_output_mask()
        self.assertTrue(tgt_output_mask[vocab.tok_to_idx("the")])
        self.assertFalse(tgt_output_mask[vocab.tok_to_idx("dog")])
        self.assertTrue(tgt_output_mask[vocab.tok_to_idx("ogday")])
        self.assertFalse(tgt_output_mask[vocab.tok_to_idx(SpecialTokens.BOS)])
        self.assertTrue(tgt_output_mask[vocab.tok_to_idx(SpecialTokens.EOS)])

        tgt_output_mask = vocab.get_tgt_output_mask(bool_mask=False)
        self.assertEqual(tgt_output_mask[vocab.tok_to_idx("the")], 1.0)
        self.assertEqual(tgt_output_mask[vocab.tok_to_idx("dog")], 0.0)
        self.assertEqual(tgt_output_mask[vocab.tok_to_idx("ogday")], 1.0)
        self.assertEqual(tgt_output_mask[vocab.tok_to_idx(SpecialTokens.BOS)], 0.0)
        self.assertEqual(tgt_output_mask[vocab.tok_to_idx(SpecialTokens.EOS)], 1.0)
