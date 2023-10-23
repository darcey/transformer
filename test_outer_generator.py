import math
import unittest
from configuration import read_config
from vocabulary import Vocabulary, SpecialTokens
from outer_generator import *

class TestMBR(unittest.TestCase):

    def setUp(self):
        device = "cpu"
        config = read_config("configuration.toml")
        vocab = Vocabulary()
        vocab.initialize_from_data([], [])
        gen = None

        self.outer_gen = OuterGenerator(gen, vocab, config, device)

    def testMBR(self):
        BOS = SpecialTokens.BOS
        EOS = SpecialTokens.EOS
        PAD = SpecialTokens.PAD
        sent1 = [BOS, 'I', 'wanted', 'to', 'understand', 'who', 'was', 'doing', 'the', 'job', '.', EOS]
        sent2 = [BOS, 'I', 'wanted', 'to', 'know', 'which', 'sandwich', 'to', 'eat', '.', EOS]
        sent3 = [BOS, 'If', 'the', 'job', 'is', 'to', 'eat', 'a', 'sandwich']
        sent4 = [BOS, 'Give', 'me', 'a', 'bigger', 'cookie', '.', EOS]
        sent5 = [BOS, 'My', 'mother', 'has', 'to', 'eat', 'the', 'sandwich']
        sent6 = ['Nobody', 'wants', 'to', 'trade', 'their', 'cookie', 'for', 'a', 'sandwich', '.', EOS]
        
        sents1 = [sent1, sent2, sent3]
        sents2 = [sent4, sent5, sent6]
        sents_batch = [sents1, sents2]
        #probs1 = [0.1, 0.01, 0.001]
        #probs2 = [0.001, 0.01, 0.1]
        #probs_batch = [probs1, probs2]

        sent1_trim = ['I', 'wanted', 'to', 'understand', 'who', 'was', 'doing', 'the', 'job', '.']
        sent2_trim = ['I', 'wanted', 'to', 'know', 'which', 'sandwich', 'to', 'eat', '.']
        sent3_trim = ['If', 'the', 'job', 'is', 'to', 'eat', 'a', 'sandwich']
        sent4_trim = ['Give', 'me', 'a', 'bigger', 'cookie', '.']
        sent5_trim = ['My', 'mother', 'has', 'to', 'eat', 'the', 'sandwich']
        sent6_trim = ['Nobody', 'wants', 'to', 'trade', 'their', 'cookie', 'for', 'a', 'sandwich', '.']

        def mock_score_func(sent_a, sent_b):
            if sent_a == sent_b:
                return 1
            else:
                if sent_a == sent1_trim and sent_b == sent2_trim:
                    return 0.9
                elif sent_a == sent1_trim and sent_b == sent3_trim:
                    return 0.8
                elif sent_a == sent2_trim and sent_b == sent1_trim:
                    return 0.7
                elif sent_a == sent2_trim and sent_b == sent3_trim:
                    return 0.6
                elif sent_a == sent3_trim and sent_b == sent1_trim:
                    return 0.5
                elif sent_a == sent3_trim and sent_b == sent2_trim:
                    return 0.4
                elif sent_a == sent4_trim and sent_b == sent5_trim:
                    return 0.1
                elif sent_a == sent4_trim and sent_b == sent6_trim:
                    return 0.15
                elif sent_a == sent5_trim and sent_b == sent4_trim:
                    return 0.2
                elif sent_a == sent5_trim and sent_b == sent6_trim:
                    return 0.25
                elif sent_a == sent6_trim and sent_b == sent4_trim:
                    return 0.3
                elif sent_a == sent6_trim and sent_b == sent5_trim:
                    return 0.35
                else:
                    self.assertTrue(False)
        self.outer_gen.sentence_bleu = mock_score_func

        cands_final, cands_all, scores = self.outer_gen.mbr(sents_batch)
        
        #sent1_score = 1*0.1 + 0.9*0.01 + 0.8*0.001  # 0.1098
        #sent2_score = 0.7*0.1 + 1*0.01 + 0.6*0.001  # 0.0806
        #sent3_score = 0.5*0.1 + 0.4*0.01 + 1*0.001  # 0.055
        #sent4_score = 1*0.001 + 0.35*0.01 + 0.3*0.1 # 0.0345
        #sent5_score = 0.25*0.001 + 1*0.01 + 0.2*0.1 # 0.03025
        #sent6_score = 0.15*0.001 + 0.1*0.01 + 1*0.1 # 0.10115
        sent1_score = 1/3*(1 + 0.9 + 0.8)
        sent2_score = 1/3*(0.7 + 1 + 0.6)
        sent3_score = 1/3*(0.5 + 0.4 + 1)
        sent4_score = 1/3*(1 + 0.1 + 0.15)
        sent5_score = 1/3*(0.2 + 1 + 0.25)
        sent6_score = 1/3*(0.3 + 0.35 + 1)
        self.assertEqual(cands_final, [sent1, sent6])
        self.assertEqual(cands_all, [[sent1, sent2, sent3], [sent6, sent5, sent4]])
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(scores, [[sent1_score, sent2_score, sent3_score], [sent6_score, sent5_score, sent4_score]])))
