import toml
import math
import random
import unittest
from configuration import *
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

    def testRecursiveMBR(self):
        config_string = """
                        [generation]
                        # How many sentences can the GPU handle at a time
                        max_parallel_sentences = 100

                        # Number of generations
                        num_beams_or_samples   = 5

                        # Length of generations
                        use_rel_max_len        = true
                        rel_max_len            = 50
                        abs_max_len            = 300

                        # Decoding method
                        decoding_method        = "MBR"

                        # Sampling params
                        sampling_k             = 0
                        sampling_p             = 0.9
                        sampling_temp          = 1.0

                        # Beam search params
                        allow_empty_string     = true
                        length_normalization   = "None"
                        length_reward_gamma    = 0.0
                        length_norm_alpha      = 0.0

                        # MBR params
                        mbr_share_sents        = false
                        weight_hypos_equally   = true
                        [generation.cand]
                        decoding_method        = "MBR"
                        mbr_share_sents        = true
                        num_beams_or_samples   = 30
                        [generation.cand.share]
                        decoding_method        = "MBR"
                        mbr_share_sents        = false
                        sampling_k             = 15
                        num_beams_or_samples   = 100
                        [generation.cand.share.cand]
                        decoding_method        = "Sampling"
                        num_beams_or_samples   = 75
                        [generation.cand.share.hypo]
                        decoding_method        = "Sampling"
                        sampling_p             = 0.8
                        sampling_k             = 10
                        num_beams_or_samples   = 50
                        [generation.hypo]
                        decoding_method        = "Beam_Search"
                        allow_empty_string     = false
                        num_beams_or_samples   = 40
                        """
        config_dict = toml.loads(config_string)
        config      = Namespace(**config_dict["generation"])
        parse_gen_options(config)
        parse_mbr_options(config, config_dict["generation"])
        
        toks = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
        lengths = range(10,21)
        def mock_inner_outer_generate(src, unbatch_func, config):
            tgt_all = []
            probs_all = []
            for i in range(src):
                tgt = []
                probs = []
                for j in range(config.num_beams_or_samples):
                    l = random.choice(lengths)
                    t = [random.choice(toks) for i in range(l)]
                    tgt.append(t)
                    probs.append(random.random())
                tgt_all.append(tgt)
                probs_all.append(probs)
            return [tgt_all[i][0] for i in range(len(tgt_all))], tgt_all, probs_all
        self.outer_gen.inner_outer_generate = mock_inner_outer_generate

        tgt_final, tgt_all, probs_all = self.outer_gen.middle_outer_generate(20, None, config)
        self.assertEqual(len(tgt_final), 20)
        self.assertEqual(len(tgt_all), 20)
        self.assertEqual(len(tgt_all[0]), 5)
        self.assertEqual(len(probs_all), 20)
        self.assertEqual(len(probs_all[0]), 5)

    def testMBRWeightEqually(self):
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
        probs1 = [0.1, 0.01, 0.001]
        probs2 = [0.001, 0.01, 0.1]
        probs_batch = [probs1, probs2]

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

        cands_final, cands_all, scores, probs = self.outer_gen.mbr(sents_batch, probs_batch, sents_batch, probs_batch, True)
        
        sent1_score = 1/3*(1 + 0.9 + 0.8)
        sent2_score = 1/3*(0.7 + 1 + 0.6)
        sent3_score = 1/3*(0.5 + 0.4 + 1)
        sent4_score = 1/3*(1 + 0.1 + 0.15)
        sent5_score = 1/3*(0.2 + 1 + 0.25)
        sent6_score = 1/3*(0.3 + 0.35 + 1)
        self.assertEqual(cands_final, [sent1, sent6])
        self.assertEqual(cands_all, [[sent1, sent2, sent3], [sent6, sent5, sent4]])
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(scores, [[sent1_score, sent2_score, sent3_score], [sent6_score, sent5_score, sent4_score]])))
        self.assertEqual(probs, [[0.1, 0.01, 0.001], [0.1, 0.01, 0.001]])

    def testMBRWeightByProbs(self):
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
        probs1 = [0.1, 0.01, 0.001]
        probs2 = [0.001, 0.01, 0.1]
        probs_batch = [probs1, probs2]

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

        cands_final, cands_all, scores, probs = self.outer_gen.mbr(sents_batch, probs_batch, sents_batch, probs_batch, False)
        
        sent1_score = 1*0.1 + 0.9*0.01 + 0.8*0.001  # 0.1098
        sent2_score = 0.7*0.1 + 1*0.01 + 0.6*0.001  # 0.0806
        sent3_score = 0.5*0.1 + 0.4*0.01 + 1*0.001  # 0.055
        sent4_score = 1*0.001 + 0.1*0.01 + 0.15*0.1 # 0.017
        sent5_score = 0.2*0.001 + 1*0.01 + 0.25*0.1 # 0.0352
        sent6_score = 0.3*0.001 + 0.35*0.01 + 1*0.1 # 0.1038
        self.assertEqual(cands_final, [sent1, sent6])
        self.assertEqual(cands_all, [[sent1, sent2, sent3], [sent6, sent5, sent4]])
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(scores, [[sent1_score, sent2_score, sent3_score], [sent6_score, sent5_score, sent4_score]])))
        self.assertEqual(probs, [[0.1, 0.01, 0.001], [0.1, 0.01, 0.001]])
