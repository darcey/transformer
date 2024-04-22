import toml
import math
import random
import os
import subprocess
from tempfile import NamedTemporaryFile
import unittest
from configuration import *
from vocabulary import Vocabulary, SpecialTokens
from outer_generator import *

class TestMBR(unittest.TestCase):

    def setUp(self):
        device = "cpu"
        config = read_config("configuration.toml")
        config.gen.decoding_method = DecodingMethod.MBR
        vocab = Vocabulary()
        vocab.initialize_from_data([], [])
        gen = None
        
        self.outer_gen = OuterGenerator(gen, vocab, config, device)

    def setUpToyData(self):
        self.sent1 = ['I', 'wanted', 'to', 'understand', 'who', 'was', 'doing', 'the', 'job', '.']
        self.sent2 = ['I', 'wanted', 'to', 'know', 'which', 'sandwich', 'to', 'eat', '.']
        self.sent3 = ['If', 'the', 'job', 'is', 'to', 'eat', 'a', 'sandwich']
        self.sent4 = ['Give', 'me', 'a', 'bigger', 'cookie', '.']
        self.sent5 = ['My', 'mother', 'has', 'to', 'eat', 'the', 'sandwich']
        self.sent6 = ['Nobody', 'wants', 'to', 'trade', 'their', 'cookie', 'for', 'a', 'sandwich', '.']
        
        self.sents1 = [self.sent1, self.sent2, self.sent3]
        self.sents2 = [self.sent4, self.sent5, self.sent6]
        self.sents_batch = [self.sents1, self.sents2]
        self.probs1 = [math.log(0.1), math.log(0.01), math.log(0.001)]
        self.probs2 = [math.log(0.001), math.log(0.01), math.log(0.1)]
        self.probs_batch = [self.probs1, self.probs2]

        def mock_score_func(sent_a, sent_b):
            if sent_a == sent_b:
                return 1
            else:
                if sent_a == self.sent1 and sent_b == self.sent2:
                    return 0.9
                elif sent_a == self.sent1 and sent_b == self.sent3:
                    return 0.8
                elif sent_a == self.sent2 and sent_b == self.sent1:
                    return 0.7
                elif sent_a == self.sent2 and sent_b == self.sent3:
                    return 0.6
                elif sent_a == self.sent3 and sent_b == self.sent1:
                    return 0.5
                elif sent_a == self.sent3 and sent_b == self.sent2:
                    return 0.4
                elif sent_a == self.sent4 and sent_b == self.sent5:
                    return 0.1
                elif sent_a == self.sent4 and sent_b == self.sent6:
                    return 0.15
                elif sent_a == self.sent5 and sent_b == self.sent4:
                    return 0.2
                elif sent_a == self.sent5 and sent_b == self.sent6:
                    return 0.25
                elif sent_a == self.sent6 and sent_b == self.sent4:
                    return 0.3
                elif sent_a == self.sent6 and sent_b == self.sent5:
                    return 0.35
                else:
                    self.assertTrue(False)
        self.mock_score_func = mock_score_func

    def testProcessForMetricNoScript(self):
        self.setUpToyData()
        correct = [["I wanted to understand who was doing the job .",\
                    "I wanted to know which sandwich to eat .",\
                    "If the job is to eat a sandwich"],\
                   ["Give me a bigger cookie .",\
                    "My mother has to eat the sandwich",\
                    "Nobody wants to trade their cookie for a sandwich ."]]
        out = self.outer_gen.process_for_metric(self.sents_batch)
        self.assertEqual(out, correct)

    def testProcessForMetricWithScript(self):
        script_file = NamedTemporaryFile(delete=True)
        with open(script_file.name, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("sed s/sandwich/cake/g $1\n")
        os.chmod(script_file.name, 0o777)
        script_file.file.close()
        self.outer_gen.postproc_script = script_file.name
        #self.outer_gen.postproc_script = "/afs/crc.nd.edu/group/nlp/09/darcey/mt-system/prod/transformer/testing.sh"

        self.setUpToyData()
        correct = [["I wanted to understand who was doing the job .",\
                    "I wanted to know which cake to eat .",\
                    "If the job is to eat a cake"],\
                   ["Give me a bigger cookie .",\
                    "My mother has to eat the cake",\
                    "Nobody wants to trade their cookie for a cake ."]]
        out = self.outer_gen.process_for_metric(self.sents_batch)
        self.assertEqual(out, correct)

    def testMBRWeightEqually(self):
        self.setUpToyData()
        self.outer_gen.sentence_bleu = self.mock_score_func
        cands_final, cands_all, scores, probs = self.outer_gen.mbr(self.sents_batch, self.sents_batch, self.probs_batch, self.sents_batch, self.sents_batch, self.probs_batch, MBRMetric.BLEU_TOK, True)
        
        sent1_score = 1/3*(1 + 0.9 + 0.8)
        sent2_score = 1/3*(0.7 + 1 + 0.6)
        sent3_score = 1/3*(0.5 + 0.4 + 1)
        sent4_score = 1/3*(1 + 0.1 + 0.15)
        sent5_score = 1/3*(0.2 + 1 + 0.25)
        sent6_score = 1/3*(0.3 + 0.35 + 1)
        self.assertEqual(cands_final, [self.sent1, self.sent6])
        self.assertEqual(cands_all, [[self.sent1, self.sent2, self.sent3], [self.sent6, self.sent5, self.sent4]])
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(scores, [[sent1_score, sent2_score, sent3_score], [sent6_score, sent5_score, sent4_score]])))
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(probs, [[0.1, 0.01, 0.001], [0.1, 0.01, 0.001]])))

    def testMBRWeightByProbs(self):
        self.setUpToyData()
        self.outer_gen.sentence_bleu = self.mock_score_func
        cands_final, cands_all, scores, probs = self.outer_gen.mbr(self.sents_batch, self.sents_batch, self.probs_batch, self.sents_batch, self.sents_batch, self.probs_batch, MBRMetric.BLEU_TOK, False)
        
        sent1_score = 1*0.1 + 0.9*0.01 + 0.8*0.001  # 0.1098
        sent2_score = 0.7*0.1 + 1*0.01 + 0.6*0.001  # 0.0806
        sent3_score = 0.5*0.1 + 0.4*0.01 + 1*0.001  # 0.055
        sent4_score = 1*0.001 + 0.1*0.01 + 0.15*0.1 # 0.017
        sent5_score = 0.2*0.001 + 1*0.01 + 0.25*0.1 # 0.0352
        sent6_score = 0.3*0.001 + 0.35*0.01 + 1*0.1 # 0.1038
        self.assertEqual(cands_final, [self.sent1, self.sent6])
        self.assertEqual(cands_all, [[self.sent1, self.sent2, self.sent3], [self.sent6, self.sent5, self.sent4]])
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(scores, [[sent1_score, sent2_score, sent3_score], [sent6_score, sent5_score, sent4_score]])))
        self.assertTrue(all(all(math.isclose(out, actual) for out, actual in zip(outs, actuals)) for outs, actuals in zip(probs, [[0.1, 0.01, 0.001], [0.1, 0.01, 0.001]])))

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
                        mbr_metric             = "BLEU_tok"
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
        def mock_call_generator(src, unbatch_func, config):
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
        self.outer_gen.call_generator = mock_call_generator

        tgt_final, tgt_all, probs_all = self.outer_gen.outer_generate(20, None, config)
        self.assertEqual(len(tgt_final), 20)
        self.assertEqual(len(tgt_all), 20)
        self.assertEqual(len(tgt_all[0]), 5)
        self.assertEqual(len(probs_all), 20)
        self.assertEqual(len(probs_all[0]), 5)
