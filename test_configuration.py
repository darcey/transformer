from argparse import Namespace
import toml
import unittest
from configuration import *

class TestConfiguration(unittest.TestCase):

    def testParseMBROptionsShared(self):
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
                        mbr_share_sents        = true
                        weight_hypos_equally   = true
                        [generation.share]
                        decoding_method        = "Sampling"
                        sampling_k             = 15
                        num_beams_or_samples   = 100
                        """
        config_dict   = toml.loads(config_string)
        config_actual = Namespace(**config_dict["generation"])
        parse_gen_options(config_actual)
        parse_mbr_options(config_actual, config_dict["generation"])

        config_correct_dict = dict()
        config_correct_dict["max_parallel_sentences"] = 100
        config_correct_dict["num_beams_or_samples"]   = 5
        config_correct_dict["use_rel_max_len"]        = True
        config_correct_dict["rel_max_len"]            = 50
        config_correct_dict["abs_max_len"]            = 300
        config_correct_dict["decoding_method"]        = DecodingMethod.MBR
        config_correct_dict["sampling_k"]             = 0
        config_correct_dict["sampling_p"]             = 0.9
        config_correct_dict["sampling_temp"]          = 1.0
        config_correct_dict["allow_empty_string"]     = True
        config_correct_dict["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["length_reward_gamma"]    = 0.0
        config_correct_dict["length_norm_alpha"]      = 0.0
        config_correct_dict["mbr_share_sents"]        = True
        config_correct_dict["weight_hypos_equally"]   = True
        config_correct_dict["share"]                         = dict()
        config_correct_dict["share"]["max_parallel_sentences"] = 100
        config_correct_dict["share"]["num_beams_or_samples"]   = 100
        config_correct_dict["share"]["use_rel_max_len"]        = True
        config_correct_dict["share"]["rel_max_len"]            = 50
        config_correct_dict["share"]["abs_max_len"]            = 300
        config_correct_dict["share"]["decoding_method"]        = DecodingMethod.SAMPLING
        config_correct_dict["share"]["sampling_k"]             = 15
        config_correct_dict["share"]["sampling_p"]             = 0.9
        config_correct_dict["share"]["sampling_temp"]          = 1.0
        config_correct_dict["share"]["allow_empty_string"]     = True
        config_correct_dict["share"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["share"]["length_reward_gamma"]    = 0.0
        config_correct_dict["share"]["length_norm_alpha"]      = 0.0
        config_correct_dict["share"]["mbr_share_sents"]        = True
        config_correct_dict["share"]["weight_hypos_equally"]   = True
        config_correct = Namespace(**config_correct_dict)
        config_correct.share = Namespace(**config_correct_dict["share"])

        self.assertEqual(config_actual, config_correct)

    def testParseMBROptionsSeparate(self):
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
                        decoding_method        = "Sampling"
                        sampling_k             = 15
                        num_beams_or_samples   = 100
                        [generation.hypo]
                        decoding_method        = "Beam_Search"
                        allow_empty_string     = false
                        num_beams_or_samples   = 40
                        """
        config_dict   = toml.loads(config_string)
        config_actual = Namespace(**config_dict["generation"])
        parse_gen_options(config_actual)
        parse_mbr_options(config_actual, config_dict["generation"])

        config_correct_dict = dict()
        config_correct_dict["max_parallel_sentences"] = 100
        config_correct_dict["num_beams_or_samples"]   = 5
        config_correct_dict["use_rel_max_len"]        = True
        config_correct_dict["rel_max_len"]            = 50
        config_correct_dict["abs_max_len"]            = 300
        config_correct_dict["decoding_method"]        = DecodingMethod.MBR
        config_correct_dict["sampling_k"]             = 0
        config_correct_dict["sampling_p"]             = 0.9
        config_correct_dict["sampling_temp"]          = 1.0
        config_correct_dict["allow_empty_string"]     = True
        config_correct_dict["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["length_reward_gamma"]    = 0.0
        config_correct_dict["length_norm_alpha"]      = 0.0
        config_correct_dict["mbr_share_sents"]        = False
        config_correct_dict["weight_hypos_equally"]   = True
        config_correct_dict["cand"]                         = dict()
        config_correct_dict["cand"]["max_parallel_sentences"] = 100
        config_correct_dict["cand"]["num_beams_or_samples"]   = 100
        config_correct_dict["cand"]["use_rel_max_len"]        = True
        config_correct_dict["cand"]["rel_max_len"]            = 50
        config_correct_dict["cand"]["abs_max_len"]            = 300
        config_correct_dict["cand"]["decoding_method"]        = DecodingMethod.SAMPLING
        config_correct_dict["cand"]["sampling_k"]             = 15
        config_correct_dict["cand"]["sampling_p"]             = 0.9
        config_correct_dict["cand"]["sampling_temp"]          = 1.0
        config_correct_dict["cand"]["allow_empty_string"]     = True
        config_correct_dict["cand"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["cand"]["length_reward_gamma"]    = 0.0
        config_correct_dict["cand"]["length_norm_alpha"]      = 0.0
        config_correct_dict["cand"]["mbr_share_sents"]        = False
        config_correct_dict["cand"]["weight_hypos_equally"]   = True
        config_correct_dict["hypo"]                         = dict()
        config_correct_dict["hypo"]["max_parallel_sentences"] = 100
        config_correct_dict["hypo"]["num_beams_or_samples"]   = 40
        config_correct_dict["hypo"]["use_rel_max_len"]        = True
        config_correct_dict["hypo"]["rel_max_len"]            = 50
        config_correct_dict["hypo"]["abs_max_len"]            = 300
        config_correct_dict["hypo"]["decoding_method"]        = DecodingMethod.BEAM_SEARCH
        config_correct_dict["hypo"]["sampling_k"]             = 0
        config_correct_dict["hypo"]["sampling_p"]             = 0.9
        config_correct_dict["hypo"]["sampling_temp"]          = 1.0
        config_correct_dict["hypo"]["allow_empty_string"]     = False
        config_correct_dict["hypo"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["hypo"]["length_reward_gamma"]    = 0.0
        config_correct_dict["hypo"]["length_norm_alpha"]      = 0.0
        config_correct_dict["hypo"]["mbr_share_sents"]        = False
        config_correct_dict["hypo"]["weight_hypos_equally"]   = True
        config_correct = Namespace(**config_correct_dict)
        config_correct.cand = Namespace(**config_correct_dict["cand"])
        config_correct.hypo = Namespace(**config_correct_dict["hypo"])

        self.assertEqual(config_actual, config_correct)

    def testParseMBROptionsRecursive(self):
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
                        sampling_p             = 0.6
                        [generation.cand.share]
                        decoding_method        = "MBR"
                        mbr_share_sents        = false
                        sampling_k             = 15
                        num_beams_or_samples   = 100
                        [generation.cand.share.cand]
                        decoding_method        = "Sampling"
                        [generation.cand.share.hypo]
                        decoding_method        = "Sampling"
                        sampling_p             = 0.8
                        sampling_k             = 10
                        [generation.hypo]
                        decoding_method        = "Beam_Search"
                        allow_empty_string     = false
                        num_beams_or_samples   = 40
                        """
        config_dict   = toml.loads(config_string)
        config_actual = Namespace(**config_dict["generation"])
        parse_gen_options(config_actual)
        parse_mbr_options(config_actual, config_dict["generation"])

        config_correct_dict = dict()
        config_correct_dict["max_parallel_sentences"] = 100
        config_correct_dict["num_beams_or_samples"]   = 5
        config_correct_dict["use_rel_max_len"]        = True
        config_correct_dict["rel_max_len"]            = 50
        config_correct_dict["abs_max_len"]            = 300
        config_correct_dict["decoding_method"]        = DecodingMethod.MBR
        config_correct_dict["sampling_k"]             = 0
        config_correct_dict["sampling_p"]             = 0.9
        config_correct_dict["sampling_temp"]          = 1.0
        config_correct_dict["allow_empty_string"]     = True
        config_correct_dict["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["length_reward_gamma"]    = 0.0
        config_correct_dict["length_norm_alpha"]      = 0.0
        config_correct_dict["mbr_share_sents"]        = False
        config_correct_dict["weight_hypos_equally"]   = True
        config_correct_dict["cand"]                           = dict()
        config_correct_dict["cand"]["max_parallel_sentences"] = 100
        config_correct_dict["cand"]["num_beams_or_samples"]   = 5
        config_correct_dict["cand"]["use_rel_max_len"]        = True
        config_correct_dict["cand"]["rel_max_len"]            = 50
        config_correct_dict["cand"]["abs_max_len"]            = 300
        config_correct_dict["cand"]["decoding_method"]        = DecodingMethod.MBR
        config_correct_dict["cand"]["sampling_k"]             = 0
        config_correct_dict["cand"]["sampling_p"]             = 0.6
        config_correct_dict["cand"]["sampling_temp"]          = 1.0
        config_correct_dict["cand"]["allow_empty_string"]     = True
        config_correct_dict["cand"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["cand"]["length_reward_gamma"]    = 0.0
        config_correct_dict["cand"]["length_norm_alpha"]      = 0.0
        config_correct_dict["cand"]["mbr_share_sents"]        = True
        config_correct_dict["cand"]["weight_hypos_equally"]   = True
        config_correct_dict["cand"]["share"]                           = dict()
        config_correct_dict["cand"]["share"]["max_parallel_sentences"] = 100
        config_correct_dict["cand"]["share"]["num_beams_or_samples"]   = 100
        config_correct_dict["cand"]["share"]["use_rel_max_len"]        = True
        config_correct_dict["cand"]["share"]["rel_max_len"]            = 50
        config_correct_dict["cand"]["share"]["abs_max_len"]            = 300
        config_correct_dict["cand"]["share"]["decoding_method"]        = DecodingMethod.MBR
        config_correct_dict["cand"]["share"]["sampling_k"]             = 15
        config_correct_dict["cand"]["share"]["sampling_p"]             = 0.6
        config_correct_dict["cand"]["share"]["sampling_temp"]          = 1.0
        config_correct_dict["cand"]["share"]["allow_empty_string"]     = True
        config_correct_dict["cand"]["share"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["cand"]["share"]["length_reward_gamma"]    = 0.0
        config_correct_dict["cand"]["share"]["length_norm_alpha"]      = 0.0
        config_correct_dict["cand"]["share"]["mbr_share_sents"]        = False
        config_correct_dict["cand"]["share"]["weight_hypos_equally"]   = True
        config_correct_dict["cand"]["share"]["cand"]                           = dict()
        config_correct_dict["cand"]["share"]["cand"]["max_parallel_sentences"] = 100
        config_correct_dict["cand"]["share"]["cand"]["num_beams_or_samples"]   = 100
        config_correct_dict["cand"]["share"]["cand"]["use_rel_max_len"]        = True
        config_correct_dict["cand"]["share"]["cand"]["rel_max_len"]            = 50
        config_correct_dict["cand"]["share"]["cand"]["abs_max_len"]            = 300
        config_correct_dict["cand"]["share"]["cand"]["decoding_method"]        = DecodingMethod.SAMPLING
        config_correct_dict["cand"]["share"]["cand"]["sampling_k"]             = 15
        config_correct_dict["cand"]["share"]["cand"]["sampling_p"]             = 0.6
        config_correct_dict["cand"]["share"]["cand"]["sampling_temp"]          = 1.0
        config_correct_dict["cand"]["share"]["cand"]["allow_empty_string"]     = True
        config_correct_dict["cand"]["share"]["cand"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["cand"]["share"]["cand"]["length_reward_gamma"]    = 0.0
        config_correct_dict["cand"]["share"]["cand"]["length_norm_alpha"]      = 0.0
        config_correct_dict["cand"]["share"]["cand"]["mbr_share_sents"]        = False
        config_correct_dict["cand"]["share"]["cand"]["weight_hypos_equally"]   = True
        config_correct_dict["cand"]["share"]["hypo"]                           = dict()
        config_correct_dict["cand"]["share"]["hypo"]["max_parallel_sentences"] = 100
        config_correct_dict["cand"]["share"]["hypo"]["num_beams_or_samples"]   = 100
        config_correct_dict["cand"]["share"]["hypo"]["use_rel_max_len"]        = True
        config_correct_dict["cand"]["share"]["hypo"]["rel_max_len"]            = 50
        config_correct_dict["cand"]["share"]["hypo"]["abs_max_len"]            = 300
        config_correct_dict["cand"]["share"]["hypo"]["decoding_method"]        = DecodingMethod.SAMPLING
        config_correct_dict["cand"]["share"]["hypo"]["sampling_k"]             = 10
        config_correct_dict["cand"]["share"]["hypo"]["sampling_p"]             = 0.8
        config_correct_dict["cand"]["share"]["hypo"]["sampling_temp"]          = 1.0
        config_correct_dict["cand"]["share"]["hypo"]["allow_empty_string"]     = True
        config_correct_dict["cand"]["share"]["hypo"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["cand"]["share"]["hypo"]["length_reward_gamma"]    = 0.0
        config_correct_dict["cand"]["share"]["hypo"]["length_norm_alpha"]      = 0.0
        config_correct_dict["cand"]["share"]["hypo"]["mbr_share_sents"]        = False
        config_correct_dict["cand"]["share"]["hypo"]["weight_hypos_equally"]   = True
        config_correct_dict["hypo"]                           = dict()
        config_correct_dict["hypo"]["max_parallel_sentences"] = 100
        config_correct_dict["hypo"]["num_beams_or_samples"]   = 40
        config_correct_dict["hypo"]["use_rel_max_len"]        = True
        config_correct_dict["hypo"]["rel_max_len"]            = 50
        config_correct_dict["hypo"]["abs_max_len"]            = 300
        config_correct_dict["hypo"]["decoding_method"]        = DecodingMethod.BEAM_SEARCH
        config_correct_dict["hypo"]["sampling_k"]             = 0
        config_correct_dict["hypo"]["sampling_p"]             = 0.9
        config_correct_dict["hypo"]["sampling_temp"]          = 1.0
        config_correct_dict["hypo"]["allow_empty_string"]     = False
        config_correct_dict["hypo"]["length_normalization"]   = LengthNormalization.NONE
        config_correct_dict["hypo"]["length_reward_gamma"]    = 0.0
        config_correct_dict["hypo"]["length_norm_alpha"]      = 0.0
        config_correct_dict["hypo"]["mbr_share_sents"]        = False
        config_correct_dict["hypo"]["weight_hypos_equally"]   = True
        config_correct = Namespace(**config_correct_dict)
        config_correct.cand = Namespace(**config_correct_dict["cand"])
        config_correct.cand.share = Namespace(**config_correct_dict["cand"]["share"])
        config_correct.cand.share.cand = Namespace(**config_correct_dict["cand"]["share"]["cand"])
        config_correct.cand.share.hypo = Namespace(**config_correct_dict["cand"]["share"]["hypo"])
        config_correct.hypo = Namespace(**config_correct_dict["hypo"])

        self.assertEqual(config_actual, config_correct)

    def testMaxNumBeamsOrSamples(self):
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
                        sampling_p             = 0.6
                        num_beams_or_samples   = 200
                        [generation.cand.share]
                        decoding_method        = "MBR"
                        mbr_share_sents        = false
                        sampling_k             = 15
                        num_beams_or_samples   = 100
                        [generation.cand.share.cand]
                        decoding_method        = "Sampling"
                        num_beams_or_samples   = 25
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
        
        num_beams_or_samples = max_num_beams_or_samples(config)
        self.assertEqual(num_beams_or_samples, 50)
