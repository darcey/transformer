from enum import Enum
from argparse import Namespace
import toml

# Allows the three standard transformer configurations (Encoder-Decoder, 
# Encoder only, Decoder only) as well as two "Custom" configurations that
# permit maximum flexibility.

# Encoder-Decoder model:
#   encoder: takes one sequence,  no masked self-attention,  outputs encodings
#   decoder: takes two sequences, has masked self-attention, outputs probs
# Encoder Only model:
#   encoder: takes one sequence,  no masked self-attention,  outputs encodings
# Decoder Only model:
#   decoder: takes one sequence,  has masked self-attention, outputs probs
# Custom, Two-Sequence:
#   encoder: takes one sequence,  [use_masked_att_encoder],  outputs encodings
#   decoder: takes two sequences, [use_masked_att_decoder],  [output_probs]
# Custom, One-Sequence (arbitrarily chosen to be decoder):
#   decoder: takes one sequence,  [use_masked_att_decoder],  [output_probs]

class TransformerType(Enum):
    ENCODER_DECODER = "Encoder_Decoder"
    ENCODER_ONLY = "Encoder_Only"
    DECODER_ONLY = "Decoder_Only"
    CUSTOM_TWO_SEQ = "Custom_Two_Seq"
    CUSTOM_ONE_SEQ = "Custom_One_Seq"

class PositionalEncodingType(Enum):
    NONE = "None"
    SINUSOIDAL = "Sinusoidal"

class NormType(Enum):
    NONE = "None"
    LAYER_NORM = "Layer_Norm"
    SCALE_NORM = "Scale_Norm"

class LearningRateStrategy(Enum):
    WARMUP_INV_SQRT_DECAY = "Warmup_InvSqrtDecay"
    WARMUP_VAL_DECAY = "Warmup_ValDecay"
    NO_WARMUP_VAL_DECAY = "NoWarmup_ValDecay"

class ClipGrad(Enum):
    NONE = "None"
    MAX = "Max"
    NORM = "Norm"

class DecodingMethod(Enum):
    SAMPLING = "Sampling"
    BEAM_SEARCH = "Beam_Search"

class SamplingMethod(Enum):
    ANCESTRAL   = "Ancestral"
    TOP_K       = "Top_k"
    TOP_P       = "Top_p"

def read_config(filename):
    with open(filename) as config_file:
        config_dict = toml.load(config_file)

    config_arch = Namespace(**config_dict["architecture"])
    config_arch.transformer_type = TransformerType(config_arch.transformer_type)
    config_arch.pos_enc_type = PositionalEncodingType(config_arch.pos_enc_type)
    config_arch.norm_type = NormType(config_arch.norm_type)

    config_train = Namespace(**config_dict["training"])
    config_train_lr = Namespace(**config_dict["training"]["lr"])
    config_train_lr.lr_strategy = LearningRateStrategy(config_train_lr.lr_strategy)
    config_train.lr = config_train_lr
    config_train.clip_grad = ClipGrad(config_train.clip_grad)
    
    config_gen = Namespace(**config_dict["generation"])
    config_gen.decoding_method = DecodingMethod(config_gen.decoding_method)
    config_gen.sampling_method = SamplingMethod(config_gen.sampling_method)

    config = Namespace()
    config.arch = config_arch
    config.train = config_train
    config.gen = config_gen
    return config
