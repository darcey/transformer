from enum import Enum
from argparse import Namespace

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
    ENCODER_DECODER = 1
    ENCODER_ONLY = 2
    DECODER_ONLY = 3
    CUSTOM_TWO_SEQ = 4
    CUSTOM_ONE_SEQ = 5

class PositionalEncodingType(Enum):
    NONE = 0
    SINUSOIDAL = 1

class NormType(Enum):
    NONE = 0
    LAYER_NORM = 1
    SCALE_NORM = 2

def get_config_arch():
    config_arch = dict()

    # Major architectural options
    config_arch["transformer_type"]       = TransformerType.ENCODER_DECODER
    config_arch["num_encoder_layers"]     = 6
    config_arch["num_decoder_layers"]     = 6
    config_arch["d_model"]                = 512
    # These options are only relevant if TransformerType is CUSTOM*
    config_arch["output_probs"]           = True
    config_arch["use_masked_att_encoder"] = False
    config_arch["use_masked_att_decoder"] = True

    # Layer options
    config_arch["use_resid_connection"]   = True
    config_arch["pre_norm"]               = True

    # Attention options
    config_arch["num_attention_heads"]    = 8

    # Feed-forward options
    config_arch["d_ff"]                   = 2048
    
    # Positional encoding options
    config_arch["pos_enc_type"]           = PositionalEncodingType.SINUSOIDAL
    config_arch["context_window_length"]  = 512

    # Normalization options
    config_arch["norm_type"]              = NormType.SCALE_NORM
    config_arch["layer_norm_epsilon"]     = 1e-5
    
    # Embedding options
    config_arch["fix_norm"]               = True

    return Namespace(**config_arch)

def get_config_train():
    config_train = dict()

    # Length of training
    config_train["max_epochs"]   = 200
    config_train["epoch_size"]   = 1000

    # Dropout options
    config_train["dropout"]      = 0.3
    config_train["att_dropout"]  = 0.3
    config_train["ff_dropout"]   = 0.3
    config_train["word_dropout"] = 0.1

    return Namespace(**config_train)
