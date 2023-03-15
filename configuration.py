from enum import Enum
from argparse import Namespace
 
class NormType(Enum):
    NONE = 0
    LAYER_NORM = 1

def get_config_arch():
    config_arch = dict()
    config_arch["context_window_length"] = 512
    config_arch["num_encoder_layers"]    = 6
    config_arch["num_decoder_layers"]    = 6
    config_arch["num_attention_heads"]   = 8
    config_arch["d_model"]               = 512
    config_arch["d_ff"]                  = 2048
    config_arch["norm_type"]             = NormType.LAYER_NORM
    config_arch["layer_norm_epsilon"]    = 1e-5
    return Namespace(**config_arch)
