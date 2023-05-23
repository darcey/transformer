# TODO(darcey): implement everything
# TODO(darcey): figure out what the return type of the generator is (there will probably be tgt_final, tgt_all, probs, score... need to update the batch format, the unbatching, the printing, etc.)

# TODO(darcey): consider changing generate() to be a yield-style function, in order to accommodate extremely large numbers of samples

import torch

class Generator:

    def __init__(self, model, config):
        self.model = model

    def generate(self, src):
        tgt_all = src.clone().unsqueeze(1).expand(-1, num_beams_or_samples, -1)
        return tgt_all
