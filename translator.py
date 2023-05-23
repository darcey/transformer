# TODO(darcey): consider moving the torch.no_grad part out of the iterator? feels like bad practice to have it in there

import torch

class Translator:

    def __init__(self, model, generator, device):
        self.model = model
        self.generator = generator
        self.device = device

    def translate(self, data, print_every=0):
        output_batches = []
        num_sents = 0
        self.model.eval()
        with torch.no_grad():
            for batch in data.batches:
                src = batch.src.to(self.device)
                tgt_all = self.generator.generate(src).cpu()
                new_batch = batch.with_translation(tgt_all)
                output_batches.append(new_batch)
                num_sents += tgt_all.size(0) * tgt_all.size(1)
                
                if print_every > 0 and num_sents > print_every:
                    yield output_batches
                    output_batches = []
                    num_sents = 0

        self.model.train()
        if len(output_batches) > 0:
            yield output_batches
