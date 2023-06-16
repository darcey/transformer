import torch

class Translator:

    def __init__(self, model, generator, device):
        self.model = model
        self.generator = generator
        self.device = device

    def translate(self, data_src, yield_interval=0):
        # Outer loop runs once per yield
        num_batches = 0
        while num_batches < len(data_src):
            data_tgt = data_src.get_empty_tgt_dataset()
            num_sents = 0
            # Inner loop translates batches until it is time to yield
            self.model.eval()
            with torch.no_grad():
                while True:
                    batch = data_src.batches[num_batches]
                    src = batch.src.to(self.device)
                    tgt_final, tgt_all, probs_all = self.generator.generate(src)
                    tgt_final = tgt_final.cpu()
                    tgt_all = tgt_all.cpu()
                    probs_all = probs_all.cpu()

                    new_batch = batch.with_translation(tgt_final, tgt_all, probs_all)
                    data_tgt.add_batch(new_batch)

                    num_sents += tgt_all.size(0) * tgt_all.size(1)
                    num_batches += 1

                    if yield_interval > 0 and num_sents > yield_interval:
                        break
                    if num_batches >= len(data_src):
                        break
            self.model.train()
            yield data_tgt
