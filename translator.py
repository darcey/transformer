import torch

class Translator:

    def __init__(self, model, outer_generator):
        self.model = model
        self.outer_generator = outer_generator

    def translate(self, test):
        # Outer loop runs once per yield
        for sub_test in test.subdatasets:
            # Inner loop translates batches until it is time to yield
            self.model.eval()
            with torch.no_grad():
                tgt_finals = []
                tgt_alls   = []
                probs_alls = []
                for src in sub_test.batches:
                    tgt_final, tgt_all, probs_all = self.outer_generator.outer_generate(src, sub_test.unbatch)
                    tgt_finals.extend(tgt_final)
                    tgt_alls.extend(tgt_all)
                    probs_alls.extend(probs_all)
                tgt_finals = sub_test.restore_order(tgt_finals)
                tgt_alls = sub_test.restore_order(tgt_alls)
                probs_alls = sub_test.restore_order(probs_alls)
            self.model.train()
            yield tgt_finals, tgt_alls, probs_alls
