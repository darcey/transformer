import torch

class Translator:

    def __init__(self, model, generator, device):
        self.model = model
        self.generator = generator
        self.device = device

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
                    src = src.to(self.device)
                    tgt_final, tgt_all, probs_all = self.generator.generate(src)

                    tgt_final = tgt_final.cpu()
                    tgt_all = tgt_all.cpu()
                    probs_all = probs_all.cpu()

                    tgt_final, tgt_all, probs_all = sub_test.unbatch(tgt_final, tgt_all, probs_all)
                    
                    tgt_finals.extend(tgt_final)
                    tgt_alls.extend(tgt_all)
                    probs_alls.extend(probs_all)
                tgt_finals = sub_test.restore_order(tgt_finals)
                tgt_alls = sub_test.restore_order(tgt_alls)
                probs_alls = sub_test.restore_order(probs_alls)
            self.model.train()
            yield tgt_finals, tgt_alls, probs_alls
