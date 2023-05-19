# TODO(darcey): consider moving the torch.no_grad part out of the iterator? feels like bad practice to have it in there

class Translator:

    def __init__(self, model, generator):
        self.model = model
        self.generator = generator

    def translate(self, data, print_every=0):
        output_batches = []
        num_sents = 0
        self.model.eval():
        with torch.no_grad():
            for batch in data.batches:
                src = batch.src.to(self.device)
                tgt_all = self.generator(src).cpu()
                new_batch = batch.with_translation(tgt_all)
                output_batches.append(new_batch)
                num_sents += src.size(0) * src.size(1)
                
                if print_every > 0 and num_sents > print_every:
                    yield output_batches
                    output_batches = []
                    num_sents = 0

        self.model.train()
        if len(output_batches) > 0:
            yield output_batches
