# TODO(darcey): add more config options for MBR

import numpy as np
import logging
from sacrebleu.metrics import BLEU
from configuration import DecodingMethod
from vocabulary import SpecialTokens

class OuterGenerator:

    def __init__(self, generator, vocab, config, device):
        self.generator = generator
        self.vocab = vocab
        self.config = config.gen
        self.device = device

        if self.config.decoding_method == DecodingMethod.MBR:
            self.bleu = BLEU(smooth_method='add-k', smooth_value=1)
            logging.getLogger('sacrebleu').setLevel(logging.CRITICAL)

    def outer_generate(self, src, unbatch_func):
        src = src.to(self.device)
        tgt_final, tgt_all, probs_all = self.generator.generate(src)

        tgt_final = tgt_final.cpu()
        tgt_all = tgt_all.cpu()
        probs_all = probs_all.cpu()

        tgt_final, tgt_all, probs_all = unbatch_func(tgt_final, tgt_all, probs_all)
        tgt_final = self.vocab.idx_to_tok_data(tgt_final)
        tgt_all = self.vocab.idx_to_tok_data(tgt_all, nesting=3)
        
        if self.config.decoding_method == DecodingMethod.MBR:
            tgt_final, tgt_all, scores = self.mbr(tgt_all)

        tgt_final = self.vocab.remove_bos_eos_data(tgt_final)

        return tgt_final, tgt_all, probs_all

    def mbr(self, cand_hypo_batch):
        cand_finals = []
        cand_alls = []
        cand_alls_scores = []
        for cand_hypo in cand_hypo_batch:
            num_cands = len(cand_hypo)
            num_hypos = len(cand_hypo)

            cand_scores = []
            for cand_idx in range(num_cands):
                cand_score = 0.0
                for hypo_idx in range(num_hypos):

                    cand = cand_hypo[cand_idx]
                    cand_trim = self.vocab.remove_bos_eos_sent(cand)

                    hypo = cand_hypo[hypo_idx]
                    hypo_trim = self.vocab.remove_bos_eos_sent(hypo)
                    
                    score = self.sentence_bleu(cand_trim, hypo_trim)
                    cand_score += score
                cand_scores.append(cand_score)
            cand_scores = [score / num_hypos for score in cand_scores]
            order = reversed(list(np.argsort(np.array(cand_scores))))
            cand_scores = sorted(cand_scores, reverse=True)
            cand_all = [cand_hypo[i] for i in order]
            cand_alls.append(cand_all)
            cand_finals.append(list(cand_all[0]))
            cand_alls_scores.append(cand_scores)
        return cand_finals, cand_alls, cand_alls_scores

    def sentence_bleu(self, cand, hypo):
        cand_str = " ".join([tok if isinstance(tok, str) else tok.value for tok in cand])
        hypo_str = " ".join([tok if isinstance(tok, str) else tok.value for tok in hypo])
        bleu_score = self.bleu.sentence_score(cand_str, [hypo_str]).score
        return bleu_score
