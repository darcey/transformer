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

        # this is just used in MBR
        self.bleu = BLEU(smooth_method='add-k', smooth_value=1)
        logging.getLogger('sacrebleu').setLevel(logging.CRITICAL)

    def outer_generate(self, src, unbatch_func):
        tgt_final, tgt_all, probs_all = self.middle_outer_generate(src, unbatch_func)
        tgt_final = self.vocab.remove_bos_eos_data(tgt_final)
        return tgt_final, tgt_all, probs_all

    def middle_outer_generate(self, src, unbatch_func, config=None):
        config = config if config else self.config
        if config.decoding_method == DecodingMethod.MBR:
            return self.handle_mbr(src, unbatch_func, config)
        else:
            return self.inner_outer_generate(src, unbatch_func, config)

    def inner_outer_generate(self, src, unbatch_func, config):
        src = src.to(self.device)
        tgt_final, tgt_all, probs_all = self.generator.generate(src, config)

        tgt_final = tgt_final.cpu()
        tgt_all = tgt_all.cpu()
        probs_all = probs_all.cpu()

        tgt_final, tgt_all, probs_all = unbatch_func(tgt_final, tgt_all, probs_all)
        tgt_final = self.vocab.idx_to_tok_data(tgt_final)
        tgt_all = self.vocab.idx_to_tok_data(tgt_all, nesting=3)
        
        return tgt_final, tgt_all, probs_all

    def handle_mbr(self, src, unbatch_func, config):
        if config.mbr_share_sents:
            tgt_final, tgt_all, probs_all = self.middle_outer_generate(src, unbatch_func, config.share)
            cand_final, cand_all, cand_scores_all, cand_probs_all = self.mbr(tgt_all, probs_all, tgt_all, probs_all, config.weight_hypos_equally)
        else:
            cand_final, cand_all, cand_probs_all = self.middle_outer_generate(src, unbatch_func, config.cand)
            hypo_final, hypo_all, hypo_probs_all = self.middle_outer_generate(src, unbatch_func, config.hypo)
            cand_final, cand_all, cand_scores_all, cand_probs_all = self.mbr(cand_all, cand_probs_all, hypo_all, hypo_probs_all, config.weight_hypos_equally)

        keep = min(len(cand_all[0]), config.num_beams_or_samples)
        cand_all = [cands[:keep] for cands in cand_all]
        cand_probs_all = [probs[:keep] for probs in cand_probs_all]

        return cand_final, cand_all, cand_probs_all

    def mbr(self, cand_batch, cand_batch_probs, hypo_batch, hypo_batch_probs, weight_hypos_equally):
        cand_final = []
        cand_all = []
        cand_scores_all = []
        cand_probs_all = []
        for cands, cand_probs, hypos, hypo_probs in zip(cand_batch, cand_batch_probs, hypo_batch, hypo_batch_probs):
            num_cands = len(cands)
            num_hypos = len(hypos)

            cand_scores = []
            for cand_idx in range(num_cands):
                cand_score = 0.0
                for hypo_idx in range(num_hypos):

                    cand = cands[cand_idx]
                    cand_trim = self.vocab.remove_bos_eos_sent(cand)

                    hypo = hypos[hypo_idx]
                    hypo_trim = self.vocab.remove_bos_eos_sent(hypo)
                    
                    score = self.sentence_bleu(cand_trim, hypo_trim)
                    if not weight_hypos_equally:
                        score *= hypo_probs[hypo_idx]
                    cand_score += score
                cand_scores.append(cand_score)
            if weight_hypos_equally:
                cand_scores = [score / num_hypos for score in cand_scores]
            order = list(reversed(list(np.argsort(np.array(cand_scores)))))
            cands = [cands[i] for i in order]
            cand_all.append(cands)
            cand_final.append(list(cands[0]))
            cand_scores = sorted(cand_scores, reverse=True)
            cand_scores_all.append(cand_scores)
            cand_probs = [cand_probs[i] for i in order]
            cand_probs_all.append(cand_probs)
        return cand_final, cand_all, cand_scores_all, cand_probs_all

    def sentence_bleu(self, cand, hypo):
        cand_str = " ".join([tok if isinstance(tok, str) else tok.value for tok in cand])
        hypo_str = " ".join([tok if isinstance(tok, str) else tok.value for tok in hypo])
        bleu_score = self.bleu.sentence_score(cand_str, [hypo_str]).score
        return bleu_score
