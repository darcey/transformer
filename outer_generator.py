import re
import math
import numpy as np
import logging
import subprocess
import os
from tempfile import NamedTemporaryFile
from sacrebleu.metrics import BLEU
from configuration import DecodingMethod, MBRMetric
from vocabulary import SpecialTokens

class OuterGenerator:

    def __init__(self, generator, vocab, config, device):
        self.generator = generator
        self.vocab = vocab
        self.config = config.gen
        self.device = device

        if self.config.decoding_method == DecodingMethod.MBR:
            if self.config.mbr_metric == MBRMetric.BLEU_DETOK:
                self.bleu = BLEU(smooth_method='add-k', smooth_value=1, effective_order=True)
                logging.getLogger('sacrebleu').setLevel(logging.CRITICAL)
            elif self.config.mbr_metric == MBRMetric.BLEU_TOK or self.config.mbr_metric == MBRMetric.BLEU_BPE:
                self.bleu = BLEU(smooth_method='add-k', smooth_value=1, effective_order=True, tokenize="none")
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

    # returns sentences as lists of BPE tokens
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
            cand_final, cand_all, cand_scores_all, cand_probs_all = self.mbr(tgt_all, probs_all, tgt_all, probs_all, config.mbr_metric, config.weight_hypos_equally)
        else:
            cand_final, cand_all, cand_probs_all = self.middle_outer_generate(src, unbatch_func, config.cand)
            hypo_final, hypo_all, hypo_probs_all = self.middle_outer_generate(src, unbatch_func, config.hypo)
            cand_final, cand_all, cand_scores_all, cand_probs_all = self.mbr(cand_all, cand_probs_all, hypo_all, hypo_probs_all, config.mbr_metric, config.weight_hypos_equally)

        keep = min(len(cand_all[0]), config.num_beams_or_samples)
        cand_all = [cands[:keep] for cands in cand_all]
        cand_probs_all = [probs[:keep] for probs in cand_probs_all]

        return cand_final, cand_all, cand_probs_all

    # cand_batch and hypo_batch are lists of BPE tokens
    def mbr(self, cand_batch, cand_batch_log_probs, hypo_batch, hypo_batch_log_probs, mbr_metric, weight_hypos_equally):
        cand_final = []
        cand_all = []
        cand_scores_all = []
        cand_probs_all = []

        cand_batch_processed, hypo_batch_processed = self.process_for_metric(cand_batch, hypo_batch, mbr_metric)

        cand_batch_probs = [[math.exp(log_prob) for log_prob in cand_log_probs] for cand_log_probs in cand_batch_log_probs]
        hypo_batch_probs = [[math.exp(log_prob) for log_prob in hypo_log_probs] for hypo_log_probs in hypo_batch_log_probs]
        for cands, cand_strs, cand_probs, hypos, hypo_strs, hypo_probs in zip(cand_batch, cand_batch_processed, cand_batch_probs, hypo_batch, hypo_batch_processed, hypo_batch_probs):
            num_cands = len(cands)
            num_hypos = len(hypos)

            cand_scores = []
            for cand_idx in range(num_cands):
                cand_score = 0.0
                for hypo_idx in range(num_hypos):
                    cand = cand_strs[cand_idx]
                    hypo = hypo_strs[hypo_idx]

                    score = self.sentence_bleu(cand, hypo)

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

    def process_for_metric(self, cand_batch, hypo_batch, mbr_metric):
        cand_batch = self.vocab.remove_bos_eos_data(cand_batch, nesting=3)
        hypo_batch = self.vocab.remove_bos_eos_data(hypo_batch, nesting=3)
        
        assert(all([(all([not "\n" in hypo for hypo in hypos]) for hypos in hypo_batch)]))

        cand_batch_strs = [[" ".join([tok if isinstance(tok, str) else tok.value for tok in cand]) for cand in cands] for cands in cand_batch]
        hypo_batch_strs = [[" ".join([tok if isinstance(tok, str) else tok.value for tok in hypo]) for hypo in hypos] for hypos in hypo_batch]

        cand_batch_strs_join = "\n\n".join(["\n".join(cands) for cands in cand_batch_strs])
        hypo_batch_strs_join = "\n\n".join(["\n".join(hypos) for hypos in hypo_batch_strs])

        if mbr_metric == MBRMetric.BLEU_TOK or mbr_metric == MBRMetric.BLEU_DETOK:
            bpe = re.compile('(@@ )|(@@ ?$)')
            cand_batch_strs_join = bpe.sub("", cand_batch_strs_join)
            hypo_batch_strs_join = bpe.sub("", hypo_batch_strs_join)
            
        if mbr_metric == MBRMetric.BLEU_DETOK:
            # hard coding this until I know whether it's useful
            detok = "/afs/crc.nd.edu/group/nlp/software/moses/3.0/mosesdecoder/scripts/tokenizer/detokenizer.perl"
            cand_batch_strs_join = subprocess.run([detok, "-q", "-l en"], input=cand_batch_strs_join, capture_output=True, text=True).stdout
            hypo_batch_strs_join = subprocess.run([detok, "-q", "-l en"], input=hypo_batch_strs_join, capture_output=True, text=True).stdout

        cand_batch_strs = [cands_join.split("\n") for cands_join in cand_batch_strs_join.split("\n\n")]
        hypo_batch_strs = [hypos_join.split("\n") for hypos_join in hypo_batch_strs_join.split("\n\n")]
        return cand_batch_strs, hypo_batch_strs

    def sentence_bleu(self, cand_str, hypo_str):
        bleu_score = self.bleu.sentence_score(cand_str, [hypo_str]).score
        return bleu_score
