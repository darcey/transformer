import re
import math
import numpy as np
import logging
import subprocess
from tempfile import NamedTemporaryFile
from sacrebleu.metrics import BLEU
from configuration import DecodingMethod, MBRMetric
from vocabulary import SpecialTokens

# Calls the generator, and converts its outputs from tensors
# back to lists of tokens. If we're doing MBR, this is also
# where MBR happens.
class OuterGenerator:

    def __init__(self, generator, vocab, config, device, postproc=None):
        self.generator = generator
        self.vocab = vocab
        self.config = config.gen
        self.device = device
        self.postproc_script = postproc

        if self.config.decoding_method == DecodingMethod.MBR:
            if self.config.mbr_metric == MBRMetric.BLEU_DETOK:
                self.bleu = BLEU(smooth_method='add-k', smooth_value=1, effective_order=True)
                logging.getLogger('sacrebleu').setLevel(logging.CRITICAL)
            elif self.config.mbr_metric == MBRMetric.BLEU_TOK:
                self.bleu = BLEU(smooth_method='add-k', smooth_value=1, effective_order=True, tokenize="none")
                logging.getLogger('sacrebleu').setLevel(logging.CRITICAL)

    # If doing MBR, call handle_mbr (which will recursively call this function
    # to actually generate its candidate and hypothesis sets).
    # If not doing MBR, just call the generator.
    def outer_generate(self, src, unbatch_func, config=None):
        config = config if config else self.config
        if config.decoding_method == DecodingMethod.MBR:
            return self.handle_mbr(src, unbatch_func, config)
        else:
            return self.call_generator(src, unbatch_func, config)

    # Call the generator on the source batch,
    # then process the translations from tensors back to tokens.
    # tgt_final: list of lists of tokens
    # tgt_all:   list of lists of lists of tokens
    # probs_all: list of lists of probs
    def call_generator(self, src, unbatch_func, config):
        src = src.to(self.device)
        tgt_final, tgt_all, probs_all = self.generator.generate(src, config)

        tgt_final = tgt_final.cpu()
        tgt_all = tgt_all.cpu()
        probs_all = probs_all.cpu()

        tgt_final, tgt_all, probs_all = unbatch_func(tgt_final, tgt_all, probs_all)
        tgt_final = self.vocab.idx_to_tok_data(tgt_final)
        tgt_final = self.vocab.remove_bos_eos_data(tgt_final)
        tgt_all = self.vocab.idx_to_tok_data(tgt_all, nesting=3)
        tgt_all = self.vocab.remove_bos_eos_data(tgt_all, nesting=3)

        return tgt_final, tgt_all, probs_all

    # Everything below this point is MBR-specific.

    # Handle MBR by first generating the candidate/hypothesis sets
    # (which may themselves be recursively generated via MBR),
    # then running MBR on the candidate/hypothesis sets.
    def handle_mbr(self, src, unbatch_func, config):
        if config.mbr_share_sents:
            tgt_final, tgt_all, probs_all = self.outer_generate(src, unbatch_func, config.share)
            tgt_all_proc = self.process_for_metric(tgt_all)
            cand_final, cand_all, cand_scores_all, cand_probs_all = self.mbr(tgt_all, tgt_all_proc, probs_all, tgt_all, tgt_all_proc, probs_all, config.mbr_metric, config.weight_hypos_equally)
        else:
            cand_final, cand_all, cand_probs_all = self.outer_generate(src, unbatch_func, config.cand)
            hypo_final, hypo_all, hypo_probs_all = self.outer_generate(src, unbatch_func, config.hypo)
            cand_all_proc = self.process_for_metric(cand_all)
            hypo_all_proc = self.process_for_metric(hypo_all)
            cand_final, cand_all, cand_scores_all, cand_probs_all = self.mbr(cand_all, cand_all_proc, cand_probs_all, hypo_all, hypo_all_proc, hypo_probs_all, config.mbr_metric, config.weight_hypos_equally)

        keep = min(len(cand_all[0]), config.num_beams_or_samples)
        cand_all = [cands[:keep] for cands in cand_all]
        cand_probs_all = [probs[:keep] for probs in cand_probs_all]

        return cand_final, cand_all, cand_probs_all

    # The sentences are normally stored in lists of lists of tokens.
    # But the metric may expect them in a different form (a string, maybe detokenized).
    # Convert sentences to this processed form for use by the MBR code.
    def process_for_metric(self, batch):
        num_batch_items = len(batch)
        num_per_batch = len(batch[0])
    
        batch_strs = [[" ".join([tok if isinstance(tok, str) else tok.value for tok in sent]) for sent in sents] for sents in batch]

        if self.postproc_script != None:
            batch_strs_join = "\n".join(["\n".join(sents) for sents in batch_strs])

            temp = NamedTemporaryFile()
            with open(temp.name, 'w') as f:
                f.write(batch_strs_join)
            batch_strs_join = subprocess.run([f"{self.postproc_script} {temp.name}"], shell=True, capture_output=True, text=True).stdout

            batch_strs = batch_strs_join.split("\n")
            batch_strs = [batch_strs[i*num_per_batch:(i+1)*num_per_batch] for i in range(num_batch_items)]

        return batch_strs

    # Run MBR
    def mbr(self, cand_batch, cand_batch_processed, cand_batch_log_probs, hypo_batch, hypo_batch_processed, hypo_batch_log_probs, mbr_metric, weight_hypos_equally):
        cand_final = []
        cand_all = []
        cand_scores_all = []
        cand_probs_all = []

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

    # BLEU metric
    def sentence_bleu(self, cand_str, hypo_str):
        bleu_score = self.bleu.sentence_score(cand_str, [hypo_str]).score
        return bleu_score
