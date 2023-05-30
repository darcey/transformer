# TODO(darcey): copy beam search code over from Toan's code
# TODO(darcey): implement MBR
# TODO(darcey): implement cluster search
# TODO(darcey): implement exact search
# TODO(darcey): implement exact cluster search
# TODO(darcey): implement temperature for sampling
# TODO(darcey): implement diverse beam search
# TODO(darcey): implement repetition constraints

# TODO(darcey): learn where torch uses a copy vs. view, and make sure I am using clone() in all/only the right places
# TODO(darcey): think about whether there's a way to decrease space usage during truncated sampling (probably not, since top-p is ragged)
# TODO(darcey): consider whether to make top-p, top-k, and temperature usable at the same time
# TODO(darcey): consider whether beam search and sampling should multiply in the EOS probability if they reach the max length
# TODO(darcey): also should max length include BOS?
# TODO(darcey): in sampling outer loop, should I move samples off GPU?
# TODO(darcey): consider changing generate() to be a yield-style function, in order to accommodate extremely large numbers of samples where we need to print midway through
# TODO(darcey): come up with a better return type for the generator -- an object? a dict?

import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from configuration import DecodingMethod, SamplingMethod
from cache import BeamCache

class Generator:


    def __init__(self, model, config, device, pad_idx, bos_idx, eos_idx):
        self.pad = pad_idx
        self.bos = bos_idx
        self.eos = eos_idx

        self.device = device
        self.config = config.gen
        self.model = model

    # src: [batch_size, length]
    def generate(self, src):
        batch_size = src.size(0)

        if self.config.use_rel_max_len:
            max_lengths = torch.sum(src != self.pad, dim=-1) + self.config.rel_max_len
        else:
            max_lengths = torch.tensor([self.config.abs_max_len] * batch_size, device=self.device)
        max_possible_length = max_lengths.max().item()

        cache = BeamCache()
        autoregressive_fn = self.model.get_autoregressive_one_step_fn(src, cache)
        match self.config.decoding_method:
            case DecodingMethod.SAMPLING:
                return self.sample_outer_loop(src.size(0), max_lengths, max_possible_length, autoregressive_fn, cache)

    def sample_outer_loop(self, batch_size, max_lengths, max_possible_length, autoregressive_fn, cache):
        max_sents     = self.config.max_parallel_sentences
        num_samples   = self.config.num_beams_or_samples
        total_samples = batch_size * num_samples

        if total_samples <= max_sents:
            all_samples, all_probs = self.sample(batch_size, num_samples, max_lengths, max_possible_length, autoregressive_fn, cache)
            final_samples = all_samples[:,0,:].clone()
            return final_samples, all_samples, all_probs

        if batch_size > 1:
            raise Exception("If number of samples exceeds maximum number of parallel sentences, batch size must be 1")

        all_samples = []
        all_probs = []
        while total_samples > 0:
            curr_num_samples = min(total_samples, max_sents)
            cache_copy = copy.deepcopy(cache)
            samples, probs = self.sample(1, curr_num_samples, max_lengths.clone(), max_possible_length, autoregressive_fn, cache_copy)
            all_samples.append(samples)
            all_probs.append(probs)
            total_samples -= curr_num_samples

        max_seq_len   = max([samples.size(2) for samples in all_samples])
        all_samples   = [torch.nn.functional.pad(samples, (0, max_seq_len - samples.size(2)), value=self.pad) for samples in all_samples]
        all_samples   = torch.cat(all_samples, dim=1)
        final_samples = all_samples[:,0,:].clone()
        all_probs     = torch.cat(all_probs, dim=1)
        return final_samples, all_samples, all_probs

    # max_lengths: [batch_size]
    # ret_symbols: [batch_size, num_samples, tgt_len]
    # ret_probs:   [batch_size, num_samples]
    def sample(self, batch_size, num_samples, max_lengths, max_possible_length, autoregressive_fn, cache):
        size = batch_size * num_samples
        cumulative_symbols = torch.tensor([self.bos] * size, device=self.device).unsqueeze(1) # [size, tgt_seq=1]
        cumulative_probs   = torch.zeros(size=(size, 1), device=self.device)                  # [size, dummy dimension for V]
        max_lengths        = max_lengths.unsqueeze(1).expand(-1, num_samples).reshape(size)   # [size]
        cache.expand_to_num_samples(num_samples)

        ret_symbols = [None] * size
        ret_probs   = [None] * size
        orig_idxs   = torch.arange(size, device=self.device)

        # at beginning of timestep n, n tokens have been generated so far (not incl. BOS)
        for time_step in range(0, max_possible_length + 1):
            # when a sample is finished, can stop doing computation on it
            reached_max_length = (max_lengths <= time_step) + (time_step == max_possible_length) # [size]
            reached_eos        = (cumulative_symbols[:, -1] == self.eos)                           # [size]
            finished_sents     = reached_max_length + reached_eos                                  # [size]

            if finished_sents.any():
                for j in range(finished_sents.size(0)):
                    if finished_sents[j]:
                        ret_symbols[orig_idxs[j]] = cumulative_symbols[j].clone()
                        ret_probs[orig_idxs[j]]   = cumulative_probs[j].clone()

                # size = size - number of finished sents
                cumulative_symbols = cumulative_symbols[~finished_sents] # [size, tgt_seq]
                cumulative_probs   = cumulative_probs[~finished_sents]   # [size, 1]
                max_lengths        = max_lengths[~finished_sents]        # [size]
                orig_idxs          = orig_idxs[~finished_sents]          # [size]
                cache.trim_finished_sents(finished_sents)

            if finished_sents.all():
                break

            # compute next token probabilities
            next_token_probs = autoregressive_fn(cumulative_symbols, cache)    # [size, V]
            all_choices_cumulative_probs = cumulative_probs + next_token_probs # [size, 1] + [size, V] = [size, V]

            # convert from log probs to probs; do any truncation etc.
            next_token_probs = torch.exp(next_token_probs)
            next_token_probs = self.adjust_or_truncate_probs(next_token_probs)

            # sample next token
            chosen_idxs = torch.multinomial(next_token_probs, 1, replacement=True)         # [size, 1]
            cumulative_probs = torch.gather(all_choices_cumulative_probs, -1, chosen_idxs) # [size, 1]
            cumulative_symbols = torch.cat((cumulative_symbols, chosen_idxs), -1)          # [size, tgt_len]

        # get return values into proper format
        ret_symbols = pad_sequence(ret_symbols, batch_first=True, padding_value=self.pad).reshape(batch_size, num_samples, -1)
        ret_probs   = torch.stack(ret_probs).reshape(batch_size, num_samples)
        return ret_symbols, ret_probs

    def adjust_or_truncate_probs(self, probs):
        match self.config.sampling_method:
            case SamplingMethod.ANCESTRAL:
                return probs
            case SamplingMethod.TOP_K:
                vals, idxs = torch.topk(probs, k=self.config.sampling_k, dim=-1)
                trunc_probs = torch.zeros_like(probs)
                trunc_probs.scatter_(dim=-1, index=idxs, src=vals)
                return trunc_probs
            # Based on Ari Holtzman's implementation
            # https://github.com/ari-holtzman/degen/blob/master/gen.py
            case SamplingMethod.TOP_P:
                vals, idxs = torch.sort(probs, dim=-1, descending=True)
                cum_vals = torch.cumsum(vals, dim=-1)
                to_remove = cum_vals > self.config.sampling_p
                to_remove[:, 1:] = to_remove[:,:-1].clone()
                to_remove[:, 0] = 0.0
                vals[to_remove] = 0.0
                trunc_probs = torch.empty_like(probs)
                trunc_probs.scatter_(dim=-1, index=idxs, src=vals)
                return trunc_probs
