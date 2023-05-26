# TODO(darcey): copy beam search code over from Toan's code
# TODO(darcey): implement MBR
# TODO(darcey): implement cluster search

# TODO(darcey): implement exact search
# TODO(darcey): implement exact cluster search
# TODO(darcey): consider whether beam search and sampling should multiply in the EOS probability if they reach the max length
# TODO(darcey): also should max length include BOS?
# TODO(darcey): in sampling outer loop, should I move samples off GPU?
# TODO(darcey): consider changing generate() to be a yield-style function, in order to accommodate extremely large numbers of samples where we need to print midway through

import copy
import torch
from torch.nn.utils.rnn import pad_sequence
from configuration import DecodingMethod
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
        autoregressive_fn = model.get_autoregressive_one_step_fn(src, cache)
        match config.decoding_method:
            case DecodingMethod.SAMPLING:
                return sample_outer_loop(src.size(0), max_lengths, max_possible_length, autoregressive_fn, cache)

    def sample_outer_loop(self, batch_size, max_lengths, max_possible_length, autoregressive_fn, cache):
        max_sents     = self.config.max_parallel_sentences
        num_samples   = self.config.num_beams_or_samples
        total_samples = batch_size * num_samples

        if total_samples < max_sents:
            return self.sample(batch_size, num_samples, max_lengths, max_possible_length, autoregressive_fn, cache)

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

        max_seq_len = max([samples.size(2) for samples in all_samples])
        all_samples = [torch.nn.functional.pad(samples, (0, max_seq_len - samples.size(2)), value=self.pad) for samples in all_samples]
        all_samples = torch.cat(all_samples, dim=1)
        all_probs   = torch.cat(all_probs, dim=1)
        return all_samples, all_probs

    # max_lengths: [batch_size]
    # ret_symbols: [batch_size, num_samples, tgt_len]
    # ret_probs:   [batch_size, num_samples]
    def sample(self, batch_size, num_samples, max_lengths, max_possible_length, autoregressive_fn, cache):
        size = batch_size * num_samples
        cumulative_symbols = torch.tensor([self.bos] * size, device=self.device).unsqueeze(1) # [size, tgt_seq=1]
        cumulative_probs   = torch.zeros(size=(size, 1), device=self.device)                  # [size, dummy dimension for V]
        max_lengths        = max_lengths.unsqueeze(1).expand(-1, num_samples).reshape(size)   # [size]
        cache.expand(num_samples)

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

            # sample next token
            chosen_idxs = torch.multinomial(torch.exp(next_token_probs), 1, replacement=True) # [size, 1]
            cumulative_probs = torch.gather(all_choices_cumulative_probs, -1, chosen_idxs)    # [size, 1]
            cumulative_symbols = torch.cat((cumulative_symbols, chosen_idxs), -1)             # [size, tgt_len]

        # get return values into proper format
        ret_symbols = pad_sequence(ret_symbols, batch_first=True, padding_value=self.pad).reshape(batch_size, num_samples, -1)
        ret_probs   = torch.stack(ret_probs).reshape(batch_size, num_samples)
        return ret_symbols, ret_probs
