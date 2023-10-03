# TODO(darcey): implement various length rewards for beam search
# TODO(darcey): implement MBR
# TODO(darcey): implement cluster search
# TODO(darcey): implement exact search
# TODO(darcey): implement exact cluster search
# TODO(darcey): implement diverse beam search
# TODO(darcey): implement repetition constraints

# TODO(darcey): consider having the transformer / beam manager return logits vs. probs selectively to save computation
# TODO(darcey): when returning final samples and beam search results should I trim extra padding?
# TODO(darcey): learn where torch uses a copy vs. view, and make sure I am using clone() in all/only the right places
# TODO(darcey): in sampling outer loop, should I move samples off GPU?
# TODO(darcey): consider changing generate() to be a yield-style function, in order to accommodate extremely large numbers of samples where we need to print midway through
# TODO(darcey): come up with a better return type for the generator -- an object? a dict?
# TODO(darcey): update generation to allow transformer to handle things larger than context window?

import warnings
import copy
import torch
from configuration import DecodingMethod, LengthNormalization
from beam_manager import BeamManager
from cache import BeamCache

class Generator:

    def __init__(self, model, config, device, vocab_size, pad_idx, bos_idx, eos_idx):
        self.vocab_size = vocab_size
        self.pad = pad_idx
        self.bos = bos_idx
        self.eos = eos_idx

        self.device = device
        self.config = config.gen
        self.window = config.arch.context_window_length
        self.num_layers = config.arch.num_decoder_layers
        self.model = model

    # src: [batch_size, src_len]
    # return:
    #   tgt_final: [batch_size, tgt_len]
    #   tgt_all:   [batch_size, beam_size, tgt_len]
    #   probs_all: [batch_size, beam_size]
    def generate(self, src):
        max_lengths, max_possible_length = self.get_max_lengths(src)

        batch = src.size(0)
        cache = BeamCache(batch, 1, self.num_layers, self.device)

        autoregressive_fn = self.model.get_autoregressive_one_step_fn(src, cache)
        match self.config.decoding_method:
            case DecodingMethod.SAMPLING:
                return self.sample_outer_loop(src.size(0), max_lengths, max_possible_length, autoregressive_fn, cache)
            case DecodingMethod.BEAM_SEARCH:
                return self.beam_search(src.size(0), self.config.num_beams_or_samples, max_lengths, max_possible_length, autoregressive_fn, cache)

    # src:         [batch_size, src_len]
    # max_lengths: [batch_size]
    def get_max_lengths(self, src):
        batch_size = src.size(0)

        if self.config.use_rel_max_len:
            max_lengths = torch.sum(src != self.pad, dim=-1) + self.config.rel_max_len
        else:
            max_lengths = torch.tensor([self.config.abs_max_len] * batch_size, device=self.device)

        min_max_length = max_lengths.min().item()
        if min_max_length < 1:
            raise Exception("Max length should be at least 1 to allow for BOS")

        max_possible_length = max_lengths.max().item()
        if max_possible_length > self.window:
            warnings.warn("Warning: some sentences have max length higher than the context window length. Reducing max length to context window length.")
            max_lengths = torch.minimum(max_lengths, torch.tensor(self.window))
            max_possible_length = self.window

        return max_lengths, max_possible_length

    # max_lengths: [batch_size]
    # return:
    #   final_samples: [batch_size, tgt_len]
    #   all_samples:   [batch_size, beam_size, tgt_len]
    #   all_probs:     [batch_size, beam_size]
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
        beam_manager = BeamManager(batch_size=batch_size,
                                   beam_size=num_samples,
                                   vocab_size=self.vocab_size,
                                   max_lengths=max_lengths,
                                   max_possible_length=max_possible_length,
                                   pad=self.pad,
                                   bos=self.bos,
                                   eos=self.eos,
                                   autoregressive_fn=autoregressive_fn,
                                   cache=cache,
                                   device=self.device)

        while True:
            beam_manager.prune_finished()
            if beam_manager.all_done():
                break

            # get relevant info for choosing the next timestep's tokens
            next_token_logits, _, _ = beam_manager.compute_next_token_probs()      # [batch, num_samples, V]

            # adjust probs as needed and sample next token
            next_token_probs = self.process_logits(next_token_logits)              # [batch, num_samples, V]
            next_token_probs = self.truncate_probs(next_token_probs)               # [batch, num_samples, V]
            curr_size = next_token_probs.size(0)
            next_token_probs = next_token_probs.reshape(curr_size*num_samples, -1) # [batch*num_samples, 1]
            chosen_idxs = torch.multinomial(next_token_probs, 1, replacement=True) # [batch*num_samples, 1]
            chosen_idxs = chosen_idxs.reshape(curr_size, num_samples)              # [batch, num_samples]

            # tell the beam manager which idxs we're keeping
            beam_manager.select_idxs_independent(chosen_idxs)

        return beam_manager.get_final()

    # logits: [batch, num_samples, V]
    # ret:    [batch, num_samples, V]
    def process_logits(self, logits):
        logits = logits / self.config.sampling_temp
        return torch.nn.functional.softmax(logits, dim=-1)

    # probs: [batch, num_samples, V]
    # ret:   [batch, num_samples, V]
    def truncate_probs(self, probs):
        # If not doing top-p or top-k, this is a no-op
        if (self.config.sampling_p == 1.0) and (self.config.sampling_k <= 0):
            return probs

        # Based on Ari Holtzman's implementation
        # https://github.com/ari-holtzman/degen/blob/master/gen.py
        vals, idxs = torch.sort(probs, dim=-1, descending=True)
        cum_vals = torch.cumsum(vals, dim=-1)
        to_remove_p = cum_vals >= self.config.sampling_p
        to_remove_p[:,:,1:] = to_remove_p[:,:,:-1].clone()
        to_remove_p[:,:,0] = False
        to_remove_k = torch.zeros_like(to_remove_p)
        k = probs.size(-1)
        if self.config.sampling_k > 0:
            k = min(k, self.config.sampling_k)
        to_remove_k[:,:,k:] = True
        to_remove = to_remove_p + to_remove_k
        vals[to_remove] = 0.0
        trunc_probs = torch.empty_like(probs)
        trunc_probs.scatter_(dim=-1, index=idxs, src=vals)
        return trunc_probs

    # max_lengths: [batch_size]
    # ret_symbols: [batch_size, beam_size, tgt_len]
    # ret_probs:   [batch_size, beam_size]
    def beam_search(self, batch_size, beam_size, max_lengths, max_possible_length, autoregressive_fn, cache):
        beam_manager = BeamManager(batch_size=batch_size,
                                   beam_size=beam_size,
                                   vocab_size=self.vocab_size,
                                   max_lengths=max_lengths,
                                   max_possible_length=max_possible_length,
                                   pad=self.pad,
                                   bos=self.bos,
                                   eos=self.eos,
                                   autoregressive_fn=autoregressive_fn,
                                   cache=cache,
                                   device=self.device)

        # beam should not contain duplicates, so at the beginning,
        # start with just one BOS in the beam, and the rest of the
        # beam should be filled with dummy sentences that are just PAD
        start_symbols = torch.full((batch_size, beam_size, 1), self.pad, device=self.device)
        start_symbols[:,0,:] = self.bos
        start_probs = torch.full((batch_size, beam_size), float("-inf"), device=self.device)
        start_probs[:,0] = 0.0
        beam_manager.manually_initialize(start_symbols, start_probs)

        # in each iteration, get the probabilities of all choices
        # from the beam manager, then choose the top k
        time_step = 0
        while True:
            # tell the beam manager to prune anything that ends with EOS
            # since we don't have to do computation on it anymore;
            # when everything is pruned, can return
            beam_manager.prune_finished()
            if beam_manager.all_done():
                break

            # tell the beam manager to extend the beams by one token,
            # and return the probabilities of all the possible new beams
            _, next_token_probs, all_choices_cumulative_probs = beam_manager.compute_next_token_probs() # [batch, beam, vocab]

            # modify the beams' probs as needed
            if time_step == 0 and not self.config.allow_empty_string:
                all_choices_cumulative_probs[:,:,self.eos] = float("-inf")
            all_choices_cumulative_probs = self.length_normalize(beam_manager, all_choices_cumulative_probs)

            # for finished sentences (e.g. ones that end in PAD),
            # the beam manager will set the next token log probability
            # of PAD to be 0, and everything else to be -inf.
            # but this runs into trouble with the dummy sentences that are all PAD:
            # since their cumulative probability is already -inf,
            # it doesn't matter what the next token probability is.
            # so, find the sentences that are all PAD, set the probability of PAD
            # to something slightly higher than -inf, so PAD will be chosen as
            # the next token.
            all_pad = (all_choices_cumulative_probs == float("-inf"))
            min_val = all_choices_cumulative_probs[~all_pad].min() - 10
            all_pad_plus_pad = (next_token_probs == 0.0) * all_pad
            all_choices_cumulative_probs[all_pad_plus_pad] = min_val

            # reshape, choose top k
            batch_size = all_choices_cumulative_probs.size(0)                                       # (may have changed if sents were pruned)
            all_choices_cumulative_probs = all_choices_cumulative_probs.reshape(batch_size, -1)     # [batch, beam*V]
            chosen_probs, chosen_idxs = torch.topk(all_choices_cumulative_probs, beam_size, dim=-1) # ([batch, beam], [batch, beam])

            # tell the beam manager which of the possible beams to keep
            # note that each index in chosen_idxs is a value (beam_item*V + vocab_item)
            # indicating that the extended beam item to keep is [batch_idx, beam_item, vocab_item]
            beam_manager.select_idxs_dependent(chosen_idxs)

            time_step += 1

        symbols, probs = beam_manager.get_final()
        return symbols[:,0,:].clone(), symbols, probs

    def length_normalize(self, beam_manager, log_probs):
        if self.config.length_normalization == LengthNormalization.NONE:
            return log_probs

        lengths = beam_manager.get_all_choices_lengths()
        if self.config.length_normalization == LengthNormalization.LENGTH_REWARD:
            return log_probs + lengths * self.config.length_reward_gamma
        elif self.config.length_normalization == LengthNormalization.LENGTH_NORM:
            return log_probs / lengths
        elif self.config.length_normalization == LengthNormalization.GOOGLE_METHOD:
            alpha = self.config.length_norm_alpha
            return log_probs / ((5.0 + lengths) ** alpha / 6.0 ** alpha)
