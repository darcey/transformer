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
from beam_manager import BeamManager, BeamManagerNoBall
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
    def generate(self, src, config=None):
        config = config if config else self.config
        max_lengths, max_possible_length = self.get_max_lengths(config, src)

        batch = src.size(0)
        cache = BeamCache(batch, 1, self.num_layers, self.device)

        autoregressive_fn = self.model.get_autoregressive_one_step_fn(src, cache)
        match config.decoding_method:
            case DecodingMethod.SAMPLING:
                return self.sample_outer_loop(config, src.size(0), max_lengths, max_possible_length, autoregressive_fn, cache)
            case DecodingMethod.BEAM_SEARCH:
                return self.beam_search(config, src.size(0), config.num_beams_or_samples, max_lengths, max_possible_length, autoregressive_fn, cache)
            case DecodingMethod.MBR:
                raise Exception("The decoding method passed to 'generate' should not be MBR.")

    # src:         [batch_size, src_len]
    # max_lengths: [batch_size]
    def get_max_lengths(self, config, src):
        batch_size = src.size(0)

        if config.use_rel_max_len:
            max_lengths = torch.sum(src != self.pad, dim=-1) + config.rel_max_len
        else:
            max_lengths = torch.tensor([config.abs_max_len] * batch_size, device=self.device)

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
    def sample_outer_loop(self, config, batch_size, max_lengths, max_possible_length, autoregressive_fn, cache):
        max_sents     = config.max_parallel_sentences
        num_samples   = config.num_beams_or_samples
        total_samples = batch_size * num_samples

        if total_samples <= max_sents:
            all_samples, all_probs = self.sample(config, batch_size, num_samples, max_lengths, max_possible_length, autoregressive_fn, cache)
            final_samples = all_samples[:,0,:].clone()
            return final_samples, all_samples, all_probs

        if batch_size > 1:
            raise Exception("If number of samples exceeds maximum number of parallel sentences, batch size must be 1")

        all_samples = []
        all_probs = []
        while total_samples > 0:
            curr_num_samples = min(total_samples, max_sents)
            cache_copy = copy.deepcopy(cache)
            samples, probs = self.sample(config, 1, curr_num_samples, max_lengths.clone(), max_possible_length, autoregressive_fn, cache_copy)
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
    def sample(self, config, batch_size, num_samples, max_lengths, max_possible_length, autoregressive_fn, cache):
        beam_manager = BeamManagerNoBall(batch_size=batch_size,
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
            next_token_probs = self.process_logits(config, next_token_logits)      # [batch, num_samples, V]
            next_token_probs = self.truncate_probs(config, next_token_probs)       # [batch, num_samples, V]
            curr_size = next_token_probs.size(0)
            next_token_probs = next_token_probs.reshape(curr_size*num_samples, -1) # [batch*num_samples, 1]
            chosen_idxs = torch.multinomial(next_token_probs, 1, replacement=True) # [batch*num_samples, 1]
            chosen_idxs = chosen_idxs.reshape(curr_size, num_samples)              # [batch, num_samples]

            # tell the beam manager which idxs we're keeping
            beam_manager.select_idxs_sampling(chosen_idxs)

        return beam_manager.get_final()

    # logits: [batch, num_samples, V]
    # ret:    [batch, num_samples, V]
    def process_logits(self, config, logits):
        logits = logits / config.sampling_temp
        return torch.nn.functional.softmax(logits, dim=-1)

    # probs: [batch, num_samples, V]
    # ret:   [batch, num_samples, V]
    def truncate_probs(self, config, probs):
        # If not doing top-p or top-k, this is a no-op
        if (config.sampling_p == 1.0) and (config.sampling_k <= 0):
            return probs

        # Based on Ari Holtzman's implementation
        # https://github.com/ari-holtzman/degen/blob/master/gen.py
        vals, idxs = torch.sort(probs, dim=-1, descending=True)
        cum_vals = torch.cumsum(vals, dim=-1)
        to_remove_p = cum_vals >= config.sampling_p
        to_remove_p[:,:,1:] = to_remove_p[:,:,:-1].clone()
        to_remove_p[:,:,0] = False
        to_remove_k = torch.zeros_like(to_remove_p)
        k = probs.size(-1)
        if config.sampling_k > 0:
            k = min(k, config.sampling_k)
        to_remove_k[:,:,k:] = True
        to_remove = to_remove_p + to_remove_k
        vals[to_remove] = 0.0
        trunc_probs = torch.empty_like(probs)
        trunc_probs.scatter_(dim=-1, index=idxs, src=vals)
        return trunc_probs

    # max_lengths: [batch_size]
    # ret_symbols: [batch_size, beam_size, tgt_len]
    # ret_probs:   [batch_size, beam_size]
    def beam_search(self, config, batch_size, beam_size, max_lengths, max_possible_length, autoregressive_fn, cache):
        beam_manager = BeamManagerNoBall(batch_size=batch_size,
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
            if time_step == 0 and not config.allow_empty_string:
                all_choices_cumulative_probs[:,:,self.eos] = float("-inf")
            all_choices_cumulative_probs = self.length_normalize(config, beam_manager, all_choices_cumulative_probs)

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
            beam_manager.select_idxs_beam(chosen_idxs)

            time_step += 1

        symbols, probs = beam_manager.get_final()
        return symbols[:,0,:].clone(), symbols, probs

    def length_normalize(self, config, beam_manager, log_probs):
        if config.length_normalization == LengthNormalization.NONE:
            return log_probs

        lengths = beam_manager.get_all_choices_lengths()
        if config.length_normalization == LengthNormalization.LENGTH_REWARD:
            return log_probs + lengths * config.length_reward_gamma
        elif config.length_normalization == LengthNormalization.LENGTH_NORM:
            return log_probs / lengths
        elif config.length_normalization == LengthNormalization.GOOGLE_METHOD:
            alpha = config.length_norm_alpha
            return log_probs / ((5.0 + lengths) ** alpha / 6.0 ** alpha)

    # max_lengths: [batch_size]
    # ret_symbols: ????
    # ret_probs:   ????
    def cluster_beam_search(self, config, batch_size, beam_size, ball_size, max_lengths, max_possible_length, distance_class, autoregressive_fn, cache):
        # The beam items + ball items will be stored in a tensor of size [batch, beam, ball+1, ...]
        # where elements [:, :, 0, ...] are beam items and [:, :, 1:, ...] are the ball items.

        beam_manager = BeamManager(batch_size=batch_size,
                                   beam_size=beam_size,
                                   ball_size=ball_size,
                                   vocab_size=self.vocab_size,
                                   max_lengths=max_lengths,
                                   max_possible_length=max_possible_length,
                                   pad=self.pad,
                                   bos=self.bos,
                                   eos=self.eos,
                                   autoregressive_fn=autoregressive_fn,
                                   cache=cache,
                                   device=self.device)
        distance = distance_class(batch_size=batch_size,
                                  beam_size=beam_size,
                                  ball_size=ball_size,
                                  vocab_size=self.vocab_size,
                                  beam_manager=beam_manager)

        # At the start, the beam will just contain one item (the empty string)
        # and its ball will not contain anything. The rest of the beam / ball
        # items should be filled with dummy sentences that are just PAD.
        start_symbols = torch.full((batch_size, beam_size, ball_size+1, 1), self.pad, device=self.device)
        start_symbols[:,0,0,:] = self.bos
        start_probs = torch.full((batch_size, beam_size, ball_size+1), float("-inf"), device=self.device)
        start_probs[:,0,0] = 0.0
        beam_manager.manually_initialize(start_symbols, start_probs)

        time_step = 0
        while True:
            # Prune things from the batch when every beam item is finished
            # (this ignores ball items, which do not need to be finished).
            beam_manager.prune_finished()
            if beam_manager.all_done():
                break

            # Tell the beam manager to extend all the beam and ball items by one token.
            _, _, all_choices_cumulative_probs = beam_manager.compute_next_token_probs() # [batch, beam, ball+1, vocab]
            batch_size = all_choices_cumulative_probs.size(0)                            # (may have changed if sents were pruned)

            # Compute the distance-weighted probabilities that will be used to score the ball items.
            distances = distance.get_distance_estimates()                                    # [batch, beam, vocab, ball+1, vocab]
            ball_distance_weights = torch.exp(-config.cluster_beam_search_alpha * distances) # [batch, beam, vocab, ball+1, vocab]
            ball_scores = all_choices_cumulative_probs.unsqueeze(2) * ball_distance_weights  # [batch, beam, vocab, ball+1, vocab]
            ball_scores_old = ball_scores.clone()
            # If a sentence contains PAD followed by a non-PAD token, it will have distance
            # of infinity and probability of -infinity, as it is invalid. It is still better
            # to select these (since the rest of the code is equipped to handle invalid stuff)
            # than to select a ball item which is the same as the beam item, so set these to a
            # low value that isn't quite -infinity.
            invalid = torch.logical_or(torch.isnan(ball_scores), ball_scores == float("-inf"))
            min_val = ball_scores[~invalid].min() - 10
            ball_scores[invalid] = min_val
            ball_scores_old[invalid] = float("-inf")
            # For each new beam item, there will be one item in the ball that's the same as the beam item;
            # set the score of this to -infinity so it cannot be chosen.
            ball_same_as_beam_mask = torch.full((batch_size, beam_size, ball_size+1, self.vocab_size, self.vocab_size), False, device=self.device)
            ball_same_as_beam_mask[:, :, 0:1] = torch.eye(self.vocab_size, dtype=torch.bool, device=self.device)
            ball_same_as_beam_mask = ball_same_as_beam_mask.permute(0, 1, 3, 2, 4)
            ball_scores[ball_same_as_beam_mask] = float("-inf")

            # Select a new ball for each possible beam item yv.
            ball_scores = ball_scores.reshape(batch_size, beam_size, self.vocab_size, -1) # [batch, beam, vocab, (ball+1)*vocab]
            _, chosen_ball_idxs = torch.topk(ball_scores, ball_size, dim=-1)              # [batch, beam, vocab, ball]
            # Add the beam item (which we previously set to -inf) back into its ball.
            beam_item_scores = ball_scores_old[ball_same_as_beam_mask]
            beam_item_scores = beam_item_scores.reshape(batch_size, beam_size, self.vocab_size, 1)          # [batch, beam, vocab, 1]
            ball_scores_old = ball_scores_old.reshape(batch_size, beam_size, self.vocab_size, -1)           # [batch, beam, vocab, (ball+1)*vocab]
            ball_item_scores = torch.gather(ball_scores_old, -1, chosen_ball_idxs)
            chosen_ball_scores = torch.cat((beam_item_scores, ball_item_scores), dim=-1)                    # [batch, beam, vocab, ball+1]
            beam_item_idxs = torch.arange(self.vocab_size).expand(batch_size, beam_size, -1).unsqueeze(-1)  # [batch, beam, vocab, 1]
            chosen_ball_idxs = torch.cat((beam_item_idxs, chosen_ball_idxs), dim=-1)                        # [batch, beam, vocab, ball+1]

            # Use the ball scores to compute the score of each beam item.
            beam_scores = torch.sum(chosen_ball_scores, dim=-1) # [batch, beam, vocab]
            # Use the beam scores to select the new beams.
            beam_scores = beam_scores.reshape(batch_size, -1)   # [batch, beam*vocab]
            chosen_beam_probs, chosen_beam_idxs = torch.topk(beam_scores, beam_size, dim=-1) # [batch, beam]

            # Pass this info to the distance metric + beam manager to tell it what to prune.
            beam_manager.select_idxs_cluster(chosen_ball_idxs, chosen_beam_idxs)
            distance.select_idxs(chosen_ball_idxs, chosen_beam_idxs)
            
            time_step += 1
        
        symbols, probs = beam_manager.get_final() # [batch, beam, ball+1, seq_len], [batch, beam, ball+1]
        return symbols[:,0,0,:].clone(), symbols, probs
