# TODO(darcey): maybe write a second beam manager which allows different beam sizes for each sentence in the batch

import math
import torch
from torch.nn.utils.rnn import pad_sequence

# Maintains a set of beams for beam-based decoding algorithms, such as
# sampling, beam search, or cluster search. It is batched, and assumes
# that all sentences in the batch have the same beam size and ball size.
class BeamManager:

    # Initializes the beam manager with the default initial beams
    # (all beam items start with BOS and have probability 1). If you want
    # other initial beams/probs, you should call manually_initialize()
    # after creating the beam manager.
    # max_lengths: [batch]
    def __init__(self, batch_size, beam_size, ball_size, vocab_size, max_lengths, max_possible_length, pad, bos, eos, autoregressive_fn, cache, device):
        # Model information
        self.auto_fn = autoregressive_fn
        self.cache   = cache
        self.device  = device

        # Special token information
        self.pad = pad
        self.bos = bos
        self.eos = eos

        # Shape information
        self.batch_size          = batch_size
        self.curr_size           = batch_size
        self.beam_size           = beam_size
        self.ball_size           = ball_size
        self.vocab_size          = vocab_size
        self.seq_len             = 1
        self.max_lengths         = max_lengths
        self.max_possible_length = max_possible_length

        # Maintain the working copies of the beams and probabilities
        # Initialize them with the default initial beams/probs
        # Indices [batch, beam, 0] represent the beam and
        # indices [batch, beam, 1:ball_size] represent the ball.
        self.symbols = torch.full((batch_size, beam_size, ball_size+1, 1), bos, device=self.device) # [batch, beam, ball+1, seq_len]
        self.probs   = torch.zeros(batch_size, beam_size, ball_size+1, device=self.device)          # [batch, beam, ball+1]

        # Also maintain a list of the items previously known to be finished.
        self.prev_finished = torch.full((batch_size, beam_size, ball_size+1), False, device=self.device)

        # Update the shape of the cache
        self.cache.expand_to_beam_size(beam_size*(ball_size+1))

        # Maintain the final copies of the beams and probabilities,
        # to be returned when decoding is finished
        self.ret_symbols = [None] * batch_size
        self.ret_probs   = [None] * batch_size
        self.orig_idxs   = torch.arange(batch_size, device=self.device)

    # Manually set the initial beams and probs.
    # Assumes everything is the correct size and that seq_len is 1.
    # symbols: [batch, beam, ball+1, 1]
    # probs:   [batch, beam, ball+1]
    def manually_initialize(self, symbols, probs):
        self.symbols = symbols
        self.probs = probs
        self.prev_finished = torch.logical_or(symbols[:, :, :, -1] == self.pad,\
                                              symbols[:, :, :, -1] == self.eos)

    # For each sentence in the batch, checks whether everything in the 
    # beam has reached EOS. If so there is no need to do further
    # computation on that sentence; can just remove it and save it to 
    # be returned later.
    # In the case of cluster beam search (where ball_size > 0), this
    # just checks whether the main beam items are finished, not whether
    # the ball items are finished also.
    def prune_finished(self):
        reached_max_length = (self.seq_len >= self.max_lengths) + (self.seq_len == self.max_possible_length)          # [batch]
        eos_or_pad         = torch.logical_or(torch.logical_or(self.symbols[:, :, 0, -1] == self.eos,
                                                               self.symbols[:, :, 0, -1] == self.pad),
                                              self.prev_finished[:, :, 0])                                            # [batch, beam]
        reached_eos        = torch.sum(eos_or_pad.type(torch.int), dim=-1) == self.beam_size                          # [batch]
        finished_sents     = reached_max_length + reached_eos                                                         # [batch]

        if finished_sents.any():
            for j in range(finished_sents.size(0)):
                if finished_sents[j]:
                    self.ret_symbols[self.orig_idxs[j]] = self.symbols[j].clone()
                    self.ret_probs[self.orig_idxs[j]]   = self.probs[j].clone()

            self.symbols       = self.symbols[~finished_sents]
            self.probs         = self.probs[~finished_sents]
            self.max_lengths   = self.max_lengths[~finished_sents]
            self.orig_idxs     = self.orig_idxs[~finished_sents]
            self.prev_finished = self.prev_finished[~finished_sents]
            self.cache.register_finished_beams(finished_sents)
            self.curr_size     = self.symbols.size(0)

    # Check whether decoding is finished.
    def all_done(self):
        return self.curr_size == 0

    # Returns a tensor containing all of the generated sentences.
    # Should only be called once all_done() returns true.
    def get_final(self):
        # Since all sentences in the batch have the same beam and ball size,
        # can just stack them in order to return. But the generations
        # may have different target lengths so they should be padded
        # before stacking.
        for i in range(self.batch_size):
            self.ret_symbols[i] = self.ret_symbols[i].permute((2,0,1))
        ret_symbols = pad_sequence(self.ret_symbols, batch_first=True, padding_value=self.pad).permute((0,2,3,1))
        ret_probs   = torch.stack(self.ret_probs)
        return ret_symbols, ret_probs

    # Compute the lengths of all the one-token extensions of the beam items.
    # Should be called after compute_next_token_probs
    # lengths: [batch, beam, ball+1, vocab]
    def get_all_choices_lengths(self):
        lengths = self.seq_len - (self.symbols == self.eos).sum(-1) - (self.symbols == self.pad).sum(-1) # [batch, beam, ball+1]
        finished = lengths < self.seq_len                                                                # [batch, beam, ball+1]
        end_toks = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.symbols.device)            # [vocab]
        end_toks[self.pad] = True
        end_toks[self.eos] = True
        still_not_finished = ~finished.unsqueeze(-1) * ~end_toks                                         # [batch, beam, ball+1, vocab]
        all_choices_lengths = lengths.unsqueeze(-1).expand(-1, -1, -1, self.vocab_size).clone()          # [batch, beam, ball+1, vocab]
        all_choices_lengths[still_not_finished] = all_choices_lengths[still_not_finished] + 1            # [batch, beam, ball+1, vocab]
        return all_choices_lengths

    # Make a call to the underlying model to get the next token
    # probabilities for each item in the beam and ball. Use these
    # to compute what the cumulative probability would be for every
    # expansion of every beam and ball item. Return these so that
    # the decoding algorithm can algorithm can make its choices.
    # next_token_logits: [batch, beam, ball+1, vocab]
    # next_token_probs:  [batch, beam, ball+1, vocab]
    # all_choices_probs: [batch, beam, ball+1, vocab]
    def compute_next_token_probs(self):
        # Separate out the sentences which already hit EOS;
        # no need to pass these through the transformer.
        last_symbols = self.symbols[:, :, :, -1]                                 # [batch, beam, ball+1]
        curr_finished = (last_symbols == self.eos) + (last_symbols == self.pad)  # [batch, beam, ball+1]
        self.prev_finished = torch.logical_or(curr_finished, self.prev_finished) # [batch, beam, ball+1]
        finished_mask = self.prev_finished.reshape(-1)                           # [batch*beam*(ball+1)]
        if finished_mask.any():
            next_token_logits = torch.full((self.curr_size*self.beam_size*(self.ball_size+1), self.vocab_size),
                                           float("-inf"), device=self.device)               # [batch*beam*(ball+1), vocab]
            next_token_probs = torch.full((self.curr_size*self.beam_size*(self.ball_size+1), self.vocab_size),
                                          float("-inf"), device=self.device)                # [batch*beam*(ball+1), vocab]
            active_symbols = self.symbols.reshape(-1, self.seq_len)[~finished_mask]         # [batch*beam*(ball+1) - |finished|, seq_len]
            self.cache.register_finished_sents(finished_mask)
            active_logits, active_probs = self.auto_fn(active_symbols[:,-1:], self.seq_len-1, self.cache) # [batch*beam*(ball+1) - |finished|, seq, vocab]
            next_token_logits[~finished_mask] = active_logits[:,-1,:]
            next_token_logits[finished_mask,self.pad] = 0.0
            next_token_probs[~finished_mask] = active_probs[:,-1,:]
            next_token_probs[finished_mask,self.pad] = 0.0
        else:
            next_token_logits, next_token_probs = self.auto_fn(self.symbols.reshape(-1, self.seq_len)[:,-1:], self.seq_len-1, self.cache) # [batch*beam*(ball+1), seq_len, vocab]
            next_token_logits = next_token_logits[:,-1,:] # [batch*beam*(ball+1), vocab]
            next_token_probs = next_token_probs[:,-1,:]   # [batch*beam*(ball+1), vocab]

        next_token_logits = next_token_logits.reshape(self.curr_size, self.beam_size, self.ball_size+1, self.vocab_size) # [batch, beam, ball+1, vocab]
        next_token_probs = next_token_probs.reshape(self.curr_size, self.beam_size, self.ball_size+1, self.vocab_size)   # [batch, beam, ball+1, vocab]
        self.all_choices_probs = self.probs.unsqueeze(-1) + next_token_probs                                             # [batch, beam, ball+1, vocab]
        return next_token_logits, next_token_probs, self.all_choices_probs.clone()

    # In sampling, for each original beam or ball item, we choose one successor.
    # The decisions for each beam/ball item are independent of each other,
    # making this very easy to implement.
    # chosen_idxs: [batch, beam, ball+1]
    def select_idxs_sampling(self, chosen_idxs):
        chosen_idxs = chosen_idxs.unsqueeze(-1)
        self.probs = torch.gather(self.all_choices_probs, -1, chosen_idxs).squeeze(-1)
        self.symbols = torch.cat((self.symbols, chosen_idxs), -1)
        self.seq_len += 1

    # In beam search, for each set of beam*vocab items, we will choose beam successors.
    # To do this, we combine the beam and vocab dimensions into a single dimension
    # and select from that. We assume that each element in chosen_idxs takes the form:
    # chosen_idxs[batch, beam] = orig_beam*vocab_size + orig_vocab_item.
    # It is essentially a base-vocab number telling where to find the chosen
    # beams in self.all_choices_probs. Since in beam search, the ball size is 0,
    # we can essentially ignore the ball dimension.
    # chosen_idxs: [batch, beam]
    def select_idxs_beam(self, chosen_idxs):
        # Make sure the ball size is 0.
        assert(self.ball_size == 0)

        # Get the correct new probabilities.
        all_choices_probs = self.all_choices_probs.reshape(self.curr_size, -1)      # [batch, beam*vocab]
        self.probs = torch.gather(all_choices_probs, -1, chosen_idxs).unsqueeze(-1) # [batch, beam, ball]

        # Get the correct new beams.
        # Need to first select the correct original beams.
        beam_parent_idxs = torch.div(chosen_idxs, self.vocab_size, rounding_mode='floor') # [batch, beam]
        beam_parent_idxs_expanded = beam_parent_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,1,self.seq_len) # [batch, beam, ball+1, seq_len]
        symbols = torch.gather(self.symbols, 1, beam_parent_idxs_expanded) # [batch, beam, ball+1, seq_len]
        # Then need to extend the chosen original beams by the correct token.
        new_symbol_idxs = chosen_idxs - beam_parent_idxs * self.vocab_size # [batch, beam]
        self.symbols = torch.cat((symbols, new_symbol_idxs.unsqueeze(-1).unsqueeze(-1)), dim=-1) # [batch, beam, ball+1, seq_len+1]
        # Update prev_finished and the cache.
        self.prev_finished = torch.gather(self.prev_finished, 1, beam_parent_idxs.unsqueeze(-1))
        self.cache.select_idxs(beam_parent_idxs)
        
        self.seq_len += 1

    # In cluster beam search, the set of choices has shape [batch, beam, vocab, ball+1, vocab]
    # and the choice of idxs proceeds in two stages: first, selecting the ball successors,
    # and then selecting the beam successors. For both of these, the principle is the same as
    # for beam search: combine the beam*vocab dimensions or the ball*vocab dimensions, then
    # use base-vocab numbers to select the appropriate idxs.
    # chosen_idxs_ball: [batch, beam, vocab, ball+1]
    # chosen_idxs_beam: [batch, beam]
    def select_idxs_cluster(self, chosen_idxs_ball, chosen_idxs_beam):
        # Expand the all_choices_probs to the correct size.
        all_choices_probs = self.all_choices_probs.unsqueeze(2).expand(-1,-1,self.vocab_size,-1,-1) # [batch, beam, vocab, ball+1, vocab]
        symbols = self.symbols.unsqueeze(2).expand(-1,-1,self.vocab_size,-1,-1)                     # [batch, beam, vocab, ball+1, seq_len]

        # Do the ball selection for the probs.
        all_choices_probs = all_choices_probs.reshape(self.curr_size, self.beam_size, self.vocab_size, -1) # [batch, beam, vocab, (ball+1)*vocab]
        beam_choices_probs = torch.gather(all_choices_probs, -1, chosen_idxs_ball)                         # [batch, beam, vocab, ball+1]

        # Do the ball selection for the symbols.
        # Need to first select the correct original ball items.
        ball_parent_idxs = torch.div(chosen_idxs_ball, self.vocab_size, rounding_mode='floor') # [batch, beam, vocab, ball+1]
        ball_parent_idxs_expanded = ball_parent_idxs.unsqueeze(-1).expand(-1,-1,-1,-1,self.seq_len) # [batch, beam, vocab, ball+1, seq_len]
        symbols = torch.gather(symbols, 3, ball_parent_idxs_expanded) # [batch, beam, vocab, ball+1, seq_len]
        # Then need to extend the chosen original ball items by the correct token.
        new_symbol_idxs = chosen_idxs_ball - ball_parent_idxs * self.vocab_size # [batch, beam, vocab, ball+1]
        symbols = torch.cat((symbols, new_symbol_idxs.unsqueeze(-1)), dim=-1) # [batch, beam, vocab, ball+1, seq_len+1]

        # Do the ball selection for prev_finished.
        prev_finished = self.prev_finished.unsqueeze(2).expand(-1, -1, self.vocab_size, -1) # [batch, beam, vocab, ball+1]
        prev_finished = torch.gather(prev_finished, -1, ball_parent_idxs) # [batch, beam, vocab, ball+1]

        # Do the beam selection for the probs.
        beam_choices_probs = beam_choices_probs.reshape(self.curr_size,-1,self.ball_size+1) # [batch, beam*vocab, ball+1]
        chosen_idxs_beam_expanded = chosen_idxs_beam.unsqueeze(-1).expand(-1,-1,self.ball_size+1) # [batch, beam, ball+1]
        self.probs = torch.gather(beam_choices_probs, 1, chosen_idxs_beam_expanded) # [batch, beam, ball+1]

        # Do the beam selection for the symbols.
        # Just need to select the correct beam+vocab items.
        # Don't need to extend anything by 1 token because we already did that.
        symbols = symbols.reshape(self.curr_size, -1, self.ball_size+1, self.seq_len+1) # [batch, beam*vocab, ball+1, seq_len+1]
        chosen_idxs_beam_expanded_again = chosen_idxs_beam_expanded.unsqueeze(-1).expand(-1,-1,-1,self.seq_len+1) # [batch, beam, ball+1, seq_len+1]
        self.symbols = torch.gather(symbols, 1, chosen_idxs_beam_expanded_again) # [batch, beam, ball+1, seq_len+1]

        # Do the beam selection for prev_finished.
        prev_finished = prev_finished.reshape(self.curr_size, -1, self.ball_size+1) # [batch, beam*vocab, ball+1]
        self.prev_finished = torch.gather(prev_finished, 1, chosen_idxs_beam_expanded) # [batch, beam, ball+1]

        # Update the cache. Cache expects something in [batch, beam*(ball+1)] format.
        ball_parent_idxs = ball_parent_idxs.reshape(self.curr_size, -1, self.ball_size+1) # [batch, beam*vocab, ball+1]
        orig_beam_ball_idxs = torch.gather(ball_parent_idxs, 1, chosen_idxs_beam_expanded) # [batch, beam, ball+1]
        orig_beam_ball_idxs = orig_beam_ball_idxs.reshape(self.curr_size, -1) # [batch, beam*(ball+1)]
        self.cache.select_idxs(orig_beam_ball_idxs)

        self.seq_len += 1

# This is just a wrapper around BeamManager to allow easier use
# by sampling and beam search, which don't have a ball dimension.
class BeamManagerNoBall(BeamManager):

    # max_lengths: [batch]
    def __init__(self, batch_size, beam_size, vocab_size, max_lengths, max_possible_length, pad, bos, eos, autoregressive_fn, cache, device):
        super().__init__(batch_size, beam_size, 0, vocab_size, max_lengths, max_possible_length, pad, bos, eos, autoregressive_fn, cache, device)

    # symbols: [batch, beam, 1]
    # probs:   [batch, beam]
    def manually_initialize(self, symbols, probs):
        super().manually_initialize(symbols.unsqueeze(2), probs.unsqueeze(2))

    # These accessors are just for testing;
    # probably shouldn't be used elsewhere.
    def get_symbols(self):
        return self.symbols.clone().squeeze(2)
    def get_probs(self):
        return self.probs.clone().squeeze(2)
    def set_symbols(self, symbols):
        self.symbols = symbols.unsqueeze(2)
    def set_probs(self, probs):
        self.probs = probs.unsqueeze(2)

    def get_final(self):
        ret_symbols, ret_probs = super().get_final()
        return ret_symbols.squeeze(2), ret_probs.squeeze(2)

    def get_all_choices_lengths(self):
        lengths = super().get_all_choices_lengths()
        return lengths.squeeze(2)

    def compute_next_token_probs(self):
        next_token_logits, next_token_probs, all_choices_probs = super().compute_next_token_probs()
        return next_token_logits.squeeze(2), next_token_probs.squeeze(2), all_choices_probs.squeeze(2)

    # chosen_idxs: [batch, beam]
    def select_idxs_sampling(self, chosen_idxs):
        super().select_idxs_sampling(chosen_idxs.unsqueeze(-1))
