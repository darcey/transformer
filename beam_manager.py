# TODO(darcey): update the logic to handle a sub-beam dimension for cluster search
# TODO(darcey): maybe write a second beam manager which allows different beam sizes for each sentence in the batch

import math
import torch
from torch.nn.utils.rnn import pad_sequence

# Maintains a set of beams for beam-based decoding algorithms, such as
# sampling, beam search, or cluster search. It is batched, and assumes
# that all sentences in the batch have the same beam size.
class BeamManager:

    # Initializes the beam manager with the default initial beams
    # (all beam items start with BOS and have probability 1). If you want
    # other initial beams/probs, you should call manually_initialize()
    # after creating the beam manager.
    def __init__(self, batch_size, beam_size, vocab_size, max_lengths, max_possible_length, pad, bos, eos, autoregressive_fn, cache, device):
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
        self.vocab_size          = vocab_size
        self.seq_len             = 1
        self.max_lengths         = max_lengths
        self.max_possible_length = max_possible_length

        # Maintain the working copies of the beams and probabilities
        # Initialize them with the default initial beams/probs
        self.symbols = torch.full((batch_size, beam_size, 1), bos, device=self.device) # [batch, beam, seq_len]
        self.probs   = torch.zeros(batch_size, beam_size, device=self.device)          # [batch, beam]

        # Update the shape of the cache
        self.cache.expand_to_beam_size(beam_size)

        # Maintain the final copies of the beams and probabilities,
        # to be returned when decoding is finished
        self.ret_symbols = [None] * batch_size
        self.ret_probs   = [None] * batch_size
        self.orig_idxs   = torch.arange(batch_size, device=self.device)

    # Manually set the initial beams and probs.
    # Assumes everything is the correct size and that seq_len is 1.
    # symbols: [batch, beam, 1]
    # probs:   [batch, beam]
    def manually_initialize(self, symbols, probs):
        self.symbols = symbols
        self.probs = probs

    # For each sentence in the batch, checks whether everything in the 
    # beam has reached EOS. If so there is no need to do further
    # computation on that sentence; can just remove it and save it to 
    # be returned later.
    def prune_finished(self):
        reached_max_length = (self.seq_len >= self.max_lengths) + (self.seq_len == self.max_possible_length)          # [batch]
        eos_or_pad         = torch.logical_or(torch.logical_or(self.symbols[:, :, -1] == self.eos,
                                                               self.symbols[:, :, -1] == self.pad),
                                              self.symbols[:, :, 0] == self.pad)                                      # [batch, beam]
        reached_eos        = torch.sum(eos_or_pad.type(torch.int), dim=-1) == self.beam_size                          # [batch]
        finished_sents     = reached_max_length + reached_eos                                                         # [batch]

        if finished_sents.any():
            for j in range(finished_sents.size(0)):
                if finished_sents[j]:
                    self.ret_symbols[self.orig_idxs[j]] = self.symbols[j].clone().permute((1,0))
                    self.ret_probs[self.orig_idxs[j]]   = self.probs[j].clone()

            self.symbols     = self.symbols[~finished_sents]
            self.probs       = self.probs[~finished_sents]
            self.max_lengths = self.max_lengths[~finished_sents]
            self.orig_idxs   = self.orig_idxs[~finished_sents]
            self.cache.register_finished_beams(finished_sents)
            self.curr_size   = self.symbols.size(0)

    # Check whether decoding is finished.
    def all_done(self):
        return self.curr_size == 0

    # Returns a tensor containing all of the generated sentences.
    # Should only be called once all_done() returns true.
    def get_final(self):
        # Since all sentences in the batch have the same beam size,
        # can just stack them in order to return. But the generations
        # may have different target lengths so they should be padded
        # before stacking.
        ret_symbols = pad_sequence(self.ret_symbols, batch_first=True, padding_value=self.pad).permute((0,2,1))
        ret_probs   = torch.stack(self.ret_probs)
        return ret_symbols, ret_probs

    # Make a call to the underlying model to get the next token
    # probabilities for each item in the beam. Use these to compute
    # what the cumulative probability would be for every expansion
    # of every beam. Return these so that the decoding algorithm can
    # make its choices.
    # next_token_probs:  [batch, beam, vocab]
    # all_choices_probs: [batch, beam, vocab]
    def compute_next_token_probs(self):
        # Separate out the sentences which already hit EOS;
        # no need to pass these through the transformer.
        last_symbols = self.symbols[:, :, -1].reshape(-1)                       # [batch*beam]
        finished_mask = (last_symbols == self.eos) + (last_symbols == self.pad) # [batch*beam]
        if finished_mask.any():
            next_token_probs = torch.full((self.curr_size*self.beam_size, self.vocab_size),
                                          float("-inf"), device=self.device)                # [batch*beam, vocab]
            active_symbols = self.symbols.reshape(-1, self.seq_len)[~finished_mask]         # [batch*beam, seq_len]
            self.cache.register_finished_sents(finished_mask)
            next_token_probs[~finished_mask] = self.auto_fn(active_symbols[:,-1:], self.seq_len-1, self.cache)[:,-1,:]
            next_token_probs[finished_mask,self.pad] = 0.0
        else:
            next_token_probs = self.auto_fn(self.symbols.reshape(-1, self.seq_len)[:,-1:], self.seq_len-1, self.cache)[:,-1,:] # [batch*beam, vocab] # [batch*beam, vocab]

        self.next_token_probs = next_token_probs.reshape(self.curr_size, self.beam_size, self.vocab_size) # [batch, beam, vocab]
        self.all_choices_probs = self.probs.unsqueeze(-1) + self.next_token_probs                         # [batch, beam, vocab]
        return self.next_token_probs.clone(), self.all_choices_probs.clone()

    # Assumes that for each original beam item, exactly one successor is
    # chosen (meaning there is no intermixing of the different items in
    # the beam), hence why this is the "independent" index selection.
    # chosen_idxs: [batch, beam]
    def select_idxs_independent(self, chosen_idxs):
        chosen_idxs = chosen_idxs.unsqueeze(-1)
        self.probs = torch.gather(self.all_choices_probs, -1, chosen_idxs).squeeze(-1)
        self.symbols = torch.cat((self.symbols, chosen_idxs), -1)
        self.seq_len += 1

    # Note that each element in chosen_idxs takes the following format:
    # chosen_idxs[batch, beam] = orig_beam*vocab_size + orig_vocab_item.
    # It is essentially a base-V number telling where to find chosen beam
    # in self.all_choices_probs. This allows intermixing of the beam items,
    # hence why this is the "dependent" index selection.
    # chosen_idxs: [batch, beam]
    def select_idxs_dependent(self, chosen_idxs):
        # Get the correct new probabilities.        
        all_choices_probs = self.all_choices_probs.reshape(self.curr_size, -1) # [batch, beam*V]
        self.probs = torch.gather(all_choices_probs, -1, chosen_idxs) # [batch, beam]
        
        # Get the correct new beams.
        # Need to first select the correct original beams.
        beam_parent_idxs = torch.div(chosen_idxs, self.vocab_size, rounding_mode='floor') # [batch, beam]
        beam_parent_idxs_expanded = beam_parent_idxs.unsqueeze(-1).expand(-1,-1,self.seq_len) # [batch, beam, seq_len]
        symbols = torch.gather(self.symbols, 1, beam_parent_idxs_expanded) # [batch, beam, seq_len]
        # Then need to extend the chosen original beams by the correct token.
        new_symbol_idxs = chosen_idxs - beam_parent_idxs * self.vocab_size # [batch, beam]
        self.symbols = torch.cat((symbols, new_symbol_idxs.unsqueeze(-1)), dim=-1)
        # Update the cache.
        self.cache.select_idxs(beam_parent_idxs)
        
        self.seq_len += 1
