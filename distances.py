from abc import ABC, abstractmethod
import torch



class Distance(ABC):

    @abstractmethod
    def __init__(self, batch_size, beam_size, ball_size, vocab_size):
        pass

    @abstractmethod
    def get_distance_estimates(self, sents): 
        pass

    @abstractmethod
    def select_idxs(self, ball_idxs, beam_idxs):
        pass



class Identity(Distance):

    def __init__(self, batch_size, beam_size, ball_size, vocab_size, beam_manager):
        self.beam_size = beam_size
        self.ball_size = ball_size
        self.vocab_size = vocab_size
        self.beam_manager = beam_manager

    # sents: [batch, beam, ball+1, seq]
    def get_distance_estimates(self):
        sents = self.beam_manager.symbols.clone()
    
        curr_size = sents.size(0)
        sents = sents.unsqueeze(3).expand(-1, -1, -1, self.vocab_size, -1)  # [batch, beam, ball+1, vocab, seq]
        next_tok = torch.arange(self.vocab_size).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(4).expand(curr_size, self.beam_size, self.ball_size+1, -1, 1) # [batch, beam, ball+1, vocab, 1]
        sents = torch.cat((sents, next_tok), -1) # [batch, beam, ball+1, vocab, seq+1]

        dists = torch.full((curr_size, self.beam_size, self.vocab_size, self.ball_size+1, self.vocab_size), float("inf"))
#        for i in range(curr_size):
#            for j in range(self.beam_size):
#                for k in range(self.vocab_size):
#                    for l in range(self.ball_size+1):
#                        for m in range(self.vocab_size):
#                            if (sents[i,j,0,k,:] == sents[i,j,l,m,:]).all():
#                                dists[i,j,k,l,m] = 0.0
        sents_1 = sents[:,:,0,:,:].clone() # [batch, beam, vocab, seq+1]
        sents_1 = sents_1.unsqueeze(3).unsqueeze(3).expand(-1, -1, -1, self.ball_size+1, self.vocab_size, -1) # [batch, beam, vocab, ball+1, vocab, seq+1]
        sents_2 = sents.clone().unsqueeze(2).expand(-1, -1, self.vocab_size, -1, -1, -1) # [batch, beam, vocab, ball+1, vocab, seq+1]
        eq = (sents_1 != sents_2).sum(dim=-1) == 0
        dists[eq] = 0.0
        return dists

    def select_idxs(self, ball_idxs, beam_idxs):
        return




# TODO does this code need to worry at all about sentences that end in PAD? probably because it's just maintaining the outer bits of the lev table? figure this out

#string1: batch x beam x len
#string2: batch x beam x (ball + 1) x len
#lev:     batch x beam x (ball + 1) x len x len

#extend string1: batch x beam x vocab x len
#extend string2: batch x beam x (ball + 1) x vocab x len
#extend lev:     batch x beam x vocab x (ball + 1) x vocab x len+1 x len+1

#weights: vocab x vocab

class Levenshtein:

    # TODO also need PAD index (and BOS, EOS? or just PAD?)
    def __init__(self, batch_size, beam_size, ball_size, vocab_size):
        self.batch_size = batch_size
        self.beam_size  = beam_size
        self.ball_size  = ball_size
        self.vocab_size = vocab_size

        self.weights = initialize_unweighted(vocab_size) # [vocab, vocab]

        # at each time step, we have beam item y, ball items z in ball.
        # need to levenshtein y against itself, + against all ball items z.
        # string1 stores all the ys
        # string2 stores all the ys + ball items z
        self.len     = 0
        self.string1 = torch.empty(batch_size, beam_size, self.len) # [batch, beam, len]
        self.string2 = torch.empty(batch_size, beam_size, ball_size+1, self.len) # [batch, beam, ball+1, len]
        
        # TODO store this more efficiently -- just need to maintain top row and side row at each point in time
        self.lev = torch.empty(batch_size, beam_size, ball_size+1, self.len+1, self.len+1)
        #self.lev_string1 = torch.empty(batch_size, beam_size, 
        #self.lev_string2 =
        #self.lev_corner  = 
        
        # Initialize Levenshtein table for all pairs
        # TODO change to more efficient representation
        self.lev[:,:,:,0,0] = 0

    def extend_strings_by_one_token(self):
        self.string1_ext = self.string1.unsqueeze(2).expand(-1,-1,self.vocab_size,-1)
        self.string1_one_tok = torch.arange(self.vocab_size).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(self.batch_size, self.beam_size, -1, -1)
        self.string1_ext = torch.cat(self.string1_ext, self.string1_one_tok)
        
        # TODO do this for string2 also

    def extend_lev_tables_by_one_token(self):
        lev = self.lev.unsqueeze(2).unsqueeze(4).expand(-1,-1,self.vocab_size,-1,self.vocab_size,-1,-1)
        # just do the Levenshtein algorithm one position at a time
        
        # need to be careful with EOS and PAD
        # also need to be careful of sentences that contain EOS or PAD followed by some other token
        # these sentences are invalid so their distance to everything should always be infinity

    def take_min_of_stuff(self):
        return
        # this will just do a min over the outer rows of the Levenshtein table and return that

    def get_distance_estimates(self):
        self.extend_strings_by_one_token()
        self.extend_lev_tables_by_one_token()
        return self.take_min_of_stuff()

    def select_top_k(self):
        return
        # this is presumably going to be the same kind of top-k selection logic as anywhere else in the code


def initialize_unweighted(vocab_size):
    # TODO add stuff to accommodate infinite distances for PAD
    return torch.full((vocab_size, vocab_size), 1.0)
    

# THIS IS OLD DRAFT CODE, SAVING IT JUST IN CASE
            # for each batch item, for each beam item y, for each extension yv of y,
            # for each extension zv' of every ball item z (including y),
            # compute the distance between yv and zv' and store it.
            # sentences which end in EOS require some special handling:
            # extensions other than PAD should just get a distance of infinity.
            # TODO: confirm that this will handle EOS sentences correctly
            #distances = torch.empty((batch_size, beam_size, vocab, (ball_size+1), vocab), device=self.device)
            #for batch_item in range(batch_size):
            #    for beam_item in range(beam_size):
            #        # TODO: check if y ends in EOS; if it does, just need to consider y+PAD and not any other extensions..... maybe do this by ignoring everything with prob -inf??
            #        for v1 in range(self.vocab_size):
            #            # TODO: this needs the symbols. in sampling and vanilla beam search, the generator doesn't need the symbols, 
            #            # so the beam manager doesn't return them -- here it needs to
            #            # cand = symbols[batch_item,beam_item,...]
            #            for ball_item in range(ball_size+1):
            #                for v2 in range(self.vocab_size):
            #                    if ball_item == beam_item and v1 == v2:
            #                        distances[batch_item, beam_item, v1, beam_item, v1] = 0.0
            #                    else:
            #                        # hypo = symbols[...]
            #                        distances[batch_item, beam_item, v1, ball_item, v2] = distance_func(cand, hypo) # TODO
