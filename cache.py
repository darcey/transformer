import torch

# All functions assume the data is in [batch*beam, ...] format.
# They can switch it into [batch, beam, ...] format as necessary.
class BeamCache:

    def __init__(self, batch_size, beam_size, num_layers, device):
        self.device = device
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_layers = num_layers
        self.src_id_to_layer_num = dict()

        self.src_mask = None
        self.src_k_num_to_temp_cache = dict()
        self.src_k_num_cached = 0
        self.src_k = None
        self.src_v_num_to_temp_cache = dict()
        self.src_v_num_cached = 0
        self.src_v = None

        self.finished_mask = torch.tensor([False]*batch_size*beam_size, device=device)
        self.idx_mapping = torch.arange(batch_size * beam_size, dtype=torch.int, device=device)

    # Store the source mask.
    # src_mask: [batch_size * beam_size, 1, src_seq]
    def cache_src_mask(self, src_mask):
        self.src_mask = src_mask
    # Retrieve the source mask.
    # src_mask: [(batch_size * beam_size) - |finished|, 1, src_seq]
    def get_src_mask(self):
        return self.src_mask

    # Store the k and v projections.
    # Group all the layer-specific caches into a single tensor for efficiency.
    # k, v: [batch_size * beam_size, seq, d_attention]
    def cache_src_k(self, layer_id, layer_num, k):
        self.src_id_to_layer_num[layer_id] = layer_num
        self.src_k_num_to_temp_cache[layer_num] = k
        self.src_k_num_cached += 1
        if self.src_k_num_cached == self.num_layers:
            self.src_k = torch.stack([self.src_k_num_to_temp_cache[i] for i in range(self.num_layers)])
            self.src_k_num_cached = 0
    def cache_src_v(self, layer_id, layer_num, v):
        self.src_id_to_layer_num[layer_id] = layer_num
        self.src_v_num_to_temp_cache[layer_num] = v
        self.src_v_num_cached += 1
        if self.src_v_num_cached == self.num_layers:
            self.src_v = torch.stack([self.src_v_num_to_temp_cache[i] for i in range(self.num_layers)])
            self.src_v_num_cached = 0

    # Retrieve the k and v projections, for either src or tgt.
    # k, v: [(batch_size * beam_size) - |finished|, seq, d_attention]
    def get_k(self, layer_id):
        if layer_id in self.src_id_to_layer_num:
            layer_num = self.src_id_to_layer_num[layer_id]
            k = self.src_k[layer_num,:,:,:]
            return k
        else:
            # target stuff goes here
            return None
        return k
    def get_v(self, layer_id):
        if layer_id in self.src_id_to_layer_num:
            layer_num = self.src_id_to_layer_num[layer_id]
            v = self.src_v[layer_num,:,:,:]
            return v
        else:
            # target stuff goes here
            return None

    # Reshape the cache to accommodate beam_size beams or samples per sentence.
    # Should be done after the src stuff has been cached, but before any decoding.
    def expand_to_beam_size(self, beam_size):
        self.beam_size = beam_size
        src_len = self.src_k.size(-2)
        if self.src_mask is not None:
            self.src_mask = self.src_mask.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(self.batch_size * beam_size, 1, src_len)
        self.src_k = self.src_k.unsqueeze(2).expand(-1, -1, beam_size, -1, -1).reshape(self.num_layers, self.batch_size * beam_size, src_len, -1)
        self.src_v = self.src_v.unsqueeze(2).expand(-1, -1, beam_size, -1, -1).reshape(self.num_layers, self.batch_size * beam_size, src_len, -1)
        self.finished_mask = self.finished_mask.unsqueeze(1).expand(-1, beam_size).reshape(self.batch_size * beam_size)
        self.update_idx_mapping()

    # Register individual finished sentences.
    # These can be permanently removed from the cache,
    # since once a sentence is finished we never need to do
    # computation on it again.
    # finished_mask: [batch_size * beam_size]
    def register_finished_sents(self, finished_mask):
        # If finished mask hasn't changed, can just return
        if torch.equal(finished_mask, self.finished_mask):
            return

        # Update the finished mask
        finished_mask_old = self.finished_mask
        finished_mask_curr = finished_mask[~finished_mask_old]
        self.finished_mask = finished_mask
        self.update_idx_mapping()

        # Update the caches in light of the new finished mask
        self.src_mask = self.src_mask[~finished_mask_curr]
        self.src_k = self.src_k.permute((1,0,2,3))[~finished_mask_curr].permute((1,0,2,3))
        self.src_v = self.src_v.permute((1,0,2,3))[~finished_mask_curr].permute((1,0,2,3))

    # Register finished beams.
    # This is the same as register_finished_sents, except that
    # it also truncates the full form of the finished_mask,
    # since now the beam manager will also be storing it
    # in a truncated form.
    # finished_mask: [batch_size]
    def register_finished_beams(self, finished_mask):
        finished_mask_expanded = finished_mask.unsqueeze(1).expand(-1, self.beam_size).reshape(-1)
        self.register_finished_sents(torch.logical_or(self.finished_mask, finished_mask_expanded))
        self.batch_size = (~finished_mask).sum()
        self.finished_mask = self.finished_mask[~finished_mask_expanded]
        self.update_idx_mapping()

    # This is needed for select_idxs.
    def update_idx_mapping(self):
        idx_mapping = torch.empty(self.batch_size * self.beam_size, dtype=torch.int, device=self.device)
        idx_mapping[self.finished_mask] = -1
        idx_mapping[~self.finished_mask] = torch.arange((~self.finished_mask).sum(), dtype=torch.int, device=self.device)
        self.idx_mapping = idx_mapping

    # Counterpart of analogous function in beam_manager.
    # chosen_idxs: [batch_size, beam_size]
    def select_idxs(self, chosen_idxs):
        # Convert chosen_idxs into a batch*beam format
        # Before: chosen_idxs[batch,beam] = beam'
        # After:  chosen_idxs[batch*beam] = batch*beam_size + beam'
        base_beam_size = torch.arange(self.batch_size, device=self.device).unsqueeze(1)*self.beam_size
        chosen_idxs = (chosen_idxs + base_beam_size).reshape(-1)

        # Update mask, create truncated version of chosen_idxs
        self.finished_mask = self.finished_mask[chosen_idxs]
        chosen_idxs = chosen_idxs[~self.finished_mask]
        mapped_idxs = self.idx_mapping[chosen_idxs]
        self.update_idx_mapping()

        # Now update everything else.
        self.src_mask = torch.index_select(self.src_mask, 0, mapped_idxs)
        self.src_k    = torch.index_select(self.src_k, 1, mapped_idxs)
        self.src_v    = torch.index_select(self.src_v, 1, mapped_idxs)
