# TODO(darcey): write tests for the BeamManager on GPU

import time
import copy
import random
import torch
import torch.testing
import unittest
from tempfile import TemporaryDirectory
from configuration import read_config, DecodingMethod, LengthNormalization
from trainer import *
from vocabulary import *
from generator import *
from cache import *
from transformer import *



class TestTrainerSameOnGPU(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data([],[])
        self.assertEqual(self.vocab.tok_to_idx(SpecialTokens.PAD), 0)
        self.config = read_config("configuration.toml")
        self.checkpt_dir = TemporaryDirectory()

    def tearDown(self):
        self.checkpt_dir.cleanup()

    def testWordDropout(self):
        if not torch.cuda.is_available():
            return

        trainer_cpu = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, "cpu")
        trainer_gpu = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, "cuda:0")
        in_cpu = torch.rand(20,10)
        in_gpu = in_cpu.to("cuda:0")
        out_cpu = trainer_cpu.word_dropout(in_cpu, 0.0)
        out_gpu = trainer_gpu.word_dropout(in_gpu, 0.0)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testLoss(self):
        if not torch.cuda.is_available():
            return

        self.config.train.label_smoothing = 0.5
        trainer_cpu = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, "cpu")
        trainer_gpu = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, "cuda:0")

        predicted_cpu = torch.rand(2,3,4)
        predicted_gpu = predicted_cpu.cuda()
        gold_cpu = torch.randint(high=4, size=(2,3))
        gold_gpu = gold_cpu.cuda()
        out_cpu = trainer_cpu.loss(predicted_cpu, gold_cpu)
        out_gpu = trainer_gpu.loss(predicted_gpu, gold_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testCrossEntropy(self):
        if not torch.cuda.is_available():
            return

        self.config.train.label_smoothing = 0.5
        trainer_cpu = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, "cpu")
        trainer_gpu = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, "cuda:0")

        predicted_cpu = torch.rand(2,3,4)
        predicted_gpu = predicted_cpu.cuda()
        gold_cpu = torch.randint(high=4, size=(2,3))
        gold_gpu = gold_cpu.cuda()
        ce_cpu, nt_cpu = trainer_cpu.cross_ent(predicted_cpu, gold_cpu)
        ce_gpu, nt_gpu = trainer_gpu.cross_ent(predicted_gpu, gold_gpu)
        torch.testing.assert_close(ce_cpu, ce_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(nt_cpu, nt_gpu.to("cpu"), atol=0.00001, rtol=0)



class MockModelDoesNothing:
    def get_autoregressive_one_step_fn(self, src, cache):
        return 5, 6

class MockCacheDoesNothing:
    def expand_to_beam_size(self, beam_size):
        return
    def register_finished_sents(self, mask):
        return
    def register_finished_beams(self, mask):
        return
    def select_idxs(self, chosen_idxs):
        return

class TestGeneratorWorksOnGPU(unittest.TestCase):

    def setUp(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2

        config = read_config("configuration.toml")
        model = MockModelDoesNothing()

        self.gen_cpu = Generator(model, config, "cpu", 5, self.pad_idx, self.bos_idx, self.eos_idx)
        self.gen_gpu = Generator(model, config, "cuda:0", 5, self.pad_idx, self.bos_idx, self.eos_idx)
        self.cache = MockCacheDoesNothing()
        self.config = config.gen

    # This is the same test function as in test_generator;
    # just need to make sure it also works on GPU.
    def testSampleSimpleDistribution(self):
        if not torch.cuda.is_available():
            return

        def mock_autoregressive_fn(symbols, timestep, cache):
            lh       = math.log(0.5)
            ni       = float("-inf")
            a_dist   = torch.tensor([[[ni, ni, lh, lh, ni]]])
            b_dist   = torch.tensor([[[ni, ni, lh, ni, lh]]])
            ab_dist  = torch.tensor([[[ni, ni, ni, lh, lh]]])
            all_dist = torch.empty((symbols.size(0), 5))
            all_dist[(symbols[:,-1] == 1)] = ab_dist
            all_dist[(symbols[:,-1] == 3)] = a_dist
            all_dist[(symbols[:,-1] == 4)] = b_dist
            all_dist = all_dist.unsqueeze(1).to(symbols.device)
            return all_dist, torch.log_softmax(all_dist, dim=-1)

        max_lengths = torch.tensor([40]*1000, device="cuda:0")
        symbols_out, probs_out = self.gen_gpu.sample(self.config, 1000, 5, max_lengths, 40, mock_autoregressive_fn, self.cache)

        # Every sample should have just a or just b
        a_samples = torch.eq(symbols_out, 3).sum(dim=-1).type(torch.bool)
        b_samples = torch.eq(symbols_out, 4).sum(dim=-1).type(torch.bool)
        self.assertFalse(torch.logical_and(a_samples, b_samples).any())

        # Should be roughly half a, half b
        self.assertAlmostEqual(a_samples.sum()/5000, 0.5, delta=0.02)
        self.assertAlmostEqual(b_samples.sum()/5000, 0.5, delta=0.02)

    def testTopKSameOnGPU(self):
        if not torch.cuda.is_available():
            return

        self.config.decoding_method = DecodingMethod.SAMPLING
        self.config.sampling_k = 5

        dist_cpu = torch.rand(5,6,50)
        dist_gpu = dist_cpu.clone().cuda()

        dist_out_cpu = self.gen_cpu.truncate_probs(self.config, dist_cpu)
        dist_out_gpu = self.gen_gpu.truncate_probs(self.config, dist_gpu)
        self.assertTrue(torch.equal(dist_out_cpu, dist_out_gpu.cpu()))

    def testTopPSameOnGPU(self):
        if not torch.cuda.is_available():
            return

        self.config.decoding_method = DecodingMethod.SAMPLING
        self.config.sampling_p = 0.6

        dist_cpu = torch.nn.functional.softmax(torch.rand(5,6,50), dim=-1)
        dist_gpu = dist_cpu.clone().cuda()

        dist_out_cpu = self.gen_cpu.truncate_probs(self.config, dist_cpu)
        dist_out_gpu = self.gen_gpu.truncate_probs(self.config, dist_gpu)
        self.assertTrue(torch.equal(dist_out_cpu, dist_out_gpu.cpu()))

    def testBeamSearchSameOnGPU(self):
        if not torch.cuda.is_available():
            return

        self.config.decoding_method = DecodingMethod.BEAM_SEARCH
        self.config.length_normalization = LengthNormalization.GOOGLE_METHOD

        dist_cpu = torch.nn.functional.softmax(torch.rand(30,50), dim=-1)
        dist_gpu = dist_cpu.clone().cuda()
        logits_cpu = torch.nn.functional.softmax(torch.rand(30,50), dim=-1)
        logits_gpu = dist_cpu.clone().cuda()

        def auto_fn_cpu(symbols, timestep, cache):
            dist = dist_cpu[0:symbols.size(0)].unsqueeze(1)
            logits = logits_cpu[0:symbols.size(0)].unsqueeze(1)
            return logits, dist
        def auto_fn_gpu(symbols, timestep, cache):
            dist = dist_gpu[0:symbols.size(0)].unsqueeze(1)
            logits = logits_gpu[0:symbols.size(0)].unsqueeze(1)
            return logits, dist

        self.gen_cpu.vocab_size = 50
        self.gen_gpu.vocab_size = 50
        max_lengths_cpu = torch.tensor([5]*5)
        max_lengths_gpu = torch.tensor([5]*5).cuda()
        cache = MockCacheDoesNothing()

        symbols_final_out_cpu, symbols_all_out_cpu, probs_all_out_cpu = self.gen_cpu.beam_search(self.config, 5, 6, max_lengths_cpu, 40, auto_fn_cpu, cache)
        symbols_final_out_gpu, symbols_all_out_gpu, probs_all_out_gpu = self.gen_gpu.beam_search(self.config, 5, 6, max_lengths_gpu, 40, auto_fn_gpu, cache)

        self.assertTrue(torch.equal(symbols_final_out_cpu, symbols_final_out_gpu.cpu()))
        self.assertTrue(torch.equal(symbols_all_out_cpu, symbols_all_out_gpu.cpu()))
        self.assertTrue(torch.equal(probs_all_out_cpu, probs_all_out_gpu.cpu()))



class TestCacheSameOnGPU(unittest.TestCase):

    def setUpCache(self, beam_size):
        self.batch_size = random.randint(3,10)
        self.cache_cpu = BeamCache(batch_size=self.batch_size,
                                   beam_size=beam_size,
                                   num_layers=2,
                                   device="cpu")
        self.cache_gpu = BeamCache(batch_size=self.batch_size,
                                   beam_size=beam_size,
                                   num_layers=2,
                                   device="cuda:0")

        self.src_len = random.randint(5,15)
        self.src_mask_cpu = torch.log((torch.rand((self.batch_size*beam_size,1,self.src_len)) < 0.5).type(torch.int))
        self.src_mask_gpu = self.src_mask_cpu.to("cuda:0")
        self.tgt_mask_cpu = torch.log((torch.rand((self.batch_size*beam_size,1,self.src_len)) < 0.5).type(torch.int))
        self.tgt_mask_gpu = self.tgt_mask_cpu.to("cuda:0")

        self.d_embed = random.randint(10,20)
        self.src_k1_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.src_k1_gpu = self.src_k1_cpu.to("cuda:0")
        self.src_k2_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.src_k2_gpu = self.src_k2_cpu.to("cuda:0")
        self.src_v1_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.src_v1_gpu = self.src_v1_cpu.to("cuda:0")
        self.src_v2_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.src_v2_gpu = self.src_v2_cpu.to("cuda:0")
        self.tgt_k1_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.tgt_k1_gpu = self.tgt_k1_cpu.to("cuda:0")
        self.tgt_k2_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.tgt_k2_gpu = self.tgt_k2_cpu.to("cuda:0")
        self.tgt_v1_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.tgt_v1_gpu = self.tgt_v1_cpu.to("cuda:0")
        self.tgt_v2_cpu = torch.rand((self.batch_size*beam_size,self.src_len,self.d_embed))
        self.tgt_v2_gpu = self.tgt_v2_cpu.to("cuda:0")

        self.cache_cpu.cache_src_mask(self.src_mask_cpu)
        self.cache_cpu.cache_tgt_mask(self.tgt_mask_cpu)
        self.cache_cpu.cache_src_k(23, 0, self.src_k1_cpu)
        self.cache_cpu.cache_src_k(54, 1, self.src_k2_cpu)
        self.cache_cpu.cache_src_v(23, 0, self.src_v1_cpu)
        self.cache_cpu.cache_src_v(54, 1, self.src_v2_cpu)
        self.cache_cpu.cache_tgt_k(67, 0, self.tgt_k1_cpu)
        self.cache_cpu.cache_tgt_k(89, 1, self.tgt_k2_cpu)
        self.cache_cpu.cache_tgt_v(67, 0, self.tgt_v1_cpu)
        self.cache_cpu.cache_tgt_v(89, 1, self.tgt_v2_cpu)

        self.cache_gpu.cache_src_mask(self.src_mask_gpu)
        self.cache_gpu.cache_tgt_mask(self.tgt_mask_gpu)
        self.cache_gpu.cache_src_k(23, 0, self.src_k1_gpu)
        self.cache_gpu.cache_src_k(54, 1, self.src_k2_gpu)
        self.cache_gpu.cache_src_v(23, 0, self.src_v1_gpu)
        self.cache_gpu.cache_src_v(54, 1, self.src_v2_gpu)
        self.cache_gpu.cache_tgt_k(67, 0, self.tgt_k1_gpu)
        self.cache_gpu.cache_tgt_k(89, 1, self.tgt_k2_gpu)
        self.cache_gpu.cache_tgt_v(67, 0, self.tgt_v1_gpu)
        self.cache_gpu.cache_tgt_v(89, 1, self.tgt_v2_gpu)

    def evalCache(self):
        src_mask_out_cpu = self.cache_cpu.get_src_mask()
        src_mask_out_gpu = self.cache_gpu.get_src_mask()
        tgt_mask_out_cpu = self.cache_cpu.get_tgt_mask()
        tgt_mask_out_gpu = self.cache_gpu.get_tgt_mask()
        src_k1_out_cpu = self.cache_cpu.get_k(23)
        src_k1_out_gpu = self.cache_gpu.get_k(23)
        src_k2_out_cpu = self.cache_cpu.get_k(54)
        src_k2_out_gpu = self.cache_gpu.get_k(54)
        src_v1_out_cpu = self.cache_cpu.get_v(23)
        src_v1_out_gpu = self.cache_gpu.get_v(23)
        src_v2_out_cpu = self.cache_cpu.get_v(54)
        src_v2_out_gpu = self.cache_gpu.get_v(54)
        tgt_k1_out_cpu = self.cache_cpu.get_k(67)
        tgt_k1_out_gpu = self.cache_gpu.get_k(67)
        tgt_k2_out_cpu = self.cache_cpu.get_k(89)
        tgt_k2_out_gpu = self.cache_gpu.get_k(89)
        tgt_v1_out_cpu = self.cache_cpu.get_v(67)
        tgt_v1_out_gpu = self.cache_gpu.get_v(67)
        tgt_v2_out_cpu = self.cache_cpu.get_v(89)
        tgt_v2_out_gpu = self.cache_gpu.get_v(89)

        torch.testing.assert_close(src_mask_out_cpu, src_mask_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(tgt_mask_out_cpu, tgt_mask_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(src_k1_out_cpu, src_k1_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(src_k2_out_cpu, src_k2_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(src_v1_out_cpu, src_v1_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(src_v2_out_cpu, src_v2_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(tgt_k1_out_cpu, tgt_k1_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(tgt_k2_out_cpu, tgt_k2_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(tgt_v1_out_cpu, tgt_v1_out_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(tgt_v2_out_cpu, tgt_v2_out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testCacheAndGetSameOnGPU(self):
        self.setUpCache(1)
        self.evalCache()

    def testExpandToBeamSizeSameOnGPU(self):
        beam_size = random.randint(3,6)
        self.setUpCache(1)
        self.cache_cpu.expand_to_beam_size(beam_size)
        self.cache_gpu.expand_to_beam_size(beam_size)
        self.evalCache()

    def testRegisterFinishedSentsSameOnGPU(self):
        beam_size = random.randint(3,6)
        self.setUpCache(beam_size)
        mask1_cpu = torch.rand((self.batch_size * beam_size)) < 0.3
        mask1_gpu = mask1_cpu.to("cuda:0")
        mask2_cpu = torch.logical_and(mask1_cpu, torch.rand((self.batch_size * beam_size)) < 0.3)
        mask2_gpu = mask2_cpu.to("cuda:0")
        self.cache_cpu.register_finished_sents(mask1_cpu)
        self.cache_gpu.register_finished_sents(mask1_gpu)
        self.cache_cpu.register_finished_sents(mask2_cpu)
        self.cache_gpu.register_finished_sents(mask2_gpu)
        self.evalCache()

    def testRegisterFinishedBeamsSameOnGPU(self):
        beam_size = random.randint(3,6)
        self.setUpCache(beam_size)
        mask1_cpu = torch.rand((self.batch_size * beam_size)) < 0.3
        mask1_gpu = mask1_cpu.to("cuda:0")
        mask2_cpu = torch.rand((self.batch_size)) < 0.5
        mask2_gpu = mask2_cpu.to("cuda:0")
        self.cache_cpu.register_finished_sents(mask1_cpu)
        self.cache_gpu.register_finished_sents(mask1_gpu)
        self.cache_cpu.register_finished_beams(mask2_cpu)
        self.cache_gpu.register_finished_beams(mask2_gpu)
        self.evalCache()

    def testSelectIdxsSameOnGPU(self):
        beam_size = random.randint(6,15)
        self.setUpCache(beam_size)
        mask1_cpu = torch.rand((self.batch_size * beam_size)) < 0.3
        mask1_gpu = mask1_cpu.to("cuda:0")
        self.cache_cpu.register_finished_sents(mask1_cpu)
        self.cache_gpu.register_finished_sents(mask1_gpu)
        chosen_idxs_cpu = torch.randint(beam_size, (self.batch_size, beam_size))
        chosen_idxs_gpu = chosen_idxs_cpu.to("cuda:0")
        self.cache_cpu.select_idxs(chosen_idxs_cpu)
        self.cache_gpu.select_idxs(chosen_idxs_gpu)
        self.evalCache()



class TestTransformerSameOnGPU(unittest.TestCase):

    def setUp(self):
        self.config = read_config("configuration.toml")

    def testEmbedding(self):
        if not torch.cuda.is_available():
            return
    
        x = torch.randint(high=10,size=(100,40))
        x_emb = torch.rand(100,40,4)

        emb_cpu = Embedding(10,4,fix_norm=False)
        emb_gpu = copy.deepcopy(emb_cpu).to("cuda:0")
        out_cpu = emb_cpu(x)
        out_gpu = emb_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)
        out_cpu = emb_cpu(x_emb, reverse=True)
        out_gpu = emb_gpu(x_emb.to("cuda:0"), reverse=True)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)
        
        emb_cpu = Embedding(10,4,fix_norm=True)
        emb_gpu = copy.deepcopy(emb_cpu).to("cuda:0")
        out_cpu = emb_cpu(x)
        out_gpu = emb_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)
        out_cpu = emb_cpu(x_emb, reverse=True)
        out_gpu = emb_gpu(x_emb.to("cuda:0"), reverse=True)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testNullPositionalEncoding(self):
        if not torch.cuda.is_available():
            return

        npe_cpu = NullPositionalEncoding()
        npe_gpu = copy.deepcopy(npe_cpu).to("cuda:0")
        x = torch.rand(8,9,10)
        out_cpu = npe_cpu(x)
        out_gpu = npe_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testSinusoidalPositionalEncoding(self):
        if not torch.cuda.is_available():
            return

        spe_cpu = NullPositionalEncoding()
        spe_gpu = copy.deepcopy(spe_cpu).to("cuda:0")
        x = torch.rand(8,9,10)
        out_cpu = spe_cpu(x)
        out_gpu = spe_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testLayerNorm(self):
        if not torch.cuda.is_available():
            return

        ln_cpu = LayerNorm(4,1e-5)
        ln_gpu = copy.deepcopy(ln_cpu).to("cuda:0")
        x = torch.rand(2,3,4)
        out_cpu = ln_cpu(x)
        out_gpu = ln_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testScaleNorm(self):
        if not torch.cuda.is_available():
            return

        sn_cpu = ScaleNorm(scale=1)
        sn_gpu = copy.deepcopy(sn_cpu).to("cuda:0")
        x = torch.rand(2,3,4)
        out_cpu = sn_cpu(x)
        out_gpu = sn_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testFeedForward(self):
        if not torch.cuda.is_available():
            return

        ff_cpu = ff = FeedForward(64, 256, use_toan_init=False, dropout=0.0)
        ff_gpu = copy.deepcopy(ff_cpu).to("cuda:0")
        x = torch.rand(2,3,64)
        out_cpu = ff_cpu(x)
        out_gpu = ff_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testMultiHeadAttention(self):
        if not torch.cuda.is_available():
            return

        ni = float("-inf")
        xmask_cpu = torch.tensor([[[1, ni, ni, ni, ni],
                                   [1, 1,  ni, ni, ni],
                                   [1, 1,  1,  ni, ni],
                                   [1, 1,  1,  1,  ni],
                                   [1, 1,  1,  1,   1]]])
        xmask_gpu = xmask_cpu.to("cuda:0")
        xymask_cpu = torch.rand(100, 1, 5)
        xymask_gpu = xymask_cpu.to("cuda:0")
        x_cpu = torch.rand(100, 5, 64)
        x_gpu = x_cpu.to("cuda:0")
        y_cpu = torch.rand(100, 20, 64)
        y_gpu = y_cpu.to("cuda:0")

        mha_cpu = MultiHeadAttention(64, 8, dropout=0.0)
        mha_gpu = copy.deepcopy(mha_cpu).to("cuda:0")
        # self attention
        out_cpu = mha_cpu(x_cpu, x_cpu, x_cpu, xmask_cpu)
        out_gpu = mha_gpu(x_gpu, x_gpu, x_gpu, xmask_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)
        # cross attention
        out_cpu = mha_cpu(y_cpu, x_cpu, x_cpu, xymask_cpu)
        out_gpu = mha_gpu(y_gpu, x_gpu, x_gpu, xymask_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testSublayerConnection(self):
        if not torch.cuda.is_available():
            return

        def mock_sublayer_func(y):
            return y
        y_cpu = torch.rand(100,20,512)
        y_gpu = y_cpu.to("cuda:0")
        self.config.train.dropout = 0.0

        # pre norm, resid
        self.config.arch.pre_norm = True
        self.config.arch.use_resid_connection = True

        slc_cpu = SublayerConnection(self.config)
        slc_gpu = copy.deepcopy(slc_cpu).to("cuda:0")
        out_cpu = slc_cpu(y_cpu, mock_sublayer_func)
        out_gpu = slc_gpu(y_gpu, mock_sublayer_func)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

        # pre norm, no resid
        self.config.arch.pre_norm = True
        self.config.arch.use_resid_connection = False

        slc_cpu = SublayerConnection(self.config)
        slc_gpu = copy.deepcopy(slc_cpu).to("cuda:0")
        out_cpu = slc_cpu(y_cpu, mock_sublayer_func)
        out_gpu = slc_gpu(y_gpu, mock_sublayer_func)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

        # post norm, resid
        self.config.arch.pre_norm = False
        self.config.arch.use_resid_connection = True

        slc_cpu = SublayerConnection(self.config)
        slc_gpu = copy.deepcopy(slc_cpu).to("cuda:0")
        out_cpu = slc_cpu(y_cpu, mock_sublayer_func)
        out_gpu = slc_gpu(y_gpu, mock_sublayer_func)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

        # post norm, no resid
        self.config.arch.pre_norm = False
        self.config.arch.use_resid_connection = False

        slc_cpu = SublayerConnection(self.config)
        slc_gpu = copy.deepcopy(slc_cpu).to("cuda:0")
        out_cpu = slc_cpu(y_cpu, mock_sublayer_func)
        out_gpu = slc_gpu(y_gpu, mock_sublayer_func)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testLayer(self):
        if not torch.cuda.is_available():
            return

        x_cpu = torch.rand(100,10,512)
        x_gpu = x_cpu.to("cuda:0")
        y_cpu = torch.rand(100,20,512)
        y_gpu = y_cpu.to("cuda:0")
        xymask_cpu = torch.rand(100,20,10)
        xymask_gpu = xymask_cpu.to("cuda:0")
        ymask_cpu = torch.rand(100,20,20)
        ymask_gpu = ymask_cpu.to("cuda:0")
        self.config.train.dropout = 0.0
        self.config.train.att_dropout = 0.0
        self.config.train.ff_dropout = 0.0

        l_cpu = Layer(self.config, take_two_seqs=True)
        l_gpu = copy.deepcopy(l_cpu).to("cuda:0")
        out_cpu = l_cpu(y_cpu, ymask_cpu, x_cpu, xymask_cpu)
        out_gpu = l_gpu(y_gpu, ymask_gpu, x_gpu, xymask_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testEncoderOrDecoder(self):
        if not torch.cuda.is_available():
            return

        x_cpu = torch.rand(30,10,512)
        x_gpu = x_cpu.to("cuda:0")
        y_cpu = torch.rand(30,20,512)
        y_gpu = y_cpu.to("cuda:0")
        xmask_cpu = torch.rand(30,1,10)
        xmask_gpu = xmask_cpu.to("cuda:0")
        ymask_cpu = torch.rand(30,1,20)
        ymask_gpu = ymask_cpu.to("cuda:0")
        self.config.train.dropout = 0.0
        self.config.train.att_dropout = 0.0
        self.config.train.ff_dropout = 0.0

        # decoder
        eod_cpu = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, masked_self_att=True)
        eod_gpu = copy.deepcopy(eod_cpu).to("cuda:0")
        out_cpu = eod_cpu(y_cpu, ymask_cpu, x_cpu, xmask_cpu)
        out_gpu = eod_gpu(y_gpu, ymask_gpu, x_gpu, xmask_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testInputLayer(self):
        if not torch.cuda.is_available():
            return

        x_cpu = torch.randint(high=1000,size=(5,10))
        x_gpu = x_cpu.to("cuda:0")

        emb = get_embedding(self.config, 1000)
        il_cpu = InputLayer(self.config, emb)
        il_gpu = copy.deepcopy(il_cpu).to("cuda:0")
        out_cpu = il_cpu(x_cpu)
        out_gpu = il_gpu(x_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testOutputLayer(self):
        if not torch.cuda.is_available():
            return

        x_cpu = torch.rand(5,10,self.config.arch.d_model)
        x_gpu = x_cpu.to("cuda:0")

        emb = get_embedding(self.config, 1000)
        ol_cpu = OutputLayer(emb, 1000, support_mask=None)
        ol_gpu = copy.deepcopy(ol_cpu).to("cuda:0")
        logits_cpu, probs_cpu = ol_cpu(x_cpu)
        logits_gpu, probs_gpu = ol_gpu(x_gpu)
        torch.testing.assert_close(logits_cpu, logits_gpu.to("cpu"), atol=0.00001, rtol=0)
        torch.testing.assert_close(probs_cpu, probs_gpu.to("cpu"), atol=0.00001, rtol=0)

    def testTransformerTwoSeq(self):
        if not torch.cuda.is_available():
            return

        x_cpu = torch.randint(high=1000,size=(5,10))
        x_gpu = x_cpu.to("cuda:0")
        y_cpu = torch.randint(high=1000,size=(5,20))
        y_gpu = y_cpu.to("cuda:0")
        self.config.train.dropout = 0.0
        self.config.train.att_dropout = 0.0
        self.config.train.ff_dropout = 0.0

        t_cpu = TransformerTwoSeq(self.config, num_enc_layers=6, masked_self_att_enc=True, num_dec_layers=6, masked_self_att_dec=True, output_probs=True, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        t_gpu = copy.deepcopy(t_cpu).to("cuda:0")
        out_cpu = t_cpu(x_cpu, y_cpu)
        out_gpu = t_gpu(x_gpu, y_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00005, rtol=0)

        cache_cpu = BeamCache(5,1,6,"cpu")
        cache_gpu = BeamCache(5,1,6,"cuda:0")
        auto_fn_cpu = t_cpu.get_autoregressive_one_step_fn(x_cpu, cache_cpu)
        auto_fn_gpu = t_gpu.get_autoregressive_one_step_fn(x_gpu, cache_gpu)
        logits_cpu, probs_cpu = auto_fn_cpu(y_cpu, 0, cache_cpu)
        logits_gpu, probs_gpu = auto_fn_gpu(y_gpu, 0, cache_gpu)
        torch.testing.assert_close(logits_cpu, logits_gpu.to("cpu"), atol=0.00005, rtol=0)
        torch.testing.assert_close(probs_cpu, probs_gpu.to("cpu"), atol=0.00005, rtol=0)

    def testTransformerOneSeq(self):
        if not torch.cuda.is_available():
            return

        y_cpu = torch.randint(high=1000,size=(5,20))
        y_gpu = y_cpu.to("cuda:0")
        self.config.train.dropout = 0.0
        self.config.train.att_dropout = 0.0
        self.config.train.ff_dropout = 0.0

        t_cpu = TransformerOneSeq(self.config, num_layers=6, masked_self_att=True, output_probs=True, vocab_size=1000, pad_idx=0, support_mask=None)
        t_gpu = copy.deepcopy(t_cpu).to("cuda:0")
        out_cpu = t_cpu(y_cpu)
        out_gpu = t_gpu(y_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.00005, rtol=0)

        cache_cpu = BeamCache(5,1,6,"cpu")
        cache_gpu = BeamCache(5,1,6,"cuda:0")
        auto_fn_cpu = t_cpu.get_autoregressive_one_step_fn(cache_cpu,5,"cpu")
        auto_fn_gpu = t_gpu.get_autoregressive_one_step_fn(cache_gpu,5,"cuda:0")
        logits_cpu, probs_cpu = auto_fn_cpu(y_cpu, 0, cache_cpu)
        logits_gpu, probs_gpu = auto_fn_gpu(y_gpu, 0, cache_gpu)
        torch.testing.assert_close(logits_cpu, logits_gpu.to("cpu"), atol=0.00005, rtol=0)
        torch.testing.assert_close(probs_cpu, probs_gpu.to("cpu"), atol=0.00005, rtol=0)
