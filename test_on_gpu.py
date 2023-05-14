import copy
import torch
import torch.testing
import unittest
from tempfile import TemporaryDirectory
from configuration import *
from trainer import *
from vocabulary import *
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

        sn_cpu = ScaleNorm()
        sn_gpu = copy.deepcopy(sn_cpu).to("cuda:0")
        x = torch.rand(2,3,4)
        out_cpu = sn_cpu(x)
        out_gpu = sn_gpu(x.to("cuda:0"))
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

    def testFeedForward(self):
        if not torch.cuda.is_available():
            return

        ff_cpu = ff = FeedForward(64, 256, dropout=0.0)
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

        x_cpu = torch.rand(100,10,512)
        x_gpu = x_cpu.to("cuda:0")
        y_cpu = torch.rand(100,20,512)
        y_gpu = y_cpu.to("cuda:0")
        xmask_cpu = torch.rand(100,1,10)
        xmask_gpu = xmask_cpu.to("cuda:0")
        ymask_cpu = torch.rand(100,1,20)
        ymask_gpu = ymask_cpu.to("cuda:0")
        self.config.train.dropout = 0.0
        self.config.train.att_dropout = 0.0
        self.config.train.ff_dropout = 0.0

        # decoder
        eod_cpu = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, masked_self_att=True)
        eod_gpu = copy.deepcopy(eod_cpu).to("cuda:0")
        out_cpu = eod_cpu(y_cpu, ymask_cpu, x_cpu, xmask_cpu)
        out_gpu = eod_gpu(y_gpu, ymask_gpu, x_gpu, xmask_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

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

        t_cpu = TransformerTwoSeq(self.config, num_enc_layers=6, masked_self_att_enc=True, num_dec_layers=6, masked_self_att_dec=False, output_probs=True, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        t_gpu = copy.deepcopy(t_cpu).to("cuda:0")
        out_cpu = t_cpu(x_cpu, y_cpu)
        out_gpu = t_gpu(x_gpu, y_gpu)
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)

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
        torch.testing.assert_close(out_cpu, out_gpu.to("cpu"), atol=0.000001, rtol=0)
