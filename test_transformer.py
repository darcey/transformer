# TODO(darcey): write tests which confirm that the model has all the parameters it's supposed to
# TODO(darcey): factor the tests so that mocks exist somewhere centralized instead of being created inside each test (and copied over to test_transformer_gpu sometimes also)

import torch
import torch.testing
import unittest
from configuration import *
from transformer import *



class TestEmbedding(unittest.TestCase):
    
    def testShape(self):
        x = torch.randint(high=10,size=(100,40))
        x_emb = torch.rand(100,40,4)

        emb = Embedding(10,4,fix_norm=False)
        out = emb(x)
        self.assertEqual(out.shape, (100,40,4))
        out = emb(x_emb, reverse=True)
        self.assertEqual(out.shape, (100,40,10))
        
        emb = Embedding(10,4,fix_norm=True)
        out = emb(x)
        self.assertEqual(out.shape, (100,40,4))
        out = emb(x_emb, reverse=True)
        self.assertEqual(out.shape, (100,40,10))

    def testCorrectnessNoFixNorm(self):
        emb = Embedding(2,4,fix_norm=False)
        emb.embedding = torch.nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                                         [2.0, 2.0, 2.0, 2.0]]))

        input_tensor = torch.tensor([[0,1]])
        correct_tensor = torch.tensor([[[2.0, 2.0, 2.0, 2.0],
                                        [4.0, 4.0, 4.0, 4.0]]])
        actual_tensor = emb(input_tensor, reverse=False)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        input_tensor = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        correct_tensor = torch.tensor([[[4.0, 8.0]]])
        actual_tensor = emb(input_tensor, reverse=True)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

    def testCorrectnessFixNorm(self):
        emb = Embedding(2,4,fix_norm=True)
        emb.embedding = torch.nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                                         [2.0, 2.0, 2.0, 2.0]]))

        input_tensor = torch.tensor([[0]])
        correct_tensor = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        actual_tensor = emb(input_tensor, reverse=False)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        input_tensor = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
        correct_tensor = torch.tensor([[[2.0, 2.0]]])
        actual_tensor = emb(input_tensor, reverse=True)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestNullPositionalEncoding(unittest.TestCase):

    def testCorrectness(self):
        npe = NullPositionalEncoding()
        correct_tensor = torch.zeros(6, 17, 12)
        actual_tensor = npe(torch.rand(6, 17, 12))
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

class TestSinusoidalPositionalEncoding(unittest.TestCase):

    def testShape(self):
        spe = SinusoidalPositionalEncoding(5, 4, 10000)
        x = torch.rand(10,5,4)
        out = spe(x)
        self.assertEqual(out.shape, (5,4))

    def testCorrectness(self):
        from math import sin, cos
        spe = SinusoidalPositionalEncoding(5, 4, 100)
        correct_tensor = torch.tensor([[sin(0/1), cos(0/1), sin(0/10), cos(0/10)],
                                       [sin(1/1), cos(1/1), sin(1/10), cos(1/10)],
                                       [sin(2/1), cos(2/1), sin(2/10), cos(2/10)],
                                       [sin(3/1), cos(3/1), sin(3/10), cos(3/10)],
                                       [sin(4/1), cos(4/1), sin(4/10), cos(4/10)]])
        actual_tensor = spe(torch.zeros(50, 5, 4))
        self.assertEqual(actual_tensor.shape, correct_tensor.shape)
        torch.testing.assert_close(actual_tensor, correct_tensor, atol=0.005, rtol=0)

    def testOddDimension(self):
        from math import sin, cos
        spe = SinusoidalPositionalEncoding(5, 3, 100)
        exp = math.pow(100, 2/3)
        correct_tensor = torch.tensor([[sin(0/1), cos(0/1), sin(0/exp)],
                                       [sin(1/1), cos(1/1), sin(1/exp)],
                                       [sin(2/1), cos(2/1), sin(2/exp)],
                                       [sin(3/1), cos(3/1), sin(3/exp)],
                                       [sin(4/1), cos(4/1), sin(4/exp)]])
        actual_tensor = spe(torch.zeros(50, 5, 3))
        self.assertEqual(actual_tensor.shape, correct_tensor.shape)
        torch.testing.assert_close(actual_tensor, correct_tensor, atol=0.005, rtol=0)



class TestLayerNorm(unittest.TestCase):

    def testShape(self):
        ln = LayerNorm(4,1e-5)
        x = torch.rand(6,5,4)
        out = ln(x)
        self.assertEqual(out.shape, (6,5,4))

    def testZeros(self):
        ln = LayerNorm(4,1e-5)
        correct_tensor = ln.beta.unsqueeze(0).unsqueeze(0).expand(6,5,-1)
        actual_tensor = ln(torch.zeros(6,5,4))
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

    def testMeanVar(self):
        ln = LayerNorm(2,0)
        ln.gamma = torch.nn.Parameter(torch.ones(2))
        ln.beta = torch.nn.Parameter(torch.zeros(2))
        input_tensor = torch.tensor([[[3.0,1.0]]])
        correct_tensor = torch.tensor([[[1.0,-1.0]]])
        actual_tensor = ln(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

class TestScaleNorm(unittest.TestCase):

    def testShape(self):
        sn = ScaleNorm(scale=1)
        x = torch.rand(6,5,4)
        out = sn(x)
        self.assertEqual(out.shape, (6,5,4))

    def testCorrectness(self):
        sn = ScaleNorm(scale=1)
        input_tensor = torch.tensor([[[1.0,1.0,1.0,1.0]]])
        
        sn.g = torch.nn.Parameter(torch.tensor(1.0))
        correct_tensor = torch.tensor([[[0.5,0.5,0.5,0.5]]])
        actual_tensor = sn(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))
        
        sn.g = torch.nn.Parameter(torch.tensor(5.0))
        correct_tensor = torch.tensor([[[2.5,2.5,2.5,2.5]]])
        actual_tensor = sn(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestFeedForward(unittest.TestCase):

    def testShape(self):
        ff = FeedForward(64, 256, use_toan_init=False, dropout=0.3)
        x = torch.rand(100, 10, 64)
        out = ff(x)
        self.assertEqual(out.shape, (100, 10, 64))

        ff = FeedForward(64, 256, use_toan_init=True, dropout=0.3)
        x = torch.rand(100, 10, 64)
        out = ff(x)
        self.assertEqual(out.shape, (100, 10, 64))

    def testCorrectness(self):
        ff = FeedForward(5, 5, use_toan_init=False, dropout=0.0)
        ff.layer1.weight = torch.nn.Parameter(torch.eye(5))
        ff.layer1.bias = torch.nn.Parameter(torch.zeros(5))
        ff.layer2.weight = torch.nn.Parameter(torch.eye(5))
        ff.layer2.bias = torch.nn.Parameter(torch.zeros(5))

        input_tensor = torch.tensor([[[-1.0, 1.0, 3.0, -4.0, 2.0]]])
        correct_tensor = torch.tensor([[[0.0, 1.0, 3.0, 0.0, 2.0]]])
        actual_tensor = ff(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

    def testDropout(self):
        ff = FeedForward(5, 5, use_toan_init=False, dropout=1.0)
        ff.layer2.bias = torch.nn.Parameter(torch.zeros(5))
        ff.train()
        input_tensor = torch.rand(10, 12, 5)
        correct_tensor = torch.zeros(10, 12, 5)
        actual_tensor = ff(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestMultiHeadAttention(unittest.TestCase):

    def testInitBadParams(self):
        with self.assertRaises(ValueError):
            mha = MultiHeadAttention(20, 8)
        with self.assertRaises(ValueError):
            mha = MultiHeadAttention(20, 8, qk_dim=4)
        with self.assertRaises(ValueError):
            mha = MultiHeadAttention(20, 8, v_dim=4)
        mha = MultiHeadAttention(24, 8)
        mha = MultiHeadAttention(20, 8, qk_dim=4, v_dim=4)

    def testShape(self):
        x = torch.rand(100, 5, 64)
        y = torch.rand(100, 20, 64)
        xmask1 = torch.rand(100, 1, 5)
        xmask2 = torch.rand(100, 5, 5)
        xymask1 = torch.rand(100, 1, 20)
        xymask2 = torch.rand(100, 5, 20)

        # one head
        mha = MultiHeadAttention(64,1)
        out = mha(x, x, x, xmask1)
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, x, x, xmask2)
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, y, y, xymask1)
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, y, y, xymask2)
        self.assertEqual(out.shape, (100,5,64))

        # multiple heads
        mha = MultiHeadAttention(64,8)
        out = mha(x, x, x, xmask1)
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, x, x, xmask2)
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, y, y, xymask1)
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, y, y, xymask2)
        self.assertEqual(out.shape, (100,5,64))

        # both init strategies
        mha = MultiHeadAttention(64,8,use_toan_init=False)
        out = mha(x, x, x, xmask1)
        self.assertEqual(out.shape, (100,5,64))
        mha = MultiHeadAttention(64,8,use_toan_init=True)
        out = mha(x, x, x, xmask1)
        self.assertEqual(out.shape, (100,5,64))

    def testMask(self):
        mha = MultiHeadAttention(4,1,dropout=0.0)
        mha.proj_q.weight = torch.nn.Parameter(torch.eye(4))
        mha.proj_k.weight = torch.nn.Parameter(torch.eye(4))
        mha.proj_v.weight = torch.nn.Parameter(torch.eye(4))
        mha.proj_out.weight = torch.nn.Parameter(torch.eye(4))

        ni = float("-inf")
        mask = torch.tensor([[[1, ni, ni, ni, ni],
                              [1, 1,  ni, ni, ni],
                              [1, 1,  1,  ni, ni],
                              [1, 1,  1,  1,  ni],
                              [1, 1,  1,  1,   1]]])
        input_tensor = torch.rand(5, 4).unsqueeze(0)
        output_tensor = mha(input_tensor, input_tensor, input_tensor, mask)
        self.assertTrue(torch.equal(output_tensor[0,0], input_tensor[0,0]))

    def testDropout(self):
        mha = MultiHeadAttention(64,8,dropout=1.0)
        mha.train()

        input_tensor = torch.rand(100,5,64)
        correct_tensor = torch.zeros(100,5,64)
        mask = torch.zeros(100,1,5)
        actual_tensor = mha(input_tensor, input_tensor, input_tensor, mask)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestSublayerConnection(unittest.TestCase):

    def setUp(self):
        self.config = read_config("configuration.toml")

    def testShape(self):
        def mock_sublayer_func(y):
            return y
        y = torch.rand(100,20,512)

        # pre norm, resid
        self.config.arch.pre_norm = True
        self.config.arch.use_resid_connection = True

        slc = SublayerConnection(self.config)
        out = slc(y, mock_sublayer_func)
        self.assertEqual(out.shape, (100,20,512))

        # pre norm, no resid
        self.config.arch.pre_norm = True
        self.config.arch.use_resid_connection = False

        slc = SublayerConnection(self.config)
        out = slc(y, mock_sublayer_func)
        self.assertEqual(out.shape, (100,20,512))

        # post norm, resid
        self.config.arch.pre_norm = False
        self.config.arch.use_resid_connection = True

        slc = SublayerConnection(self.config)
        out = slc(y, mock_sublayer_func)
        self.assertEqual(out.shape, (100,20,512))

        # post norm, no resid
        self.config.arch.pre_norm = False
        self.config.arch.use_resid_connection = False

        slc = SublayerConnection(self.config)
        out = slc(y, mock_sublayer_func)
        self.assertEqual(out.shape, (100,20,512))

    def testCorrectnessMocked(self):
        class MockNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, y):
                return 10 * y
        mock_norm = MockNorm()
        mock_sublayer_func = lambda y: y + 2
        input_tensor = torch.ones(3,4,5)
        self.config.train.dropout = 0.0

        # pre norm, resid
        self.config.arch.pre_norm = True
        self.config.arch.use_resid_connection = True
        slc = SublayerConnection(self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 13
        actual_tensor = slc(input_tensor, mock_sublayer_func)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        # pre norm, no resid
        self.config.arch.pre_norm = True
        self.config.arch.use_resid_connection = False
        slc = SublayerConnection(self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 12
        actual_tensor = slc(input_tensor, mock_sublayer_func)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        # post norm, resid
        self.config.arch.pre_norm = False
        self.config.arch.use_resid_connection = True
        slc = SublayerConnection(self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 40
        actual_tensor = slc(input_tensor, mock_sublayer_func)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        # post norm, no resid
        self.config.arch.pre_norm = False
        self.config.arch.use_resid_connection = False
        slc = SublayerConnection(self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 30
        actual_tensor = slc(input_tensor, mock_sublayer_func)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

    def testDropout(self):
        def mock_sublayer_func(y):
            return y
        self.config.arch.pre_norm = True
        self.config.train.dropout = 1.0

        slc = SublayerConnection(self.config)
        slc.train()
        input_tensor = torch.rand(100,20,512)
        actual_tensor = slc(input_tensor, mock_sublayer_func)
        self.assertTrue(torch.equal(actual_tensor, input_tensor))



class TestLayer(unittest.TestCase):

    def setUp(self):
        self.config = read_config("configuration.toml")

    def testBadInput(self):
        x = torch.rand(100,10,512)
        y = torch.rand(100,20,512)
        xmask = torch.rand(100,1,10)
        ymask = torch.rand(100,1,20)

        # one sequence
        l = Layer(self.config, take_two_seqs=False)
        with self.assertRaises(ValueError):
            out = l(y, ymask, x, xmask)
        with self.assertRaises(ValueError):
            out = l(y, ymask, x)
        with self.assertRaises(ValueError):
            out = l(y, ymask, prev_mask=xmask)

        # two sequence
        l = Layer(self.config, take_two_seqs=True)
        with self.assertRaises(ValueError):
            out = l(y, ymask)
        with self.assertRaises(ValueError):
            out = l(y, ymask, x)
        with self.assertRaises(ValueError):
            out = l(y, ymask, prev_mask=xmask)

    def testShape(self):
        x = torch.rand(100,10,512)
        y = torch.rand(100,20,512)
        xmask = torch.rand(100,1,10)
        ymask = torch.rand(100,1,20)

        # one sequence
        l = Layer(self.config, take_two_seqs=False)
        out = l(y, ymask)
        self.assertEqual(out.shape, (100,20,512))

        # two sequence
        l = Layer(self.config, take_two_seqs=True)
        out = l(y, ymask, x, xmask)
        self.assertEqual(out.shape, (100,20,512))



class TestEncoderOrDecoder(unittest.TestCase):

    def setUp(self):
        self.config = read_config("configuration.toml")

    def testShape(self):
        x = torch.rand(50,10,512)
        y = torch.rand(50,20,512)
        xmask = torch.rand(50,1,10)
        ymask = torch.rand(50,1,20)

        # encoder
        e = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=False, masked_self_att=False)
        out = e(y, ymask)
        self.assertEqual(out.shape, (50,20,512))

        # decoder only
        do = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=False, masked_self_att=True)
        out = do(y, ymask)
        self.assertEqual(out.shape, (50,20,512))

        # ???
        weird = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, masked_self_att=False)
        out = weird(y, ymask, x, xmask)
        self.assertEqual(out.shape, (50,20,512))

        # decoder
        d = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, masked_self_att=True)
        out = d(y, ymask, x, xmask)
        self.assertEqual(out.shape, (50,20,512))

        # prenorm
        self.config.arch.pre_norm = True
        d_pre = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, masked_self_att=True)
        out = d_pre(y, ymask, x, xmask)
        self.assertEqual(out.shape, (50,20,512))

        # postnorm
        self.config.arch.pre_norm = False
        d_post = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, masked_self_att=True)
        out = d_post(y, ymask, x, xmask)
        self.assertEqual(out.shape, (50,20,512))



class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.config = read_config("configuration.toml")

    def testTwoSeqShape(self):
        x = torch.randint(high=1000,size=(5,10))
        y = torch.randint(high=1000,size=(5,20))

        t = TransformerTwoSeq(self.config, num_enc_layers=6, masked_self_att_enc=True, num_dec_layers=6, masked_self_att_dec=False, output_probs=False, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        out = t(x, y)
        self.assertEqual(out.shape, (5,20,512))

        t = TransformerTwoSeq(self.config, num_enc_layers=6, masked_self_att_enc=True, num_dec_layers=6, masked_self_att_dec=False, output_probs=True, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        out = t(x, y)
        self.assertEqual(out.shape, (5,20,1000))

    def testOneSeqShape(self):
        y = torch.randint(high=1000,size=(5,20))

        t = TransformerOneSeq(self.config, num_layers=6, masked_self_att=True, output_probs=False, vocab_size=1000, pad_idx=0, support_mask=None)
        out = t(y)
        self.assertEqual(out.shape, (5,20,512))

        t = TransformerOneSeq(self.config, num_layers=6, masked_self_att=True, output_probs=True, vocab_size=1000, pad_idx=0, support_mask=None)
        out = t(y)
        self.assertEqual(out.shape, (5,20,1000))

    def testEncoderDecoderShape(self):
        x = torch.randint(high=1000,size=(5,10))
        y = torch.randint(high=1000,size=(5,20))

        self.config.arch.transformer_type = TransformerType.ENCODER_DECODER
        t = get_transformer(self.config, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        out = t(x, y)
        self.assertEqual(out.shape, (5,20,1000))

    def testEncoderOnlyShape(self):
        x = torch.randint(high=1000,size=(5,10))

        self.config.arch.transformer_type = TransformerType.ENCODER_ONLY
        t = get_transformer(self.config, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        out = t(x)
        self.assertEqual(out.shape, (5,10,512))

    def testDecoderOnlyShape(self):
        y = torch.randint(high=1000,size=(5,20))

        self.config.arch.transformer_type = TransformerType.DECODER_ONLY
        t = get_transformer(self.config, vocab_size=1000, pad_idx=0, tgt_support_mask=None)
        out = t(y)
        self.assertEqual(out.shape, (5,20,1000))

    def testGetPadMask(self):
        seq = torch.tensor([[3,4,5,6,0,0],
                            [0,2,0,0,1,0]])
        ni = float("-inf")
        correct_mask = torch.tensor([[[0.,0.,0.,0.,ni,ni]],
                                     [[ni,0.,ni,ni,0.,ni]]])
        actual_mask = get_pad_mask(seq, pad_idx=0)
        self.assertTrue(torch.equal(actual_mask, correct_mask))

    def testDropout(self):
        self.config.train.dropout = 1.0
        self.config.train.ff_dropout = 0.0
        self.config.train.att_dropout = 0.0

        x1 = torch.randint(low=1,high=30,size=(5,10))
        x2 = torch.randint(low=1,high=30,size=(5,10))
        y1 = torch.randint(low=1,high=30,size=(5,20))
        y2 = torch.randint(low=1,high=30,size=(5,20))

        t = TransformerTwoSeq(self.config, num_enc_layers=6, masked_self_att_enc=True, num_dec_layers=6, masked_self_att_dec=False, output_probs=False, vocab_size=30, pad_idx=0, tgt_support_mask=None)
        t.train()
        out1 = t(x1, y1)
        out2 = t(x2, y2)
        self.assertTrue(torch.equal(out1, out2))

        t = TransformerOneSeq(self.config, num_layers=6, masked_self_att=True, output_probs=False, vocab_size=30, pad_idx=0, support_mask=None)
        t.train()
        out1 = t(y1)
        out2 = t(y2)
        self.assertTrue(torch.equal(out1, out2))

    def testTgtVocabMask(self):
        x = torch.randint(low=1,high=5,size=(4,3))
        y = torch.randint(low=1,high=5,size=(4,3))
        tgt_support_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])

        t = TransformerTwoSeq(self.config, num_enc_layers=6, masked_self_att_enc=False, num_dec_layers=6, masked_self_att_dec=True, output_probs=True, vocab_size=5, pad_idx=0, tgt_support_mask=tgt_support_mask)
        out = t(x, y)
        self.assertEqual(out.shape, (4,3,5))
        self.assertTrue(out[0,0,0] > float('-inf'))
        self.assertTrue(out[0,0,1] > float('-inf'))
        self.assertTrue(out[0,0,2] == float('-inf'))
        self.assertTrue(out[0,0,3] > float('-inf'))
        self.assertTrue(out[0,0,4] == float('-inf'))

        t = TransformerOneSeq(self.config, num_layers=6, masked_self_att=True, output_probs=True, vocab_size=5, pad_idx=0, support_mask=tgt_support_mask)
        out = t(y)
        self.assertEqual(out.shape, (4,3,5))
        self.assertTrue(out[0,0,0] > float('-inf'))
        self.assertTrue(out[0,0,1] > float('-inf'))
        self.assertTrue(out[0,0,2] == float('-inf'))
        self.assertTrue(out[0,0,3] > float('-inf'))
        self.assertTrue(out[0,0,4] == float('-inf'))
