# TODO(darcey): write tests which confirm that stuff gets gradient updated
# TODO(darcey): do the unit tests need to be written to run on GPU?

import torch
import torch.testing
import unittest
from configuration import *
from transformer import *



class TestEmbedding(unittest.TestCase):
    
    def testShape(self):
        x = torch.rand(100,40,10)
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

        input_tensor = torch.tensor([[[1.0, 0.0],
                                      [0.0, 1.0]]])
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
        emb.g = torch.nn.Parameter(torch.tensor(1.0))
        emb.embedding = torch.nn.Parameter(torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                                         [2.0, 2.0, 2.0, 2.0]]))

        input_tensor = torch.tensor([[[1.0, 0.0]]])
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
        sn = ScaleNorm()
        x = torch.rand(6,5,4)
        out = sn(x)
        self.assertEqual(out.shape, (6,5,4))

    def testCorrectness(self):
        sn = ScaleNorm()
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
        ff = FeedForward(64, 256)
        x = torch.rand(100, 10, 64)
        out = ff(x)
        self.assertEqual(out.shape, (100, 10, 64))

    def testCorrectness(self):
        ff = FeedForward(5, 5)
        ff.layer1.weight = torch.nn.Parameter(torch.eye(5))
        ff.layer1.bias = torch.nn.Parameter(torch.zeros(5))
        ff.layer2.weight = torch.nn.Parameter(torch.eye(5))
        ff.layer2.bias = torch.nn.Parameter(torch.zeros(5))

        input_tensor = torch.tensor([[[-1.0, 1.0, 3.0, -4.0, 2.0]]])
        correct_tensor = torch.tensor([[[0.0, 1.0, 3.0, 0.0, 2.0]]])
        actual_tensor = ff(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestMultiHeadAttention(unittest.TestCase):

    def testBadInput(self):
        with self.assertRaises(ValueError):
            mha = MultiHeadAttention(20, 8)
        with self.assertRaises(ValueError):
            mha = MultiHeadAttention(20, 8, qk_dim=4)
        with self.assertRaises(ValueError):
            mha = MultiHeadAttention(20, 8, v_dim=4)
        mha = MultiHeadAttention(24, 8)
        mha = MultiHeadAttention(20, 8, qk_dim=4, v_dim=4)

    def testShape(self):
        ni = float("-inf")
        mask = torch.tensor([[1, ni, ni, ni, ni],
                             [1, 1,  ni, ni, ni],
                             [1, 1,  1,  ni, ni],
                             [1, 1,  1,  1,  ni],
                             [1, 1,  1,  1,   1]])
        x = torch.rand(100, 5, 64)
        y = torch.rand(100, 20, 64)

        # one head
        mha = MultiHeadAttention(64,1)
        out = mha(x, x, x)        # self attention
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, x, x, mask)  # masked self attention
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, y, y)        # cross attention
        self.assertEqual(out.shape, (100,5,64))

        # multiple heads
        mha = MultiHeadAttention(64,8)
        out = mha(x, x, x)        # self attention
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, x, x, mask)  # masked self attention
        self.assertEqual(out.shape, (100,5,64))
        out = mha(x, y, y)        # cross attention
        self.assertEqual(out.shape, (100,5,64))

    def testMask(self):
        mha = MultiHeadAttention(4,1)
        mha.proj_q.weight = torch.nn.Parameter(torch.eye(4))
        mha.proj_k.weight = torch.nn.Parameter(torch.eye(4))
        mha.proj_v.weight = torch.nn.Parameter(torch.eye(4))
        mha.proj_out.weight = torch.nn.Parameter(torch.eye(4))

        ni = float("-inf")
        mask = torch.tensor([[1, ni, ni, ni, ni],
                             [1, 1,  ni, ni, ni],
                             [1, 1,  1,  ni, ni],
                             [1, 1,  1,  1,  ni],
                             [1, 1,  1,  1,   1]])
        input_tensor = torch.rand(5, 4).unsqueeze(0)
        output_tensor = mha(input_tensor, input_tensor, input_tensor, mask=mask)
        self.assertTrue(torch.equal(output_tensor[0,0], input_tensor[0,0]))



class TestSublayerConnection(unittest.TestCase):

    def setUp(self):
        self.config = get_config_arch()

    def testShape(self):
        mock_sublayer = None
        def mock_sublayer_func(s, y, *other_inputs):
            return y
        y = torch.rand(100,20,512)

        # pre norm, resid
        self.config.pre_norm = True
        self.config.use_resid_connection = True

        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        out = slc(y)
        self.assertEqual(out.shape, (100,20,512))

        # pre norm, no resid
        self.config.pre_norm = True
        self.config.use_resid_connection = False

        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        out = slc(y)
        self.assertEqual(out.shape, (100,20,512))

        # post norm, resid
        self.config.pre_norm = False
        self.config.use_resid_connection = True

        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        out = slc(y)
        self.assertEqual(out.shape, (100,20,512))

        # post norm, no resid
        self.config.pre_norm = False
        self.config.use_resid_connection = False

        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        out = slc(y)
        self.assertEqual(out.shape, (100,20,512))

    def testPassesArgsCorrectly(self):
        # multiple args in a weird order
        y = torch.rand(100,20,512)
        x = torch.rand(100,10,512)
        m = torch.rand(20,20)
        def mock_sublayer(y, m, x1, x2):
            self.assertEqual(y.shape, (100,20,512))
            self.assertEqual(m.shape, (20,20))
            self.assertEqual(x1.shape, (100,10,512))
            self.assertTrue(torch.equal(x1, x2))
            return y
        mock_sublayer_func = lambda s, y, x, m: s(y, m, x, x)

        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        slc(y, x, m)

        # no args
        y = torch.rand(100,20,512)
        def mock_sublayer(y):
            self.assertEqual(y.shape, (100,20,512))
            return y
        mock_sublayer_func = lambda s, y: s(y)

        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        slc(y)

    def testCorrectnessMocked(self):
        class MockNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, y):
                return 10 * y
        mock_norm = MockNorm()
        mock_sublayer = lambda y: y + 2
        mock_sublayer_func = lambda s, y: s(y)
        input_tensor = torch.ones(3,4,5)

        # pre norm, resid
        self.config.pre_norm = True
        self.config.use_resid_connection = True
        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 13
        actual_tensor = slc(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        # pre norm, no resid
        self.config.pre_norm = True
        self.config.use_resid_connection = False
        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 12
        actual_tensor = slc(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        # post norm, resid
        self.config.pre_norm = False
        self.config.use_resid_connection = True
        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 40
        actual_tensor = slc(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

        # post norm, no resid
        self.config.pre_norm = False
        self.config.use_resid_connection = False
        slc = SublayerConnection(mock_sublayer_func, mock_sublayer, self.config)
        slc.norm = mock_norm

        correct_tensor = torch.ones(3,4,5) * 30
        actual_tensor = slc(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestLayer(unittest.TestCase):

    def setUp(self):
        self.config = get_config_arch()

    def testBadInput(self):
        x = torch.rand(100,10,512)
        y = torch.rand(100,20,512)
        ymask = torch.triu(torch.full((20,20), float('-inf')), diagonal=1)

        # encoder
        el = Layer(self.config, take_two_seqs=False, use_mask=False)
        with self.assertRaises(ValueError):
            out = el(y, mask=ymask)
        with self.assertRaises(ValueError):
            out = el(y, x)
        with self.assertRaises(ValueError):
            out = el(y, x, ymask)

        # decoder only
        dol = Layer(self.config, take_two_seqs=False, use_mask=True)
        with self.assertRaises(ValueError):
            out = dol(y)
        with self.assertRaises(ValueError):
            out = dol(y, x)
        with self.assertRaises(ValueError):
            out = dol(y, x, ymask)

        # ???
        weirdl = Layer(self.config, take_two_seqs=True, use_mask=False)
        with self.assertRaises(ValueError):
            out = weirdl(y)
        with self.assertRaises(ValueError):
            out = weirdl(y, mask=ymask)
        with self.assertRaises(ValueError):
            out = weirdl(y, x, ymask)

        # decoder
        dl = Layer(self.config, take_two_seqs=True, use_mask=True)
        with self.assertRaises(ValueError):
            out = dl(y)
        with self.assertRaises(ValueError):
            out = dl(y, mask=ymask)
        with self.assertRaises(ValueError):
            out = dl(y, x)

    def testShape(self):
        x = torch.rand(100,10,512)
        y = torch.rand(100,20,512)
        ymask = torch.triu(torch.full((20,20), float('-inf')), diagonal=1)

        # encoder
        el = Layer(self.config, take_two_seqs=False, use_mask=False)
        out = el(y)
        self.assertEqual(out.shape, (100,20,512))

        # decoder only
        dol = Layer(self.config, take_two_seqs=False, use_mask=True)
        out = dol(y, mask=ymask)
        self.assertEqual(out.shape, (100,20,512))

        # ???
        weirdl = Layer(self.config, take_two_seqs=True, use_mask=False)
        out = weirdl(y, x)
        self.assertEqual(out.shape, (100,20,512))

        # decoder
        dl = Layer(self.config, take_two_seqs=True, use_mask=True)
        out = dl(y, x, ymask)
        self.assertEqual(out.shape, (100,20,512))



class TestEncoderOrDecoder(unittest.TestCase):

    def setUp(self):
        self.config = get_config_arch()

    def testBadInput(self):
        x = torch.rand(50,10,512)
        y = torch.rand(50,20,512)

        # encoder
        e = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=False, use_mask=False)
        with self.assertRaises(ValueError):
            out = e(y, x)

        # decoder
        d = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, use_mask=True)
        with self.assertRaises(ValueError):
            out = d(y)

    def testShape(self):
        x = torch.rand(50,10,512)
        y = torch.rand(50,20,512)

        # encoder
        e = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=False, use_mask=False)
        out = e(y)
        self.assertEqual(out.shape, (50,20,512))

        # decoder only
        do = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=False, use_mask=True)
        out = do(y)
        self.assertEqual(out.shape, (50,20,512))

        # ???
        weird = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, use_mask=False)
        out = weird(y, x)
        self.assertEqual(out.shape, (50,20,512))

        # decoder
        d = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, use_mask=True)
        out = d(y, x)
        self.assertEqual(out.shape, (50,20,512))

        # prenorm
        self.config.pre_norm = True
        d_pre = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, use_mask=True)
        out = d_pre(y, x)
        self.assertEqual(out.shape, (50,20,512))

        # postnorm
        self.config.pre_norm = False
        d_post = EncoderOrDecoder(self.config, num_layers=6, take_two_seqs=True, use_mask=True)
        out = d_post(y, x)
        self.assertEqual(out.shape, (50,20,512))



class TestTransformer(unittest.TestCase):

    def setUp(self):
        self.config = get_config_arch()

    def testTwoSeqShape(self):
        x = torch.rand(5,10,1000)
        y = torch.rand(5,20,1000)

        t = TransformerTwoSeq(self.config, num_enc_layers=6, use_mask_enc=True, num_dec_layers=6, use_mask_dec=False, output_probs=False, vocab_size=1000, tgt_vocab_mask=None)
        out = t(x, y)
        self.assertEqual(out.shape, (5,20,512))

        t = TransformerTwoSeq(self.config, num_enc_layers=6, use_mask_enc=True, num_dec_layers=6, use_mask_dec=False, output_probs=True, vocab_size=1000, tgt_vocab_mask=None)
        out = t(x, y)
        self.assertEqual(out.shape, (5,20,1000))

    def testOneSeqShape(self):
        y = torch.rand(5,20,1000)

        t = TransformerOneSeq(self.config, num_layers=6, use_mask=True, output_probs=False, vocab_size=1000, vocab_mask=None)
        out = t(y)
        self.assertEqual(out.shape, (5,20,512))

        t = TransformerOneSeq(self.config, num_layers=6, use_mask=True, output_probs=True, vocab_size=1000, vocab_mask=None)
        out = t(y)
        self.assertEqual(out.shape, (5,20,1000))

    def testEncoderDecoderShape(self):
        x = torch.rand(5,10,1000)
        y = torch.rand(5,20,1000)

        self.config.transformer_type = TransformerType.ENCODER_DECODER
        t = get_transformer(self.config, vocab_size=1000, tgt_vocab_mask=None)
        out = t(x, y)
        self.assertEqual(out.shape, (5,20,1000))

    def testEncoderOnlyShape(self):
        x = torch.rand(5,10,1000)

        self.config.transformer_type = TransformerType.ENCODER_ONLY
        t = get_transformer(self.config, vocab_size=1000, tgt_vocab_mask=None)
        out = t(x)
        self.assertEqual(out.shape, (5,10,512))

    def testDecoderOnlyShape(self):
        y = torch.rand(5,20,1000)

        self.config.transformer_type = TransformerType.DECODER_ONLY
        t = get_transformer(self.config, vocab_size=1000, tgt_vocab_mask=None)
        out = t(y)
        self.assertEqual(out.shape, (5,20,1000))
