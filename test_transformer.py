# TODO(darcey): come up with better unit tests for all the classes
# TODO(darcey): write tests which confirm that stuff gets gradient updated
# TODO(darcey): do the unit tests need to be written to run on GPU?

import torch
import torch.testing
import unittest
from configuration import *
from transformer import *



class TestEmbedding(unittest.TestCase):
    
    def setUp(self):
        self.emb = Embedding(10,4)
        
    def testShape(self):
        x = torch.rand(100,40,10)
        out = self.emb(x)
        self.assertEqual(out.shape, (100,40,4))
    
    def testForwardZeros(self):
        self.emb.embedding = torch.nn.Parameter(torch.zeros(10,4))
        input_tensor = torch.tensor([[[1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                      [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]]])
        correct_tensor = torch.tensor([[[0.0,0.0,0.0,0.0],
                                        [0.0,0.0,0.0,0.0]]])
        actual_tensor = self.emb(input_tensor)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))

    def testReverseZeros(self):
        self.emb.embedding = torch.nn.Parameter(torch.zeros(10,4))
        input_tensor = torch.tensor([[[0.0,0.0,0.0,0.0],
                                      [0.0,0.0,0.0,0.0]]])
        correct_tensor = torch.tensor([[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]])
        actual_tensor = self.emb(input_tensor, reverse=True)
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



class TestFeedForward(unittest.TestCase):
    
    def testShape(self):
        ff = FeedForward(64, 256)
        x = torch.rand(100, 10, 64)
        out = ff(x)
        self.assertEqual(out.shape, (100, 10, 64))



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





class TestEncodersDecoders(unittest.TestCase):

    def setUp(self):
        self.config = get_config_arch()

    def testEncoderLayerShape(self):
        el = EncoderLayer(self.config)
        x = torch.rand(100,10,512)
        out = el(x)
        self.assertEqual(out.shape, (100,10,512))
    
    def testEncoderShape(self):
        e = Encoder(self.config)
        x = torch.rand(100,10,512)
        out = e(x)
        self.assertEqual(out.shape, (100,10,512))
    
    def testDecoderLayerShape(self):
        dl = DecoderLayer(self.config)
        x = torch.rand(100,10,512)
        y = torch.rand(100,20,512)
        mask = torch.triu(torch.full((20,20), float('-inf')), diagonal=1)
        out = dl(y, x, mask)
        self.assertEqual(out.shape, (100,20,512))
    
    def testDecoderShape(self):
        d = Decoder(self.config)
        x = torch.rand(100,10,512)
        y = torch.rand(100,20,512)
        out = d(y, x)
        self.assertEqual(out.shape, (100,20,512))

    def testDecoderOnlyLayerShape(self):
        dol = DecoderOnlyLayer(self.config)
        y = torch.rand(100,20,512)
        mask = torch.triu(torch.full((20,20), float('-inf')), diagonal=1)
        out = dol(y, mask)
        self.assertEqual(out.shape, (100,20,512))
    
    def testDecoderOnlyShape(self):
        do = DecoderOnly(self.config)
        y = torch.rand(100,20,512)
        out = do(y)
        self.assertEqual(out.shape, (100,20,512))

    def testTransformerEncoderDecoderShape(self):
        vocab_size = 1000
        tgt_mask = torch.tensor([True]*1000)
        ted = TransformerEncoderDecoder(self.config, vocab_size, tgt_mask)
        x = torch.rand(5,10,1000)
        y = torch.rand(5,20,1000)
        out = ted(x, y)
        self.assertEqual(out.shape, (5,20,1000))

    def testTransformerEncoderOnlyShape(self):
        vocab_size = 1000
        teo = TransformerEncoderOnly(self.config, vocab_size)
        x = torch.rand(5,10,1000)
        out = teo(x)
        self.assertEqual(out.shape, (5,10,512))

    def testTransformerDecoderOnlyShape(self):
        vocab_size = 1000
        tgt_mask = torch.tensor([True]*1000)
        tdo = TransformerDecoderOnly(self.config, vocab_size, tgt_mask)
        y = torch.rand(5,20,1000)
        out = tdo(y)
        self.assertEqual(out.shape, (5,20,1000))
