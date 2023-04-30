import torch
import torch.testing
import unittest
import copy
from configuration import *
from vocabulary import *
from trainer import *



class MockModel(torch.nn.Module):

    def __init__(self, l):
        super().__init__()
        self.lin1 = torch.nn.Linear(l,l)
        self.lin2 = torch.nn.Linear(l,l)

    def forward(self, in1, in2):
        return self.lin1(in1) + self.lin2(in2)



class TestInit(unittest.TestCase):

    def testLabelSmoothingMask(self):
        fake_src = [["the", "dog", "walked", "to", "the", "park"]]
        fake_tgt = [["the", "ogday", "alkedway", "to", "the", "arkpay"]]
        vocab = Vocabulary(fake_src, fake_tgt)
        model = MockModel(len(vocab))
        config = get_config_train()

        config.label_smooth_eos = True
        config.label_smooth_unk = True
        trainer = Trainer(model, vocab, config)
        ls_counts = trainer.label_smoothing_counts
        self.assertEqual(ls_counts[vocab.tok_to_idx("the")], 1.0/7.0)
        self.assertEqual(ls_counts[vocab.tok_to_idx(SpecialTokens.EOS)], 1.0/7.0)
        self.assertEqual(ls_counts[vocab.tok_to_idx(SpecialTokens.UNK)], 1.0/7.0)

        config.label_smooth_eos = False
        config.label_smooth_unk = False
        trainer = Trainer(model, vocab, config)
        ls_counts = trainer.label_smoothing_counts
        self.assertEqual(ls_counts[vocab.tok_to_idx("the")], 1.0/5.0)
        self.assertEqual(ls_counts[vocab.tok_to_idx(SpecialTokens.EOS)], 0.0)
        self.assertEqual(ls_counts[vocab.tok_to_idx(SpecialTokens.UNK)], 0.0)



class TestTrainOneStep(unittest.TestCase):

    def testParamsUpdate(self):
        config = get_config_train()
        vocab = Vocabulary([],[])
        l = len(vocab)

        model = MockModel(l)
        model_old = copy.deepcopy(model)
        inputs1 = torch.rand(2,5,l)
        inputs2 = torch.rand(2,5,l)
        targets = torch.rand(2,5,l)

        trainer = Trainer(model, vocab, config)
        trainer.train_one_step(inputs1, inputs2, targets)

        old_params = {name:param for name, param in model_old.named_parameters()}
        for name, param in model.named_parameters():
            self.assertFalse(torch.equal(param, old_params[name]))



class TestPerplexity(unittest.TestCase):

    def setUp(self):
        self.config = get_config_train()
        self.vocab = Vocabulary([['4','5','6','7','8','9']],[['10','11','12','13','14','15']])

        class MockData:
            def __init__(self, vocab):
                PAD = vocab.tok_to_idx(SpecialTokens.PAD)
                BOS = vocab.tok_to_idx(SpecialTokens.BOS)
                EOS = vocab.tok_to_idx(SpecialTokens.EOS)
                src_1 = torch.tensor([[5,6,EOS,PAD,PAD,PAD],[5,6,7,8,9,EOS]])
                src_2 = torch.tensor([[5,6,8,9,EOS],[5,6,7,8,EOS]])
                tgt_in_1 = torch.tensor([[BOS,10,11,12,PAD],[BOS,10,11,12,13]])
                tgt_in_2 = torch.tensor([[BOS,10,11,12,13,PAD],[BOS,10,11,12,13,14]])
                tgt_out_1 = torch.tensor([[10,11,12,EOS,PAD],[10,11,12,13,EOS]])
                tgt_out_2 = torch.tensor([[10,11,12,13,EOS,PAD],[10,11,12,13,14,EOS]])

                self.batch1 = {
                    "src": src_1,
                    "tgt_in": tgt_in_1,
                    "tgt_out": tgt_out_1,
                }
                self.batch2 = {
                    "src": src_2,
                    "tgt_in": tgt_in_2,
                    "tgt_out": tgt_out_2,
                }

                self.use_batch_1 = True
            def __len__(self):
                return 2
            def get_batch(self):
                if self.use_batch_1:
                    return self.batch1
                else:
                    return self.batch2
                self.use_batch_1 = not self.use_batch_1
        self.data = MockData(self.vocab)

    def testPerfect(self):
        class MockModelIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.eye = torch.nn.Linear(16, 16, bias=False)
                self.eye.weight = torch.nn.Parameter(torch.eye(16))
            def forward(self, in1, in2):
                return self.eye(in2)
        self.model = MockModelIdentity()
        self.trainer = Trainer(self.model, self.vocab, self.config)

        ppl = self.trainer.perplexity(self.data)
        self.assertEqual(ppl, 1.0)



class TestPrepBatch(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary([['4','5','6','7','8','9']],[['10','11','12','13','14','15']])
        self.config = get_config_train()
        self.trainer = Trainer(self.model, self.vocab, self.config)

        PAD = self.vocab.tok_to_idx(SpecialTokens.PAD)
        BOS = self.vocab.tok_to_idx(SpecialTokens.BOS)
        EOS = self.vocab.tok_to_idx(SpecialTokens.EOS)
        src = torch.tensor([[5,6,7,8,EOS,PAD],
                            [5,6,7,8,9,EOS]])
        tgt_in = torch.tensor([[BOS,10,11,12,13,14,15],
                               [BOS,10,11,12,13,PAD,PAD]])
        tgt_out = torch.tensor([[10,11,12,13,14,15,EOS],
                                [10,11,12,13,EOS,PAD,PAD]])
        self.batch = {
            "src": src,
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "num_src_toks": 11,
            "num_tgt_toks": 12,
        }

    def testShape(self):
        src, tgt_in, tgt_out = self.trainer.prep_batch(self.batch, do_dropout=False)
        self.assertEqual(src.shape, (2,6,16))
        self.assertEqual(tgt_in.shape, (2,7,16))
        self.assertEqual(tgt_out.shape, (2,7,16))



class TestWordDropout(unittest.TestCase):

    def setUp(self):
        self.config = get_config_train()
        self.vocab = Vocabulary([],[])
        l = len(self.vocab)
        self.model = torch.nn.Linear(l, l)
        self.trainer = Trainer(self.model, self.vocab, self.config)

    def testWordDropoutNoUnk(self):
        input_tensor = torch.rand(20,10)
        actual_tensor = self.trainer.word_dropout(input_tensor, 0.0)
        self.assertTrue(torch.equal(actual_tensor, input_tensor))

    def testWordDropoutAllUnk(self):
        input_tensor = torch.rand(20,10)
        correct_tensor = torch.full((20,10), self.vocab.tok_to_idx(SpecialTokens.UNK))
        actual_tensor = self.trainer.word_dropout(input_tensor, 1.0)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))



class TestLossAndCrossEnt(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary([],[])
        self.config = get_config_train()

    def testShape(self):
        trainer = Trainer(self.model, self.vocab, self.config)

        predicted = torch.rand(2, 3, 4)
        actual    = torch.rand(2, 3, 4)
        c_e, n_t  = trainer.cross_ent(predicted, actual)
        loss      = trainer.loss(predicted, actual)
        self.assertEqual(c_e.shape, ())
        self.assertEqual(n_t.shape, ())
        self.assertEqual(loss.shape, ())

    def testCrossEntSmoothParamDoesSomething(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0.5

        predicted = torch.rand(2, 3, 4)
        actual    = torch.rand(2, 3, 4)
        ce1, nt1  = trainer.cross_ent(predicted, actual, smooth=False)
        ce2, nt2  = trainer.cross_ent(predicted, actual, smooth=True)
        self.assertFalse(torch.equal(ce1, ce2))
        self.assertTrue(torch.equal(nt1, nt2))

    # no label smoothing, no PAD, perfect predictions
    def testLossPerfectPrediction(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        # note: this isn't a valid probability distribution
        #       but when I tried using nh = float("-inf") I got a nan
        nh = -100
        predicted = torch.tensor([[[nh, 0, nh, nh], [nh, nh, 0, nh], [nh, nh, nh, 0]],
                                  [[nh, nh, nh, 0], [nh, nh, 0, nh], [nh, 0, nh, nh]]])
        actual    = torch.tensor([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor(0.0)))

    # no label smoothing, no PAD, random predictions
    def testLossNoLabelSmoothing(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.rand(1, 2, 4)
        actual    = torch.tensor([[[0,1,0,0], [0,0,1,0]]])
        loss      = trainer.loss(predicted, actual)
        correct   = (predicted[0,0,1] + predicted[0,1,2]) / 2
        self.assertTrue(torch.equal(loss, correct))

    # no label smoothing, PAD
    def testLossNoLabelSmoothingPad(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.tensor([[[1, 3, 3, 3], [30, 30, 10, 30], [300, 300, 300, 100]],
                                  [[3000, 3000, 3000, 1000], [10000, 30000, 30000, 30000], [300000, 100000, 300000, 300000]]])
        actual    = torch.tensor([[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor(101110.0/4.0)))

    # maximum label smoothing, no PAD, no label smoothing mask
    def testLossMaxLabelSmoothing(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 1
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
        actual    = torch.tensor([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor(2.5)))

    # maximum label smoothing, no PAD, label smoothing mask
    def testLossMaxLabelSmoothingMask(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 1
        trainer.label_smoothing_counts = torch.tensor([0.0, 0.0, 0.5, 0.5])

        predicted = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
        actual    = torch.tensor([[[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor(3.5)))
    
    # normal loss and label smoothing, no PAD, no mask
    def testLossInterpolation(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0.5
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
        actual    = torch.tensor([[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor((2.5 + 4)/2)))
