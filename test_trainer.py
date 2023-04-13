import torch
import torch.testing
import unittest
import copy
from configuration import *
from vocabulary import *
from trainer import *



class TestLoss(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary([],[])
        self.config = get_config_train()

    def testShape(self):
        trainer = Trainer(self.model, self.vocab, self.config)

        predicted = torch.rand(2, 3, 4)
        actual    = torch.rand(2, 3, 4)
        loss      = trainer.loss(predicted, actual)
        self.assertEqual(loss.shape, ())

    # no label smoothing, no PAD, perfect predictions
    def testPerfectPrediction(self):
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
    def testNoLabelSmoothing(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.rand(1, 2, 4)
        actual    = torch.tensor([[[0,1,0,0], [0,0,1,0]]])
        loss      = trainer.loss(predicted, actual)
        correct   = (predicted[0,0,1] + predicted[0,1,2]) / 2
        self.assertTrue(torch.equal(loss, correct))

    # no label smoothing, PAD
    def testNoLabelSmoothingPad(self):
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
    def testMaxLabelSmoothing(self):
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
    def testMaxLabelSmoothingMask(self):
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
    def testInterpolation(self):
        trainer = Trainer(self.model, self.vocab, self.config)
        trainer.label_smoothing = 0.5
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
        actual    = torch.tensor([[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor((2.5 + 4)/2)))



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
