import torch
import torch.testing
import unittest
import copy
from trainer import *
from vocabulary import *



class TestLoss(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary([],[])

    def testShape(self):
        trainer = Trainer(self.model, self.vocab)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.rand(2, 3, 4)
        actual    = torch.rand(2, 3, 4)
        loss      = trainer.loss(predicted, actual)
        self.assertEqual(loss.shape, ())

    # no label smoothing, no PAD, perfect predictions
    def testPerfectPrediction(self):
        trainer = Trainer(self.model, self.vocab)
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
        trainer = Trainer(self.model, self.vocab)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.rand(1, 2, 4)
        actual    = torch.tensor([[[0,1,0,0], [0,0,1,0]]])
        loss      = trainer.loss(predicted, actual)
        correct   = (predicted[0,0,1] + predicted[0,1,2]) / 2
        self.assertTrue(torch.equal(loss, correct))

    # no label smoothing, PAD
    def testNoLabelSmoothingPad(self):
        trainer = Trainer(self.model, self.vocab)
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
        trainer = Trainer(self.model, self.vocab)
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
        trainer = Trainer(self.model, self.vocab)
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
        trainer = Trainer(self.model, self.vocab)
        trainer.label_smoothing = 0.5
        trainer.label_smoothing_counts = torch.ones(4)/4.0

        predicted = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],
                                   [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
        actual    = torch.tensor([[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                  [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]])
        loss      = trainer.loss(predicted, actual)
        self.assertTrue(torch.equal(loss, torch.tensor((2.5 + 4)/2)))



class TestTrainOneStep(unittest.TestCase):

    def testParamsUpdate(self):
        vocab = Vocabulary([],[])
        l = len(vocab)

        class FakeModelClass(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = torch.nn.Linear(l,l)
                self.lin2 = torch.nn.Linear(l,l)
            def forward(self, in1, in2):
                return self.lin1(in1) + self.lin2(in2)

        model = FakeModelClass()
        model.label_smoothing = 0.0
        model_old = copy.deepcopy(model)
        inputs1 = torch.rand(2,5,l)
        inputs2 = torch.rand(2,5,l)
        targets = torch.rand(2,5,l)

        trainer = Trainer(model, vocab)
        trainer.train_one_step(inputs1, inputs2, targets)

        old_params = {name:param for name, param in model_old.named_parameters()}
        for name, param in model.named_parameters():
            self.assertFalse(torch.equal(param, old_params[name]))
