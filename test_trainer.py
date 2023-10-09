# TODO(darcey): make better mocks, and organize the mocks better

import copy
import torch
import torch.testing
import unittest
from tempfile import TemporaryDirectory
from configuration import *
from vocabulary import *
from dataset import Seq2SeqTrainBatch
from trainer import *



class MockModel(torch.nn.Module):

    def __init__(self, l):
        super().__init__()
        self.lin1 = torch.nn.Linear(l,l)
        self.lin2 = torch.nn.Linear(l,l)
        self.l    = l

    def forward(self, in1, in2):
        in1 = in1.unsqueeze(-1).expand(-1,-1,self.l).type(torch.float)
        in2 = in2.unsqueeze(-1).expand(-1,-1,self.l).type(torch.float)
        return self.lin1(in1) + self.lin2(in2)



class TestInit(unittest.TestCase):

    def setUp(self):
        fake_src = [["the", "dog", "walked", "to", "the", "park"]]
        fake_tgt = [["the", "ogday", "alkedway", "to", "the", "arkpay"]]
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data(fake_src, fake_tgt)
        self.model = MockModel(len(self.vocab))
        self.config = read_config("configuration.toml")
        self.checkpt_dir = TemporaryDirectory()
        self.device = "cpu"

    def tearDown(self):
        self.checkpt_dir.cleanup()

    def testLabelSmoothingMaskSubsetOfSupport(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        for i in range(len(self.vocab)):
            if not trainer.support_mask[i]:
                self.assertEqual(trainer.label_smoothing_counts[i], 0.0)

    def testLabelSmoothingCounts(self):
        self.config.train.label_smooth_eos = True
        self.config.train.label_smooth_unk = True
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        ls_counts = trainer.label_smoothing_counts
        self.assertEqual(ls_counts[self.vocab.tok_to_idx("the")], 1.0/7.0)
        self.assertEqual(ls_counts[self.vocab.tok_to_idx(SpecialTokens.EOS)], 1.0/7.0)
        self.assertEqual(ls_counts[self.vocab.tok_to_idx(SpecialTokens.UNK)], 1.0/7.0)

        self.config.train.label_smooth_eos = False
        self.config.train.label_smooth_unk = False
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        ls_counts = trainer.label_smoothing_counts
        self.assertEqual(ls_counts[self.vocab.tok_to_idx("the")], 1.0/5.0)
        self.assertEqual(ls_counts[self.vocab.tok_to_idx(SpecialTokens.EOS)], 0.0)
        self.assertEqual(ls_counts[self.vocab.tok_to_idx(SpecialTokens.UNK)], 0.0)



class TestTrainOneStep(unittest.TestCase):

    def testParamsUpdate(self):
        device = "cpu"
        checkpt_dir = TemporaryDirectory()
        config = read_config("configuration.toml")
        vocab = Vocabulary()
        vocab.initialize_from_data([],[])
        l = len(vocab)

        model = MockModel(l)
        model_old = copy.deepcopy(model)
        inputs1 = torch.randint(high=l, size=(2,5))
        inputs2 = torch.randint(high=l, size=(2,5))
        targets = torch.randint(high=l, size=(2,5))

        trainer = Trainer(model, vocab, config, checkpt_dir.name, device)
        trainer.train_one_step(inputs1, inputs2, targets)

        old_params = {name:param for name, param in model_old.named_parameters()}
        for name, param in model.named_parameters():
            self.assertFalse(torch.equal(param, old_params[name]))

        checkpt_dir.cleanup()


class TestPerplexity(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.checkpt_dir = TemporaryDirectory()
        self.config = read_config("configuration.toml")
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data([['4','5','6','7','8','9']],[['10','11','12','13','14','15']])

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

                self.batch1 = Seq2SeqTrainBatch(src_1, tgt_in_1, tgt_out_1, 0, 0)
                self.batch2 = Seq2SeqTrainBatch(src_2, tgt_in_2, tgt_out_2, 0, 0)

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

    def tearDown(self):
        self.checkpt_dir.cleanup()

    def testPerfect(self):
        class MockModelIdentity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.eye = torch.nn.Linear(16, 16, bias=False)
                self.eye.weight = torch.nn.Parameter(torch.eye(16))
            def forward(self, in1, in2):
                in2 = torch.nn.functional.one_hot(in2, 16).type(torch.float)
                return self.eye(in2)
        self.model = MockModelIdentity()
        self.trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)

        ppl = self.trainer.perplexity(self.data)
        self.assertEqual(ppl, 1.0)



class TestLearningRate(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data([],[])
        self.config = read_config("configuration.toml")
        self.checkpt_dir = TemporaryDirectory()
        self.device = "cpu"

    def tearDown(self):
        self.checkpt_dir.cleanup()

    def testWarmup(self):
        for strategy in [LearningRateStrategy.WARMUP_INV_SQRT_DECAY, LearningRateStrategy.WARMUP_VAL_DECAY]:
            self.config.train.lr.lr_strategy = strategy
            self.config.train.lr.lr_scale = 2
            self.config.train.lr.warmup_steps = 4
            self.config.arch.d_model = 256
            trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)

            # Sets init rate correctly?
            start_lr = trainer.get_initial_learning_rate()
            self.assertEqual(start_lr, 1/64)
            self.assertEqual(trainer.lr, 1/64)

            # Set num steps to during warmup
            trainer.num_steps = 1

            # During warmup, after each step, updates lr correctly?
            trainer.adjust_learning_rate_step()
            self.assertEqual(trainer.lr, 1/32)

            # During warmup, after each epoch, does not modify lr?
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)

            # During warmup, no early stopping
            trainer.num_steps = 1
            trainer.lr_config.stop_lr = 0.0005
            trainer.lr                = 0.00005
            self.assertFalse(trainer.early_stopping())
            
    def testNoWarmup(self):
        for strategy in [LearningRateStrategy.NO_WARMUP_VAL_DECAY]:
            self.config.train.lr.lr_strategy = strategy
            self.config.train.lr.start_lr = 0.1
            trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)

            # Sets init rate correctly?
            start_lr = trainer.get_initial_learning_rate()
            self.assertEqual(start_lr, 0.1)
            self.assertEqual(trainer.lr, 0.1)

    def testInvSqrtDecay(self):
        for strategy in [LearningRateStrategy.WARMUP_INV_SQRT_DECAY]:
            self.config.train.lr.lr_strategy = strategy
            self.config.train.lr.lr_scale = 2
            self.config.train.lr.warmup_steps = 4
            self.config.arch.d_model = 256
            trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
            
            # Set num steps to after warmup
            trainer.num_steps = 8

            # After warmup, after each step, updates lr correctly?
            trainer.adjust_learning_rate_step()
            self.assertEqual(trainer.lr, 1/24)

            # After warmup, after each epoch, does not modify lr?
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)

            # No early stopping
            trainer.num_steps = 8
            self.assertFalse(trainer.early_stopping())
            trainer.lr_config.stop_lr = 0.0005
            trainer.lr                = 0.00005
            self.assertFalse(trainer.early_stopping())

    def testValDecay(self):
        for strategy in [LearningRateStrategy.NO_WARMUP_VAL_DECAY, LearningRateStrategy.NO_WARMUP_VAL_DECAY]:
            self.config.train.lr.lr_strategy = strategy
            self.config.train.lr.lr_scale = 2
            self.config.train.lr.warmup_steps = 4
            self.config.arch.d_model = 256
            self.config.train.lr.patience = 3
            self.config.train.lr.lr_decay = 0.5
            self.config.train.lr.start_lr = 0.1
            trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)

            # Set num steps to after warmup in case of warmup
            trainer.num_steps = 8

            # After warmup, after each step, does not modify lr?
            lr_before = trainer.lr
            trainer.adjust_learning_rate_step()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)

            # After each epoch
            # Should do nothing if not enough epochs
            trainer.num_epochs = 3
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)
            # After enough epochs
            trainer.num_epochs = 5
            # Should do nothing if perplexity going down / BLEU going up
            trainer.eval_metric = EvalMetric.PPL
            trainer.dev_ppls = [0.5, 0.4, 0.3, 0.2, 0.1]
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)
            trainer.eval_metric = EvalMetric.BLEU
            trainer.dev_bleus = [0.1, 0.2, 0.3, 0.4, 0.5]
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)
            # Should do nothing if perplexity going up / BLEU going down, 
            # but not worst in patience
            trainer.eval_metric = EvalMetric.PPL
            trainer.dev_ppls = [0.3, 0.1, 0.5, 0.2, 0.4]
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)
            trainer.eval_metric = EvalMetric.BLEU
            trainer.dev_bleus = [0.3, 0.5, 0.1, 0.4, 0.2]
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before, lr_after)
            # Should decay if perplexity going up / BLEU going down,
            # compared to last patience ppls
            trainer.eval_metric = EvalMetric.PPL
            trainer.dev_ppls = [0.5, 0.1, 0.3, 0.2, 0.4]
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before * 0.5, lr_after)
            trainer.eval_metric = EvalMetric.BLEU
            trainer.dev_bleus = [0.1, 0.5, 0.3, 0.4, 0.2]
            lr_before = trainer.lr
            trainer.adjust_learning_rate_epoch()
            lr_after = trainer.lr
            self.assertEqual(lr_before * 0.5, lr_after)

            # Early stopping
            trainer.lr_config.stop_lr = 0.0005
            trainer.lr                = 0.005
            self.assertFalse(trainer.early_stopping())
            trainer.lr_config.stop_lr = 0.0005
            trainer.lr                = 0.00005
            self.assertTrue(trainer.early_stopping())



class TestWordDropout(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"
        self.checkpt_dir = TemporaryDirectory()
        self.config = read_config("configuration.toml")
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data([],[])
        l = len(self.vocab)
        self.model = torch.nn.Linear(l, l)
        self.trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)

    def tearDown(self):
        self.checkpt_dir.cleanup()

    def testWordDropoutNoUnk(self):
        input_tensor = torch.rand(20,10)
        actual_tensor = self.trainer.word_dropout(input_tensor, 0.0)
        self.assertTrue(torch.equal(actual_tensor, input_tensor))

    def testWordDropoutAllUnk(self):
        input_tensor = torch.rand(20,10)
        correct_tensor = torch.full((20,10), self.vocab.tok_to_idx(SpecialTokens.UNK))
        actual_tensor = self.trainer.word_dropout(input_tensor, 1.0)
        self.assertTrue(torch.equal(actual_tensor, correct_tensor))


# these tests assume the vocab has assigned PAD an idx of zero
# so tests which rely on this assumption confirm this first
class TestLossAndCrossEnt(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Linear(10,10)
        self.vocab = Vocabulary()
        self.vocab.initialize_from_data([],[])
        self.assertEqual(self.vocab.tok_to_idx(SpecialTokens.PAD), 0)
        self.config = read_config("configuration.toml")
        self.checkpt_dir = TemporaryDirectory()
        self.device = "cpu"

    def tearDown(self):
        self.checkpt_dir.cleanup()

    def testShape(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)

        predicted = torch.rand(2, 3, 4)
        gold      = torch.randint(high=4, size=(2, 3))
        c_e, n_t  = trainer.cross_ent(predicted, gold)
        loss      = trainer.loss(predicted, gold)
        self.assertEqual(c_e.shape, ())
        self.assertEqual(n_t.shape, ())
        self.assertEqual(loss.shape, ())

    def testCrossEntSmoothParamDoesSomething(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 0.5

        predicted = torch.rand(2, 3, 4)
        gold      = torch.randint(high=4, size=(2, 3))
        ce1, nt1  = trainer.cross_ent(predicted, gold, smooth=False)
        ce2, nt2  = trainer.cross_ent(predicted, gold, smooth=True)
        self.assertFalse(torch.equal(ce1, ce2))
        self.assertTrue(torch.equal(nt1, nt2))

    # no label smoothing, no PAD, perfect predictions
    def testPerfectPrediction(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0
        trainer.support_mask = torch.tensor([True]*4)

        # note: this isn't a valid probability distribution
        #       but this function can't handle negative infinities
        #       unless they are outside the support of the distribution
        #       (expressed with support_mask)
        nh = -100
        predicted = torch.tensor([[[nh, 0, nh, nh], [nh, nh, 0, nh], [nh, nh, nh, 0]],
                                  [[nh, nh, nh, 0], [nh, nh, 0, nh], [nh, 0, nh, nh]]])
        gold      = torch.tensor([[1, 2, 3],
                                  [3, 2, 1]])
        loss      = trainer.loss(predicted, gold)
        self.assertTrue(torch.equal(loss, torch.tensor(0.0)))

    # no label smoothing, no PAD, random predictions
    def testNoLabelSmoothing(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0
        trainer.support_mask = torch.tensor([True]*4)

        predicted = torch.rand(1, 2, 4)
        gold      = torch.tensor([[1,2]])
        loss      = trainer.loss(predicted, gold)
        correct   = (- predicted[0,0,1] - predicted[0,1,2]) / 2
        self.assertTrue(torch.equal(loss, correct))

    # no label smoothing, PAD
    def testNoLabelSmoothingPad(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 0
        trainer.label_smoothing_counts = torch.ones(4)/4.0
        trainer.support_mask = torch.tensor([True]*4)

        predicted = torch.tensor([[[-1, -3, -3, -3], [-30, -30, -10, -30], [-300, -300, -300, -100]],
                                  [[-3000, -3000, -3000, -1000], [-10000, -30000, -30000, -30000], [-300000, -100000, -300000, -300000]]])
        gold      = torch.tensor([[0, 2, 3],
                                  [3, 0, 1]])
        loss      = trainer.loss(predicted, gold)
        self.assertTrue(torch.equal(loss, torch.tensor(101110.0/4.0)))

    # maximum label smoothing, no PAD, no label smoothing mask
    def testMaxLabelSmoothing(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 1
        trainer.label_smoothing_counts = torch.ones(4)/4.0
        trainer.support_mask = torch.tensor([True]*4)

        predicted = torch.tensor([[[-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4],
                                   [-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4]]])
        gold      = torch.tensor([[1, 2, 3],
                                  [3, 2, 1]])
        loss      = trainer.loss(predicted, gold)
        self.assertTrue(torch.equal(loss, torch.tensor(2.5)))

    # normal loss and label smoothing, no PAD, no mask
    def testInterpolation(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 0.5
        trainer.label_smoothing_counts = torch.ones(4)/4.0
        trainer.support_mask = torch.tensor([True]*4)

        predicted = torch.tensor([[[-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4],
                                   [-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4]]])
        gold      = torch.tensor([[3, 3, 3],
                                  [3, 3, 3]])
        loss      = trainer.loss(predicted, gold)
        self.assertTrue(torch.equal(loss, torch.tensor((2.5 + 4)/2)))

    # tests label smoothing mask (max label smoothing, no PAD)
    def testLabelSmoothingMask(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 1
        trainer.label_smoothing_counts = torch.tensor([0.0, 0.0, 0.5, 0.5])
        trainer.support_mask = torch.tensor([True]*4)

        predicted = torch.tensor([[[-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4],
                                   [-1, -2, -3, -4], [-1, -2, -3, -4], [-1, -2, -3, -4]]])
        gold      = torch.tensor([[1, 2, 3],
                                  [3, 2, 1]])
        loss      = trainer.loss(predicted, gold)
        self.assertTrue(torch.equal(loss, torch.tensor(3.5)))

    # tests support mask (max label smoothing, no PAD)
    def testSupportMask(self):
        trainer = Trainer(self.model, self.vocab, self.config, self.checkpt_dir.name, self.device)
        trainer.label_smoothing = 0.5
        trainer.support_mask = torch.tensor([False, True, True, True])
        trainer.label_smoothing_counts = torch.tensor([0.0, 0.0, 0.0, 1.0])

        ni        = float("-inf")
        predicted = torch.tensor([[[ni, -1, -3, -3], [ni, -30, -10, -30], [ni, -300, -100, -300]],
                                  [[ni, -3000, -1000, -3000], [ni, -10000, -30000, -30000], [ni, -300000, -100000, -300000]]])
        gold      = torch.tensor([[1, 2, 2],
                                  [2, 1, 2]])
        loss      = trainer.loss(predicted, gold)
        self.assertTrue(torch.equal(loss, torch.tensor(222222.0/6.0)))
