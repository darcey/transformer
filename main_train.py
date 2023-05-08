# TODO(darcey): figure out how to handle custom config files
# TODO(darcey): make the generator, pass it into the trainer

# TODO(darcey): consider moving tok-to-idx and unking into dataset creation
# TODO(darcey): make it possible to resume training from a checkpoint

import argparse
import torch

from configuration import *
from trainer import Trainer
from dataset import Seq2SeqTrainDataset
from vocabulary import Vocabulary
from transformer import get_transformer

def read_data(filename):
    with open(filename) as f:
        data = []
        for line in f:
            line = line.rstrip()
            if line == '':
                data.append([])
            else:
                data.append(line.split(' '))
    return data

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-src', type=str, required=True,
                        help='File location for src side training data')
    parser.add_argument('--train-tgt', type=str, required=True,
                        help='File location for tgt side training data')
    parser.add_argument('--dev-src', type=str, required=True,
                        help='File location for src side dev data')
    parser.add_argument('--dev-tgt', type=str, required=True,
                        help='File location for tgt side dev data')

    parser.add_argument('--vocab', type=str, required=True,
                        help='File location to write the vocabulary to')

    return parser

if __name__ == '__main__':
    # process args
    args = get_parser().parse_args()

    # read in data from file
    train_src = read_data(args.train_src)
    train_tgt = read_data(args.train_tgt)
    dev_src   = read_data(args.dev_src)
    dev_tgt   = read_data(args.dev_tgt)
    
    # read in configs from config file
    # TODO(darcey): figure out how to handle custom config files
    config_arch = get_config_arch()
    config_train = get_config_train()
    config_train.batch_size = 512
    config_train.epoch_size = 50
    config_train.max_epochs = 2000
    
    # make the vocabulary; use it to preprocess the data
    vocab = Vocabulary()
    vocab.initialize_from_data(train_src, train_tgt)
    vocab.write_to_file(args.vocab)

    train_src_unk = [vocab.unk_src(sent) for sent in train_src]
    train_tgt_unk = [vocab.unk_tgt(sent) for sent in train_tgt]
    dev_src_unk = [vocab.unk_src(sent) for sent in dev_src]
    dev_tgt_unk = [vocab.unk_tgt(sent) for sent in dev_tgt]

    train_src_idxs = [vocab.tok_to_idx(sent) for sent in train_src_unk]
    train_tgt_idxs = [vocab.tok_to_idx(sent) for sent in train_tgt_unk]
    dev_src_idxs = [vocab.tok_to_idx(sent) for sent in dev_src_unk]
    dev_tgt_idxs = [vocab.tok_to_idx(sent) for sent in dev_tgt_unk]
    
    # make the data batches
    train_batches = Seq2SeqTrainDataset(train_src_idxs, train_tgt_idxs, vocab, config_train.batch_size, randomize=True)
    dev_batches = Seq2SeqTrainDataset(dev_src_idxs, dev_tgt_idxs, vocab, config_train.batch_size, randomize=False)

    # make the model
    tgt_support_mask = vocab.get_tgt_support_mask()
    model = get_transformer(config_arch, config_train, len(vocab), tgt_support_mask)
    
    # TODO(darcey): if using dev BLEU, make the generator
    
    # make the trainer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, vocab, config_train, device)
    trainer.train(train_batches, dev_batches)
