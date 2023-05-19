# TODO(darcey): make the generator and translator; pass them into the trainer
# TODO(darcey): make it possible to resume training from a checkpoint (will need to save learning rate details etc. in addition to the checkpoint itself; probably best way to do this is to write out a custom config file representing the current state)

import os
import argparse
import torch

from configuration import read_config
from trainer import Trainer
from reader_writer import read_data, print_translations
from vocabulary import Vocabulary
from dataset import Seq2SeqTranslateDataset, Seq2SeqTrainDataset
#from generator import Generator
from transformer import get_transformer

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file location (toml format)')

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
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory to save model checkpoints to')

    return parser

if __name__ == '__main__':
    # process args
    args = get_parser().parse_args()

    # read in configs from config file
    config = read_config(args.config)

    # read in data from file
    train_src = read_data(args.train_src)
    train_tgt = read_data(args.train_tgt)
    dev_src   = read_data(args.dev_src)
    dev_tgt   = read_data(args.dev_tgt)
    
    # make the vocabulary
    vocab = Vocabulary()
    vocab.initialize_from_data(train_src, train_tgt)
    vocab.write_to_file(args.vocab)

    # do unking
    train_src_unk = vocab.unk_data(train_src, src=True)
    train_tgt_unk = vocab.unk_data(train_tgt, src=False)
    dev_src_unk = vocab.unk_data(dev_src, src=True)
    dev_tgt_unk = vocab.unk_data(dev_tgt, src=False)

    # convert from toks to idxs
    train_src_idxs = vocab.tok_to_idx_data(train_src_unk)
    train_tgt_idxs = vocab.tok_to_idx_data(train_tgt_unk)
    dev_src_idxs = vocab.tok_to_idx_data(dev_src_unk)
    dev_tgt_idxs = vocab.tok_to_idx_data(dev_tgt_unk)
    
    # make the data batches
    train_batches = Seq2SeqTrainDataset(train_src_idxs, train_tgt_idxs, vocab, config.train.batch_size, sort_by_tgt_only=config.train.sort_by_tgt_only, randomize=True)
    dev_batches = Seq2SeqTrainDataset(dev_src_idxs, dev_tgt_idxs, vocab, config.train.batch_size, sort_by_tgt_only=False, randomize=False)
    #if config.train.compute_bleu:
    #    dev_translate_batches = Seq2SeqTranslateDataset(dev_src_idxs, vocab, config.generation.max_parallel_sentences, ....)

    # determine the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make the model
    tgt_support_mask = vocab.get_tgt_support_mask()
    model = get_transformer(config, len(vocab), vocab.pad_idx(), tgt_support_mask)
    model.to(device)
    
    # if using dev BLEU, make the generator and translator
    #if config.train.compute_bleu:
    #    generator = Generator(model, ...)
    #    translator = Translator(...)
    #    def print_func(data):
    #        return print_translations(vocab.idx_to_tok_data(dev_translate_batches.restore_original_order(data)))
    
    # make the trainer
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    #if config.train.compute_bleu:
    #    trainer = Trainer(model, vocab, config, args.checkpoint_dir, device, translator, print_func)
    # else
    trainer = Trainer(model, vocab, config, args.checkpoint_dir, device)

    # train
    trainer.train(train_batches, dev_batches)
