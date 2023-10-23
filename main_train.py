# TODO(darcey): make it possible to resume training from a checkpoint (will need to save learning rate details etc. in addition to the checkpoint itself; probably best way to do this is to write out a custom config file representing the current state)

import os
import argparse
import torch

from configuration import read_config
from trainer import Trainer
from translator import Translator
from reader_writer import read_data, print_translations, compute_bleu
from vocabulary import Vocabulary
from dataset import Seq2SeqTranslateDataset, Seq2SeqTrainDataset
from generator import Generator
from outer_generator import OuterGenerator
from transformer import get_transformer

PRINT_INTERVAL = 100000

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
    parser.add_argument('--dev-tgt-gold', type=str, required=True,
                        help='File location for the dev data to be used during BLEU computations (may be different than dev-tgt due to tokenization)')

    parser.add_argument('--vocab', type=str, required=True,
                        help='File location to write the vocabulary to')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Directory to save model checkpoints to')
    parser.add_argument('--bleu-script', type=str, required=False,
                        help='Script that does postprocessing and computes BLEU score')

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
    dev_src_unk   = vocab.unk_data(dev_src, src=True)
    dev_tgt_unk   = vocab.unk_data(dev_tgt, src=False)

    # convert from toks to idxs
    train_src_idxs = vocab.tok_to_idx_data(train_src_unk)
    train_tgt_idxs = vocab.tok_to_idx_data(train_tgt_unk)
    dev_src_idxs   = vocab.tok_to_idx_data(dev_src_unk)
    dev_tgt_idxs   = vocab.tok_to_idx_data(dev_tgt_unk)

    # make the training data batches
    train_batches = Seq2SeqTrainDataset(src=train_src_idxs,
                                        tgt=train_tgt_idxs,
                                        toks_per_batch=config.train.batch_size,
                                        pad_idx=vocab.pad_idx(),
                                        bos_idx=vocab.bos_idx(),
                                        eos_idx=vocab.eos_idx(),
                                        sort_by_tgt_only=config.train.sort_by_tgt_only,
                                        randomize=True)
    dev_batches = Seq2SeqTrainDataset(src=dev_src_idxs,
                                      tgt=dev_tgt_idxs,
                                      toks_per_batch=config.train.batch_size,
                                      pad_idx=vocab.pad_idx(),
                                      bos_idx=vocab.bos_idx(),
                                      eos_idx=vocab.eos_idx(),
                                      sort_by_tgt_only=False,
                                      randomize=False)

    # determine the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make the model
    tgt_support_mask = vocab.get_tgt_support_mask()
    model = get_transformer(config, len(vocab), vocab.pad_idx(), tgt_support_mask)
    model.to(device)

    # make the directory for saving checkpoints + translations
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # computations pertaining to the batch size for dev BLEU computation
    max_parallel_sentences = config.gen.max_parallel_sentences
    num_beams_or_samples = config.gen.num_beams_or_samples
    if max_parallel_sentences < num_beams_or_samples:
        translate_batch_size = 1
    else:
        translate_batch_size = int(max_parallel_sentences / num_beams_or_samples)

    num_translations = len(dev_src) * num_beams_or_samples
    print_interval = min(num_translations, PRINT_INTERVAL)

    # make the translation data batches for dev BLEU computation
    dev_translate_batches = Seq2SeqTranslateDataset(dev_src_idxs, translate_batch_size, print_interval, vocab.pad_idx(), vocab.bos_idx(), vocab.eos_idx())

    # construct the function for computing BLEU scores
    generator = Generator(model, config, device, len(vocab), vocab.pad_idx(), vocab.bos_idx(), vocab.eos_idx())
    outer_generator = OuterGenerator(generator, vocab, config, device)
    translator = Translator(model, outer_generator)
    def translate_and_bleu_func(epoch_num, max_epochs):
        for dev_translated_final, dev_translated_all, probs_all in translator.translate(dev_translate_batches):
            dev_translated_filename = f"{os.path.basename(args.dev_tgt)}.{epoch_num:0{len(str(max_epochs))}d}"
            dev_translated_filepath = os.path.join(args.checkpoint_dir, dev_translated_filename)
            print_translations(dev_translated_filepath, dev_translated_final, dev_translated_all, probs_all)
        bleu = compute_bleu(args.bleu_script, dev_translated_filepath, args.dev_tgt_gold)
        return bleu

    # make the trainer
    trainer = Trainer(model, vocab, config, args.checkpoint_dir, device)

    # train
    trainer.train(train_batches, dev_batches, translate_and_bleu_func)
