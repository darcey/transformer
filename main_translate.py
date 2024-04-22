# TODO(darcey): check to make sure that, if num_beams_or_samples > max_parallel_sentences, then the generation method is compatible with being split up

import argparse
import torch

from configuration import read_config, max_num_beams_or_samples
from reader_writer import read_data, print_translations, compute_bleu
from vocabulary import Vocabulary
from dataset import Seq2SeqTranslateDataset
from translator import Translator
from generator import Generator
from outer_generator import OuterGenerator
from transformer import get_transformer

PRINT_INTERVAL = 100000

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file location (toml format)')
    parser.add_argument('--vocab', type=str, required=True,
                        help='File location for model vocabulary')
    parser.add_argument('--model', type=str, required=True,
                        help='File location for model weights')

    parser.add_argument('--src', type=str, required=True,
                        help='File location for src data to be translated')
    parser.add_argument('--tgt', type=str, required=True,
                        help='File path to print completed translations')
    parser.add_argument('--gold', type=str, required=False,
                        help='Optional gold truth data for computing BLEU')
    parser.add_argument('--compute-bleu', action='store_true',
                        help='If this argument is used, will compute BLEU score of translations')
    parser.add_argument('--bleu-script', type=str, required=False,
                        help='Script that does postprocessing and computes BLEU score')
    parser.add_argument('--mbr-postproc-script', type=str, required=False,
                        help='Script that postprocesses the translations (e.g. to detokenize them) before doing MBR.')

    return parser

if __name__ == '__main__':

    # process args; read in config file
    args = get_parser().parse_args()
    config = read_config(args.config)

    # read in vocab and data from file
    vocab = Vocabulary()
    vocab.read_from_file(args.vocab)
    PAD = vocab.pad_idx()
    BOS = vocab.bos_idx()
    EOS = vocab.eos_idx()
    src = read_data(args.src)

    # computations pertaining to the batch size
    max_parallel_sentences = config.gen.max_parallel_sentences
    num_beams_or_samples = max_num_beams_or_samples(config.gen)
    if max_parallel_sentences < num_beams_or_samples:
        batch_size = 1
    else:
        batch_size = int(max_parallel_sentences / num_beams_or_samples)

    num_translations = len(src) * num_beams_or_samples
    print_interval = min(num_translations, PRINT_INTERVAL)

    # prepare the data
    src_unk = vocab.unk_data(src, src=True)
    src_idxs = vocab.tok_to_idx_data(src_unk)
    src_batches = Seq2SeqTranslateDataset(src_idxs, batch_size, print_interval, PAD, BOS, EOS)

    # load the model from file
    tgt_support_mask = vocab.get_tgt_support_mask()
    model = get_transformer(config, len(vocab), PAD, tgt_support_mask)
    model.load_state_dict(torch.load(args.model))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # make generator and translator
    generator = Generator(model, config, device, len(vocab), PAD, BOS, EOS)
    outer_generator = OuterGenerator(generator, vocab, config, device, args.mbr_postproc_script)
    translator = Translator(model, outer_generator)

    # translate the data
    for tgt_final, tgt_all, probs_all in translator.translate(src_batches):
        print_translations(args.tgt, tgt_final, tgt_all, probs_all)

    if args.compute_bleu:
        bleu = compute_bleu(args.bleu_script, args.tgt, args.gold)
        print(f"BLEU score:\t{bleu}")
