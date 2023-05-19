# TODO(darcey): dev BLEU computation
# TODO(darcey): check to make sure that, if num_beams_or_samples > max_parallel_sentences, then the generation method is compatible with being split up

import argparse
import torch

from configuration import read_config
from translator import Translator
from reader_writer import read_data, print_translations
from vocabulary import Vocabulary
from dataset import Seq2SeqTranslateDataset
from generator import Generator
from transformer import get_transformer

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

    return parser

if __name__ == '__main__':

    # process args; read in config file
    args = get_parser().parse_args()
    config = read_config(args.config)

    # read in vocab and data from file
    vocab = Vocabulary()
    vocab.read_from_file(args.vocab)
    src = read_data(args.src)    

    # computations pertaining to the batch size
    max_parallel_sentences = config.generation.max_parallel_sentences
    num_beams_or_samples = config.generation.num_beams_or_samples
    if max_parallel_sentences < num_beams_or_samples:
        batch_size = 1
    else:
        batch_size = int(max_parallel_sentences / num_beams_or_samples)

    num_translations = len(src) * num_beams_or_samples
    dump_translations = (num_translations > 20000)

    # prepare the data
    src_unk = vocab.unk_data(src, src=True)
    src_idxs = vocab.tok_to_idx_data(src_unk)
    batches = Seq2SeqTranslateDataset(src_idxs, vocab, batch_size, dump_translations)

    # load the model from file
    tgt_support_mask = vocab.get_tgt_support_mask()
    model = get_transformer(config, len(vocab), vocab.pad_idx(), tgt_support_mask)
    model.load_state_dict(torch.load(args.model))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # make generator and translator
    generator = Generator(model, config)
    translator = Translator(model, generator)

    # translate the data
    print_every = 20000 if dump_translations else 0
    for translations in translator.translate(batches, print_every):
        translations_all = batches.unbatch(translations)
        translations_all = vocab.idx_to_tok_data(translations_all)
        print_translations(translations_all)

    #if args.compute_bleu:
        # load gold
        # compute bleu
