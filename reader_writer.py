from vocabulary import SpecialTokens

# Read in the data as a list of lists of tokens
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

# Print out the translations
def print_translations(filename, translations_final, translations_all, probs_all):
    final_filename = filename + ".final"
    all_filename = filename + ".all"

    with open(final_filename, "a") as f:
        for tgt_sent in translations_final:
            tgt_sent = [tok.name if isinstance(tok, SpecialTokens) else tok for tok in tgt_sent]
            f.write(" ".join(tgt_sent) + "\n")

    with open(all_filename, "a") as f:
        for tgt_sent, probs in zip(translations_all, probs_all):
            for gen, prob in zip(tgt_sent, probs):
                gen = [tok.name if isinstance(tok, SpecialTokens) else tok for tok in gen]
                f.write(" ".join(gen) + " ||| " + str(prob) + "\n")
            f.write("\n")


#def compute_bleu(out_filename, gold_filename):
    # run through de-bpe-er
    # run through detokenizer
    # call sacre bleu to compute bleu
