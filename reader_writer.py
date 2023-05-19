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
def print_translations(translations_all, filename):
    with open(filename, "a") as f:
        for tgt_sent in translations_all:
            for gen in tgt_sent:
                f.write(gen.join(" ") + "\n")
            f.write("\n")

#def compute_bleu(out_filename, gold_filename):
    # run through de-bpe-er
    # run through detokenizer
    # call sacre bleu to compute bleu
