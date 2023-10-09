import subprocess
from vocabulary import SpecialTokens

# Read in the data as a list of lists of tokens
def read_data(filepath):
    with open(filepath) as f:
        data = []
        for line in f:
            line = line.rstrip()
            if line == '':
                data.append([])
            else:
                data.append(line.split(' '))
    return data

# Print out the translations
def print_translations(filepath, translations_final, translations_all, probs_all):
    final_filepath = filepath + ".final"
    all_filepath = filepath + ".all"

    with open(final_filepath, "a") as f:
        for tgt_sent in translations_final:
            tgt_sent = [tok.name if isinstance(tok, SpecialTokens) else tok for tok in tgt_sent]
            f.write(" ".join(tgt_sent) + "\n")

    with open(all_filepath, "a") as f:
        for tgt_sent, probs in zip(translations_all, probs_all):
            for gen, prob in zip(tgt_sent, probs):
                gen = [tok.name if isinstance(tok, SpecialTokens) else tok for tok in gen]
                f.write(" ".join(gen) + " ||| " + str(prob) + "\n")
            f.write("\n")

# Call the externally supplied BLEU script that does
# the appropriate postprocessing (such as de-BPE-ing,
# detokenization) and then computes BLEU.
def compute_bleu(bleu_script, out_filepath, gold_filepath):
    out_final_filepath = out_filepath + ".final"
    bleu_output = subprocess.run([f"{bleu_script} {out_final_filepath} {gold_filepath}"], shell=True, capture_output=True, text=True)
    bleu_score = float(bleu_output.stdout)
    return bleu_score
