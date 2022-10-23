import numpy as np


def read_fasta_file(fasta_file):
    seq_dict = {}
    bag_sen = list()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line[0] == '>':
            name = line[1:]
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()

    for seq in seq_dict.values():
        seq = seq.replace('T', 'U')
        bag_sen.append(seq)

    return np.asarray(bag_sen)