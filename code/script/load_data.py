import os
import numpy as np

def load_seq(path, query):
    f = open(path)
    f.readline()
    seq = "".join(list(map(lambda line: line.strip(), f.readlines())))
    f.close()
    return seq

def get_all_mutation(seq):
    output_list = []
    AA_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i, AA1 in enumerate(seq):
        for AA2 in AA_list:
            if AA1 == AA2: continue
            output_list.append("%s%d%s" %(AA1, i+1, AA2))
    return output_list

def load_mutagenesis(path, query):
    output_list, output_dic = [], {}
    f = open(path)
    f.readline()
    for line in f.readlines():
        line = line.strip().split("\t")
        output_list.append(line[0])
        output_dic[line[0]] = float(line[1])
    f.close()

    return output_list, output_dic