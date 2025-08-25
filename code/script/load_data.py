import os
import numpy as np
from script.seq import retrieve_fasta

def load_seq(path, query):
	if os.path.isfile(path):
		f = open(path)
		f.readline()
		seq = "".join(list(map(lambda line: line.strip(), f.readlines())))
		f.close()
	else:
		seq = retrieve_fasta(query)
		fo = open(path, 'w')
		print (">%s\n%s" %(query, seq), file=fo)
		fo.close()
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
    #if query in ["P62593", "P00552"]:  # [BLAT, APH2]
    f = open(path)
    f.readline()
    for line in f.readlines():
        line = line.strip().split("\t")
        output_list.append(line[0])
        output_dic[line[0]] = float(line[1])
    f.close()

    return output_list, output_dic

def load_SCI(SCI_output_dir, mutation_list, query):
    SCI_dic = {}
    f = open(os.path.join(SCI_output_dir, '%s.sci' %query))
    f.readline()
    for line in f.readlines():
        line = line.strip().split("\t")
        SCI_dic[line[1]+line[0]+line[2]] = float(line[3])
    f.close()
    output_list = list(map(lambda mutation: SCI_dic.get(mutation, np.nan), mutation_list))
    return output_list

def load_stability(stability_output_file, overlapped_mutation):
    f = open(stability_output_file)
    f.readline()
    stability_list, stability_dic = [], {}
    for line in f.readlines():
        uni_var, pdb_var, stability_diff = line.strip().split("\t")
        stability_dic[uni_var] = float(stability_diff)
    for mut in overlapped_mutation:
        stability_list.append(stability_dic[mut])
    return stability_list
