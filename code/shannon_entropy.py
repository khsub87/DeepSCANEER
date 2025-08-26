import os, math
import numpy as np
import pandas as pd
from script import msa, iscalc,coe

AA_LIST = list("ARNDCQEGHILKMFPSTWYV")

def calc_ent_dic(output_dir, subst_list, prefix, msa_path):
    #Loading MSA
    pm = msa.ProcMsa("tmp", msa_path, "tmp", "tmp")
    pm.parse()
    msa_dic = build_msa(msa_path)

    # analyze residue interaction 
    CN_dic, w_CN_dic, coenet_dic, seq_len = get_CN(output_dir, pm, prefix)

    # calculate entropy difference
    ent_dict = calc_SCI_and_get_ent_dict(output_dir, subst_list, prefix, msa_dic, CN_dic, w_CN_dic, coenet_dic, save=True)

    data = { 'AA': AA_LIST } 
    for residue in range(1, seq_len + 1):
        data[str(residue)] = [ent_dict.get(f"{residue}{AA}", 0) for AA in AA_LIST]

    ent_df = pd.DataFrame(data)
    ent_df.to_csv(os.path.join(output_dir, "ent_dic.txt"), sep='\t', index=False)
    return ent_df

def build_msa(msa_path):
    prefix = os.path.basename(msa_path).split(".")[0]
    tmp_dic = {}
    with open(msa_path) as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2:
                name, seq = line
                if prefix in name: 
                    name = prefix
                tmp_dic.setdefault(name, "")
                tmp_dic[name] += seq

    gene = prefix.split("_")[0]
    msa_dic = {k: "" for k in tmp_dic}
    for i, _ in enumerate(tmp_dic[gene]):
        if tmp_dic[gene][i] == "-":
            continue
        for k in tmp_dic:
            msa_dic[k] += tmp_dic[k][i]
    return msa_dic

def build_coevolution_score_dic(path):
    output_dic = {}
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.strip().split("\t")
            output_dic[(line[0], line[1])] = float(line[2])
    return output_dic


def calc_CN(sorted_key, len_seq, len_threshold):
    CN_dic = {res: 0 for res in range(1, len_seq + 1)}
    for res1, res2 in sorted_key[:int(len_seq * len_threshold)]:
        CN_dic[int(res1)] += 1
        CN_dic[int(res2)] += 1
    sorted_vals = sorted(CN_dic.values())
    step = 1.0 / (len(sorted_vals) - 1)
    per_dic = {v: i * step for i, v in enumerate(sorted_vals)}
    per_CN_dic = {res: per_dic[CN_dic[res]] for res in CN_dic}
    return CN_dic, per_CN_dic


def build_coupling_dic(sorted_key, len_seq, len_threshold):
    coupling_dic = {res: [] for res in range(1, len_seq + 1)}
    for res1, res2 in sorted_key[:int(len_seq * len_threshold)]:
        coupling_dic[int(res1)].append(int(res2))
        coupling_dic[int(res2)].append(int(res1))
    return coupling_dic

def get_CN(base_pth, pm, prefix):
    coe_output_mcbasc = os.path.join(base_pth, "%s.coe_out_mcbasc" % prefix)
    pcn = iscalc.ProcCN(pm, coe_output_mcbasc, pm.result[0].id, cn_cutoff=2.0, coe_algorithm="McBASCCovariance")
    pcn.calc()
    tmp_coe = build_coevolution_score_dic(coe_output_mcbasc)
    sorted_coe = sorted(tmp_coe, key=lambda x: tmp_coe[x], reverse=True)
    CN_dic, w_CN_dic = calc_CN(sorted_coe, pcn.query_len, 2.0)
    coenet_dic = build_coupling_dic(sorted_coe, pcn.query_len, 2.0)
    return CN_dic, w_CN_dic, coenet_dic, pcn.query_len


def calc_ent_diff_coupled(msa_dic, gene, coupling_dic, residue, AA1, AA2):
    if msa_dic[gene][residue - 1] != AA1:
        return np.nan
    ent_diff_list = []
    for residue2 in coupling_dic[residue]:
        residue_list = [
            x[residue - 1] + x[residue2 - 1]
            for x in msa_dic.values()
            if "-" not in x[residue - 1:residue2]
        ]
        n_AA1 = residue_list.count(AA1 + msa_dic[gene][residue2 - 1])
        n_AA2 = residue_list.count(AA2 + msa_dic[gene][residue2 - 1])
        ent_diff_list.append(-math.log((n_AA2 + 1) / float(n_AA1)))
    return np.mean(ent_diff_list) if ent_diff_list else np.nan

def calc_ent_diff(msa_dic, gene, residue, AA1, AA2):
    if msa_dic[gene][residue - 1] != AA1:
        print(f"Not accurate information; The AA in {gene} ({residue}) is {msa_dic[gene][residue-1]} not {AA1}")
        return np.nan

    residue_list = [seq[residue - 1] for seq in msa_dic.values() if seq[residue - 1] != "-"]
    n_AA1 = residue_list.count(AA1)
    n_AA2 = residue_list.count(AA2)

    return -math.log((n_AA2 + 1) / n_AA1) if n_AA1 else np.nan

def calc_SCI_and_get_ent_dict(base_pth, subst_list, prefix, msa_dic, CN_dic, w_CN_dic, coenet_dic, save=False):
    ent_dict={}
    if save:
        fo = open(os.path.join(base_pth, "%s.sci" %prefix), 'w')
        print ("\t".join(["res", "AA1", "AA2", "SCI", "cn", "cs"]), file=fo)
    for subst in subst_list:
        AA1, AA2, res = subst[0], subst[-1], int(subst[1:-1])
        avg_ent_diff = calc_ent_diff_coupled(msa_dic, prefix.split("_")[0], coenet_dic, res, AA1, AA2)
        cs=calc_ent_diff(msa_dic, prefix.split("_")[0], res, AA1, AA2)
        cn, w_cn = CN_dic.get(res,0), w_CN_dic.get(res,0)
        sci = w_cn*avg_ent_diff*(-1)
        if save:
            print ("\t".join(list(map(str, [res, AA1, AA2, sci, cn, cs]))), file=fo)
        ent_dict[subst[1:]]=avg_ent_diff
    return ent_dict


