import os, math
import numpy as np
import pandas as pd
from script import msa, iscalc, blast


def calc_ent_dic(output_dir, fasta_path, subst_list, prefix, orthofasta_path):
    # MSA using muscle
    m_output = os.path.join(output_dir, "%s.aln" % prefix)
    m_tree = os.path.join(output_dir, "%s.dnd" % prefix)

    # BLAST search or using existed MSA
    b_orthof = os.path.join(orthofasta_path)
    
    pm = msa.ProcMsa(b_orthof, m_output, m_tree, method="muscle")
    pm.run()
    pm.parse()
    msa_dic = build_msa(os.path.join(output_dir, '%s.aln' % prefix))

    # Analyzing co-evolutionary network
    CN_dic, w_CN_dic, coenet_dic, seq_len = get_CN(output_dir, pm, prefix)

    # w_CN_dic
    ent_df = pd.DataFrame()
    AA_list = list("ARNDCQEGHILKMFPSTWYV")
    ent_df['AA'] = AA_list

    SCI_list, ent_dict = calc_SCI(output_dir, subst_list, prefix, msa_dic, CN_dic, w_CN_dic, coenet_dic, save=True)
    print(ent_dict)
    ent_key = ent_dict.keys()
    for residue in range(1, seq_len + 1):
        for AA2 in AA_list:
            res = str(residue) + AA2
            if res in ent_key:
                ent_df.loc[ent_df['AA'] == AA2, str(residue)] = ent_dict[res]
            else:
                ent_df.loc[ent_df['AA'] == AA2, str(residue)] = 0

    ent_df.to_csv('%s/ent_dic.txt' % (output_dir), sep='\t', index=False)
    return ent_df


def BLAST(fasta_path, output_dir, prefix):
    b_output = os.path.join(output_dir, '%s.blast.xml' % prefix)
    b_orthof = os.path.join(output_dir, '%s_ortho.fasta' % prefix)
    b_options = {
        'psi_itr': 3,
        'eval_coff': 0.001,
        'max_seq_id': 0.9,
        'seq_len_ratio': (0.7, 1.3),
        'max_target_seqs': 999999999
    }

    pb = blast.ProcBlast(fasta_path, b_output, b_options, db="uniref90_0817.fasta")
    pb.psiBlast()
    pb.parse()
    print("Num. of homologs: %d" % len(pb.result))

    if len(pb.result) < 10:
        b_options = {
            'psi_itr': 3,
            'eval_coff': 0.001,
            'max_seq_id': 0.9,
            'seq_len_ratio': (0.7, 1.3),
        }
        pb = blast.ProcBlast(fasta_path, b_output, b_options, db="uniref90_0817.fasta")
        pb.psiBlast()
        pb.parse()
        if len(pb.result) < 10:
            raise Exception("ERROR: Insufficient num. of homologues.")
    
    query_id = pb.makeFas(b_orthof, True)
    return query_id


def build_msa(msa_path):
    prefix = msa_path.split("/")[-1].split(".")[0]
    tmp_dic = {}
    with open(msa_path) as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 2:
                if prefix in line[0]:
                    line[0] = prefix
                if line[0] not in tmp_dic:
                    tmp_dic[line[0]] = ""
                if line[1] in tmp_dic[line[0]]:
                    continue
                tmp_dic[line[0]] += line[1]

    msa_dic = {}
    gene = prefix.split("_")[0]
    for i in range(len(tmp_dic[gene])):
        if tmp_dic[gene][i] != "-":
            for key in tmp_dic.keys():
                if key not in msa_dic:
                    msa_dic[key] = ""
                try:
                    msa_dic[key] += tmp_dic[key][i]
                except IndexError:
                    msa_dic[key] += "-"
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
    for i in range(int(len_seq * len_threshold)):
        if i >= len(sorted_key):
            break
        res1, res2 = sorted_key[i]
        CN_dic[int(res1)] += 1
        CN_dic[int(res2)] += 1
    values = list(CN_dic.values())
    step = 1.0 / (len(values) - 1)
    sorted_vals = sorted(values)
    per_dic = {v: i * step for i, v in enumerate(sorted_vals)}
    per_CN_dic = {res: per_dic[CN_dic[res]] for res in CN_dic}
    return CN_dic, per_CN_dic


def build_coupling_dic(sorted_key, len_seq, len_threshold):
    coupling_dic = {res: [] for res in range(1, len_seq + 1)}
    for i in range(int(len_seq * len_threshold)):
        if i >= len(sorted_key):
            break
        res1, res2 = sorted_key[i]
        coupling_dic[int(res1)].append(int(res2))
        coupling_dic[int(res2)].append(int(res1))
    return coupling_dic


def get_CN(base_pth, pm, prefix):
    coe_output_mcbasc = os.path.join(base_pth, "%s.coe_out_mcbasc" % prefix)
    pcn = iscalc.ProcCN(pm, coe_output_mcbasc, pm.result[0].id, cn_cutoff=2.0, coe_algorithm="McBASCCovariance")
    cn_result = pcn.calc()
    tmp_coe = build_coevolution_score_dic(coe_output_mcbasc)
    sorted_coe = sorted(tmp_coe, key=lambda x: tmp_coe[x], reverse=True)
    CN_dic, w_CN_dic = calc_CN(sorted_coe, pcn.query_len, 2.0)
    coenet_dic = build_coupling_dic(sorted_coe, pcn.query_len, 2.0)
    return CN_dic, w_CN_dic, coenet_dic, pcn.query_len


def calc_ent_diff(msa_dic, gene, residue, AA1, AA2):
    try:
        if msa_dic[gene][residue - 1] == AA1:
            residue_list = [x[residue - 1] for x in msa_dic.values()]
            residue_list = [x for x in residue_list if "-" not in x]
            n_AA1 = residue_list.count(AA1)
            n_AA2 = residue_list.count(AA2)
            return -math.log((n_AA2 + 1) / float(n_AA1))
        else:
            return np.nan
    except Exception:
        return np.nan


def calc_ent_diff_coupled(msa_dic, gene, coupling_dic, residue, AA1, AA2):
    try:
        if msa_dic[gene][residue - 1] == AA1:
            ent_diff_list = []
            if residue==1: print(coupling_dic)
            for residue2 in coupling_dic[residue]:
                residue_list = [x[residue - 1] + x[residue2 - 1] for x in msa_dic.values()]
                residue_list = [x for x in residue_list if "-" not in x]
                n_AA1 = residue_list.count(AA1 + msa_dic[gene][residue2 - 1])
                n_AA2 = residue_list.count(AA2 + msa_dic[gene][residue2 - 1])
                entropy_difference = -math.log((n_AA2 + 1) / float(n_AA1))
                ent_diff_list.append(entropy_difference)
            return np.mean(ent_diff_list)
        else:
            return np.nan
    except Exception:
        return np.nan


def calc_SCI(base_pth, subst_list, prefix, msa_dic, CN_dic, w_CN_dic, coenet_dic, save=False):
    SCI_list = []
    ent_dict = {}
    if save:
        with open(os.path.join(base_pth, "%s.sci" % prefix), 'w') as fo:
            print("\t".join(["res", "AA1", "AA2", "SCI", "cn", "cs"]), file=fo)
            for subst in subst_list:
                AA1, AA2, res = subst[0], subst[-1], int(subst[1:-1])
                avg_ent_diff = calc_ent_diff_coupled(msa_dic, prefix.split("_")[0], coenet_dic, res, AA1, AA2)
                cs = calc_ent_diff(msa_dic, prefix.split("_")[0], res, AA1, AA2)
                cn, w_cn = CN_dic.get(res, 0), w_CN_dic.get(res, 0)
                sci = w_cn * avg_ent_diff * (-1)
                print("\t".join(map(str, [res, AA1, AA2, sci, cn, cs])), file=fo)
                SCI_list.append(sci)
                ent_dict[subst[1:]] = avg_ent_diff
    return SCI_list, ent_dict
