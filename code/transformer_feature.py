import os
import numpy as np
import pandas as pd

AA_LIST = list("ARNDCQEGHILKMFPSTWYV")

def calc_transformer_feature(protein, data_dir, ent_dic, attention, attention_trans):
    """
    Calculate combined transformer features and SCI-weighted input features for a protein.
    """
    
    # Load sequence from fasta
    fasta_path = os.path.join(f'../data/{protein}', f'{protein}.fasta')
    with open(fasta_path) as f:
        f.readline()
        sequence = "".join(line.strip() for line in f)

    # Load SCI data
    sci_result = pd.read_csv(os.path.join(data_dir, f'{protein}.sci'), sep='\t')
    sci_result['mut'] = sci_result['AA1'] + sci_result['res'].astype(str) + sci_result['AA2']

    # Combine transformer attention features
    transformer_result = (attention + attention_trans) / 2

    # Prepare entropy dictionary matrix
    wg_result_np = ent_dic.iloc[:, 1:].to_numpy().T
    wg_result_np = np.nan_to_num(wg_result_np, nan=0)

    # Normalize per-column
    wg_result_np_norm = (wg_result_np - np.mean(wg_result_np)) / np.std(wg_result_np)

    # Multiply transformer features with normalized entropy
    feature_pre = np.matmul(transformer_result, wg_result_np_norm)
    calc_matrix = np.reshape(feature_pre, (64, -1))

    # Build result DataFrame
    result = pd.DataFrame()
    index = 0
    length = calc_matrix.shape[1] // len(AA_LIST)

    data_dict={}
    for res in range(1, length + 1):
        for AA in AA_LIST:
            var = sequence[res - 1] + str(res) + AA
            sci_value = 0 if var[0] == var[-1] else sci_result['cn'][sci_result['mut'] == var].values
            data_dict[var] = np.append(calc_matrix[:, index], sci_value)
            index += 1
    result = pd.DataFrame(data_dict)

    # Save
    result.to_csv(os.path.join(data_dir, 'input_feature.tsv'), sep='\t', index=False)
    return result

