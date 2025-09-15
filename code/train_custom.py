import os, time
import numpy as np
import pandas as pd

from script.load_data import load_seq, get_all_mutation, load_mutagenesis
from shannon_entropy import calc_ent_dic
from transformer_feature import calc_transformer_feature 
from train_and_eval import train_model
from script.msa_transformer.run_a_prot import calc_attention
from scipy.stats import zscore

#########################################################################################

def train_custom(test_enzyme,train_enzyme_list):
    """Run prediction performance test for a single query enzyme."""
    
    print(f"Name of test enzyme: [{test_enzyme}]")
    print(f"[{time.ctime()}] [Testing prediction performance.]")
    
    total_enzyme_dic={}
    total_enzyme_list=train_enzyme_list+[test_enzyme]

    for enzyme in total_enzyme_list:
        total_enzyme_dic[enzyme]={}
        total_enzyme_dic[enzyme]["Y"],total_enzyme_dic[enzyme]["X"], total_enzyme_dic[enzyme]["mutation_list"] = [], [], []
        data_dir = os.path.join('../data', enzyme)
        fasta_path = f"{data_dir}/{enzyme}.fasta"
        msa_path = f"{data_dir}/{enzyme}.aln"
        query_mutagenesis_path = f'{data_dir}/{enzyme}_score.txt' 

        # Load sequence & mutations
        seq = load_seq(fasta_path, enzyme)
        all_mutation = get_all_mutation(seq)
        mutagenesis_list, mutagenesis_dic = load_mutagenesis(query_mutagenesis_path,enzyme)
        overlapped_mutation = list(set(all_mutation).intersection(mutagenesis_list))
        
        # Generate features
        if os.path.isfile('../data/%s/input_feature.tsv'%enzyme):
            print(f"[{time.ctime()}] [Loading caculated input feature]")
            input_feature=pd.read_csv('../data/%s/input_feature.tsv'%enzyme,sep='\t')
        else:
            print(f"[{time.ctime()}] [Calculating pairwise mutation effect]")
            ent_dic = calc_ent_dic(data_dir, all_mutation, enzyme, msa_path)
            print(f"[{time.ctime()}] [Calculating all to all residue interaction]")
            attention, attention_trans = calc_attention(data_dir, enzyme)
            print(f"[{time.ctime()}] [Generating input feature for classifer]")
            input_feature = calc_transformer_feature(enzyme, data_dir, ent_dic, attention, attention_trans)

        for i, mut in enumerate(overlapped_mutation):
            total_enzyme_dic[enzyme]["Y"].append(mutagenesis_dic[mut])
            total_enzyme_dic[enzyme]["X"].append(input_feature[mut].tolist())
            total_enzyme_dic[enzyme]["mutation_list"].append(mut)

    # Prepare train input
    x_train, y_train = [], []
    for train_enzyme in train_enzyme_list:
        x_train.extend(zscore(total_enzyme_dic[train_enzyme]["X"]))
        y_train.extend(zscore(total_enzyme_dic[train_enzyme]["Y"]))
    x_train = np.array(x_train)
    y_train = np.array(y_train)


    # Prepare test input
    x_test = zscore(total_enzyme_dic[test_enzyme]["X"])
    y_test = zscore(total_enzyme_dic[test_enzyme]["Y"])
    
    print(f"[{time.ctime()}] [DeepSCANEER prediction by custom train set]")
    result=train_model(test_enzyme, x_train, y_train, x_test, y_test, weight_dir,version)

    # Ensemble across folds
    result['prediction_score'] = result[list(range(0, 10))].mean(axis=1)
    
    # Save per enzyme
    column_names = ["variant"] + [f"version{i}" for i in range(10)] + ["prediction_score"]
    test_enzyme_df = pd.DataFrame(
        zip(overlapped_mutation, *[result[i] for i in range(10)], result['prediction_score']),
        columns=column_names
    )   
    test_enzyme_df.to_csv(f"{result_dir}/{test_enzyme}_DeepSCANEER_prediction_{version}.tsv", sep="\t", index=False)
    return

#########################################################################################

# === Parameters ===
test_enzyme = 'P51580'
train_enzyme_list=['Q9NV35']
version='case1'

#########################################################################################

# === Path ===
data_dir = os.path.join('../data', test_enzyme)
os.makedirs(data_dir, exist_ok=True)

result_dir = os.path.join('../result', test_enzyme)
os.makedirs(result_dir, exist_ok=True)

weight_dir=f'../weight/{version}'
os.makedirs(weight_dir, exist_ok=True)

# Run
train_custom(test_enzyme,train_enzyme_list)