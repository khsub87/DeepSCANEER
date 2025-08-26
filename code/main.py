import os, time
import numpy as np
import pandas as pd

from script.load_data import load_seq, get_all_mutation, load_mutagenesis
from shannon_entropy import calc_ent_dic
from transformer_feature import calc_transformer_feature 
from trains_eval import predict_with_pretrained_model, predict_with_fine_tuning
from script.msa_transformer.run_a_prot import calc_attention

#########################################################################################

def main(test_enzyme):
    """Run prediction performance test for a single query enzyme."""
    
    print(f"Name of test enzyme: [{test_enzyme}]")
    print(f"[{time.ctime()}] [Testing prediction performance.]")
    
    # Load sequence & mutations
    seq = load_seq(fasta_path, test_enzyme)
    all_mutation = get_all_mutation(seq)
    
    # Generate features
    print(f"[{time.ctime()}] [Calculating pairwise mutation effect]")
    ent_dic = calc_ent_dic(data_dir, all_mutation, test_enzyme, msa_path)
    print(f"[{time.ctime()}] [Calculating all to all residue interaction]")
    attention, attention_trans = calc_attention(data_dir, test_enzyme)
    print(f"[{time.ctime()}] [Generating input feature for classifer]")
    input_file = calc_transformer_feature(test_enzyme, data_dir, ent_dic, attention, attention_trans)
    
    # Prepare test input
    x_test = [input_file[mut].tolist() for mut in all_mutation]
    
    # Fine-tuning if required
    if fine_tuning:
        print(f"[{time.ctime()}] [DeepSCANEER prediction by fine-tuning]")
        mutagenesis_list, mutagenesis_dic = load_mutagenesis(query_mutagenesis_path, test_enzyme)
        overlapped_mutation = list(set(all_mutation).intersection(mutagenesis_list))
        
        x_ft = [input_file[mut].tolist() for mut in overlapped_mutation]
        y_ft = [mutagenesis_dic[mut] for mut in overlapped_mutation]
        
        # Run fine-tuning prediction
        result = predict_with_fine_tuning(
            test_enzyme, pre_train_path,x_ft, y_ft, x_test,result_dir
        )
    else:
        # Run pretrained model prediction
        print(f"[{time.ctime()}] [DeepSCANEER prediction using pre-trained weight]")
        result = predict_with_pretrained_model(pre_train_path,x_test)
    
    # Ensemble across folds
    result['ensemble'] = result[list(range(0, 10))].mean(axis=1)
    
    # Save per enzyme
    column_names = ["mut"] + [f"fold{i}" for i in range(10)] + ["ensemble"]
    data = [all_mutation] + [result[i] for i in range(10)] + [result['ensemble']]
    test_enzyme_df = pd.DataFrame(data).T
    test_enzyme_df.columns = column_names
    test_enzyme_df.to_csv(f"{result_dir}/{test_enzyme}_DeepSCANEER_prediction_2.tsv", sep="\t", index=False)
    
    return

#########################################################################################

# === Parameters ===
test_enzyme = 'Q9NV35'
data_dir = os.path.join('../data', test_enzyme)
os.makedirs(data_dir, exist_ok=True)

fasta_path = f"{data_dir}/{test_enzyme}.fasta"
msa_path = f"{data_dir}/{test_enzyme}.aln"

result_dir = os.path.join('../result', test_enzyme)
os.makedirs(result_dir, exist_ok=True)

pre_train_path = f'../pre_train_weight'

fine_tuning = False
query_mutagenesis_path = f'{data_dir}/{test_enzyme}_score.txt' if fine_tuning else None

#########################################################################################

# Run
main(test_enzyme)