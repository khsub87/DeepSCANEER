# DeepSCANEER
This repository provides the implementation and scripts to run DeepSCANEER.

**DeepSCANEER** is designed to predict changes in enzyme activity caused by mutations. 
In addition, DeepSCANEER incorporates enzyme-specific functions trained on low-throughput data, enabling ML-assisted directed evolution.

**DeepSCANEER** is an upgraded version of **SCANEER**, incorporating parts of the original SCANEER code. \
SCANEER github link: https://github.com/SBIlab/SCANEER

## Installation
Clone the repository and navigate into the project directory:

```
git clone [repository-url]
cd [repository-name]
```

## Input Data
Before running DeepSCANEER, make sure that all required input files are placed under the data/ directory:

+ **FASTA file ((enzyme_id).fasta)** - Query sequence(s) in FASTA format.
+ **Alignment file ((enzyme_id).aln)** - Multiple sequence alignment for the query sequence(s).
+ **low-throughput file ((enzyme_id)_score.txt)** (optional) - Required only for enzyme-specific predictions.

## Running DeepSCANEER
DeepSCANEER can be executed in one of two modes, depending on your analysis needs:

1. Zero-shot prediction - general-purpose prediction without fine-tuning
2. Enzyme-specific prediction - requires the corresponding low-throughput file

## Running:

You need to set the following parameters in main.py according to your situation.

```
#########################################################################################
# === Parameters ===
test_enzyme = '(enzyme_id)'
enzyme_specific_prediction = (True or False)
enzyme_specific_data_num= (num of low-throughput data)

#########################################################################################
```
Then, by running the following command in the terminal, a prediction_result file will be generated inside the result folder.
```
python main.py
```

## Notes
+ Zero-shot mode works directly without any additional fine-tuning.
+ Enzyme-specific mode uses the provided fine-tuning file (*_ft.txt) to adapt predictions to a specific enzyme.
