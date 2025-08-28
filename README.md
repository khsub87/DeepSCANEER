# DeepSCANEER
This repository provides the implementation and scripts to run DeepSCANEER.

**DeepSCANEER** is designed to predict changes in enzyme activity caused by mutations. 
In addition, DeepSCANEER incorporates enzyme-specific functions trained on low-throughput data, enabling ML-assisted directed evolution.

**DeepSCANEER** is an upgraded version of **SCANEER**, incorporating parts of the original SCANEER code.

## Installation
Clone the repository and navigate into the project directory:

```
git clone [repository-url]
cd [repository-name]
```

## Input Data
Before running DeepSCANEER, make sure that all required input files are placed under the data/ directory:

+ **FASTA file (*.fasta)** - Query sequence(s) in FASTA format.
+ **Alignment file (*.aln)** - Multiple sequence alignment for the query sequence(s).
+ **Fine-tuning file (*_ft.txt)** (optional) - Required only for enzyme-specific predictions.

## Running DeepSCANEER
DeepSCANEER can be executed in one of two modes, depending on your analysis needs:

1. Zero-shot prediction - (general-purpose prediction without fine-tuning)
2. Enzyme-specific prediction - (requires the corresponding fine-tuning file)

## Running:
```
python main.py
```

## Notes
+ Zero-shot mode works directly without any additional fine-tuning.
+ Enzyme-specific mode uses the provided fine-tuning file (*_ft.txt) to adapt predictions to a specific enzyme.
