# Annotation-Assisted Learning of Treatment Policies From Multimodal Electronic Health Records
This repository contains the code accompanying the paper: `Annotation-Assisted Learning of Treatment Policies From Multimodal Electronic Health Records`.

The goal is to learn treatment policies from multimodal electronic health records (EHRs) that combine tabular variables and clinical text. The repository supports three practical strategies for treatment policy learning:

1. **Risk-based modeling**, which prioritizes patients by predicted baseline risk.
2. **Representation-based causal modeling**, which applies causal treatment effect estimators directly to multimodal representations.
3. **Annotation-assisted causal modeling (\textsc{AACE})**, which uses expert-provided annotations during training to support confounding adjustment.

## Structure
- `data/`: notebooks for preparing datasets and embeddings (data files are not tracked).
- `library/`: core implementation (models, training loops, metrics, utilities).
- `embeddings.ipynb`: compute text embeddings and save them for downstream models.
- `nuisances.ipynb`: nuisance estimation.
- `aace.ipynb`: annotation-assisted treatment effect estimation and policy learning (**AACE**).
- `risk-based.ipynb`: risk-based treatment policy learning.
- `causal-rep.ipynb`: representation-based causal baselines (e.g., TARNet, DragonNet).

## Usage
The code requires Python 3 and standard scientific Python packages, including PyTorch, NumPy, pandas, NLTK, and transformers.

Run the notebooks in `data/` to generate or load datasets and embeddings, then use the notebooks in the root directory to train models and evaluate policies. The demo notebooks are configured to run on **SynSum** by default. The same notebooks also support the other datasets in `data/`, which requires changing the dataset-specific arguments (e.g., dataset name and variables) in the notebook cells.