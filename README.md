# Learning Treatment Policies From Multimodal Electronic Health Records
This repository contains the code accompanying the paper: `Learning Treatment Policies From Multimodal Electronic Health Records.`

## Structure
- `data/`: notebooks for preparing datasets and embeddings (data files are not tracked).
- `library/`: core implementation (models, training loops, metrics, utilities).
- `nuisances.ipynb`: nuisance estimation.
- `embeddings.ipynb`: compute text embeddings and save them for downstream models.
- `coarsened_effects.ipynb`: treatment policy learning from coarsened effects (**proposed method**).
- `risk-based.ipynb`: risk-based treatment policy learning.

## Usage
The code requires Python 3 and standard scientific Python packages, including PyTorch, NumPy, pandas, nltk and transformers.

Run the notebooks in `data/` to generate (or load) datasets and embeddings, then use the notebooks in the root directory to train models and evaluate policies. The demo notebooks are configured to run on **SynSum** by default. The same notebooks also support the other datasets in `data/` which requires changing the dataset-specific arguments (e.g., dataset name and variables) in the notebook cells.
