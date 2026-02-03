import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


####################################################################################

def autoc(ranked):
    # mean effect (varying k)
    cate = ranked['cate'].to_numpy()
    n = len(cate)
    cumsum = np.cumsum(cate)
    top_means = cumsum / np.arange(1, n+1)

    # improvement over ate
    ate = cate.mean()
    toc = top_means - ate
    return toc.mean()

####################################################################################

def policy_value(ranked):
    # potential outcomes
    M0 = ranked['M0'].to_numpy()
    M1 = ranked['M1'].to_numpy()
    n = M0.size

    # cumulative outcomes
    csum_M1 = np.cumsum(M1)
    csum_M0 = np.cumsum(M0)
    total_M0 = csum_M0[-1]

    # mean policy value
    ks = np.arange(1, n + 1, dtype=int)
    treated_sum   = csum_M1[ks - 1]
    untreated_sum = total_M0 - csum_M0[ks - 1]
    pv = (treated_sum + untreated_sum) / n
    return pv.mean()

####################################################################################

def pehe(ranked):
    cate = ranked['cate'].to_numpy(dtype=float)
    est  = ranked['est'].to_numpy(dtype=float)
    return float(np.sqrt(np.mean((cate - est) ** 2)))


####################################################################################

class EvalDataset(Dataset):
    def __init__(self, df, confounders_tabular, confounders_full, embeddings):
        self.df = df
        self.confounders_tabular = confounders_tabular
        self.confounders_full = confounders_full
        self.embeddings = embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.confounders_full].astype(np.float32).to_numpy(), dtype=torch.float32)

        data_idx = row.name
        emb = self.embeddings[data_idx].to(torch.float32)
        device = self.embeddings.device
        x_tab = torch.tensor(row[self.confounders_tabular].astype(np.float32).to_numpy(), dtype=torch.float32, device=device)
        phi = torch.cat([emb, x_tab])
        
        cate = torch.tensor(float(row['cate']), dtype=torch.float32)
        M0 = torch.tensor(float(row['M0']), dtype=torch.float32)
        M1 = torch.tensor(float(row['M1']), dtype=torch.float32)
        
        return phi, x, cate, M0, M1