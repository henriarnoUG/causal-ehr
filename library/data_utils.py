import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

##########################################################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_splits(df, train_frac=0.8, seed=0):
    # seed + rng
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # sizes
    assert 0 < train_frac <= 0.8
    n_total = len(df)
    n_test = int(n_total * 0.1)

    # test set
    test = df.tail(n_test).copy()
    remaining = df.iloc[:-n_test].copy()

    # val set
    n_val = int(n_total * 0.1)
    val_indices = rng.choice(remaining.index, size=n_val, replace=False)
    val = remaining.loc[val_indices].copy()

    # train set
    n_train = int(n_total * train_frac)
    train_pool = remaining.drop(val_indices)
    if len(train_pool) > n_train:
        train_indices = rng.choice(train_pool.index, size=n_train, replace=False)
        train = train_pool.loc[train_indices].copy()
    else:
        train = train_pool
    return train, val, test

##########################################################################################################

class NuisanceDataset(Dataset):
    def __init__(self, df, confounders):
        self.df = df
        self.confounders = confounders
        self.T = 'T'
        self.Y = 'Y'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.confounders].astype(np.float32).to_numpy(), dtype=torch.float32)
        t = torch.tensor(float(row[self.T]), dtype=torch.float32)
        y = torch.tensor(float(row[self.Y]), dtype=torch.float32)
        return x, t, y


class CateDataset(Dataset):
    def __init__(self, df, confounders_tabular, confounders_full, embeddings, target_col):
        self.df = df
        self.confounders_tabular = confounders_tabular
        self.confounders_full = confounders_full
        self.embeddings = embeddings
        self.target_col = target_col

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
        
        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return phi, x, target


class PredictiveDataset(Dataset):
    def __init__(self, df, confounders_tabular, embeddings):
        self.df = df
        self.confounders_tabular = confounders_tabular
        self.embeddings = embeddings
        self.target_col = 'Y'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        data_idx = row.name
        emb = self.embeddings[data_idx].to(torch.float32)
        device = self.embeddings.device
        x_tab = torch.tensor(row[self.confounders_tabular].astype(np.float32).to_numpy(), dtype=torch.float32, device=device)
        phi = torch.cat([emb, x_tab])
        
        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return phi, target

##########################################################################################################

def make_nuisance_loaders(train_df, val_df, confounders, batch_size):
    train_ds = NuisanceDataset(train_df, confounders)
    val_ds = NuisanceDataset(val_df, confounders)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def make_cate_loaders(train_df, val_df, confounders_tabular, confounders_full, embeddings, target_col, batch_size):
    train_ds = CateDataset(train_df, confounders_tabular, confounders_full, embeddings, target_col)
    val_ds = CateDataset(val_df, confounders_tabular, confounders_full, embeddings, target_col)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def make_predictive_loaders(train_df, val_df, confounders_tabular, embeddings, batch_size):
    train_ds = PredictiveDataset(train_df, confounders_tabular, embeddings)
    val_ds = PredictiveDataset(val_df, confounders_tabular, embeddings)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

##########################################################################################################

@torch.no_grad()
def score_propensity(df, confounders, model, device):
    """helper to get propensity predictions in dataframe"""
    batch_size=1024
    model.eval()
    xs = torch.tensor(df[confounders].astype(np.float32).values, dtype=torch.float32)
    probs = []
    for i in range(0, len(xs), batch_size):
        x = xs[i:i+batch_size].to(device)
        logits = model(x)
        probs.append(torch.sigmoid(logits).detach().cpu())
    return torch.cat(probs).numpy()
    
@torch.no_grad()
def score_response(df, confounders, model, device):
    """helper to get reponse surface predictions in dataframe"""
    batch_size=1024
    model.eval()
    xs = torch.tensor(df[confounders].astype(np.float32).values, dtype=torch.float32)
    preds = []
    for i in range(0, len(xs), batch_size):
        x = xs[i:i+batch_size].to(device)
        yhat = model(x)
        preds.append(yhat.detach().cpu())
    return torch.cat(preds).numpy()

def compute_dr_scores(df, confounders, prop_model, m0_model, m1_model, device):
    
    # score nuisances
    e_hat  = score_propensity(df, confounders, prop_model, device).astype(np.float32).ravel()
    m0_hat = score_response(df, confounders, m0_model, device).astype(np.float32).ravel()
    m1_hat = score_response(df, confounders, m1_model, device).astype(np.float32).ravel()

    # DR components
    T = df["T"].astype(np.float32).to_numpy()
    Y = df["Y"].astype(np.float32).to_numpy()

    # stabilize denominator
    eps = 1e-3
    denom_e   = np.clip(e_hat,   eps, 1.0 - eps)
    denom_1_e = np.clip(1.0 - e_hat, eps, 1.0 - eps)

    # compute DR scores
    term_treated = (T * (Y - m1_hat)) / denom_e
    term_control = ((1.0 - T) * (Y - m0_hat)) / denom_1_e
    dr = term_treated - term_control + (m1_hat - m0_hat)

    # store
    df["e_hat"]  = e_hat
    df["m0_hat"] = m0_hat
    df["m1_hat"] = m1_hat
    df["DR"] = dr.astype(np.float32)
    return df