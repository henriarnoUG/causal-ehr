# general imports
import random
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
import json
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


#########################################################################################################################

class TARNetDataset(Dataset):
    def __init__(self, df, confounders_tabular, embeddings):
        self.df = df
        self.confounders_tabular = confounders_tabular
        self.embeddings = embeddings
        self.T = 'T'
        self.Y = 'Y'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        data_idx = row.name
        emb = self.embeddings[data_idx].to(torch.float32)
        device = self.embeddings.device
        x_tab = torch.tensor(row[self.confounders_tabular].astype(np.float32).to_numpy(), dtype=torch.float32, device=device)
        phi = torch.cat([emb, x_tab])
        
        t = torch.tensor(float(row[self.T]), dtype=torch.float32)
        y = torch.tensor(float(row[self.Y]), dtype=torch.float32)
        return phi, t, y



def make_tarnet_loaders(train_df, val_df, confounders_tabular, embeddings, batch_size):
    train_ds = TARNetDataset(train_df, confounders_tabular, embeddings)
    val_ds = TARNetDataset(val_df, confounders_tabular, embeddings)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class DragonNetDataset(Dataset):
    def __init__(self, df, confounders_tabular, embeddings):
        self.df = df
        self.confounders_tabular = confounders_tabular
        self.embeddings = embeddings
        self.T = 'T'
        self.Y = 'Y'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        data_idx = row.name
        emb = self.embeddings[data_idx].to(torch.float32)
        device = self.embeddings.device
        x_tab = torch.tensor(row[self.confounders_tabular].astype(np.float32).to_numpy(), dtype=torch.float32, device=device)
        phi = torch.cat([emb, x_tab])
        
        t = torch.tensor(float(row[self.T]), dtype=torch.float32)
        y = torch.tensor(float(row[self.Y]), dtype=torch.float32)
        return phi, t, y



def make_dragonnet_loaders(train_df, val_df, confounders_tabular, embeddings, batch_size):
    train_ds = DragonNetDataset(train_df, confounders_tabular, embeddings)
    val_ds = DragonNetDataset(val_df, confounders_tabular, embeddings)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

#########################################################################################################################


class TARNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        # shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # potential outcome heads
        self.y0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.y1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        z = self.shared(x)

        y0 = self.y0_head(z).squeeze(1)
        y1 = self.y1_head(z).squeeze(1)

        return y0, y1



class DragonNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()

        # shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # propensity head
        self.propensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # potential outcome heads
        self.y0_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.y1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        z = self.shared(x)

        e_logit = self.propensity_head(z).squeeze(1)  # logits
        y0 = self.y0_head(z).squeeze(1)
        y1 = self.y1_head(z).squeeze(1)

        return e_logit, y0, y1


#########################################################################################################################


def train_tarnet(model, train_loader, val_loader,
                    device, lr=3e-4, weight_decay=1e-5,
                    max_epochs=50, patience=5, seed=0):
    set_seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.MSELoss(reduction="mean")

    best_state, best_val = None, float("inf")
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        # -------- train --------
        model.train()
        running_loss, n_train = 0.0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for phi, t, y in pbar:
                phi, t, y = phi.to(device), t.to(device), y.to(device)

                y0, y1 = model(phi)

                # factual prediction
                y_factual = t * y1 + (1.0 - t) * y0
                loss = criterion(y_factual, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # -------- validate --------
        model.eval()
        val_sum, n_val = 0.0, 0

        with torch.no_grad():
            for phi_val, t_val, y_val in val_loader:
                phi_val, t_val, y_val = phi_val.to(device), t_val.to(device), y_val.to(device)

                y0_val, y1_val = model(phi_val)
                y_factual_val = t_val * y1_val + (1.0 - t_val) * y0_val
                loss_val = criterion(y_factual_val, y_val)

                bs_val = y_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val

        val_loss = val_sum / max(n_val, 1)

        tqdm.write(f"Epoch {epoch:02d} | val_loss={val_loss:.6f}")

        # -------- early stopping --------
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"val_loss": best_val}




def train_dragonnet(model, train_loader, val_loader,
                    device, lr=3e-4, weight_decay=1e-5,
                    max_epochs=50, patience=5, seed=0, alpha=1.0):
    set_seed(seed)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion_y = nn.MSELoss(reduction="mean")
    criterion_t = nn.BCEWithLogitsLoss(reduction="mean")

    best_state, best_val = None, float("inf")
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        # -------- train --------
        model.train()
        running_loss, n_train = 0.0, 0

        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for phi, t, y in pbar:
                phi, t, y = phi.to(device), t.to(device), y.to(device)

                e_logit, y0, y1 = model(phi)

                # factual prediction
                y_factual = t * y1 + (1.0 - t) * y0

                loss_y = criterion_y(y_factual, y)
                loss_t = criterion_t(e_logit, t)
                loss = loss_y + alpha * loss_t

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # -------- validate --------
        model.eval()
        val_sum, n_val = 0.0, 0

        with torch.no_grad():
            for phi_val, t_val, y_val in val_loader:
                phi_val, t_val, y_val = phi_val.to(device), t_val.to(device), y_val.to(device)

                e_logit_val, y0_val, y1_val = model(phi_val)
                y_factual_val = t_val * y1_val + (1.0 - t_val) * y0_val

                loss_y_val = criterion_y(y_factual_val, y_val)
                loss_t_val = criterion_t(e_logit_val, t_val)
                loss_val = loss_y_val + alpha * loss_t_val

                bs_val = y_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val

        val_loss = val_sum / max(n_val, 1)

        tqdm.write(f"Epoch {epoch:02d} | val_loss={val_loss:.6f}")

        # -------- early stopping --------
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"val_loss": best_val}