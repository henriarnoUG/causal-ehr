from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from library.data_utils import set_seed


####################################################################################################################################


def train_propensity(model, train_loader, val_loader, device, lr=3e-4, 
                     weight_decay=1e-5, max_epochs=50, patience=5, seed=0):
    """ Expects loaders that yield (x, t, y). """
    set_seed(seed)
    
    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # early stopping
    best_state, best_val, patience_left = None, float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for x, t, _ in pbar:

                # forward pass
                x, t = x.to(device), t.to(device)
                logits = model(x)

                # backward pass
                loss = criterion(logits, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = t.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loop
        model.eval()
        running_loss_val, n_val = 0.0, 0
        with torch.no_grad():
            for x_val, t_val, _ in val_loader:
                x_val, t_val = x_val.to(device), t_val.to(device)
                logits_val = model(x_val)
                loss_val = criterion(logits_val, t_val)

                # progress
                bs_val = t_val.size(0)
                running_loss_val += loss_val.item() * bs_val
                n_val += bs_val
        val_loss = running_loss_val / max(n_val, 1)

        # progress
        tqdm.write(f"Epoch {epoch:02d} | val_loss={val_loss:.4f}")

        
        # early stopping
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}



####################################################################################################################################



def train_response(model, train_loader, val_loader, device, lr=3e-4, 
                   weight_decay=1e-5, max_epochs=50, patience=5, seed=0):
    """ Expects loaders that yield (x, t, y). """
    set_seed(seed)

    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="mean")

    # early stopping
    best_state, best_val, patience_left = None, float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for x, _, y in pbar:

                # forward pass
                x, y = x.to(device), y.to(device)
                preds = model(x)

                # backward pass
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = y.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loops
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for x_val, _, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds_val = model(x_val)
                loss_val = criterion(preds_val, y_val)

                # progress
                bs_val = y_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val
        val_mse = val_sum / max(n_val, 1)

        # progress
        tqdm.write(f"Epoch {epoch:02d} | val_mse={val_mse:.6f}")

        
        # early stopping
        if val_mse + 1e-9 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}



####################################################################################################################################



def train_cate(model, train_loader, val_loader, device, lr=3e-4, 
               weight_decay=1e-5, max_epochs=50, patience=5, seed=0, tabular=False):
    """ Expects loaders that yield (phi, x, dr)."""
    set_seed(seed)

    # init
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="mean")

     # early stopping
    best_state, best_val, patience_left = None, float("inf"), patience

    # loop over epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, n_train = 0.0, 0

        # progress
        with tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False) as pbar:
            for phi, x, dr in pbar:

                # set correct inputs
                if tabular:
                    inp = x
                else:
                    inp = phi

                # forward pass
                inp, dr = inp.to(device), dr.to(device)
                preds = model(inp)

                # backward pass
                loss = criterion(preds, dr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # progress
                bs = dr.size(0)
                running_loss += loss.item() * bs
                n_train += bs
                pbar.set_postfix(loss=running_loss / n_train)

        # validation loop
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for phi_val, x_val, dr_val in val_loader:
                # set correct inputs
                if tabular:
                    inp_val = x_val
                else:
                    inp_val = phi_val

                inp_val, dr_val = inp_val.to(device), dr_val.to(device)
                preds_val = model(inp_val)
                loss_val = criterion(preds_val, dr_val)

                # progress
                bs_val = dr_val.size(0)
                val_sum += loss_val.item() * bs_val
                n_val += bs_val
        val_mse = val_sum / max(n_val, 1)

        # progress 
        tqdm.write(f"Epoch {epoch:02d} | val_mse={val_mse:.6f}")

        # early stopping
        if val_mse + 1e-9 < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    # reset best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"val_loss": best_val}