import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from typing import List


def _generate_tensors(data, device = "cpu"):

    tensors = []
    unpacked_data = []

    for sublist in data:
        if isinstance(sublist,list):
            unpacked_data.extend(sublist)
        else:
            unpacked_data.append(sublist)

    for d in unpacked_data:
        arr = np.array(d, dtype=np.float32)
        t = torch.tensor(arr, dtype=torch.float32).to(device)
        t = torch.tensor(arr, dtype=torch.float32)
        tensors.append(t)

    return tensors


def _normalise_dataset(x, x_val = [], scaler_type = StandardScaler, scalers = None):
    
    x_train_scaled = []
    x_val_scaled = []

    if not scalers:

        scalers = []

        for i, x_tr in enumerate(x):
            if x_tr.ndim == 3:
                # x_tr shape: (samples, timesteps, features)
                n_samples, n_timesteps, n_features = x_tr.shape
                x_tr_flat = x_tr.reshape(-1, n_features)
                scaler = scaler_type()
                x_tr_scaled_flat = scaler.fit_transform(x_tr_flat)
                scalers.append(scaler)
                x_tr_scaled = x_tr_scaled_flat.reshape(n_samples, n_timesteps, n_features)

                if x_val:
                    x_val_flat = x_val[i].reshape(-1, n_features)
                    x_val_scaled_flat = scaler.transform(x_val_flat)
                    x_val_scaled_i = x_val_scaled_flat.reshape(x_val[i].shape[0], n_timesteps, n_features)

            elif x_tr.ndim == 2:
                scaler = scaler_type()
                x_tr_scaled = scaler.fit_transform(x_tr)
                scalers.append(scaler)
                if x_val:
                    x_val_scaled_i = scaler.transform(x_val[i])

            else:
                raise ValueError(f"Unsupported input dimension: {x_tr.ndim}")

            x_train_scaled.append(x_tr_scaled)
            if x_val:
                x_val_scaled.append(x_val_scaled_i)
    
    else:
        for i, x_tr in enumerate(x):
            if x_tr.ndim == 3:
                n_samples, n_timesteps, n_features = x_tr.shape
                x_flat = x_tr.reshape(-1, n_features)
                x_scaled_flat = scalers[i].transform(x_flat)
                x_scaled_i = x_scaled_flat.reshape(x_tr.shape[0], n_timesteps, n_features)

            elif x_tr.ndim == 2:
                x_scaled_i = scalers[i].transform(x_tr)

            x_train_scaled.append(x_scaled_i)

    return x_train_scaled, x_val_scaled, scalers


def _sort_features(idx: np.ndarray, x: List[np.ndarray], x_to_sort: List[int] = [0]) -> List[np.ndarray]:
    """
    Riordina le feature di x in base agli indici specificati.

    Parameters
    ----------
    idx : np.ndarray
        Array di indici secondo cui effettuare l'ordinamento (tipicamente (n,) o (n, k)).
    x : list of np.ndarray
        Lista di array da ordinare o replicare.
    x_to_sort : list of int
        Indici delle feature in `x` che devono essere ordinate usando `idx`.

    Returns
    -------
    x_new : list of np.ndarray
        Lista di array ordinati o replicati coerentemente con `idx`.
    """
    x_new = x.copy()
    x_new = [t.detach().cpu().numpy() if hasattr(t, 'detach') else t.numpy() for t in x_new]
    n = len(idx[0])

    for j in x_to_sort:
        x_new[j] = np.repeat(x_new[j], idx.shape[0], axis=0)

        for k in range(x_new[j].shape[0]):
            sliced = x_new[j][k][0:n, ...]
            ordered = np.take(sliced, idx[k], axis=0)
            padding_rows = x_new[j].shape[1] - ordered.shape[0]

            if padding_rows > 0:
                padding = np.zeros((padding_rows, ordered.shape[1]), dtype=ordered.dtype)
                ordered_padded = np.vstack([ordered, padding])
            else:
                ordered_padded = ordered

            x_new[j][k] = ordered_padded

    for j in range(len(x)):
        if j not in x_to_sort:
            x_new[j] = np.repeat(x_new[j], idx.shape[0], axis=0)
            
    return x_new

    x1 = np.zeros((population.shape[0], self.max_seq_len, self.features_rec.shape[2]))
    for i in range(population.shape[0]):
        seq = population[i]
        x1[i, :len(seq)] = self.features_rec[0, seq]
    x2 = np.repeat(self.features_lin[:], population.shape[0], axis=0)