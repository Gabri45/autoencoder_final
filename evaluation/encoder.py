"""
Latent space encoding for single model and ensemble.

Outputs latent coordinates, predictions, confidence, and binary features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def encode_single(model, scaler, features, n_continuous, n_binary, batch_size=1024, device=None):
    """
    Encode all frames with a single VAE model.

    Returns:
        latent_coords, y_pred, confidence, x_bin
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)

    if n_continuous > 0:
        x_cont = features[:, :n_continuous]
        x_cont_scaled = scaler.transform(x_cont).astype(np.float32)
    else:
        x_cont_scaled = np.zeros((features.shape[0], 0), dtype=np.float32)

    if n_binary > 0:
        x_bin = features[:, n_continuous:]
        x_scaled = np.concatenate([x_cont_scaled, x_bin], axis=1).astype(np.float32)
    else:
        x_bin = None
        x_scaled = x_cont_scaled

    x_tensor = torch.tensor(x_scaled, device=device)
    all_mu, all_preds, all_conf = [], [], []

    with torch.no_grad():
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i:i + batch_size]
            _, mu, _, logits = model(batch)
            probs = F.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)
            all_mu.append(mu.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_conf.append(conf.cpu().numpy())

    latent_coords = np.concatenate(all_mu, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    confidence = np.concatenate(all_conf, axis=0)
    return latent_coords, y_pred, confidence, x_bin


def predict_ensemble(models, scaler, features, batch_size=2048):
    """
    Run ensemble prediction.

    Returns:
        mean_mu, std_mu, latent_uncertainty, y_pred, confidence, x_bin
    """
    device = next(models[0].parameters()).device
    n_continuous = models[0].n_continuous
    n_binary = models[0].n_binary

    if n_continuous > 0:
        x_cont = features[:, :n_continuous]
        x_cont_scaled = scaler.transform(x_cont).astype(np.float32)
    else:
        x_cont_scaled = np.zeros((features.shape[0], 0), dtype=np.float32)

    if n_binary > 0:
        x_bin = features[:, n_continuous:]
        x_scaled = np.concatenate([x_cont_scaled, x_bin], axis=1).astype(np.float32)
    else:
        x_bin = None
        x_scaled = x_cont_scaled

    x_tensor = torch.tensor(x_scaled)
    n_samples = len(x_tensor)

    all_mean_mu, all_std_mu, all_mean_probs = [], [], []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = x_tensor[i:i + batch_size].to(device)
            batch_mus, batch_probs = [], []

            for model in models:
                _, mu, _, logits = model(batch)
                batch_mus.append(mu.cpu().numpy())
                batch_probs.append(F.softmax(logits, dim=1).cpu().numpy())

            batch_mus = np.stack(batch_mus)
            batch_probs = np.stack(batch_probs)

            all_mean_mu.append(np.mean(batch_mus, axis=0))
            all_std_mu.append(np.std(batch_mus, axis=0))
            all_mean_probs.append(np.mean(batch_probs, axis=0))

    final_mean_mu = np.concatenate(all_mean_mu, axis=0)
    final_std_mu = np.concatenate(all_std_mu, axis=0)
    final_mean_probs = np.concatenate(all_mean_probs, axis=0)

    latent_uncertainty = np.linalg.norm(final_std_mu, axis=1)
    y_pred = np.argmax(final_mean_probs, axis=1)
    confidence = np.max(final_mean_probs, axis=1)

    return final_mean_mu, final_std_mu, latent_uncertainty, y_pred, confidence, x_bin
