"""
VAE loss function with reconstruction, KL, classification, and contrastive terms.

Ported from autoencoder_contrastive_classifier/train_vae.py.
YAML: training.loss (beta_max, gamma, margin, free_bits)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss_function(
    recon_x,
    x,
    mu,
    logvar,
    class_logits,
    labels,
    n_continuous,
    beta=0.25,
    gamma=1.0,
    margin=1.0,
    free_bits=0.5,
):
    """
    Loss for Semi-Supervised VAE with conditional contrastive loss.

    Returns:
        total_loss, recon_loss, kl_loss, contrastive_loss, class_loss
    """
    batch_size = x.size(0)

    recon_cont = recon_x[:, :n_continuous]
    x_cont = x[:, :n_continuous]
    recon_loss_cont = F.mse_loss(recon_cont, x_cont, reduction="sum") / batch_size

    if recon_x.shape[1] > n_continuous:
        recon_bin = recon_x[:, n_continuous:]
        x_bin = x[:, n_continuous:]
        epsilon = 1e-7
        recon_bin_clamped = torch.clamp(recon_bin, epsilon, 1.0 - epsilon)
        recon_loss_bin = F.binary_cross_entropy(recon_bin_clamped, x_bin, reduction="sum") / batch_size
        recon_loss = recon_loss_cont + recon_loss_bin
    else:
        recon_loss = recon_loss_cont
        x_bin = None

    kl_per_dim = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
    kl_per_dim = kl_per_dim / batch_size
    kl_free_bits = torch.maximum(
        kl_per_dim,
        torch.tensor(free_bits, device=kl_per_dim.device),
    )
    kl_loss = torch.sum(kl_free_bits)

    mask_cert = labels != -1
    mask_interm = labels == -1

    class_loss = torch.tensor(0.0, device=mu.device)
    if mask_cert.sum() > 0:
        class_loss = F.cross_entropy(class_logits[mask_cert], labels[mask_cert])

    contrastive_loss = torch.tensor(0.0, device=mu.device)
    if x.shape[1] > n_continuous and mask_interm.sum() > 0:
        mu_interm = mu[mask_interm]
        x_bin_interm = x_bin[mask_interm]

        dist_z = torch.cdist(mu_interm, mu_interm, p=2)
        dist_b = torch.cdist(x_bin_interm, x_bin_interm, p=1)

        mask_pos = (dist_b == 0).float()
        mask_pos.fill_diagonal_(0)
        mask_neg = (dist_b > 0).float()

        num_pos = mask_pos.sum() + 1e-8
        num_neg = mask_neg.sum() + 1e-8

        loss_pos = (mask_pos * dist_z.pow(2)).sum() / num_pos
        loss_neg = (mask_neg * F.relu(margin - dist_z).pow(2)).sum() / num_neg

        contrastive_loss = loss_pos + loss_neg

    total_loss = recon_loss + beta * kl_loss + gamma * class_loss + gamma * contrastive_loss

    return total_loss, recon_loss, kl_loss, contrastive_loss, class_loss
