"""
Semi-Supervised VAE model.

Ported from autoencoder_contrastive_classifier/train_vae.py.
YAML: training.model.latent_dim, training.model.n_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SemiSupervisedVAE(nn.Module):
    """Variational autoencoder with classifier head on latent space."""

    def __init__(self, n_continuous=240, n_binary=0, latent_dim=2, n_classes=5):
        super().__init__()
        self.n_continuous = n_continuous
        self.n_binary = n_binary
        self.latent_dim = latent_dim
        input_dim = n_continuous + n_binary

        self.decoder_bottleneck = max(128, min(512, n_continuous))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder_shared = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.decoder_bottleneck),
            nn.ReLU(),
        )

        self.decoder_cont = nn.Linear(self.decoder_bottleneck, n_continuous)
        if n_binary > 0:
            self.decoder_bin = nn.Sequential(
                nn.Linear(self.decoder_bottleneck, n_binary),
                nn.Sigmoid(),
            )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_shared(z)
        out_cont = self.decoder_cont(h)
        if self.n_binary > 0:
            out_bin = self.decoder_bin(h)
            return torch.cat([out_cont, out_bin], dim=1)
        return out_cont

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        class_logits = self.classifier(z)
        return recon_x, mu, logvar, class_logits
