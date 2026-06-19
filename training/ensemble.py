"""
Ensemble VAE trainer.

Trains N SemiSupervisedVAE models, saves ensemble_model_{i}.pt and ensemble_scaler.pkl.
YAML: training.mode=ensemble, training.n_ensemble
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.loss import vae_loss_function
from model.vae import SemiSupervisedVAE
from training.monitor import TrainingMonitor


class EnsembleTrainer:
    """Train an ensemble of Semi-Supervised VAEs."""

    def __init__(self, config: dict, monitor: TrainingMonitor):
        self.config = config
        self.monitor = monitor
        self.train_cfg = config["training"]
        self.loss_cfg = self.train_cfg.get("loss", {})
        self.output_dir = config["project"]["output_dir"]
        self.n_models = self.train_cfg.get("n_ensemble", 5)

        device_name = self.train_cfg.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name if device_name != "cuda" else "cuda:0")

    def _prepare_data(self, features, traj_labels, n_continuous, n_binary):
        """Scale and split data (same split for all ensemble members)."""
        scaler = StandardScaler()
        x_cont = features[:, :n_continuous]
        x_bin = features[:, n_continuous:] if n_binary > 0 else np.zeros((len(features), 0))

        x_cont_scaled = scaler.fit_transform(x_cont).astype(np.float32)
        x_scaled = np.concatenate([x_cont_scaled, x_bin], axis=1).astype(np.float32)

        scaler_path = os.path.join(self.output_dir, "ensemble_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        self.monitor.log(f"Saved ensemble scaler to {scaler_path}")

        seed = self.config["project"].get("seed", 42)
        n_samples = len(x_scaled)
        val_split = self.train_cfg.get("val_split", 0.3)
        split = int((1.0 - val_split) * n_samples)
        indices = np.random.RandomState(seed).permutation(n_samples)
        train_idx, val_idx = indices[:split], indices[split:]

        x_tensor = torch.tensor(x_scaled)
        y_tensor = torch.tensor(traj_labels, dtype=torch.long)

        batch_size = self.train_cfg.get("batch_size", 128)
        train_loader = DataLoader(
            TensorDataset(x_tensor[train_idx], y_tensor[train_idx]),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_tensor[val_idx], y_tensor[val_idx]),
            batch_size=batch_size, shuffle=False,
        )
        return train_loader, val_loader, n_continuous, n_binary

    def train(self, features, traj_labels, n_continuous, n_binary):
        """Train all ensemble models sequentially."""
        train_loader, val_loader, n_continuous, n_binary = self._prepare_data(
            features, traj_labels, n_continuous, n_binary
        )

        model_cfg = self.train_cfg.get("model", {})
        latent_dim = model_cfg.get("latent_dim", 2)
        n_classes = model_cfg.get("n_classes", 5)

        epochs = self.train_cfg.get("epochs", 200)
        lr = self.train_cfg.get("lr", 1e-3)
        beta_max = self.loss_cfg.get("beta_max", 1.0)
        warmup_epochs = self.loss_cfg.get("warmup_epochs", 50)
        gamma = self.loss_cfg.get("gamma", 1.0)
        margin = self.loss_cfg.get("margin", 1.0)
        free_bits = self.loss_cfg.get("free_bits", 0.5)
        patience = self.train_cfg.get("early_stopping", {}).get("patience", 20)

        checkpoint_paths = []

        for model_id in range(self.n_models):
            self.monitor.log(f"\n{'='*60}\nTraining Ensemble Model {model_id + 1}/{self.n_models}\n{'='*60}")

            torch.manual_seed(model_id * 42 + self.config["project"].get("seed", 42))

            model = SemiSupervisedVAE(
                n_continuous=n_continuous,
                n_binary=n_binary,
                latent_dim=latent_dim,
                n_classes=n_classes,
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=10
            )

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None
            best_epoch = 0
            checkpoint_path = os.path.join(self.output_dir, f"ensemble_model_{model_id}.pt")

            for epoch in range(1, epochs + 1):
                if epoch <= warmup_epochs:
                    beta = beta_max * (epoch / warmup_epochs)
                else:
                    beta = beta_max

                model.train()
                train_loss_sum = 0
                n_batches = 0

                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    optimizer.zero_grad()

                    recon, mu, logvar, class_logits = model(batch_x)
                    loss, recon_loss, kl_loss, contrastive_loss, class_loss = vae_loss_function(
                        recon, batch_x, mu, logvar, class_logits, batch_y,
                        n_continuous=n_continuous, beta=beta, gamma=gamma,
                        margin=margin, free_bits=free_bits,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    train_loss_sum += loss.item()
                    n_batches += 1

                avg_train = train_loss_sum / n_batches

                model.eval()
                val_loss_sum = 0
                n_val = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        recon, mu, logvar, class_logits = model(batch_x)
                        loss, _, _, _, _ = vae_loss_function(
                            recon, batch_x, mu, logvar, class_logits, batch_y,
                            n_continuous=n_continuous, beta=beta, gamma=gamma,
                            margin=margin, free_bits=free_bits,
                        )
                        val_loss_sum += loss.item()
                        n_val += 1

                avg_val = val_loss_sum / n_val
                scheduler.step(avg_val)
                current_lr = optimizer.param_groups[0]["lr"]

                self.monitor.record_epoch(
                    epoch, model_id=model_id,
                    metrics={"train_loss": avg_train, "val_loss": avg_val, "beta": beta,
                             "recon": 0, "kl": 0, "class": 0, "contrastive": 0},
                    lr=current_lr,
                )

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_epoch = epoch
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.monitor.log(f"Early stopping model {model_id} at epoch {epoch}")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            torch.save({
                "model_state_dict": model.state_dict(),
                "n_continuous": n_continuous,
                "n_binary": n_binary,
                "latent_dim": latent_dim,
                "val_loss": best_val_loss,
            }, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            self.monitor.set_best(model_id, best_epoch, best_val_loss, checkpoint_path)
            self.monitor.log(f"Model {model_id} best val={best_val_loss:.4f} epoch={best_epoch}")

        return checkpoint_paths
