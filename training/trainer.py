"""
Single-model VAE trainer.

Trains one SemiSupervisedVAE, saves vae_model.pt and scaler.pkl.
YAML: training.mode=single, training.* hyperparameters
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


class SingleModelTrainer:
    """Train a single Semi-Supervised VAE."""

    def __init__(self, config: dict, monitor: TrainingMonitor):
        self.config = config
        self.monitor = monitor
        self.train_cfg = config["training"]
        self.loss_cfg = self.train_cfg.get("loss", {})
        self.output_dir = config["project"]["output_dir"]

        device_name = self.train_cfg.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name if device_name != "cuda" else "cuda:0")

    def _prepare_data(self, features, traj_labels, n_continuous, n_binary):
        """Scale continuous features and split train/val."""
        scaler = StandardScaler()
        x_cont = features[:, :n_continuous]
        x_bin = features[:, n_continuous:] if n_binary > 0 else np.zeros((len(features), 0))

        x_cont_scaled = scaler.fit_transform(x_cont).astype(np.float32)
        x_scaled = np.concatenate([x_cont_scaled, x_bin], axis=1).astype(np.float32)

        scaler_path = os.path.join(self.output_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        self.monitor.log(f"Saved scaler to {scaler_path}")

        seed = self.config["project"].get("seed", 42)
        n_samples = len(x_scaled)
        val_split = self.train_cfg.get("val_split", 0.3)
        split = int((1.0 - val_split) * n_samples)
        indices = np.random.RandomState(seed).permutation(n_samples)
        train_idx, val_idx = indices[:split], indices[split:]

        x_train = torch.tensor(x_scaled[train_idx])
        x_val = torch.tensor(x_scaled[val_idx])
        y_train = torch.tensor(traj_labels[train_idx], dtype=torch.long)
        y_val = torch.tensor(traj_labels[val_idx], dtype=torch.long)

        batch_size = self.train_cfg.get("batch_size", 256)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader, scaler, n_continuous, n_binary

    def train(self, features, traj_labels, n_continuous, n_binary):
        """Run full training loop."""
        train_loader, val_loader, scaler, n_continuous, n_binary = self._prepare_data(
            features, traj_labels, n_continuous, n_binary
        )

        model_cfg = self.train_cfg.get("model", {})
        latent_dim = model_cfg.get("latent_dim", 2)
        n_classes = model_cfg.get("n_classes", 5)

        model = SemiSupervisedVAE(
            n_continuous=n_continuous,
            n_binary=n_binary,
            latent_dim=latent_dim,
            n_classes=n_classes,
        ).to(self.device)

        lr = self.train_cfg.get("lr", 1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        epochs = self.train_cfg.get("epochs", 200)
        beta_max = self.loss_cfg.get("beta_max", 0.25)
        warmup_epochs = self.loss_cfg.get("warmup_epochs", 0)
        gamma = self.loss_cfg.get("gamma", 1.0)
        margin = self.loss_cfg.get("margin", 1.0)
        free_bits = self.loss_cfg.get("free_bits", 0.5)
        patience = self.train_cfg.get("early_stopping", {}).get("patience", 20)

        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0
        checkpoint_path = os.path.join(self.output_dir, "vae_model.pt")

        self.monitor.log(f"Training single model on {self.device}")

        for epoch in range(1, epochs + 1):
            if warmup_epochs > 0 and epoch <= warmup_epochs:
                beta = beta_max * (epoch / warmup_epochs)
            else:
                beta = beta_max

            model.train()
            train_metrics = {"train_loss": 0, "recon": 0, "kl": 0, "class": 0, "contrastive": 0}
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

                train_metrics["train_loss"] += loss.item()
                train_metrics["recon"] += recon_loss.item()
                train_metrics["kl"] += kl_loss.item()
                train_metrics["class"] += class_loss.item()
                train_metrics["contrastive"] += contrastive_loss.item()
                n_batches += 1

            for k in train_metrics:
                train_metrics[k] /= n_batches

            model.eval()
            val_loss_sum = 0
            val_recon = val_kl = val_class = val_contrastive = 0
            n_val = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    recon, mu, logvar, class_logits = model(batch_x)
                    loss, recon_loss, kl_loss, contrastive_loss, class_loss = vae_loss_function(
                        recon, batch_x, mu, logvar, class_logits, batch_y,
                        n_continuous=n_continuous, beta=beta, gamma=gamma,
                        margin=margin, free_bits=free_bits,
                    )
                    val_loss_sum += loss.item()
                    val_recon += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_class += class_loss.item()
                    val_contrastive += contrastive_loss.item()
                    n_val += 1

            avg_val = val_loss_sum / n_val
            scheduler.step(avg_val)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_metrics = {
                **train_metrics,
                "val_loss": avg_val,
                "recon": (train_metrics["recon"] + val_recon / n_val) / 2,
                "kl": (train_metrics["kl"] + val_kl / n_val) / 2,
                "class": (train_metrics["class"] + val_class / n_val) / 2,
                "contrastive": (train_metrics["contrastive"] + val_contrastive / n_val) / 2,
                "beta": beta,
            }
            self.monitor.record_epoch(epoch, model_id=0, metrics=epoch_metrics, lr=current_lr)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "n_continuous": n_continuous,
                    "n_binary": n_binary,
                    "latent_dim": latent_dim,
                    "epoch": epoch,
                    "val_loss": avg_val,
                }, checkpoint_path)
                self.monitor.set_best(0, epoch, avg_val, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.monitor.log(f"Early stopping at epoch {epoch}")
                    break

        self.monitor.log(f"Best val loss {best_val_loss:.4f} at epoch {best_epoch}")
        return checkpoint_path
