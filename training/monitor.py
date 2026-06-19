"""
Training monitor for SLURM-friendly logging.

Writes to output_dir:
  training_log.txt, training_history.csv, training_loss.png,
  config_resolved.yaml, training_summary.json

YAML: monitoring.log_every_epochs, monitoring.plot_every_epochs,
      monitoring.save_csv, monitoring.flush_stdout
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt

from config.loader import save_config


class TrainingMonitor:
    """Structured training logger with CSV, text log, and loss plots."""

    CSV_COLUMNS = [
        "epoch", "model_id", "train_loss", "val_loss",
        "recon", "kl", "class", "contrastive", "beta", "lr",
    ]

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config["project"]["output_dir"]
        self.mon_cfg = config.get("monitoring", {})
        self.flush_stdout = self.mon_cfg.get("flush_stdout", True)
        self.plot_every = self.mon_cfg.get("plot_every_epochs", 10)
        self.log_every = self.mon_cfg.get("log_every_epochs", 1)
        self.save_csv = self.mon_cfg.get("save_csv", True)

        os.makedirs(self.output_dir, exist_ok=True)

        self.log_path = os.path.join(self.output_dir, "training_log.txt")
        self.csv_path = os.path.join(self.output_dir, "training_history.csv")
        self.plot_path = os.path.join(self.output_dir, "training_loss.png")
        self.summary_path = os.path.join(self.output_dir, "training_summary.json")
        self.config_path = os.path.join(self.output_dir, "config_resolved.yaml")

        save_config(config, self.config_path)

        self._history: list[dict[str, Any]] = []
        self._per_model: dict[int, dict[str, list]] = {}
        self._start_time = time.time()
        self._best: dict[str, Any] = {}

        if self.save_csv:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()

        self.log("TrainingMonitor initialized")
        self.log(f"Output directory: {self.output_dir}")

    def log(self, message: str) -> None:
        """Append to training_log.txt and optionally flush stdout."""
        line = f"{message}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        print(message)
        if self.flush_stdout:
            sys.stdout.flush()

    def record_epoch(
        self,
        epoch: int,
        model_id: int,
        metrics: dict[str, float],
        lr: float,
    ) -> None:
        """Record one epoch of metrics."""
        row = {
            "epoch": epoch,
            "model_id": model_id,
            "train_loss": metrics.get("train_loss", 0),
            "val_loss": metrics.get("val_loss", 0),
            "recon": metrics.get("recon", 0),
            "kl": metrics.get("kl", 0),
            "class": metrics.get("class", 0),
            "contrastive": metrics.get("contrastive", 0),
            "beta": metrics.get("beta", 0),
            "lr": lr,
        }
        self._history.append(row)

        if model_id not in self._per_model:
            self._per_model[model_id] = {
                "train_loss": [], "val_loss": [], "recon": [], "kl": [],
                "class": [], "contrastive": [], "beta": [],
            }
        for key in ("train_loss", "val_loss", "recon", "kl", "class", "contrastive", "beta"):
            self._per_model[model_id][key].append(metrics.get(key, 0))

        if self.save_csv:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
                writer.writerow(row)

        if epoch % self.log_every == 0 or epoch == 1:
            self.log(
                f"[model {model_id}] Epoch {epoch:4d}: "
                f"Train={row['train_loss']:.4f}, Val={row['val_loss']:.4f}, "
                f"β={row['beta']:.3f}"
            )

        if epoch % self.plot_every == 0:
            self.plot_losses()

    def set_best(self, model_id: int, epoch: int, val_loss: float, checkpoint: str) -> None:
        """Track best checkpoint for a model."""
        key = str(model_id)
        prev = self._best.get(key)
        if prev is None or val_loss < prev["val_loss"]:
            self._best[key] = {
                "model_id": model_id,
                "best_epoch": epoch,
                "val_loss": val_loss,
                "checkpoint": checkpoint,
            }

    def plot_losses(self) -> None:
        """Update training_loss.png (single: 3 panels; ensemble: per-model + mean)."""
        n_models = len(self._per_model)
        if n_models == 0:
            return

        if n_models == 1:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            hist = next(iter(self._per_model.values()))
            best_key = "0"
            best_epoch = self._best.get(best_key, {}).get("best_epoch", 1)

            axes[0].plot(hist["train_loss"], label="Train", alpha=0.8)
            axes[0].plot(hist["val_loss"], label="Validation", alpha=0.8)
            axes[0].axvline(best_epoch - 1, color="red", linestyle="--", alpha=0.5)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Total Loss")
            axes[0].set_title("Total Loss")
            axes[0].legend()
            axes[0].set_yscale("log")

            axes[1].plot(hist["recon"], label="Train Recon", alpha=0.8)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Reconstruction Loss")
            axes[1].set_title("Reconstruction Loss")
            axes[1].set_yscale("log")

            axes[2].plot(hist["kl"], label="KL", alpha=0.8)
            axes[2].plot(hist["class"], label="Class", alpha=0.8, linestyle=":")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Loss")
            axes[2].set_title("KL & Classification")
            axes[2].legend()
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            for mid, hist in sorted(self._per_model.items()):
                axes[0].plot(hist["val_loss"], label=f"Model {mid}", alpha=0.7)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Validation Loss")
            axes[0].set_title("Ensemble Validation Loss per Model")
            axes[0].legend()
            axes[0].set_yscale("log")

            mean_val = []
            max_len = max(len(h["val_loss"]) for h in self._per_model.values())
            for i in range(max_len):
                vals = [h["val_loss"][i] for h in self._per_model.values() if i < len(h["val_loss"])]
                if vals:
                    mean_val.append(sum(vals) / len(vals))
            axes[1].plot(mean_val, color="black", linewidth=2, label="Mean val loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Mean Validation Loss")
            axes[1].set_title("Ensemble Mean Validation Loss")
            axes[1].legend()
            axes[1].set_yscale("log")

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=200, bbox_inches="tight")
        plt.close()

    def finalize(self, mode: str, checkpoint_paths: list[str]) -> dict:
        """Write training_summary.json and return summary dict."""
        elapsed = time.time() - self._start_time
        summary = {
            "mode": mode,
            "elapsed_seconds": elapsed,
            "best_models": self._best,
            "checkpoints": checkpoint_paths,
            "output_dir": self.output_dir,
        }
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.plot_losses()
        self.log(f"Training complete in {elapsed:.1f}s")
        self.log(f"Summary saved to {self.summary_path}")
        return summary
