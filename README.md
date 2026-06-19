# autoencoder_final

Train and evaluate a semi-supervised VAE on molecular trajectories.  
Everything is driven by a YAML config file.

## Folder structure

```
autoencoder_final/
├── train.py              # Training entry point
├── evaluate.py           # Evaluation entry point
├── configs/              # YAML configuration files
├── slurm/                # SLURM job template
├── outputs/              # Results (checkpoints, plots, logs)
│
├── config/               # YAML loader
├── features/             # Pluggable feature encoders (see features/GUIDE.md)
├── model/                # VAE architecture + loss
├── training/             # Trainers + training monitor
├── evaluation/           # Artifact loading, encoding, plots
└── utils/                # Trajectories, Q-score, free energy
```

## Quick start

```bash
cd autoencoder_final
```

### 1. Edit the config

Open a file in `configs/` and set your paths:

```yaml
project:
  output_dir: ./outputs/rmd_hn_ensemble

data:
  data_dir: /path/to/trajectories
  conf_dir: /path/to/reference_pdbs
```

### 2. Train

```bash
# Ensemble (5 models) — default for production
python -u train.py --config configs/train_ensemble.yaml

# Single model
python -u train.py --config configs/train_single.yaml
```

Override values without editing the file:

```bash
python -u train.py --config configs/train_ensemble.yaml --override training.epochs=50
```

**Training outputs** (in `project.output_dir`):

| File | Description |
|------|-------------|
| `ensemble_model_*.pt` / `vae_model.pt` | Model checkpoints |
| `ensemble_scaler.pkl` / `scaler.pkl` | Feature scaler |
| `features_mixed.npy` | Cached feature matrix |
| `training_log.txt` | Epoch-by-epoch log (SLURM-friendly) |
| `training_history.csv` | Loss history |
| `training_loss.png` | Loss curves |

### 3. Evaluate

Run **after** training, pointing to the same `output_dir`:

```bash
python evaluate.py --config configs/train_ensemble.yaml
```

Evaluation auto-detects single vs ensemble from the checkpoints in `output_dir`.  
Plots to generate are listed under `evaluation.plots` in the YAML.

Disable a plot:

```bash
python evaluate.py --config configs/train_ensemble.yaml --override evaluation.plots.hamming=false
```

### 4. SLURM

```bash
sbatch slurm/train_ensemble.slurm
```

## Config files

| File | Purpose |
|------|---------|
| `configs/train_ensemble.yaml` | Train ensemble of 5 models |
| `configs/train_single.yaml` | Train a single model |
| `configs/default_rna.yaml` | Full config (train + eval settings) |

Key YAML sections:

| Section | Controls |
|---------|----------|
| `project.output_dir` | Where checkpoints and plots are saved |
| `data` | Trajectory paths, stride, feature cache |
| `features` | Which encoder to use (`rna_g4_enriched` by default) |
| `training` | Mode (`single` / `ensemble`), epochs, loss params |
| `monitoring` | Log frequency, loss plots |
| `evaluation.plots` | Which standard plots to generate |
| `evaluation.by_target` | Per-target landscapes on the training dataset |
| `evaluation.new_dataset` | Project a new dataset into the trained latent space |

## Standard evaluation

```bash
python evaluate.py --config configs/default_rna.yaml
```

Plots are controlled by `evaluation.plots` in the YAML.

## By-target evaluation

Analyzes trajectories grouped by folding target (143D, 1KF1, 2HY9, 2JPZ, NEVER_FOLDED) using the latent coordinates from the training dataset.

```yaml
evaluation:
  by_target:
    enabled: true
    folding_report: /path/to/folding_by_target.txt
```

**Outputs** (in `output_dir`):
- `vae_target_landscape_{TARGET}_reference_ensemble_classic.png`
- `vae_target_trajectories_grid_{TARGET}.png`
- `exploration_areas_by_target.txt`

```bash
python evaluate.py --config configs/default_rna.yaml --override evaluation.by_target.enabled=true
```

## New dataset evaluation

Projects trajectories from a **different** simulation into the trained ensemble.

### Quick mode (no folding report)

```yaml
evaluation:
  new_dataset:
    enabled: true
    data_dir: /path/to/new/trajectories
    per_target: false
    plot_latent_overview: true
```

```bash
python evaluate.py --config configs/eval_new_quick.yaml
```

**Output:** `outputs/.../new_dataset/vae_new_dataset_latent_overview.png` + `ensemble_mean_latent_new.npy`

### Full mode (with folding report)

Add `folding_report` (or set `per_target: true`) for per-target landscapes and grids.

```yaml
evaluation:
  new_dataset:
    enabled: true
    data_dir: /path/to/new/trajectories
    folding_report: /path/to/folding_by_target.txt
    per_target: true
```

```bash
python evaluate.py --config configs/eval_new_dataset.yaml
```

## Typical workflow

```
edit configs/  →  train.py  →  evaluate.py
                      ↓              ↓
                 checkpoints      plots + .npy
```

Training and evaluation are **independent**: you can re-run `evaluate.py` on existing checkpoints without retraining.

## Adding a custom feature encoder

See [`features/GUIDE.md`](features/GUIDE.md).
