# Adding a new encoder

## Generalized encoders (recommended)

For any molecular system, use the **composite** or **cmap** encoder via YAML — no new Python file required.

### Composite (`features.encoder: composite`)

```yaml
features:
  encoder: composite
  params:
    reference_pdb: /path/to/reference.pdb   # optional: auto from data.conf_dir/*.pdb
    preset: rna_general                     # or rna_g4_enriched
  cache_prefix: features_composite
```

**Presets:**

| Preset | Blocks |
|--------|--------|
| `rna_g4_enriched` | G4 H-bond distances + guanine χ + binary (same as legacy `rna_g4_enriched`) |
| `rna_general` | Watson–Crick distances + all χ + CMAP top-500 + binaries |

**Custom blocks:**

```yaml
features:
  encoder: composite
  params:
    blocks:
      - type: distance_pairs
        template: watson_crick          # g4_hoogsteen | watson_crick | all_base_pairs
        transform: raw_distance         # raw_distance | sigmoid_squared
        binary_cutoff_nm: 0.35
        spatial_filter_nm: 0.45         # optional: keep pairs near in reference
        top_k: 200                      # optional: limit pair count

      - type: dihedral
        template: all_nucleotide_chi    # purine_chi | guanine_chi | pyrimidine_chi

      - type: cmap
        selection: backbone             # backbone | heavy | MDTraj expression
        signature_mode: top_k             # top_k | threshold | all
        top_k: 500
        binary_cutoff: 0.6
```

### CMAP only (`features.encoder: cmap`)

```yaml
features:
  encoder: cmap
  params:
    selection: backbone
    signature_mode: top_k
    top_k: 500
    binary_cutoff: 0.6
```

### Cache outputs

| File | Content |
|------|---------|
| `{prefix}.npy` | Feature matrix |
| `{prefix}_meta.json` | `n_continuous`, `n_binary`, encoder info |
| `{prefix}_signature.json` | Fixed atom pairs / dihedrals |

See `configs/train_composite.yaml` and `configs/train_cmap.yaml`.

---

## Custom encoder (escape hatch)

### How it works (4 files)

| File | Role |
|------|------|
| `base.py` | Contract: `FeatureEncoder.compute()` → `(features, FeatureMeta)` |
| `registry.py` | `@register_encoder("name")` + `get_encoder(name, **params)` |
| `pipeline.py` | Loads trajectories, calls `get_encoder(...)`, saves cache |
| `rna_g4.py` | Reference example |

The pipeline **does not need to be modified**. It always calls:

```python
encoder = get_encoder(feat_cfg["encoder"], **feat_cfg.get("params", {}))
features, meta = encoder.compute(traj)
n_continuous = meta.n_continuous
n_binary = meta.n_binary
```

---

## Steps

### 1. Create `features/my_encoder.py`

Copy the structure of `rna_g4.py`:

```python
"""
Short description of the encoder.

YAML: features.encoder=my_encoder, features.params.<param>
"""

from __future__ import annotations

import mdtraj as md
import numpy as np

from features.base import FeatureEncoder, FeatureMeta
from features.registry import register_encoder


def compute_my_features(traj, cutoff: float = 0.35):
    """
    Helper function: compute raw features.
    Returns:
        features: np.ndarray (n_frames, n_features), or None on failure
    """
    # ... computation with mdtraj ...
    features = ...  # float32 recommended
    return features


@register_encoder("my_encoder")
class MyEncoder(FeatureEncoder):
    """Short description."""

    name = "my_encoder"

    def __init__(self, cutoff: float = 0.35):
        self.cutoff = cutoff

    def compute(self, traj) -> tuple[np.ndarray, FeatureMeta]:
        result = compute_my_features(traj, cutoff=self.cutoff)
        if result is None:
            raise ValueError("Could not compute features from trajectory")
        features = result
        n_continuous = features.shape[1]
        n_binary = 0
        return features, FeatureMeta(n_continuous=n_continuous, n_binary=n_binary)

    def describe(self) -> dict:
        info = super().describe()
        info["cutoff"] = self.cutoff
        return info
```

### 2. Register the import in `train.py`

Same pattern as `rna_g4`:

```python
from features import rna_g4  # noqa: F401 — register encoder
from features import my_encoder  # noqa: F401 — register encoder
```

Without this import, `get_encoder()` will not find the name.

### 3. Configure the YAML

```yaml
features:
  encoder: my_encoder
  params:
    cutoff: 0.35
  cache_prefix: features_mixed
```

`params` → passed to the class `__init__` (same pattern as `hbond_cutoff` in `rna_g4.py`).

To recompute features (new encoder or new data):

```yaml
data:
  recompute_features: true
```

---

## `n_continuous` vs `n_binary`

The feature vector is **a single matrix** `(n_frames, n_features)` with two trailing blocks:

```text
[ ... continuous ... | ... binary ... ]
```

| Type | Treatment | Example in `rna_g4.py` |
|------|-----------|------------------------|
| **continuous** | `StandardScaler` during training | distances, sin(χ), cos(χ) |
| **binary** | no scaling; BCE + contrastive loss | `(distances < hbond_cutoff)` |

**Continuous only** — as in the example above:

```python
n_continuous = features.shape[1]
n_binary = 0
return features, FeatureMeta(n_continuous=n_continuous, n_binary=n_binary)
```

**Continuous + binary** — as in `rna_g4.py`:

```python
binary_features = (distances < hbond_cutoff).astype(np.float32)
features = np.concatenate([distances, sin_chi, cos_chi, binary_features], axis=1)

n_binary = binary_features.shape[1]
n_continuous = features.shape[1] - n_binary

return features, FeatureMeta(n_continuous=n_continuous, n_binary=n_binary)
```

Rule: columns `[0 : n_continuous]` are continuous, `[n_continuous :]` are binary.

---

## Reference: `rna_g4.py`

```python
@register_encoder("rna_g4_enriched")
class RNAG4EnrichedEncoder(FeatureEncoder):

    name = "rna_g4_enriched"

    def __init__(self, hbond_cutoff: float = 0.35):
        self.hbond_cutoff = hbond_cutoff

    def compute(self, traj) -> tuple[np.ndarray, FeatureMeta]:
        result = compute_features_enriched(traj, hbond_cutoff=self.hbond_cutoff)
        if result is None:
            raise ValueError("Could not compute G4 enriched features from trajectory")
        features, n_continuous, n_binary = result
        return features, FeatureMeta(n_continuous=n_continuous, n_binary=n_binary)

    def describe(self) -> dict:
        info = super().describe()
        info["hbond_cutoff"] = self.hbond_cutoff
        return info
```

---

## Checklist

- [ ] Class inherits from `FeatureEncoder`
- [ ] `@register_encoder("unique_name")` above the class
- [ ] `compute()` returns `tuple[np.ndarray, FeatureMeta]`
- [ ] `n_continuous + n_binary == features.shape[1]`
- [ ] Import added in `train.py`
- [ ] `features.encoder` updated in YAML
- [ ] `recompute_features: true` on first run with the new encoder

## Quick check

```bash
python -c "from features import my_encoder; from features.registry import list_encoders; print(list_encoders())"
# must include 'my_encoder'
```
