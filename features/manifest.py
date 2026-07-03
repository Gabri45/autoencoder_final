"""
Signature manifest — serializes fixed atom pairs and dihedrals used by encoders.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class SignatureManifest:
    """Fixed geometric signature for a feature encoder."""

    pairs: list[list[int]] = field(default_factory=list)
    dihedrals: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    encoder: str = ""
    params: dict = field(default_factory=dict)

    @property
    def n_pair_features(self) -> int:
        return len(self.pairs)

    @property
    def n_dihedral_features(self) -> int:
        return len(self.dihedrals)

    def pairs_array(self):
        import numpy as np

        if not self.pairs:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(self.pairs, dtype=np.int32)

    def dihedrals_array(self):
        import numpy as np

        if not self.dihedrals:
            return np.zeros((0, 4), dtype=np.int32)
        return np.asarray(self.dihedrals, dtype=np.int32)


def save_manifest(manifest: SignatureManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_manifest(path: str | Path) -> SignatureManifest:
    with open(path) as f:
        data = json.load(f)
    return SignatureManifest(**data)
