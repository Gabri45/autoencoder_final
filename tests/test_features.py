"""Tests for generalized feature encoders."""

from __future__ import annotations

import unittest

import mdtraj as md

from features import cmap, composite, rna_g4
from features.registry import get_encoder

EXAMPLE_PDB = "/home/gabriele/sims_tar/ExampleStruct_A.pdb"


@unittest.skipUnless(__import__("os").path.isfile(EXAMPLE_PDB), "example PDB missing")
class TestCompositeEncoder(unittest.TestCase):
    def setUp(self):
        self.traj = md.load(EXAMPLE_PDB)

    def test_g4_preset_matches_legacy(self):
        legacy, ml = get_encoder("rna_g4_enriched").compute(self.traj)
        preset, mp = get_encoder(
            "composite",
            preset="rna_g4_enriched",
            reference_pdb=EXAMPLE_PDB,
        ).compute(self.traj)
        self.assertEqual(legacy.shape, preset.shape)
        self.assertEqual(ml.n_continuous, mp.n_continuous)
        self.assertEqual(ml.n_binary, mp.n_binary)

    def test_cmap_top_k_dimension(self):
        features, meta = get_encoder(
            "cmap",
            reference_pdb=EXAMPLE_PDB,
            top_k=50,
            selection="backbone",
        ).compute(self.traj)
        self.assertEqual(features.shape[1], 100)  # 50 continuous + 50 binary
        self.assertEqual(meta.n_continuous, 50)
        self.assertEqual(meta.n_binary, 50)


if __name__ == "__main__":
    unittest.main()
