"""
Chemical templates for distance pairs and dihedrals.

Templates map topology to fixed atom-index lists using residue names.
"""

from __future__ import annotations

import itertools

import mdtraj as md
import numpy as np

PURINE_NAMES = frozenset({"A", "G", "DA", "DG", "ADE", "GUA", "G5", "A5", "RA", "RG"})
PYRIMIDINE_NAMES = frozenset({"C", "U", "DC", "DT", "CYT", "URA", "C3", "U3", "RC", "RU"})
GUANINE_NAMES = frozenset({"G", "DG", "GUA", "DGU", "G5", "RG"})


def _base_class(resname: str) -> str | None:
    r = resname.strip().upper()
    if r in PURINE_NAMES:
        return "purine"
    if r in PYRIMIDINE_NAMES:
        return "pyrimidine"
    if r.startswith("G"):
        return "purine"
    if r in ("A", "C", "U"):
        return {"A": "purine", "C": "pyrimidine", "U": "pyrimidine"}[r]
    return None


def _find_atom(residue, names: tuple[str, ...]):
    for name in names:
        for atom in residue.atoms:
            if atom.name == name:
                return atom.index
    return None


def build_pairs_from_template(traj: md.Trajectory, template: str) -> tuple[np.ndarray, list[str]]:
    """
    Build distance pair signature from a named chemical template.

    Templates:
        g4_hoogsteen — all G×G N1-O6 and N2-N7 (permutations)
        watson_crick — WC/wobble atom pairs for all qualifying residue pairs
        all_base_pairs — centroid-proxied N1/N9 pairs for all base×base (i<j)
    """
    topology = traj.topology
    builders = {
        "g4_hoogsteen": _pairs_g4_hoogsteen,
        "watson_crick": _pairs_watson_crick,
        "all_base_pairs": _pairs_all_base_nonlocal,
    }
    if template not in builders:
        raise ValueError(f"Unknown pair template {template!r}. Available: {list(builders)}")
    return builders[template](topology)


def _pairs_g4_hoogsteen(topology) -> tuple[np.ndarray, list[str]]:
    guanines = [r for r in topology.residues if r.name in GUANINE_NAMES or r.name.strip().upper().startswith("G")]
    guanines.sort(key=lambda x: x.index)
    pairs, labels = [], []
    for g1, g2 in itertools.permutations(guanines, 2):
        n1 = _find_atom(g1, ("N1",))
        o6 = _find_atom(g2, ("O6",))
        if n1 is not None and o6 is not None:
            pairs.append([n1, o6])
            labels.append(f"G{g1.index+1}_N1-G{g2.index+1}_O6")
        n2 = _find_atom(g1, ("N2",))
        n7 = _find_atom(g2, ("N7",))
        if n2 is not None and n7 is not None:
            pairs.append([n2, n7])
            labels.append(f"G{g1.index+1}_N2-G{g2.index+1}_N7")
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32), []
    return np.asarray(pairs, dtype=np.int32), labels


def _wc_atom_pairs(res_i, res_j, name_i: str, name_j: str) -> tuple[list, list]:
    pairs, labels = [], []
    a = _find_atom(res_i, (name_i,))
    b = _find_atom(res_j, (name_j,))
    if a is not None and b is not None:
        pairs.append([a, b])
        labels.append(f"r{res_i.index+1}_{name_i}-r{res_j.index+1}_{name_j}")
    return pairs, labels


def _pairs_watson_crick(topology) -> tuple[np.ndarray, list[str]]:
    """RNA WC / wobble donor-acceptor pairs for all residue combinations."""
    residues = list(topology.residues)
    pairs, labels = [], []

    def res_type(r):
        n = r.name.strip().upper()
        if n in ("G", "DG", "GUA", "G5", "RG") or (n.startswith("G") and "C" not in n[:2]):
            return "G"
        if n in ("A", "DA", "ADE", "A5", "RA") or n.startswith("A"):
            return "A"
        if n in ("C", "DC", "CYT", "C3", "RC") or n.startswith("C"):
            return "C"
        if n in ("U", "DT", "URA", "U3", "RU") or n.startswith("U"):
            return "U"
        return None

    for i, ri in enumerate(residues):
        ti = res_type(ri)
        if ti is None:
            continue
        for rj in residues[i + 1 :]:
            tj = res_type(rj)
            if tj is None:
                continue
            key = frozenset((ti, tj))
            if key == frozenset(("G", "C")):
                for a, b, la, lb in [("O6", "N4", "O6", "N4"), ("N1", "N3", "N1", "N3"), ("N2", "O2", "N2", "O2")]:
                    p, l = _wc_atom_pairs(ri, rj, a, b)
                    pairs.extend(p)
                    labels.extend(l)
                    p, l = _wc_atom_pairs(rj, ri, a, b)
                    pairs.extend(p)
                    labels.extend(l)
            elif key == frozenset(("A", "U")):
                for a, b in [("N1", "N3"), ("N6", "O4")]:
                    p, l = _wc_atom_pairs(ri, rj, a, b)
                    pairs.extend(p)
                    labels.extend(l)
                    p, l = _wc_atom_pairs(rj, ri, a, b)
                    pairs.extend(p)
                    labels.extend(l)
            elif key == frozenset(("G", "U")):
                for a, b in [("N1", "O2"), ("O6", "N3"), ("N2", "O2")]:
                    p, l = _wc_atom_pairs(ri, rj, a, b)
                    pairs.extend(p)
                    labels.extend(l)
                    p, l = _wc_atom_pairs(rj, ri, a, b)
                    pairs.extend(p)
                    labels.extend(l)

    if not pairs:
        return np.zeros((0, 2), dtype=np.int32), []
    return np.asarray(pairs, dtype=np.int32), labels


def _pairs_all_base_nonlocal(topology, min_residues_apart: int = 2) -> tuple[np.ndarray, list[str]]:
    """N1 (pyrimidine) or N9 (purine) distance for non-adjacent bases."""
    residues = [r for r in topology.residues if _base_class(r.name)]
    pairs, labels = [], []
    for i, ri in enumerate(residues):
        ai = _find_atom(ri, ("N9",)) or _find_atom(ri, ("N1",))
        if ai is None:
            continue
        for rj in residues[i + 1 :]:
            if abs(rj.index - ri.index) < min_residues_apart:
                continue
            aj = _find_atom(rj, ("N9",)) or _find_atom(rj, ("N1",))
            if aj is None:
                continue
            pairs.append([ai, aj])
            labels.append(f"base_{ri.index+1}_{rj.index+1}")
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32), []
    return np.asarray(pairs, dtype=np.int32), labels


def build_dihedrals_from_template(traj: md.Trajectory, template: str) -> tuple[np.ndarray, list[str]]:
    """
    Templates:
        purine_chi — O4'-C1'-N9-C4 on purines
        guanine_chi — same, only guanines
        pyrimidine_chi — O4'-C1'-N1-C2 on pyrimidines
        all_nucleotide_chi — purine + pyrimidine chi
    """
    topology = traj.topology
    builders = {
        "purine_chi": lambda t: _chi_purines(t, guanine_only=False),
        "guanine_chi": lambda t: _chi_purines(t, guanine_only=True),
        "pyrimidine_chi": _chi_pyrimidines,
        "all_nucleotide_chi": _chi_all,
    }
    if template not in builders:
        raise ValueError(f"Unknown dihedral template {template!r}. Available: {list(builders)}")
    return builders[template](topology)


def _chi_purines(topology, guanine_only: bool = False) -> tuple[np.ndarray, list[str]]:
    names = GUANINE_NAMES if guanine_only else PURINE_NAMES
    dihedrals, labels = [], []
    for residue in topology.residues:
        if residue.name not in names and not (
            guanine_only and residue.name.strip().upper().startswith("G")
        ):
            if not guanine_only and residue.name.strip().upper().startswith("A"):
                pass
            elif residue.name not in names:
                continue
        o4 = _find_atom(residue, ("O4'", "O4*"))
        c1 = _find_atom(residue, ("C1'", "C1*"))
        n9 = _find_atom(residue, ("N9",))
        c4 = _find_atom(residue, ("C4",))
        if all(x is not None for x in (o4, c1, n9, c4)):
            dihedrals.append([o4, c1, n9, c4])
            labels.append(f"chi_purine_r{residue.index+1}")
    if not dihedrals:
        return np.zeros((0, 4), dtype=np.int32), []
    return np.asarray(dihedrals, dtype=np.int32), labels


def _chi_pyrimidines(topology) -> tuple[np.ndarray, list[str]]:
    dihedrals, labels = [], []
    for residue in topology.residues:
        bc = _base_class(residue.name)
        if bc != "pyrimidine":
            continue
        o4 = _find_atom(residue, ("O4'", "O4*"))
        c1 = _find_atom(residue, ("C1'", "C1*"))
        n1 = _find_atom(residue, ("N1",))
        c2 = _find_atom(residue, ("C2",))
        if all(x is not None for x in (o4, c1, n1, c2)):
            dihedrals.append([o4, c1, n1, c2])
            labels.append(f"chi_pyr_r{residue.index+1}")
    if not dihedrals:
        return np.zeros((0, 4), dtype=np.int32), []
    return np.asarray(dihedrals, dtype=np.int32), labels


def _chi_all(topology) -> tuple[np.ndarray, list[str]]:
    d1, l1 = _chi_purines(topology, guanine_only=False)
    d2, l2 = _chi_pyrimidines(topology)
    if len(d1) == 0:
        return d2, l2
    if len(d2) == 0:
        return d1, l1
    return np.vstack([d1, d2]), l1 + l2
