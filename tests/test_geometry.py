"""Tests for geometry module."""

import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rapiat import geometry


def test_align_permutations():
    """Test that permutation alignment recovers the original distance matrix."""
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMultipleConfs(mol, numConfs=1)
    mol = Chem.RemoveHs(mol)

    X = geometry.get_conformer_positions(mol)
    R = geometry.inverse_distance_matrix(X)

    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    elements = list(set(atomic_symbols))

    # Create a random permutation of the inverse distance matrix
    Rp = R[0].copy()
    random.seed(42)
    Pref = np.arange(len(atomic_symbols))
    for element in elements:
        atom_ids = np.where(atomic_symbols == element)[0]
        P = atom_ids.copy()
        random.shuffle(P)
        Pref[atom_ids] = P
    Rp = Rp[Pref][:, Pref]

    initial_error = np.linalg.norm(R[0] - Rp)
    R_best, _ = geometry.sample_align_permutations(
        R[0], Rp, mol, 100, rng=np.random.default_rng(42)
    )
    final_error = np.linalg.norm(R[0] - R_best)

    assert final_error < 1e-1 * initial_error
