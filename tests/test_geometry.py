import random

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from rapiat import geometry


def test_align_permutations():
    biliverdin = (
        "CC1=C(CCC(=O)N1)C2=CC(=C(N2)C3=CC(=C(C(=N3)C4=CC(=C(C(=N4)C(=O)O)CCC(=O)O)C)C)CCC(=O)O)C"
    )
    mol = Chem.AddHs(Chem.MolFromSmiles(biliverdin))
    AllChem.EmbedMultipleConfs(mol, numConfs=1)
    mol = Chem.RemoveHs(mol)  # remove hydrogens for distance calculations

    X = geometry.get_conformer_positions(mol)
    R = geometry.inverse_distance_matrix(X)

    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    elements = list(set(atomic_symbols))

    print("\n")
    # Create a random permutation of the inverse distance matrix
    Rp = R[0].copy()
    random.seed(42)
    Pref = np.arange(len(atomic_symbols))
    for element in elements:
        atom_ids = np.where(atomic_symbols == element)[0]
        # if element == 'C':
        # continue
        P = atom_ids.copy()
        random.shuffle(P)
        print(f"Permuted element: {element}, {atom_ids}, permutation: {P}")
        Pref[atom_ids] = P
    print(f"Final permutation: {Pref}\n\n")
    Rp = Rp[Pref][:, Pref]

    R_best, P_best = geometry.sample_align_permutations(
        R[0], Rp, mol, 100, rng=np.random.default_rng(42)
    )
    assert np.linalg.norm(R[0] - R_best) < 1e-10
