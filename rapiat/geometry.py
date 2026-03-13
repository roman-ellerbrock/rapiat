import numpy as np
from rdkit import Chem
from scipy.optimize import quadratic_assignment


# Compute moment of inertia tensor for a single conformer (positions: shape (num_atoms, 3))
def moment_of_inertia_tensor(positions, masses):
    r2 = np.sum(positions**2, axis=1)
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I[i, i] = np.sum(masses * (r2 - positions[:, i] ** 2))
            else:
                I[i, j] = -np.sum(masses * positions[:, i] * positions[:, j])
    return I


def set_conformer_positions(mol, positions_array):
    for conf_id, pos in enumerate(positions_array):
        conf = mol.GetConformer(conf_id)
        for atom_idx in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(atom_idx, pos[atom_idx])


def get_conformer_positions(mol) -> np.array:
    positions = np.array(
        [mol.GetConformer(i).GetPositions() for i in range(mol.GetNumConformers())]
    )
    return positions


def distance_matrix(positions):
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    return D


def inverse_distance_matrix(positions, diag_value=0.0):
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    D = np.linalg.norm(diff, axis=-1)
    with np.errstate(divide="ignore"):
        inv_D = 1.0 / D
    for i in range(inv_D.shape[0]):
        np.fill_diagonal(inv_D[i], diag_value)
    return inv_D


def featurize_molecules(mol: Chem.Mol) -> np.ndarray:
    """
    Featurize a molecule into a 2D array of atom features.
    Each row corresponds to a conformer, and each column corresponds to a feature.
    """
    X = get_conformer_positions(mol)
    R = inverse_distance_matrix(X)
    Ravg = np.mean(R, axis=0)
    features = []
    for i in range(R.shape[0]):
        ev, _ = np.linalg.eigh(R[i] - Ravg)
        features.append(ev)
    return np.array(features)


def align_permutations(R1, R2, mol, rng=np.random.default_rng()) -> np.ndarray:
    """
    Aligns the second molecules matrix to the first by finding the optimal permutation of atoms

    R1: inverse distance matrix of the first molecule
    R2: inverse distance matrix of the second molecule
    mol: RDKit molecule object

    """
    R2 = R2.copy()

    atomic_symbols = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    elements = list(set(atomic_symbols))
    P = np.arange(R2.shape[0])
    for element in elements:
        fixed_atoms = np.where(atomic_symbols != element)[0]
        atom_ids = np.where(atomic_symbols == element)[0]
        # print(f"Aligning element: {element}, atom ids: {atom_ids}")

        partial_match = np.array([[i, i] for i in fixed_atoms])

        result = quadratic_assignment(
            R1,
            R2,
            options={"maximize": True, "partial_match": partial_match, "rng": rng},
            method="2opt",
        )
        perm = result["col_ind"]
        # print(f"Permutation for element {element}: {perm[atom_ids]}\n")
        P[atom_ids] = perm[atom_ids]
        R2 = R2[perm, :][:, perm]

    return R2, P


def sample_align_permutations(R1, R2, mol, nsample, rng):
    R2 = R2.copy()
    delta_best = np.linalg.norm(R1 - R2)
    R_best = R2.copy()
    P_best = np.arange(R2.shape[0])

    for _ in range(nsample):
        Rtry, P = align_permutations(R1, R2, mol, rng)
        delta = np.linalg.norm(R1 - Rtry)
        if delta < delta_best:
            print(_, delta, delta_best)
            R_best = Rtry.copy()
            R2 = Rtry
            delta_best = delta
            P_best
        if delta < 1e-6:
            break

    return R_best, P_best


def permute_R(R1, R2, mol):
    masses = np.array([atom.GetMass() for atom in mol.GetAtoms()])
    # multiply row/cols by sum of masses
    M = np.outer(masses, masses)
    M = M / np.mean(masses) ** 2  # normalize by total mass
    R1m = R1 * M
    R2m = R2 * M
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # R1m[i, j] *= 10
        # R1m[j, i] *= 10
        # R2m[i, j] *= 10
        # R2m[j, i] *= 10

    result = quadratic_assignment(R1m, R2m, options={"maximize": True}, method="2opt")
    P = result["col_ind"]
    R2 = R2[P, :][:, P]
    return R2, P
