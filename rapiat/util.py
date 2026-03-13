import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D


def remove_all_bonds(mol):
    mol = Chem.RWMol(mol)
    for bond in reversed([b for b in mol.GetBonds()]):  # reverse to avoid index shifting
        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return mol


def assign_bonds_by_distance(mol, max_CH=1.2, max_CC=1.6):
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atoms = [a for a in mol.GetAtoms()]
    mol = Chem.RWMol(mol)
    existing = set(
        (min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in mol.GetBonds()
    )
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in existing:
                continue
            d = np.linalg.norm(positions[i] - positions[j])
            ai, aj = atoms[i], atoms[j]
            if (ai.GetSymbol(), aj.GetSymbol()) in [("C", "H"), ("H", "C")] and d < max_CH:
                mol.AddBond(i, j, Chem.BondType.SINGLE)
            elif ai.GetSymbol() == aj.GetSymbol() == "C" and d < max_CC:
                mol.AddBond(i, j, Chem.BondType.AROMATIC)
    # rdmolops.SanitizeMol(mol)
    return mol


def read_xyz_conformers(xyz_file, mol):
    with open(xyz_file) as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    n_confs = len(lines) // (n_atoms + 2)
    mol = Chem.AddHs(mol)
    for i in range(n_confs):
        block = lines[i * (n_atoms + 2) : (i + 1) * (n_atoms + 2)]
        conf = Chem.Conformer(mol.GetNumAtoms())
        for idx, line in enumerate(block[2 : 2 + n_atoms]):
            x, y, z = map(float, line.split()[1:4])
            conf.SetAtomPosition(idx, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)
    return mol
