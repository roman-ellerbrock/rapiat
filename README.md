# rapiat

[![License](https://img.shields.io/github/license/roman-ellerbrock/rapiat)](https://github.com/roman-ellerbrock/rapiat/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/roman-ellerbrock/rapiat/test.yml?branch=master&logo=github-actions)](https://github.com/roman-ellerbrock/rapiat/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/roman-ellerbrock/rapiat)](https://codecov.io/gh/roman-ellerbrock/rapiat)

Clustering of molecular conformers based on inverse distance matrix features, with interactive 3D visualization via py3Dmol.

## Overview

rapiat generates conformer ensembles (via RDKit) or reads them from XYZ trajectory files, computes inverse distance matrix representations, and clusters the conformers using dimensionality reduction (t-SNE, UMAP) combined with k-means++. Each cluster is visualized as an overlaid 3D structure in py3Dmol, making it easy to inspect the distinct conformational families of a molecule.

The repository ships example XYZ datasets of minimum-energy conical intersection (MECI) seam geometries for several molecules (benzene, ethylene, CPO, DMPCO, HBDI) in `data/`.

## Scripts

The main entry points are Jupyter notebooks in `scripts/`:

- **`clustering.ipynb`** — Conformer generation from SMILES, inverse distance featurization, t-SNE/UMAP embedding, k-means++ clustering, and per-cluster 3D visualization.
- **`meci_clustering.ipynb`** — Same workflow applied to MECI seam geometries read from XYZ files, with mass-weighted distance matrices and SVD-based features.

## Installation

Requires [Pixi](https://pixi.sh):

```bash
pixi install
```

## Usage

```bash
pixi run jupyter lab
```

Then open one of the notebooks in `scripts/`.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
