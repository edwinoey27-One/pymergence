# PyMergence

A Python toolkit for calculating and visualizing causal emergence in complex systems.

## Overview

PyMergence provides a comprehensive set of tools for analyzing causal emergence phenomena, including implementations of measures from the CE2.0 framework and visualization capabilities for understanding emergent causal structures.

## Features

- **Causal Emergence Measures**: Implementation of single path measures from the CE2.0 paper
- **Visualization Tools**: Create and analyze Hasse diagrams and causal structures
- **Coarse-Graining**: Tools for systematic coarse-graining of complex systems
- **Stochastic Analysis**: Support for stochastic matrices and probabilistic measures

## Installation Instructions

[to finalize, should probably be replaced with a proper PyPI link once available]

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Matplotlib ≥ 3.3.0
- NetworkX ≥ 3.4.0
- PyGraphviz ≥ 1.14

### Install from source

```bash
git clone https://github.com/yourusername/pymergence.git
cd pymergence
pip install -e .
```

## Quick Start

An interactive Jupyter notebook that shows how to get started with PyMergence is available in the `Tutorial.ipynb` notebook.
To recreate the figures from the CE2.0 paper, you can run the `recreate_CE2_plots.ipynb` notebook.

## Documentation

Documentation is available at `pymergence/docs/_build/html/index.html`.


## Testing

Run the test suite using pytest:

```bash
python -m pytest tests
```

## License
[To discuss: which license to use?]
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyMergence in your research, please cite:

```bibtex
@article{2025causalEmergence,
  title={Causal Emergence 2.0: Quantifying emergent complexity},
  author={PlaceholderName},
  journal={arXiv preprint arXiv:2503.13395},
  year={2025}
}
```

---

*PyMergence is under active development. API may change between versions until official v1.0.0 release.*

## Roadmap
- [X] Add documentation.
- [X] Add the single path measure from the CE2.0 paper
- [X] Add a notebook that recreates the figures from the CE2.0 paper.
- [X] Add a jupyter notebook to present and explain package use. 
- [ ] Add more tests for less regular TPMs.
- [ ] Add test for finding correct single path.
- [ ] Should we add a 'lattice' class?
- [ ] Make our delta-CP measure independent of `networkx` and faster by using adjacency matrices. 
- [ ] Speed up construction of Hasse diagram (Adj mat and then transitive reduction?). 
- [ ] Add option to calculate möbius inverse of CP. 