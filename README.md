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

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- Matplotlib ≥ 3.3.0
- NetworkX ≥ 3.4.0
- PyGraphviz ≥ 1.14

### Install from source

```bash
git clone https://github.com/EI-research-group/pymergence.git
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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use PyMergence in your research, please cite:

```bibtex
@article{2025causalEmergence,
  title={Causal Emergence 2.0: Quantifying emergent complexity},
  author={Erik Hoel},
  journal={arXiv preprint arXiv:2503.13395},
  year={2025}
}

@software{pymergence2025,
  title={PyMergence: A Python toolkit for causal emergence 2.0},
  author={Jansma, Abel and Hoel, Erik},
  year={2025},
  url={https://github.com/EI-research-group/pymergence},
  doi={10.5281/zenodo.17210078}
}
```

---

*PyMergence is under active development.*
