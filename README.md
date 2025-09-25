# MetaSimOpt - A library for neural network-based simulation optimization

MetaSimOpt is a Python library for to perform neural-network-based simulation optimization.  
It provides a set of tools to develop, train, and test simulations surrogate models and use them in combination with advanced optimisation algorithms.

## Features
- Support for metamodel or surrogate models developement and training
- Metamodel hyperparameter optimisation
- Easy integration with PyTorch for implementation of deep learning models
- Combination of neural networks and optimization algorithms for advance metamodel-based simulation optimization
- Visualisation of training curves and optimisation results

## Installation

Clone the repository and install dependencies with [Poetry](https://python-poetry.org/):
```
git clone https://github.com/your-username/metasimopt.git
cd metasimopt
poetry install
```

## Main Dependencies
The main dependencies of the project are:
- NumPy and Pandas for data handling
- Matplotlib and Seaborn for visualisation
- Scikit-learn for models and ML algorithms
- SciPy for scientific functions
- Optuna for hyperparameter optimisation
- PyTorch (with optional CUDA support) for complex models
- Pathos and Joblib for parallelisation

## Project Structure
The project has the following structure:
```
MetaSimOpt/
├── MetaSimOpt/          # Main source code
    ├── handlers/        # Handlers for training, hyperparameter search and models
    ├── metamodels/      # Surrogate models
    ├── opt_algorithms/  # A set of optimization algorithms
    ├── utils/           # Other utilities
├── Example/             # A usage example with files and Jupyeter notebooks
├── poetry.lock          # Dependencies version
├── pyproject.toml       # Poetry configuration
└── README.md            # This file
```

## Citation
The example refers to the content of the following two papers:

**Metamodel-Based Order Picking for Automated Storage and Retrieval Systems**  
Andrea Ferrari, Canan Gunes Corlu  
*2024 Winter Simulation Conference (WSC), IEEE, pp. 1457–1468*

The second paper extend the capabilities of the metamodel adding context features and use the neural networks in combination with advanced optimization algorithms to solve the order sequencing problem.

If you use **MetaSimOpt** in your research or you find this work interesting, please cite the following papers:

```bibtex
@inproceedings{ferrari2024metamodel,
  title={Metamodel-Based Order Picking for Automated Storage and Retrieval Systems},
  author={Ferrari, Andrea and Corlu, Canan Gunes},
  booktitle={2024 Winter Simulation Conference (WSC)},
  pages={1457--1468},
  year={2024},
  organization={IEEE}
}
```

## License

This project is licensed under the terms of the [GNU General Public License v3.0](LICENSE).  
You may redistribute and/or modify it under the terms of the GNU GPL, either version 3 or (at your option) any later version.