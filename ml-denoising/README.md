# ML Denoising: Quantum Circuit Error Mitigation with Machine Learning

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com/?gitHubUrl=https://github.com/qBraid/ai4quantum.git&redirectUrl=ml-denoising/README.md)


A Python package that uses machine learning techniques, specifically Graph Neural Networks (GNNs), to mitigate quantum circuit errors. This package implements advanced error mitigation strategies using PyTorch Geometric and Qiskit-Aer for quantum circuit simulation.

## Features

- **Graph Neural Network Models**: Simple and robust GNN models for quantum error mitigation
- **Rich Circuit Representation**: Converts quantum circuits to graphs with node, edge, and global features that are fed into PyTorch Geometric
- **Noise Modeling**: Implements realistic noise models based on experimental quantum hardware using Qiskit-Aer
- **Comprehensive Evaluation**: Built-in metrics and visualization tools for model performance analysis
- **Scalable Training**: Efficient training pipeline with early stopping and learning rate scheduling

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/qBraid/ai4quantum.git
cd ai4quantum/ml-denoising/

# Install using poetry
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
pip install -e .
```


## Project Structure

```
ml-denoising/
├── src/
│   └── ml_denoising/
│       ├── __init__.py
│       ├── model.py          # Neural network models
│       ├── train.py          # Training pipeline
│       ├── data_generation.py # Data generation utilities
│       ├── circuit.py        # Circuit-to-graph conversion
│       └── noise_modeling.py # Noise model implementations
├── examples/
│   ├── run_experiment.py     # Full-scale experiment runner
│   ├── run_small_experiment.py # Small-scale test experiment
│   ├── run_medium_experiment.py # Medium-scale experiment
│   └── introductory_example_updated.ipynb # Jupyter notebook tutorial
├── pyproject.toml           # Poetry configuration
└── README.md
```

## Backend Migration

This package has been migrated from CUDA-Q to Qiskit-Aer for quantum circuit simulation. The migration enables:
- Better compatibility across different systems
- More flexible noise modeling capabilities
- Integration with the broader Qiskit ecosystem

## Citation

If you use this package in your research, please cite:

```bibtex
@software{ml_denoising,
  title={ML Denoising: Quantum Circuit Error Mitigation with Machine Learning},
  author={Kenny Heitritter},
  organization={qBraid},
  year={2025},
  url={https://github.com/qbraid/ai4quantum/ml-denoising}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

For questions and support, please open an issue on GitHub or contact [kenny@qbraid.com](mailto:kenny@qbraid.com). 