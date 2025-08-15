"""
ML Denoising: Quantum Circuit Error Mitigation with Machine Learning

A Python package that uses machine learning techniques, specifically Graph Neural Networks (GNNs), 
to mitigate quantum circuit errors. This package implements error mitigation strategies 
using PyTorch Geometric and Qiskit.

Developed by Kenny Heitritter - qBraid
"""

__version__ = "0.1.0"
__author__ = "Kenny Heitritter"
__email__ = "kenny@qbraid.com"
__license__ = "MIT"

# Core model classes
from .model import (
    QErrorMitigationModel,
    build_model,
    visualize_embeddings,
)

# Training and evaluation utilities
from .train import (
    train_mitigator,
    evaluate_mitigator,
)

# Data generation utilities
from .data_generation import (
    generate_proper_circuits,
    generate_proper_observables,
    prepare_mitigator_dataset,
)

# Circuit processing utilities
from .circuit import (
    circuit_to_graph_data,
)

# Noise modeling utilities
from .noise_modeling import (
    get_quera_noise_model,
)

__all__ = [
    # Model classes
    "QErrorMitigationModel",
    "build_model",
    "visualize_embeddings",
    
    # Training functions
    "train_mitigator",
    "evaluate_mitigator",
    
    # Data generation
    "generate_proper_circuits",
    "generate_proper_observables", 
    "prepare_mitigator_dataset",
    
    # Circuit utilities
    "circuit_to_graph_data",
    
    # Noise modeling
    "get_quera_noise_model",
] 