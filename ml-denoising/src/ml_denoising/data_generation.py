"""Data generation utilities for quantum error mitigation datasets.

This module provides functions to generate quantum circuits, observables, and
prepare datasets for training quantum error mitigation models.
"""

from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import List, Tuple, Dict, Callable
import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp

from .circuit import circuit_to_graph_data



def generate_proper_circuits(
    num_circuits: int = 500,
    min_qubits: int = 2,
    max_qubits: int = 6,
    min_depth: int = 50,
    max_depth: int = 200,
    seed: int = 42,
    circuit_types: list = ['random', 'su2', 'uccsd']
) -> List[QuantumCircuit]:
    """
    Generate proper quantum circuits with actual gates.
    Args:
        num_circuits: Number of circuits to generate
        min_qubits: Minimum number of qubits
        max_qubits: Maximum number of qubits
        min_depth: Minimum circuit depth
        max_depth: Maximum circuit depth
        seed: Random seed
        circuit_types: Types of circuits to generate ('random', 'su2', 'uccsd')
    Returns:
        List of QuantumCircuit objects
    """
    np.random.seed(seed)
    circuits_qiskit = []
    
    # Generate random circuits
    if 'random' in circuit_types:
        for i in range(num_circuits):
            # Random number of qubits and depth
            num_qubits = np.random.randint(min_qubits, max_qubits + 1)
            depth = np.random.randint(min_depth, max_depth + 1)
            # Generate a sequence of gates and parameters first
            # This ensures both Qiskit and CUDA-Q circuits will be identical
            gates_sequence = []
            two_qubit_gates = []
            params = []
            # For 1-qubit gates
            for d in range(depth):
                for q in range(num_qubits):
                    gate_type = np.random.randint(0, 6)  # 0=H, 1=X, 2=Y, 3=Z, 4=RX, 5=RY
                    qubit = q
                    if gate_type in [4, 5]:  # Rotation gates need parameters
                        param_val = np.random.uniform(0, 2*np.pi)
                        params.append(param_val)
                        gates_sequence.append((gate_type, qubit, len(params)-1))
                    else:
                        gates_sequence.append((gate_type, qubit, None))
                # For 2-qubit gates (if we have at least 2 qubits)
                if num_qubits >= 2:
                    for _ in range(min(num_qubits // 2, 5)):
                        control = np.random.randint(0, num_qubits)
                        target = np.random.randint(0, num_qubits)
                        while target == control:
                            target = np.random.randint(0, num_qubits)
                        # Two-qubit gate type: 0=CX, 1=CZ
                        two_q_gate = np.random.randint(0, 2)
                        two_qubit_gates.append((two_q_gate, control, target))
            # Create Qiskit circuit
            qiskit_circuit = QuantumCircuit(num_qubits, name=f"random_circuit_{i}")
            # Apply gates to Qiskit circuit using the generated sequence
            for gate_type, qubit, param_idx in gates_sequence:
                if gate_type == 0:  # H
                    qiskit_circuit.h(qubit)
                elif gate_type == 1:  # X
                    qiskit_circuit.x(qubit)
                elif gate_type == 2:  # Y
                    qiskit_circuit.y(qubit)
                elif gate_type == 3:  # Z
                    qiskit_circuit.z(qubit)
                elif gate_type == 4:  # RX
                    qiskit_circuit.rx(params[param_idx], qubit)
                elif gate_type == 5:  # RY
                    qiskit_circuit.ry(params[param_idx], qubit)
            # Apply 2-qubit gates to Qiskit circuit
            for two_q_gate, control, target in two_qubit_gates:
                if two_q_gate == 0:  # CX
                    qiskit_circuit.cx(control, target)
                else:  # CZ
                    qiskit_circuit.cz(control, target)
            
            # Store the circuits in their respective dictionaries
            circuits_qiskit.append(qiskit_circuit)
            
    # Generate SU2 circuits (similar to hardware-efficient ansatz)
    if 'su2' in circuit_types:
        for i in range(num_circuits):
            # Random number of qubits
            num_qubits = np.random.randint(min_qubits, max_qubits + 1)
            # Random number of repetitions (layers)
            reps = np.random.randint(1, 4)
            # Define the gate sequence and parameters first
            gates_sequence = []
            params = []
            # Build the SU2 ansatz manually to have precise control
            for r in range(reps):
                # Rotation layer
                for q in range(num_qubits):
                    # RY rotations
                    param_val = np.random.uniform(0, 2*np.pi)
                    params.append(param_val)
                    gates_sequence.append(('ry', q, len(params)-1))
                    # RZ rotations
                    param_val = np.random.uniform(0, 2*np.pi)
                    params.append(param_val)
                    gates_sequence.append(('rz', q, len(params)-1))
                # Entanglement layer - linear connectivity
                for q in range(num_qubits - 1):
                    gates_sequence.append(('cx', q, q+1))
            # Create Qiskit circuit
            qiskit_circuit = QuantumCircuit(num_qubits, name=f"efficient_su2_{i}")
            # Apply gates to Qiskit circuit
            for gate_type, qubit1, qubit2_or_param in gates_sequence:
                if gate_type == 'ry':
                    qiskit_circuit.ry(params[qubit2_or_param], qubit1)
                elif gate_type == 'rz':
                    qiskit_circuit.rz(params[qubit2_or_param], qubit1)
                elif gate_type == 'cx':
                    qiskit_circuit.cx(qubit1, qubit2_or_param)
            
            # Store the circuits
            circuits_qiskit.append(qiskit_circuit)
            
    # Generate UCCSD-like circuits with CNOT ladders and rotations
    if 'uccsd' in circuit_types:
        for i in range(num_circuits):
            # Random number of qubits
            num_qubits = np.random.randint(min_qubits, max_qubits + 1)
            # Random number of layers
            num_layers = np.random.randint(1, 4)
            # Define the gate sequence first
            gates_sequence = []
            params = []
            # Initial Hadamard layer
            for q in range(num_qubits):
                gates_sequence.append(('h', q, None))
            # Build the UCCSD-like ansatz
            for layer in range(num_layers):
                # Forward ladder
                for q in range(num_qubits - 1):
                    gates_sequence.append(('cx', q, q+1))
                # Rotation layer
                for q in range(num_qubits):
                    param_val = np.random.uniform(0, 2*np.pi)
                    params.append(param_val)
                    gates_sequence.append(('rz', q, len(params)-1))
                # Backward ladder
                for q in range(num_qubits - 1, 0, -1):
                    gates_sequence.append(('cx', q-1, q))
            # Create Qiskit circuit
            qiskit_circuit = QuantumCircuit(num_qubits, name=f"uccsd_like_{i}")
            # Apply gates to Qiskit circuit
            for gate_type, qubit1, qubit2_or_param in gates_sequence:
                if gate_type == 'h':
                    qiskit_circuit.h(qubit1)
                elif gate_type == 'rz':
                    qiskit_circuit.rz(params[qubit2_or_param], qubit1)
                elif gate_type == 'cx':
                    qiskit_circuit.cx(qubit1, qubit2_or_param)
            
            # Store the circuits
            circuits_qiskit.append(qiskit_circuit)
            
    return circuits_qiskit


def generate_proper_observables(circuits_qiskit,
                                max_terms = 8,
                                seed      = 42):
    """
    Produces observables for quantum circuits.
    Args:
        circuits_qiskit: List of Qiskit quantum circuits
        max_terms: Maximum number of terms in each observable
        seed: Random seed for reproducibility
    
    Returns:
        List of SparsePauliOp observables
    """
    np.random.seed(seed)
    obs_qiskit = []

    for qc in circuits_qiskit:
        n = qc.num_qubits
        T = np.random.randint(1, max_terms+1)
        paulis, coeffs = [], []
        for _ in range(T):
            pstr  = ''.join(np.random.choice(['I','X','Y','Z'], n))
            coeff = np.random.uniform(-1, 1)
            paulis.append(pstr); coeffs.append(coeff)

        # Qiskit representation (for feature extraction) ----------------------
        sp_op = SparsePauliOp(paulis, coeffs)
        obs_qiskit.append(sp_op)

    return obs_qiskit

def extract_observable_features(observable: BaseOperator) -> List[float]:
    """Extract numerical features from a quantum observable.
    
    Analyzes the structure of a quantum observable and extracts features
    such as the number of terms, Pauli operator counts, and average weight
    for use in machine learning models.
    
    Args:
        observable (BaseOperator): The quantum observable to analyze 
                                 (typically a SparsePauliOp).
        
    Returns:
        List[float]: Feature vector containing:
                    - Number of terms in the observable
                    - Number of X Pauli operators
                    - Number of Y Pauli operators  
                    - Number of Z Pauli operators
                    - Average weight (number of non-identity Paulis per term)
    """
    features = []
    
    # # Convert to SparsePauliOp if needed
    # if isinstance(observable, PauliSumOp):
    #     pauli_op = observable.primitive
    if hasattr(observable, 'to_pauli_op'):
        pauli_op = observable.to_pauli_op()
    else:
        # If we can't convert, return minimal features
        features.append(1.0)  # Number of terms
        features.append(0.0)  # Number of X terms
        features.append(0.0)  # Number of Y terms
        features.append(0.0)  # Number of Z terms
        features.append(0.0)  # Average weight
        return features
    
    # Number of terms
    features.append(float(len(pauli_op)))
    
    # Count Pauli types
    x_count = 0
    y_count = 0
    z_count = 0
    weights = []
    
    for pauli_string, _ in zip(pauli_op.paulis, pauli_op.coeffs):
        pauli_str = pauli_string.to_label()
        x_count += pauli_str.count('X')
        y_count += pauli_str.count('Y')
        z_count += pauli_str.count('Z')
        weight = len(pauli_str) - pauli_str.count('I')
        weights.append(weight)
    
    features.append(float(x_count))
    features.append(float(y_count))
    features.append(float(z_count))
    
    # Average weight
    avg_weight = np.mean(weights) if weights else 0.0
    features.append(float(avg_weight))
    
    return features

def prepare_mitigator_dataset(circuits, observables, noisy_values, noiseless_values, noise_factor):
    """Prepare a dataset for training quantum error mitigation models.
    
    Converts quantum circuits, observables, and expectation values into a format
    suitable for training graph neural networks. Each circuit is converted to a
    graph representation and combined with observable features and noise information.
    
    Args:
        circuits: List of quantum circuits or list of lists of circuits for 
                 multiple noise factors.
        observables: List of observables or list of lists of observables for 
                   multiple noise factors.
        noisy_values: List of noisy expectation values or list of lists for 
                     multiple noise factors.
        noiseless_values: List of noiseless expectation values or list of lists 
                        for multiple noise factors.
        noise_factor: Single noise factor value or list of noise factor values.
    
    Returns:
        List[dict]: List of dataset entries, each containing:
                   - circuit_graph: PyTorch Geometric graph representation
                   - observable_features: Extracted observable features  
                   - noise_factor: Noise level used
                   - noisy_exp: Noisy expectation value
                   - true_exp: True (noiseless) expectation value
                   - correction: Difference between true and noisy values
                   - num_qubits: Number of qubits in the circuit
                   - circuit: Original quantum circuit
    """
    # from ml_qem.features import extract_observable_features
    
    dataset = []
    
    # Check if we have multiple noise factors
    if isinstance(noise_factor, (list, tuple, np.ndarray)):
        # Process multiple noise factors
        for nf_idx, nf in enumerate(noise_factor):
            # Get the corresponding data for this noise factor
            if isinstance(circuits[0], list):
                curr_circuits = circuits[nf_idx]
                curr_observables = observables[nf_idx]
                curr_noisy_values = noisy_values[nf_idx]
                curr_noiseless_values = noiseless_values[nf_idx]
            else:
                # If only noise_factor is a list but other inputs aren't lists of lists,
                # use the same data for each noise factor
                curr_circuits = circuits
                curr_observables = observables
                curr_noisy_values = noisy_values
                curr_noiseless_values = noiseless_values
            
            # Process the data for this noise factor
            for i in range(len(curr_circuits)):
                circuit_graph = circuit_to_graph_data(curr_circuits[i])
                obs_features = torch.tensor(extract_observable_features(curr_observables[i]), dtype=torch.float).unsqueeze(0)
                correction = curr_noiseless_values[i] - curr_noisy_values[i]
                
                dataset.append({
                    'circuit_graph': circuit_graph,
                    'observable_features': obs_features,
                    'noise_factor': torch.tensor([nf], dtype=torch.float).view(1, 1),
                    'noisy_exp': torch.tensor([curr_noisy_values[i]], dtype=torch.float).view(1, 1),
                    'true_exp': torch.tensor([curr_noiseless_values[i]], dtype=torch.float).view(1, 1),
                    'correction': torch.tensor([correction], dtype=torch.float).view(1, 1),
                    'num_qubits': curr_circuits[i].num_qubits,
                    'circuit': curr_circuits[i]  # Store the individual circuit, not the full list
                })
    else:
        # Original single noise factor case
        for i in range(len(circuits)):
            # Convert circuit to graph
            circuit_graph = circuit_to_graph_data(circuits[i])
            
            # Extract observable features
            obs_features = torch.tensor(extract_observable_features(observables[i]), dtype=torch.float).unsqueeze(0)
            
            # Calculate the correction (true - noisy)
            correction = noiseless_values[i] - noisy_values[i]
            
            # Store as dictionary
            dataset.append({
                'circuit_graph': circuit_graph,
                'observable_features': obs_features,
                'noise_factor': torch.tensor([noise_factor], dtype=torch.float).view(1, 1),
                'noisy_exp': torch.tensor([noisy_values[i]], dtype=torch.float).view(1, 1),
                'true_exp': torch.tensor([noiseless_values[i]], dtype=torch.float).view(1, 1),
                'correction': torch.tensor([correction], dtype=torch.float).view(1, 1),
                'num_qubits': circuits[i].num_qubits,
                'circuit': circuits[i]  # Store the individual circuit
            })
    
    return dataset
