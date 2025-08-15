"""Quantum circuit to graph conversion utilities.

This module provides functions to convert Qiskit quantum circuits into
PyTorch Geometric graph representations suitable for graph neural networks.
Includes comprehensive feature extraction and edge type classification.
"""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGInNode, DAGOutNode
import torch
from torch_geometric.data import Data
import numpy as np
from qiskit import qasm3
import hashlib


def circuit_to_graph_data(circuit: QuantumCircuit, max_qubits_feature: int = 16, max_params: int = 3) -> Data:
    """Converts a Qiskit quantum circuit to a PyTorch Geometric graph with rich features.
    
    This function transforms a quantum circuit into a graph representation suitable for
    graph neural networks. The conversion includes comprehensive node features, edge
    features, and global circuit properties.
    
    Features included:
    - Node features: Gate types, qubit indices, layer positions, gate parameters
    - Edge features: Data flow vs control flow connections
    - Global features: Circuit statistics and gate counts
    
    Args:
        circuit (QuantumCircuit): Input quantum circuit to convert.
        max_qubits_feature (int, optional): Maximum number of qubits to encode in 
                                          node features. Defaults to 16.
        max_params (int, optional): Maximum number of gate parameters to encode
                                   per node. Defaults to 3.
    
    Returns:
        Data: PyTorch Geometric Data object containing the graph representation
             with node features, edge indices, edge attributes, and global features.
    """
    dag = circuit_to_dag(circuit)

    # --- 1. Define Feature Mappings ---
    # Expanded list of common gates
    op_types = {'x', 'h', 'cx', 'rz', 'ry', 'rx', 'p', 'u', 'u1', 'u2', 'u3', 'id',
                'barrier', 'measure', 'cz', 'ecr', 'sx', 'sxdg', 'y', 'z', 'swap'}
    op_type_map = {op: i for i, op in enumerate(sorted(list(op_types)))}
    
    # NEW: Node features now include qubit index encoding
    node_feature_size = len(op_types) + 3 + max_qubits_feature + 1 + (max_params * 2)

    # --- 2. Build Node Features ---
    node_features = []
    node_map = {} 

    layer_map = {}
    try:
        dag_layers_list = list(dag.layers())
        total_layers = len(dag_layers_list) if dag_layers_list else 1
        for i, layer in enumerate(dag_layers_list):
            for node in layer['graph'].op_nodes():
                layer_map[node] = i
    except Exception: # Handle potential DAG-related errors
        dag_layers_list = []
        total_layers = 1


    all_nodes = list(dag.op_nodes()) + list(dag.input_map.values()) + list(dag.output_map.values())

    for i, node in enumerate(all_nodes):
        node_map[node] = i
        feature = torch.zeros(node_feature_size)

        if isinstance(node, DAGOpNode):
            op_name = node.op.name.lower()
            feature[op_type_map.get(op_name, len(op_types) - 1)] = 1.0

            # Add qubit index as a feature
            qubit_idx_start = len(op_types) + 3
            for qarg in node.qargs:
                try:
                    q_idx = circuit.qubits.index(qarg)
                    if q_idx < max_qubits_feature:
                        feature[qubit_idx_start + q_idx] = 1.0
                except ValueError:

                    continue

            pos_encoding_idx = qubit_idx_start + max_qubits_feature
            feature[pos_encoding_idx] = layer_map.get(node, -1) / total_layers

            param_start_idx = pos_encoding_idx + 1
            if hasattr(node.op, 'params') and node.op.params:
                for j, param in enumerate(node.op.params):
                    if j < max_params:
                        try:
                            p_val = float(param)
                            feature[param_start_idx + j * 2] = np.sin(p_val)
                            feature[param_start_idx + j * 2 + 1] = np.cos(p_val)
                        except (TypeError, ValueError):
                            continue
        
        elif isinstance(node, DAGInNode):
            feature[len(op_types)] = 1.0
        elif isinstance(node, DAGOutNode):
            feature[len(op_types) + 1] = 1.0

        node_features.append(feature)

    node_features_tensor = torch.stack(node_features) if node_features else torch.empty(0, node_feature_size)

    # --- 3. Build Edges ---
    edge_list = []
    edge_attrs = []

    for edge in dag.edges():
        source_node, target_node, edge_data = edge
        if source_node in node_map and target_node in node_map:
            source_idx = node_map[source_node]
            target_idx = node_map[target_node]
            edge_list.append([source_idx, target_idx])
            
            # Simple edge attributes: [is_data_flow, is_control_flow]
            # For the simple model, we'll just mark all edges as data flow
            edge_attrs.append([1.0, 0.0])  # data flow edge

    # Convert to tensors
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle case with no edges (single node circuits)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    # --- 4. Global Features (simple version) ---
    global_features = torch.tensor([
        circuit.num_qubits,
        circuit.depth(),
        len([op for op in circuit.data if op.operation.name.lower() == 'cx']),  # CNOT count
        len([op for op in circuit.data if op.operation.name.lower() == 'h']),   # Hadamard count
        len([op for op in circuit.data if 'r' in op.operation.name.lower()]),   # Rotation gate count
        circuit.num_parameters if hasattr(circuit, 'num_parameters') else 0,
        len(circuit.data),  # Total number of operations
        len(set(qarg for op in circuit.data for qarg in op.qubits)),  # Number of active qubits
        len([op for op in circuit.data if len(op.qubits) > 1]),  # Multi-qubit gate count
        circuit.width(),  # Circuit width (includes classical bits)
        max([len(op.qubits) for op in circuit.data] + [1]),  # Max gate connectivity
        len([op for op in circuit.data if hasattr(op.operation, 'params') and op.operation.params])  # Parameterized gate count
    ], dtype=torch.float)

    return Data(
        x=node_features_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=global_features.unsqueeze(0)  # Add batch dimension
    )


def qasm_hash(circ: QuantumCircuit) -> str:
    """Creates a stable hash from a circuit's QASM3 representation.
    
    Generates a SHA-256 hash of the circuit's QASM3 string representation,
    providing a stable identifier for circuits that can be used for caching
    or deduplication.
    
    Args:
        circ (QuantumCircuit): The quantum circuit to hash.
    
    Returns:
        str: SHA-256 hexadecimal hash string of the circuit.
    
    Note:
        Falls back to hashing the string representation of circuit.data
        if QASM3 conversion fails.
    """
    try:
        qasm = qasm3.dumps(circ)
        return hashlib.sha256(qasm.encode()).hexdigest()
    except Exception:
        # Fallback for circuits that might have issues with QASM3 conversion
        return hashlib.sha256(str(circ.data).encode()).hexdigest()


