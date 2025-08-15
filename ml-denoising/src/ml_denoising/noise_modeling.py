"""
Noise modeling utilities for quantum circuit error mitigation.

This module provides realistic noise models based on experimental quantum hardware
parameters, particularly focusing on the QuEra noise model derived from 
Bluvstein et al., Nature Vol 626, (2024).
"""
from __future__ import annotations
from qiskit_aer.noise import NoiseModel, pauli_error
import numpy as np


def get_quera_noise_model(config_quera_noise_factor: float = 1.0) -> NoiseModel:
    """
    Constructs a quantum noise model based on error parameters
    derived from the Bluvstein et al., Nature Vol 626, (2024) paper.

    This noise model includes realistic error rates for:
    - Reset operations
    - Measurement operations  
    - Single-qubit gates (U1, U2, U3)
    - Two-qubit gates (CZ, CNOT, etc.)

    Args:
        config_quera_noise_factor (float): A factor to scale the error probabilities.
                                           Default is 1.0 (no scaling).

    Returns:
        NoiseModel: The Qiskit Aer noise model configured with QuEra-based parameters.
    """
    quera_noise_model = NoiseModel()

    # 1. Reset error
    # Supported by SPAM error (Ref. [8] cited by paper) and measurement fidelity (paper p2, p10).
    # e.g., SPAM ~0.006, Meas_Error ~0.002 => Reset_Error ~0.004
    p_reset = 0.004 * config_quera_noise_factor
    error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])
    quera_noise_model.add_all_qubit_quantum_error(error_reset, "reset")

    # 2. Measurement error
    # Paper (p2, Fig 1c; p10) mentions ~99.8% readout fidelity (0.002 error).
    # SPAM data from Ref. [8] also supports p_meas around 0.003.
    p_meas = 0.003 * config_quera_noise_factor
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])
    quera_noise_model.add_all_qubit_quantum_error(error_meas, "measure")

    # 3. Entangling errors only for the active qubits in the CZ gate
    # Paper (p4, p9) states two-qubit gate fidelity of 99.5% (0.005 error).
    p_cz_active_qub = 0.005 * config_quera_noise_factor
    # The specific Pauli distribution X(p/4), Y(p/4), Z(p/2) is a modeling choice.
    cz_single_qubit_error = pauli_error(
        [
            ("X", 1 / 4 * p_cz_active_qub),
            ("Y", 1 / 4 * p_cz_active_qub),
            ("Z", 1 / 2 * p_cz_active_qub),
            ("I", 1 - p_cz_active_qub),
        ]
    )
    cz_error = cz_single_qubit_error.tensor(cz_single_qubit_error)
    quera_noise_model.add_quantum_error(
        cz_error, ["cx", "ecr", "cz"], [0, 1]  # Apply to specific qubits for 2Q gates
    )
    # To apply to all instances of 2Q gates if qubits are not specified
    # quera_noise_model.add_all_qubit_quantum_error(cz_error, ["cx", "ecr", "cz"]) # This is less precise for 2Q gates

    # 4. Single-qubit gate errors
    # p_u1: Supported by Ref. [8] (avg 1Q fidelity 99.94%, error 6e-4) and
    #       paper (p10, Raman scattering limit error ~7e-4).
    p_u1 = 5e-4 * config_quera_noise_factor
    # p_u2: Supported by paper (p10, local Z(pi/2) fidelity 99.912%, error ~8.8e-4).
    p_u2 = 1e-3 * config_quera_noise_factor
    # p_u3: Plausible for more complex/longer 1Q gates, qualitatively supported
    #       by paper's discussion of variable gate durations (p10, Ext. Data Fig. 2b).
    p_u3 = 1.5e-3 * config_quera_noise_factor

    # Depolarizing-like Pauli error model for single-qubit gates.
    sq_error_u1 = pauli_error(
        [
            ("X", 1 / 3 * p_u1),
            ("Y", 1 / 3 * p_u1),
            ("Z", 1 / 3 * p_u1),
            ("I", 1 - p_u1),
        ]
    )
    # Apply to gates corresponding to simpler/shorter rotations
    quera_noise_model.add_all_qubit_quantum_error(
        sq_error_u1, ["u1", "rz", "ry", "rx", "sx", "sxdg", "x", "y", "z", "h"]  # Common 1Q gates
    )

    sq_error_u2 = pauli_error(
        [
            ("X", 1 / 3 * p_u2),
            ("Y", 1 / 3 * p_u2),
            ("Z", 1 / 3 * p_u2),
            ("I", 1 - p_u2),
        ]
    )
    # Apply to gates corresponding to moderately complex rotations (e.g., U2)
    quera_noise_model.add_all_qubit_quantum_error(sq_error_u2, ["u2"])

    sq_error_u3 = pauli_error(
        [
            ("X", 1 / 3 * p_u3),
            ("Y", 1 / 3 * p_u3),
            ("Z", 1 / 3 * p_u3),
            ("I", 1 - p_u3),
        ]
    )
    # Apply to gates corresponding to more complex rotations (e.g., U3)
    quera_noise_model.add_all_qubit_quantum_error(sq_error_u3, ["u3", "u"])  # u is a general 1Q gate

    return quera_noise_model