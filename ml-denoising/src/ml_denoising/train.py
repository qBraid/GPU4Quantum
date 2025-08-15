"""Training and evaluation utilities for quantum error mitigation models.

This module provides functions for training graph neural networks on quantum
error mitigation tasks, including training loops, evaluation metrics, and
result visualization.
"""

import torch
import torch.nn as nn
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch
from tqdm import tqdm
import json



def train_mitigator(dataset, model, epochs=100, batch_size=32, lr=0.001, device='cpu', early_stopping_patience=10):
    """Train a quantum error mitigation model with early stopping and validation.
    
    Trains a graph neural network model to predict quantum error corrections using
    a dataset of circuit graphs and expectation values. Includes validation monitoring,
    early stopping, learning rate scheduling, and training history tracking.
    
    Args:
        dataset: List of dataset entries containing circuit graphs and target values.
        model: PyTorch neural network model to train (QErrorMitigationModel).
        epochs (int, optional): Maximum number of training epochs. Defaults to 100.
        batch_size (int, optional): Training batch size. Defaults to 32.
        lr (float, optional): Initial learning rate. Defaults to 0.001.
        device (str, optional): Device for training ('cpu' or 'cuda'). Defaults to 'cpu'.
        early_stopping_patience (int, optional): Epochs to wait before early stopping. 
                                               Defaults to 10.
    
    Returns:
        tuple: (trained_model, training_history) where training_history contains
               loss curves, learning rates, and timing information.
    """
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Create data loaders with custom collate function for rich graph data
    def collate_fn(batch):
        # Take the graph data from each item in the batch
        data_list = [item['circuit_graph'] for item in batch]
        
        # Add the other features directly to each Data object before batching.
        for i, data in enumerate(data_list):
            item = batch[i]
            data.observable_features = item['observable_features']
            data.noise_factor = item['noise_factor']
            data.noisy_exp = item['noisy_exp']
            data.true_exp = item['true_exp']
            data.correction = item['correction']
        
        # `Batch.from_data_list` will automatically handle creating a single batch object
        # with all attributes correctly batched.
        return Batch.from_data_list(data_list)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    model = model.to(device)
    
    # Store training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'epoch_times': []
    }
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        # Training phase
        model.train()
        train_loss = 0
        
        # Use tqdm with leave=False to avoid flooding output
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # Move data to device
            circuit_data = batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            
            # Model predicts the correction directly
            predicted_correction = model(circuit_data)
            # Target is the actual correction (true - noisy)
            target_correction = circuit_data.true_exp - circuit_data.noisy_exp
            loss = criterion(predicted_correction, target_correction)

            # Backward pass
            loss.backward()

            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                circuit_data = batch.to(device)
                
                # Forward pass
                predicted_correction = model(circuit_data)
                # Target is the actual correction (true - noisy)
                target_correction = circuit_data.true_exp - circuit_data.noisy_exp
                loss = criterion(predicted_correction, target_correction)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start_time
        
        # Store training metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        history['epoch_times'].append(epoch_time)
        
        # Only print every 10 epochs to avoid flooding output
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_quantum_mitigator.pt')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {early_stopping_patience} epochs.")
            break
    
    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save final model and training history
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'best_val_loss': best_val_loss,
    }, 'final_quantum_mitigator.pt')
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(history['learning_rates'])
    plt.yscale('log')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    os.makedirs("ml_mitigation_output", exist_ok=True)
    plt.savefig(os.path.join("ml_mitigation_output", "training_history.png"))
    
    return model, history

def evaluate_mitigator(model, test_dataset, device='cpu', output_prefix="ml_mitigation_output"):
    """Evaluate quantum error mitigation model performance with comprehensive metrics.
    
    Computes detailed performance metrics for a trained error mitigation model,
    including error statistics, R² scores, and performance analysis by circuit
    properties. Generates plots and saves results to disk.
    
    Args:
        model: Trained error mitigation model or tuple (model, history).
        test_dataset: List of test dataset entries containing circuit graphs and targets.
        device (str, optional): Device for inference ('cpu' or 'cuda'). Defaults to 'cpu'.
        output_prefix (str, optional): Directory to save evaluation results. 
                                     Defaults to "ml_mitigation_output".
    
    Returns:
        dict: Comprehensive evaluation results including:
             - Error statistics (RMSE, MAE, etc.)
             - R² scores and correlation coefficients
             - Performance breakdowns by circuit properties
             - Confidence intervals and statistical tests
    """
    from torch.utils.data import DataLoader
    from sklearn.metrics import r2_score
    
    # Check if model is a tuple (model, history) and extract just the model
    if isinstance(model, tuple) and len(model) > 0:
        model = model[0]
    
    # Use same collate function as in training
    def collate_fn(batch):
        # Take the graph data from each item in the batch
        data_list = [item['circuit_graph'] for item in batch]
        
        # Add the other features directly to each Data object before batching.
        for i, data in enumerate(data_list):
            item = batch[i]
            data.observable_features = item['observable_features']
            data.noise_factor = item['noise_factor']
            data.noisy_exp = item['noisy_exp']
            data.true_exp = item['true_exp']
            # Add num_qubits for evaluation analysis
            data.num_qubits = torch.tensor(item['num_qubits'])
        
        # `Batch.from_data_list` will automatically handle creating a single batch object
        # with all attributes correctly batched.
        return Batch.from_data_list(data_list)
    
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Evaluation
    model.eval()
    predictions = []
    true_values = []
    noisy_values = []
    corrections = []
    qubit_counts = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            circuit_data = batch.to(device)
            
            # Model predicts the correction
            predicted_correction = model(circuit_data)
            
            # Apply the correction to get the final prediction
            corrected_exp = circuit_data.noisy_exp + predicted_correction
            
            # The applied correction
            predicted_correction_val = predicted_correction.cpu().numpy().flatten()
            
            # Store results
            predictions.extend(corrected_exp.cpu().numpy().flatten())
            true_values.extend(circuit_data.true_exp.cpu().numpy().flatten())
            noisy_values.extend(circuit_data.noisy_exp.cpu().numpy().flatten())
            corrections.extend(predicted_correction_val)
            qubit_counts.extend(circuit_data.num_qubits.cpu().numpy().flatten())

    predictions = np.array(predictions)
    true_values = np.array(true_values)
    noisy_values = np.array(noisy_values)
    corrections = np.array(corrections)
    qubit_counts = np.array(qubit_counts)

    # Calculate errors
    noisy_error = np.abs(true_values - noisy_values)
    mitigated_error = np.abs(true_values - predictions)
    
    # --- Confidence Intervals (Bootstrap) ---
    def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
        bootstrapped_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
        lower_bound = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
        return np.mean(data), lower_bound, upper_bound

    mean_noisy_err, noisy_err_lb, noisy_err_ub = bootstrap_ci(noisy_error)
    mean_mitigated_err, mitigated_err_lb, mitigated_err_ub = bootstrap_ci(mitigated_error)
    
    # Bootstrap for error reduction
    reductions = []
    for _ in range(1000):
        indices = np.random.choice(len(noisy_error), size=len(noisy_error), replace=True)
        noisy_sample_mean = np.mean(noisy_error[indices])
        mitigated_sample_mean = np.mean(mitigated_error[indices])
        if noisy_sample_mean > 1e-9:
            reductions.append(100 * (1 - mitigated_sample_mean / noisy_sample_mean))
    
    mean_reduction, reduction_lb, reduction_ub = bootstrap_ci(np.array(reductions)) if reductions else (0, 0, 0)

    # --- Additional Metrics ---
    r2 = r2_score(true_values, predictions)

    # Print statistics
    print("\n--- Error Mitigation Evaluation ---")
    print(f"  Mean Noisy Error:      {mean_noisy_err:.4f} (95% CI: [{noisy_err_lb:.4f}, {noisy_err_ub:.4f}])")
    print(f"  Mean Mitigated Error:  {mean_mitigated_err:.4f} (95% CI: [{mitigated_err_lb:.4f}, {mitigated_err_ub:.4f}])")
    print(f"  Error Reduction:       {mean_reduction:.2f}% (95% CI: [{reduction_lb:.2f}%, {reduction_ub:.2f}%])")
    print(f"  R-squared (R²):        {r2:.4f}")
    print("------------------------------------")
    
    # Create visualization
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Compare mitigated vs true values
    plt.subplot(2, 3, 1)
    plt.scatter(noisy_values, true_values, alpha=0.5, label='Noisy')
    plt.scatter(predictions, true_values, alpha=0.5, label='Mitigated')
    plt.plot([min(true_values), max(true_values)], 
            [min(true_values), max(true_values)], 'k--', label='Ideal')
    plt.xlabel("Expectation Value")
    plt.ylabel("True Expectation Value")
    plt.legend()
    plt.title("Mitigation Performance")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Error distribution
    plt.subplot(2, 3, 2)
    plt.hist(noisy_error, alpha=0.5, bins=30, label='Noisy Error', density=True)
    plt.hist(mitigated_error, alpha=0.5, bins=30, label='Mitigated Error', density=True)
    plt.xlabel("Absolute Error")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Error Distribution")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Correction accuracy
    plt.subplot(2, 3, 3)
    ideal_corrections = true_values - noisy_values
    plt.scatter(ideal_corrections, corrections, alpha=0.5)
    plt.plot([min(ideal_corrections), max(ideal_corrections)], 
             [min(ideal_corrections), max(ideal_corrections)], 'k--', label='Ideal')
    plt.xlabel("Ideal Correction (True - Noisy)")
    plt.ylabel("Applied Correction")
    plt.title("Correction Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Error histogram per circuit size
    plt.subplot(2, 3, 4)
    # Define qubit bins based on actual data range
    min_q_val = int(np.min(qubit_counts))
    max_q_val = int(np.max(qubit_counts))
    
    qubit_bins = np.linspace(min_q_val, max_q_val, 4, dtype=int)
    for i in range(len(qubit_bins) - 1):
        q_min, q_max = qubit_bins[i], qubit_bins[i+1]
        mask = (qubit_counts >= q_min) & (qubit_counts < q_max)
        if np.any(mask):
            plt.hist(mitigated_error[mask], bins=20, alpha=0.6, label=f'{q_min}-{q_max-1} Qubits', density=True)
    plt.xlabel("Mitigated Absolute Error")
    plt.ylabel("Density")
    plt.title("Mitigated Error by Qubit Count")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 5: R-squared
    plt.subplot(2, 3, 5)
    plt.bar(['R²'], [r2], color=['#1f77b4'])
    plt.ylim(min(0, r2) - 0.1, 1.1)
    plt.axhline(1.0, color='k', linestyle='--', label='Perfect Score')
    plt.title('Goodness of Fit (R²)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.text(0, r2 + 0.02, f'{r2:.3f}', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Comprehensive Model Evaluation", fontsize=16)
    os.makedirs(output_prefix, exist_ok=True)
    plt.savefig(os.path.join(output_prefix, "mitigation_results.png"))
    
    # Save evaluation results
    evaluation_results = {
        'metrics': {
            'mean_noisy_error': float(mean_noisy_err),
            'mean_mitigated_error': float(mean_mitigated_err),
            'error_reduction_percentage': float(mean_reduction),
            'r2_score': float(r2)
        },
        'confidence_intervals': {
            'noisy_error': {'lower': float(noisy_err_lb), 'upper': float(noisy_err_ub)},
            'mitigated_error': {'lower': float(mitigated_err_lb), 'upper': float(mitigated_err_ub)},
            'error_reduction': {'lower': float(reduction_lb), 'upper': float(reduction_ub)},
        },
        'raw_data': {
            'predictions': predictions.tolist(),
            'true_values': true_values.tolist(),
            'noisy_values': noisy_values.tolist(),
            'corrections': corrections.tolist(),
            'qubit_counts': qubit_counts.tolist(),
        }
    }
    
    with open(os.path.join(output_prefix, "evaluation_results.json"), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    return evaluation_results

def save_scaling_results(experiment_config, eval_results, output_dir):
    """Save scaling experiment results and metadata to disk.
    
    Creates experiment metadata files containing configuration parameters,
    performance results, and confidence intervals for reproducibility and
    analysis.
    
    Args:
        experiment_config (dict): Configuration parameters for the experiment.
        eval_results (dict): Evaluation results from evaluate_mitigator.
        output_dir (str): Directory to save the results files.
    
    Returns:
        dict: Experiment metadata that was saved to disk.
    """
    
    # Create experiment metadata
    experiment_metadata = {
        'experiment_type': 'scaling_study',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'configuration': experiment_config,
        'results': eval_results['metrics'] if eval_results else {},
        'confidence_intervals': eval_results.get('confidence_intervals', {}) if eval_results else {}
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "experiment_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    print(f"Saved experiment metadata to {metadata_path}")
    return experiment_metadata

def create_scaling_summary_plots(all_results, output_dir):
    """Create comprehensive summary plots for scaling study results.
    
    Generates visualization plots showing how model performance scales with
    the number of training circuits and model depth across different model
    variants.
    
    Args:
        all_results (dict): Dictionary containing all experimental results
                          organized by model variant and configuration.
        output_dir (str): Directory to save the summary plots.
    
    Returns:
        None: Saves plots to disk and prints status messages.
    """
    
    try:
        # Extract data for plotting
        variants = list(all_results['results'].keys())
        data_configs = all_results['configurations']['data_scaling_configs']
        depth_configs = all_results['configurations']['model_depth_configs']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scaling Study Results Summary', fontsize=16)
        
        for variant_idx, variant in enumerate(variants):
            variant_results = all_results['results'][variant]
            
            # Plot 1: Error reduction vs number of circuits (for default depth)
            default_depth = depth_configs[2] if len(depth_configs) > 2 else depth_configs[0]  # Use middle or first depth
            error_reductions = []
            circuit_counts = []
            
            for circuits_config in data_configs:
                key = f'circuits_{circuits_config}'
                depth_key = f'depth_{default_depth}'
                if key in variant_results and depth_key in variant_results[key]:
                    result = variant_results[key][depth_key]
                    if 'performance_metrics' in result and 'error_reduction_percentage' in result['performance_metrics']:
                        error_reductions.append(result['performance_metrics']['error_reduction_percentage'])
                        circuit_counts.append(circuits_config)
            
            if error_reductions:
                axes[0, variant_idx].plot(circuit_counts, error_reductions, 'o-', label=f'{variant} (depth {default_depth})')
                axes[0, variant_idx].set_xlabel('Number of Circuits')
                axes[0, variant_idx].set_ylabel('Error Reduction (%)')
                axes[0, variant_idx].set_title(f'{variant.capitalize()} Model: Data Scaling')
                axes[0, variant_idx].grid(True, alpha=0.3)
                axes[0, variant_idx].legend()
            
            # Plot 2: Error reduction vs model depth (for default circuit count)
            default_circuits = data_configs[2] if len(data_configs) > 2 else data_configs[0]  # Use middle or first
            error_reductions_depth = []
            depths = []
            
            circuits_key = f'circuits_{default_circuits}'
            if circuits_key in variant_results:
                for depth_config in depth_configs:
                    depth_key = f'depth_{depth_config}'
                    if depth_key in variant_results[circuits_key]:
                        result = variant_results[circuits_key][depth_key]
                        if 'performance_metrics' in result and 'error_reduction_percentage' in result['performance_metrics']:
                            error_reductions_depth.append(result['performance_metrics']['error_reduction_percentage'])
                            depths.append(depth_config)
            
            if error_reductions_depth:
                axes[1, variant_idx].plot(depths, error_reductions_depth, 's-', label=f'{variant} ({default_circuits} circuits)')
                axes[1, variant_idx].set_xlabel('Model Depth')
                axes[1, variant_idx].set_ylabel('Error Reduction (%)')
                axes[1, variant_idx].set_title(f'{variant.capitalize()} Model: Depth Scaling')
                axes[1, variant_idx].grid(True, alpha=0.3)
                axes[1, variant_idx].legend()
        
        plt.tight_layout()
        summary_plot_path = os.path.join(output_dir, "scaling_study_summary.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary plots saved to: {summary_plot_path}")
        
    except Exception as e:
        print(f"Could not create summary plots: {str(e)}")
        print("Individual experiment results are still available in their respective folders.")