import argparse
import os
import pickle
from math import log, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments():
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train and compare full-dimensional and top-dimension Logistic Regression probes on multiple language model hidden states.'
    )

    # Configuration file
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')

    # Optional argument to disable plotting
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting of results.')

    return parser.parse_args()


def load_config(config_path):
    """
    Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def assign_layer_ranges(models):
    """
    Assign default layer ranges based on the model name.

    Args:
        models (list): List of model dictionaries.

    Returns:
        list: Updated list of model dictionaries with 'layer_range' key.
    """
    default_layer_mapping = {
        'Llama3.2-3B': 28,
        'Llama3.1-8B': 32,
        'Llama3.1-70B': 80,
        'Gemma2-9B': 42
    }
    for model in models:
        name = model['model_name']
        if 'layer_range' in model:
            # Use provided layer_range
            model['layer_range'] = model['layer_range']
        elif name in default_layer_mapping:
            total_layers = default_layer_mapping[name]
            model['layer_range'] = list(range(total_layers))
        else:
            raise ValueError(f"Unknown model '{name}'. Please define its layer range manually in the YAML file.")
    return models


def load_fact_scores(filepath):
    """
    Load and flatten fact scores from a pickle file.

    Args:
        filepath (str): Path to the fact scores pickle file.

    Returns:
        list: A flattened list of factoids with 'atom' and 'is_supported' fields.
    """
    with open(filepath, 'rb') as f:
        scores = pickle.load(f)

    # Flatten scores
    if isinstance(scores, dict) and 'decisions' in scores:
        flat_scores = [factoid for decision_group in scores['decisions'] for factoid in decision_group]
    elif isinstance(scores, list):
        # Assuming list of [atom, is_supported]
        flat_scores = [{'atom': score[0], 'is_supported': score[1]} for score in scores]
    else:
        raise ValueError("Unsupported fact scores format.")

    # Ensure each factoid has 'atom' and 'is_supported'
    for factoid in flat_scores:
        if 'atom' not in factoid or 'is_supported' not in factoid:
            raise ValueError("Each factoid must have 'atom' and 'is_supported' fields.")

    return flat_scores


def initialize_model(hf_model_name, device, max_memory):
    """
    Initialize the tokenizer and language model.

    Args:
        hf_model_name (str): Hugging Face model name.
        device (str): Device to load the model on (e.g., 'cuda:0').
        max_memory (str): Maximum memory per device (e.g., '80GIB').

    Returns:
        tuple: (tokenizer, model)
    """
    if 'llama-3.1-70b' in hf_model_name.lower():
        path = snapshot_download(
            repo_id=hf_model_name,
            allow_patterns=['*.json', '*.model', '*.safetensors'],
            ignore_patterns=['pytorch_model.bin.index.json']
        )
    else:
        path = hf_model_name

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto" if 'llama-3.1-70b' in hf_model_name.lower() else "cuda:0",  # Ensures the model is loaded entirely on the specified device
        torch_dtype=torch.float16,
    )
    print(f'Model "{hf_model_name}" loaded on device "{device}".')
    return tokenizer, model


def extract_hidden_states(tokenizer, model, flat_scores, layer_range, token_pos, device):
    """
    Extract hidden states for each factoid at specified layers and token position.

    Args:
        tokenizer (AutoTokenizer): Tokenizer instance.
        model (AutoModelForCausalLM): Language model instance.
        flat_scores (list): List of factoids.
        layer_range (list): List of layer indices to extract.
        token_pos (str): Token position to extract ('lt', 'slt', 'fgt').
        device (str): Device to perform computations on.

    Returns:
        tuple: (X_all, y, hidden_size)
            X_all (dict): Dictionary mapping layer index to feature matrix.
            y (np.ndarray): Labels.
            hidden_size (int): Size of each layer's hidden state.
    """
    X_all = {}
    y = []
    hidden_size = None

    for layer_idx in layer_range:
        X_all[layer_idx] = []

    for factoid in tqdm(flat_scores, desc='Extracting Hidden States'):
        atom = factoid['atom']
        is_supported = factoid['is_supported']
        y.append(is_supported)

        # Tokenize input
        inputs = tokenizer([atom], return_tensors="pt").to(device)  # Ensure inputs are on the correct device

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = outputs.hidden_states  # Tuple of (layers + embedding)

        for layer_idx in layer_range:
            if layer_idx >= len(hidden_states):
                print(f"Warning: Layer index {layer_idx} is out of bounds. Skipping.")
                continue

            if token_pos == 'lt':
                token_index = -1
            elif token_pos == 'slt':
                token_index = -2 if hidden_states[layer_idx].size(1) > 1 else -1
            elif token_pos == 'fgt':
                token_index = 0
            else:
                raise ValueError(f"Unsupported token_pos: {token_pos}")

            # Extract hidden state for the specified token
            hidden_state = hidden_states[layer_idx][0, token_index, :].cpu().numpy()
            X_all[layer_idx].append(hidden_state)

            # Determine hidden_size based on the first sample
            if hidden_size is None and len(hidden_state) > 0:
                hidden_size = len(hidden_state)

    y = np.array(y)
    return X_all, y, hidden_size


def train_probe(X, y, C, n_splits=5):
    """
    Train a Logistic Regression probe using K-Fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        C (float): Regularization parameter.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        tuple: (average_auroc, average_coef)
            average_auroc (float): Average AUROC score across folds.
            average_coef (np.ndarray): Averaged coefficients across folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auroc_scores = []
    coef_list = []

    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the Logistic Regression model
        probe = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
        probe.fit(X_train, y_train)

        # Predict probabilities on the test set
        y_proba = probe.predict_proba(X_test)[:, 1]

        # Calculate AUROC
        auroc = roc_auc_score(y_test, y_proba)
        auroc_scores.append(auroc)

        # Get coefficients
        coef = probe.coef_[0]
        coef_list.append(coef)

        print(f"  Fold {fold}: AUROC = {auroc:.4f}")

    average_auroc = np.mean(auroc_scores)
    average_coef = np.mean(coef_list, axis=0)

    return average_auroc, average_coef


def select_top_dimension(coef):
    """
    Select the single dimension with the highest absolute average coefficient.

    Args:
        coef (np.ndarray): Averaged coefficient array.

    Returns:
        int: Index of the top dimension.
    """
    non_zero_indices = np.where(coef != 0)[0]
    if len(non_zero_indices) == 0:
        return None  # No non-zero coefficients
    top_dim = non_zero_indices[np.argmax(np.abs(coef[non_zero_indices]))]
    return top_dim


def pool_top_dimensions(top_dims_per_layer, auroc_full_probes, layer_range, top_n=3):
    """
    Pool and select the top dimensions based on the new metric: log(occurrences) * mean(AUROC).

    Args:
        top_dims_per_layer (dict): Mapping from layer index to top dimension.
        auroc_full_probes (list): List of AUROC scores for each layer.
        layer_range (list): List of layer indices.
        top_n (int): Number of top dimensions to select based on the new metric.

    Returns:
        list: List of top_n dimension indices sorted by the new metric.
    """
    dim_auroc = {}

    for layer_idx, top_dim in top_dims_per_layer.items():
        if layer_idx in layer_range:
            pos = layer_range.index(layer_idx)
            auroc = auroc_full_probes[pos]
            if top_dim not in dim_auroc:
                dim_auroc[top_dim] = []
            dim_auroc[top_dim].append(auroc)
        else:
            print(f"Warning: Layer index {layer_idx} not found in layer_range.")

    dim_scores = {}

    for dim, auroc_list in dim_auroc.items():
        occurrences = len(auroc_list)
        mean_auroc = np.mean(auroc_list)
        if occurrences > 0:
            # Compute the new metric: log(occurrences) * mean(AUROC)
            score = log(occurrences) * mean_auroc
            dim_scores[dim] = score
        else:
            print(f"Dimension {dim} has zero occurrences, skipping.")

    # Sort dimensions by the computed score in descending order
    sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top_n dimensions based on the sorted scores
    top_dims = [dim for dim, score in sorted_dims[:top_n]]

    return top_dims


def plot_auroc(ax, layer_indices, auroc_full, auroc_sparse, model_name, top_n):
    """
    Plot AUROC scores comparing full-neuron probes and sparse (top dimensions) probes.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        layer_indices (list): List of layer indices.
        auroc_full (list): List of AUROC scores for full-neuron probes.
        auroc_sparse (list): List of AUROC scores for sparse probes.
        model_name (str): Identifier for the model.
        top_n (int): Number of top dimensions used in the sparse probe.
    """
    ax.plot(layer_indices, auroc_full, label='Full-Neuron Probe', marker='o')
    ax.plot(layer_indices, auroc_sparse, label=f'Sparse Probe ({top_n} Dims)', marker='x')
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUROC Score')
    ax.set_title(f'{model_name}: AUROC Comparison Across Layers')
    ax.grid(True)
    # Note: Legends are handled separately in the main function


def plot_layerwise_activation_with_std(ax, top_dimensions, X_all, y, model_name, layer_range):
    """
    Plot layer-wise average activation with standard deviation as shaded areas for the top dimensions.
    Dimensions are represented with unique colors within each subplot. Legends for "Supported" and "Not Supported"
    are handled separately to appear outside the plots.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        top_dimensions (list): List of selected dimension indices.
        X_all (dict): Dictionary mapping layer index to feature matrix.
        y (np.ndarray): Labels.
        model_name (str): Identifier for the model.
        layer_range (list): List of layer indices.
    """
    # Initialize dictionaries to hold mean and std for each layer and class
    activation_stats = {
        dim: {
            'supported_mean': [],
            'supported_std': [],
            'not_supported_mean': [],
            'not_supported_std': []
        } for dim in top_dimensions
    }

    # Iterate over each layer and compute statistics
    for layer_idx in layer_range:
        X = np.array(X_all[layer_idx])
        for dim in top_dimensions:
            if dim >= X.shape[1]:
                # If the dimension does not exist in this layer, append NaN
                activation_stats[dim]['supported_mean'].append(np.nan)
                activation_stats[dim]['supported_std'].append(np.nan)
                activation_stats[dim]['not_supported_mean'].append(np.nan)
                activation_stats[dim]['not_supported_std'].append(np.nan)
                continue

            activations = X[:, dim]
            supported = activations[y == 1]
            not_supported = activations[y == 0]

            activation_stats[dim]['supported_mean'].append(np.mean(supported) if len(supported) > 0 else np.nan)
            activation_stats[dim]['supported_std'].append(np.std(supported, ddof=1) if len(supported) > 1 else np.nan)
            activation_stats[dim]['not_supported_mean'].append(np.mean(not_supported) if len(not_supported) > 0 else np.nan)
            activation_stats[dim]['not_supported_std'].append(np.std(not_supported, ddof=1) if len(not_supported) > 1 else np.nan)

    # Define colors for dimensions
    dimension_colors = plt.cm.viridis(np.linspace(0, 1, len(top_dimensions)))

    for idx, dim in enumerate(top_dimensions):
        color = dimension_colors[idx]

        # Supported Class (Solid Line)
        mean_supported = activation_stats[dim]['supported_mean']
        std_supported = activation_stats[dim]['supported_std']
        # Compute standard error
        se_supported = [std / sqrt(len(X_all[layer_range[i]][y == 1])) if not np.isnan(std) else np.nan
                        for i, std in enumerate(std_supported)]
        ax.plot(layer_range, mean_supported, label=f'Dim. {dim} Supported', color=color, linestyle='-', alpha=0.7)
        ax.errorbar(layer_range, mean_supported, yerr=se_supported, color=color, alpha=0.7, fmt='none', ecolor=color, capsize=3)

        # Not Supported Class (Dashed Line)
        mean_not_supported = activation_stats[dim]['not_supported_mean']
        std_not_supported = activation_stats[dim]['not_supported_std']
        # Compute standard error
        se_not_supported = [std / sqrt(len(X_all[layer_range[i]][y == 0])) if not np.isnan(std) else np.nan
                            for i, std in enumerate(std_not_supported)]
        ax.plot(layer_range, mean_not_supported, label=f'Dim. {dim} Not Supported', color=color, linestyle='--', alpha=0.7)
        ax.errorbar(layer_range, mean_not_supported, yerr=se_not_supported, color=color, alpha=0.7, fmt='none', ecolor='gray', capsize=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Activation')
    ax.set_title(f'{model_name}: Layer-wise Average Activation with Std Error for Top Dimensions')
    ax.grid(True)
    # Note: Legends are handled separately in the main function


def train_sparse_probe(X_sparse, y, C, n_splits=5):
    """
    Train a Logistic Regression probe using only selected dimensions with K-Fold cross-validation.

    Args:
        X_sparse (np.ndarray): Feature matrix with selected dimensions.
        y (np.ndarray): Labels.
        C (float): Regularization parameter.
        n_splits (int): Number of folds for cross-validation.

    Returns:
        float: Average AUROC score across folds.
    """
    average_auroc, _ = train_probe(X_sparse, y, C, n_splits=n_splits)
    return average_auroc


def main():
    args = parse_arguments()
    plotting_enabled = not args.no_plot

    # Load configuration
    config = load_config(args.config)

    # Extract configurations
    models = config.get('models', [])
    if not models:
        raise ValueError("No models specified in the configuration file.")

    # Assign default layer ranges based on model names or use provided layer ranges
    models = assign_layer_ranges(models)

    # Extract probe parameters and general settings
    probe_params = config.get('probe_parameters', {})
    C_full = probe_params.get('C_full', 0.01)
    C_sparse = probe_params.get('C_sparse', 0.1)

    general_settings = config.get('general_settings', {})
    token_pos = general_settings.get('token_pos', 'lt')
    device = general_settings.get('device', 'cuda:0')  # Ensure it's specific, e.g., 'cuda:0'
    max_memory = general_settings.get('max_memory', '80GIB')
    results_dir = general_settings.get('results_dir', './results')

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Prepare separate figures for AUROC and Activation if plotting is enabled
    if plotting_enabled:
        num_models = len(models)
        fig_auroc, axes_auroc = plt.subplots(nrows=1, ncols=num_models, figsize=(8 * num_models, 6))
        fig_activation, axes_activation = plt.subplots(nrows=1, ncols=num_models, figsize=(8 * num_models, 6))

        # Ensure axes are iterable
        if num_models == 1:
            axes_auroc = [axes_auroc]
            axes_activation = [axes_activation]

        # To collect unique dimension labels across all models for legends
        all_dimension_labels = set()

    for idx, model in enumerate(models):
        print(f"\nProcessing model: {model['model_name']}")

        # Initialize tokenizer and model for the current model
        print(f"Initializing model: {model['model_name']}...")
        tokenizer, model_obj = initialize_model(model['hf_model_name'], device, max_memory)

        try:
            # Load and flatten fact scores for the current model
            print(f"Loading and flattening fact scores for {model['model_name']}...")
            flat_scores = load_fact_scores(model['scores_filepath'])
            print(f'Number of factoids: {len(flat_scores)}')

            # Extract hidden states
            print(f"Extracting hidden states for {model['model_name']}...")
            X_all, y, hidden_size = extract_hidden_states(
                tokenizer=tokenizer,
                model=model_obj,
                flat_scores=flat_scores,
                layer_range=model['layer_range'],
                token_pos=token_pos,
                device=device
            )
            print(f'Hidden size per layer: {hidden_size}')

            # Train probes on each layer and select top dimensions using K-Fold cross-validation
            print("Training probes on each layer using K-Fold cross-validation to select top dimensions...")
            top_dims_per_layer = {}
            auroc_full_probes = []
            for layer_idx in model['layer_range']:
                X = np.array(X_all[layer_idx])
                y_labels = y
                print(f"  Training probe for Layer {layer_idx}...")
                average_auroc, average_coef = train_probe(X, y_labels, C=C_full, n_splits=5)
                auroc_full_probes.append(average_auroc)
                top_dim = select_top_dimension(average_coef)
                if top_dim is not None:
                    top_dims_per_layer[layer_idx] = top_dim
                    print(f'    Layer {layer_idx}: Average AUROC = {average_auroc:.4f}, Top Dimension = {top_dim}')
                else:
                    print(f'    Layer {layer_idx}: No non-zero coefficients found.')

            # Retrieve top_n_dims for the current model
            top_n = model.get('top_n_dims', 3)  # Default to 3 if not specified

            # Pool all selected top dimensions based on the new metric: log(occurrences) * mean(AUROC)
            all_top_dims = list(top_dims_per_layer.values())
            if all_top_dims:
                top_dims = pool_top_dimensions(top_dims_per_layer, auroc_full_probes, model['layer_range'], top_n=top_n)
                print(f'Top {top_n} dimensions based on log(occurrences) * mean(AUROC): {top_dims}')
            else:
                top_dims = []
                print("No top dimensions selected across layers.")

            if not top_dims:
                print(f"No top dimensions selected for {model['model_name']}. Skipping plotting for activations.")
                # Still proceed to plot AUROC if enabled
                if plotting_enabled:
                    auroc_sparse_probes = [np.nan] * len(model['layer_range'])
            else:
                if plotting_enabled:
                    # Collect dimension labels for legends
                    for dim in top_dims:
                        all_dimension_labels.add(f'Dim. {dim}')

                # Train sparse probes on each layer using the selected top dimensions with K-Fold cross-validation
                print(f"Training sparse probes on each layer using the top {top_n} dimensions with K-Fold cross-validation...")
                auroc_sparse_probes = []
                for layer_idx in model['layer_range']:
                    X = np.array(X_all[layer_idx])
                    # Ensure top dimensions are within the layer's hidden size
                    valid_dims = [dim for dim in top_dims if dim < X.shape[1]]
                    if not valid_dims:
                        print(f"  Warning: None of the top dimensions are valid for layer {layer_idx}. Assigning AUROC as NaN.")
                        auroc_sparse_probes.append(np.nan)
                        continue
                    X_sparse = X[:, valid_dims]
                    y_labels = y
                    print(f"  Training sparse probe for Layer {layer_idx} using dimensions {valid_dims}...")
                    mean_auroc_sparse = train_sparse_probe(X_sparse, y_labels, C=C_sparse, n_splits=5)
                    auroc_sparse_probes.append(mean_auroc_sparse)
                    print(f'    Layer {layer_idx}: Sparse Probe Average AUROC = {mean_auroc_sparse:.4f}')

            # Identify and print the layer with the highest AUROC
            if auroc_full_probes:
                max_auroc_idx = np.argmax(auroc_full_probes)
                best_layer = model['layer_range'][max_auroc_idx]
                best_auroc = auroc_full_probes[max_auroc_idx]
                print(f'\nLayer with highest AUROC: Layer {best_layer} with AUROC = {best_auroc:.4f}')

                # If top dimensions are selected, compute activation statistics on the best layer
                if top_dims:
                    X_best_layer = np.array(X_all[best_layer])
                    # Select top dimensions that are valid for the best layer
                    valid_top_dims = [dim for dim in top_dims if dim < X_best_layer.shape[1]]
                    if not valid_top_dims:
                        print(f"No valid top dimensions for the best layer {best_layer}. Skipping activation statistics.")
                    else:
                        # Initialize dictionaries to hold mean and std error for each dimension and class
                        activation_stats = {}

                        for dim in valid_top_dims:
                            activations_supported = X_best_layer[y == 1, dim]
                            activations_not_supported = X_best_layer[y == 0, dim]

                            mean_sup = np.mean(activations_supported) if len(activations_supported) > 0 else np.nan
                            std_sup = np.std(activations_supported, ddof=1) if len(activations_supported) > 1 else np.nan
                            se_sup = std_sup / sqrt(len(activations_supported)) if len(activations_supported) > 1 else np.nan

                            mean_not_sup = np.mean(activations_not_supported) if len(activations_not_supported) > 0 else np.nan
                            std_not_sup = np.std(activations_not_supported, ddof=1) if len(activations_not_supported) > 1 else np.nan
                            se_not_sup = std_not_sup / sqrt(len(activations_not_supported)) if len(activations_not_supported) > 1 else np.nan

                            activation_stats[dim] = {
                                'Supported': {'mean': mean_sup, 'se': se_sup},
                                'Not Supported': {'mean': mean_not_sup, 'se': se_not_sup}
                            }

                        # Print activation statistics for each dimension
                        print(f'\nActivation Statistics on Best Layer (Layer {best_layer}):')
                        for dim, stats in activation_stats.items():
                            print(f'  Dimension {dim}:')
                            print(f'    Supported Class: Average Activation = {stats["Supported"]["mean"]:.4f}, Standard Error = {stats["Supported"]["se"]:.4f}')
                            print(f'    Not Supported Class: Average Activation = {stats["Not Supported"]["mean"]:.4f}, Standard Error = {stats["Not Supported"]["se"]:.4f}')
            else:
                print("No AUROC scores available to determine the best layer.")

            # Plot AUROC comparison if plotting is enabled
            if plotting_enabled:
                print("Plotting AUROC comparison...")
                plot_auroc(
                    ax=axes_auroc[idx],
                    layer_indices=model['layer_range'],
                    auroc_full=auroc_full_probes,
                    auroc_sparse=auroc_sparse_probes,
                    model_name=model['model_name'],
                    top_n=top_n
                )

                # Plot layer-wise activation averages with std for the top dimensions
                if top_dims:
                    print("Plotting layer-wise activation averages with standard error...")
                    plot_layerwise_activation_with_std(
                        ax=axes_activation[idx],
                        top_dimensions=top_dims,
                        X_all=X_all,
                        y=y,
                        model_name=model['model_name'],
                        layer_range=model['layer_range']
                    )
                else:
                    axes_activation[idx].set_title(f'{model["model_name"]}: No Top Dimensions Selected')
                    axes_activation[idx].axis('off')

            # Save AUROC data
            print("Saving AUROC data...")
            auroc_data = pd.DataFrame({
                'Layer': model['layer_range'],
                'Full Probe AUROC': auroc_full_probes,
                'Sparse Probe AUROC': auroc_sparse_probes
            })
            auroc_data_filepath = os.path.join(results_dir, f'{model["model_name"]}_AUROC_Data.csv')
            auroc_data.to_csv(auroc_data_filepath, index=False)
            print(f'AUROC data saved to {auroc_data_filepath}')

            # Save Activation data
            if top_dims:
                print("Saving Activation data...")
                # Prepare activation data for top dimensions
                activation_rows = []
                for dim in top_dims:
                    for layer_idx in model['layer_range']:
                        if dim >= hidden_size:
                            # If the dimension does not exist in this layer, skip
                            continue
                        activations = X_all[layer_idx]
                        if not activations:
                            continue
                        activations = np.array(activations)
                        if dim >= activations.shape[1]:
                            continue
                        # Calculate mean and std for supported and not supported
                        supported_activations = activations[y == 1, dim]
                        not_supported_activations = activations[y == 0, dim]
                        avg_sup = np.mean(supported_activations) if len(supported_activations) > 0 else np.nan
                        std_sup = np.std(supported_activations, ddof=1) if len(supported_activations) > 1 else np.nan
                        se_sup = std_sup / sqrt(len(supported_activations)) if len(supported_activations) > 1 else np.nan

                        avg_not_sup = np.mean(not_supported_activations) if len(not_supported_activations) > 0 else np.nan
                        std_not_sup = np.std(not_supported_activations, ddof=1) if len(not_supported_activations) > 1 else np.nan
                        se_not_sup = std_not_sup / sqrt(len(not_supported_activations)) if len(not_supported_activations) > 1 else np.nan

                        activation_rows.append({
                            'Layer': layer_idx,
                            'Neuron': 'N/A',  # Placeholder as neuron info is not retained
                            'Dimension': dim,
                            'Average Activation (Supported)': avg_sup,
                            'Std Activation (Supported)': std_sup,
                            'Standard Error (Supported)': se_sup,
                            'Average Activation (Not Supported)': avg_not_sup,
                            'Std Activation (Not Supported)': std_not_sup,
                            'Standard Error (Not Supported)': se_not_sup
                        })

                activation_df = pd.DataFrame(activation_rows)
                activation_csv_filename = f'{model["model_name"]}_top{top_n}_activations.csv'
                activation_csv_path = os.path.join(results_dir, activation_csv_filename)
                activation_df.to_csv(activation_csv_path, index=False)
                print(f'Activation data saved to {activation_csv_path}')
            else:
                print(f"No activation data to save for {model['model_name']}.")

            # Save top dimensions data
            if top_dims_per_layer:
                print("Saving top dimensions per layer data...")
                top_dims_data = pd.DataFrame([
                    {'Layer': layer, 'Top Dimension': dim} for layer, dim in top_dims_per_layer.items()
                ])
                top_dims_data_filepath = os.path.join(results_dir, f'{model["model_name"]}_Top_Dimensions_Per_Layer.csv')
                top_dims_data.to_csv(top_dims_data_filepath, index=False)
                print(f'Top dimensions per layer data saved to {top_dims_data_filepath}')
            else:
                print(f"No top dimensions data to save for {model['model_name']}.")

        finally:
            # Unload the model to free GPU memory
            del model_obj
            torch.cuda.empty_cache()
            print(f'Model "{model["model_name"]}" unloaded from GPU.')

    # After processing all models, handle legends and save plots if plotting is enabled
    if plotting_enabled:
        # ----- AUROC Plots Legend -----
        # Collect handles and labels from all AUROC subplots
        handles_auroc = []
        labels_auroc = []
        for ax in axes_auroc:
            h, l = ax.get_legend_handles_labels()
            handles_auroc.extend(h)
            labels_auroc.extend(l)
        # Remove duplicates
        by_label_auroc = dict(zip(labels_auroc, handles_auroc))
        # Add the legend above the AUROC plots
        fig_auroc.legend(by_label_auroc.values(), by_label_auroc.keys(), loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

        # Save AUROC plot as PDF
        auroc_pdf_path = os.path.join(results_dir, 'AUROC_Plots.pdf')
        fig_auroc.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for legend
        fig_auroc.savefig(auroc_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f'\nAUROC plots saved to {auroc_pdf_path}')
        plt.close(fig_auroc)

        # ----- Activation Plots Legend -----
        if all_dimension_labels:
            # Create handles for dimensions
            dim_handles = []
            dim_labels = []
            sorted_dims = sorted(all_dimension_labels, key=lambda x: int(x.split('. ')[-1]))
            for dim_label in sorted_dims:
                color_idx = list(all_dimension_labels).index(dim_label) / len(all_dimension_labels)
                color = plt.cm.viridis(color_idx)
                handle, = plt.plot([], [], color=color, label=dim_label, linestyle='-')
                dim_handles.append(handle)
                dim_labels.append(dim_label)

            # Create handles for support classes
            support_handles = [
                plt.Line2D([0], [0], color='black', linestyle='-', label='Supported'),
                plt.Line2D([0], [0], color='black', linestyle='--', label='Not Supported')
            ]
            support_labels = ['Supported', 'Not Supported']

            # Combine all handles and labels
            handles_activation = dim_handles + support_handles
            labels_activation = dim_labels + support_labels

            # Add the legend above the Activation plots
            fig_activation.legend(handles_activation, labels_activation, loc='upper center', ncol=len(handles_activation), bbox_to_anchor=(0.5, 1.05))
        else:
            # If no dimensions were selected, only add support class legends
            support_handles = [
                plt.Line2D([0], [0], color='black', linestyle='-', label='Supported'),
                plt.Line2D([0], [0], color='black', linestyle='--', label='Not Supported')
            ]
            support_labels = ['Supported', 'Not Supported']
            fig_activation.legend(support_handles, support_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

        # Save Activation plot as PDF
        activation_pdf_path = os.path.join(results_dir, 'Activation_Plots.pdf')
        fig_activation.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make space for legend
        fig_activation.savefig(activation_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f'Activation plots saved to {activation_pdf_path}')
        plt.close(fig_activation)


if __name__ == '__main__':
    main()
