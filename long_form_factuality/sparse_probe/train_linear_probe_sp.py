import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


def parse_arguments():
    """
    Parse and return command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Train and compare full-dimensional and top-dimension Logistic Regression probes on concatenated language model hidden states.'
    )
    
    # Model and data configurations
    parser.add_argument('--model_name', type=str, required=True,
                        help='Identifier for the model (e.g., Llama3-8B)')
    parser.add_argument('--hf_model_name', type=str, required=True,
                        help='Hugging Face model name (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)')
    parser.add_argument('--scores_filepath', type=str, required=True,
                        help='Path to the fact scores pickle file')
    parser.add_argument('--token_pos', type=str, default='lt', choices=['lt', 'slt', 'fgt'],
                        help='Token position to extract: last token (lt), second last token (slt), first token (fgt)')
    
    # Probe configurations
    parser.add_argument('--layer_range', type=str, default='0-32',
                        help='Comma-separated list of layers or ranges to concatenate (e.g., "0-4,5-9")')
    parser.add_argument('--C_full', type=float, default=0.025,
                        help='Regularization parameter for the full-dimensional probe')
    parser.add_argument('--C_top', type=float, default=0.5,
                        help='Regularization parameter for the top-dimension probe')
    parser.add_argument('--top_dimensions', type=int, default=10,
                        help='Number of top dimensions to select for the secondary probe')
    
    # Device and memory configurations
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation (e.g., cuda, cpu)')
    parser.add_argument('--max_memory', type=str, default='80GIB',
                        help='Maximum memory per device for the model')
    
    # Output configurations
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results and plots')
    
    return parser.parse_args()


def parse_layer_range(layer_range_str):
    """
    Parse a string representing layer indices and ranges into a sorted list of unique integers.
    
    Supported formats:
        - "0,2,4,5"
        - "0-5,10,15"
    """
    layer_set = set()
    parts = layer_range_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                if start > end:
                    raise ValueError(f"Invalid range '{part}': start > end.")
                layer_set.update(range(start, end + 1))
            except Exception as e:
                raise argparse.ArgumentTypeError(f"Invalid layer range '{part}': {e}")
        else:
            try:
                layer = int(part)
                layer_set.add(layer)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid layer index '{part}': must be an integer.")
    return sorted(layer_set)


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
        device (str): Device to load the model on.
        max_memory (str): Maximum memory per device.
    
    Returns:
        tuple: (tokenizer, model)
    """
    if 'llama-3.1-70b' in hf_model_name.lower():
        # path = '/scratch/ms23jh/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/'
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
        device_map="auto", 
        torch_dtype=torch.float16
    )
    print(f'Model "{hf_model_name}" loaded on device "{device}".')
    return tokenizer, model


def extract_hidden_states(tokenizer, model, flat_scores, layer_range, token_pos, device):
    """
    Extract and concatenate hidden states for each factoid at specified layers and token position.
    
    Args:
        tokenizer (AutoTokenizer): Tokenizer instance.
        model (AutoModelForCausalLM): Language model instance.
        flat_scores (list): List of factoids.
        layer_range (list): List of layer indices to extract.
        token_pos (str): Token position to extract ('lt', 'slt', 'fgt').
        device (str): Device to perform computations on.
    
    Returns:
        tuple: (X, y, hidden_size)
            X (np.ndarray): Feature matrix with concatenated hidden states.
            y (np.ndarray): Labels.
            hidden_size (int): Size of each layer's hidden state.
    """
    X_list = []
    y = []
    hidden_size = None
    
    for factoid in tqdm(flat_scores, desc='Extracting Hidden States'):
        atom = factoid['atom']
        is_supported = factoid['is_supported']
        y.append(is_supported)
        
        # Tokenize input
        inputs = tokenizer([atom], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        hidden_states = outputs.hidden_states  # Tuple of (layers + embedding)
        
        # Extract hidden states for the specified layers and token position
        concatenated_features = []
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
            concatenated_features.append(hidden_state)
        
        # Concatenate hidden states across layers
        concatenated_features = np.concatenate(concatenated_features)
        X_list.append(concatenated_features)
        
        # Determine hidden_size based on the first sample
        if hidden_size is None and len(concatenated_features) > 0:
            hidden_size = len(concatenated_features) // len(layer_range)
    
    X = np.array(X_list)
    y = np.array(y)
    
    return X, y, hidden_size


def train_and_evaluate_probe(X, y, C):
    """
    Train Logistic Regression probe with cross-validation and return mean AUROC and coefficients.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        C (float): Regularization parameter.
    
    Returns:
        tuple: (mean_auroc, final_coef)
            mean_auroc (float): Mean AUROC over cross-validation folds.
            final_coef (np.ndarray): Averaged coefficients across folds.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auroc_scores = []
    coefs = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        probe = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
        probe.fit(X_train, y_train)
        
        y_proba = probe.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_proba)
        auroc_scores.append(auroc)
        coefs.append(probe.coef_[0])
    
    mean_auroc = np.mean(auroc_scores)
    final_coef = np.mean(coefs, axis=0)  # Average coefficients across folds
    
    return mean_auroc, final_coef


def select_top_dimensions(coef, top_n):
    """
    Select top N dimensions based on absolute coefficient values.
    
    Args:
        coef (np.ndarray): Coefficient array.
        top_n (int): Number of top dimensions to select.
    
    Returns:
        list: List of top dimension indices.
    """
    top_dims = np.argsort(np.abs(coef))[-top_n:]
    return top_dims.tolist()


def compute_activation_metrics(X, y, top_dimensions, results_dir, model_name, top_n, hidden_size, layer_range):
    """
    Compute and save average activations of top dimensions for each class.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        top_dimensions (list): List of top dimension indices.
        results_dir (str): Directory to save the metrics.
        model_name (str): Identifier for the model.
        top_n (int): Number of top dimensions.
        hidden_size (int): Size of each layer's hidden state.
        layer_range (list): List of layer indices.
    """
    metrics = []
    
    for dim in top_dimensions:
        # Map concatenated feature index to layer and neuron
        layer_idx_in_layer_range = dim // hidden_size
        neuron = dim % hidden_size
        if layer_idx_in_layer_range >= len(layer_range):
            print(f"Warning: Dimension {dim} exceeds the number of layers. Skipping.")
            continue
        layer_number = layer_range[layer_idx_in_layer_range] + 1  # Assuming layer indices start at 0
        
        # Extract activations
        activations_supported = X[y == 1, dim]
        activations_not_supported = X[y == 0, dim]
        
        avg_act_supported = np.mean(activations_supported) if len(activations_supported) > 0 else 0
        std_act_supported = np.std(activations_supported) if len(activations_supported) > 0 else 0
        avg_act_not_supported = np.mean(activations_not_supported) if len(activations_not_supported) > 0 else 0
        std_act_not_supported = np.std(activations_not_supported) if len(activations_not_supported) > 0 else 0
        
        metrics.append({
            'Layer': layer_number,
            'Neuron': neuron,
            'Dimension': dim,
            'Average Activation (Supported)': avg_act_supported,
            'Std Activation (Supported)': std_act_supported,
            'Average Activation (Not Supported)': avg_act_not_supported,
            'Std Activation (Not Supported)': std_act_not_supported
        })
    
    metrics_df = pd.DataFrame(metrics)
    os.makedirs(results_dir, exist_ok=True)
    metrics_filepath = os.path.join(results_dir, f'{model_name}_top{top_n}_activations.csv')
    metrics_df.to_csv(metrics_filepath, index=False)
    print(f'Activation metrics saved to {metrics_filepath}')


def plot_auroc(auroc_full, auroc_top, results_dir, model_name, top_n, C_full, C_top):
    """
    Plot AUROC scores for full and top-dimension probes.
    
    Args:
        auroc_full (float): AUROC score for the full-dimensional probe.
        auroc_top (float): AUROC score for the top-dimension probe.
        results_dir (str): Directory to save the plot.
        model_name (str): Identifier for the model.
        top_n (int): Number of top dimensions.
        C_full (float): Regularization parameter for the full probe.
        C_top (float): Regularization parameter for the top probe.
    """
    plt.figure(figsize=(8, 6))
    probes = ['Full Probe', f'Top {top_n} Dimensions Probe']
    auroc_values = [auroc_full, auroc_top]
    
    plt.bar(probes, auroc_values, color=['blue', 'red'])
    plt.ylim(0.5, 1.0)
    plt.ylabel('AUROC Score')
    plt.title('AUROC Comparison: Full Probe vs Top Dimensions Probe')
    for i, v in enumerate(auroc_values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    os.makedirs(results_dir, exist_ok=True)
    plot_filepath = os.path.join(results_dir, f'{model_name}_AUROC_Comparison.png')
    plt.savefig(plot_filepath, dpi=300)
    print(f'AUROC comparison plot saved to {plot_filepath}')
    plt.close()


def plot_activation_metrics(metrics_filepath, results_dir, model_name, top_n):
    """
    Plot average activations of top dimensions over layers as a single line chart with distinct colors.
    
    Args:
        metrics_filepath (str): Path to the activation metrics CSV file.
        results_dir (str): Directory to save the plots.
        model_name (str): Identifier for the model.
        top_n (int): Number of top dimensions.
    """
    metrics_df = pd.read_csv(metrics_filepath)
    
    plt.figure(figsize=(12, 8))
    
    # Get unique dimensions
    dimensions = metrics_df['Dimension'].unique()
    
    for dim in dimensions:
        dim_metrics = metrics_df[metrics_df['Dimension'] == dim]
        layers = dim_metrics['Layer']
        avg_act_supported = dim_metrics['Average Activation (Supported)']
        avg_act_not_supported = dim_metrics['Average Activation (Not Supported)']
        
        # Plot average activation for supported class
        plt.plot(layers, avg_act_supported, label=f'Dimension {dim} (Supported)', marker='o')
        
        # Plot average activation for not supported class
        plt.plot(layers, avg_act_not_supported, label=f'Dimension {dim} (Not Supported)', marker='x')
    
    plt.title(f'Average Activation Over Layers for Top {top_n} Dimensions')
    plt.xlabel('Layer')
    plt.ylabel('Average Activation')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(results_dir, exist_ok=True)
    plot_filepath = os.path.join(results_dir, f'{model_name}_Top{top_n}_Dimensions_Activation.png')
    plt.savefig(plot_filepath, dpi=300)
    print(f'Activation metrics plot saved to {plot_filepath}')
    plt.close()


def main():
    args = parse_arguments()
    
    # Parse layer range
    layer_range = parse_layer_range(args.layer_range)
    print(f'Layers selected for concatenation: {layer_range}')
    
    # Load and flatten fact scores
    print("Loading and flattening fact scores...")
    flat_scores = load_fact_scores(args.scores_filepath)
    print(f'Number of factoids: {len(flat_scores)}')
    
    # Initialize tokenizer and model
    print("Initializing tokenizer and model...")
    tokenizer, model = initialize_model(args.hf_model_name, args.device, args.max_memory)
    model.eval()  # Set model to evaluation mode
    
    # Extract hidden states and concatenate
    print("Extracting and concatenating hidden states...")
    X, y, hidden_size = extract_hidden_states(tokenizer, model, flat_scores, layer_range, args.token_pos, args.device)
    print(f'Feature matrix shape after concatenation: {X.shape}')
    
    # Train and evaluate full-dimensional probe
    print("\nTraining and evaluating full-dimensional probe...")
    mean_auroc_full, coef_full = train_and_evaluate_probe(X, y, args.C_full)
    print(f'Full Probe AUROC Mean: {mean_auroc_full:.4f}')
    
    # Select top dimensions
    print(f"\nSelecting top {args.top_dimensions} dimensions based on full probe coefficients...")
    top_dims = select_top_dimensions(coef_full, args.top_dimensions)
    print(f'Top {args.top_dimensions} dimensions: {top_dims}')
    
    # Train and evaluate top-dimension probe
    print("\nTraining and evaluating top-dimension probe...")
    X_top = X[:, top_dims]
    mean_auroc_top, coef_top = train_and_evaluate_probe(X_top, y, args.C_top)
    print(f'Top Dimensions Probe AUROC Mean: {mean_auroc_top:.4f}')
    
    # Compare AUROC
    print("\nComparing AUROC between full-dimensional and top-dimension probes...")
    plot_auroc(
        auroc_full=mean_auroc_full,
        auroc_top=mean_auroc_top,
        results_dir=args.results_dir,
        model_name=args.model_name,
        top_n=args.top_dimensions,
        C_full=args.C_full,
        C_top=args.C_top
    )
    
    # Compute and save activation metrics
    print("\nComputing and saving activation metrics...")
    compute_activation_metrics(
        X=X,
        y=y,
        top_dimensions=top_dims,
        results_dir=args.results_dir,
        model_name=args.model_name,
        top_n=args.top_dimensions,
        hidden_size=hidden_size,
        layer_range=layer_range
    )
    
    print("\nAll tasks completed successfully!")


if __name__ == '__main__':
    main()
