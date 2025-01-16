"""Deprecated training script -> see fp_exps for latest ones"""

import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

import argparse

from utils.eval_utils import auroc, bootstrap_func

# Define a mapping from model names to their corresponding number of layers
MODEL_LAYER_COUNTS = {
    'Llama3.1-8B': 33,
    'Gemma2-9B': 43,
    'Llama3.1-70B': 81,
    'Llama3.2-3B': 29,
}

# Set the default number of layers if the model is not in the mapping
DEFAULT_LAYER_COUNT = 33

def load_fact_scores(model_name, data_dir='./FActScore/data/', scores_filename=None):
    """Load fact scores from pickled file."""
    if scores_filename is not None:
        filepath = scores_filename
    else:
        filepath = os.path.join(data_dir, f'{model_name}_fact_scores_nent_30_temp_0.5_maxtok_512_api_gpt-4o-mini.pkl')
    with open(filepath, 'rb') as f:
        scores = pickle.load(f)
    return scores

def flatten_scores(scores):
    """Flatten the nested list of scores into a flat list of factoids."""
    if isinstance(scores, dict) and 'decisions' in scores:
        # Original format: {'decisions': [[factoid1, factoid2, ...], [factoid3, factoid4, ...], ...]}
        flat_scores = [factoid for decision_group in scores['decisions'] for factoid in decision_group]
        # Ensure each factoid has 'is_supported' key; if not, assign default label (e.g., 0)
        for factoid in flat_scores:
            if 'is_supported' not in factoid:
                factoid['is_supported'] = 0  # or another default value as appropriate
        return flat_scores
    elif isinstance(scores, list):
        # Check if the data is in the new format
        if all(isinstance(inner, list) and len(inner) == 2 for inner in scores):
            # New format: list of [ 'atomic claim', True/False ]
            flattened = []
            for atom, is_supported in scores:
                is_supported_int = 1 if is_supported else 0
                flattened.append({'atom': atom, 'is_supported': is_supported_int})
            return flattened
        elif all(isinstance(inner, dict) and len(inner) == 2 for inner in scores):
            return scores  # already flattened
        else:
            # Handle other list formats (if any)
            # Existing code to handle list of data points, each containing extracted_facts and truth_fact_level
            flattened = []
            for datum in scores:
                extracted_facts = datum[4]
                truth_fact_level = datum[5]
                for fact, truth in zip(extracted_facts, truth_fact_level):
                    # Map 'is_supported': 1 if True, else 0 (for MINOR and MAJOR)
                    is_supported = 1 if truth == True else 0
                    flattened.append({'atom': fact, 'is_supported': is_supported})
            return flattened
    else:
        raise ValueError("Unsupported score format.")

def initialize_model(hf_model_name, device='cuda', max_memory='80GIB'):
    """Initialize the tokenizer and model."""
    if 'llama-3.1-405b' in hf_model_name.lower():
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

def load_or_compute_inputs(model_name, tokenizer, llm, scores, token_pos='lt', layer_range=None, cache_dir='/scratch/ms23jh/facts', data_name=None):
    """Load or compute the hidden states for each factoid."""
    if layer_range is None:
        layer_range = list(range(33))  # Default layer range if not provided

    # Use data_name if provided, else default to model_name
    if data_name is None:
        data_name = model_name

    # Define layer range string for filename (e.g., '5-9')
    layer_range_str = f"{layer_range[0]}-{layer_range[-1]}" if layer_range else "all"

    token_pos_suffix = '' if token_pos == 'lt' else f'_{token_pos}'
    cache_filepath = os.path.join(cache_dir, f'{data_name}_layers_{layer_range_str}{token_pos_suffix}_probe_inputs.pkl')

    if False and os.path.exists(cache_filepath):
        with open(cache_filepath, 'rb') as f:
            X_all, y = pickle.load(f)
        print(f'Loaded precomputed inputs from {cache_filepath}')
    else:
        print('Computing inputs...')
        X_all = defaultdict(list)  # layer -> samples
        y = []
        for factoid in tqdm(scores, desc='Processing factoids'):
            try:
                inputs = tokenizer([factoid['atom']], return_tensors="pt").to('cuda')
                with torch.no_grad():
                    outputs = llm(**inputs, output_hidden_states=True, return_dict=True)
                hiddens = outputs.hidden_states

                sequence_length = hiddens[0].size(1)
                for layer_idx in layer_range:
                    if token_pos == 'lt':
                        token_index = -1  # Last token
                    elif token_pos == 'slt':
                        token_index = -2 if hiddens[layer_idx].size(1) > 1 else -1  # Second last token if possible
                    elif token_pos == 'fgt':
                        token_index = 0   # First token
                    else:
                        raise ValueError(f"Unsupported token_pos: {token_pos}")

                    # Ensure layer_idx is within the hidden states
                    if layer_idx >= len(hiddens):
                        print(f"Warning: Layer index {layer_idx} is out of bounds for the model. Skipping this layer.")
                        continue

                    # Handle cases where token_index might be out of bounds
                    if abs(token_index) > sequence_length:
                        print(f"Warning: token_index {token_index} out of bounds for sequence length {sequence_length}. Using last token.")
                        token_index = -1

                    hidden_state = hiddens[layer_idx][0, token_index, :].cpu()
                    X_all[layer_idx].append(hidden_state)
                y.append(factoid['is_supported'])
            except Exception as e:
                print(f"Error processing factoid: {factoid}")
                print(e)
                continue

        # os.makedirs(cache_dir, exist_ok=True)
        # with open(cache_filepath, 'wb') as f:
        #     pickle.dump((X_all, y), f)
        # print(f'Saved computed inputs to {cache_filepath}')
    return X_all, y

def run_concat_layer_probing(
    X_all, y, layer_range, group_size=5, C=0.1, max_iter=1000, 
    results_dir='./metrics', model_name='model', token_pos='lt'
):
    """
    Run concatenated layer probing experiments without a test split.

    Args:
        X_all (dict): Dictionary mapping layer indices to hidden states.
        y (list): List of labels corresponding to each factoid.
        layer_range (list): List of layer indices to include in probing.
        group_size (int, optional): Number of layers per probe group. Defaults to 5.
        C (float, optional): Regularization parameter for Logistic Regression. Defaults to 0.1.
        max_iter (int, optional): Maximum iterations for Logistic Regression solver. Defaults to 1000.
        results_dir (str, optional): Directory to save results. Defaults to './metrics'.
        model_name (str, optional): Name identifier for the model. Defaults to 'model'.
        token_pos (str, optional): Token position identifier. Defaults to 'lt'.

    Returns:
        str: Filepath to the best performing probe.
    """
    mean_aurocs = []
    
    # Adjust layer_groups to ensure all have exactly group_size layers
    layer_groups = []
    total_layers = len(layer_range)
    overlap = group_size - 1  # Overlap by group_size -1 to shift window
    
    for i in range(0, total_layers, group_size - overlap):
        group = layer_range[i:i+group_size]
        if len(group) < group_size:
            # Shift the window back to include the last group_size layers
            group = layer_range[-group_size:]
            layer_groups.append(group)
            break
        layer_groups.append(group)
    
    # Remove potential duplicates
    unique_layer_groups = []
    seen = set()
    for group in layer_groups:
        group_tuple = tuple(group)
        if group_tuple not in seen:
            unique_layer_groups.append(group)
            seen.add(group_tuple)
    
    layer_groups = unique_layer_groups

    # Initialize variables to track the best probe based on validation AUROC
    best_val_auroc = -np.inf
    best_probe = None
    best_layer_group = None
    token_pos_suffix = '' if token_pos == 'lt' else f'_{token_pos}'

    for layers_in_group in layer_groups:
        layer_start = layers_in_group[0]
        layer_end = layers_in_group[-1]
        print(f'\nTesting probes on layers {layers_in_group}')

        # Collect and concatenate hidden states from the specified layers
        X_list = [X_all[layer_idx] for layer_idx in layers_in_group]
        X_list = [torch.stack(X_layer).cpu().numpy() for X_layer in X_list]
        X_group = np.concatenate(X_list, axis=-1)
        y_np = np.array(y)

        print(f'Feature matrix shape: {X_group.shape}, Labels shape: {y_np.shape}, Positive class ratio: {np.mean(y_np):.4f}')

        # Cross-validation on the entire dataset
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        val_auroc_scores = []

        for fold_idx, (train_index, val_index) in enumerate(skf.split(X_group, y_np)):
            X_train, X_val = X_group[train_index], X_group[val_index]
            y_train, y_val = y_np[train_index], y_np[val_index]

            # Initialize Logistic Regression probe with specified parameters
            probe = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=max_iter)

            # Fit the model on the training data
            probe.fit(X_train, y_train)

            # Predict probabilities on the validation data
            y_val_proba = probe.predict_proba(X_val)[:, 1]

            # Compute AUROC score for the current fold
            val_auroc = roc_auc_score(y_val, y_val_proba)
            val_auroc_scores.append(val_auroc)

            print(f'Fold {fold_idx + 1}: Validation AUROC = {val_auroc:.4f}')

        # Calculate mean and standard deviation of AUROC scores on validation folds
        mean_auroc_val = np.mean(val_auroc_scores)
        std_auroc_val = np.std(val_auroc_scores, ddof=1)

        # Retrain the model on the entire dataset
        probe_full = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=max_iter)
        probe_full.fit(X_group, y_np)

        # Record the results
        mean_aurocs.append({
            'Layer Group': f'{layer_start}-{layer_end}',
            'Validation AUROC Mean': mean_auroc_val,
            'Validation AUROC Std': std_auroc_val
        })

        print(f'Layer Group {layer_start}-{layer_end}: Val AUROC Mean={mean_auroc_val:.4f}, Val AUROC Std={std_auroc_val:.4f}')

        # Check if this is the best performing probe based on validation AUROC mean
        if mean_auroc_val > best_val_auroc:
            best_val_auroc = mean_auroc_val
            best_probe = probe_full  # Trained on the entire dataset
            best_layer_group = (layer_start, layer_end)

    # Save results to CSV
    df_results = pd.DataFrame(mean_aurocs)
    os.makedirs(results_dir, exist_ok=True)
    results_filepath = os.path.join(results_dir, f'{model_name}{token_pos_suffix}_layer_group{group_size}_aurocs.csv')
    df_results.to_csv(results_filepath, index=False)
    print(f'\nResults saved to {results_filepath}')
    print(df_results)

    # Save the best performing probe based on validation AUROC
    layer_start, layer_end = best_layer_group
    probe_filename = f'{model_name}{token_pos_suffix}_best_probe_layers_{layer_start}-{layer_end}_C_{C}.pkl'
    probe_filepath = os.path.join(results_dir, probe_filename)
    with open(probe_filepath, 'wb') as f:
        pickle.dump({
            'probe': best_probe,
            'layer_group': best_layer_group,
            'token_pos': token_pos,
            'C': C,
            'max_iter': max_iter
        }, f)
    print(f'Best performing probe saved to {probe_filepath} based on Validation AUROC of {best_val_auroc:.4f}')

    return probe_filepath

def test_on_new_data(model_name, tokenizer, llm, probe_filepath, new_scores_filepath, cache_dir='/scratch/ms23jh/facts', test_data_name=None):
    """Test the saved probe on new unseen fact scores."""
    # Load the best probe
    with open(probe_filepath, 'rb') as f:
        saved_data = pickle.load(f)
    best_probe = saved_data['probe']
    best_layer_group = saved_data['layer_group']
    token_pos = saved_data['token_pos']

    # Load new fact scores
    new_scores = load_fact_scores(model_name, scores_filename=new_scores_filepath)
    new_scores = flatten_scores(new_scores)

    # Load or compute inputs for new data
    # We'll need to get the hidden states for the best layer group
    layer_range = list(range(best_layer_group[0], best_layer_group[1]+1))
    X_all_new, y_new = load_or_compute_inputs(
        model_name,
        tokenizer,
        llm,
        new_scores,
        token_pos=token_pos,
        layer_range=layer_range,
        cache_dir=cache_dir,
        data_name=test_data_name  # Use test_data_name for cache file naming
    )

    # Concatenate the hidden states from the specified layers
    X_list_new_raw = [X_all_new[layer_idx] for layer_idx in layer_range]

    # Check for empty layers and handle them
    for idx, X_layer in enumerate(X_list_new_raw):
        if not X_layer:
            print(f"Warning: Layer {layer_range[idx]} has no data. Inserting a dummy tensor to maintain dimensionality.")
            # Insert a dummy tensor to maintain dimensions
            dummy_tensor = torch.zeros(llm.config.hidden_size)
            X_list_new_raw[idx] = [dummy_tensor]

    try:
        X_list_new = [torch.stack(X_layer).cpu().numpy() for X_layer in X_list_new_raw]
    except RuntimeError as e:
        print("Error stacking tensors:", e)
        return

    # Concatenate along the last axis
    X_group_new = np.concatenate(X_list_new, axis=-1)
    y_np_new = np.array(y_new)

    # Predict using the best probe
    y_pred_proba_new = best_probe.predict_proba(X_group_new)[:, 1]
    bs_auc = bootstrap_func(y_np_new, y_pred_proba_new, auroc, rn=42)
    
    # Compute AUROC
    if np.unique(y_np_new).size > 1:
        test_auroc_new = bs_auc['mean']
        test_auroc_new_std = bs_auc['bootstrap']['std_err']
        print(f'New Data AUROC: {test_auroc_new:.4f}, standard error: {test_auroc_new_std}')
    else:
        test_auroc_new = None
        print('Cannot compute AUROC: Only one class present in y_true.')

    return test_auroc_new

def main():
    parser = argparse.ArgumentParser(description='Probing Experiments')
    parser.add_argument('--hf_model_name', type=str, default='meta-llama/Llama-3.1-70B-Instruct', help='Name of the Hugging Face model')
    parser.add_argument('--model_name', type=str, default='Llama3.1-70B', help='Model name identifier')
    parser.add_argument('--token_pos', type=str, default='lt', choices=['lt', 'slt', 'fgt'], help='Token position: last token (lt), second last token (slt), first token (fgt)')
    parser.add_argument('--C', type=float, default=0.5, help='Regularization parameter for Logistic Regression')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for solver convergence')
    parser.add_argument('--group_size', type=int, default=5, help='Number of layers to concatenate for probing')
    parser.add_argument('--cache_dir', type=str, default='/scratch/ms23jh/facts', help='Directory to cache computed inputs')
    parser.add_argument('--results_dir', type=str, default='./metrics', help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='./FActScore/data', help='Directory containing data')
    parser.add_argument('--layer_range', type=int, nargs='+', default=None, help='Range of layers to use')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--max_memory', type=str, default='80GIB', help='Max memory per device for model')
    parser.add_argument('--test_new_data', type=str, default=None, help='Path to new unseen fact scores to test on')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Mode: train or eval')
    parser.add_argument('--probe_filepath', type=str, default=None, help='Path to the pre-trained probe file (required in eval mode)')
    parser.add_argument('--scores_filepath', type=str, default=None, help='Path to the scores data file')
    parser.add_argument('--use_train_test_split', action='store_true', help='Flag to perform a train/test split')
    parser.add_argument('--test_set_dir', type=str, default='./FActScore/data/', help='Path to save the 20% test set')
    parser.add_argument('--train_fraction', type=float, default=1.0, help='Fraction of training data to use (between 0 and 1). For example, 0.5 means 50% of the training data.')

    args = parser.parse_args()

    # **Validation for train_fraction**
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError("Argument --train_fraction must be between 0 (exclusive) and 1 (inclusive).")

    if args.layer_range is None:
        # Retrieve the number of layers for the specified model_name, defaulting to DEFAULT_LAYER_COUNT
        num_layers = MODEL_LAYER_COUNTS.get(args.model_name, DEFAULT_LAYER_COUNT)
        
        # Generate the layer_range as a list of layer indices
        args.layer_range = list(range(num_layers))
        
        print(f"Layer range not specified. Using all {num_layers} layers for model '{args.model_name}'.")
    else:
        # Remove duplicates and sort
        args.layer_range = sorted(list(set(args.layer_range)))
        print(f'Using specified layer range: {args.layer_range}')

    # Initialize model
    # tokenizer, llm = initialize_model(args.hf_model_name, device=args.device, max_memory=args.max_memory)

    if args.mode == 'train':
        # Load fact scores
        scores = load_fact_scores(args.model_name, args.data_dir, scores_filename=args.scores_filepath)

        # Flatten the scores
        flat_scores = flatten_scores(scores)
        print(f'Number of samples: {len(flat_scores)}')

        if args.use_train_test_split:
            # Perform train-test split
            print('Performing train-test split...')
            original_decisions = scores['decisions']
            y_labels = [fact['is_supported'] for fact in flat_scores]
            train_flat_scores, test_flat_scores = train_test_split(
                flat_scores, test_size=0.2, stratify=y_labels, random_state=42
            )

            # **Apply train_fraction to the training set**
            if args.train_fraction < 1.0:
                print(f'Applying train_fraction: Using {args.train_fraction*100:.0f}% of the training data.')
                train_flat_scores, _ = train_test_split(
                    train_flat_scores, train_size=args.train_fraction, stratify=[fact['is_supported'] for fact in train_flat_scores], random_state=42
                )
                print(f'Number of training samples after applying train_fraction: {len(train_flat_scores)}')
            else:
                print('Using 100% of the training data.')

            # Create a set for faster lookup
            test_set = set()
            for fact in test_flat_scores:
                # Assuming each factoid is unique, you can use tuple of (atom, is_supported) as identifier
                test_set.add((fact['atom'], fact['is_supported']))

            # Reconstruct the test_decisions preserving the original nested structure
            test_decisions = []
            for decision_group in original_decisions:
                test_group = []
                for factoid in decision_group:
                    if (factoid['atom'], factoid['is_supported']) in test_set:
                        test_group.append(factoid)
                test_decisions.append(test_group)

            # Save the test set in the desired nested format
            test_set_dict = {'decisions': test_decisions}
            test_set_file = os.path.join(args.test_set_dir, f'{args.model_name}_test_set.pkl')
            with open(test_set_file, 'wb') as f:
                pickle.dump(test_set_dict, f)
            print(f'Test set saved to {test_set_file}')
            import sys; sys.exit(0)
            # Optionally save the test set if needed (as a nested list)
            # test_set_file = os.path.join(args.test_set_dir, f'{args.model_name}_test_set.pkl')
            # with open(test_set_file, 'wb') as f:
            #     pickle.dump(test_flat_scores, f)
            # print(f'Test set saved to {test_set_file}')

            # Use the training set for further processing
            flat_scores = train_flat_scores
            print(f'Number of training samples after split: {len(flat_scores)}')
        else:
            print('Using the entire dataset without train-test split.')

            # **Apply train_fraction to the entire dataset**
            if args.train_fraction < 1.0:
                print(f'Applying train_fraction: Using {args.train_fraction*100:.0f}% of the data.')
                flat_scores, _ = train_test_split(
                    flat_scores, train_size=args.train_fraction, stratify=[fact['is_supported'] for fact in flat_scores], random_state=42
                )
                print(f'Number of training samples after applying train_fraction: {len(flat_scores)}')
            else:
                print('Using 100% of the data.')

        # Load or compute inputs
        X_all, y = load_or_compute_inputs(
            args.model_name,
            tokenizer,
            llm,
            flat_scores,
            args.token_pos,
            args.layer_range,
            args.cache_dir,
            data_name=args.model_name
        )

        # Run concatenated layer probing
        probe_filepath = run_concat_layer_probing(
            X_all, y,
            args.layer_range,
            args.group_size,
            args.C,
            args.max_iter,
            args.results_dir,
            args.model_name,
            args.token_pos
        )

        # Test on new data if provided
        if args.test_new_data is not None:
            test_on_new_data(
                args.model_name,
                tokenizer,
                llm,
                probe_filepath,
                args.test_new_data,
                cache_dir=args.cache_dir,
            )

    elif args.mode == 'eval':
        if args.probe_filepath is None:
            print('Error: In eval mode, --probe_filepath must be specified.')
            return

        if args.test_new_data is None:
            print('Error: In eval mode, --test_new_data must be specified.')
            return

        # Test on new data using the pre-trained probe
        test_on_new_data(
            args.model_name,
            tokenizer,
            llm,
            args.probe_filepath,
            args.test_new_data,
            cache_dir=args.cache_dir,
        )

if __name__ == '__main__':
    main()
