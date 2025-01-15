import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse
import time  # Importing time module for benchmarking

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_utils import auroc, bootstrap_func  # Ensure this is available in your environment

# Import additional classifiers
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Define a mapping from model names to their corresponding number of layers
MODEL_LAYER_COUNTS = {
    'Llama3.1-8B': 33,
    'Gemma2-9B': 43,
    'Llama3.1-70B': 81,
    'Llama3.2-3B': 29,
}

# Set the default number of layers if the model is not in the mapping
DEFAULT_LAYER_COUNT = 33

def load_fact_scores(model_name, data_dir='./FActScore/data/labeled', scores_filename=None):
    """Load fact scores from pickled file."""
    if scores_filename is not None:
        filepath = scores_filename
    else:
        filepath = os.path.join(data_dir, f'{model_name}_fact_scores_nent_200_temp_0.1_maxtok_128_api_gpt-4o-mini.pkl')
    with open(filepath, 'rb') as f:
        scores = pickle.load(f)
    return scores

def flatten_scores(scores):
    """Flatten the nested list of scores into a flat list of factoids."""
    if isinstance(scores, dict) and 'decisions' in scores:
        flat_scores = [factoid for decision_group in scores['decisions'] for factoid in decision_group]
        for factoid in flat_scores:
            if 'is_supported' not in factoid:
                factoid['is_supported'] = 0
        return flat_scores
    elif isinstance(scores, list):
        if all(isinstance(inner, list) and len(inner) == 2 for inner in scores):
            flattened = []
            for atom, is_supported in scores:
                is_supported_int = 1 if is_supported else 0
                flattened.append({'atom': atom, 'is_supported': is_supported_int})
            return flattened
        elif all(isinstance(inner, dict) and len(inner) == 2 for inner in scores):
            return scores
        else:
            flattened = []
            for datum in scores:
                extracted_facts = datum[4]
                truth_fact_level = datum[5]
                for fact, truth in zip(extracted_facts, truth_fact_level):
                    is_supported = 1 if truth == True else 0
                    flattened.append({'atom': fact, 'is_supported': is_supported})
            return flattened
    else:
        raise ValueError("Unsupported score format.")

def initialize_model(hf_model_name, device='cuda', max_memory='80GIB'):
    """Initialize the tokenizer and model."""
    # Determine the path based on hf_model_name
    if 'llama-3.1-70b' in hf_model_name.lower():
        path = '/scratch/ms23jh/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/'
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

def parse_layer_range(s):
    """
    Parse a string representing layer indices and ranges into a sorted list of unique integers.
    
    Supported formats:
        - "0,2,4,5"
        - "0-5,10,20-25"
        - "20-80"
    
    Example:
        Input: "0-2,4,5"
        Output: [0, 1, 2, 4, 5]
    """
    layer_set = set()
    parts = s.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range
            try:
                start, end = part.split('-')
                start = int(start)
                end = int(end)
                if start > end:
                    raise ValueError(f"Invalid range '{part}': start ({start}) > end ({end}).")
                layer_set.update(range(start, end + 1))
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"Invalid layer range '{part}': {e}")
        else:
            # Handle single layer
            try:
                layer = int(part)
                layer_set.add(layer)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid layer index '{part}': must be an integer.")
    return sorted(layer_set)

def load_or_compute_inputs(model_name, tokenizer, llm, scores, token_pos='lt', layer_range=None, cache_dir='/scratch/ms23jh/facts', data_name=None):
    """Load or compute the hidden states for each factoid."""
    if layer_range is None:
        layer_range = list(range(33))  # Default layer range if not provided

    if data_name is None:
        data_name = model_name

    layer_range_str = f"{layer_range[0]}-{layer_range[-1]}" if layer_range else "all"
    token_pos_suffix = '' if token_pos == 'lt' else f'_{token_pos}'
    cache_filepath = os.path.join(cache_dir, f'{data_name}_layers_{layer_range_str}{token_pos_suffix}_probe_inputs.pkl')

    if False and os.path.exists(cache_filepath):
        with open(cache_filepath, 'rb') as f:
            X_all, y = pickle.load(f)
        print(f'Loaded precomputed inputs from {cache_filepath}')
    else:
        print('Computing inputs...')
        X_all = defaultdict(list)
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
                        token_index = -1
                    elif token_pos == 'slt':
                        token_index = -2 if sequence_length > 1 else -1
                    elif token_pos == 'fgt':
                        token_index = 0
                    else:
                        raise ValueError(f"Unsupported token_pos: {token_pos}")

                    if layer_idx >= len(hiddens):
                        print(f"Warning: Layer index {layer_idx} is out of bounds for the model. Skipping this layer.")
                        continue

                    if abs(token_index) > sequence_length:
                        print(f"Warning: token_index {token_index} out of bounds for sequence length {sequence_length}. Using last token.")
                        token_index = -1

                    hidden_state = hiddens[layer_idx][:, token_index, :].squeeze(0)
                    X_all[layer_idx].append(hidden_state.cpu())
                y.append(factoid['is_supported'])
            except Exception as e:
                print(f"Error processing factoid: {factoid}")
                print(e)
                continue

    return X_all, y

def get_classifier(classifier_name, C=0.1, max_iter=1000, random_state=42):
    """Instantiate the classifier based on the provided name."""
    if classifier_name == 'logistic_regression':
        return LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=max_iter, random_state=random_state)
    elif classifier_name == 'catboost':
        return CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=random_state,
            verbose=0
        )
    elif classifier_name == 'lightgbm':
        return LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            objective='binary',
            random_state=random_state
        )
    elif classifier_name == 'xgboost':
        return XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='auc',
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

def run_concat_layer_probing(
    X_all, y, layer_range, group_size=5, classifiers=['logistic_regression'], C=0.1, max_iter=1000, 
    results_dir='./metrics', model_name='model', token_pos='lt'
):
    """
    Run concatenated layer probing experiments with multiple classifiers, adding time benchmarks.
    """
    mean_aurocs_all = defaultdict(list)
    best_val_auroc_all = {}
    best_probes_all = {}
    best_layer_groups_all = {}
    time_benchmarks_all = {}
    token_pos_suffix = '' if token_pos == 'lt' else f'_{token_pos}'

    layer_groups = []
    total_layers = len(layer_range)
    overlap = group_size - 1

    # Sort the layer_range to ensure consistent grouping
    sorted_layers = sorted(layer_range)

    for i in range(0, total_layers, group_size - overlap):
        group = sorted_layers[i:i+group_size]
        if len(group) < group_size:
            group = sorted_layers[-group_size:]
            layer_groups.append(group)
            break
        layer_groups.append(group)

    unique_layer_groups = []
    seen = set()
    for group in layer_groups:
        group_tuple = tuple(group)
        if group_tuple not in seen:
            unique_layer_groups.append(group)
            seen.add(group_tuple)

    layer_groups = unique_layer_groups

    for classifier_name in classifiers:
        print(f"\n{'='*40}\nProcessing Classifier: {classifier_name}\n{'='*40}")
        mean_aurocs = []
        best_val_auroc = -np.inf
        best_probe = None
        best_layer_group = None
        time_benchmarks = {
            'total_time': 0.0,  # Total time for this classifier
            'per_group_time': []  # Time per layer group
        }

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
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            val_auroc_scores = []
            group_start_time = time.time()  # Start time for this layer group

            for fold_idx, (train_index, val_index) in enumerate(skf.split(X_group, y_np)):
                X_train, X_val = X_group[train_index], X_group[val_index]
                y_train, y_val = y_np[train_index], y_np[val_index]

                probe = get_classifier(classifier_name, C=C, max_iter=max_iter)

                start_time = time.time()  # Start time for this fold
                probe.fit(X_train, y_train)
                y_val_proba = probe.predict_proba(X_val)[:, 1]
                val_auroc = roc_auc_score(y_val, y_val_proba)
                end_time = time.time()  # End time for this fold

                fold_time = end_time - start_time
                time_benchmarks['per_group_time'].append({
                    'layer_group': f'{layer_start}-{layer_end}',
                    'fold': fold_idx + 1,
                    'time_seconds': fold_time
                })

                val_auroc_scores.append(val_auroc)

                print(f'Fold {fold_idx + 1}: Validation AUROC = {val_auroc:.4f}, Time = {fold_time:.2f} seconds')

            group_end_time = time.time()  # End time for this layer group
            group_time = group_end_time - group_start_time
            time_benchmarks['total_time'] += group_time

            mean_auroc_val = np.mean(val_auroc_scores)
            std_auroc_val = np.std(val_auroc_scores, ddof=1)

            probe_full = get_classifier(classifier_name, C=C, max_iter=max_iter)
            probe_full.fit(X_group, y_np)

            mean_aurocs.append({
                'Layer Group': f'{layer_start}-{layer_end}',
                'Validation AUROC Mean': mean_auroc_val,
                'Validation AUROC Std': std_auroc_val
            })

            print(f'Layer Group {layer_start}-{layer_end}: Val AUROC Mean={mean_auroc_val:.4f}, Val AUROC Std={std_auroc_val:.4f}, Group Time = {group_time:.2f} seconds')

            if mean_auroc_val > best_val_auroc:
                best_val_auroc = mean_auroc_val
                best_probe = probe_full
                best_layer_group = (layer_start, layer_end)

        # Save results for the current classifier
        df_results = pd.DataFrame(mean_aurocs)
        os.makedirs(results_dir, exist_ok=True)
        results_filepath = os.path.join(results_dir, f'{model_name}{token_pos_suffix}_{classifier_name}_layer_group_aurocs.csv')
        df_results.to_csv(results_filepath, index=False)
        print(f'\nResults for {classifier_name} saved to {results_filepath}')
        print(df_results)

        # Save the best performing probe for the current classifier
        layer_start, layer_end = best_layer_group
        probe_filename = f'{model_name}{token_pos_suffix}_best_probe_layers_{layer_start}-{layer_end}_{classifier_name}.pkl'
        probe_filepath = os.path.join(results_dir, probe_filename)
        with open(probe_filepath, 'wb') as f:
            pickle.dump({
                'probe': best_probe,
                'layer_group': best_layer_group,
                'token_pos': token_pos,
                'classifier_name': classifier_name,
                'C': C,
                'max_iter': max_iter
            }, f)
        print(f'Best performing probe for {classifier_name} saved to {probe_filepath} based on Validation AUROC of {best_val_auroc:.4f}\n')

        # Record time benchmarks
        time_benchmarks_all[classifier_name] = time_benchmarks

        # Aggregate best probes
        best_val_auroc_all[classifier_name] = best_val_auroc
        best_probes_all[classifier_name] = probe_filepath
        best_layer_groups_all[classifier_name] = best_layer_group
        mean_aurocs_all[classifier_name] = df_results

    # Save aggregated results
    aggregated_results = []
    for classifier_name, df in mean_aurocs_all.items():
        for _, row in df.iterrows():
            aggregated_results.append({
                'Classifier': classifier_name,
                'Layer Group': row['Layer Group'],
                'Validation AUROC Mean': row['Validation AUROC Mean'],
                'Validation AUROC Std': row['Validation AUROC Std']
            })
    df_aggregated = pd.DataFrame(aggregated_results)

    # Add time benchmarks to the aggregated results
    for classifier_name, benchmarks in time_benchmarks_all.items():
        df_aggregated.loc[df_aggregated['Classifier'] == classifier_name, 'Total Training Time (s)'] = benchmarks['total_time']

    aggregated_filepath = os.path.join(results_dir, f'{model_name}{token_pos_suffix}_aggregated_layer_group_aurocs.csv')
    df_aggregated.to_csv(aggregated_filepath, index=False)
    print(f'\nAggregated results with time benchmarks saved to {aggregated_filepath}')
    print(df_aggregated)

    # Save all best probes in a single pickle file
    best_probes_summary = {
        classifier: {
            'probe_filepath': probe_filepath,
            'layer_group': best_layer_groups_all[classifier],
            'validation_auroc': best_val_auroc_all[classifier],
            'training_time_seconds': time_benchmarks_all[classifier]['total_time']
        }
        for classifier, probe_filepath in best_probes_all.items()
    }
    best_probes_summary_filepath = os.path.join(results_dir, f'{model_name}{token_pos_suffix}_best_probes_summary.pkl')
    with open(best_probes_summary_filepath, 'wb') as f:
        pickle.dump(best_probes_summary, f)
    print(f'\nBest probes summary saved to {best_probes_summary_filepath}')

    return best_probes_summary_filepath

def test_on_new_data(model_name, tokenizer, llm, best_probes_summary, new_scores_filepath, cache_dir='/scratch/ms23jh/facts', test_data_name=None, results_dir='./metrics'):
    """Test the saved probes on new unseen fact scores, adding time benchmarks."""
    # Load new fact scores
    new_scores = load_fact_scores(model_name, scores_filename=new_scores_filepath)
    new_scores = flatten_scores(new_scores)

    # Initialize a DataFrame to store test AUROC results
    test_results = []
    time_benchmarks_test = []

    for classifier_name, probe_info in best_probes_summary.items():
        print(f"\n{'='*40}\nTesting Classifier: {classifier_name}\n{'='*40}")
        best_probe = None

        # Check if 'probe' is already present to avoid re-loading
        if 'probe' in probe_info:
            best_probe = probe_info['probe']
            print(f'Using already loaded probe for classifier "{classifier_name}".')
        elif 'probe_filepath' in probe_info:
            best_probe_filepath = probe_info['probe_filepath']
            if os.path.exists(best_probe_filepath):
                with open(best_probe_filepath, 'rb') as f:
                    saved_data = pickle.load(f)
                best_probe = saved_data['probe']
                print(f'Loaded probe from "{best_probe_filepath}" for classifier "{classifier_name}".')
            else:
                print(f"Error: Probe file '{best_probe_filepath}' does not exist. Skipping classifier '{classifier_name}'.")
                continue
        else:
            print(f"Error: No valid probe information found for classifier '{classifier_name}'. Skipping.")
            continue

        best_layer_group = probe_info['layer_group']
        token_pos = 'lt'

        # Determine the layer range
        layer_range = list(range(best_layer_group[0], best_layer_group[1]+1))

        # Load or compute inputs for new data
        start_time = time.time()
        X_all_new, y_new = load_or_compute_inputs(
            model_name,
            tokenizer,
            llm,
            new_scores,
            token_pos=token_pos,
            layer_range=layer_range,
            cache_dir=cache_dir,
            data_name=test_data_name
        )
        end_time = time.time()
        input_time = end_time - start_time
        print(f'Input computation time for {classifier_name}: {input_time:.2f} seconds')

        # Concatenate the hidden states from the specified layers
        X_list_new_raw = [X_all_new[layer_idx] for layer_idx in layer_range]

        # Handle empty layers by inserting dummy tensors
        for idx, X_layer in enumerate(X_list_new_raw):
            if not X_layer:
                print(f"Warning: Layer {layer_range[idx]} has no data. Inserting a dummy tensor to maintain dimensionality.")
                # Determine hidden_size
                hidden_size = llm.config.hidden_size if hasattr(llm, 'config') else 768  # Default hidden size
                dummy_tensor = torch.zeros(hidden_size)
                X_list_new_raw[idx] = [dummy_tensor]

        try:
            X_list_new = [torch.stack(X_layer).cpu().numpy() for X_layer in X_list_new_raw]
        except RuntimeError as e:
            print("Error stacking tensors:", e)
            continue

        # Concatenate along the last axis
        X_group_new = np.concatenate(X_list_new, axis=-1)
        y_np_new = np.array(y_new)

        # Predict using the best probe
        start_predict = time.time()
        y_pred_proba_new = best_probe.predict_proba(X_group_new)[:, 1]
        end_predict = time.time()
        predict_time = end_predict - start_predict

        bs_auc = bootstrap_func(y_np_new, y_pred_proba_new, auroc, rn=42)

        # Compute AUROC
        if np.unique(y_np_new).size > 1:
            test_auroc_new = bs_auc['mean']
            test_auroc_new_std = bs_auc['bootstrap']['std_err']
            print(f'New Data AUROC for {classifier_name}: {test_auroc_new:.4f}, standard error: {test_auroc_new_std}, Prediction Time = {predict_time:.2f} seconds')
            test_results.append({
                'Classifier': classifier_name,
                'Test AUROC Mean': test_auroc_new,
                'Test AUROC Std': test_auroc_new_std,
                'Prediction Time (s)': predict_time
            })
            time_benchmarks_test.append({
                'Classifier': classifier_name,
                'Input Computation Time (s)': input_time,
                'Prediction Time (s)': predict_time
            })
        else:
            print(f'Cannot compute AUROC for {classifier_name}: Only one class present in y_true.')
            test_results.append({
                'Classifier': classifier_name,
                'Test AUROC Mean': None,
                'Test AUROC Std': None,
                'Prediction Time (s)': predict_time
            })
            time_benchmarks_test.append({
                'Classifier': classifier_name,
                'Input Computation Time (s)': input_time,
                'Prediction Time (s)': predict_time
            })

    # Save test results to CSV
    test_results_df = pd.DataFrame(test_results)
    # Ensure the directory exists
    os.makedirs(results_dir, exist_ok=True)
    test_results_filepath = os.path.join(results_dir, f'{model_name}_test_aurocs.csv')
    test_results_df.to_csv(test_results_filepath, index=False)
    print(f'\nTest AUROC results saved to {test_results_filepath}')
    print(test_results_df)

    # Save time benchmarks for testing
    time_benchmarks_df = pd.DataFrame(time_benchmarks_test)
    time_benchmarks_filepath = os.path.join(results_dir, f'{model_name}_test_time_benchmarks.csv')
    time_benchmarks_df.to_csv(time_benchmarks_filepath, index=False)
    print(f'\nTest time benchmarks saved to {time_benchmarks_filepath}')
    print(time_benchmarks_df)

    return test_results_df

def main():
    parser = argparse.ArgumentParser(description='Probing Experiments with Multiple Classifiers and Time Benchmarks')
    parser.add_argument('--hf_model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Name of the Hugging Face model')
    parser.add_argument('--model_name', type=str, default='Llama3.1-8B', help='Model name identifier')
    parser.add_argument('--token_pos', type=str, default='lt', choices=['lt', 'slt', 'fgt'], help='Token position: last token (lt), second last token (slt), first token (fgt)')
    parser.add_argument('--classifiers', type=str, default='logistic_regression', help='Comma-separated list of classifiers to use for probing (options: logistic_regression, catboost, lightgbm, xgboost)')
    parser.add_argument('--C', type=float, default=0.5, help='Regularization parameter for Logistic Regression (ignored for other classifiers)')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for solver convergence (applicable for Logistic Regression)')
    parser.add_argument('--group_size', type=int, default=5, help='Number of layers to concatenate for probing')
    parser.add_argument('--cache_dir', type=str, default='/scratch/ms23jh/facts', help='Directory to cache computed inputs')
    parser.add_argument('--results_dir', type=str, default='./metrics', help='Directory to save results')
    parser.add_argument('--data_dir', type=str, default='./FActScore/data/unlabeled', help='Directory containing data')
    parser.add_argument('--layer_range', type=parse_layer_range, default=None, help='Comma-separated list of layers to use or ranges (e.g., "0,2,4,5" or "0-5,10,20-25")')
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

    # Parse classifiers into a list
    classifiers = [cls.strip().lower() for cls in args.classifiers.split(',')]
    valid_classifiers = ['logistic_regression', 'catboost', 'lightgbm', 'xgboost']
    for cls in classifiers:
        if cls not in valid_classifiers:
            raise ValueError(f"Unsupported classifier: {cls}. Supported classifiers are: {valid_classifiers}")

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

    tokenizer, llm = initialize_model(args.hf_model_name, device=args.device, max_memory=args.max_memory)

    if args.mode == 'train':
        scores = load_fact_scores(args.model_name, args.data_dir, scores_filename=args.scores_filepath)
        flat_scores = flatten_scores(scores)
        print(f'Number of samples: {len(flat_scores)}')

        if args.use_train_test_split:
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
                if test_group:
                    test_decisions.append(test_group)

            # Save the test set in the desired nested format
            test_set_dict = {'decisions': test_decisions}
            test_set_file = os.path.join(args.test_set_dir, f'{args.model_name}_test_set.pkl')
            with open(test_set_file, 'wb') as f:
                pickle.dump(test_set_dict, f)
            print(f'Test set saved to {test_set_file}')

            # Exit after saving the test set
            import sys
            sys.exit(0)
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

        # Run concatenated layer probing for all classifiers with time benchmarks
        best_probes_summary_filepath = run_concat_layer_probing(
            X_all, y,
            args.layer_range,
            args.group_size,
            classifiers=classifiers,
            C=args.C,
            max_iter=args.max_iter,
            results_dir=args.results_dir,
            model_name=args.model_name,
            token_pos=args.token_pos
        )

        # Load the best probes summary
        with open(best_probes_summary_filepath, 'rb') as f:
            best_probes_summary = pickle.load(f)

        # Test on new data if provided
        if args.test_new_data is not None:
            test_on_new_data(
                args.model_name,
                tokenizer,
                llm,
                best_probes_summary,
                args.test_new_data,
                cache_dir=args.cache_dir,
                test_data_name=None,  # Modify as needed
                results_dir=args.results_dir
            )

    elif args.mode == 'eval':
        if args.probe_filepath is None:
            print('Error: In eval mode, --probe_filepath must be specified.')
            return

        if args.test_new_data is None:
            print('Error: In eval mode, --test_new_data must be specified.')
            return

        # Load the probe file
        with open(args.probe_filepath, 'rb') as f:
            loaded_data = pickle.load(f)

        # Determine if it's a summary or a single probe
        if isinstance(loaded_data, dict) and 'probe' in loaded_data:
            # It's a single probe
            classifier_name = loaded_data.get('classifier_name', 'single_probe')
            # Add 'probe' directly to the summary to avoid re-loading
            best_probes_summary = {classifier_name: loaded_data}
            print(f'Detected a single probe for classifier "{classifier_name}".')
        elif isinstance(loaded_data, dict):
            # Assume it's a summary containing multiple probes
            best_probes_summary = loaded_data
            print(f'Detected a summary containing multiple probes for classifiers: {", ".join(best_probes_summary.keys())}.')
        else:
            raise ValueError('Unsupported probe file format. Must be a dict containing either a single probe or a summary of probes.')

        test_on_new_data(
            args.model_name,
            tokenizer,
            llm,
            best_probes_summary,
            args.test_new_data,
            cache_dir=args.cache_dir,
            test_data_name=None,  # Modify as needed
            results_dir=args.results_dir
        )

if __name__ == '__main__':
    main()
