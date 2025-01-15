import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from eval_utils import auroc, bootstrap_func  # Ensure this is available in your environment

# Constants
MODEL_NAME = 'Llama3.2-3B'
HF_MODEL_NAME = 'meta-llama/Llama-3.2-3B-Instruct'
NUM_LAYERS = 28

DEFAULT_TRAIN_DATA_DIR = './train_data/'
DEFAULT_TEST_DATA_DIR = './test_data/'

RESULTS_DIR = './metrics'
LOG_DIR = './logs'

# Ensure necessary directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize logging
log_filename = os.path.join(LOG_DIR, f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def load_fact_scores(scores_filepath):
    """Load fact scores from a pickled file."""
    logger.info(f'Loading fact scores from: {scores_filepath}')
    try:
        with open(scores_filepath, 'rb') as f:
            scores = pickle.load(f)
        logger.info('Fact scores loaded successfully.')
        return scores
    except Exception as e:
        logger.error(f'Failed to load fact scores from {scores_filepath}: {e}')
        raise

def flatten_scores(scores):
    """Flatten the nested list of scores into a flat list of factoids."""
    logger.info('Flattening fact scores.')
    try:
        if isinstance(scores, dict) and 'decisions' in scores:
            flat_scores = [factoid for decision_group in scores['decisions'] for factoid in decision_group]
            for factoid in flat_scores:
                if 'is_supported' not in factoid:
                    factoid['is_supported'] = 0
            logger.info(f'Flattened scores to {len(flat_scores)} factoids.')
            return flat_scores
        elif isinstance(scores, list):
            if all(isinstance(inner, list) and len(inner) == 2 for inner in scores):
                flattened = []
                for atom, is_supported in scores:
                    is_supported_int = 1 if is_supported else 0
                    flattened.append({'atom': atom, 'is_supported': is_supported_int})
                logger.info(f'Flattened scores to {len(flattened)} factoids.')
                return flattened
            elif all(isinstance(inner, dict) and len(inner) == 2 for inner in scores):
                logger.info('Scores are already flattened as a list of dictionaries.')
                return scores
            else:
                flattened = []
                for datum in scores:
                    extracted_facts = datum[4]
                    truth_fact_level = datum[5]
                    for fact, truth in zip(extracted_facts, truth_fact_level):
                        is_supported = 1 if truth else 0
                        flattened.append({'atom': fact, 'is_supported': is_supported})
                logger.info(f'Flattened scores to {len(flattened)} factoids.')
                return flattened
        else:
            raise ValueError("Unsupported score format.")
    except Exception as e:
        logger.error(f'Error flattening scores: {e}')
        raise

def initialize_model():
    """Initialize the tokenizer and model for Llama3.2-3B."""
    logger.info(f'Downloading and initializing model "{MODEL_NAME}" from Hugging Face...')
    try:
        path = snapshot_download(
            repo_id=HF_MODEL_NAME,
            allow_patterns=['*.json', '*.model', '*.safetensors'],
            ignore_patterns=['pytorch_model.bin.index.json']
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        logger.info(f'Model "{MODEL_NAME}" loaded successfully.')
        return tokenizer, model
    except Exception as e:
        logger.error(f'Failed to initialize model "{MODEL_NAME}": {e}')
        raise

def compute_hidden_states(model_name, tokenizer, llm, scores, layer_range=None):
    """Compute the hidden states for each factoid without caching."""
    if layer_range is None:
        layer_range = list(range(NUM_LAYERS))  # Default layer range if not provided

    X_all = defaultdict(list)
    y = []
    try:
        for factoid in tqdm(scores, desc='Processing factoids'):
            inputs = tokenizer([factoid['atom']], return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = llm(**inputs, output_hidden_states=True, return_dict=True)
            hiddens = outputs.hidden_states

            for layer_idx in layer_range:
                if layer_idx >= len(hiddens):
                    logger.warning(f"Layer index {layer_idx} is out of bounds for the model. Skipping this layer.")
                    continue

                hidden_state = hiddens[layer_idx][:, -1, :].squeeze(0)  # last token
                X_all[layer_idx].append(hidden_state.cpu())
            y.append(factoid['is_supported'])
    except Exception as e:
        logger.error(f'Error during hidden state computation: {e}')
        raise

    return X_all, y

def get_classifier(classifier_name, C=None, max_iter=1000, random_state=42):
    """Instantiate the classifier based on the provided name."""
    logger.info(f'Initializing classifier: {classifier_name} with settings: C={C}, max_iter={max_iter}, random_state={random_state}')
    if classifier_name == 'logistic_regression':
        if C is None:
            raise ValueError("Regularization parameter C must be specified for Logistic Regression.")
        return LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=max_iter, random_state=random_state)
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

def run_probing(
    X_all, y, layer_range, group_size, classifier_settings, classifier_name, results_dir=RESULTS_DIR, model_name=MODEL_NAME, identifier=''
):
    """
    Run probing experiments for a specific classifier with given settings.

    Args:
        X_all (dict): Dictionary of layer_idx to list of hidden states.
        y (list): List of labels.
        layer_range (list): List of layer indices.
        group_size (int): Number of layers to group.
        classifier_settings (dict): Settings specific to the classifier.
        classifier_name (str): Name of the classifier.
        results_dir (str): Directory to save results.
        model_name (str): Name of the model.
        identifier (str): Unique identifier for naming outputs based on training file.

    Returns:
        dict: Information about the best performing probe.
    """
    C_value = classifier_settings.get('C', 'default')
    logger.info(f'--- Starting probing for classifier: {classifier_name}, C={C_value}, group_size={group_size} ---')
    logger.info(f'Layer range: {layer_range}, Group size: {group_size}')
    logger.info(f'Classifier settings: {classifier_settings}')

    mean_aurocs = []
    best_val_auroc = -np.inf
    best_probe = None
    best_layer_group = None
    time_benchmarks = {
        'total_time': 0.0,
        'per_group_time': []
    }

    # Create layer groups
    layer_groups = []
    total_layers = len(layer_range)
    overlap = group_size - 1
    sorted_layers = sorted(layer_range)

    for i in range(0, total_layers, group_size - overlap):
        group = sorted_layers[i:i+group_size]
        if len(group) < group_size:
            group = sorted_layers[-group_size:]
            layer_groups.append(group)
            break
        layer_groups.append(group)

    # Remove duplicate groups
    unique_layer_groups = []
    seen = set()
    for group in layer_groups:
        group_tuple = tuple(group)
        if group_tuple not in seen:
            unique_layer_groups.append(group)
            seen.add(group_tuple)

    layer_groups = unique_layer_groups

    logger.info(f'Number of unique layer groups to process: {len(layer_groups)}')

    for layers_in_group in layer_groups:
        layer_start = layers_in_group[0]
        layer_end = layers_in_group[-1]
        logger.info(f'\n--- Processing Layer Group: {layer_start}-{layer_end} ---')

        X_list = [X_all[layer_idx] for layer_idx in layers_in_group]
        try:
            X_list = [torch.stack(X_layer).cpu().numpy() for X_layer in X_list]
        except Exception as e:
            logger.error(f'Error stacking hidden states for layers {layers_in_group}: {e}')
            continue  # Skip this layer group if there's an error

        X_group = np.concatenate(X_list, axis=-1)
        y_np = np.array(y)

        logger.info(f'Feature matrix shape: {X_group.shape}')
        logger.info(f'Labels shape: {y_np.shape}, Positive class ratio: {np.mean(y_np):.4f}')

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_auroc_scores = []
        group_start_time = time.time()

        for fold_idx, (train_index, val_index) in enumerate(skf.split(X_group, y_np), 1):
            logger.info(f'\n--- Fold {fold_idx} ---')
            X_train, X_val = X_group[train_index], X_group[val_index]
            y_train, y_val = y_np[train_index], y_np[val_index]

            probe = get_classifier(classifier_name, **classifier_settings)

            start_time = time.time()
            try:
                probe.fit(X_train, y_train)
                y_val_proba = probe.predict_proba(X_val)[:, 1]
                val_auroc = roc_auc_score(y_val, y_val_proba)
                logger.info(f'Fold {fold_idx}: Validation AUROC = {val_auroc:.4f}')
            except Exception as e:
                logger.error(f'Error during training/prediction in fold {fold_idx}: {e}')
                val_auroc = None
            end_time = time.time()

            fold_time = end_time - start_time
            time_benchmarks['per_group_time'].append({
                'layer_group': f'{layer_start}-{layer_end}',
                'fold': fold_idx,
                'time_seconds': fold_time
            })

            if val_auroc is not None:
                val_auroc_scores.append(val_auroc)
                logger.info(f'Fold {fold_idx}: Time = {fold_time:.2f} seconds')

        group_end_time = time.time()
        group_time = group_end_time - group_start_time
        time_benchmarks['total_time'] += group_time

        if not val_auroc_scores:
            logger.warning(f'No valid AUROC scores obtained for layer group {layer_start}-{layer_end}. Skipping.')
            continue  # Skip if no valid scores

        mean_auroc_val = np.mean(val_auroc_scores)
        std_auroc_val = np.std(val_auroc_scores, ddof=1)

        try:
            probe_full = get_classifier(classifier_name, **classifier_settings)
            probe_full.fit(X_group, y_np)
        except Exception as e:
            logger.error(f'Failed to fit the full probe for layer group {layer_start}-{layer_end}: {e}')
            continue

        mean_aurocs.append({
            'Layer Group': f'{layer_start}-{layer_end}',
            'Validation AUROC Mean': mean_auroc_val,
            'Validation AUROC Std': std_auroc_val
        })

        logger.info(f'\nLayer Group {layer_start}-{layer_end}:')
        logger.info(f'  Validation AUROC Mean: {mean_auroc_val:.4f}')
        logger.info(f'  Validation AUROC Std: {std_auroc_val:.4f}')
        logger.info(f'  Group Time: {group_time:.2f} seconds')

        if mean_auroc_val > best_val_auroc:
            best_val_auroc = mean_auroc_val
            best_probe = probe_full
            best_layer_group = (layer_start, layer_end)

    if best_layer_group is None:
        logger.warning('No best layer group found. No probe will be saved.')
    else:
        # Save results for the current classifier
        df_results = pd.DataFrame(mean_aurocs)
        os.makedirs(results_dir, exist_ok=True)
        # Incorporate identifier, C, and group_size into the results file name
        C_value_str = f"C{classifier_settings.get('C')}"
        results_filepath = os.path.join(results_dir, f'{model_name}_{classifier_name}_aurocs_{C_value_str}_group{group_size}_{identifier}.csv')
        try:
            df_results.to_csv(results_filepath, index=False)
            logger.info(f'\nResults for {classifier_name} (C={C_value_str}, group_size={group_size}) saved to {results_filepath}')
        except Exception as e:
            logger.error(f'Failed to save results to {results_filepath}: {e}')
            raise

        logger.info('\n--- Probing Results ---')
        logger.info(df_results)

        # Save the best performing probe for the current classifier
        layer_start, layer_end = best_layer_group
        # Incorporate identifier, C, and group_size into the probe file name
        probe_filename = f'{model_name}_best_probe_layers_{layer_start}-{layer_end}_{classifier_name}_{C_value_str}_group{group_size}_{identifier}.pkl'
        probe_filepath = os.path.join(results_dir, probe_filename)
        try:
            with open(probe_filepath, 'wb') as f:
                pickle.dump({
                    'probe': best_probe,
                    'layer_group': best_layer_group,
                    'classifier_name': classifier_name,
                    'settings': classifier_settings,
                    'group_size': group_size,
                    'C': classifier_settings.get('C')
                }, f)
            logger.info(f'Best performing probe for {classifier_name} (C={C_value_str}, group_size={group_size}) saved to {probe_filepath} based on Validation AUROC of {best_val_auroc:.4f}\n')
        except Exception as e:
            logger.error(f'Failed to save probe to {probe_filepath}: {e}')
            raise

    # Record time benchmarks
    logger.info('--- Time Benchmarks ---')
    logger.info(f'Total Time for {classifier_name} (C={C_value_str}, group_size={group_size}): {time_benchmarks["total_time"]:.2f} seconds')
    for benchmark in time_benchmarks['per_group_time']:
        logger.info(f'  Layer Group {benchmark["layer_group"]}, Fold {benchmark["fold"]}: {benchmark["time_seconds"]:.2f} seconds')

    return {
        'classifier_name': classifier_name,
        'C': classifier_settings.get('C'),
        'group_size': group_size,
        'best_val_auroc': best_val_auroc,
        'probe_filepath': probe_filepath if best_layer_group else None,
        'layer_group': best_layer_group,
        'time_benchmarks': time_benchmarks
    }

def run_xgboost_on_best(probe_summaries, X_all, y, layer_ranges, results_dir=RESULTS_DIR, model_name=MODEL_NAME, identifier=''):
    """
    Run XGBoost on the best layer ranges selected by Logistic Regression with C=0.5.

    Args:
        probe_summaries (list): List of dictionaries containing probe summaries from Logistic Regression.
        X_all (dict): Dictionary of layer_idx to list of hidden states.
        y (list): List of labels.
        layer_ranges (list): List of layer ranges to run XGBoost on.
        results_dir (str): Directory to save results.
        model_name (str): Name of the model.
        identifier (str): Unique identifier for naming outputs based on training file.

    Returns:
        list: List of summaries from XGBoost training.
    """
    logger.info('--- Starting XGBoost Training on Best Layer Groups ---')

    # Initialize a set to keep track of already processed layer groups
    run_best_layer_groups = set()

    xgboost_summaries = []

    for summary in probe_summaries:
        classifier_name = summary['classifier_name']
        best_layer_group = summary['layer_group']
        C_value = summary['C']
        group_size = summary['group_size']

        # Convert the layer group to a tuple to make it hashable for the set
        layer_group_tuple = tuple(best_layer_group)

        # Check if this layer group has already been processed
        if layer_group_tuple in run_best_layer_groups:
            logger.info(f'Skipping XGBoost training for already processed layer group: {layer_group_tuple}')
            continue  # Skip to the next summary to avoid duplicate training

        # Add the current layer group to the set to mark it as processed
        run_best_layer_groups.add(layer_group_tuple)

        # Only proceed if the classifier is Logistic Regression with C=0.5
        if classifier_name != 'logistic_regression':
            logger.info(f'Skipping classifier "{classifier_name}" as it is not Logistic Regression.')
            continue

        if 'C' not in summary or summary['C'] != 0.5:
            logger.info(f'Skipping layer group {best_layer_group} as C={summary.get("C")} != 0.5')
            continue

        logger.info(f'\n--- Running XGBoost on Best Layer Group {best_layer_group} selected by {classifier_name} (C=0.5, group_size={group_size}) ---')

        layer_start, layer_end = best_layer_group
        selected_layers = list(range(layer_start, layer_end + 1))

        X_list = [X_all[layer_idx] for layer_idx in selected_layers]
        try:
            X_list = [torch.stack(X_layer).cpu().numpy() for X_layer in X_list]
        except Exception as e:
            logger.error(f'Error stacking hidden states for layers {selected_layers}: {e}')
            continue

        X_group = np.concatenate(X_list, axis=-1)
        y_np = np.array(y)

        logger.info(f'XGBoost Feature matrix shape: {X_group.shape}')
        logger.info(f'Labels shape: {y_np.shape}, Positive class ratio: {np.mean(y_np):.4f}')

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        val_auroc_scores = []
        time_benchmarks = {
            'total_time': 0.0,
            'per_fold_time': []
        }

        for fold_idx, (train_index, val_index) in enumerate(skf.split(X_group, y_np), 1):
            logger.info(f'\n--- XGBoost Fold {fold_idx} ---')
            X_train, X_val = X_group[train_index], X_group[val_index]
            y_train, y_val = y_np[train_index], y_np[val_index]

            probe = get_classifier('xgboost')

            start_time = time.time()
            try:
                probe.fit(X_train, y_train)
                y_val_proba = probe.predict_proba(X_val)[:, 1]
                val_auroc = roc_auc_score(y_val, y_val_proba)
                logger.info(f'Fold {fold_idx}: Validation AUROC = {val_auroc:.4f}')
            except Exception as e:
                logger.error(f'Error during XGBoost training/prediction in fold {fold_idx}: {e}')
                val_auroc = None
            end_time = time.time()

            fold_time = end_time - start_time
            time_benchmarks['per_fold_time'].append({
                'layer_group': f'{layer_start}-{layer_end}',
                'fold': fold_idx,
                'time_seconds': fold_time
            })

            if val_auroc is not None:
                val_auroc_scores.append(val_auroc)
                logger.info(f'Fold {fold_idx}: Time = {fold_time:.2f} seconds')

        if not val_auroc_scores:
            logger.warning(f'No valid AUROC scores obtained for XGBoost on layer group {layer_group_tuple}. Skipping.')
            continue  # Skip if no valid scores

        mean_auroc_val = np.mean(val_auroc_scores)
        std_auroc_val = np.std(val_auroc_scores, ddof=1)
        time_benchmarks['total_time'] = sum([fb['time_seconds'] for fb in time_benchmarks['per_fold_time']])

        try:
            probe_full = get_classifier('xgboost')
            probe_full.fit(X_group, y_np)
        except Exception as e:
            logger.error(f'Failed to fit the full XGBoost probe for layer group {layer_start}-{layer_end}: {e}')
            continue

        logger.info(f'\nXGBoost on Layer Group {layer_start}-{layer_end}:')
        logger.info(f'  Validation AUROC Mean: {mean_auroc_val:.4f}')
        logger.info(f'  Validation AUROC Std: {std_auroc_val:.4f}')
        logger.info(f'  Total Training Time: {time_benchmarks["total_time"]:.2f} seconds')

        # Save XGBoost results
        results = {
            'Layer Group': f'{layer_start}-{layer_end}',
            'Validation AUROC Mean': mean_auroc_val,
            'Validation AUROC Std': std_auroc_val,
            'Total Training Time (s)': time_benchmarks['total_time']
        }
        # Incorporate identifier, C, and group_size into the results file name
        C_value_str = "C0.5"
        results_filepath = os.path.join(results_dir, f'{model_name}_xgboost_results_{C_value_str}_group{group_size}_{identifier}.csv')

        try:
            df_results = pd.DataFrame([results])
            df_results.to_csv(results_filepath, index=False)
            logger.info(f'XGBoost results saved to {results_filepath}')
        except Exception as e:
            logger.error(f'Failed to save XGBoost results to {results_filepath}: {e}')
            continue

        # Save the XGBoost probe
        probe_filename = f'{model_name}_best_probe_layers_{layer_start}-{layer_end}_xgboost_{C_value_str}_group{group_size}_{identifier}.pkl'
        probe_filepath = os.path.join(results_dir, probe_filename)

        try:
            with open(probe_filepath, 'wb') as f:
                pickle.dump({
                    'probe': probe_full,
                    'layer_group': best_layer_group,
                    'classifier_name': 'xgboost',
                    'settings': {},
                    'group_size': group_size,
                    'C': 0.5
                }, f)
            logger.info(f'XGBoost probe saved to {probe_filepath}')
        except Exception as e:
            logger.error(f'Failed to save XGBoost probe to {probe_filepath}: {e}')

        # Append summary for XGBoost
        xgboost_summaries.append({
            'classifier_name': 'xgboost',
            'C': 0.5,
            'group_size': group_size,
            'best_val_auroc': mean_auroc_val,
            'probe_filepath': probe_filepath,
            'layer_group': best_layer_group,
            'time_benchmarks': time_benchmarks
        })

    return xgboost_summaries

def test_on_new_data(model_name, tokenizer, llm, best_probes_summary, test_data_dir=DEFAULT_TEST_DATA_DIR, results_dir=RESULTS_DIR):
    """Test the saved probes on new unseen fact scores."""
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.pkl')]
    if not test_files:
        logger.warning(f'No test files found in directory {test_data_dir}. Skipping testing on new data.')
        return

    logger.info('--- Starting Testing Phase on New Data ---')
    test_results = []
    time_benchmarks_test = []

    for test_file in test_files:
        test_scores_filepath = os.path.join(test_data_dir, test_file)
        logger.info(f'\n--- Testing on New Data File: {test_scores_filepath} ---')
        try:
            new_scores = load_fact_scores(test_scores_filepath)
            new_scores = flatten_scores(new_scores)
        except Exception as e:
            logger.error(f'Failed to load or flatten scores from {test_scores_filepath}: {e}')
            continue

        for classifier_key, probe_info in best_probes_summary.items():
            classifier_name = classifier_key  # e.g., 'logistic_regression_C0.5_group5'
            C_value = probe_info.get('C', 'default')
            group_size = probe_info.get('group_size', 'default')
            logger.info(f'\n--- Testing Classifier: {classifier_name}, C={C_value}, group_size={group_size} on Dataset: {test_file} ---')

            # Retrieve the probe directly from the summary
            best_probe = probe_info.get('probe')
            if best_probe is None:
                logger.error(f"No probe object found for classifier '{classifier_name}'. Skipping.")
                continue

            best_layer_group = probe_info['layer_group']
            selected_layers = list(range(best_layer_group[0], best_layer_group[1] + 1))
            logger.info(f'Using layer group: {selected_layers}')

            # Compute hidden states for new data
            start_time = time.time()
            try:
                X_all_new, y_new = compute_hidden_states(
                    model_name,
                    tokenizer,
                    llm,
                    new_scores,
                    layer_range=selected_layers
                )
            except Exception as e:
                logger.error(f'Failed to compute hidden states for {test_file} with classifier {classifier_name}: {e}')
                continue
            end_time = time.time()
            input_time = end_time - start_time
            logger.info(f'Input computation time for {classifier_name} on {test_file}: {input_time:.2f} seconds')

            # Concatenate the hidden states from the specified layers
            X_list_new_raw = [X_all_new[layer_idx] for layer_idx in selected_layers]

            try:
                X_list_new = [torch.stack(X_layer).cpu().numpy() for X_layer in X_list_new_raw]
            except RuntimeError as e:
                logger.error(f'Error stacking tensors for {classifier_name} on {test_file}: {e}', exc_info=True)
                continue

            X_group_new = np.concatenate(X_list_new, axis=-1)
            y_np_new = np.array(y_new)

            # Predict using the best probe
            start_predict = time.time()
            try:
                y_pred_proba_new = best_probe.predict_proba(X_group_new)[:, 1]
            except Exception as e:
                logger.error(f'Failed to predict probabilities for {classifier_name} on {test_file}: {e}')
                continue
            end_predict = time.time()
            predict_time = end_predict - start_predict

            try:
                bs_auc = bootstrap_func(y_np_new, y_pred_proba_new, auroc, rn=42)
                test_auroc_new = bs_auc['mean']
                test_auroc_new_std = bs_auc['bootstrap']['std_err']
                logger.info(f'New Data AUROC for {classifier_name} on {test_file}: {test_auroc_new:.4f} ± {test_auroc_new_std:.4f}')
            except Exception as e:
                logger.error(f'Failed to compute AUROC for {classifier_name} on {test_file}: {e}')
                test_auroc_new = None
                test_auroc_new_std = None

            logger.info(f'Prediction Time: {predict_time:.2f} seconds')

            # Create a unique identifier for the test result
            C_value_str = f"C{C_value}"
            test_result_identifier = f'{classifier_name}_C{C_value}_group{group_size}'

            test_results.append({
                'Test File': test_file,
                'Classifier': classifier_name,
                'C': C_value,
                'Group Size': group_size,
                'Test AUROC Mean': test_auroc_new,
                'Test AUROC Std': test_auroc_new_std,
                'Prediction Time (s)': predict_time
            })
            time_benchmarks_test.append({
                'Classifier': classifier_name,
                'C': C_value,
                'Group Size': group_size,
                'Input Computation Time (s)': input_time,
                'Prediction Time (s)': predict_time
            })

    # Save test results to CSV
    if test_results:
        try:
            test_results_df = pd.DataFrame(test_results)
            test_results_filepath = os.path.join(results_dir, f'{model_name}_test_aurocs.csv')
            test_results_df.to_csv(test_results_filepath, index=False)
            logger.info(f'\nTest AUROC results saved to {test_results_filepath}')
            logger.info(test_results_df)
        except Exception as e:
            logger.error(f'Failed to save test AUROC results to {test_results_filepath}: {e}')
    else:
        logger.warning('No test results to save.')

    # Save time benchmarks for testing
    if time_benchmarks_test:
        try:
            time_benchmarks_df = pd.DataFrame(time_benchmarks_test)
            time_benchmarks_filepath = os.path.join(results_dir, f'{model_name}_test_time_benchmarks.csv')
            time_benchmarks_df.to_csv(time_benchmarks_filepath, index=False)
            logger.info(f'\nTest time benchmarks saved to {time_benchmarks_filepath}')
            logger.info(time_benchmarks_df)
        except Exception as e:
            logger.error(f'Failed to save test time benchmarks to {test_scores_filepath}: {e}')
    else:
        logger.warning('No time benchmarks to save.')

def main():
    parser = argparse.ArgumentParser(description='Probing Experiments with Logistic Regression and XGBoost for Llama3.2-3B')
    parser.add_argument('--train_data_dir', type=str, default=DEFAULT_TRAIN_DATA_DIR, help='Directory containing training data files (.pkl)')
    parser.add_argument('--test_data_dir', type=str, default=DEFAULT_TEST_DATA_DIR, help='Directory containing test data files (.pkl)')
    args = parser.parse_args()

    logger.info('--- Starting Probing Experiment Script ---')
    logger.info(f'Training Data Directory: {args.train_data_dir}')
    logger.info(f'Test Data Directory: {args.test_data_dir}')

    try:
        tokenizer, llm = initialize_model()
    except Exception as e:
        logger.error('Model initialization failed. Exiting.')
        return

    # Define group sizes and C values
    group_sizes = [1, 5]
    C_values = [0.1, 0.5]

    # Iterate over each training file
    training_files = [f for f in os.listdir(args.train_data_dir) if f.endswith('.pkl')]
    if not training_files:
        logger.error(f'No training files found in directory {args.train_data_dir}. Exiting.')
        return

    for train_file in training_files:
        train_filepath = os.path.join(args.train_data_dir, train_file)
        logger.info(f'\n--- Processing Training File: {train_filepath} ---')

        # Unique identifier based on training file name (without extension)
        identifier = os.path.splitext(train_file)[0]

        # Load and preprocess training data
        try:
            scores = load_fact_scores(train_filepath)
            flat_scores = flatten_scores(scores)
            logger.info(f'Number of training samples in {train_file}: {len(flat_scores)}')
        except Exception as e:
            logger.error(f'Failed to load and preprocess training data from {train_filepath}: {e}')
            continue

        # Compute hidden states
        try:
            X_all, y = compute_hidden_states(
                MODEL_NAME,
                tokenizer,
                llm,
                flat_scores,
                layer_range=list(range(NUM_LAYERS))
            )
        except Exception as e:
            logger.error(f'Failed to compute hidden states for {train_file}: {e}')
            continue

        # Define Logistic Regression settings
        logistic_settings = [
            {'C': 0.5},
            {'C': 0.1}
        ]

        # Train Logistic Regression models
        probe_summaries = []
        for settings in logistic_settings:
            C_value = settings['C']
            for group_size in group_sizes:
                classifier_name = 'logistic_regression'
                logger.info(f'\n{"="*40}\nTraining Logistic Regression with C={C_value}, group_size={group_size}\n{"="*40}')
                summary = run_probing(
                    X_all=X_all,
                    y=y,
                    layer_range=list(range(NUM_LAYERS)),
                    group_size=group_size,
                    classifier_settings=settings,
                    classifier_name=classifier_name,
                    results_dir=RESULTS_DIR,
                    model_name=MODEL_NAME,
                    identifier=identifier
                )
                probe_summaries.append(summary)

        # After training Logistic Regression models, select best layer ranges (C=0.5)
        logger.info('--- Selecting Best Layer Ranges Based on Logistic Regression (C=0.5) ---')
        best_layer_ranges = []
        for summary in probe_summaries:
            if summary['classifier_name'] == 'logistic_regression' and summary['C'] == 0.5 and summary['best_val_auroc'] >= 0.5:  # Adjust threshold as needed
                best_layer_ranges.append(summary['layer_group'])
                logger.info(f'Layer range {summary["layer_group"]} selected with Validation AUROC: {summary["best_val_auroc"]:.4f}')

        if not best_layer_ranges:
            logger.warning(f'No suitable layer ranges found based on Logistic Regression for {train_file}. Skipping XGBoost training.')
            continue

        logger.info(f'Best layer ranges selected by Logistic Regression (C=0.5) for {train_file}: {best_layer_ranges}')

        # Train XGBoost models on the best layer ranges
        xgb_summaries = run_xgboost_on_best(
            probe_summaries=[summary for summary in probe_summaries if summary['classifier_name'] == 'logistic_regression' and summary['C'] == 0.5],
            X_all=X_all,
            y=y,
            layer_ranges=best_layer_ranges,
            results_dir=RESULTS_DIR,
            model_name=MODEL_NAME,
            identifier=identifier
        )
        probe_summaries.extend(xgb_summaries)  # Append XGBoost summaries to the overall summaries

        # Prepare best_probes_summary for testing
        best_probes_summary = {}
        for summary in probe_summaries:
            classifier = summary['classifier_name']
            C = summary['C']
            group_size = summary['group_size']
            probe_filepath = summary['probe_filepath']
            if probe_filepath and os.path.exists(probe_filepath):
                try:
                    with open(probe_filepath, 'rb') as f:
                        saved_data = pickle.load(f)
                    best_probe = saved_data['probe']
                    best_layer_group = saved_data['layer_group']
                    best_probes_summary[f'{classifier}_C{C}_group{group_size}'] = {
                        'probe': best_probe,
                        'layer_group': best_layer_group,
                        'C': C,
                        'group_size': group_size
                    }
                except Exception as e:
                    logger.error(f'Failed to load probe from {probe_filepath}: {e}')
                    continue
            else:
                logger.error(f'Probe file "{probe_filepath}" does not exist. Skipping classifier "{classifier}".')

        if not best_probes_summary:
            logger.warning(f'No valid probes found for {train_file}. Skipping testing.')
            continue

        # Test the trained probes on test data
        logger.info('--- Initiating Testing Phase on Unseen Data ---')
        test_on_new_data(
            model_name=MODEL_NAME,
            tokenizer=tokenizer,
            llm=llm,
            best_probes_summary=best_probes_summary,
            test_data_dir=args.test_data_dir,
            results_dir=RESULTS_DIR
        )

    logger.info('--- Probing Experiment Script Completed ---')

if __name__ == '__main__':
    main()
