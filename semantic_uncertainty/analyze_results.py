"""Analyze uncertainty predictions."""
import argparse
import functools
import logging
import os
import pickle

import numpy as np
from sklearn.metrics import roc_curve, auc
import wandb

from uncertainty.utils import utils
from uncertainty.utils.eval_utils import (
    bootstrap, compatible_bootstrap, auroc, accuracy_at_quantile,
    area_under_thresholded_accuracy)


utils.setup_logger()

result_dict = {}

UNC_MEAS = 'uncertainty_measures.pkl'


def init_wandb(wandb_runid, assign_new_wandb_id):
    '''Initialize wandb session.'''
    user = os.environ['USER']
    kwargs = dict(
        entity='goatml',
        project='semantic_uncertainty',
        dir=f'/scratch-ssd/{user}/uncertainty',
    )
    if not assign_new_wandb_id:
        # Restore wandb session.
        wandb.init(
            id=wandb_runid,
            resume=True,
            **kwargs
        )
        wandb.restore(UNC_MEAS)
    else:
        api = wandb.Api()
        wandb.init(**kwargs)

        old_run = api.run(f'goatml/semantic_uncertainty/{wandb_runid}')
        old_run.file(UNC_MEAS).download(
            replace=True, exist_ok=False, root=wandb.run.dir)


def analyze_run(wandb_runid, assign_new_wandb_id=False, answer_fractions_mode='default'):
    '''Analyze the uncertainty measures for a given wandb run id.'''
    logging.info('Analyzing wandb_runid `%s`.', wandb_runid)

    # Set up evaluation metrics
    if answer_fractions_mode == 'default':
        answer_fractions = [0.8, 0.9, 0.95]
    elif answer_fractions_mode == 'finegrained':
        answer_fractions = np.linspace(0, 1, 20+1)
    else:
        raise ValueError

    rng = np.random.default_rng(41)
    eval_metrics = dict(zip(
        ['AUROC', 'area_under_thresholded_accuracy', 'mean_uncertainty'],
        list(zip(
            [auroc, area_under_thresholded_accuracy, np.mean],
            [compatible_bootstrap, compatible_bootstrap, bootstrap]
        )),
    ))
    for answer_fraction in answer_fractions:
        key = f'accuracy_at_{answer_fraction}_answer_fraction'
        eval_metrics[key] = [
            functools.partial(accuracy_at_quantile, quantile=answer_fraction),
            compatible_bootstrap]

    if wandb.run is None:
        init_wandb(wandb_runid, assign_new_wandb_id=assign_new_wandb_id)

    elif wandb.run.id != wandb_runid:
        raise

    # Load the results dictionary from a pickle file.
    with open(f'{wandb.run.dir}/{UNC_MEAS}', 'rb') as file:
        results_old = pickle.load(file)

    result_dict = {'performance': {}, 'uncertainty': {}}

    # First: Compute Simple Accuracy metrics of the model predictions
    all_accuracies = {name: 1 - np.array(data) for name, data in results_old['alt_validation_is_false'].items()}
    all_accuracies['accuracy'] = 1 - np.array(results_old['validation_is_false'])
    for name, target in all_accuracies.items():
        result_dict['performance'][name] = {}
        result_dict['performance'][name]['mean'] = np.mean(target)
        result_dict['performance'][name]['bootstrap'] = bootstrap(np.mean, rng)(target)

    rum = results_old['uncertainty_measures']
    if 'p_false' in rum and 'p_false_fixed' not in rum:
        # restore log probs true: y = 1 - x --> x = 1 - y
        # convert to probs --> np.exp(1 - y)
        # conert to p_false --> 1- np.exp(1-y)
        rum['p_false_fixed'] = [1 - np.exp(1 - x) for x in rum['p_false']]

    # Next: Uncertainty Measures
    # Iterate through the dictionary and compute additional metrics for each measure.
    for measure_name, measure_values in rum.items():
        logging.info('Computing for uncertainty measure `%s`.', measure_name)

        # Validation accuracy.
        validation_is_falses = [
            results_old['validation_is_false'],
            results_old['validation_unanswerable']
        ]

        logging_names = ['', '_UNANSWERABLE']

        # Check if we have additional predictions for this measure.
        if measure_name in (u_m := results_old['alt_validation_is_false']):
            validation_is_falses.append(u_m[measure_name])
            logging_names.append(f'_max_from_{measure_name}')

        # Iterate over predictions of 'falseness' or 'answerability'.
        for validation_is_false, logging_name in zip(validation_is_falses, logging_names):
            name = measure_name + logging_name
            result_dict['uncertainty'][name] = {}

            validation_is_false = np.array(validation_is_false)
            validation_accuracy = 1 - validation_is_false

            fargs = {
                'AUROC': [validation_is_false, measure_values],
                'area_under_thresholded_accuracy': [validation_accuracy, measure_values],
                'mean_uncertainty': [measure_values]}

            for answer_fraction in answer_fractions:
                fargs[f'accuracy_at_{answer_fraction}_answer_fraction'] = [validation_accuracy, measure_values]

            for fname, (function, bs_function) in eval_metrics.items():
                metric_i = function(*fargs[fname])
                result_dict['uncertainty'][name][fname] = {}
                result_dict['uncertainty'][name][fname]['mean'] = metric_i
                logging.info("%s for measure name `%s`: %f", fname, name, metric_i)
                result_dict['uncertainty'][name][fname]['bootstrap'] = bs_function(
                    function, rng)(*fargs[fname])

    wandb.log(result_dict)
    logging.info(
        'Analysis for wandb_runid `%s` finished. Full results dict: %s',
        wandb_runid, result_dict
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_runids', nargs='+', type=str,
                        help='Wandb run ids of the datasets to evaluate on.')
    parser.add_argument('--assign_new_wandb_id', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--answer_fractions_mode', type=str, default='default')

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    # Parse wandb run ids
    wandb_runids = args.wandb_runids

    for wid in wandb_runids:
        logging.info('Evaluating wandb_runid `%s`.', wid)
        analyze_run(wid, args.assign_new_wandb_id, args.answer_fractions_mode)
