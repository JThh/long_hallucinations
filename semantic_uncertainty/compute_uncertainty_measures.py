"""Compute uncertainty measures after generating answers."""
import argparse
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from analyze_results import analyze_run

from uncertainty.uncertainty_measures.p_ik import get_p_ik
from uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from uncertainty.utils import utils


utils.setup_logger()


def main(args):

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    user = os.environ['USER']
    wandb_dir = f'/scratch-ssd/{user}/uncertainty'
    slurm_jobid = os.getenv('SLURM_JOB_ID')
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        wandb.init(
            entity=args.entity,
            # set the wandb project where this run will be logged
            project=project,
            dir=wandb_dir,
            notes=f'slurm_id: {slurm_jobid}',
        )
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.config.update(old_run.config)

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')
        def restore(filename):
            pass

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)

    else:
        is_ood_eval = False  # pylint: disable=invalid-name
        train_generations_pickle = restore('train_generations.pkl')
        with open(train_generations_pickle.name, 'rb') as infile:
            train_generations = pickle.load(infile)

    wandb.config.update({
        "compute_predictive_entropy": args.compute_predictive_entropy,
        "compute_p_ik": args.compute_p_ik,
        "is_ood_eval": is_ood_eval
    },
        allow_val_change=True
    )

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)


    entropies, accuracies = defaultdict(list), defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    count = 0  # pylint: disable=invalid-name

    if len(validation_generations) == 400:
        raise ValueError("Very likely this is a bug where validation data contains train data.")


    def is_answerable(generation):
        return len(generation['reference']['answers']['text']) > 0


    # Loop over datapoints and compute validation embeddings, accuracies and entropies.
    for tid in validation_generations:

        question = validation_generations[tid]['question']
        full_responses = validation_generations[tid]["responses"]
        most_likely_answer = validation_generations[tid]['most_likely_answer']

        responses = [full_response[0] for full_response in full_responses]

        validation_answerable.append(is_answerable(validation_generations[tid]))

        validation_embeddings.append(most_likely_answer['embedding'])
        validation_is_true.append(most_likely_answer['accuracy'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            log_liks = [r[1] for r in full_responses]
            for i in log_liks:
                assert i

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(responses, model=model, tokenizer=tokenizer)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Compute entropies with and without length normalized token probabilities.
            # NOTE: Averaging is default. For compatibility, we do not mention it in the name.
            for agg_name, agg_func in zip(['', '_sum'], [np.mean, np.sum]):
                log_liks_agg = [agg_func(log_lik) for log_lik in log_liks]

                # Compute standard entropy.
                entropies['regular_entropy' + agg_name].append(predictive_entropy(log_liks_agg))

                # Compute semantic entropies with summing and with averaging probabilities within the cluster.
                cluster_agg_names = ['', '_sum-normalized', '_sum-normalized-rao', '_cmean']
                cluster_aggs = ['sum', 'sum_normalized', 'sum_normalized', 'mean']
                for cluster_agg_name, cluster_agg in zip(cluster_agg_names, cluster_aggs):
                    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg=cluster_agg)
                    name = 'semantic_entropy' + agg_name + cluster_agg_name

                    if cluster_agg_name != '_sum-normalized-rao':
                        pe = predictive_entropy(log_likelihood_per_semantic_id)
                    else:
                        pe = predictive_entropy_rao(log_likelihood_per_semantic_id)

                    entropies[name].append(pe)

                    # For the semantic uncertainties, we can also change the prediction, by first selecting the semantic
                    # cluster with the highest probability, and then selecting the generation with the highest probability
                    # within that cluster.
                    # NOTE: nanargmax because we currently have some clusters with empty generations.
                    max_cluster_id = np.nanargmax(log_likelihood_per_semantic_id)
                    # Filter log_liks to max cluster.
                    generations_in_cluster = np.array(log_liks_agg)
                    generations_in_cluster[np.array(semantic_ids) != max_cluster_id] = -np.inf
                    # Select generation with new max probability.
                    max_idx_in_cluster = np.argmax(generations_in_cluster)
                    # Accuracies for alternative generations saved at last index.
                    accuracies[name].append(full_responses[max_idx_in_cluster][-1])

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM at id=`%s`.', tid)
            logging.info('Context:')
            logging.info(validation_generations[tid]['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(validation_generations[tid]['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))


    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)
        accuracies_mean = {k: np.mean(v) for k, v in accuracies.items()}
        logging.info('Accuracy on original task from cluster-based generations: %s', accuracies_mean)

        result_dict['alt_validation_accuracies_mean'] = accuracies_mean
        result_dict['alt_validation_is_false'] = {k: [1 - vi for vi in v] for k, v in accuracies.items()}

    if args.compute_p_ik or args.compute_p_ik_answerable:
        # Assemble training data for classifier.
        train_is_true, train_embeddings, train_answerable = [], [], []
        for tid in train_generations:
            most_likely_answer = train_generations[tid]['most_likely_answer']
            train_embeddings.append(most_likely_answer['embedding'])
            train_is_true.append(most_likely_answer['accuracy'])
            train_answerable.append(is_answerable(train_generations[tid]))
        train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
        train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
        logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    if args.compute_p_ik:
        # Train classifier of correct/incorrect.
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        result_dict['uncertainty_measures']['p_ik'] = p_ik_predictions

    if args.compute_p_ik_answerable:
        # Train classifier of answerable/unanswerable:
        p_ik_predictions = get_p_ik(
            train_embeddings=train_embeddings, is_false=train_unanswerable,
            eval_embeddings=validation_embeddings, eval_is_false=validation_unanswerable)
        result_dict['uncertainty_measures']['p_ik_unanswerable'] = p_ik_predictions

    # write the dictionary to a pickle file
    with open(f'{wandb.run.dir}/uncertainty_measures.pkl', 'wb') as f:
        pickle.dump(result_dict, f)

    wandb.save(f'{wandb.run.dir}/uncertainty_measures.pkl')

    if args.analyze_run:
        analyze_run(wandb.run.id)


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
