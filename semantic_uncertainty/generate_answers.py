"""Predict with LLM on task."""
import argparse
import os
import logging
import pickle
import random
from tqdm import tqdm

import numpy as np
import openai
import wandb
from evaluate import load

from uncertainty.data.data_utils import load_ds
from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.models.oai_models import OpenAIModel
from uncertainty.utils import utils

from uncertainty.uncertainty_measures import p_true as p_true_utils


utils.setup_logger()
random.seed(10)
# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

# Implement argparsers
parser = argparse.ArgumentParser()
parser.add_argument(
    "--debug", action=argparse.BooleanOptionalAction, default=False,
    help="Keep default wandb clean.")
parser.add_argument(
    "--experiment_lot", type=str, default='Unnamed Experiment',
    help="Keep default wandb clean.")
parser.add_argument(
    "--model_name", type=str, default="oai.code-davinci-002", help="Model name",
)
parser.add_argument(
    "--dataset", type=str, default="record",
    choices=['trivia_qa', 'squad', 'med_qa', 'bioasq', 'record'],
    help="Dataset to use")
parser.add_argument(
    "--num_samples", type=int, default=200,
    help="Number of samples to use")
parser.add_argument(
    "--num_few_shot", type=int, default=5,
    help="Number of few shot examples to use")
parser.add_argument(
    "--num_generations", type=int, default=5,
    help="Number of generations to use")
parser.add_argument(
    "--temperature", type=float, default=1.0,
    help="Temperature")
parser.add_argument(
    "--use_mc_options", type=bool, default=True,
    help="Include MC options question?")
parser.add_argument(
    "--get_training_set_generations", default=True,
    action=argparse.BooleanOptionalAction,
    help="Get generations for training set?")
parser.add_argument(
    "--get_training_set_generations_most_likely_only", default=True,
    action=argparse.BooleanOptionalAction,
    help=(
        "Only get embedding of most likely answer for training set. "
        "This is all that's needed for p_true."))
parser.add_argument('--compute_p_true', default=True,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--entity', type=str, default='goatml')
parser.add_argument(
    "--brief_always", default=False, action=argparse.BooleanOptionalAction)


args, unknown = parser.parse_known_args()
logging.info('Starting new run with args: %s', args)
experiment_details = {'args': args}
if unknown:
    raise ValueError(f'Unkown args: {unknown}')
# Load SQuAD dataset from Hugging Face

user = os.environ['USER']
slurm_jobid = os.getenv('SLURM_JOB_ID')
if not os.path.exists(f"/scratch-ssd/{user}/uncertainty"):
    os.makedirs(f"/scratch-ssd/{user}/uncertainty")

wandb.init(
    entity=args.entity,
    project="uncertainty" if not args.debug else "uncertainty_debug",
    dir=f"/scratch-ssd/{user}/uncertainty",
    config={
        "dataset": args.dataset,
        "model": args.model_name,
        "num_samples": args.num_samples,
        "num_few_shot": args.num_few_shot,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "compute_p_true": args.compute_p_true
    },
    notes=f'slurm_id: {slurm_jobid}, experiment_lot: {args.experiment_lot}',
)
logging.info('Finished wandb init.')


train_dataset, validation_dataset = load_ds(
    args.dataset, add_options=args.use_mc_options)
logging.info('Train dataset: %s', train_dataset)
squad_metric = load("squad_v2")


STOP_SEQUENCES = ['\n', 'Question:', 'Context:']


def init_model(args):
    mn = args.model_name
    if 'llama' in mn.lower() or 'falcon' in mn:
        model = HuggingfaceModel(mn, stop_sequences=STOP_SEQUENCES)
    elif mn.startswith('oai'):
        model = OpenAIModel(mn.split('.')[1], stop_sequences=STOP_SEQUENCES)
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model


model = init_model(args)


# Get indices of answerable and unanswerable questions and construct prompt.
answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)
prompt_indices = random.sample(answerable_indices, args.num_few_shot)
experiment_details['prompt_indices'] = prompt_indices


def make_prompt(context, question, answer, brief, brief_always):
    prompt = ''
    if brief_always:
        prompt += brief
    if context is not None:
        prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\n"
    if answer:
        prompt += f"Answer: {answer}\n\n"
    else:
        prompt += 'Answer:'
    return prompt


BRIEF = "Answer the following question as briefly as possible.\n"

prompt = utils.construct_fewshot_prompt_from_indices(
    train_dataset, prompt_indices, BRIEF, args.brief_always, make_prompt)
logging.info('Prompt is: %s', prompt)


if args.compute_p_true:
    logging.info(80*'#')
    logging.info('Constructing few-shot prompt for p_true.')
    p_true_few_shot_prompt = p_true_utils.construct_few_shot_prompt(
        model=model, dataset=train_dataset, n_shots=args.num_few_shot,
        prompt=prompt, brief=BRIEF, brief_always=args.brief_always,
        make_prompt=make_prompt)
    logging.info('Finished constructing few-shot prompt for p_true.')
    logging.info(80*'#')
    logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
    logging.info(80*'#')


logging.info(80 * '=')
logging.info('Generating answers: ')
logging.info(80 * '=')
for dataset_split in ['train', 'validation']:
    logging.info('Starting with dataset_split %s.', dataset_split)

    # This will store all input data and model predictions.
    accuracies, generations, results_dict, p_trues = [], {}, {}, []

    if dataset_split == 'train':
        if not args.get_training_set_generations:
            logging.info('Skip training data.')
            continue
        dataset = train_dataset
    else:
        dataset = validation_dataset

    # Evaluate over random subset of the datasets.
    indices = random.sample(range(0, len(dataset)), min(args.num_samples, len(dataset)))
    experiment_details[dataset_split] = {'indices': indices}

    if args.num_samples > len(dataset):
        logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

    it = 0
    for index in tqdm(indices):
        it += 1

        # Grab example at index.
        example = dataset[index]
        question, context = example["question"], example['context']
        generations[example['id']] = {'question': question, 'context': context}
        correct_answer = example['answers']['text']
        reference = {
            'answers': {
                'answer_start': example['answers']['answer_start'],
                'text': correct_answer},
            'id': example['id']}

        current_input = make_prompt(context, question, None, BRIEF, args.brief_always)
        local_prompt = prompt + current_input

        logging.info('Current input: '.ljust(15) + current_input)

        full_responses = []
        # We sample 1 low temperature answer on which we will compute the
        # accuracy and args.num_generation high temperature answers which will
        # be used to estimate the entropy.

        if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
            num_generations = 1
        else:
            num_generations = args.num_generations + 1

        for i in range(num_generations):

            # Temperature for first generation is always `0.1`.
            temperature = 0.1 if i == 0 else args.temperature

            predicted_answer, token_log_likelihoods, embedding = model.predict(
                local_prompt, temperature)
            embedding = embedding.cpu() if embedding is not None else None

            # Assemble `prediction` and `reference` for squad_metric.compute().
            prediction = {
                'prediction_text': predicted_answer,
                'id': example['id'],
                'no_answer_probability': 0.0  # Required by squad_metric.
            }

            # Only compute accuracy if question is answerable.
            if correct_answer:
                # Evaluate prediction with
                results = squad_metric.compute(
                    predictions=[prediction], references=[reference])
                acc = 1.0 if (results['f1'] >= 50.0) else 0.0  # pylint: disable=invalid-name
            else:
                results = None
                acc = 0.0  # pylint: disable=invalid-name

            if i == 0:
                logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                logging.info('context: '.ljust(15) + str(context))
                logging.info('question: '.ljust(15) + question)
                logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                logging.info('correct answer: '.ljust(15) + str(correct_answer))
                logging.info('accuracy: '.ljust(15) + str(acc))
                logging.info('results: '.ljust(15) + str(results))

                accuracies.append(acc)
                most_likely_answer_dict = {
                    'response': predicted_answer,
                    'token_log_likelihoods': token_log_likelihoods,
                    'embedding': embedding,
                    'accuracy': acc,
                }

                generations[example['id']].update({
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': reference,
                })
            else:
                logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                # Aggregate predictions over num_generations.
                full_responses.append(
                    (predicted_answer, token_log_likelihoods, embedding, acc))

        # Append all predictions for this example to `generations`.
        generations[example['id']]['responses'] = full_responses

        if args.compute_p_true and dataset_split == 'validation':
            # Already compute p_true here. Avoid heavy lifting in downstream scripts.
            p_true = p_true_utils.calculate_p_true(
                model, question, most_likely_answer_dict['response'],
                [r[0] for r in full_responses], p_true_few_shot_prompt)
            p_trues.append(p_true)
            logging.info('p_true: %s', p_true)

    # Save generations for that split.
    with open(f'{wandb.run.dir}/{dataset_split}_generations.pkl', 'wb') as f:
        pickle.dump(generations, f)
    wandb.save(f'{wandb.run.dir}/{dataset_split}_generations.pkl')

    # Print overall accuracy.
    accuracy = np.mean(accuracies)
    print(f"Overall {dataset_split} split accuracy: {accuracy}")
    wandb.log({f"{dataset_split}_accuracy": accuracy})

    # Already compute p_true here. Avoid heavy lifting in downstream scripts.
    if dataset_split == 'validation':
        if args.compute_p_true:
            p_false = [1 - p for p in p_trues]
            results_dict['uncertainty_measures'] = {'p_false':  p_false}

        with open(f'{wandb.run.dir}/uncertainty_measures.pkl', 'wb') as f:
            pickle.dump(results_dict, f)
        wandb.save(f'{wandb.run.dir}/uncertainty_measures.pkl')

with open(f'{wandb.run.dir}/experiment_details.pkl', 'wb') as f:
    pickle.dump(experiment_details, f)
wandb.save(f'{wandb.run.dir}/experiment_details.pkl')
logging.info('Run complete.')
