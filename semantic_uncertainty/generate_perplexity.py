"""Predict with LLM on task."""
import argparse
import os
import logging
import pickle
import random
from tqdm import tqdm

import numpy as np
import torch
import openai
import wandb
from evaluate import load

from uncertainty.data.data_utils import load_ds
from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.models.oai_models import OpenAIModel
from uncertainty.utils import utils

from uncertainty.uncertainty_measures.p_true import calculate_p_true, construct_few_shot_prompt


utils.setup_logger()
random.seed(10)
# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

# Implement argparsers
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="oai.code-davinci-002", help="Model name",
    choices=[
        'oai.code-davinci-002', 'oai.text-davinci-002',
        'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b',
        'FlanUL2', 'T5', 'alpaca-lora', 'gpt-neo-2.7B',
        'falcon-7b', 'falcon-40b', 'falcon-7b-instruct', 'falcon-40b-instruct'])
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
parser.add_argument("--restore_id", type=str, default=None)
parser.add_argument('--compute_p_true', default=True,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--entity', type=str, default='goatml')
parser.add_argument(
    "--brief_always", default=False, action=argparse.BooleanOptionalAction)


args, unknown = parser.parse_known_args()
if unknown:
    raise ValueError(f'Unkown args: {unknown}')
# Load SQuAD dataset from Hugging Face

train_dataset, validation_dataset = load_ds(
    args.dataset, add_options=args.use_mc_options)
logging.info('Train dataset: %s', train_dataset)

squad_metric = load("squad_v2")

# Get indices of answerable and unanswerable questions and construct prompt.
answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)
answerable_sample = random.sample(answerable_indices, args.num_few_shot)
brief = "Answer the following question as briefly as possible.\n"  # pylint: disable=invalid-name



stop_sequences = ['\n', 'Question:', 'Context:']


def init_model(args):
    if args.model_name == "FlanUL2":
        model = HuggingfaceModel('FlanUL2', stop_sequences=stop_sequences)
    elif args.model_name == 'T5':
        model = HuggingfaceModel('T5', stop_sequences=stop_sequences)
    elif ('llama' in args.model_name) or ('alpaca' in args.model_name) or ('neo' in args.model_name)\
            or ('falcon' in args.model_name):
        model = HuggingfaceModel(args.model_name, stop_sequences=stop_sequences)
    elif args.model_name.startswith('oai'):
        model = OpenAIModel(args.model_name.split('.')[1], stop_sequences=stop_sequences)
    else:
        raise ValueError(f'Unknown model_name `{args.model_name}`.')
    return model


model = init_model(args)

user = os.environ['USER']
slurm_jobid = os.getenv('SLURM_JOB_ID')
if not os.path.exists(f"/scratch-ssd/{user}/uncertainty"):
    os.makedirs(f"/scratch-ssd/{user}/uncertainty")

if args.restore_id is not None:
    logging.warning('Restoring existing run at %s', args.restore_id)
    kwargs = {'resume': True, 'id': args.restore_id}
else:
    kwargs = {}

wandb.init(
    # set the wandb project where this run will be logged
    project="uncertainty",
    entity=args.entity,
    dir=f"/scratch-ssd/{user}/uncertainty",
    # track hyperparameters and run metadata
    config={
        "dataset": args.dataset,
        "model": args.model_name,
        "num_samples": args.num_samples,
        "num_few_shot": args.num_few_shot,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "compute_p_true": args.compute_p_true
    },
    notes=f'slurm_id: {slurm_jobid}',
    **kwargs
)
logging.info('Finished wandb init.')


logging.info('Generating answers: ')

perplexities = []
example_ids = []

for dataset_split in ['validation']:
    logging.info('Starting with dataset_split %s.', dataset_split)

    if dataset_split == 'train':
        if not args.get_training_set_generations:
            continue
        dataset = train_dataset

    else:
        dataset = validation_dataset

    # Evaluate over random subset of the datasets.
    indices = random.sample(range(0, len(dataset)), min(args.num_samples, len(dataset)))

    if args.num_samples > len(dataset):
        logging.info('Not enough samples in dataset. Using all %d samples.', len(dataset))
    it = 0
    for index in tqdm(indices):
        it += 1
        # torch.cuda.empty_cache()  # fix memory leaks?
        # if it % 30 == 0:
            # logging.info('REINIT MODEL TO FIGHT MEMORY LEAKS.')
            # torch.cuda.empty_cache()  # fix memory leaks?
            # model = init_model(args)

        # Grab example at index.
        example = dataset[index]
        question, context, answer = example["question"], example['context'], example['answers']['text']
        prompt = ''
        if context:
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}"
        if answer:
            prompt += f"\nAnswer: {answer[0]}"

        logging.info(prompt)
        # We sample 1 low temperature answer on which we will compute the
        # accuracy and args.num_generation high temperature answers which will
        # be used to estimate the entropy.
        perplexity = model.get_perplexity(prompt)

        # Assemble `prediction` and `reference` for
        # squad_metric.compute().
        perplexities.append(perplexity)
        example_ids.append(example['id'])
        logging.info('iteration %i id %s perplexity %f', it, example['id'], perplexity)
        wandb.log({
            'id': example['id'],
            'perplexity': perplexity
        })

    logging.info('mean_perplexity: %f', np.mean(perplexities))
    wandb.log({"mean_perplexity": np.mean(perplexities)})
    logging.info('std_perplexity: %f', np.std(perplexities))
    wandb.log({"std_perplexity": np.std(perplexities)})

    # write generations file to json
    with open(f'{wandb.run.dir}/{dataset_split}_perplexities.pkl', 'wb') as f:
        pickle.dump(dict(perplexities=perplexities, example_ids=example_ids), f)

    logging.info('finished!')
