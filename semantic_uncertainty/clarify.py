import pathlib
from evaluate import load
import argparse
import json
import os

import datasets
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
# Compute auroc between clarification_need and log_probs
# from sklearn.metrics import roc_auc_score
# import sklearn
import seaborn as sns
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from clarification import prompts
from clarification_uncertainty import get_entropy_for_question
from uncertainty.utils.utils import check_for_clarification_request

from uncertainty.models.huggingface_models import HuggingfaceModel
from uncertainty.models.oai_models import OpenAIModel

sns.set_palette('pastel')

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="oai.code-davinci-002", help="Model name",
    choices=[
        'oai.code-davinci-002', 'oai.text-davinci-002',
        'llama-7b', 'llama-13b', 'llama-30b', 'llama-65b',
        'FlanUL2', 'T5', 'alpaca-lora', 'gpt-neo-2.7B',
        'falcon-7b', 'falcon-40b', 'falcon-7b-instruct', 'falcon-40b-instruct'])
parser.add_argument('--dataset', type=str, default='clariq',
                    choices=['ambiguous-trivia-qa', 'clariq', 'claqua_singleturn_classification',
                             'claqua_singleturn_clarification', 'claqua_multiturn_classification',
                             'claqua_multiturn_clarification'])
parser.add_argument('--type_of_question', type=str, default='all', choices=['all', 'ambiguous', 'precise'])
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--stage_name', type=str, default='prompting_baseline',
                    choices=['detect_ambiguity', 'entropy', 'prompting_baseline', 'give_initial_answer',
                             'ask_clarifying_question', 'provide_clarifying_information', 'give_final_answer'])

args, unknown = parser.parse_known_args()
if unknown:
    raise ValueError(f'Unkown args: {unknown}')
# Load SQuAD dataset from Hugging Face

print(80*'*')
print(f"STARTING {args.stage_name}")
print(80*'*')


squad_metric = load("squad")
# rouge = load('rouge')

openai.api_key = os.getenv("OPENAI_API_KEY")


base_path = pathlib.Path('/scratch-ssd/jansen/clarifying_questions/data/')


stop_sequences = ['Bot:', 'User:', '###', 'The user wants', '\n', 'Question:', 'Context:']
LOW_TEMPERATURE = 0.1
HIGH_TEMPERATURE = 1.0


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

path = base_path / args.model_name / args.dataset

# Create path if it doesn't exists
path.mkdir(parents=True, exist_ok=True)

if args.stage_name == 'detect_ambiguity':
    if args.dataset == 'ambiguous-trivia-qa':
        dataset_path = path / 'ambiguous_questions'
        dataset = datasets.load_from_disk(dataset_path)
        dataset = dataset['train'].select(list(range(args.n_samples)))
    elif args.dataset == 'claqua_singleturn_classification':
        dataset_path = path / 'singleturn_classification_dataset'
        dataset = datasets.load_from_disk(dataset_path).select(list(range(args.n_samples)))

    elif args.dataset == 'claqua_multiturn_classification':
        dataset_path = path / 'claqua_multiturn_classification_dataset'
        dataset = datasets.load_from_disk(dataset_path).select(list(range(args.n_samples)))

    elif args.dataset == 'clariq':
        dataset_path = path / 'clariq_classification_dataset'
        dataset = datasets.load_from_disk(dataset_path).select(list(range(args.n_samples)))

    target_dataset_path = path / 'ambiguous_questions_with_p_true'

elif args.stage_name == 'entropy':
    dataset_path = path / 'ambiguous_questions_with_p_true'
    dataset = datasets.load_from_disk(dataset_path)
    target_dataset_path = path / 'ambiguous_questions_with_p_true_and_entropy'

elif args.stage_name == 'prompting_baseline':
    dataset_path = path / 'ambiguous_questions_with_p_true'
    dataset = datasets.load_from_disk(dataset_path)
    target_dataset_path = path / 'ambiguous_questions_with_p_true_and_prompting_baseline'

elif args.stage_name == 'give_initial_answer':

    if args.dataset == 'ambiguous-trivia-qa':
        dataset_path = path / 'ambiguous_questions_with_p_true_and_prompting_baseline'
        dataset = datasets.load_from_disk(dataset_path)

    elif args.dataset == 'claqua_multiturn_classification' or args.dataset == 'claqua_singleturn_classification':
        dataset_path = path / 'ambiguous_questions_with_p_true_and_prompting_baseline'
        dataset = datasets.load_from_disk(dataset_path)

    elif args.dataset == 'claqua_multiturn_clarification':
        dataset_path = path / 'claqua_multiturn_clarification_dataset'
        dataset = datasets.load_from_disk(dataset_path).select(list(range(args.n_samples)))

    elif args.dataset == 'claqua_singleturn_clarification':
        dataset_path = path / 'claqua_singleturn_clarification_dataset'
        dataset = datasets.load_from_disk(dataset_path).select(list(range(args.n_samples)))

    elif args.dataset == 'clariq':
        dataset_path = path / 'ambiguous_questions_with_p_true_and_prompting_baseline'
        dataset = datasets.load_from_disk(dataset_path)
    # elif args.dataset == 'claqua':
    #     dataset_path = path / 'claqua_single_turn_generation_dataset'
    #     dataset = datasets.load_from_disk(dataset_path).select(list(range(args.n_samples)))
    target_dataset_path = path / 'ambiguous_questions_with_p_true_and_initial_answer'

elif args.stage_name == 'ask_clarifying_question':
    dataset_path = path / 'ambiguous_questions_with_p_true_and_initial_answer'
    dataset = datasets.load_from_disk(dataset_path)
    target_dataset_path = path / 'ambiguous_questions_with_clarifying_questions'

elif args.stage_name == 'provide_clarifying_information':
    dataset_path = path / 'ambiguous_questions_with_clarifying_questions'
    dataset = datasets.load_from_disk(dataset_path)
    target_dataset_path = path / 'ambiguous_questions_with_clarifying_information'

elif args.stage_name == 'give_final_answer':
    dataset_path = path / 'ambiguous_questions_with_clarifying_information'
    dataset = datasets.load_from_disk(dataset_path)
    target_dataset_path = path / 'ambiguous_questions_with_final_answer'

else:
    print('Stage name {} does not exist'.format(args.stage_name))

print('Length of dataset ', len(dataset))

generations = []
embeddings = []

if args.type_of_question == 'all':
    type_of_question_list = ['precise_question', 'ambiguous_question']

elif args.type_of_question == 'precise':
    type_of_question_list = ['precise_question']
elif args.type_of_question == 'ambiguous':
    type_of_question_list = ['ambiguous_question']

for type_of_question in type_of_question_list:
    dataset_key = 'question' if type_of_question == 'precise_question' else 'ambiguous_questions'
    generations = [None] * args.n_samples
    embeddings = [None] * args.n_samples

    for i in range(args.n_samples):
        print(i)
        embedding = None
        if args.stage_name == 'detect_ambiguity':
            if args.dataset == 'ambiguous-trivia-qa':
                prompt = prompts.UNIFIED_P_TRUE_PROMPT.format(question=dataset[i][dataset_key])
            elif 'claqua_singleturn' in args.dataset:
                question_parts = dataset[i][dataset_key].split('\n')
                entity = question_parts[0].split('<S>')[0].split(': ')[1]
                question_core = question_parts[-1]
                question_core = '\"' + question_core + '\" could refer to both entities \"' + entity + '\": True'
                question_parts[-1] = question_core
                question = '\n'.join(question_parts)
                prompt = prompts.P_TRUE_PROMPT_SINGLE_TURN_CLAQUA.format(question=question)
            elif 'claqua_multiturn' in args.dataset:
                question_parts = dataset[i][dataset_key].split('<SPEC>')
                question_parts.pop(1)
                question = ''.join(question_parts)

                prompt = prompts.P_TRUE_PROMPT_MULITURN_CLAQUA.format(question=question)
            elif args.dataset == 'clariq':
                prompt = prompts.UNIFIED_P_TRUE_PROMPT.format(question=dataset[i][dataset_key])

            # print(prompt)
            try:
                generation = model.get_p_true(prompt)
                print(generation)
            except Exception as e:
                print('Generation failed ', e)
                generation = -10

        elif args.stage_name == 'entropy':
            generation = get_entropy_for_question(model, dataset[i][dataset_key], None)
            print(generation)

        elif args.stage_name == 'prompting_baseline':
            if args.dataset == 'claqua_multiturn_classification':
                question = ''.join(dataset[i][dataset_key].split('<SPEC>')[:-1])
                prompt = prompts.PROMPT_THAT_ENCOURAGES_CLARIFICATION.format(initial_question=question)
            else:
                prompt = prompts.PROMPT_THAT_ENCOURAGES_CLARIFICATION.format(initial_question=dataset[i][dataset_key])
            try:
                response, _, _ = model.predict(prompt, temperature=LOW_TEMPERATURE)
                generation = check_for_clarification_request(response)
            except:
                generation = 0.0

        elif args.stage_name == 'give_initial_answer':
            if args.dataset == 'claqua_multiturn_clarification':
                prompt = prompts.GENERIC_QUESTION_PROMPT_CLAQUA.format(initial_question=dataset[i][dataset_key])

            elif args.dataset == 'claqua_singleturn_clarification' and args.model_name == 'text-davinci-002':
                prompt = prompts.GENERIC_QUESTION_PROMPT_CLAQUA_SINGLETURN.format(
                    initial_question=dataset[i][dataset_key])
            else:
                prompt = prompts.GENERIC_QUESTION_PROMPT.format(initial_question=dataset[i][dataset_key])

            try:
                generation, _, embedding = model.predict(prompt, temperature=LOW_TEMPERATURE)
            except:
                generation = None

        elif args.stage_name == 'ask_clarifying_question':

            if args.dataset == 'ambiguous-trivia-qa':
                prompt = prompts.CLARIFYING_QUESTION_PROMPT.format(initial_question=dataset[i][dataset_key])
            elif args.dataset == 'claqua_singleturn_clarification':
                filtered_question = dataset[i][dataset_key].split('\n')[-1]
                prompt = prompts.CLAQUA_SINGLE_TURN_CLARIFYING_QUESTION_PROMPT.format(
                    initial_question=filtered_question)
            elif args.dataset == 'claqua_multiturn_clarification':
                filtered_question = dataset[i][dataset_key].split('\n')[-1]
                prompt = prompts.CLAQUA_MULTI_TURN_CLARIFYING_QUESTION_PROMPT.format(initial_question=filtered_question)
            elif args.dataset == 'clariq':
                prompt = prompts.CLARIFYING_QUESTION_PROMPT.format(initial_question=dataset[i][dataset_key])

            try:
                generation, _, _ = model.predict(prompt, temperature=LOW_TEMPERATURE)

            except:
                generation = None

        elif args.stage_name == 'provide_clarifying_information':
            if args.dataset == 'claqua_singleturn_clarification':
                prompt = prompts.CLAQUA_SINGLETURN_CLARIFYING_INFORMATION_PROMPT.format(
                    precise_question=dataset[i]['question'],
                    initial_question=dataset[i][dataset_key],
                    clarifying_question=dataset[i]['clarifying_question_for_{}'.format(type_of_question)])

            elif args.dataset == 'claqua_multiturn_clarification':
                filtered_question = dataset[i][dataset_key].split('\n')[-1]
                prompt = prompts.CLAQUA_MULTITURN_CLARIFYING_INFORMATION_PROMPT.format(
                    precise_question=dataset[i]['question'],
                    initial_question=filtered_question,
                    clarifying_question=dataset[i]['clarifying_question_for_{}'.format(type_of_question)])
            else:
                prompt = prompts.CLARIFYING_INFORMATION_PROMPT.format(
                    precise_question=dataset[i]['question'],
                    initial_question=dataset[i][dataset_key],
                    clarifying_question=dataset[i]['clarifying_question_for_{}'.format(type_of_question)])

            try:
                generation, _, _ = model.predict(prompt, temperature=LOW_TEMPERATURE)

            except:
                generation = None

        elif args.stage_name == 'give_final_answer':
            if args.dataset == 'claqua_multiturn_clarification':
                prompt = prompts.CLAQUA_MULTITURN_FINAL_ANSWER_PROMPT.format(
                    initial_question=dataset[i][dataset_key],
                    clarifying_question=dataset[i]['clarifying_question_for_ambiguous_question'],
                    clarifying_information=dataset[i]['clarifying_information_for_ambiguous_question'])

            elif args.dataset == 'claqua_singleturn_clarification':
                prompt = prompts.CLAQUA_SINGLETURN_FINAL_ANSWER_PROMPT.format(
                    initial_question=dataset[i][dataset_key],
                    clarifying_question=dataset[i]['clarifying_question_for_ambiguous_question'],
                    clarifying_information=dataset[i]['clarifying_information_for_ambiguous_question'])

            else:

                prompt = prompts.FINAL_ANSWER_PROMPT.format(
                    initial_question=dataset[i][dataset_key],
                    clarifying_question=dataset[i]['clarifying_question_for_ambiguous_question'],
                    clarifying_information=dataset[i]['clarifying_information_for_ambiguous_question'])

            generation, _, _ = model.predict(prompt, temperature=LOW_TEMPERATURE)

        generations[i] = generation

        if embedding is not None:
            embeddings[i] = embedding

    if args.stage_name == 'detect_ambiguity':

        if 'p_true_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('p_true_{}'.format(type_of_question))

        dataset = dataset.add_column('p_true_{}'.format(type_of_question), generations)

    elif args.stage_name == 'entropy':
        if 'entropy_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('entropy_{}'.format(type_of_question))

        if 'semantic_entropy_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('semantic_entropy_{}'.format(type_of_question))
        entropies = []
        semantic_entropies = []

        for generation in generations:
            entropies.append(generation['entropy'])
            semantic_entropies.append(generation['semantic_entropy'])

        dataset = dataset.add_column('entropy_{}'.format(type_of_question), entropies)
        dataset = dataset.add_column('semantic_entropy_{}'.format(type_of_question), semantic_entropies)

    elif args.stage_name == 'give_initial_answer':

        if 'initial_answer_for_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('initial_answer_for_{}'.format(type_of_question))

        if 'embedding_for_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('embedding_for_{}'.format(type_of_question))

        dataset = dataset.add_column('initial_answer_for_{}'.format(type_of_question), generations)
        # dataset = dataset.add_column('embedding_for_{}'.format(type_of_question), embeddings)

    elif args.stage_name == 'prompting_baseline':
        if 'clarification_requested_for_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('clarification_requested_for_{}'.format(type_of_question))

        dataset = dataset.add_column('clarification_requested_for_{}'.format(type_of_question), generations)
    elif args.stage_name == 'ask_clarifying_question':
        if 'clarifying_question_for_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('clarifying_question_for_{}'.format(type_of_question))
        dataset = dataset.add_column('clarifying_question_for_{}'.format(type_of_question), generations)

    elif args.stage_name == 'provide_clarifying_information':

        if 'clarifying_information_for_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('clarifying_information_for_{}'.format(type_of_question))
        dataset = dataset.add_column('clarifying_information_for_{}'.format(type_of_question), generations)

    elif args.stage_name == 'give_final_answer':
        if 'final_answer_for_{}'.format(type_of_question) in dataset.column_names:
            dataset = dataset.remove_columns('final_answer_for_{}'.format(type_of_question))
        dataset = dataset.add_column('final_answer_for_{}'.format(type_of_question), generations)


dataset.save_to_disk(target_dataset_path)
