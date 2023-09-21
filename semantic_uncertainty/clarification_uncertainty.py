import argparse
import os
import pathlib
import random

import numpy as np
import openai
import torch
from datasets import load_dataset
from evaluate import load
from sklearn.metrics import auc, roc_curve
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id, get_semantic_ids

random.seed(10)

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load SQuAD dataset from Hugging Face
squad_dataset = load_dataset("squad_v2")

predictive_entropies = []
true_labels = []

meteor = load('meteor')
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v2-xlarge-mnli")


def get_entropy_for_question(model, question, context, num_generations=5):
    """Get entropy for a question and context pair."""

    responses_log_likelihoods = []

    prompt = "Answer the following question as briefly as possible.\n"
    if context is not None:
        prompt += f"Context: {context}\n"
    prompt += f"Question: {question}\nAnswer: "
    responses = []
    print('Question', question)
    for _ in range(num_generations):

        # prompt = f"Answer the following question as briefly as possible. Context: {context}\nQuestion: {question}\nAnswer: ",

        predicted_answer, token_log_likelihoods, _ = model.predict(prompt, temperature=1.0)

        responses.append(predicted_answer)
        avg_token_log_likelihood = np.mean(token_log_likelihoods)
        responses_log_likelihoods.append(avg_token_log_likelihood)
        # print(f"Question: {question}\Answer: {response['choices'][0]['text']}\nAverage Token Log-likelihood: {avg_token_log_likelihood}\n")

        print('predicted answer:', predicted_answer)

    semantic_ids = get_semantic_ids(responses)
    print('semantic ids', semantic_ids)
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, responses_log_likelihoods)
    print('log_likelihood_per_semantic_id', log_likelihood_per_semantic_id)
    # Compute the logsumexp of the response likelihood for every semantic id in semantc_ids

    print('responses_log_likelihoods', responses_log_likelihoods)
    semantic_entropy = predictive_entropy(log_likelihood_per_semantic_id)
    entropy = predictive_entropy(responses_log_likelihoods)

    # total_meteor_score = 0
    # num_comparisons = 0
    # for i in range(len(responses)):
    #     for j in range(i+1, len(responses)):
    #         results = meteor.compute(predictions=[responses[i]], references=[responses[j]])
    #         total_meteor_score += results['meteor']
    #         num_comparisons += 1

    # average_meteor_score = -total_meteor_score / num_comparisons

    return {'entropy': entropy, 'semantic_entropy': semantic_entropy}


def predictive_entropy(log_probs):
    # probs = np.exp(log_probs)
    entropy = -np.sum(log_probs) / len(log_probs)
    return entropy
