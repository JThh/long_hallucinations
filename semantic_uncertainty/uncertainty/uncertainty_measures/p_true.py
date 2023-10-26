"""Compute p_true uncertainty metric."""
import logging
import random
from evaluate import load


squad_metric = load("squad_v2")


def construct_few_shot_prompt(
        *, model, dataset, indices, prompt, brief, brief_always, make_prompt, num_generations, metric):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times
    few_shot_prompt = ''

    # TODO: Why are we not using the context to construct the p_true few-shot prompt?

    for it, i in enumerate(indices):
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            few_shot_prompt += '\n'
        few_shot_prompt += 'Question: ' + question
        few_shot_prompt += '\nBrainstormed Answers: '
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            responses.append(response)
            few_shot_prompt += f'{response.strip()} \n'
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        few_shot_prompt += 'Possible answer: ' + most_likely_response + '\n'
        few_shot_prompt += 'Is the possible answer:\n'
        few_shot_prompt += 'A) True\n'
        few_shot_prompt += 'B) False\n'
        few_shot_prompt += 'The possible answer is:'
        few_shot_prompt += ' A' if is_correct else ' B'

    return few_shot_prompt


def calculate_p_true(model, question, most_probable_answer, brainstormed_answers, few_shot_prompt, hint=False):
    """Calculate p_true uncertainty metric."""

    if few_shot_prompt:
        prompt = few_shot_prompt + '\n'
    else:
        prompt = ''

    prompt += 'Question: ' + question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    if not hint:
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
    else:
        prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

    log_prob = model.get_p_true(prompt)

    return log_prob
