"""Compute p_true uncertainty metric."""
import logging
import random
from evaluate import load


squad_metric = load("squad_v2")


PROMPT_TEMPLATE = """Question: Who was the third president of the United States?
Brainstormed Answers: James Monroe
Thomas Jefferson
John Adams
Thomas Jefferson
George Washington
Possible Answer: James Monroe
Is the possible answer:
A) True
B) False
The possible answer is: B"""


def construct_few_shot_prompt(model, dataset, indices, prompt, brief, brief_always, make_prompt):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times
    few_shot_prompt = ''

    # TODO: Why are we not using the context to construct the p_true few-shot prompt?

    for i in indices:
        example = dataset[i]
        question = example["question"]
        context = example["context"]

        few_shot_prompt += '\nQuestion: ' + question
        few_shot_prompt += '\nBrainstormed Answers: '
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) +  current_question)

        responses = []
        for j in range(5):

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
                prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': example['id']}
                answer_starts = [answer_start for answer_start in example['answers']['answer_start']]
                answers = [answer for answer in example['answers']['text']]
                reference = {'answers': {'answer_start': answer_starts, 'text': answers}, 'id': example['id']}
                results = squad_metric.compute(predictions=[prediction], references=[reference])
                is_correct = results['f1'] > 50.0
                logging.info('P_TRUE >> LOW-T >> answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> results: '.ljust(35) + str(results))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        few_shot_prompt += 'Possible answer: ' + most_likely_response + '\n'
        few_shot_prompt += 'Is the possible answer:\n'
        few_shot_prompt += 'A) True\n'
        few_shot_prompt += 'B) False\n'
        few_shot_prompt += 'The possible answer is:'
        few_shot_prompt += ' A' if is_correct else ' B'

    return few_shot_prompt


def calculate_p_true(model, question, most_probable_answer, brainstormed_answers, few_shot_prompt):
    """Calculate p_true uncertainty metric."""

    prompt = PROMPT_TEMPLATE + few_shot_prompt

    prompt += '\nQuestion: ' + question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    prompt += 'Is the possible answer:\n'
    prompt += 'A) True\n'
    prompt += 'B) False\n'
    prompt += 'The possible answer is:'

    log_prob = model.get_p_true(prompt)

    return log_prob
