"""Compute p_true uncertainty metric."""
import logging
import random
from evaluate import load


squad_metric = load("squad_v2")


PROMPT_TEMPLATE = """Question: Who was the third president of the United States?
Here are some brainstormed ideas: James Monroe
Thomas Jefferson
John Adams
Thomas Jefferson
George Washington
Possible Answer: James Monroe
Is the possible answer:
A) True
B) False
The possible answer is: B"""


def construct_few_shot_prompt(model, dataset, n_shots, prompt, brief, brief_always, make_prompt):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times
    few_shot_prompt = ''

    # sample n_shot integers without replacement from the range 0, len(dataset) - 1
    indices = random.sample(range(0, len(dataset) - 1), n_shots)

    for i in indices:
        example = dataset[i]
        question = example["question"]
        context = example["context"]

        few_shot_prompt += '\nQuestion: ' + question
        few_shot_prompt += '\nBrainstormed Answers: '
        local_prompt = prompt + make_prompt(context, question, None, brief, brief_always)

        responses = []
        for j in range(5):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)

            responses.append(response)
            few_shot_prompt += f'{response.strip()} \n'
            if j == 0:
                most_likely_response = response
                prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': example['id']}
                answer_starts = [answer_start for answer_start in example['answers']['answer_start']]
                answers = [answer for answer in example['answers']['text']]
                reference = {'answers': {'answer_start': answer_starts, 'text': answers}, 'id': example['id']}
                results = squad_metric.compute(predictions=[prediction], references=[reference])
                logging.info('Fewshot prompt results: %s', results)
                is_correct = results['f1'] > 50.0

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

