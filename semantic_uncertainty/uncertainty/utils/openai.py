import os
import logging
import hashlib
from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

from openai import OpenAI


CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY_OX'])


@retry(wait=wait_random_exponential(min=5, max=20))
def predict(prompt, temperature=1.0, model='gpt-4'):
    """Predict with GPT-4 model."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    if model == 'gpt-4':
        model = 'gpt-4-0613'
    elif model == 'gpt-4-turbo':
        model = 'gpt-4-1106-preview'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo-1106'

    output = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
