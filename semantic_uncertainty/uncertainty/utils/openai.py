import os
import logging
import hashlib
from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

import openai


openai.api_key = os.environ['OPENAI_API_KEY_OX']


@retry(wait=wait_random_exponential(min=5, max=60))
def predict(prompt, temperature=1.0):
    """Predict with GPT-4 model."""
    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    output = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output['choices'][0]['message']['content']
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
