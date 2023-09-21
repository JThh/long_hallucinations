from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

import openai


@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(100))
def call_openai_api(prompt, engine, temperature=1.0, max_tokens=25, logprobs=1,
                    stop=["Question:", "Answer", "Context:"], echo=False):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=logprobs,
        temperature=temperature,
        echo=echo,
        stop=stop
    )

    return response
