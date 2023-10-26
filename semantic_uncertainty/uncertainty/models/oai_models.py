"""OpenAI API predictions."""
import os

import openai
import tiktoken

from uncertainty.models.base_model import BaseModel
# from uncertainty.utils.openai import call_openai_api
import numpy as np
import logging

openai.api_key = os.environ["OPENAI_API_KEY"]


class OpenAIModel(BaseModel):
    """Query OpenAI API for predictions."""

    def __init__(self, model_name='text-davinci-002', stop_sequences=None):
        self.model_name = model_name

        self.stop_sequences = stop_sequences
        # TODO: Figure out which OAI models need this.
        if model_name == 'text-davinci-002':
            self.use_oai_stop = False
        else:
            self.use_oai_stop = True
        self.encoding = tiktoken.encoding_for_model(self.model_name)

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.encoding.encode(string))
        return num_tokens

    def predict(self, input_data, temperature):
        raise

        # local_prompt = input_data

        # We only pass the first four stop sequences due to API limits, and then post-process the response with
        # all stop sequences.
        # response = call_openai_api(
            # local_prompt, self.model_name, temperature=temperature,
            # stop=self.stop_sequences[:4] if self.use_oai_stop else None)

        # token_log_likelihoods = response["choices"][0]["logprobs"]["token_logprobs"]

        # contains_stop_sequence = False
        # if self.use_oai_stop and len(self.stop_sequences) > 4:
        #     # Check if any of the self.stop_sequences are in the response.
        #     for word in self.stop_sequences:
        #         index = response['choices'][0]['text'].find(word)
        #         if index != -1:
        #             contains_stop_sequence = True

        # char_stop_index = len(response['choices'][0]['text'])
        # token_stop_index = len(token_log_likelihoods)
        # if contains_stop_sequence or not self.use_oai_stop:
        #     # Postprocess generation
        #     # We pass the empty string as input_data because the openai api doesn't echo back the input
        #     # in the response.
        #     char_start_index, char_stop_index = self.get_character_start_stop_indices(0, response['choices'][0]['text'])
        #     token_stop_index = len(self.encoding.encode(response['choices'][0]['text'][:char_stop_index]))
        #     sliced_response = response['choices'][0]['text'][char_start_index:char_stop_index]
        # else:
        #     sliced_response = response['choices'][0]['text']
        # return sliced_response, token_log_likelihoods[:token_stop_index], None

    def get_p_true(self, input_data):
        """Returns the probability of the last token of the given input under the model."""
        raise
        # response = call_openai_api(
        #     prompt=input_data, engine=self.model_name, max_tokens=2, logprobs=100, temperature=0, echo=True)
        # logprobs = response["choices"][0]["logprobs"]["top_logprobs"]

        # if " A" in logprobs[-1]:
        #     return logprobs[-1][" A"]
        # elif " A" in logprobs[-2]:
        #     return logprobs[-2][" A"]
        # else:
        #     logging.warning('Token \" A\" not found in response: %s', response)
