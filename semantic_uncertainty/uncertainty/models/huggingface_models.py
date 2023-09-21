"""Implement HuggingfaceModel models."""
import logging
from collections import Counter
import numpy as np
import torch
import torch.utils._pytree as pytree

import accelerate

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import snapshot_download


from uncertainty.models.base_model import BaseModel


def remove_split_layer(device_map):
    """Modify device maps s.t. individual layers are not spread across devices."""

    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            raise ValueError('More than one split layer')

        print(f'Split layer is {layer}')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None):

        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            else:
                kwargs = {}

            if 'Llama-2' in model_name:
                base = 'meta-llama'
                model_name = model_name + '-hf'
            else:
                base = 'huggyllama'

            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None)

            llama65b = '65b' in model_name and base == 'huggyllama'
            llama2_70b = '70b' in model_name and base == 'meta-llama'

            if '7b' in model_name or '13b' in model_name:
                self.model = AutoModelForCausalLM.from_pretrained(
                    f"{base}/{model_name}", device_map="auto", **kwargs)

            elif llama2_70b or llama65b:
                path = snapshot_download(
                    repo_id=f'{base}/{model_name}',
                    allow_patterns=['*.json', '*.model', '*.safetensors'],
                    ignore_patterns=['pytorch_model.bin.index.json']
                )
                config = AutoConfig.from_pretrained(f"{base}/{model_name}")
                # config.load_in_8bit = True
                with accelerate.init_empty_weights():
                    self.model = AutoModelForCausalLM.from_config(config)
                self.model.tie_weights()

                max_mem = 15 * 4686198491 # 4G*15
                device_map = accelerate.infer_auto_device_map(
                    self.model.model,
                    max_memory={0: max_mem, 1: max_mem},
                    dtype='float16'
                )
                device_map = remove_split_layer(device_map)
                full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
                full_model_device_map["lm_head"] = 0

                # get snapshot folder
                self.model = accelerate.load_checkpoint_and_dispatch(
                    self.model, path, device_map=full_model_device_map,
                    dtype='float16')
            else:
                raise ValueError

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = stop_sequences

    def predict(self, input_data, temperature):

        # TODO @lorenz: Investigate this for clarify. Why are the inputs tuples sometimes?
        if isinstance(input_data, tuple):
            input_data = input_data[0]

        # Implement prediction.
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")

        if 'llama' in self.model_name or 'falcon' in self.model_name:
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

        logging.debug('temperature: %f', temperature)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=25,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                do_sample=True,
            )

        answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)

        # For some models, we need to remove the input_data from the answer.
        if answer.startswith(input_data):
            input_data_offset = len(input_data)
            n_tokens_in_input = inputs['input_ids'].shape[1]
        else:
            input_data_offset, n_tokens_in_input = 0, 0

        start_index, stop_index = self.get_character_start_stop_indices(input_data_offset, answer)

        if start_index < stop_index:
            sliced_answer = answer[start_index:stop_index]
        else:
            sliced_answer = answer[start_index:]
            logging.warning(
                'Problematic generation: start_index %d, stop_index %d, ignoring stop_index!', start_index, stop_index)
            stop_index = -1

        logging.debug('Answer: %s', sliced_answer)

        # Get token index of first stop sequence to cut off likelihoods/embeddings correctly.
        token_stop_index = self.tokenizer(answer[:stop_index], return_tensors="pt")['input_ids'].shape[1]

        # Get the last hidden state (last layer) and the last token's embedding of the answer.
        # Note: The output embeddings have the shape (batch_size, generated_length, hidden_size). We do not get
        # embeddings for input_data! We thus subtract the n_tokens_in_input from
        # token_stop_index to arrive at the right output.
        if 'decoder_hidden_states' in outputs.keys():
            last_hidden_state = outputs.decoder_hidden_states[-1][token_stop_index - 1 - n_tokens_in_input]
        else:
            last_hidden_state = outputs.hidden_states[-1][token_stop_index - 1 - n_tokens_in_input]
        last_token_embedding = last_hidden_state[:, -1, :].cpu()

        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)
        # transition_scores[0] only contains the scores for the first generated tokens.
        start_off = 0
        log_likelihoods = [score.item() for score in transition_scores[0]]
        log_likelihoods = log_likelihoods[start_off:token_stop_index - n_tokens_in_input]

        if len(log_likelihoods) == 0:
            logging.warning(
                (
                    'len(log_likelihoods) == 0 after answer slicing, take last '
                    'loglik instead.\n'
                    'Answer: \n""""\n%s\n"""\nSliced Answer:\n""""\n%s\n"""'
                ),
                answer, sliced_answer)
            log_likelihoods = [transition_scores[0][-1].item()]
        return sliced_answer, log_likelihoods, last_token_embedding

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = torch.tensor(self.tokenizer([input_data])['input_ids'], device='cpu')

        # This computation of the negative log likelihoods follows this tutorial:
        #  https://huggingface.co/docs/transformers/perplexity

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -1.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        # This computation of the negative log likelihoods follows this tutorial:
        #  https://huggingface.co/docs/transformers/perplexity

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()

        # skip first token. (will be start token for llama anyways)
        # output : B A C
        # input  : A B C
        # I've checked that the above implementation is equivalent to what I think is reasonable, which is:
        # -torch.mean(torch.tensor([
        #     torch.nn.functional.log_softmax(output, 0)[token] for output, token
        #     in zip(model_output_true['logits'][0], tokenized_data[0][1:])]))

        return perplexity


def demo():
    """Demo."""
    model = HuggingfaceModel('FlanUL2')
    input_string = "Answer the following question by reasoning step by step. The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apple do they have?"  # pylint: disable=line-too-long # noqa: E501
    print(model.predict(input_string, temperature=1))


if __name__ == "__main__":
    demo()
