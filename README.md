s# Detecting Hallucinations in Large Language Models Using Semantic Entropy

This repository contains the code necessary to reproduce the results of the Nature submission 'Detecting Hallucinations in Large Language Models Using Semantic Entropy'.


## System Requirements

We here discuss system requirements to execute the code that reproduces the experiments.

### Hardware Dependencies

Our experiments require modern computer hardware which is suited for usage with large language models (LLMs).

Requirements regarding the system's CPU and RAM size are relatively modest: any reasonably modern system should suffice, e.g. a system with an Intel 10th generation CPU and 32 GB of system memory.

More importantly, all our experiments make use of one or more Graphics Processor Units (GPUs) to speed up LLM inference.
Without a GPU, it is not feasible to reproduce our results in a reasonable amount of time.
The particular GPU necessary depends on the choice of LLM: LLMs with more parameters require GPUs with more memory.
For smaller 7B models, desktop GPUs such as a Nvidia TitanRTX (24 GB) are sufficient.
For larger 13B models, server GPUs, such as the Nvidia A100 (80GB), are required.
Our largest models with 70B parameters require the use of two Nvidia A100 (2*80GB).


### Software Dependencies

Our code relies on Python 3.11 with PyTorch 2.1.

Our systems run Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-89-generic x86_64).

In `environment_export.yaml`, we list the precise versions for all Python packages.

We would expect our code to run across other operating systems and Python versions.


## Installation Guide


To install Python with all necessary dependencies, we recommend you use conda.

We refer to [https://conda.io/](https://conda.io/) for an installation guide.

After installing conda, you can set up and activate the conda environment by executing the following commands in a linux shell at the root folder of this repository:


```
conda-env update -f environment.yaml
conda activate semantic_uncertainty
```

The installation should take around 15 minutes.

Our experiments rely on [Weights & Biases](https://wandb.ai/) to log results.
While wandb will be installed automatically with the above conda script, you may need to log in with your wandb API key upon initial execution.
We do not support execution without wandb.

Our experiments rely on Hugging Face for all LLM models and most of the datasets.
It may be necessary to set the environment variable `HUGGING_FACE_HUB_TOKEN` to the token associated with your Hugging Face account.
Further, it may be necessary to [apply for access](https://huggingface.co/meta-llama) to use the official repository of Meta's LLaMa-2 models.
We recommend setting the `XDG_CACHE_HOME` environment variable to a directory on a device with sufficient space, as models and datasets will be downloaded to this folder.


Our experiments with sentence-length generation use GPT models from the OpenAI API.
Please set the environment variable `OPENAI_API_KEY` to your OpenAI API key in order to use these models.
Note that OpenAI charges a cost per input token and per generated token.
Costs for reproducing our results vary depending on experiment configuration, but, without any guarantee, should lie somewhere between 10 and 100 USD.


For almost all tasks, the dataset is downloaded automatically from HuggingFace Datasets library upon first execution.
Only for bioasq, data needs to be [downloaded](http://participants-area.bioasq.org/datasets) manually and put in the following directory `$SCRATCH_DIR/$USER/uncertainty`, where `$SCRATCH_DIR` defaults to `.`.



## Demo

Execute

```
python generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
```

to reproduce results for short-phrase generation with LLaMa-2 Chat (7B) on the TriviaQA dataset.

The expected runtime of this demo is 1 hour using an A100 GPU, 24 cores of a Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz, and 192 GB of RAM.
Runtime may be longer upon first execution, as models need to be downloaded first.


Note down the wandb id assigned to your demo run.

To obtain a barplot similar to those of the paper, open the the iPython notebook in `notebooks/example_evaluation.ipynb`, populate `wandb_id` with the id of your demo run, and execute all cells.


We refer to [https://jupyter.org/](https://jupyter.org/) for more information on how to start the jupter notebook (or jupterlab) server.


## Further Instructions


### Repository Structure

We here give an overview over the various components of the code.

By default, a standard run executes the following three scripts in order

* `generate_answers.py`: Sample responses (and their likelihods/hidden states) from the models for the questions.
* `compute_uncertainties.py`: Compute uncertainty metrics given responses.
* `analyze_results.py`: Compute aggregate performance metrics.

While it is possible to run scripts individually, e.g. when recomputing results, and we are happy to provide guidance on how to do so upon request.


### Reproducing the Experiments

To reproduce the experiments of the paper, one just needs to run the above demo for the various combinations of models and datasets.

In other words, execute

```
python generate_answers.py --model_name=$MODEL --dataset=$DATASET $EXTRA_CFG
```

where

* `$MODEL` is one of: `[Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-2-7b-chat, Llama-2-13b-chat, Llama-2-70b-chat, falcon-7b, falcon-40b, falcon-7b-instruct, falcon-40b-instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1]`,
* `$DATASET` is one of `[trivia_qa, squad, med_qa, bioasq, record, nq, svamp],
* and `$EXTRA_CFG` is empty for short-phrase generation and for sentence-length generation, `EXTRA_CFG=--num_few_shot=0 --model_max_new_tokens=100 --brief_prompt=chat --metric=llm_gpt-4 --entailment_model=gpt-3.5 --no-compute_accuracy_at_all_temps`.


The results for any run can be obtained by passing their `wandb_id` to an evaluation notebook identical to the demonstration in `notebooks/example_evaluation.ipynb`.
