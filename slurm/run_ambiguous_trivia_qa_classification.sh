#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --nodelist=oat15
#SBATCH --job-name="nlg_uncertainty"
``
export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
# export HF_DATASETS_CACHE=/scratch-ssd/loruhn/uncertainty
# export TRANSFORMERS_CACHE=/scratch-ssd/loruhn/uncertainty

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
# source /scratch-ssd/oatml/miniconda3/bin/activate llm

n_samples=200
# model_name='llama-7b'
model_name='falcon-7b-instruct'
# model_name='falcon-7b'
dataset='ambiguous-trivia-qa'
type_of_question="all"
srun python code/clarify.py --model_name=$model_name --stage_name='detect_ambiguity' --dataset=$dataset --n_samples=$n_samples
srun python code/clarify.py --model_name=$model_name --stage_name='entropy' --dataset=$dataset --n_samples=$n_samples
srun python code/clarify.py --model_name=$model_name --stage_name='prompting_baseline' --dataset=$dataset --n_samples=$n_samples
srun python code/clarify.py --model_name=$model_name --stage_name='give_initial_answer' --dataset=$dataset --n_samples=$n_samples
srun python code/clarify.py --model_name=$model_name --stage_name='ask_clarifying_question' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples
srun python code/clarify.py --model_name=$model_name --stage_name='provide_clarifying_information' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples
srun python code/clarify.py --model_name=$model_name --stage_name='give_final_answer' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples
