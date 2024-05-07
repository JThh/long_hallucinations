#!/bin/bash
# SBATCH --cpus-per-task=8
# SBATCH --gres=gpu:1
# SBATCH --nodelist=oat13
# SBATCH --job-name="nlg_uncertainty_linearprobe"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export HF_DATASETS_CACHE=/scratch-ssd/$USER/cache
export TRANSFORMERS_CACHE=/scratch-ssd/$USER/cache

source /scratch-ssd/oatml/miniconda3/bin/activate llm

srun python ../semantic_uncertainty/generate_answers.py --model_name=Phi-3-mini-128k-instruct --dataset=trivia_qa --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=Phi-3-mini-128k-instruct --dataset=squad --num_samples=2000 --random_seed=20 --answerable_only --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=Phi-3-mini-128k-instruct --dataset=svamp --num_samples=1000 --use_context --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
# srun python ../semantic_uncertainty/generate_answers.py --model_name=Phi-3-mini-128k-instruct --dataset=bioasq --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
# python ../semantic_uncertainty/generate_answers.py --model_name=Phi-3-mini-128k-instruct --dataset=nq --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
