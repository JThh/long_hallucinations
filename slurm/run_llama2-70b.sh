#!/bin/bash
# SBATCH --cpus-per-task=24
# SBATCH --partition=jiatong
# SBATCH --gres=gpu:a100:2
# SBATCH --job-name="llama2-70b-nlg_uncertainty_linearprobe"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export HF_DATASETS_CACHE=/scratch-ssd/$USER/cache
export HF_HOME=/scratch-ssd/$USER/cache

# /scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ../environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate llm

srun python ../semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=svamp --num_samples=1000 --use_context --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=squad --num_samples=2000 --random_seed=20 --answerable_only --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=nq --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable