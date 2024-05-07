#!/bin/bash
# SBATCH --cpus-per-task=24
# SBATCH --gres=gpu:1
# SBATCH --partition=jiatong
# SBATCH --job-name="gemma-nlg_uncertainty_linearprobe"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export HF_DATASETS_CACHE=/scratch-ssd/$USER/cache
export TRANSFORMERS_CACHE=/scratch-ssd/$USER/cache

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ../environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate llm

srun python ../semantic_uncertainty/generate_answers.py --model_name=gemma-7b-it --dataset=trivia_qa --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=gemma-7b-it --dataset=squad --num_samples=2000 --answerable_only --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=gemma-7b-it --dataset=nq --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=gemma-7b-it --dataset=bioasq --num_samples=2000 --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable
srun python ../semantic_uncertainty/generate_answers.py --model_name=gemma-7b-it --dataset=svamp --num_samples=1000 --use_context --random_seed=20 --no-get_training_set_generations --no-compute_p_ik --no-compute_p_ik_answerable