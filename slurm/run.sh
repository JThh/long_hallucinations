#!/bin/bash
# SBATCH --cpus-per-task=4
# SBATCH --gres=gpu:1
# SBATCH --job-name="nlg_uncertainty"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export HF_DATASETS_CACHE=/scratch-ssd/oatml/data/
export TRANSFORMERS_CACHE=/scratch-ssd/oatml/huggingface/hub

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate llm


srun python ../semantic_uncertainty/generate_answers.py --model_name=Mistral-7B-v0.1 --temperature=1 --dataset=squad --num_samples=200
#srun python code/compute_uncertainty_measures.py --wandb_runid='1dtxdla5'
