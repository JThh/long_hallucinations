#!/bin/bash
# SBATCH --cpus-per-task=10
# SBATCH --gres=gpu:a100:1
# SBATCH --nodelist=oat11
# SBATCH --job-name="nlg_uncertainty_linearprobe"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export HF_DATASETS_CACHE=/scratch-ssd/$USER/cache
export HF_HOME=/scratch-ssd/$USER/cache

# /scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f ../environment.yaml
# source /scratch-ssd/oatml/miniconda3/bin/activate semantic_uncertainty


# /scratch-ssd/$USER/conda_envs/semantic_uncertainty/bin/python -c "import torch; torch.cuda.is_available()"

# pip install -U datasets

python ../semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b --temperature=1 --dataset=squad --num_samples=2000 --num_eval_samples 200
python ../semantic_uncertainty/generate_answers.py --model_name=falcon-7b-instruct --temperature=1 --dataset=trivia_qa --num_samples=2000 --num_eval_samples 200
# srun python ../semantic_uncertainty/generate_answers.py --model_name=Mistral-7B-v0.1-4bit --temperature=1 --dataset=svamp --num_samples=200
#srun python code/compute_uncertainty_measures.py --wandb_runid='1dtxdla5'