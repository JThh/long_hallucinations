#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --nodelist=oat17
#SBATCH --job-name="notebook"
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.out


export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export SLURM_CLUSTER_NAME=oatcloud
export XDG_CACHE_HOME=/scratch-ssd/oatml/

# Start a notebook on some oat machine.

echo "Starting jupyter notebook"

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate /scratch-ssd/jansen/conda_envs/llm_unc

jupyter-lab --no-browser --port=8888 --ip 127.0.0.1


# run with sbatch slurm/notebook.sh
