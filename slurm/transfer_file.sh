#!/bin/bash
#SBATCH --job-name=file-transfer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=transfer-%j.out
#SBATCH --error=transfer-%j.err
#SBATCH --partition=msc
#SBATCH --nodelist=oat11

# Rsync over Slurm's internal network
scp -r /scratch-ssd/ms23jh/uncertainty/wandb/run-20240423_152924-2979rbip ms23jh@oat0:/scratch-ssd/ms23jh/uncertainty/wandb/
