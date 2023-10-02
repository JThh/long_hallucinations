#!/bin/bash
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:2
# SBATCH --cpus-per-task=8
# SBATCH --gres=gpu:2g.20gb:1
# SBATCH --nodelist=oat15

#SBATCH --job-name="SE"
#SBATCH --output=log/slurm-%j.out
#SBATCH --error=log/slurm-%j.out

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/


/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate llm
# pip uninstall -y tokenizers
# pip uninstall -y tokenizers
# pip install tokenizers==0.13.3


# 156437
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa
# 156438
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=record
# 156439
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=squad
# 156440
srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq







# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='up7kvqqr'


# export FINEGRAINED_THRESH_ACC=TRUE
# srun python semantic_uncertainty/analyze_results.py --wandb_runids '2uriyhaa'
# srun "${@}"

# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='558dqnib' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# srun python semantic_uncertainty/generate_perplexity.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=bioasq
# srun python semantic_uncertainty/generate_perplexity.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=squad
# srun python semantic_uncertainty/generate_perplexity.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=trivia_qa


# export FIX_40B_INST_TRIVIA=TRUE
# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b-instruct --temperature=1 --num_samples=200 --dataset=trivia_qa --restore_id=crw6miul
# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=trivia_qa

# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-7b --temperature=1 --num_samples=200 --dataset=bioasq
# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=trivia_qa --num_generations=20

# falcon7b, bioasq, hyz9iqca
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='hyz9iqca' --train_wandb_runid='eq8dr7r7' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='eq8dr7r7' --train_wandb_runid='hyz9iqca' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# falcon-7b, trivia-qa, eq8dr7r7


# New 40b  results
# 146903 --> jlko/q4x222zx --> 146955 --> goatml/2uriyhaa
# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=bioasq
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='q4x222zx' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik

# 146900 --> jlko/c7z5xw6s -->
# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b --temperature=1 --num_samples=200
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='c7z5xw6s' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik


# 146902 --> jlko/w5yhlnws --> 146952 --> goatml/udi392s2
# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=trivia-qa
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='w5yhlnws' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik

# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-40b --temperature=1 --num_samples=200 --dataset=trivia_qa


# srun python semantic_uncertainty/generate_answers.py --model_name=falcon-7b --temperature=1 --num_samples=200 --dataset=squad


# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='mhd7keb5' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik


# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='mhd7keb5' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik


# trivia_qa to ....
# to bioasq
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='mhd7keb5' --train_wandb_runid='q4x222zx' --restore_entity_train='jlko' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# to record
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='mhd7keb5' --train_wandb_runid='c7z5xw6s' --restore_entity_train='jlko' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# to squad
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='mhd7keb5' --train_wandb_runid='w5yhlnws' --restore_entity_train='jlko' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# to trivia_qa
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='mhd7keb5' --train_wandb_runid='mhd7keb5' --restore_entity_train='jlko' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik



# bioasq to trivia-qa
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='q4x222zx' --restore_entity_eval='jlko' --train_wandb_runid='mhd7keb5' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# record to trivia-qa
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='c7z5xw6s' --restore_entity_eval='jlko' --train_wandb_runid='mhd7keb5' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik
# squad to trivia-qa
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='w5yhlnws' --restore_entity_eval='jlko' --train_wandb_runid='mhd7keb5' --assign_new_wandb_id --compute_predictive_entropy --compute_p_ik







