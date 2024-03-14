#!/bin/bash
# SBATCH --cpus-per-task=24
# SBATCH --gres=gpu:a100:2
# SBATCH
# SBATCH --gres=gpu:2g.20gb:1
# SBATCH --gres=gpu:3g.40gb:1
# SBATCH --gres=gpu:titanrtx:1
# SBATCH --exclude=oat2


# SBATCH --nodelist=oat15

SBATCH --job-name="SE-test-13Mar"
SBATCH --output=log/slurm-%j.out
SBATCH --error=log/slurm-%j.out

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs
export XDG_CACHE_HOME=/scratch-ssd/oatml/


/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yaml
source /scratch-ssd/oatml/miniconda3/bin/activate semantic_uncertainty
# pip install safetensors
# pip uninstall -y tokenizers
# pip uninstall -y tokenizers
# pip install tokenizers==0.13.3

srun "${@}"

# OOD evals



# is gpt35 or gpt-4-turbo enough?
# extra_cfg="--entailment_model=gpt-3.5 --no-use_all_generations --use_num_generations=8"
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=dqtye228 --num_eval_samples=200 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=dqtye228 --num_eval_samples=200 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=dqtye228 --num_eval_samples=200 $extra_cfg



# 30/10 Ablation over num generations:
# extra_cfg="--p_true_num_fewshot=20 --num_generations=15 --num_few_shot=0 --model_max_new_tokens=100 --model_name=Llama-2-70b-chat-8bit --brief_prompt=chat --metric=llm --entailment_model=gpt-4 --num_samples=400"
# sbatch --gres=gpu:a100:1 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --dataset=bioasq $extra_cfg
# sbatch --gres=gpu:a100:1 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --dataset=trivia_qa --no-use_context $extra_cfg
# sbatch --gres=gpu:a100:1 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --dataset=squad --answerable_only --no-use_context $extra_cfg

# 31/10 Due to bug: need to run compute_uncertainty separately for these runs
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=1gxo9oef --entailment_model=gpt-4
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=uvfbxm6d --entailment_model=gpt-4

# HERE
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=dqtye228 --entailment_model=gpt-4 --num_eval_samples=200

# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/analyze_results.py --wandb_runids=i855qzau

# 31/10 Uncertainties at other num_generations

# HERE
# extra_cfg="--entailment_model=gpt-4 --compute_p_true_in_compute_stage --no-use_all_generations --entailment_cache_only  --eval_wandb_runid=dqtye228 --entailment_cache_id=goatml/semantic_uncertainty/i855qzau --num_eval_samples=200"
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=3 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=4 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=5 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=6 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=7 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=8 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=9 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=10 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=11 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=12 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=13 $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --use_num_generations=14 $extra_cfg




# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/analyze_results.py --wandb_runids='ooslwgb9' --assign_new_wandb_id
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/analyze_results.py --wandb_runids='au7s3uxg' --assign_new_wandb_id
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/analyze_results.py --wandb_runids='e5ptgcwr' --assign_new_wandb_id


# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=A --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='ooslwgb9' --entailment_model=Llama-2-70b-chat-8bit --no-strict_entailment
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=A --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='au7s3uxg' --entailment_model=Llama-2-70b-chat-8bit --no-strict_entailment
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=A --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='e5ptgcwr' --entailment_model=Llama-2-70b-chat-8bit --no-strict_entailment


# llama-entailment
# ~~164212--164214~~ --> strict entailment (old runs strict entailment by accident)
# ~164226~, ~~164227~, ~164228~ --> no strict entailment (new runs now with strict entailment disabled)
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=A --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='ooslwgb9' --entailment_model=Llama-2-70b-chat-8bit --no-strict_entailment
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=B --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='au7s3uxg' --entailment_model=Llama-2-70b-chat-8bit --no-strict_entailment
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=C --gres=gpu:a100:1 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='e5ptgcwr' --entailment_model=Llama-2-70b-chat-8bit --no-strict_entailment

# ~164232~ (rerun gpt-4 without strict entailment)
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='ooslwgb9' --entailment_model=gpt-4 --no-strict_entailment

# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='au7s3uxg' --entailment_model=gpt-4 --strict_entailment
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='e5ptgcwr' --entailment_model=gpt-4 --strict_entailment

# e5ptgcwr,

# rerun with strict entailment (compare against gpt-4 --> are gains just from strict entailment?!)
# ~~164233~~
# sbatch --gres=gpu:1 --cpus-per-task=8 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='ooslwgb9' --strict_entailment


# rerun with strict entailment
# ~164238~ -- 164239
# sbatch --gres=gpu:1 --cpus-per-task=8 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='au7s3uxg' --strict_entailment
# sbatch --gres=gpu:1 --cpus-per-task=8 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid='e5ptgcwr' --strict_entailment




# LONG GENERATIONS
# 162438 -- 16240
# extra_cfg="--p_true_num_fewshot=7 --num_generations=6 --num_few_shot=0 --model_max_new_tokens=100 --metric=llm --brief_prompt=chat --no-strict_entailment"
# sbatch --gres=gpu:titanrtx:1 --cpus-per-task=8 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat-8bit --dataset=bioasq $extra_cfg
# sbatch --gres=gpu:titanrtx:1 --cpus-per-task=8 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat-8bit --dataset=trivia_qa --no-use_context $extra_cfg
# sbatch --gres=gpu:titanrtx:1 --cpus-per-task=8 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat-8bit --dataset=squad --answerable_only --no-use_context $extra_cfg



# P_FALSE OOD!
# 163648,163680, 163681, 163682
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=A --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq --ood_train_dataset=trivia_qa $extra_cfg
# extra_cfg="--p_true_num_fewshot=5 --num_generations=6 --num_few_shot=0 --model_max_new_tokens=100 --metric=llm --brief_prompt=chat --no-strict_entailment"
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=B --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq --ood_train_dataset=squad $extra_cfg
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=A --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa  --no-use_context $extra_cfg --ood_train_dataset=bioasq $extra_cfg
# sbatch --cpus-per-task=24 --dependency=singleton  --job-name=B --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa  --no-use_context $extra_cfg --ood_train_dataset=squad $extra_cfg


# P_FALSE ZERO-SHOT
# extra_cfg="--p_true_num_fewshot=0 --num_generations=6 --num_few_shot=0 --model_max_new_tokens=100 --metric=llm --brief_prompt=chat --no-strict_entailment"
# sbatch --dependency=singleton  --job-name=B --gres=gpu:a100:2 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq $extra_cfg
# sbatch --dependency=singleton  --job-name=A --gres=gpu:a100:2 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa --no-use_context $extra_cfg
# sbatch --dependency=singleton  --job-name=C --gres=gpu:a100:2 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=squad --answerable_only --no-use_context $extra_cfg
# sbatch --dependency=singleton  --job-name=A --gres=gpu:a100:2 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq --p_true_hint $extra_cfg
# sbatch --dependency=singleton  --job-name=C --gres=gpu:a100:2 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa --no-use_context --p_true_hint $extra_cfg
# sbatch --dependency=singleton  --job-name=A --gres=gpu:a100:2 --cpus-per-task=24 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=squad --answerable_only --no-use_context --p_true_hint $extra_cfg


# sbatch --cpus-per-task=24 --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa --no-use_context $extra_cfg
# sbatch --cpus-per-task=24 --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=squad --answerable_only --no-use_context

# 163517
# extra_cfg="--p_true_num_fewshot=5 --num_generations=6 --num_few_shot=0 --model_max_new_tokens=100 --metric=llm --brief_prompt=chat --no-strict_entailment"
# sbatch --qos=priority --cpus-per-task=24 --gres=gpu:a100:2 slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=record $extra_cfg



# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b --dataset=record
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b --dataset=trivia_qa --no-use_context
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b --dataset=squad --answerable_only --no-use_context


# 162099-162103
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-8bit --dataset=trivia_qa --no-use_context
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-8bit --dataset=trivia_qa
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-8bit --dataset=squad
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-8bit --dataset=squad --answerable_only
# sbatch slurm/run_uncertainty_estimates.sh python semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-8bit --dataset=squad --answerable_only --no-use_context



# MORE GENERATIONS
# 160176
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b --dataset=trivia_qa --num_generations=15

# 160177
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b --dataset=bioasq --num_generations=15


# Num-Generations baseline for
# 160229
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=o161v4v5 --no-use_all_generations --use_num_generations=12
# 160230
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=o161v4v5 --no-use_all_generations --use_num_generations=10
# 160231
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=o161v4v5 --no-use_all_generations --use_num_generations=8
# 160232
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=o161v4v5 --no-use_all_generations --use_num_generations=6
# 160233
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=o161v4v5 --no-use_all_generations --use_num_generations=4
# 160234
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=o161v4v5 --no-use_all_generations --use_num_generations=2


# 160235
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=l4gww8m1 --no-use_all_generations --use_num_generations=12
# 160236
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=l4gww8m1 --no-use_all_generations --use_num_generations=10
# 160237
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=l4gww8m1 --no-use_all_generations --use_num_generations=8
# 160238
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=l4gww8m1 --no-use_all_generations --use_num_generations=6
# 160239
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=l4gww8m1 --no-use_all_generations --use_num_generations=4
# 160240
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=l4gww8m1 --no-use_all_generations --use_num_generations=2



# we cannto do this for bioasq (b/c no context), so let's just do record
# 160178
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b --dataset=record --compute_context_entails_response
# srun python semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=4ug9fcne --compute_context_entails_response


# 156437
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=trivia_qa
# 156438
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=record
# 156439
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=squad
# 156440
# srun python semantic_uncertainty/generate_answers.py --model_name=Llama-2-70b-chat --dataset=bioasq

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







