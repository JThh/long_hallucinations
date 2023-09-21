dataset='claqua_multiturn_clarification'
model_name='oai.code-davinci-002'
type_of_question='ambiguous'
n_samples=5

python code/clarify.py --model_name=$model_name --stage_name='give_initial_answer' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples
python code/clarify.py --model_name=$model_name --stage_name='ask_clarifying_question' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples
python code/clarify.py --model_name=$model_name --stage_name='provide_clarifying_information' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples
python code/clarify.py --model_name=$model_name --stage_name='give_final_answer' --dataset=$dataset --type_of_question=$type_of_question --n_samples=$n_samples