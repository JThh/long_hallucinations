import json

with open('./llama3_405b_100_gen_results.json', 'r') as f:
    results = json.load(f)
# results: a list of {prompt, response} pairs

with open('./long-form-factuality/results/2024-10-03-llama3.1-405b.json', 'r') as f:
    tosave = json.load(f)
    
# import pdb; pdb.set_trace()

tosave['max_num_examples'] = 40
tosave['responder_model'] = 'LLAMA:llama3.1-405b'
tosave['responder_model_short'] = 'llama3.1-405b'

per_prompt_data = []

for i, result in enumerate(results):
    if i >= 40:
        break
    data = {}
    print(i, 'prompt:', result['prompt'])
    print('response:', result['response'])
    if "Unexpected response format: 'choices'" in result['response']:
        print(i)
        continue
    data['prompt'] = result['prompt']
    data['correct_answers'] = []
    data['incorrect_answers'] = []
    data['side1_response'] = '[PLACEHOLDER RESPONSE]'
    data['side2_response'] =  result['prompt'] + '\n' + result['response']
    
    per_prompt_data.append(data)   

tosave['per_prompt_data'] = per_prompt_data

with open('./long-form-factuality/results/2024-10-03-llama3.1-405b.json', 'w') as f:
    json.dump(tosave, f)
    