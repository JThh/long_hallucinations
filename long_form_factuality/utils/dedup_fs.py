import pickle

# path = './FActScore/data/unlabeled/Llama3.1-8B_fact_scores_nent_30_temp_0.5_maxtok_512_api_gpt-4o-mini.pkl'
# path = './FActScore/data/unlabeled/Gemma2-9B_fact_scores_nent_30_temp_0.5_maxtok_512_api_gpt-4o-mini.pkl'
# path = './FActScore/data/unlabeled/Llama3.2-3B_fact_scores_nent_30_temp_0.5_maxtok_512_api_gpt-4o-mini.pkl'
path = './FActScore/data/unlabeled/Llama3.1-70B_fact_scores_nent_25_temp_0.5_maxtok_512_api_gpt-4o-mini.pkl'

with open(path, 'rb') as f:
    sc = pickle.load(f)
    

for i, claims in enumerate(sc['decisions']):
    decisions = []
    claim_set = set()  
    for claim in claims:
        if claim['atom'] not in claim_set:
            decisions.append(claim)
            claim_set.add(claim['atom'])
    sc['decisions'][i] = decisions
    

with open(path, 'wb') as f:
    pickle.dump(sc, f)