import seaborn as sns

COLORS = {
    "semantic_entropy_sum-normalized-rao": '#0173b2', #'#89CFF0',
    "FQADebertaEntailment": '#56b4e9',
    "FQADebertaEntailment-A1": '#67d8ff',
    "FQADebertaEntailment-A2": '#5ec6ff',
    "FQADebertaEntailment-A4": '#4da2d1',
    "FQADebertaEntailment-A5": '#4490ba',
    "FQADebertaEntailment-A6": '#3c7da3',
    "cluster_assignment_entropy": '#56b4e9',
    "regular_entropy": "#029e73",
    "p_ik": '#ece133',
    "p_ik_ood": '#ece133',
    "p_false": '#d55e00',
    "FptrueOrig": '#d55e00',
    "ER - BioASQ" : "#ece133",
    "ER - TriviaQA": "#ece103",
    "ER - SQuAD" : "#ece133",
    "Fself_check": "#fbafe4",
    'deberta_not_answerable': sns.color_palette("hls", 5)[0],
    'entailment': sns.color_palette("hls", 5)[1],
    'entailment_llm': sns.color_palette("hls", 5)[2],
    'equivalent_llm': sns.color_palette("hls", 5)[3],
    '1way_entailment_llm': sns.color_palette("hls", 5)[4]
}

CROSSHATCH = {
    "Semantic Entropy": '',
    "Naive Entropy": '',
    "Embedding Regression": '',
    "p(True) - Kadavath et al.": '',
    "clarify_when_appropriate": '',
    "detect_ambiguity": '',
    "ER - BioASQ" : "///",
    "ER - TriviaQA": "---",
    "ER - SQuAD" : "\\\\\\",
}

MODEL_NAMES = {
    'falcon-7b': 'Falcon 7B',
    'falcon-40b': 'Falcon 40B',
    'falcon-7b-instruct': 'Falcon 7B Instruct',
    'falcon-40b-instruct': 'Falcon 40B Instruct',
    'llama-7b': 'LLaMA 7B',
    'llama-13b': 'LLaMA 13B',
    'llama-65b': 'LLaMA 65B',
    'Llama-2-70b-chat': 'LLaMA 2 Chat 70B',
    'Llama-2-7b-chat': 'LLaMA 2 Chat 7B',
    'Llama-2-13b-chat': 'LLaMA 2 Chat 13B',
    'Llama-2-70b': 'LLaMA 2 70B',
    'Llama-2-7b': 'LLaMA 2 7B',
    'Llama-2-13b': 'LLaMA 2 13B',
    'falcon-40b-instruct': 'Falcon 40B Instruct',
    'Mistral-7B-Instruct-v0.1': 'Mistral 7B Instruct',
    'Mistral-7B-v0.1': 'Mistral 7B',
}

PRETTY_NAMES = {
    "bioasq": "BioASQ",
    "trivia_qa": "TriviaQA",
    "squad": "SQuAD",
    "svamp": "SVAMP",
    "nq": "NQ Open",
    "cluster_assignment_entropy": "Discrete Semantic Entropy",
    "regular_entropy": "Naive Entropy",
    "semantic_entropy_sum-normalized-rao": "Semantic Entropy",
    "p_ik": "Embedding Regression",
    "p_ik_ood": "Embedding Regression - OOD",
    "p_false": "p(True) - Kadavath et al.",
    "Fself_check": "Self-check Baseline",
    "FptrueOrig": "P(True) - Kadavath et al. variant",
    "FQADebertaEntailment": "Discrete Semantic Entropy",
}

METRIC_NAMES = {
    "auroc": "AUROC",
    "accuracy_at_0.1_answer_fraction": "10%",
    "accuracy_at_0.2_answer_fraction": "20%",
    "accuracy_at_0.3_answer_fraction": "30%",
    "accuracy_at_0.4_answer_fraction": "40%",
    "accuracy_at_0.5_answer_fraction": "50%",
    "accuracy_at_0.6_answer_fraction": "60%",
    "accuracy_at_0.7_answer_fraction": "70%",
    "accuracy_at_0.8_answer_fraction": "80%",
    "accuracy_at_0.9_answer_fraction": "90%",
    "accuracy_at_0.95_answer_fraction": "95%",
    "accuracy_at_1_answer_fraction": "100%",
    "accuracy_at_1.0_answer_fraction": "100%",
    "area_under_thresholded_accuracy": "AU Rejection\nAccuracy\nCurve",
}

TEXTWIDTH = 5.14838
GOLDEN_RATIO = 1.618