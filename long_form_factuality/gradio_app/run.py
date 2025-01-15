import logging
import random
import os
import json
import pandas as pd
import torch
import gradio as gr
import nltk
import threading
import html  # Importing html module for escaping
import spacy
from factscore.factprobescorer import FactProbeScorer

# Initialize NLTK tokenizer
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

# Define constants
PROMPT_TEMPLATE = "{}\n\nProvide as many specific details and examples as possible (such as names of people, numbers, events, locations, dates, times, etc.)."
DEFAULT_PROMPTS = [
    'Who is Quoc V. Le?',
    'Who is Xochitl Gomez?',
    'What happened in the first modern Olympics?',
    'Tell me about the company Acorns.',
    'Give me an introduction of the city of Bucaramanga.',
    'Tell me about the Antiguan racer snake.',
]

# Define supported models
SUPPORTED_MODELS = {
    "LLaMA 3.1-8B": {
        "model_name": "llama3.1-8B",
        "hf_model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "probe_path": "./metrics/Llama3.1-8B_best_probe_layers_13-17_C_0.5.pkl",
        "probe_path": "./metrics/Llama3.1-8B_best_probe_layers_13-17_xgboost.pkl",
    },
    "Gemma 2-9B": {
        "model_name": "gemma2-9B",
        "hf_model_name": "google/gemma-2-9b-it",
        # "probe_path": "./metrics/Gemma2-9B_best_probe_layers_24-28_C_0.5.pkl",
        "probe_path": "./metrics/Gemma2-9B_best_probe_layers_20-20_xgboost.pkl",
    },
    # Add more models here as needed
}

# Define cache file path
CACHE_FILE = 'cache.json'

def load_cache(file_path):
    """
    Load the cache from a JSON file.
    If the file doesn't exist or is corrupted, return an empty dictionary.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("Cache file is corrupted. Starting with an empty cache.")
            return {}
    return {}

def save_cache(cache_dict, file_path):
    """
    Save the cache to a JSON file without atomic replacement.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def initialize_fact_probe_scorer(model_selection):
    """
    Initialize the FactProbeScorer based on the selected model.
    """
    model_info = SUPPORTED_MODELS.get(model_selection)
    if not model_info:
        raise ValueError(f"Model '{model_selection}' is not supported.")

    fact_probe_scorer = FactProbeScorer(
        model_name=model_info["model_name"],
        probe_path=model_info["probe_path"],
        hf_model_name=model_info["hf_model_name"],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        openai_key=os.getenv('OPENAI_API_KEY', 'your-openai-api-key'),  # Ensure you set this in your environment
        cache_dir='/scratch/ms23jh'
    )
    return fact_probe_scorer

# Initialize a global stop event
stop_event = threading.Event()

def generate_and_analyze(selected_sample, custom_prompt, model_selection, num_tokens, use_cache):
    """
    Generates a response based on the selected or custom prompt,
    processes it to extract atomic facts, and yields the colored response text
    sentence by sentence, including atomic claims details.
    """
    # Determine the prompt
    if custom_prompt and custom_prompt.strip():
        prompt = custom_prompt.strip()
    elif selected_sample and selected_sample.strip() and selected_sample != "--Select a Sample Prompt--":
        prompt = selected_sample.strip()
    else:
        prompt = random.choice(DEFAULT_PROMPTS)

    logger.info(f"Selected Prompt: {prompt}")

    # Check if the prompt is already in cache
    if use_cache:
        cached_result = cache.get(prompt)
        if cached_result:
            logger.info(f"Retrieving cached result for prompt: {prompt}")
            # Wrap the entire response in an expandable section
            expandable_response = f"""
            <details>
                <summary>View Generated Response</summary>
                {cached_result['response_html']}
            </details>
            """
            yield expandable_response, cached_result['atomic_claims_html']
            return

    # Basic validation: Ensure the prompt ends with a question mark or period
    if not prompt.endswith('?') and not prompt.endswith('.'):
        yield "<p>Please ensure your prompt ends with a '?' or a '.'</p>", ""
        return

    # Initialize FactProbeScorer based on model selection
    try:
        fact_probe_scorer = initialize_fact_probe_scorer(model_selection)
    except ValueError as ve:
        logger.error(str(ve))
        yield f"<p>Error: {str(ve)}</p>", ""
        return

    # Prepare the full prompt
    full_prompt = PROMPT_TEMPLATE.format(prompt)

    # Generate response
    try:
        response = fact_probe_scorer.generate_response(
            full_prompt,
            max_new_tokens=int(num_tokens)
        )
        logger.info("Response generated successfully.")
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        yield f"<p>Error during response generation: {str(e)}</p>", ""
        return

    # Wrap the entire response in an expandable section
    expandable_response = f"""
    <details>
        <summary>View Generated Response</summary>
        <div><p style='color:white;'>{response}</p></div>
    </details>
    """
    yield expandable_response, ""

    # Process the response using process_generation_yield
    try:
        # Initialize variables to store atomic claims
        atomic_claims_html = ""

        # Use process_generation_yield to get incremental facts and sentences
        for facts, sentence_text in fact_probe_scorer.process_generation_yield(prompt, response):
            # Tokenize the sentence
            doc = nlp(sentence_text)
            sentence_tokens = [token.text for token in doc]
            token_to_claim = {}
            contributing_tokens_dict = {}  # Dictionary to store contributing tokens per claim

            # Map tokens to claims and support probabilities
            for result in facts:
                token_indices = result['token_indices']  # Indices within the sentence
                revised_atomic_claim = result['revised_atomic_claim']
                support_probability = result['support_probability']
                is_relevant = result['is_relevant']

                # Collect contributing tokens for this claim
                contributing_tokens = []
                for idx in token_indices:
                    if idx < len(sentence_tokens):
                        token = sentence_tokens[idx]
                        contributing_tokens.append(token)
                    else:
                        contributing_tokens.append(f"<span style='color:red;'>Invalid Token Index: {idx}</span>")

                # Log the fact, sentence, score, and contributing tokens
                logger.info(f"Sentence: {sentence_text}")
                logger.info(f"Claim: {revised_atomic_claim}")
                logger.info(f"Support Probability: {support_probability}")
                logger.info(f"Contributing Tokens: {' '.join(contributing_tokens)}")

                # Map tokens to claims and support probabilities
                for idx in token_indices:
                    token_to_claim[idx] = {
                        'revised_atomic_claim': revised_atomic_claim,
                        'support_probability': support_probability,
                        'is_relevant': is_relevant
                    }

                # Store contributing tokens in the dictionary for this claim
                contributing_tokens_dict[revised_atomic_claim] = contributing_tokens

            # Build the colored HTML for the current sentence
            colored_sentence = build_colored_sentence(fact_probe_scorer, sentence_tokens, token_to_claim)

            # Build the atomic claims details for the sentence, using the actual sentence as the summary
            sentence_claims_html = build_atomic_claims_html(sentence_text, facts, contributing_tokens_dict)

            # Append to the cumulative response_html
            # Since the original response is already in an expandable section, we append sentences outside it
            # For exact text replication, ensure no additional spaces or line breaks are added
            sentence_html = f"{colored_sentence}"

            # Append to the cumulative response_html
            expandable_response += sentence_html

            # Append atomic claims details
            atomic_claims_html += sentence_claims_html

            # Check if stop_event is set
            if stop_event.is_set():
                yield expandable_response, atomic_claims_html
                logger.info("Generation stopped by user.")
                return

            # Yield the updated response_html and atomic_claims_html
            yield expandable_response, atomic_claims_html

    except Exception as e:
        logger.error(f"Error during response processing: {e}")
        yield f"<p>Error during response processing: {str(e)}</p>", ""
        return

    # After processing all sentences, save to cache
    if use_cache:
        cache[prompt] = {'response_html': expandable_response, 'atomic_claims_html': atomic_claims_html}
        save_cache(cache, CACHE_FILE)


def build_colored_sentence(fact_probe_scorer, tokens, token_claims_slice):
    """
    Build an HTML string for a sentence with tokens colored based on support probabilities
    and enhanced tooltips showing claims and support scores.
    """
    colored_tokens = []
    for idx, token in enumerate(tokens):
        claim_info = token_claims_slice.get(idx)
        if claim_info:
            support_probability = claim_info['support_probability']
            revised_atomic_claim = claim_info['revised_atomic_claim']
            is_relevant = claim_info['is_relevant']
            # Map the support_probability to a color (green for high, red for low)
            red = int(255 * (1 - support_probability))
            green = int(255 * support_probability)
            blue = 0
            color = f'rgb({red},{green},{blue})'
            hover_text = f"Claim: {revised_atomic_claim}<br>Is Relevant: {'Yes' if is_relevant else 'No'}<br>Support Probability: {support_probability:.2f}"
            # Create tooltip using CSS classes
            colored_token = f'''
                <span class="tooltip" style="color:{color};">
                    {token}
                    <span class="tooltiptext">{hover_text}</span>
                </span>
            '''
        else:
            # Default color for tokens not in any claim (white)
            colored_token = f'''
                <span class="tooltip" style="color:white;">
                    {token}
                </span>
            '''
        colored_tokens.append(colored_token)
    # Join all tokens without adding extra spaces
    colored_sentence = ''.join(colored_tokens)
    return f"<p>{colored_sentence}</p>"


def build_atomic_claims_html(sentence, facts, contributing_tokens_dict):
    """
    Build HTML for atomic claims details for a sentence.
    The summary of the details tag is the actual sentence.
    Also includes contributing tokens for each claim.
    """
    if not facts:
        return ""

    # Sanitize the sentence to prevent HTML injection
    sanitized_sentence = sentence.replace('<', '&lt;').replace('>', '&gt;')

    html_content = f"<details><summary>{sanitized_sentence}</summary>"
    html_content += "<ul>"
    for result in facts:
        revised_atomic_claim = result['revised_atomic_claim']
        support_probability = result['support_probability']
        # Sanitize the claim to prevent HTML injection
        sanitized_claim = revised_atomic_claim.replace('<', '&lt;').replace('>', '&gt;')

        # Get contributing tokens from the dictionary
        contributing_tokens = contributing_tokens_dict.get(revised_atomic_claim, [])
        # Escape contributing tokens for HTML
        escaped_contributing_tokens = [html.escape(token) for token in contributing_tokens]
        contributing_tokens_str = " ".join(escaped_contributing_tokens)

        html_content += f"<li><strong>Claim:</strong> {sanitized_claim}<br>"
        html_content += f"<strong>Support Probability:</strong> {support_probability:.2f}<br>"
        html_content += f"<strong>Contributing Tokens:</strong> {contributing_tokens_str}</li>"
    html_content += "</ul></details>"
    return html_content


def stop_generation():
    """
    Sets the stop_event to True to halt the generation and analysis process.
    """
    stop_event.set()
    logger.info("Stop signal received.")
    return "Generation stopped."

with gr.Blocks() as demo:
    # Add custom CSS for tooltips and remove spacing between chunks
    gr.HTML("""
    <style>
    /* Tooltip container */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    /* Tooltip text */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 220px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position above the text */
        left: 50%;
        margin-left: -110px; /* Center the tooltip */
        opacity: 0;
        transition: opacity 0.3s;
    }

    /* Tooltip arrow */
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%; /* At the bottom of the tooltip */
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #555 transparent transparent transparent;
    }

    /* Show the tooltip text when you mouse over the tooltip container */
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Remove spacing between yielded chunks */
    #response_box div, #response_box p {
        margin-bottom: 0px;
    }
    </style>
    """)

    gr.Markdown("# FactProbeScorer Visualization")
    gr.Markdown("""
    Generate and analyze responses with colored tokens based on support probabilities.
    The response will render incrementally, sentence by sentence, with tokens colored accordingly.
    
    **Hover over tokens** to see the associated atomic claim and its support probability.
    """)

    with gr.Row():
        with gr.Column():
            model_selection = gr.Dropdown(
                choices=list(SUPPORTED_MODELS.keys()),
                label="Select Model",
                value="LLaMA 3.1-8B"
            )
            selected_sample = gr.Dropdown(
                choices=["--Select a Sample Prompt--"] + DEFAULT_PROMPTS,
                label="Sample Prompts",
                value="--Select a Sample Prompt--"
            )
            custom_prompt = gr.Textbox(
                label="Or enter a custom prompt (e.g., 'Who is XXX?')",
                placeholder="Enter your custom prompt here..."
            )
            num_tokens = gr.Number(
                label="Max Number of Tokens",
                value=512,
                precision=0
            )
            use_cache = gr.Checkbox(
                label="Use Cached Response",
                value=False
            )
            generate_button = gr.Button("Generate and Analyze")
            stop_button = gr.Button("Stop")
        with gr.Column():
            output_response = gr.HTML(
                label="Response",
                elem_id="response_box",
                value="<p></p>"
            )
            atomic_claims_details = gr.HTML(
                label="Atomic Claims Details",
                elem_id="atomic_claims_details",
                value=""
            )

    # Link the Stop button to set the stop_event
    stop_button.click(
        stop_generation,
        inputs=None,
        outputs=None
    )

    def on_generate(selected_sample, custom_prompt, model_selection, num_tokens, use_cache):
        # Reset stop_event before starting
        stop_event.clear()
        # Start the generate_and_analyze function and yield results
        yield from generate_and_analyze(selected_sample, custom_prompt, model_selection, num_tokens, use_cache)

    # Link the Generate button to the on_generate function
    generate_button.click(
        on_generate,
        inputs=[selected_sample, custom_prompt, model_selection, num_tokens, use_cache],
        outputs=[output_response, atomic_claims_details],
        show_progress=True
    )

    gr.Markdown("## Instructions")
    gr.Markdown("""
    - **Select a Model:** Choose the model you want to use for generating and analyzing responses.
    - **Select a Sample Prompt:** Choose from the predefined prompts.
    - **Enter a Custom Prompt:** Input your own prompt.
    - **Max Number of Tokens:** Specify the maximum number of tokens to generate (default is 512).
    - **Use Cached Response:** Uncheck if you want to generate a new response even if one exists in the cache.
    - **Generate and Analyze:** Click the button to generate the response and visualize the analysis.
    - **Stop:** If you wish to halt the ongoing generation and analysis, click the "Stop" button.

    **Note:** 
    - **Incremental Rendering:** The response will appear incrementally, sentence by sentence, with tokens colored based on their support probabilities.
    - **Atomic Claims Details:** Expand each sentence's atomic claims to view detailed information, including contributing tokens.
    - **Hover Text:** Hover over a token to see the associated atomic claim and its support probability.
    - **Default Token Color:** Tokens not associated with any claim are displayed in **white**.
    """)

# Initialize the cache
cache = load_cache(CACHE_FILE)

# Launch the Gradio app
demo.launch(share=True)
