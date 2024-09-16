import pickle
import random
from pprint import pprint
import os
import re

# Removed cosine similarity imports as they are no longer needed
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# Constants
# FACTSCORE_FILE_NAME = '../FActScore/data/labeled/Llama3.1-8B_fact_scores_nent_183_temp_0.1_maxtok_128_api_gpt-4o-mini.pkl'
FACTSCORE_FILE_NAME = '../FActScore/data/Gemma2-9B_test_set.pkl'
ENTITIES_FILE = '../FActScore/data/unlabeled/prompt_entities.txt'
NUM_ENTITIES = 200  # Use only 21 for consistency with the Nature paper
# NUM_ENTITIES = 21  # Use only 21 for consistency with the Nature paper
MAJOR_FLAG = 'Major False'  # Flag as False by Longhallu in the SE paper
MAX_CLAIMS_PER_ENTITY = 6  # Maximum number of claims per entity
OUTPUT_PYTHON_FILE = 'wiki_data_gemma.py'  # Output Python file to save the data
RANDOM_SEED = 42  # Set a random seed for reproducibility
# SIMILARITY_THRESHOLD = 0.5  # Threshold for cosine similarity to consider claims as similar

# Set the random seed
random.seed(RANDOM_SEED)

def load_pickle(file_path):
    """Load a pickle file and return its content."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f'Loaded data from "{file_path}".')
        return data
    except FileNotFoundError:
        print(f'Error: File "{file_path}" not found.')
        exit(1)
    except pickle.UnpicklingError:
        print(f'Error: Failed to unpickle file "{file_path}".')
        exit(1)

def load_entities(file_path, num_entities):
    """Load a specified number of entities from a text file."""
    try:
        with open(file_path, 'r') as f:
            entities = f.read().splitlines()[:num_entities]
        print(f'Loaded {len(entities)} entities from "{file_path}".')
        return entities
    except FileNotFoundError:
        print(f'Error: File "{file_path}" not found.')
        exit(1)

def is_valid_claim(claim: str) -> bool:
    """
    Check if the claim does not contain any flag words.
    
    Args:
        claim (str): The claim text to check.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    flag_words = ['speaker', 'appear', 'incomplete', 'provide', 'want', 'bio', 'text']
    claim_lower = claim.lower()
    return not any(word in claim_lower for word in flag_words)

# Removed compute_similarity_matrix and select_diverse_claims functions

def process_claims(entities, scores, major_flag, max_claims):
    """
    Process claims for each entity and compile the data, ensuring minimal content overlap.
    
    Args:
        entities (list): List of entity names.
        scores (dict): FactScore results containing decisions.
        major_flag (str): Replacement label for unsupported claims.
        max_claims (int): Maximum number of claims per entity.
    
    Returns:
        list: Processed data for each entity.
        int: Total number of claims processed.
    """
    data = []
    total_claims = 0

    # Ensure 'decisions' key exists in scores
    decisions = scores.get('decisions', [])
    if not decisions:
        print('Error: "decisions" key not found in FactScore data.')
        exit(1)
    
    for i, (entity, claims) in enumerate(zip(entities, decisions[:len(entities)])):
        if len(claims) == 0:
            continue
        datum = []
        datum.extend([i, f'Tell me a bio of {entity}.', None, [None]])

        # Extract valid claims along with their labels
        valid_claims_with_labels = [(claim.get('atom', '').strip(), claim.get('is_supported', False)) 
                                    for claim in claims if is_valid_claim(claim.get('atom', '').strip()) and claim.get('atom', '').strip()]
        
        # Separate claims and labels
        claims_text = [c[0] for c in valid_claims_with_labels]
        claims_labels = [c[1] if c[1] else MAJOR_FLAG for c in valid_claims_with_labels]

        if not claims_text:
            # If no valid claims, append empty lists for atoms and labels
            data.append(datum + [[]] + [[]])
            continue

        # Append atoms and labels to datum
        datum.append(claims_text)
        datum.append(claims_labels)

        # Append the fully constructed datum to data
        data.append(datum)
        total_claims += len(claims_text)
    
    return data, total_claims

def write_output_file(data, output_file_path, major_flag):
    """
    Write the processed data to a Python file with MAJOR and data variables.
    
    Args:
        data (list): Processed data for each entity.
        output_file_path (str): Path to the output Python file.
        major_flag (str): The MAJOR flag string.
    """
    try:
        with open(output_file_path, 'w') as f:
            # Write the MAJOR constant
            f.write(f"MAJOR = '{major_flag}'\n\n")
            f.write(f"MINOR =  'Minor False'\n\n")
            
            # Write the data list
            f.write("data = [\n")
            for datum in data:
                f.write("    [\n")
                # Write index, qs, None, [None]
                f.write(f"        {datum[0]},\n")
                f.write(f"        '{datum[1]}',\n")
                f.write("        None,\n")
                f.write("        [None],\n")
                
                # Write the list of atoms (claims)
                f.write("        [\n")
                for claim in datum[4]:
                    # Escape single quotes in claims
                    escaped_claim = claim.replace("'", "\\'")
                    f.write(f"            '{escaped_claim}',\n")
                f.write("        ],\n")
                
                # Write the list of labels
                f.write("        [\n")
                for label in datum[5]:
                    if label == major_flag:
                        f.write("            MAJOR,\n")
                    else:
                        # Assuming labels are either True or False
                        f.write(f"            {label},\n")
                f.write("        ]\n")
                
                f.write("    ],\n")
            f.write("]\n")
        print(f'Data successfully saved to "{output_file_path}".')
    except Exception as e:
        print(f'Error while writing to "{output_file_path}": {e}')
        exit(1)

def main():
    # Load FactScore data
    scores = load_pickle(FACTSCORE_FILE_NAME)
    
    # Load entities
    entities = load_entities(ENTITIES_FILE, NUM_ENTITIES)
    
    # Process claims with random selection and minimal content overlap removed
    data, total_claims = process_claims(entities, scores, MAJOR_FLAG, MAX_CLAIMS_PER_ENTITY)
    
    # Output results
    print(f'\nTotal claims processed: {total_claims}\n')
    
    # Write to output Python file
    write_output_file(data, OUTPUT_PYTHON_FILE, MAJOR_FLAG)
    
    # Optional: Print a summary using pprint
    print('Processed Data:')
    # pprint(data, width=120, compact=True)

if __name__ == '__main__':
    main()
