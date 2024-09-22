import pickle
import random
from pprint import pprint
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
# FACTSCORE_FILE_NAME = '../FActScore/data/labeled/Llama3.1-8B_fact_scores_nent_183_temp_0.1_maxtok_128_api_gpt-4o-mini.pkl'
FACTSCORE_FILE_NAME = './FActScore/data/test_set.pkl'
ENTITIES_FILE = '../FActScore/data/labeled/prompt_entities.txt'
NUM_ENTITIES = 200  # Use only 21 for consistency with the Nature paper
# NUM_ENTITIES = 21  # Use only 21 for consistency with the Nature paper
MAJOR_FLAG = 'Major False'  # Flag as False by Longhallu in the SE paper
MAX_CLAIMS_PER_ENTITY = 6  # Maximum number of claims per entity
OUTPUT_PYTHON_FILE = 'wiki_data.py'  # Output Python file to save the data
RANDOM_SEED = 42  # Set a random seed for reproducibility
SIMILARITY_THRESHOLD = 0.5  # Threshold for cosine similarity to consider claims as similar

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

def compute_similarity_matrix(claims):
    """
    Compute the cosine similarity matrix for a list of claims.
    
    Args:
        claims (list): List of claim strings.
    
    Returns:
        ndarray: Cosine similarity matrix.
    """
    vectorizer = TfidfVectorizer().fit_transform(claims)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

def select_diverse_claims(claims, similarity_matrix, max_claims, threshold):
    """
    Select a subset of claims ensuring minimal content overlap.
    
    Args:
        claims (list): List of claim strings.
        similarity_matrix (ndarray): Precomputed cosine similarity matrix.
        max_claims (int): Maximum number of claims to select.
        threshold (float): Similarity threshold to consider claims as overlapping.
    
    Returns:
        list: Selected claims.
    """
    selected_claims = []
    selected_indices = set()
    
    # Order claims randomly to ensure diversity in selection
    indices = list(range(len(claims)))
    random.shuffle(indices)
    
    for idx in indices:
        if len(selected_claims) >= max_claims:
            break
        # Check similarity with already selected claims
        similar = False
        for sel_idx in selected_indices:
            if similarity_matrix[idx][sel_idx] >= threshold:
                similar = True
                break
        if not similar:
            selected_claims.append(claims[idx])
            selected_indices.add(idx)
    
    return selected_claims

def process_claims(entities, scores, major_flag, max_claims, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    Process claims for each entity and compile the data, ensuring minimal content overlap.
    
    Args:
        entities (list): List of entity names.
        scores (dict): FactScore results containing decisions.
        major_flag (str): Replacement label for unsupported claims.
        max_claims (int): Maximum number of claims per entity.
        similarity_threshold (float, optional): Cosine similarity threshold to consider claims as overlapping. Defaults to 0.5.
    
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
        datum = []
        datum.extend([i, f'Tell me a bio of {entity}.', None, [None]])

        atoms = []
        labels = []

        # Extract valid claims along with their labels
        valid_claims_with_labels = [(claim.get('atom', '').strip(), claim.get('is_supported', False)) 
                                    for claim in claims if is_valid_claim(claim.get('atom', '').strip()) and claim.get('atom', '').strip()]
        
        # Separate claims and labels
        claims_text = [c[0] for c in valid_claims_with_labels]
        claims_labels = [c[1] for c in valid_claims_with_labels]

        if not claims_text:
            # If no valid claims, append empty lists for atoms and labels
            data.append(datum + [[]] + [[]])
            continue

        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(claims_text)

        # Select diverse claims based on similarity threshold
        selected_claims = select_diverse_claims(claims_text, similarity_matrix, max_claims, similarity_threshold)
        
        # If selected_claims less than max_claims, pad with random claims (optional)
        if len(selected_claims) < max_claims and len(claims_text) > len(selected_claims):
            remaining_claims = list(set(claims_text) - set(selected_claims))
            num_to_add = min(max_claims - len(selected_claims), len(remaining_claims))
            if num_to_add > 0:
                selected_claims += random.sample(remaining_claims, num_to_add)
        
        # Assign labels to selected claims
        selected_labels = []
        for claim in selected_claims:
            # Find the label corresponding to this claim
            for c_text, c_label in valid_claims_with_labels:
                if c_text == claim:
                    selected_labels.append(c_label if c_label else major_flag)
                    break
        
        # Append atoms and labels to datum
        datum.append(selected_claims)
        datum.append(selected_labels)

        # Append the fully constructed datum to data
        data.append(datum)
        total_claims += len(selected_claims)
    
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
                        f.write(f"            {label},\n")  # True or False
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
    
    # Process claims with sampling and minimal content overlap
    data, total_claims = process_claims(entities, scores, MAJOR_FLAG, MAX_CLAIMS_PER_ENTITY, SIMILARITY_THRESHOLD)
    
    # Output results
    print(f'\nTotal claims processed: {total_claims}\n')
    
    # Write to output Python file
    write_output_file(data, OUTPUT_PYTHON_FILE, MAJOR_FLAG)
    
    # Optional: Print a summary using pprint
    print('Processed Data:')
    # pprint(data, width=120, compact=True)

if __name__ == '__main__':
    main()
