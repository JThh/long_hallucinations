import json
import pickle
import argparse
import os
import logging
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from factscore.factscorer import FactScorer  # Ensure this library is correctly installed


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Process a JSONL file to extract human-labeled facts (excluding 'IR' labels) "
            "and their support labels, then save them as a pickle file."
        )
    )
    parser.add_argument(
        'json_path',
        type=str,
        help="Path to the input JSONL file."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=(
            "Path to the output pickle file. "
            "If not provided, it will be inferred from the JSON file name."
        )
    )
    parser.add_argument(
        '-p', '--processes',
        type=int,
        default=cpu_count(),
        help=f"Number of parallel processes to use (default: {cpu_count()})."
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='/scratch/ms23jh/.factscore_cache',
        help=(
            "Base directory for FactScorer cache. "
            "Each data chunk will have its own subdirectory based on its index."
        )
    )
    return parser.parse_args()


def infer_pkl_path(json_path):
    """
    Infer the pickle file path based on the JSON file path.

    Args:
        json_path (str): Path to the JSONL file.

    Returns:
        str: Inferred path for the pickle file.
    """
    base, _ = os.path.splitext(json_path)
    return f"{base}.pkl"


def load_jsonl(json_path):
    """
    Load a JSONL file and return a list of JSON objects.

    Args:
        json_path (str): Path to the JSONL file.

    Returns:
        list: List of JSON objects.
    """
    sc = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading JSONL", unit=" lines"):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    json_obj = json.loads(line)
                    sc.append(json_obj)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error: {e} - Skipping line.")
    except FileNotFoundError:
        logging.error(f"File not found: {json_path}")
        exit(1)
    except IOError as e:
        logging.error(f"I/O error({e.errno}): {e.strerror}")
        exit(1)
    logging.info(f"Loaded {len(sc)} JSON objects from {json_path}")
    return sc


def split_into_chunks(lst, n):
    """
    Split list 'lst' into 'n' roughly equal chunks.

    Args:
        lst (list): The list to split.
        n (int): Number of chunks.

    Returns:
        list: A list containing 'n' sublists.
    """
    if n <= 0:
        n = 1
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def process_chunk(args):
    """
    Worker function to process a chunk of data.

    Args:
        args (tuple): Contains (sc_chunk, cache_dir_chunk)

    Returns:
        tuple: (pooled_atoms, label_counter)
    """
    sc_chunk, cache_dir_chunk = args
    # Initialize FactScorer with its own cache directory
    fact_scorer = FactScorer(cache_dir=cache_dir_chunk)
    pooled_atoms = []
    label_counter = Counter()

    for example in sc_chunk:
        annotations = example.get('annotations', [])
        if not isinstance(annotations, list):
            logging.warning("Expected 'annotations' to be a list. Skipping this example.")
            continue
        for annotation in annotations:
            atoms = annotation.get('human-atomic-facts', [])
            if not isinstance(atoms, list):
                logging.warning("Expected 'human-atomic-facts' to be a list. Skipping these atoms.")
                continue
            for atom in atoms:
                if not isinstance(atom, dict):
                    logging.warning("Expected each atom to be a dict. Skipping this atom.")
                    continue
                label = atom.get('label', 'Unknown')
                if label == 'IR':
                    continue  # Exclude 'IR' labeled atoms
                atom_text = atom.get('text', '').strip()
                if not atom_text:
                    logging.warning("Empty 'text' field encountered. Skipping this atom.")
                    continue
                support_label = label == 'S'

                response = example.get('output', '').strip()
                prompt = example.get('input', '').strip()
                revised_atom = fact_scorer.revise_fact(response, atom_text)
                # Uncomment the following lines if relevance checking is needed
                # is_relevant = fact_scorer.check_relevance(prompt, response, revised_atom, cost_estimate=None)
                # if not is_relevant:
                #     continue  # Skip irrelevant facts

                processed_atom = [revised_atom, support_label]
                pooled_atoms.append(processed_atom)
                label_counter[label] += 1

    return (pooled_atoms, label_counter)


def save_pooled_atoms(pooled_atoms, pkl_path):
    """
    Save the pooled atoms to a pickle file.

    Args:
        pooled_atoms (list): List of pooled atoms.
        pkl_path (str): Path to the pickle file.
    """
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(pooled_atoms, f)
        logging.info(f"Pooled atoms saved to {pkl_path}")
    except IOError as e:
        logging.error(f"Failed to save pickle file: {e}")
        exit(1)


def print_label_statistics(label_counter, total_atoms):
    """
    Print statistics of support labels.

    Args:
        label_counter (Counter): Counter of labels.
        total_atoms (int): Total number of atoms.
    """
    logging.info("Support Label Statistics:")
    print(f"{'Label':<10} {'Count':<10} {'Percentage':<10}")
    print("-" * 30)
    for label, count in label_counter.items():
        percentage = (count / total_atoms) * 100 if total_atoms > 0 else 0
        print(f"{label:<10} {count:<10} {percentage:.2f}%")
    print("-" * 30)
    logging.info("Finished printing label statistics.")


def main():
    setup_logging()
    args = parse_arguments()

    json_path = args.json_path
    pkl_path = args.output if args.output else infer_pkl_path(json_path)
    num_processes = args.processes
    base_cache_dir = args.cache_dir

    logging.info(f"Input JSONL file: {json_path}")
    logging.info(f"Output pickle file: {pkl_path}")
    logging.info(f"Number of processes: {num_processes}")
    logging.info(f"Base cache directory: {base_cache_dir}")

    sc = load_jsonl(json_path)

    if not sc:
        logging.error("No data loaded from the JSONL file. Exiting.")
        exit(1)

    # Split the data into chunks for parallel processing
    chunks = split_into_chunks(sc, num_processes)
    logging.info(f"Data split into {len(chunks)} chunks for processing.")

    # Assign deterministic cache directories based on chunk index
    process_args = []
    for idx, chunk in enumerate(chunks):
        cache_dir_chunk = os.path.join(base_cache_dir, f"chunk_{idx}")
        os.makedirs(cache_dir_chunk, exist_ok=True)
        process_args.append((chunk, cache_dir_chunk))

    # Initialize multiprocessing Pool
    with Pool(processes=num_processes) as pool:
        # Use tqdm to display a progress bar for the parallel processing
        results = list(tqdm(pool.imap(process_chunk, process_args), total=len(process_args), desc="Processing chunks"))

    # Aggregate results from all processes
    pooled_atoms = []
    label_counter = Counter()

    for local_pooled_atoms, local_label_counter in results:
        pooled_atoms.extend(local_pooled_atoms)
        label_counter.update(local_label_counter)

    total_atoms = len(pooled_atoms)
    print(f"Total pooled atoms (excluding 'IR' labels): {total_atoms}")
    print_label_statistics(label_counter, total_atoms)

    save_pooled_atoms(pooled_atoms, pkl_path)


if __name__ == "__main__":
    main()
