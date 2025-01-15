import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np

def plot_auroc(ax, layer_indices, auroc_full, auroc_sparse, model_name, font_size):
    """
    Plot AUROC scores comparing full-neuron probes and sparse (top dimensions) probes.

    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on.
        layer_indices (list): List of layer indices.
        auroc_full (list): List of AUROC scores for full-neuron probes.
        auroc_sparse (list): List of AUROC scores for sparse probes.
        model_name (str): Identifier for the model.
        font_size (int): Font size for plot elements.
    """
    ax.plot(layer_indices, auroc_full, label='Full-Neuron Probe', marker='o', linewidth=2)
    ax.plot(layer_indices, auroc_sparse, label='Sparse Probe', marker='x', linewidth=2)
    ax.set_title(model_name, fontsize=font_size + 2, fontweight='bold')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

def load_data(auroc_file, activation_file):
    """
    Load the CSV files for AUROC and Activation statistics.

    Args:
        auroc_file (str): Path to the AUROC CSV file.
        activation_file (str): Path to the Activation CSV file.

    Returns:
        DataFrame, DataFrame: DataFrames for AUROC and Activation data.
    """
    try:
        auroc_data = pd.read_csv(auroc_file)
    except FileNotFoundError:
        print(f"Error: AUROC file '{auroc_file}' not found.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: AUROC file '{auroc_file}' is empty.")
        return None, None

    try:
        activation_data = pd.read_csv(activation_file)
    except FileNotFoundError:
        print(f"Error: Activation file '{activation_file}' not found.")
        return auroc_data, None
    except pd.errors.EmptyDataError:
        print(f"Error: Activation file '{activation_file}' is empty.")
        return auroc_data, None

    return auroc_data, activation_data

def find_activation_file(results_dir, model_name):
    """
    Find the activation CSV file for a given model, considering files with dimension info.

    Args:
        results_dir (str): Directory containing the CSV files.
        model_name (str): Name of the model.

    Returns:
        str: The path to the activation CSV file.
    """
    # Pattern: "model_name_top{n}_activations.csv"
    pattern = rf'^{re.escape(model_name)}_top\d+_activations\.csv$'
    for filename in os.listdir(results_dir):
        if re.match(pattern, filename):
            return os.path.join(results_dir, filename)
    raise FileNotFoundError(f"Activation CSV for '{model_name}' not found in '{results_dir}'.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot AUROC statistics for multiple models.'
    )
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing the CSV files.')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='List of model names to plot.')
    parser.add_argument('--font_size', type=int, default=14,
                        help='Base font size for plots.')
    args = parser.parse_args()

    results_dir = args.results_dir
    model_names = args.models
    font_size = args.font_size

    # Check if results_dir exists
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return

    # Initialize AUROC figure
    num_models = len(model_names)
    fig_auroc, axes_auroc = plt.subplots(nrows=1, ncols=num_models, figsize=(8 * num_models, 6), squeeze=False)
    
    # If only one model, make axes iterable
    if num_models == 1:
        axes_auroc = [axes_auroc[0]]

    # Collect handles and labels for legend
    legend_elements = []

    for idx, model_name in enumerate(model_names):
        print(f"\nProcessing model: {model_name}")

        # Define file paths
        auroc_file = os.path.join(results_dir, f'{model_name}_AUROC_Data.csv')
        try:
            activation_file = find_activation_file(results_dir, model_name)
        except FileNotFoundError as e:
            print(e)
            activation_file = None

        # Load data
        auroc_data, activation_data = load_data(auroc_file, activation_file)
        if auroc_data is None or activation_data is None:
            print(f"Skipping model '{model_name}' due to missing data.")
            continue

        # Ensure 'Layer' is integer and sorted
        auroc_data['Layer'] = pd.to_numeric(auroc_data['Layer'], errors='coerce')
        auroc_data = auroc_data.dropna(subset=['Layer'])
        auroc_data['Layer'] = auroc_data['Layer'].astype(int)
        auroc_data = auroc_data.sort_values('Layer')

        activation_data['Layer'] = pd.to_numeric(activation_data['Layer'], errors='coerce')
        activation_data = activation_data.dropna(subset=['Layer'])
        activation_data['Layer'] = activation_data['Layer'].astype(int)
        activation_data = activation_data.sort_values('Layer')

        # Extract layer indices
        layer_indices = sorted(auroc_data['Layer'].unique())

        # Extract AUROC scores
        auroc_full = auroc_data.sort_values('Layer')['Full Probe AUROC'].tolist()
        auroc_sparse = auroc_data.sort_values('Layer')['Sparse Probe AUROC'].tolist()

        # Plot AUROC
        ax = axes_auroc[0][idx] if num_models > 1 else axes_auroc[0]
        plot_auroc(ax, layer_indices, auroc_full, auroc_sparse, model_name, font_size)

    # Shared Y-label
    fig_auroc.text(0.04, 0.5, 'AUROC Score', va='center', rotation='vertical', fontsize=font_size + 2)

    # Configure subplot titles to only display model names (already done in plot_auroc)

    # Adjust x-ticks: show start and end layers with smart intervals
    for ax in axes_auroc.flatten():
        if ax.lines:
            layers = [int(line.get_xdata()[0]) for line in ax.lines]  # Assuming same layer indices
            if layers:
                min_layer = min(layers)
                max_layer = max(layers)
                ax.set_xlim(min_layer, max_layer)
                ax.set_xticks([min_layer, max_layer])
        ax.tick_params(axis='x', labelsize=font_size)

    # Create a single legend for all subplots
    if num_models > 0:
        handles, labels = axes_auroc[0][0].get_legend_handles_labels() if num_models > 1 else axes_auroc[0].get_legend_handles_labels()
        # To ensure only two legends are present
        unique = dict(zip(labels, handles))
        fig_auroc.legend(unique.values(), unique.keys(), loc='upper center', ncol=2, fontsize=font_size, bbox_to_anchor=(0.5, 1.05))

    # Adjust layout to make space for the legend
    fig_auroc.tight_layout(rect=[0, 0, 1, 0.9])

    # Save AUROC plot as PDF
    auroc_pdf_path = os.path.join(results_dir, 'AUROC_Plots.pdf')
    fig_auroc.savefig(auroc_pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f'\nAUROC plots saved to {auroc_pdf_path}')
    plt.close(fig_auroc)

    # Note: Activation plotting sections have been removed as per the user's request.
    # Activation CSVs are still being saved separately in the original script's main function.

if __name__ == '__main__':
    main()

# python sp_probe_plots.py --results_dir ./results --models Llama3.2-3B Llama3.1-8B Llama3.1-70B --font_size 16