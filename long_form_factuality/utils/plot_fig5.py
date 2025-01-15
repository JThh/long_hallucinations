import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Configuration ===

# Define the path to save the plot
OUTPUT_DIR = './figures/'
OUTPUT_FILE = 'fp_test_auroc_comparison.png'

# Ensure that the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose a Matplotlib style for a polished look
plt.style.use('ggplot')  # Alternatives: 'seaborn-darkgrid', 'fivethirtyeight', etc.

# === Data Preparation ===

# Define the data as a list of dictionaries
data = [
    # Llama3.1-8B
    {
        'Model': 'Llama3.1-8B',
        'Testing Platform': 'InstructGPT',
        'Test AUROC Mean': 0.7319939117199391,
        'Test AUROC Std': 0.007813553971545468
    },
    {
        'Model': 'Llama3.1-8B',
        'Testing Platform': 'PerplexityAI',
        'Test AUROC Mean': 0.6284804688015411,
        'Test AUROC Std': 0.010087424316542859
    },
    {
        'Model': 'Llama3.1-8B',
        'Testing Platform': 'ChatGPT',
        'Test AUROC Mean': 0.6978524116583844,
        'Test AUROC Std': 0.007717033923838443
    },
    
    # Llama3.2-3B
    {
        'Model': 'Llama3.2-3B',
        'Testing Platform': 'InstructGPT',
        'Test AUROC Mean': 0.6920895846923244,
        'Test AUROC Std': 0.008271116065436396
    },
    {
        'Model': 'Llama3.2-3B',
        'Testing Platform': 'ChatGPT',
        'Test AUROC Mean': 0.6891413014354634,
        'Test AUROC Std': 0.0076856269648550475
    },
    {
        'Model': 'Llama3.2-3B',
        'Testing Platform': 'PerplexityAI',
        'Test AUROC Mean': 0.6234346618022844,
        'Test AUROC Std': 0.010346994598638136
    },
    
    # Llama3.1-70B
    {
        'Model': 'Llama3.1-70B',
        'Testing Platform': 'ChatGPT',
        'Test AUROC Mean': 0.6504062174792867,
        'Test AUROC Std': 0.008043142156040102
    },
    {
        'Model': 'Llama3.1-70B',
        'Testing Platform': 'InstructGPT',
        'Test AUROC Mean': 0.701003116619555,
        'Test AUROC Std': 0.008221976239376082
    },
    {
        'Model': 'Llama3.1-70B',
        'Testing Platform': 'PerplexityAI',
        'Test AUROC Mean': 0.6052800923177066,
        'Test AUROC Std': 0.01079510970066097
    },
    
    # Gemma2-9B
    {
        'Model': 'Gemma2-9B',
        'Testing Platform': 'InstructGPT',
        'Test AUROC Mean': 0.6121526418786692,
        'Test AUROC Std': 0.008744611616445219
    },
    {
        'Model': 'Gemma2-9B',
        'Testing Platform': 'ChatGPT',
        'Test AUROC Mean': 0.6035691737314793,
        'Test AUROC Std': 0.008337727484744529
    },
    {
        'Model': 'Gemma2-9B',
        'Testing Platform': 'PerplexityAI',
        'Test AUROC Mean': 0.5514302042512765,
        'Test AUROC Std': 0.011303813462297791
    }
]

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)

# === Plotting ===

# Create a pivot table to facilitate plotting
pivot_df = df.pivot(index='Testing Platform', columns='Model', values=['Test AUROC Mean', 'Test AUROC Std'])

# Define models and testing platforms in desired order
models = ['Llama3.2-3B', 'Llama3.1-8B', 'Gemma2-9B', 'Llama3.1-70B']
testing_platforms = ['ChatGPT', 'InstructGPT', 'PerplexityAI']

# Extract means and stds in the specified order
means = pivot_df['Test AUROC Mean'].loc[testing_platforms, models].values
stds = pivot_df['Test AUROC Std'].loc[testing_platforms, models].values

# Number of testing platforms and models
n_platforms = len(testing_platforms)
n_models = len(models)

# Set positions of the bars
bar_width = 0.2
x = np.arange(n_platforms)  # the label locations

# Adjust the positions for each model
offsets = np.linspace(-bar_width*(n_models-1)/2, bar_width*(n_models-1)/2, n_models)

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Define colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# Plot bars for each model
for i, model in enumerate(models):
    plt.bar(
        x + offsets[i],
        means[:, i],
        width=bar_width,
        yerr=stds[:, i],
        capsize=5,
        label=model,
        color=colors[i],
        edgecolor='black'
    )

# Customize the axes
plt.xlabel('Testing Models', fontsize=14)
plt.ylabel('Test AUROC Mean', fontsize=14)
plt.title('FP Test AUROC on Different Closed Models with Probes Trained on Open-Source LLMs', fontsize=16, weight='bold')

# Set x-axis ticks and labels
plt.xticks(x, testing_platforms, fontsize=12)
plt.yticks(fontsize=12)

# Set y-axis limits based on data
plt.ylim(0.5, 0.8)  # Adjust as necessary

# Add a legend with a larger font
plt.legend(title='Probed Model', fontsize=12, title_fontsize=12)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Enhance layout
plt.tight_layout()

# === Saving the Plot ===

# Save the plot as a high-resolution image
plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE), dpi=300)

# Show the plot
plt.show()
