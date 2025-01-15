import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Data Preparation ===

# Data for Llama3.1-8B
llama_data = {
    'Method': ['FP (Layer 13-17; C=0.5)', 'Baseline p(True)', 'Baseline SelfCheckGPT', 'Baseline SE'],
    'AUROC Mean': [0.7642, 0.6979, 0.6025, 0.5341],
    'AUROC Std': [0.024, 0.024, 0.021, 0.032]
}

# Data for Gemma2-9B
gemma_data = {
    'Method': ['FP (Layer 25-27; C=0.1)', 'Baseline p(True)', 'Baseline SelfCheckGPT', 'Baseline SE'],
    'AUROC Mean': [0.7142, 0.6671, 0.5415, 0.6365],
    'AUROC Std': [0.028, 0.022, 0.012, 0.024]
}

# Convert dictionaries to pandas DataFrames
llama_df = pd.DataFrame(llama_data)
gemma_df = pd.DataFrame(gemma_data)

# === Plotting ===

# Set the style for the plots
plt.style.use('ggplot')  # You can choose other styles like 'seaborn-darkgrid', 'fivethirtyeight', etc.

# Create a figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# Define the width of the bars
bar_width = 0.6

# Colors for the bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

# === Plot for Llama3.1-8B ===
axes[0].bar(
    llama_df['Method'],
    llama_df['AUROC Mean'],
    yerr=llama_df['AUROC Std'],
    capsize=5,
    color=colors,
    width=bar_width,
    edgecolor='black'
)

# Customize the first subplot
axes[0].set_title('Llama3.1-8B Model Performance', fontsize=16, weight='bold')
axes[0].set_xlabel('Method', fontsize=14)
axes[0].set_ylabel('AUROC Mean', fontsize=14)
axes[0].set_ylim(0.4, 0.85)  # Adjusted based on provided data
axes[0].set_xticklabels(llama_df['Method'], rotation=15, ha='right', fontsize=12)
axes[0].yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Add annotations on top of the bars
for idx, row in llama_df.iterrows():
    axes[0].text(
        idx, 
        row['AUROC Mean'] + row['AUROC Std'] + 0.005, 
        f"{row['AUROC Mean']:.4f} ± {row['AUROC Std']:.3f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# === Plot for Gemma2-9B ===
axes[1].bar(
    gemma_df['Method'],
    gemma_df['AUROC Mean'],
    yerr=gemma_df['AUROC Std'],
    capsize=5,
    color=colors,
    width=bar_width,
    edgecolor='black'
)

# Customize the second subplot
axes[1].set_title('Gemma2-9B Model Performance', fontsize=16, weight='bold')
axes[1].set_xlabel('Method', fontsize=14)
axes[1].set_ylim(0.4, 0.85)  # Same y-axis limits for consistency
axes[1].set_xticklabels(gemma_df['Method'], rotation=15, ha='right', fontsize=12)
axes[1].yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Add annotations on top of the bars
for idx, row in gemma_df.iterrows():
    axes[1].text(
        idx, 
        row['AUROC Mean'] + row['AUROC Std'] + 0.005, 
        f"{row['AUROC Mean']:.4f} ± {row['AUROC Std']:.3f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# === Final Adjustments ===

# Adjust layout to prevent overlap
plt.tight_layout()

# Optionally, save the plot as a high-resolution image
plt.savefig('./figures/model_performance_comparison.png', dpi=300)

# Display the plot
plt.show()
