import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# === Configuration ===

# Define the path to save the plot
OUTPUT_DIR = './figures/'
OUTPUT_FILE = 'model_scaling_performance.pdf'

# Ensure that the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose a Matplotlib style for a polished look
plt.style.use('ggplot')  # Alternatives: 'seaborn-darkgrid', 'fivethirtyeight', etc.

# === Data Preparation ===

# Define the data as a list of dictionaries
data = [
    # Llama3.1-70B
    {
        'Model': 'Llama3.1-70B',
        'Generations': 25,
        'Facts': 764,
        'Method': 'FP',
        'Test AUROC Mean': 0.6819,
        'Test AUROC Std': 0.0134
    },
    {
        'Model': 'Llama3.1-70B',
        'Generations': 25,
        'Facts': 764,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.7043,
        'Test AUROC Std': 0.0131
    },
    {
        'Model': 'Llama3.1-70B',
        'Generations': 50,
        'Facts': 1781,
        'Method': 'FP',
        'Test AUROC Mean': 0.6975,
        'Test AUROC Std': 0.0131
    },
    {
        'Model': 'Llama3.1-70B',
        'Generations': 50,
        'Facts': 1781,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.6488,
        'Test AUROC Std': 0.0136
    },
    {
        'Model': 'Llama3.1-70B',
        'Generations': 75,
        'Facts': 4227,
        'Method': 'FP',
        'Test AUROC Mean': 0.7463,
        'Test AUROC Std': 0.0121
    },
    {
        'Model': 'Llama3.1-70B',
        'Generations': 75,
        'Facts': 4227,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.7503,
        'Test AUROC Std': 0.012
    },
    
    # Llama3.1-8B
    {
        'Model': 'Llama3.1-8B',
        'Generations': 25,
        'Facts': 840,
        'Method': 'FP',
        'Test AUROC Mean': 0.6882,
        'Test AUROC Std': 0.0129,
    },
    {
        'Model': 'Llama3.1-8B',
        'Generations': 25,
        'Facts': 840,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.6862,
        'Test AUROC Std': 0.0130,
    },
    {
        'Model': 'Llama3.1-8B',
        'Generations': 25,
        'Facts': 1680,
        'Method': 'FP',
        'Test AUROC Mean': 0.7358,
        'Test AUROC Std': 0.0121
    },
    {
        'Model': 'Llama3.1-8B',
        'Generations': 25,
        'Facts': 1680,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.7330,
        'Test AUROC Std': 0.0122
    },
    {
        'Model': 'Llama3.1-8B',
        'Generations': 50,
        'Facts': 3374,
        'Method': 'FP',
        'Test AUROC Mean': 0.7353,
        'Test AUROC Std': 0.0119
    },
    {
        'Model': 'Llama3.1-8B',
        'Generations': 50,
        'Facts': 3374,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.7410,
        'Test AUROC Std': 0.0118
    },
    
    # Llama3.2-3B
    {
        'Model': 'Llama3.2-3B',
        'Generations': 25,
        'Facts': 505,
        'Method': 'FP',
        'Test AUROC Mean': 0.6853,
        'Test AUROC Std': 0.0126
    },
    {
        'Model': 'Llama3.2-3B',
        'Generations': 25,
        'Facts': 505,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.6672,
        'Test AUROC Std': 0.0127
    },
    {
        'Model': 'Llama3.2-3B',
        'Generations': 25,
        'Facts': 1025,
        'Method': 'FP',
        'Test AUROC Mean': 0.7192,
        'Test AUROC Std': 0.012
    },
    {
        'Model': 'Llama3.2-3B',
        'Generations': 25,
        'Facts': 1025,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.6981,
        'Test AUROC Std': 0.0121
    },
    {
        'Model': 'Llama3.2-3B',
        'Generations': 25,
        'Facts': 1531,
        'Method': 'FP',
        'Test AUROC Mean': 0.7236,
        'Test AUROC Std': 0.0118
    },
    {
        'Model': 'Llama3.2-3B',
        'Generations': 25,
        'Facts': 1531,
        'Method': 'XGBoost',
        'Test AUROC Mean': 0.7261,
        'Test AUROC Std': 0.0121
    }
]

# Convert the list of dictionaries to a pandas DataFrame
df = pd.DataFrame(data)

# === Plotting ===

# Create a figure and axis
plt.figure(figsize=(12, 8))

# Define color mapping for models
color_mapping = {
    'Llama3.1-70B': '#1f77b4',   # Blue
    'Llama3.1-8B': '#ff7f0e',    # Orange
    'Llama3.2-3B': '#2ca02c'     # Green
}

# Define marker mapping for methods
marker_mapping = {
    'FP': 'o',           # Circle
    'XGBoost': 's'       # Square
}

# Plot each model-method combination
for model in df['Model'].unique():
    for method in df[df['Model'] == model]['Method'].unique():
        subset = df[(df['Model'] == model) & (df['Method'] == method)]
        plt.errorbar(
            subset['Facts'],
            subset['Test AUROC Mean'],
            yerr=subset['Test AUROC Std'],
            label=f"{model} - {method}",
            marker=marker_mapping[method],
            linestyle='-',
            color=color_mapping[model],
            capsize=5,
            markersize=8,
            linewidth=2
        )

# Customize the axes
plt.xlabel('Number of Facts in Generations', fontsize=14)
plt.ylabel('Test AUROC Mean', fontsize=14)
plt.title('Test AUROC Performance Across Models, Methods, and Data Scales', fontsize=16, weight='bold')

# Set x-axis to display integer ticks
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Set y-axis limits based on data
plt.ylim(0.6, 0.8)  # Adjust as necessary based on data

# Add a legend with a larger font
plt.legend(title='Model - Method', fontsize=12, title_fontsize=12)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Enhance layout
plt.tight_layout()

# === Saving the Plot ===

# Save the plot as a high-resolution image
plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_FILE), dpi=300)

# Show the plot
plt.show()
