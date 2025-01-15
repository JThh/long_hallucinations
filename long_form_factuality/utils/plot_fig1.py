import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# === Configuration ===

# Define the paths to your layer-group-5 CSV files
# Replace 'metrics/' with the actual directory where your CSV files are located
FGT_GROUP5_CSV_PATH = 'metrics/Llama3.1-8B_fgt_layer_group5_aurocs.csv'
SLT_GROUP5_CSV_PATH = 'metrics/Llama3.1-8B_slt_layer_group5_aurocs.csv'
LT_GROUP5_CSV_PATH = 'metrics/Llama3.1-8B_layer_group5_aurocs.csv'

# Verify that the layer-group-5 CSV files exist
layer_group5_files = [FGT_GROUP5_CSV_PATH, SLT_GROUP5_CSV_PATH, LT_GROUP5_CSV_PATH]
for file_path in layer_group5_files:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

# Choose a Matplotlib style
# You can list available styles by uncommenting the following lines:
# import matplotlib.pyplot as plt
# print(plt.style.available)
# For this script, we'll use 'ggplot' which is widely supported
PLOT_STYLE = 'ggplot'  # Change this to any available style as needed

# === Functions ===

def load_csv_data(file_path):
    """
    Loads CSV data from the given file path into a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def extract_layer_group(df):
    """
    Extracts the numerical layer group from the 'Layer Group' column and adds it as a new 'Layer' column.
    """
    df['Layer'] = df['Layer Group'].apply(lambda x: int(x.split('-')[0]))
    return df

def prepare_data(file_path):
    """
    Loads and processes the CSV data from the given file path.
    """
    df = load_csv_data(file_path)
    df = extract_layer_group(df)
    df = df.sort_values('Layer')  # Ensure data is sorted by layer
    return df

# === Load and Prepare Data ===

# Load data for layer groups of size 5
fgt_group5_df = prepare_data(FGT_GROUP5_CSV_PATH)
slt_group5_df = prepare_data(SLT_GROUP5_CSV_PATH)
lt_group5_df = prepare_data(LT_GROUP5_CSV_PATH)

# === Plotting ===

# Set the chosen plot style
plt.style.use(PLOT_STYLE)

# Create a figure and axis
plt.figure(figsize=(14, 8))

# Define a list of datasets for easier iteration
datasets_group5 = [
    (fgt_group5_df, 'First Token (fgt)', '#1f77b4', '-o'),        # Blue with circle markers
    (slt_group5_df, 'Second-Last Token (slt)', '#2ca02c', '-s'), # Green with square markers
    (lt_group5_df, 'Last Token (lt)', '#d62728', '-^')            # Red with triangle markers
]

# Plot each dataset with error bars
for df, label, color, fmt in datasets_group5:
    plt.errorbar(
        df['Layer'],
        df['Validation AUROC Mean'],
        yerr=df['Validation AUROC Std'],
        label=label,
        fmt=fmt,
        capsize=5,
        markersize=8,
        linewidth=2,
        color=color
    )

# Customize the axes
plt.xlabel('Layer Group', fontsize=14)
plt.ylabel('Validation AUROC Mean', fontsize=14)
plt.title('Layer Group-5 Performance for Llama3.1-8B Model', fontsize=16, weight='bold')

# Set x-axis to show integer ticks only
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Optionally, set y-axis limits dynamically based on data
y_min = min(
    (df['Validation AUROC Mean'] - df['Validation AUROC Std']).min()
    for df in [fgt_group5_df, slt_group5_df, lt_group5_df]
)
y_max = max(
    (df['Validation AUROC Mean'] + df['Validation AUROC Std']).max()
    for df in [fgt_group5_df, slt_group5_df, lt_group5_df]
)
plt.ylim(y_min - 0.05, y_max + 0.05)  # Add some padding

# Add a legend with a larger font
plt.legend(fontsize=12)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Add annotations on top of the data points
for df, label, color, fmt in datasets_group5:
    for idx, row in df.iterrows():
        plt.text(
            row['Layer'], 
            row['Validation AUROC Mean'] + row['Validation AUROC Std'] + 0.005, 
            f"{row['Validation AUROC Mean']:.4f} ± {row['Validation AUROC Std']:.3f}",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='black'
        )

# Enhance layout
plt.tight_layout()

# === Saving the Plot ===

# Ensure that the 'figures' directory exists
os.makedirs('./figures/', exist_ok=True)

# Save the plot as a high-resolution image
plt.savefig('./figures/layer_group5_performance.png', dpi=300)

# Show the plot
plt.show()
