import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
CSV_FILE_PATH = 'drone_eval_results/evaluation_metrics.csv'
# The CONDITIONS_TO_PLOT dictionary is now used only for renaming.
# All unique conditions from the CSV will be plotted by default.
PLOT_LABEL_MAPPING = {
    'model_reconstruction': 'flow_mo'
    # Add other specific renaming if needed, e.g.:
    # 'old_condition_name_in_csv': 'New Plot Label'
}
# If you want to plot specific losses, list their original column names here.
# Otherwise, all metrics ('lpips_alex', 'lpips_vgg', 'psnr', 'ssim') will be attempted.
# SPECIFIC_LOSSES_TO_PLOT = ['lpips_alex', 'psnr'] # Example
SPECIFIC_LOSSES_TO_PLOT = None # Plot all available metrics

# --- 1. Load and Prepare Data ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    print("Please ensure the CSV file exists and the path is correct.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

if df.empty:
    print(f"The CSV file {CSV_FILE_PATH} is empty. No plots will be generated.")
    exit()

# Identify metric columns (those not 'image_name' or 'condition_name')
metric_columns = [col for col in df.columns if col not in ['image_name', 'condition_name']]
if not metric_columns:
    print("Error: No metric columns found in the CSV (e.g., lpips_alex, psnr, etc.).")
    exit()

print(f"Identified metric columns: {metric_columns}")

# Replace 'inf' with NaN for proper aggregation
for col in metric_columns:
    if df[col].dtype == 'object': # Check if column might contain 'inf' as string
        df[col] = df[col].replace('inf', np.nan).replace('-inf', -np.nan)
    df[col] = pd.to_numeric(df[col], errors='coerce') # Convert to numeric, coercing errors to NaN
df = df.replace([np.inf, -np.inf], np.nan) # Ensure any remaining infs are NaN

# Aggregate data: Calculate mean of metrics for each condition_name
aggregated_df = df.groupby('condition_name')[metric_columns].mean().reset_index()

if aggregated_df.empty:
    print("Error: Aggregated DataFrame is empty. Check if 'condition_name' exists or if data is valid.")
    exit()

# Melt the aggregated DataFrame to long format
melted_df = aggregated_df.melt(
    id_vars=['condition_name'],
    value_vars=metric_columns,
    var_name='loss_name',
    value_name='loss_value'
)

if melted_df.empty:
    print("Error: Melted DataFrame is empty. This should not happen if aggregation was successful.")
    exit()

# Create 'condition_name_plot' for plot labels, applying mapping
melted_df['condition_name_plot'] = melted_df['condition_name'].replace(PLOT_LABEL_MAPPING)

# Filter for specific losses if requested
if SPECIFIC_LOSSES_TO_PLOT:
    melted_df_filtered = melted_df[melted_df['loss_name'].isin(SPECIFIC_LOSSES_TO_PLOT)].copy()
    if melted_df_filtered.empty:
        print(f"Error: No data found for the specified losses in SPECIFIC_LOSSES_TO_PLOT: {SPECIFIC_LOSSES_TO_PLOT}")
        print(f"Available loss names after melting: {melted_df['loss_name'].unique().tolist()}")
        exit()
else:
    melted_df_filtered = melted_df.copy()


# Pivot the table for plotting
try:
    pivot_df = melted_df_filtered.pivot_table(
        index='condition_name_plot',
        columns='loss_name',
        values='loss_value'
    )
except Exception as e:
    print(f"Error pivoting data: {e}")
    print("Data before pivoting (first 5 rows of melted_df_filtered):")
    print(melted_df_filtered.head())
    exit()

if pivot_df.empty:
    print("Error: Pivot table is empty. Check the data and filtering steps.")
    print("Melted data (first 5 rows):")
    print(melted_df_filtered.head())
    exit()
    
# --- 2. Plotting ---
output_dir = "drone_eval_plots_separated"
os.makedirs(output_dir, exist_ok=True)

# Ensure the order of conditions in the plot is somewhat predictable or customizable if needed.
# For now, it will use the order from pivot_df.index, which is usually sorted alphabetically.
# If a specific order is desired, `pivot_df.reindex()` can be used with a predefined list of 'condition_name_plot' labels.

n_conditions = len(pivot_df.index)

if n_conditions == 0:
    print("Error: Not enough condition data to plot after processing.")
    exit()

if not pivot_df.columns.tolist():
    print("Error: No loss metrics found in pivot_df columns to plot.")
    exit()

for loss_name in pivot_df.columns:
    fig, ax = plt.subplots(figsize=(max(8, n_conditions * 0.7), 7)) # Adjust figure size per plot
    
    values = pivot_df[loss_name]
    bars = ax.bar(pivot_df.index, values, color=plt.cm.get_cmap('viridis')(np.linspace(0, 1, n_conditions)))
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval): # Only add text if yval is not NaN
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (0.01 * plt.ylim()[1]), f'{yval:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)

    ax.set_xlabel('Condition', fontweight='bold')
    ax.set_ylabel(f'Aggregated {loss_name} Value', fontweight='bold')
    ax.set_title(f'{loss_name} Comparison Across Conditions', fontweight='bold')
    ax.set_xticks(np.arange(n_conditions)) # Ensure correct tick positions
    ax.set_xticklabels(pivot_df.index, rotation=30, ha="right")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    plot_filename = os.path.join(output_dir, f"{loss_name.replace(' ', '_')}_comparison.png")
    try:
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.close(fig) # Close the figure to free memory

print(f"\\nAll plots saved in '{output_dir}' directory.")

print("\\n--- Data used for plotting (after pivoting) ---")
print(pivot_df)
print("\nIf the plot looks crowded or scales are too different, consider:")
print("1. Plotting fewer metrics at once (using SPECIFIC_LOSSES_TO_PLOT).")
print("2. Creating separate plots for metrics with vastly different scales (e.g., LPIPS vs PSNR).")
print("3. Using subplots within a single figure.")