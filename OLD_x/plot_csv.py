import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np # 1. Ensure numpy is imported

def create_plots_from_csv(csv_filepath="drone_eval_results/evaluation_metrics.csv", 
                          output_dir="drone_eval_plots"):
    """
    Reads an evaluation metrics CSV and generates various plots.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if df.empty:
        print(f"The CSV file {csv_filepath} is empty. No plots will be generated.")
        return

    # Replace 'inf' strings with np.inf and convert to numeric if necessary
    for col in ["lpips_alex", "lpips_vgg", "psnr", "ssim"]:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('inf', np.inf).replace('-inf', -np.inf)
            df[col] = pd.to_numeric(df[col], errors='coerce')


    os.makedirs(output_dir, exist_ok=True)
    print(f"Successfully read {csv_filepath}. DataFrame head:")
    print(df.head())
    print("\nColumns available:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)

    # 2. Initialize hue_col to None at a higher scope
    hue_col = None 

    # --- Customizable Plotting Section ---
    
    # Example 1: Bar plot of a metric (e.g., 'AP') per model
    model_col = None
    ap_col = None

    # 3. Update possible column names
    possible_model_cols = ['condition_name', 'model', 'model_name', 'config_name']
    possible_ap_cols = ['lpips_alex', 'lpips_vgg', 'psnr', 'ssim', 'AP', 'mAP', 'ap', 'map_score', 'score1'] 

    for col_name in possible_model_cols:
        if col_name in df.columns:
            model_col = col_name
            break
    
    for col_name in possible_ap_cols:
        if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            ap_col = col_name
            break

    if model_col and ap_col:
        plt.figure(figsize=(12, 7))
        # This loop attempts to update the outer 'hue_col' if a suitable candidate is found.
        # It will be used for the bar plot's hue and potentially as dataset_col later.
        possible_hue_candidates_plot1 = ['dataset', 'dataset_name', 'task'] # Add other relevant categorical columns if needed
        
        temp_hue_candidate = None
        for col_candidate in possible_hue_candidates_plot1:
            if col_candidate in df.columns and col_candidate != model_col:
                # Check for suitability as hue (not too many unique values, and more than 1)
                if 1 < df[col_candidate].nunique() < min(20, len(df) // 2):
                    temp_hue_candidate = col_candidate
                    break
        
        if temp_hue_candidate:
            hue_col = temp_hue_candidate # Update the main hue_col

        sns.barplot(data=df, x=model_col, y=ap_col, hue=hue_col) 
        plt.title(f'{ap_col} by {model_col}' + (f' (grouped by {hue_col})' if hue_col else ''))
        plt.ylabel(ap_col)
        plt.xlabel(model_col)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{ap_col}_by_{model_col}_bar.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close()
    else:
        print(f"Could not generate AP by Model bar plot. Required columns not found or not numeric (tried models: {possible_model_cols}, metrics: {possible_ap_cols}).")

    # Example 2: Box plot for a metric across different categories (e.g., 'dataset_name')
    # dataset_col will use the hue_col found in Example 1 (or None if none was suitable)
    dataset_col = hue_col 
    map_col = None
    # 3. Update possible column names
    possible_map_cols = ['psnr', 'ssim', 'lpips_alex', 'lpips_vgg', 'mAP', 'AP', 'map_score', 'score2'] 

    for col_name in possible_map_cols:
         if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            # Try to pick a different metric than ap_col for variety, if ap_col was found
            if ap_col and col_name == ap_col and len(possible_map_cols) > 1:
                for next_col_name in possible_map_cols:
                    if next_col_name != ap_col and next_col_name in df.columns and pd.api.types.is_numeric_dtype(df[next_col_name]):
                        map_col = next_col_name
                        break
                if not map_col: # Fallback to original if no other different metric found
                     map_col = col_name
            else:
                map_col = col_name
            if map_col: # If a metric is found
                break

    if dataset_col and map_col and dataset_col != map_col: # Ensure dataset_col is categorical
        # Check if dataset_col is not overly unique (e.g. not an ID)
        if df[dataset_col].nunique() < len(df) / 2 and df[dataset_col].nunique() > 1 :
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=dataset_col, y=map_col)
            plt.title(f'Distribution of {map_col} by {dataset_col}')
            plt.ylabel(map_col)
            plt.xlabel(dataset_col)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{map_col}_by_{dataset_col}_boxplot.png")
            plt.savefig(plot_path)
            print(f"Saved plot: {plot_path}")
            plt.close()
        else:
            print(f"Skipping box plot for {map_col} by {dataset_col}: '{dataset_col}' might not be suitable for grouping (too many unique values or only one).")
    else:
        print(f"Could not generate mAP by Dataset box plot. Required columns not found, not numeric, or dataset column is the same as metric column (tried datasets: {dataset_col}, metrics: {possible_map_cols}).")

    # Example 3: Scatter plot comparing two metrics, e.g., AP50 vs AP75
    # Replace 'metric1' and 'metric2'
    metric1_col = None
    metric2_col = None
    
    # Try to find AP50 and AP75 or similar
    if 'AP50' in df.columns and pd.api.types.is_numeric_dtype(df['AP50']): metric1_col = 'AP50'
    if 'AP75' in df.columns and pd.api.types.is_numeric_dtype(df['AP75']): metric2_col = 'AP75'
    
    # Fallback to first two numeric columns if specific ones not found
    if not (metric1_col and metric2_col):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if model_col in numeric_cols: numeric_cols.remove(model_col) # Don't use model_col if it's numeric by mistake
        if dataset_col in numeric_cols: numeric_cols.remove(dataset_col)
        
        if len(numeric_cols) >= 2:
            metric1_col = numeric_cols[0]
            metric2_col = numeric_cols[1]
            print(f"Using fallback metrics for scatter plot: {metric1_col} and {metric2_col}")

    if metric1_col and metric2_col and metric1_col != metric2_col:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=metric1_col, y=metric2_col, hue=model_col) # Hue by model if model_col exists
        plt.title(f'{metric1_col} vs. {metric2_col}')
        plt.xlabel(metric1_col)
        plt.ylabel(metric2_col)
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{metric1_col}_vs_{metric2_col}_scatter.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close()
    else:
        print(f"Could not generate scatter plot. Need at least two distinct numeric metric columns.")

    print(f"\nFinished generating plots. Check the '{output_dir}' directory.")

if __name__ == "__main__":
    # Assuming the CSV is in the specified path from the prompt
    csv_file = "drone_eval_results/evaluation_metrics.csv"
    
    # Create a dummy CSV for testing if it doesn't exist
    if not os.path.exists(csv_file):
        print(f"Dummy CSV created at {csv_file} for testing purposes.")
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        dummy_data = {
            'model_name': ['YOLOv8n', 'YOLOv8s', 'YOLOv8n', 'YOLOv8s', 'DETR', 'DETR'],
            'dataset_name': ['VisDrone', 'VisDrone', 'COCO', 'COCO', 'VisDrone', 'COCO'],
            'AP': [0.35, 0.42, 0.40, 0.48, 0.38, 0.45],
            'mAP': [0.33, 0.40, 0.38, 0.46, 0.36, 0.43],
            'AP50': [0.55, 0.62, 0.60, 0.68, 0.58, 0.65],
            'AP75': [0.25, 0.32, 0.30, 0.38, 0.28, 0.35],
            'some_string_col': ['infoA', 'infoB', 'infoC', 'infoD', 'infoE', 'infoF']
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(csv_file, index=False)
        
    create_plots_from_csv(csv_filepath=csv_file)