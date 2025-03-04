#!/usr/bin/env python3
import os
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuration
LOG_DIR = "ext_exps/job_logs"  # Directory containing log files
OUTPUT_DIR = "ext_exps/analysis"  # Directory for output files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex patterns to extract information
val_psnr_pattern = re.compile(r"\[ITER 3000\] val_full: L1 (\d+\.\d+), PSNR (\d+\.\d+)")
train_psnr_pattern = re.compile(r"\[ITER 3000\] train_sliding_window: L1 (\d+\.\d+), PSNR (\d+\.\d+)")
best_psnr_pattern = re.compile(r"Best PSNR: (\d+\.\d+) at iteration (\d+)")

# Function to extract parameter info from filename
def extract_params(filename):
    # Base filename without extension
    base = os.path.basename(filename).replace('.log', '')
    
    # First extract dataset
    if 'jumpingjacks' in base:
        dataset = 'jumpingjacks'
        param_part = base.replace('jumpingjacks_', '')
    elif 'bouncingballs' in base:
        dataset = 'bouncingballs'
        param_part = base.replace('bouncingballs_', '')
    else:
        dataset = 'unknown'
        param_part = base
        
    # Initialize parameter dict
    params = {
        'dataset': dataset,
        'experiment': param_part,
        'model_size': 'medium',  # default values
        'latent_dim': 32,
        'inference': 'deterministic',
        'noise_std': 0.01,
        'batch_size': 8192,
        'learning_rate': 1e-3,
        'window_obs': '40_20',
        'ode_weight': 1,
        'render_weight': 1,
        'reg_weight': 1e-3,
        'activation': 'relu',
        'model_type': 'transformer',
        'tolerance': 'high'
    }
    
    # Extract parameters from experiment name
    if 'model_size_small' in param_part:
        params['model_size'] = 'small'
    elif 'model_size_large' in param_part:
        params['model_size'] = 'large'
    
    if 'var' in param_part:
        params['inference'] = 'variational'
    elif 'det' in param_part:
        params['inference'] = 'deterministic'
    
    # Extract latent dimension
    latent_match = re.search(r'latent_dim_(\d+)', param_part)
    if latent_match:
        params['latent_dim'] = int(latent_match.group(1))
    
    # Extract noise standard deviation
    noise_match = re.search(r'noise_std_(\d+\.\d+)', param_part)
    if noise_match:
        params['noise_std'] = float(noise_match.group(1))
    
    # Extract batch size
    batch_match = re.search(r'batch_size_(\d+)', param_part)
    if batch_match:
        params['batch_size'] = int(batch_match.group(1))
    
    # Extract learning rate
    lr_match = re.search(r'learning_rate_([\d\.e\-]+)', param_part)
    if lr_match:
        params['learning_rate'] = float(lr_match.group(1))
    
    # Extract window/observation length
    window_match = re.search(r'window_obs_(\d+)_(\d+)', param_part)
    if window_match:
        params['window_obs'] = f"{window_match.group(1)}_{window_match.group(2)}"
    
    # Extract ODE weight
    ode_match = re.search(r'ode_weight_([\d\.e\-]+)', param_part)
    if ode_match:
        params['ode_weight'] = float(ode_match.group(1))
    
    # Extract render weight
    render_match = re.search(r'render_weight_(\d+)', param_part)
    if render_match:
        params['render_weight'] = int(render_match.group(1))
    
    # Extract regularization weight
    reg_match = re.search(r'reg_weight_([\d\.e\-]+)', param_part)
    if reg_match:
        params['reg_weight'] = float(reg_match.group(1))
    
    # Check if tanh activation is used
    if 'tanh' in param_part:
        params['activation'] = 'tanh'
    
    # Check if ODE-RNN is used
    if 'ode_rnn' in param_part:
        params['model_type'] = 'ode_rnn'
    
    # Extract tolerance level
    if 'tolerance_high' in param_part:
        params['tolerance'] = 'high'
    elif 'tolerance_medium' in param_part:
        params['tolerance'] = 'medium'
    elif 'tolerance_low' in param_part:
        params['tolerance'] = 'low'
    
    return params

# Function to parse a log file and extract PSNR values
def parse_log_file(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    results = {
        'val_psnr': None,
        'train_psnr': None,
        'best_psnr': None,
        'best_iter': None,
    }
    
    val_match = val_psnr_pattern.search(content)
    if val_match:
        results['val_psnr'] = float(val_match.group(2))
    
    train_match = train_psnr_pattern.search(content)
    if train_match:
        results['train_psnr'] = float(train_match.group(2))
    
    best_match = best_psnr_pattern.search(content)
    if best_match:
        results['best_psnr'] = float(best_match.group(1))
        results['best_iter'] = int(best_match.group(2))
    
    return results

# Main function to process all log files
def analyze_results():
    # Find all log files
    log_files = glob.glob(os.path.join(LOG_DIR, "*.log"))
    
    # Skip the failed jobs log if it exists
    log_files = [f for f in log_files if 'failed_jobs.log' not in f]
    
    if not log_files:
        print(f"No log files found in {LOG_DIR}")
        return
    
    print(f"Found {len(log_files)} log files to analyze")
    
    # Process each log file
    results = []
    for log_file in log_files:
        params = extract_params(log_file)
        metrics = parse_log_file(log_file)
        
        # Combine parameters and metrics
        entry = {**params, **metrics}
        results.append(entry)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save full results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "all_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved all results to {csv_path}")
    
    # Create a summary table focusing on parameters and PSNR values
    summary_columns = [
        'dataset', 'model_size', 'model_type', 'latent_dim', 'inference', 
        'noise_std', 'batch_size', 'learning_rate', 'window_obs', 
        'ode_weight', 'render_weight', 'reg_weight', 'activation', 'tolerance',
        'val_psnr', 'train_psnr', 'best_psnr'
    ]
    
    summary_df = df[summary_columns].copy()
    summary_df = summary_df.sort_values(by=['dataset', 'best_psnr'], ascending=[True, False])
    
    # Save summary to CSV
    summary_csv_path = os.path.join(OUTPUT_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary to {summary_csv_path}")
    
    # Generate HTML tables for better viewing
    html_path = os.path.join(OUTPUT_DIR, "results_table.html")
    with open(html_path, 'w') as f:
        f.write("<html><head><title>ODE Render Experiment Results</title>")
        f.write("<style>")
        f.write("table { border-collapse: collapse; width: 100%; }")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        f.write("tr:nth-child(even) { background-color: #f2f2f2; }")
        f.write("th { background-color: #4CAF50; color: white; }")
        f.write("tr:hover { background-color: #ddd; }")
        f.write("</style>")
        f.write("</head><body>")
        f.write("<h1>ODE Render Experiment Results</h1>")
        
        # Add dataset tabs
        f.write("<div style='display: flex; margin-bottom: 20px;'>")
        f.write("<button onclick=\"showDataset('all')\" style='margin-right: 10px;'>All Datasets</button>")
        f.write("<button onclick=\"showDataset('jumpingjacks')\" style='margin-right: 10px;'>Jumpingjacks</button>")
        f.write("<button onclick=\"showDataset('bouncingballs')\" style='margin-right: 10px;'>Bouncingballs</button>")
        f.write("</div>")
        
        # Table for all results
        f.write("<div id='all'>")
        f.write("<h2>All Results</h2>")
        f.write(summary_df.to_html(classes='dataframe', index=False))
        f.write("</div>")
        
        # Table for jumpingjacks
        if 'jumpingjacks' in summary_df['dataset'].values:
            f.write("<div id='jumpingjacks' style='display:none;'>")
            f.write("<h2>Jumpingjacks Results</h2>")
            f.write(summary_df[summary_df['dataset'] == 'jumpingjacks'].to_html(classes='dataframe', index=False))
            f.write("</div>")
        
        # Table for bouncingballs
        if 'bouncingballs' in summary_df['dataset'].values:
            f.write("<div id='bouncingballs' style='display:none;'>")
            f.write("<h2>Bouncingballs Results</h2>")
            f.write(summary_df[summary_df['dataset'] == 'bouncingballs'].to_html(classes='dataframe', index=False))
            f.write("</div>")
        
        # Add JavaScript for dataset tabs
        f.write("<script>")
        f.write("function showDataset(dataset) {")
        f.write("  document.getElementById('all').style.display = 'none';")
        f.write("  document.getElementById('jumpingjacks').style.display = 'none';")
        f.write("  document.getElementById('bouncingballs').style.display = 'none';")
        f.write("  document.getElementById(dataset).style.display = 'block';")
        f.write("}")
        f.write("</script>")
        
        f.write("</body></html>")
    
    print(f"Generated HTML table at {html_path}")
    
    # Create visualizations
    create_visualizations(df)
    
    return df

# Function to create visualizations
def create_visualizations(df):
    # Set up style
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    
    # Convert any string numeric values to float
    for col in ['val_psnr', 'train_psnr', 'best_psnr']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter to keep only rows with valid PSNR values
    df_filtered = df.dropna(subset=['best_psnr'])
    
    # 1. Effect of model size on PSNR
    plt.figure(figsize=(12, 6))
    model_size_plot = sns.barplot(x='model_size', y='best_psnr', hue='dataset', data=df_filtered)
    plt.title('Effect of Model Size on PSNR')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_size_effect.png'))
    
    # 2. Effect of latent dimension on PSNR
    plt.figure(figsize=(12, 6))
    latent_dim_plot = sns.lineplot(x='latent_dim', y='best_psnr', hue='dataset', style='inference', 
                                    markers=True, data=df_filtered)
    plt.title('Effect of Latent Dimension on PSNR')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'latent_dim_effect.png'))
    
    # 3. Effect of batch size on PSNR
    plt.figure(figsize=(12, 6))
    batch_size_plot = sns.lineplot(x='batch_size', y='best_psnr', hue='dataset', style='inference', 
                                    markers=True, data=df_filtered)
    plt.xscale('log')
    plt.title('Effect of Batch Size on PSNR')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'batch_size_effect.png'))
    
    # 4. Effect of learning rate on PSNR
    plt.figure(figsize=(12, 6))
    lr_plot = sns.lineplot(x='learning_rate', y='best_psnr', hue='dataset', style='inference', 
                           markers=True, data=df_filtered)
    plt.xscale('log')
    plt.title('Effect of Learning Rate on PSNR')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'learning_rate_effect.png'))
    
    # 5. Variational vs Deterministic
    plt.figure(figsize=(12, 6))
    inf_plot = sns.barplot(x='inference', y='best_psnr', hue='dataset', data=df_filtered)
    plt.title('Variational vs Deterministic Inference')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'inference_effect.png'))
    
    # 6. Effect of model type
    plt.figure(figsize=(12, 6))
    model_type_plot = sns.barplot(x='model_type', y='best_psnr', hue='dataset', data=df_filtered)
    plt.title('Transformer vs ODE-RNN Models')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_type_effect.png'))
    
    # 7. Window/observation length effect
    plt.figure(figsize=(12, 6))
    window_plot = sns.barplot(x='window_obs', y='best_psnr', hue='dataset', data=df_filtered)
    plt.title('Effect of Window/Observation Length')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'window_obs_effect.png'))
    
    # 8. Correlation heatmap of hyperparameters
    # First convert categorical variables to numeric
    df_corr = df_filtered.copy()
    
    # Convert model_size to numeric
    size_map = {'small': 0, 'medium': 1, 'large': 2}
    df_corr['model_size_num'] = df_corr['model_size'].map(size_map)
    
    # Convert inference type to numeric
    inf_map = {'deterministic': 0, 'variational': 1}
    df_corr['inference_num'] = df_corr['inference'].map(inf_map)
    
    # Convert model_type to numeric
    model_map = {'transformer': 0, 'ode_rnn': 1}
    df_corr['model_type_num'] = df_corr['model_type'].map(model_map)
    
    # Convert tolerance to numeric
    tol_map = {'high': 0, 'medium': 1, 'low': 2}
    df_corr['tolerance_num'] = df_corr['tolerance'].map(tol_map)
    
    # Convert activation to numeric
    act_map = {'relu': 0, 'tanh': 1}
    df_corr['activation_num'] = df_corr['activation'].map(act_map)
    
    # Select columns for correlation
    corr_cols = [
        'model_size_num', 'latent_dim', 'inference_num', 'batch_size', 
        'learning_rate', 'ode_weight', 'render_weight', 'reg_weight', 
        'model_type_num', 'tolerance_num', 'activation_num', 'best_psnr'
    ]
    
    # Compute correlation matrix
    try:
        corr_df = df_corr[corr_cols].corr()
        
        plt.figure(figsize=(14, 12))
        heatmap = sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation of Hyperparameters with PSNR')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'parameter_correlation.png'))
    except Exception as e:
        print(f"Error creating correlation heatmap: {e}")
    
    print(f"Saved visualizations to {OUTPUT_DIR}")

if __name__ == "__main__":
    df = analyze_results()
    
    # Print top 5 configurations for each dataset
    if df is not None and not df.empty:
        print("\n--- Top 5 Configurations for Jumpingjacks ---")
        jumpingjacks_df = df[df['dataset'] == 'jumpingjacks'].sort_values(by='best_psnr', ascending=False)
        if not jumpingjacks_df.empty:
            top5_jj = jumpingjacks_df.head(5)
            for i, row in top5_jj.iterrows():
                print(f"PSNR: {row['best_psnr']:.4f} - {row['experiment']}")
        
        print("\n--- Top 5 Configurations for Bouncingballs ---")
        bouncingballs_df = df[df['dataset'] == 'bouncingballs'].sort_values(by='best_psnr', ascending=False)
        if not bouncingballs_df.empty:
            top5_bb = bouncingballs_df.head(5)
            for i, row in top5_bb.iterrows():
                print(f"PSNR: {row['best_psnr']:.4f} - {row['experiment']}")