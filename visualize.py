import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load baseline and distillation results."""
    baseline_df = pd.read_csv('baseline_results.csv')
    distillation_df = pd.read_csv('distillation_results.csv')
    
    print(f"Loaded {len(baseline_df)} baseline results")
    print(f"Loaded {len(distillation_df)} distillation results")
    
    return baseline_df, distillation_df

def load_parent_model_metrics(parent_model, base_dir='eval_res'):
    parent_dir = Path(base_dir) / parent_model
    metrics = []

    if not parent_dir.exists():
        print(f"Parent directory not found for {parent_model}: {parent_dir}")
        return pd.DataFrame(columns=['dataset', 'numshot', 'parent_auc'])

    for dataset_dir in parent_dir.iterdir():
        if not dataset_dir.is_dir():
            continue

        for shot_dir in dataset_dir.glob('*_shot'):
            eval_file = shot_dir / 'eval_metrics.json'
            if not eval_file.exists():
                continue

            try:
                with open(eval_file, 'r') as f:
                    metrics_data = json.load(f)
                auc_value = metrics_data.get('auc')
                if auc_value is None:
                    continue

                numshot_str = shot_dir.name.replace('_shot', '')
                try:
                    numshot = int(numshot_str)
                except ValueError:
                    # Handle 'all' shot setting
                    if numshot_str == 'all':
                        numshot = 'all'
                    else:
                        continue

                metrics.append({
                    'dataset': dataset_dir.name,
                    'numshot': numshot,
                    'parent_auc': auc_value
                })
            except Exception as exc:
                print(f"Error reading {eval_file}: {exc}")

    return pd.DataFrame(metrics)

def merge_baseline_distillation(baseline_df, distillation_df):
    """Merge baseline and distillation results for comparison."""
    # Rename baseline model column to match distillation
    baseline_renamed = baseline_df.rename(columns={'baseline_model': 'student_model'})
    baseline_renamed['method'] = 'baseline'
    
    # Add method column to distillation
    distillation_df = distillation_df.copy()
    distillation_df['method'] = 'distillation'
    
    # Combine datasets
    combined_df = pd.concat([
        baseline_renamed[['dataset', 'numshot', 'student_model', 'mean_auc', 'std_auc', 'mean_complexity', 'method']],
        distillation_df[['dataset', 'numshot', 'student_model', 'mean_auc', 'std_auc', 'mean_complexity', 'method', 'parent_model']]
    ], ignore_index=True)
    
    # Shorten model names for better visualization
    model_name_map = {
        'logistic_rule_regression': 'log. rule regr.',
        'logistic_regression': 'log. reg.'
    }
    combined_df['student_model'] = combined_df['student_model'].replace(model_name_map)
    
    return combined_df

def create_comparison_plots(baseline_df, distillation_df):
    """Create comprehensive comparison plots."""
    
    # Create output directory
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Get unique parent models
    parent_models = distillation_df['parent_model'].unique()
    student_models = distillation_df['student_model'].unique()
    
    for parent_model in parent_models:
        print(f"\nCreating plots for parent model: {parent_model}")
        
        # Filter distillation results for this parent model
        parent_distill = distillation_df[distillation_df['parent_model'] == parent_model].copy()
        
        # Merge with baseline for comparison
        merged_df = merge_baseline_distillation(baseline_df, parent_distill)
        
        # Create plots for this parent model
        parent_metrics_df = load_parent_model_metrics(parent_model)
        create_parent_model_plots(merged_df, parent_model, output_dir, parent_metrics_df)


def create_parent_model_plots(merged_df, parent_model, output_dir, parent_metrics_df=None):
    """Create plots for a specific parent model."""
    if parent_metrics_df is None:
        parent_metrics_df = pd.DataFrame(columns=['dataset', 'numshot', 'parent_auc'])
    
    # 3 horizontal subplots for each parent model
    plt.figure(figsize=(21, 6))
    
    model_name_map = {
        'tabllm': 'TabLLM',
        'carte': 'CARTE', 
        'tabpfn': 'TabPFN',
        'tabm': 'TabM'
    }
    display_name = model_name_map.get(parent_model.lower(), parent_model)
    plt.suptitle(f'{display_name}', fontsize=32, fontweight='bold', y=0.95)
    
    # 1. Box plot comparison
    plt.subplot(1, 3, 1)
    student_model_order = ['decision_tree', 'log. rule regr.', 'ttnet', 'xgboost']
    sns.boxplot(data=merged_df, x='student_model', y='mean_auc', hue='method', order=student_model_order)
    plt.xticks(rotation=15, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('', fontsize=16)
    plt.ylabel('', fontsize=16)
    plt.legend(title='Method', fontsize=14, title_fontsize=16)
    
    # 2. Shot setting analysis
    plt.subplot(1, 3, 2)
    shot_values_numeric = [4, 8, 16, 32, 64, 128, 256]
    xtick_positions = [4, 16, 64, 256, 275]
    xtick_labels = ['4', '16', '64', '256', 'all']
    
    student_model_order = ['decision_tree', 'log. reg.', 'log. rule regr.', 'ttnet', 'xgboost']
    student_models = [model for model in student_model_order if model in merged_df['student_model'].unique()]
    colors = plt.cm.Set1(np.linspace(0, 1, len(student_models)))
    
    for i, student in enumerate(student_models):
        student_data = merged_df[merged_df['student_model'] == student]
        
        # --- Plot Baseline ---
        baseline_data = student_data[student_data['method'] == 'baseline']
        if len(baseline_data) > 0:
            baseline_shot = baseline_data.groupby('numshot')['mean_auc'].mean().reset_index()
            baseline_x = []
            baseline_y = []
            for _, row in baseline_shot.iterrows():
                numshot = row['numshot']
                
                # FIX: explicitly handle types
                if str(numshot) == 'all':
                    baseline_x.append(275)
                else:
                    try:
                        # Convert to int for comparison
                        val = int(numshot)
                        if val in shot_values_numeric:
                            baseline_x.append(val)
                            baseline_y.append(row['mean_auc'])
                    except ValueError:
                        continue
                        
                # Handle 'all' y-value separately if it wasn't added in the numeric block
                if str(numshot) == 'all':
                    baseline_y.append(row['mean_auc'])

            if baseline_x:
                sorted_pairs = sorted(zip(baseline_x, baseline_y))
                baseline_x, baseline_y = zip(*sorted_pairs)
                plt.plot(baseline_x, baseline_y, 
                         color=colors[i], linestyle='--', marker='o', alpha=0.7,
                         linewidth=2)
        
        # --- Plot Distillation ---
        distill_data = student_data[student_data['method'] == 'distillation']
        if len(distill_data) > 0:
            distill_shot = distill_data.groupby('numshot')['mean_auc'].mean().reset_index()
            distill_x = []
            distill_y = []
            for _, row in distill_shot.iterrows():
                numshot = row['numshot']
                
                # FIX: explicitly handle types
                if str(numshot) == 'all':
                    distill_x.append(275)
                else:
                    try:
                        val = int(numshot)
                        if val in shot_values_numeric:
                            distill_x.append(val)
                            distill_y.append(row['mean_auc'])
                    except ValueError:
                        continue
                
                if str(numshot) == 'all':
                    distill_y.append(row['mean_auc'])

            if distill_x:
                sorted_pairs = sorted(zip(distill_x, distill_y))
                distill_x, distill_y = zip(*sorted_pairs)
                plt.plot(distill_x, distill_y, 
                         color=colors[i], linestyle='-', marker='s', alpha=0.9,
                         linewidth=2)
    
    legend_elements = []
    for i, student in enumerate(student_models):
        legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=3, label=student))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Baseline'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Distillation (ours)'))
    
    plt.legend(handles=legend_elements, loc='best', fontsize=14)
    plt.gca().set_xticks(shot_values_numeric + [275], minor=True)
    plt.gca().set_xticks(xtick_positions, minor=False)
    plt.gca().tick_params(axis='x', which='minor', length=8, width=2)
    plt.gca().tick_params(axis='x', which='major', length=4, width=1)
    plt.xticks(xtick_positions, xtick_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(2, 290)
    plt.xlabel('Number of Shots', fontsize=16)
    plt.ylabel('Mean AUC', fontsize=16)
    plt.grid(True, alpha=0.3)

    # 3. Overall shot setting comparison
    plt.subplot(1, 3, 3)
    shot_comparison = merged_df.groupby(['numshot', 'method']).agg({
        'mean_auc': 'mean',
        'std_auc': 'mean'
    }).reset_index()
    
    # --- Baseline Average ---
    baseline_shot = shot_comparison[shot_comparison['method'] == 'baseline']
    baseline_x = []
    baseline_y = []
    baseline_std = []
    for _, row in baseline_shot.iterrows():
        numshot = row['numshot']
        
        # FIX: explicitly handle types
        if str(numshot) == 'all':
            baseline_x.append(275)
            baseline_y.append(row['mean_auc'])
            baseline_std.append(row['std_auc'])
        else:
            try:
                val = int(numshot)
                if val in shot_values_numeric:
                    baseline_x.append(val)
                    baseline_y.append(row['mean_auc'])
                    baseline_std.append(row['std_auc'])
            except ValueError:
                continue

    if baseline_x:
        sorted_data = sorted(zip(baseline_x, baseline_y, baseline_std))
        baseline_x, baseline_y, baseline_std = zip(*sorted_data)
        baseline_x = np.array(baseline_x)
        baseline_y = np.array(baseline_y)
        baseline_std = np.array(baseline_std)
        
        plt.plot(baseline_x, baseline_y, 
                 color='blue', linestyle='--', marker='o', alpha=0.8,
                 label='Baseline (avg)', linewidth=3, markersize=8)
        plt.fill_between(baseline_x, baseline_y - baseline_std, baseline_y + baseline_std,
                         color='blue', alpha=0.2)
    
    # --- Distillation Average ---
    distill_shot = shot_comparison[shot_comparison['method'] == 'distillation']
    distill_x = []
    distill_y = []
    distill_std = []
    for _, row in distill_shot.iterrows():
        numshot = row['numshot']
        
        # FIX: explicitly handle types
        if str(numshot) == 'all':
            distill_x.append(275)
            distill_y.append(row['mean_auc'])
            distill_std.append(row['std_auc'])
        else:
            try:
                val = int(numshot)
                if val in shot_values_numeric:
                    distill_x.append(val)
                    distill_y.append(row['mean_auc'])
                    distill_std.append(row['std_auc'])
            except ValueError:
                continue

    if distill_x:
        sorted_data = sorted(zip(distill_x, distill_y, distill_std))
        distill_x, distill_y, distill_std = zip(*sorted_data)
        distill_x = np.array(distill_x)
        distill_y = np.array(distill_y)
        distill_std = np.array(distill_std)
        
        plt.plot(distill_x, distill_y, 
                 color='red', linestyle='-', marker='s', alpha=0.8,
                 label='Distillation (ours)', linewidth=3, markersize=8)
        plt.fill_between(distill_x, distill_y - distill_std, distill_y + distill_std,
                         color='red', alpha=0.2)

    # --- Teacher Model ---
    if not parent_metrics_df.empty:
        parent_shot = parent_metrics_df.groupby('numshot')['parent_auc'].mean().reset_index()
        parent_x = []
        parent_y = []
        for _, row in parent_shot.iterrows():
            numshot = row['numshot']
            
            # FIX: explicitly handle types
            if str(numshot) == 'all':
                parent_x.append(275)
            else:
                try:
                    val = int(numshot)
                    if val in shot_values_numeric:
                        parent_x.append(val)
                        parent_y.append(row['parent_auc'])
                except ValueError:
                    continue
            
            if str(numshot) == 'all':
                parent_y.append(row['parent_auc'])

        if parent_x:
            sorted_pairs = sorted(zip(parent_x, parent_y))
            parent_x, parent_y = zip(*sorted_pairs)
            parent_y = np.array(parent_y)
            plt.plot(parent_x, parent_y,
                     color='green', linestyle='-.', marker='^', alpha=0.8,
                     label=f'{display_name} (teacher)', linewidth=3, markersize=8)

    plt.gca().set_xticks(shot_values_numeric + [275], minor=True)
    plt.gca().set_xticks(xtick_positions, minor=False)
    plt.gca().tick_params(axis='x', which='minor', length=8, width=2)
    plt.gca().tick_params(axis='x', which='major', length=4, width=1)
    plt.xticks(xtick_positions, xtick_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(2, 290)
    plt.xlabel('Number of Shots', fontsize=16)
    plt.ylabel('Mean AUC (averaged across all models)', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{parent_model}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_per_dataset_plots(baseline_df, distillation_df):
    """Create individual plots for each dataset showing baseline vs distillation."""
    output_dir = Path('visualization_results') / 'per_dataset'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter distillation data to only include TabPFN
    distillation_df = distillation_df[distillation_df['parent_model'] == 'tabpfn'].copy()
    
    # Merge baseline and distillation data
    merged_df = merge_baseline_distillation(baseline_df, distillation_df)
    
    # Get all datasets
    datasets = sorted(merged_df['dataset'].unique())
    
    # Shot settings
    shot_values_numeric = [4, 8, 16, 32, 64, 128, 256]
    xtick_positions = [4, 16, 64, 256, 275]
    xtick_labels = ['4', '16', '64', '256', 'all']
    
    # Student models in order
    student_model_order = ['decision_tree', 'log. reg.', 'log. rule regr.', 'ttnet', 'xgboost']
    
    # Load parent model metrics for TabPFN only
    parent_metrics_df = load_parent_model_metrics('tabpfn')
    
    for dataset in datasets:
        print(f"Creating plot for dataset: {dataset}")
        
        # Filter data for this dataset
        dataset_data = merged_df[merged_df['dataset'] == dataset]
        
        # Get available student models for this dataset
        student_models = [model for model in student_model_order 
                         if model in dataset_data['student_model'].unique()]
        
        if not student_models:
            print(f"  No student models found for {dataset}, skipping...")
            continue
        
        # Create figure
        plt.figure(figsize=(10, 7))
        colors = plt.cm.Set1(np.linspace(0, 1, len(student_models)))
        
        for i, student in enumerate(student_models):
            student_data = dataset_data[dataset_data['student_model'] == student]
            
            # --- Plot Baseline ---
            baseline_data = student_data[student_data['method'] == 'baseline']
            if len(baseline_data) > 0:
                baseline_shot = baseline_data.groupby('numshot')['mean_auc'].mean().reset_index()
                baseline_x = []
                baseline_y = []
                for _, row in baseline_shot.iterrows():
                    numshot = row['numshot']
                    
                    if str(numshot) == 'all':
                        baseline_x.append(275)
                        baseline_y.append(row['mean_auc'])
                    else:
                        try:
                            val = int(numshot)
                            if val in shot_values_numeric:
                                baseline_x.append(val)
                                baseline_y.append(row['mean_auc'])
                        except ValueError:
                            continue
                
                if baseline_x:
                    sorted_pairs = sorted(zip(baseline_x, baseline_y))
                    baseline_x, baseline_y = zip(*sorted_pairs)
                    plt.plot(baseline_x, baseline_y, 
                             color=colors[i], linestyle='--', marker='o', alpha=0.7,
                             linewidth=2, markersize=6)
            
            # --- Plot Distillation ---
            distill_data = student_data[student_data['method'] == 'distillation']
            if len(distill_data) > 0:
                distill_shot = distill_data.groupby('numshot')['mean_auc'].mean().reset_index()
                distill_x = []
                distill_y = []
                for _, row in distill_shot.iterrows():
                    numshot = row['numshot']
                    
                    if str(numshot) == 'all':
                        distill_x.append(275)
                        distill_y.append(row['mean_auc'])
                    else:
                        try:
                            val = int(numshot)
                            if val in shot_values_numeric:
                                distill_x.append(val)
                                distill_y.append(row['mean_auc'])
                        except ValueError:
                            continue
                
                if distill_x:
                    sorted_pairs = sorted(zip(distill_x, distill_y))
                    distill_x, distill_y = zip(*sorted_pairs)
                    plt.plot(distill_x, distill_y, 
                             color=colors[i], linestyle='-', marker='s', alpha=0.9,
                             linewidth=2, markersize=6)
        
        # --- Plot Parent Model Performance ---
        # Get TabPFN parent model data for this dataset
        if not parent_metrics_df.empty:
            # Filter for this specific dataset
            dataset_parent = parent_metrics_df[parent_metrics_df['dataset'] == dataset]
            
            if len(dataset_parent) > 0:
                parent_shot = dataset_parent.groupby('numshot')['parent_auc'].mean().reset_index()
                parent_x = []
                parent_y = []
                for _, row in parent_shot.iterrows():
                    numshot = row['numshot']
                    
                    if str(numshot) == 'all':
                        parent_x.append(275)
                        parent_y.append(row['parent_auc'])
                    else:
                        try:
                            val = int(numshot)
                            if val in shot_values_numeric:
                                parent_x.append(val)
                                parent_y.append(row['parent_auc'])
                        except ValueError:
                            continue
                
                if parent_x:
                    sorted_pairs = sorted(zip(parent_x, parent_y))
                    parent_x, parent_y = zip(*sorted_pairs)
                    plt.plot(parent_x, parent_y,
                             color='green', linestyle='-', marker='^', alpha=0.8,
                             linewidth=2.5, markersize=7,
                             label='TabPFN (teacher)')
        
        # Create legend
        legend_elements = []
        for i, student in enumerate(student_models):
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=3, label=student))
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Baseline'))
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Distillation (ours)'))
        
        # Add TabPFN to legend if it exists in the data
        if not parent_metrics_df.empty and dataset in parent_metrics_df['dataset'].values:
            legend_elements.append(plt.Line2D([0], [0], color='green', linestyle='-.', linewidth=2.5, label='TabPFN (teacher)'))
        
        plt.legend(handles=legend_elements, loc='best', fontsize=12)
        
        # Configure axes
        plt.gca().set_xticks(shot_values_numeric + [275], minor=True)
        plt.gca().set_xticks(xtick_positions, minor=False)
        plt.gca().tick_params(axis='x', which='minor', length=8, width=2)
        plt.gca().tick_params(axis='x', which='major', length=4, width=1)
        plt.xticks(xtick_positions, xtick_labels, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlim(2, 290)
        plt.xlabel('Number of Shots', fontsize=16)
        plt.ylabel('AUC', fontsize=16)
        plt.title(f'Dataset: {dataset}', fontsize=18, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / f'{dataset}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot to {output_dir / f'{dataset}_comparison.png'}")

def main():
    """Main function to create all visualizations."""
    print("Loading data...")
    baseline_df, distillation_df = load_data()
    
    print("\nCreating visualizations...")
    
    # Create output directory
    output_dir = Path('visualization_results')
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plots for each parent model
    create_comparison_plots(baseline_df, distillation_df)
    
    # Create per-dataset plots
    print("\nCreating per-dataset plots...")
    create_per_dataset_plots(baseline_df, distillation_df)
    
    print("\nAll visualizations completed!")



if __name__ == "__main__":
    main()
