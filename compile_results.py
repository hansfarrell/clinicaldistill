import os
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from scipy.stats import wilcoxon


def compile_distillation_results(base_dir="eval_res", output_csv="distillation_results.csv"):
    """
    Compile all distillation results from eval_res/{parent_model}/{dataset_name}/{numshot}_shot/{student_model}/results.txt
    into a single CSV file, extracting mean AUC and mean complexity.
    """
    all_results = []
    
    # Get all parent models (excluding baselines)
    parent_models = [d for d in os.listdir(base_dir) if d != 'baselines' and os.path.isdir(os.path.join(base_dir, d))]
    
    for parent_model in parent_models:
        parent_path = os.path.join(base_dir, parent_model)
        
        # Get all datasets
        datasets = [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d))]
        
        for dataset in datasets:
            dataset_path = os.path.join(parent_path, dataset)
            
            # Get all shot configurations
            shot_configs = [d for d in os.listdir(dataset_path) if d.endswith('_shot') and os.path.isdir(os.path.join(dataset_path, d))]
            
            for shot_config in shot_configs:
                numshot = shot_config.replace('_shot', '')
                shot_path = os.path.join(dataset_path, shot_config)
                
                # Get all student models
                student_models = [d for d in os.listdir(shot_path) if os.path.isdir(os.path.join(shot_path, d))]
                
                for student_model in student_models:
                    if student_model.lower() == 'logistic_regression':
                        continue
                    results_file = os.path.join(shot_path, student_model, 'results.txt')
                    
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                            
                            # Extract mean AUC, mean F1, std AUC, std F1, and mean complexity
                            mean_auc = None
                            mean_f1 = None
                            std_auc = None
                            std_f1 = None
                            mean_complexity = None
                            
                            for line in content.split('\n'):
                                if line.startswith('Mean AUC:'):
                                    mean_auc = float(line.split(':')[1].strip())
                                elif line.startswith('Mean F1:'):
                                    mean_f1 = float(line.split(':')[1].strip())
                                elif line.startswith('Std AUC:'):
                                    std_auc = float(line.split(':')[1].strip())
                                elif line.startswith('Std F1:'):
                                    std_f1 = float(line.split(':')[1].strip())
                                elif line.startswith('Mean Complexity:'):
                                    complexity_str = line.split(':')[1].strip()
                                    if complexity_str != 'N/A':
                                        mean_complexity = float(complexity_str)
                            
                            # Store the result
                            result = {
                                'parent_model': parent_model,
                                'dataset': dataset,
                                'numshot': int(numshot),
                                'student_model': student_model,
                                'mean_auc': mean_auc,
                                'std_auc': std_auc,
                                'mean_f1': mean_f1,
                                'std_f1': std_f1,
                                'mean_complexity': mean_complexity
                            }
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"Error processing {results_file}: {e}")
    
    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"Compiled {len(all_results)} results into {output_csv}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        return df
    else:
        print("No distillation results found")
        return None


def merge_methods(dir_path, input_merged_csv="merged_baseline_summary.csv", output_merged_csv="merged_all.csv"):

    all_dfs = []
    all_methods = os.listdir(dir_path)
    for method in all_methods:
        csv_path = os.path.join(dir_path, method, input_merged_csv)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['method'] = method
            all_dfs.append(df)

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        merged_df.to_csv(os.path.join(dir_path, output_merged_csv), index=False)
        print(f"Merged CSV saved to {output_merged_csv}")
    else:
        print("No experiment CSVs found to merge.")


def compile_baseline_results(base_dir="eval_res/baselines", output_csv="baseline_results.csv"):
    """
    Compile all baseline results from eval_res/baselines/{dataset_name}/{numshot}_shot/{model_name}/results.txt
    into a single CSV file, extracting mean AUC and mean complexity.
    """
    all_results = []
    
    if not os.path.exists(base_dir):
        print(f"Baseline directory {base_dir} does not exist")
        return None
    
    # Get all datasets
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        
        # Get all shot configurations
        shot_configs = [d for d in os.listdir(dataset_path) if d.endswith('_shot') and os.path.isdir(os.path.join(dataset_path, d))]
        
        for shot_config in shot_configs:
            numshot = shot_config.replace('_shot', '')
            shot_path = os.path.join(dataset_path, shot_config)
            
            # Get all baseline models
            baseline_models = [d for d in os.listdir(shot_path) if os.path.isdir(os.path.join(shot_path, d))]
            
            for baseline_model in baseline_models:
                if baseline_model.lower() == 'logistic_regression':
                    continue
                results_file = os.path.join(shot_path, baseline_model, 'results.txt')
                
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            content = f.read()
                        
                        # Extract mean AUC, mean F1, std AUC, std F1, and mean complexity
                        mean_auc = None
                        mean_f1 = None
                        std_auc = None
                        std_f1 = None
                        mean_complexity = None
                        
                        for line in content.split('\n'):
                            if line.startswith('Mean AUC:'):
                                mean_auc = float(line.split(':')[1].strip())
                            elif line.startswith('Mean F1:'):
                                mean_f1 = float(line.split(':')[1].strip())
                            elif line.startswith('Std AUC:'):
                                std_auc = float(line.split(':')[1].strip())
                            elif line.startswith('Std F1:'):
                                std_f1 = float(line.split(':')[1].strip())
                            elif line.startswith('Mean Complexity:'):
                                complexity_str = line.split(':')[1].strip()
                                if complexity_str != 'N/A':
                                    mean_complexity = float(complexity_str)
                        
                        # Store the result
                        result = {
                            'dataset': dataset,
                            'numshot': int(numshot),
                            'baseline_model': baseline_model,
                            'mean_auc': mean_auc,
                            'std_auc': std_auc,
                            'mean_f1': mean_f1,
                            'std_f1': std_f1,
                            'mean_complexity': mean_complexity
                        }
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"Error processing {results_file}: {e}")
    
    # Convert to DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"Compiled {len(all_results)} baseline results into {output_csv}")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        return df
    else:
        print("No baseline results found")
        return None


def extract_individual_aucs_from_files(base_dir="eval_res"):
    distill_aucs = []
    baseline_aucs = []
    
    # Get all parent models (excluding baselines)
    parent_models = [d for d in os.listdir(base_dir) if d != 'baselines' and os.path.isdir(os.path.join(base_dir, d))]
    
    # Process baseline results first to get all configurations
    baseline_dir = os.path.join(base_dir, 'baselines')
    baseline_configs = {}  # key: (dataset, numshot, model), value: list of AUCs
    
    if os.path.exists(baseline_dir):
        for dataset in os.listdir(baseline_dir):
            dataset_path = os.path.join(baseline_dir, dataset)
            if not os.path.isdir(dataset_path):
                continue
                
            for shot_config in os.listdir(dataset_path):
                if not shot_config.endswith('_shot') or not os.path.isdir(os.path.join(dataset_path, shot_config)):
                    continue
                numshot = int(shot_config.replace('_shot', ''))
                shot_path = os.path.join(dataset_path, shot_config)
                
                for model in os.listdir(shot_path):
                    model_path = os.path.join(shot_path, model)
                    if not os.path.isdir(model_path):
                        continue
                    results_file = os.path.join(model_path, 'results.txt')
                    
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                            
                            for line in content.split('\n'):
                                if line.startswith('AUCs:'):
                                    auc_str = line.split(':', 1)[1].strip()
                                    aucs = eval(auc_str)
                                    baseline_configs[(dataset, numshot, model)] = aucs
                                    break
                        except Exception as e:
                            print(f"Error processing baseline {results_file}: {e}")
    
    # Now process distillation results and match with baselines
    for parent_model in parent_models:
        parent_path = os.path.join(base_dir, parent_model)
        
        for dataset in os.listdir(parent_path):
            dataset_path = os.path.join(parent_path, dataset)
            if not os.path.isdir(dataset_path):
                continue
                
            for shot_config in os.listdir(dataset_path):
                if not shot_config.endswith('_shot') or not os.path.isdir(os.path.join(dataset_path, shot_config)):
                    continue
                numshot = int(shot_config.replace('_shot', ''))
                shot_path = os.path.join(dataset_path, shot_config)
                
                for student_model in os.listdir(shot_path):
                    student_path = os.path.join(shot_path, student_model)
                    if not os.path.isdir(student_path):
                        continue
                    results_file = os.path.join(student_path, 'results.txt')
                    
                    if os.path.exists(results_file):
                        try:
                            with open(results_file, 'r') as f:
                                content = f.read()
                            
                            for line in content.split('\n'):
                                if line.startswith('AUCs:'):
                                    auc_str = line.split(':', 1)[1].strip()
                                    distill_aucs_list = eval(auc_str)
                                    
                                    # Try to match with baseline
                                    baseline_key = (dataset, numshot, student_model)
                                    if baseline_key in baseline_configs:
                                        baseline_aucs_list = baseline_configs[baseline_key]
                                        
                                        # Match seed by seed
                                        min_len = min(len(distill_aucs_list), len(baseline_aucs_list))
                                        for i in range(min_len):
                                            distill_aucs.append(distill_aucs_list[i])
                                            baseline_aucs.append(baseline_aucs_list[i])
                                    break
                        except Exception as e:
                            print(f"Error processing distillation {results_file}: {e}")
    
    return distill_aucs, baseline_aucs


def perform_wilcoxon_test(output_file="wilcoxon_test.txt"):
    distill_aucs, baseline_aucs = extract_individual_aucs_from_files()
    stat, p = wilcoxon(distill_aucs, baseline_aucs, alternative='greater')
    
    distill_aucs = np.array(distill_aucs)
    baseline_aucs = np.array(baseline_aucs)
    diff = distill_aucs - baseline_aucs
    
    output_lines = []
    output_lines.append("Wilcoxon signed-rank test")
    output_lines.append("Null hypothesis H_0: distillation AUC <= baseline AUC")
    output_lines.append("Alternative hypothesis H_1: distillation AUC > baseline AUC")
    output_lines.append("Test significance level: 0.01")
    output_lines.append("")
    output_lines.append(f"Mean difference between distillation AUC and baseline AUC: {diff.mean()}")
    output_lines.append(f"Test statistic: {stat}")
    output_lines.append(f"p-value: {p}")
    output_lines.append(f"N (paired samples): {len(distill_aucs)}")
    output_lines.append("")
    if p < 0.01:
        output_lines.append("Conclusion: because p < 0.01, reject the null hypothesis.")
    else:
        output_lines.append("Conclusion: because p >= 0.01, insufficient evidence to reject the null hypothesis.")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nResults saved to: {output_file}")


def generate_latex_tables(baseline_csv='baseline_results.csv', 
                          distillation_csv='distillation_results.csv',
                          output_file='results_tables.tex'):
    """
    Generate LaTeX tables for k=4, 32, and 256 shots.
    Each table shows Dataset, Student Model, Baseline, TabPFN, and TabM results.
    """
    # Load the data
    baseline_df = pd.read_csv(baseline_csv)
    distillation_df = pd.read_csv(distillation_csv)
    
    # Get unique datasets
    datasets = sorted(distillation_df['dataset'].unique())
    
    # Student model mapping for better display names
    model_name_map = {
        'ttnet': 'TT',
        'xgboost': 'XGB',
        'logistic_rule_regression': 'LRR',
        'decision_tree': 'DT'
    }
    
    # Dataset name mapping for better display
    dataset_name_map = {
        'breastcancer': 'Breast',
        'breastcancer2': 'Breast 2',
        'respiratory': 'Respiratory',
        'diabetes': 'Diabetes',
        'heart': 'Heart',
        'coloncancer': 'Colon',
        'chemotherapy': 'Chemo'
    }
    
    student_models = ['ttnet', 'xgboost', 'logistic_rule_regression', 'decision_tree']
    shots = [4, 32, 256]
    
    latex_output = []
    
    for k in shots:
        latex_output.append(f"% Table for k={k} shots")
        latex_output.append("\\begin{table}")
        latex_output.append("\\centering")
        latex_output.append(f"\\caption{{Results for $k={k}$ shots. Mean AUC with standard deviation shown as superscript.}}")
        latex_output.append(f"\\label{{tab:results_k{k}}}")
        latex_output.append("\\setlength{\\tabcolsep}{3pt}")
        latex_output.append("\\begin{tabular}{llccc}")
        latex_output.append("\\toprule")
        latex_output.append("Dataset & Student & Baseline & TabPFN & TabM \\\\")
        latex_output.append("\\midrule")
        
        for dataset in datasets:
            dataset_display = dataset_name_map.get(dataset, dataset.replace('_', ' ').title())
            
            for idx, student_model in enumerate(student_models):
                model_display = model_name_map.get(student_model, student_model)
                
                # Get baseline results
                baseline_row = baseline_df[
                    (baseline_df['dataset'] == dataset) & 
                    (baseline_df['numshot'] == k) & 
                    (baseline_df['baseline_model'] == student_model)
                ]
                
                baseline_auc = None
                if len(baseline_row) > 0:
                    baseline_auc = baseline_row['mean_auc'].values[0]
                    baseline_std = baseline_row['std_auc'].values[0]
                
                # Get TabPFN results
                tabpfn_row = distillation_df[
                    (distillation_df['dataset'] == dataset) & 
                    (distillation_df['numshot'] == k) & 
                    (distillation_df['student_model'] == student_model) &
                    (distillation_df['parent_model'] == 'tabpfn')
                ]
                
                tabpfn_auc = None
                if len(tabpfn_row) > 0:
                    tabpfn_auc = tabpfn_row['mean_auc'].values[0]
                    tabpfn_std = tabpfn_row['std_auc'].values[0]
                
                # Get TabM results
                tabm_row = distillation_df[
                    (distillation_df['dataset'] == dataset) & 
                    (distillation_df['numshot'] == k) & 
                    (distillation_df['student_model'] == student_model) &
                    (distillation_df['parent_model'] == 'tabm')
                ]
                
                tabm_auc = None
                if len(tabm_row) > 0:
                    tabm_auc = tabm_row['mean_auc'].values[0]
                    tabm_std = tabm_row['std_auc'].values[0]
                
                # Find the best AUC
                aucs = []
                if baseline_auc is not None:
                    aucs.append(baseline_auc)
                if tabpfn_auc is not None:
                    aucs.append(tabpfn_auc)
                if tabm_auc is not None:
                    aucs.append(tabm_auc)
                
                best_auc = max(aucs) if aucs else None
                
                # Format baseline string
                if baseline_auc is not None:
                    is_best = best_auc is not None and baseline_auc == best_auc
                    if pd.notna(baseline_std):
                        if is_best:
                            baseline_str = f"$\\mathbf{{{baseline_auc:.3f}^{{{baseline_std:.3f}}}}}$"
                        else:
                            baseline_str = f"${baseline_auc:.3f}^{{{baseline_std:.3f}}}$"
                    else:
                        if is_best:
                            baseline_str = f"$\\mathbf{{{baseline_auc:.3f}}}$"
                        else:
                            baseline_str = f"${baseline_auc:.3f}$"
                else:
                    baseline_str = "---"
                
                # Format TabPFN string
                if tabpfn_auc is not None:
                    is_best = best_auc is not None and tabpfn_auc == best_auc
                    if pd.notna(tabpfn_std):
                        if is_best:
                            tabpfn_str = f"$\\mathbf{{{tabpfn_auc:.3f}^{{{tabpfn_std:.3f}}}}}$"
                        else:
                            tabpfn_str = f"${tabpfn_auc:.3f}^{{{tabpfn_std:.3f}}}$"
                    else:
                        if is_best:
                            tabpfn_str = f"$\\mathbf{{{tabpfn_auc:.3f}}}$"
                        else:
                            tabpfn_str = f"${tabpfn_auc:.3f}$"
                else:
                    tabpfn_str = "---"
                
                # Format TabM string
                if tabm_auc is not None:
                    is_best = best_auc is not None and tabm_auc == best_auc
                    if pd.notna(tabm_std):
                        if is_best:
                            tabm_str = f"$\\mathbf{{{tabm_auc:.3f}^{{{tabm_std:.3f}}}}}$"
                        else:
                            tabm_str = f"${tabm_auc:.3f}^{{{tabm_std:.3f}}}$"
                    else:
                        if is_best:
                            tabm_str = f"$\\mathbf{{{tabm_auc:.3f}}}$"
                        else:
                            tabm_str = f"${tabm_auc:.3f}$"
                else:
                    tabm_str = "---"
                
                # Build the row
                if idx == 0:
                    # First row for this dataset - include dataset name with multirow
                    latex_output.append(f"\\multirow{{{len(student_models)}}}{{*}}{{{dataset_display}}} & {model_display} & {baseline_str} & {tabpfn_str} & {tabm_str} \\\\")
                else:
                    # Subsequent rows - no dataset name
                    latex_output.append(f"& {model_display} & {baseline_str} & {tabpfn_str} & {tabm_str} \\\\")
            
            # Add a line between datasets (except after the last one)
            if dataset != datasets[-1]:
                latex_output.append("\\cmidrule(lr){1-5}")
        
        latex_output.append("\\bottomrule")
        latex_output.append("\\end{tabular}")
        latex_output.append("\\end{table}")
        latex_output.append("")
        latex_output.append("")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_output))
    
    print(f"LaTeX tables saved to: {output_file}")
    print(f"Generated {len(shots)} tables for k={shots}")


if __name__ == "__main__":
    # Compile all distillation results into a single CSV
    print("Compiling distillation results...")
    df = compile_distillation_results(base_dir='eval_res', output_csv='distillation_results.csv')
    
    if df is not None:
        print("\nSample of compiled results:")
        print(df.head())
        print(f"\nUnique parent models: {df['parent_model'].unique()}")
        print(f"Unique datasets: {df['dataset'].unique()}")
        print(f"Unique shot counts: {sorted(df['numshot'].unique())}")
        print(f"Unique student models: {df['student_model'].unique()}")
    
    # Compile baseline results
    print("\n" + "="*50)
    print("Compiling baseline results...")
    baseline_df = compile_baseline_results(base_dir='eval_res/baselines', output_csv='baseline_results.csv')
    
    if baseline_df is not None:
        print("\nSample of baseline results:")
        print(baseline_df.head())
        print(f"\nUnique datasets: {baseline_df['dataset'].unique()}")
        print(f"Unique shot counts: {sorted(baseline_df['numshot'].unique())}")
        print(f"Unique baseline models: {baseline_df['baseline_model'].unique()}")
    
    # Perform Wilcoxon test
    print("\n" + "="*50)
    print("Performing Wilcoxon signed-rank test...")
    perform_wilcoxon_test(output_file='wilcoxon_test.txt')
    
    # Generate LaTeX tables
    print("\n" + "="*50)
    print("Generating LaTeX tables...")
    generate_latex_tables(
        baseline_csv='baseline_results.csv',
        distillation_csv='distillation_results.csv',
        output_file='results_tables.tex'
    )