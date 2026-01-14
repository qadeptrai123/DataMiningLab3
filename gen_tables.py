import csv
import pandas as pd
import numpy as np

def format_sci(val, dev):
    if val == 0: return "0"
    # Determine magnitude
    exponent = int(np.floor(np.log10(abs(val))))
    coeff = val / (10**exponent)
    dev_coeff = dev / (10**exponent)
    
    # Clean formatting
    return f"{coeff:.4f}E{exponent:+} $\pm$ {dev_coeff:.2f}E{exponent:+}"

def format_val(val, dev):
    # For small numbers (like time), just 2 decimal places
    return f"{val:.2f} $\pm$ {dev:.2f}"

def get_method_name(m):
    mapping = {
        "kmeans++": "$k$-means++",
        "Algo1": "Ergun",
        "Det": "Det",
        "Ours": "\\textbf{Fast-Sampling}",
        "Ours1": "\\textbf{Fast-Filtering}",
        "Ours2": "\\textbf{Fast-Estimation}"
    }
    return mapping.get(m, m)

def generate_table_fixed_k(filename, dataset_name, label_prefix):
    df = pd.read_csv(filename)
    df = df[df['dataset'] == dataset_name]
    
    # Get alphas
    alphas = sorted(df['alpha'].unique())
    k_val = df['k'].iloc[0]
    
    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{So sánh các thuật toán trên tập dữ liệu {dataset_name.upper()} với $k = {k_val}$ và các $\\alpha$ khác nhau}}")
    latex.append(f"\\label{{tab:{dataset_name}_k_{k_val}}}")
    latex.append("\\footnotesize")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{lccc}") # Removed NMI/ARI columns
    latex.append("\\hline")
    latex.append("Phương pháp & $\\alpha$ & Chi phí & Thời gian (s) \\\\ \\hline")
    
    for alpha in alphas:
        sub = df[df['alpha'] == alpha]
        
        # Get OPT (Lloyd) cost if available
        opt_row = sub[sub['method'] == 'OPT']
        opt_cost = 0
        if not opt_row.empty:
            opt_cost = opt_row.iloc[0]['cost']
            
        methods = ["kmeans++", "Algo1", "Det", "Ours", "Ours1", "Ours2"]
        
        first = True
        for m in methods:
            row = sub[sub['method'] == m]
            if row.empty: continue
            
            cost = row.iloc[0]['cost']
            cost_dev = row.iloc[0]['cost_dev']
            time = row.iloc[0]['time']
            time_dev = row.iloc[0]['time_dev']
            
            # Format numbers
            if cost > 10000:
                cost_str = format_sci(cost, cost_dev)
            else:
                cost_str = f"{cost:.2f} $\pm$ {cost_dev:.2f}"
                
            time_str = format_val(time, time_dev)
            
            # Make best bold (simple heuristic: lowest cost in group)
            # But "Ours" methods might be highlighted anyway
            # For now just clean formatting.
            
            m_name = get_method_name(m)
            
            # Merged alpha cell
            if first:
                alpha_cell = f"\\multirow{{6}}{{*}}{{{alpha}}}"
                first = False
            else:
                alpha_cell = ""
                
            # Line
            line = f"{m_name} & {alpha_cell} & {cost_str} & {time_str} \\\\"
            latex.append(line)
        latex.append("\\hline")

    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")
    return "\n".join(latex)

def generate_table_fixed_alpha(filename, dataset_name, label_prefix):
    df = pd.read_csv(filename)
    df = df[df['dataset'] == dataset_name]
    
    ks = sorted(df['k'].unique())
    alpha_val = df['alpha'].iloc[0]
    
    latex = []
    latex.append("\\begin{table}[!htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{So sánh các thuật toán trên tập dữ liệu {dataset_name.upper()} với $\\alpha = {alpha_val}$ và các $k$ khác nhau}}")
    latex.append(f"\\label{{tab:{dataset_name}_alpha_{alpha_val}}}")
    latex.append("\\footnotesize")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\hline")
    latex.append("Phương pháp & $k$ & Chi phí & Thời gian (s) \\\\ \\hline")
    
    for k in ks:
        sub = df[df['k'] == k]
        
        methods = ["kmeans++", "Algo1", "Det", "Ours", "Ours1", "Ours2"]
        
        first = True
        for m in methods:
            row = sub[sub['method'] == m]
            if row.empty: continue
            
            cost = row.iloc[0]['cost']
            cost_dev = row.iloc[0]['cost_dev']
            time = row.iloc[0]['time']
            time_dev = row.iloc[0]['time_dev']
            
            if cost > 10000:
                cost_str = format_sci(cost, cost_dev)
            else:
                cost_str = f"{cost:.2f} $\pm$ {cost_dev:.2f}"
            time_str = format_val(time, time_dev)
            
            m_name = get_method_name(m)
            
            if first:
                k_cell = f"\\multirow{{6}}{{*}}{{{k}}}"
                first = False
            else:
                k_cell = ""
                
            line = f"{m_name} & {k_cell} & {cost_str} & {time_str} \\\\"
            latex.append(line)
        latex.append("\\hline")

    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")
    return "\n".join(latex)

# Generate
t1 = generate_table_fixed_k("d:/Lab3_DataMining/fixedk.csv", "mnist", "mnist")
t2 = generate_table_fixed_k("d:/Lab3_DataMining/fixedk.csv", "phy", "phy")

t3 = generate_table_fixed_alpha("d:/Lab3_DataMining/fixedalpha.csv", "mnist", "mnist")
t4 = generate_table_fixed_alpha("d:/Lab3_DataMining/fixedalpha.csv", "phy", "phy")

# Write to file
with open("d:/Lab3_DataMining/generated_tables.tex", "w", encoding="utf-8") as f:
    f.write(t1 + "\n\n" + t2 + "\n\n" + t3 + "\n\n" + t4)
