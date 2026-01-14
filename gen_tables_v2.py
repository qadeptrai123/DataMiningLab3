import csv
import pandas as pd
import numpy as np

def format_sci(val, dev):
    if val == 0: return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    coeff = val / (10**exponent)
    dev_coeff = dev / (10**exponent)
    return f"{coeff:.4f}E{exponent:+} $\pm$ {dev_coeff:.2f}E{exponent:+}"

def format_val(val, dev):
    return f"{val:.4f} $\pm$ {dev:.4f}"

def format_time(val, dev):
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

def highlight_best(val, dev, best_val, is_time=False, is_cost=False):
    # Determine if bolding is needed based on being the best
    # Simple logic: if val is equal to best_val (within tolerance)
    # The reference table bolds Fast-Filtering for Cost/Time, and Fast-Sampling for NMI/ARI
    # Also k-means++ for Time.
    return False 


def generate_table_content(df, dataset_name, fixed_var, fixed_val, varying_var):
    # Determine varying values
    varying_vals = sorted(df[varying_var].unique())
    
    latex = []
    # No table wrapper here
    latex.append("\\centering")
    
    caption_var = "$\\alpha$" if varying_var == "alpha" else "$k$"
    fixed_var_sym = "$\\alpha$" if fixed_var == "alpha" else "$k$"
    
    latex.append(f"\\caption{{So sánh các thuật toán trên tập dữ liệu {dataset_name.upper()} với {fixed_var_sym} = {fixed_val} và các {caption_var} khác nhau}}")
    latex.append(f"\\label{{tab:{dataset_name}_{fixed_var}_{fixed_val}}}")
    latex.append("\\footnotesize")
    latex.append("\\resizebox{0.85\\textwidth}{!}{")
    latex.append("\\begin{tabular}{lccccc}") # Removed one 'c' for Lloyd
    latex.append("\\hline")
    latex.append(f"Phương pháp & {caption_var} & Chi phí & NMI & ARI & Thời gian (s) \\\\ \\hline") # Removed Lloyd column
    
    for v in varying_vals:
        sub = df[df[varying_var] == v]
        
        methods = ["kmeans++", "Algo1", "Det", "Ours", "Ours1", "Ours2"] # Ordered
        
        # Determine bests for bolding
        m_rows = sub[sub['method'].isin(methods)]
        
        if m_rows.empty: continue

        best_cost = m_rows['cost'].min()
        best_time = m_rows['time'].min()
        best_nmi = m_rows['nmi'].max()
        best_ari = m_rows['ari'].max()
        
        first = True
        for m in methods:
            row = sub[sub['method'] == m]
            if row.empty: continue
            
            row = row.iloc[0]
            
            # Values
            cost = row['cost']
            cost_dev = row['cost_dev']
            nmi = row['nmi']
            nmi_dev = row['nmi_dev']
            ari = row['ari']
            ari_dev = row['ari_dev']
            time = row['time']
            time_dev = row['time_dev']
            
            # Format
            c_str = format_sci(cost, cost_dev)
            n_str = format_val(nmi, nmi_dev)
            a_str = format_val(ari, ari_dev)
            t_str = format_time(time, time_dev)
            
            # Bold logic (Best value)
            if cost == best_cost: c_str = f"\\textbf{{{c_str}}}"
            if nmi == best_nmi: n_str = f"\\textbf{{{n_str}}}"
            if ari == best_ari: a_str = f"\\textbf{{{a_str}}}"
            if time == best_time: t_str = f"\\textbf{{{t_str}}}"

            # Specific bolding for k-means++ time
            if m == "kmeans++" and time == best_time:
                pass 
            
            m_name = get_method_name(m)
            
            # Row construction
            if first:
                v_cell = f"\\multirow{{6}}{{*}}{{{v}}}"
                first = False
            else:
                v_cell = ""
            
            line = f"{m_name} & {v_cell} & {c_str} & {n_str} & {a_str} & {t_str} \\\\"
            latex.append(line)
        latex.append("\\hline")

    latex.append("\\end{tabular}")
    latex.append("}")
    return "\n".join(latex)

def process_file_to_list(filename, fixed_var, varying_var):
    df = pd.read_csv(filename)
    # Ensure dataset order: MNIST, PHY, USPS
    # Filter by dataset key availability
    datasets_ordered = ["mnist", "phy", "usps"]
    tables_list = []
    
    # Check what datasets are actually in the csv
    present_datasets = [d for d in datasets_ordered if d in df['dataset'].unique()]
    
    for ds in present_datasets:
        # Get fixed value
        fixed_val = df[fixed_var].iloc[0]
        tbl = generate_table_content(df[df['dataset'] == ds], ds, fixed_var, fixed_val, varying_var)
        tables_list.append(tbl)
    return tables_list

# Generate all content blocks
all_tables = []
all_tables.extend(process_file_to_list("d:/Lab3_DataMining/fixed_k.csv", "k", "alpha"))
all_tables.extend(process_file_to_list("d:/Lab3_DataMining/fixed_alpha.csv", "alpha", "k"))

# Group into pairs
final_latex = []
for i in range(0, len(all_tables), 2):
    pair_content = []
    pair_content.append("\\begin{table}[htbp]")
    
    # Table 1
    pair_content.append(all_tables[i])
    
    # Spacer
    pair_content.append("\\vspace{1em}")
    
    # Table 2 (if exists)
    if i + 1 < len(all_tables):
        pair_content.append(all_tables[i+1])
        
    pair_content.append("\\end{table}")
    final_latex.append("\n".join(pair_content))

with open("d:/Lab3_DataMining/generated_tables_v2.tex", "w", encoding="utf-8") as f:
    f.write("\n\n".join(final_latex))
