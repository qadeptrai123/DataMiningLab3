
import os

tex_file = r"d:\Lab3_DataMining\content\section\06_exp.tex"
new_content_file = r"d:\Lab3_DataMining\generated_tables_v2.tex"

with open(new_content_file, "r", encoding="utf-8") as f:
    new_lines = f.readlines()

with open(tex_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Lines to replace: 180 to 457 (1-based)
# Python index: 179 to 456 (inclusive)
# Slice: 179:457

start_idx = 179
end_idx = 457

# Check if the lines look right (start with \begin{table}, end with \end{table})
print(f"Replacing lines {start_idx+1} to {end_idx}:")
print(f"Start line content: {lines[start_idx].strip()}")
print(f"End line content: {lines[end_idx-1].strip()}")

if "\\begin{table}" not in lines[start_idx] and "\\centering" not in lines[start_idx]: # My previous view showed \begin{table} at 180
    print("Warning: Start line might not be \\begin{table}")

if "\\end{table}" not in lines[end_idx-1]:
    print("Warning: End line might not be \\end{table}")

# Perform replacement
final_lines = lines[:start_idx] + new_lines + lines[end_idx:]

with open(tex_file, "w", encoding="utf-8") as f:
    f.writelines(final_lines)

print("Successfully updated 06_exp.tex")
