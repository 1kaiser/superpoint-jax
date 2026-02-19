import nbformat as nbf
import os
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("# SuperPoint, SuperGlue, and LightGlue: PyTorch vs JAX Comparison"))

    # Section 1: Setup and Imports
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup and Imports"))
    with open('demo/scripts/1.1_setup_imports.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read()))

    # Section 2: Load Models
    cells.append(nbf.v4.new_markdown_cell("## 2. Load Models and Convert Weights"))
    with open('demo/scripts/1.2_load_models.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read()))

    # Section 3: Helper Functions
    cells.append(nbf.v4.new_markdown_cell("## 3. Helper Functions for Inference and Visualization"))
    with open('demo/scripts/1.3_helper_functions.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read()))

    # Section 4: Run Comparison
    cells.append(nbf.v4.new_markdown_cell("## 4. Run Comparison on Synthetic or Real Data"))
    with open('demo/scripts/1.4_run_comparison.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read()))

    # Section 5: Comparison Table
    cells.append(nbf.v4.new_markdown_cell("## 5. Comparison Table"))
    with open('demo/scripts/1.5_comparison_table.py', 'r') as f:
        cells.append(nbf.v4.new_code_cell(f.read()))

    nb['cells'] = cells

    output_path = 'demo/lightglue_jax_comparison.ipynb'
    with open(output_path, 'w') as f:
        nbf.write(nb, f)
    print(f"Notebook generated at {output_path}")

if __name__ == '__main__':
    if not os.path.exists('demo/scripts'):
        print("Error: demo/scripts/ directory not found. Please create the modular scripts first.")
    else:
        create_notebook()
