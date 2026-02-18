import subprocess
import os

def run_notebook(notebook_path):
    output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    print(f"Executing notebook: {notebook_path}")
    try:
        subprocess.run([
            'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
            notebook_path, '--output', os.path.basename(output_path)
        ], check=True)
        print(f"Notebook executed successfully. Output saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing notebook: {e}")

if __name__ == '__main__':
    run_notebook('demo/superpoint_superglue_demo.ipynb')
