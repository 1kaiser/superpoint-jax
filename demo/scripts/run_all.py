import os
import subprocess
import sys

scripts = [
    'demo/scripts/1.1_setup_imports.py',
    'demo/scripts/1.2_load_models.py',
    'demo/scripts/1.3_helper_functions.py',
    'demo/scripts/1.4_run_comparison.py'
]

# We need to run them together or maintain state.
# For verification in a script, it's better to have one combined script or run them in sequence in the same process.

combined_code = ""
for script in scripts:
    with open(script, 'r') as f:
        combined_code += f.read() + "\n"

# Execute the combined code
# We need to set __file__ for the first script if it uses it
namespace = {'__file__': 'demo/scripts/run_all.py'}
exec(combined_code, namespace)
