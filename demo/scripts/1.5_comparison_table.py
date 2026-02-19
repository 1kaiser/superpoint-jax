
# Auto-generated from benchmark_real_data.py
import time
from pandas import DataFrame
from tabulate import tabulate

results = [{'Implementation': 'LightGlue PyTorch', 'Avg Matches': '183.4', 'Avg Time (ms)': '3041.31', 'Std Time (ms)': '1250.86', 'Min Time (ms)': '2030.22', 'Max Time (ms)': '5274.84'}, {'Implementation': 'LightGlue JAX', 'Avg Matches': '171.0', 'Avg Time (ms)': '9083.23', 'Std Time (ms)': '5383.65', 'Min Time (ms)': '3207.36', 'Max Time (ms)': '17602.91'}, {'Implementation': 'SuperGlue JAX', 'Avg Matches': '119.4', 'Avg Time (ms)': '7836.59', 'Std Time (ms)': '2640.01', 'Min Time (ms)': '4903.92', 'Max Time (ms)': '11279.98'}]
df = DataFrame(results)
print("\nComparison Table (Averaged over random pairs):")
print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False))
