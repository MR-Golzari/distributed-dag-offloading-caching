import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

def find_and_organize_json_data(folder_path):
    # Dictionary to store data by algorithm and beta
    data_by_algorithm = defaultdict(lambda: defaultdict(dict))

    # Pattern to match files with specific structure
    file_pattern = os.path.join(folder_path, "*_alg_*__numberofusers.json")
    
    # Find all files matching the pattern
    json_files = glob.glob(file_pattern)
    
    for json_file in json_files:
        # Extract algorithm name and beta from the filename
        filename = os.path.basename(json_file)
        parts = filename.split('_')
        
        try:
            beta = float(parts[1])  # Extract beta value from the second position
            algorithm_index = parts.index('alg') + 1
            algorithm = '_'.join(parts[algorithm_index:parts.index('numberofusers.json')])
        except (ValueError, IndexError):
            print(f"Skipping file with unexpected format: {filename}")
            print(parts)

            continue

        # Read JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Assuming JSON data is like {"key": value}
        for key, value in data.items():
            data_by_algorithm[algorithm][beta][int(key)] = value  # Store with integer keys

    return data_by_algorithm

def plot_data(data_by_algorithm,folder):
    # Plot each algorithm's data across different beta values
    for algorithm, beta_data in data_by_algorithm.items():
        plt.figure(figsize=(10, 6))
        
        for beta, data_dict in sorted(beta_data.items()):
            # Sort data by x-values (number of users)
            x_values = sorted(data_dict.keys())
            y_values = [data_dict[x] for x in x_values]

            # Plot for the current beta value
            plt.plot(x_values, y_values, marker='o', linestyle='-', label=f'Beta {beta}')

        plt.xlabel('Number of Users')
        plt.ylabel('Metric Value')
        plt.title(f'Performance of {algorithm} Across Different Beta Values')
        plt.legend()
        plt.grid(True)
        Path(f"plot_{folder}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"plot_{folder}/{algorithm}_{beta}.png")

folder = "run_f_10_V_20_2024_11_13"
# Usage
folder_path = "./results/"+folder+"/"  # Replace with your folder path

organized_data = find_and_organize_json_data(folder_path)
plot_data(organized_data,folder)
