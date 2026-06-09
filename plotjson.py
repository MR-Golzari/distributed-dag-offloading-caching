import os
import json
import glob
from collections import defaultdict
import matplotlib.pyplot as plt

def find_and_organize_json_data(folder_path):
    # Dictionary to store data by algorithm
    data_by_algorithm = defaultdict(dict)

    # Pattern to match files with specific structure
    file_pattern = os.path.join(folder_path, "v_*_alg_*__numberofusers.json")
    
    # Find all files matching the pattern
    json_files = glob.glob(file_pattern)
    
    for json_file in json_files:
        # Extract algorithm name from the filename
        filename = os.path.basename(json_file)
        parts = filename.split('_')
        
        try:
            algorithm_index = parts.index('alg') + 1
            algorithm = '_'.join(parts[algorithm_index:parts.index('numberofusers.json')])
        except (ValueError, IndexError):
            print(f"Skipping file with unexpected format: {filename}")
            continue

        # Read JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Assuming JSON data is like {"key": value}
        for key, value in data.items():
            data_by_algorithm[algorithm][int(key)] = value  # Store with integer keys

    return data_by_algorithm

def plot_data(data_by_algorithm,folder):
    plt.figure(figsize=(10, 6))

    # Plot each algorithm's data
    for algorithm, data_dict in data_by_algorithm.items():
        # Sort data by x-values (number of users)
        x_values = sorted(data_dict.keys())
        y_values = [data_dict[x] for x in x_values]

        # Plot for the current algorithm
        plt.plot(x_values, y_values, marker='o', linestyle='-', label=algorithm)

    plt.xlabel('Number of Users')
    plt.ylabel('Metric Value')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(folder +'.pdf',dpi=300)

# Usage
folder = "run_2_10_V_200_2024_11_09"
folder_path = "./results/run2av200_2024_11_08"  # Replace with your folder path
folder_path = "./results/"+folder  # Replace with your folder path

organized_data = find_and_organize_json_data(folder_path)
plot_data(organized_data,folder)
