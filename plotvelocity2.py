import matplotlib.pyplot as plt
import os
import numpy as np
from telegram import Telegram

def read_value_txt_files(top_level_folder, search, betalist):
    data = {}  # Dictionary to store data for each tag and beta value
    for beta in betalist:
        for root, dirs, files in os.walk(top_level_folder):
            if os.path.basename(root) == os.path.basename(os.path.dirname(root)):
                continue  # Skip the top-level folder and first-level subfolders
            for sub_dir in dirs:
                value_txt_path = os.path.join(root, sub_dir, search + beta + '_', 'values.txt')
                if os.path.exists(value_txt_path):
                    with open(value_txt_path, 'r') as file:
                        content = [float(line.strip()) for line in file]
                        if content:
                            if beta not in data:
                                data[beta] = content
    return data

def moving_average(data, window_size):
    """Calculates the moving average of the data with a specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_data(data_dict1, data_dict2, max_indices, window_size, tag1, tag2):
    """Plots the moving average of data for two different tags with comparison."""
    markers = ['o', 's', '^', 'D', '*', 'p']
    colors = ['b', 'g', 'm', 'c', 'r']
    line_styles = ['-', '--', '-', '--']
    tele = Telegram()

    plt.figure(figsize=(12, 8))
    idx = 0
    for beta, data1 in data_dict1.items():
        if beta in data_dict2:
            plt.figure(figsize=(12, 8))

            data2 = data_dict2[beta]
            label1 = f'{tag1} β={beta}'
            label2 = f'{tag2} β={beta}'

            ma_data1 = moving_average(data1[:max_indices], window_size)
            ma_data2 = moving_average(data2[:max_indices], window_size)

           # marker = markers[idx % len(markers)]
           # color = colors[idx % len(colors)]
           # line_style = line_styles[idx % len(line_styles)]

            # Plot for tag1
            plt.plot(ma_data1, label=f'{label1}', marker=markers[0], color=colors[0], linestyle=line_styles[0], markersize=8, markevery=500)
            # Plot for tag2
            plt.plot(ma_data2, label=f'{label2}', marker=markers[1], color=colors[1], linestyle=line_styles[1], markersize=8, markevery=500, alpha=0.7)
            
            idx += 1

            plt.xlabel('Iteration')
            plt.ylabel('Application Finish Time')
            plt.title(f'Comparison of {tag1} and {tag2} on Application Finish Time')
            plt.legend(loc='best')
            plt.grid(True, which='both', linestyle='--', linewidth=0.7)
            plt.tight_layout()
            plt.savefig(f'plots/comparison_{tag1}_vs_{tag2}_v_{beta}.pdf', dpi=300)
            plt.savefig(f'plots/comparison_{tag1}_vs_{tag2}_v_{beta}.png')
            #plt.show()

            tele.send_photo(f'plots/comparison_{tag1}_vs_{tag2}_v_{beta}.png', f'{tag1} vs {tag2}')

def main(directory, max_indices, window_size):
    """Main function to read files from a directory and plot the moving average for comparison."""
    betalist = ['0', '10', '20', '30', '40', '70', '100', '150', '200', '300', '350', '400', '450', '500', '600', '700', '800', '900', '1000']

    tag1 = "withoutfederated_prev_servers_plus_service_per_serverDQN_v_"
    tag2 = "velocity_prev_servers_plus_service_per_serverDQN_v_"  # Replace with your second tag

    search1 = 'figs/' + tag1
    search2 = 'figs/' + tag2

    data1 = read_value_txt_files('results', search1, betalist)
    data2 = read_value_txt_files('results', search2, betalist)

    plot_data(data1, data2, max_indices, window_size, tag1, tag2)

if __name__ == "__main__":
    directory = 'values'  # Replace with your directory path
    max_indices = 500  # Set the maximum number of indices to consider
    window_size = 1  # Set the window size for the moving average
    main(directory, max_indices, window_size)
