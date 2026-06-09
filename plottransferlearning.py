import matplotlib.pyplot as plt
import os
import numpy as np
from telegram import Telegram

def read_value_txt_files(top_level_folder,search):
    #search='figs/beta_prev_servers_plus_service_per_serverDQN_beta_'
    data = {}  # Dictionary to store data from each value.txt file
    betalist= ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    #betalist= ['0.0','0.1','0.4','0.6','0.9','1.0']
    for beta in betalist:
        for root, dirs, files in os.walk(top_level_folder):
            # Check if the current folder is the second-level subfolder
            if os.path.basename(root) == os.path.basename(os.path.dirname(root)):
                continue  # Skip the top-level folder and first-level subfolders
            for sub_dir in dirs:
                value_txt_path = os.path.join(root, sub_dir, search+beta+'_','values.txt')
                if os.path.exists(value_txt_path):
                    print(value_txt_path)
                    with open(value_txt_path, 'r') as file:
                        content = [float(line.strip()) for line in file]
                        if content:
                            data[beta] = content
    return data

def read_data_from_file(file_path):
    """Reads data from a file and returns it as a list of floats."""
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data

def moving_average(data, window_size):
    """Calculates the moving average of the data with a specified window size."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_data(data_dict, max_indices, window_size,tag):
    """Plots the moving average of data from multiple lists with labels and enhanced styling."""
    markers = ['o', 's', '^', 'D', '*', 'p']
    colors = ['b', 'g', 'm', 'c', 'r']
    line_styles = ['-', '--', '-', '--']
    
    plt.figure(figsize=(12, 8))
    idx=0
    for label,data in data_dict.items():
        print(label)
        label = 'β = '+label
        ma_data = moving_average(data[:max_indices], window_size)
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        line_style = line_styles[idx % len(line_styles)]
        
        plt.plot(ma_data, label=f'{label}', 
                 marker=marker, color=color, linestyle=line_style, markersize=8, markevery=500)
        idx=idx+1
    plt.xlabel('Itreation')
    plt.ylabel('Application Finish Time')
    plt.title('Impact of β on Application Finish Time')
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig('trasnferlearning_'+tag+'.pdf',dpi=300)
    plt.savefig('trasnferlearning_'+tag+'.png')
    plt.show()
    tele = Telegram()
    tele.send_photo('trasnferlearning_'+tag+'.png',tag)

def main(directory, max_indices, window_size):
    """Main function to read files from a directory and plot the moving average of the data."""
    data_list = []
    labels = []
    #tag = 'beta_prev_servers_plus_service_per_serverDQN_beta_'
    tag = 'tanhallv0_prev_servers_plus_service_per_serverDQN_beta_'
    #tag = 'default_prev_servers_plus_service_per_serverDQN_beta_'
    tag = 'sigmoidv0_prev_servers_plus_service_per_serverDQN_beta_'
    tag = 'dqttanh_prev_servers_plus_service_per_serverDQN_beta_'
    tag = 'sametanhnumberofrun10_prev_servers_plus_service_per_serverDQN_beta_'
    search = 'figs/'+tag
    data = read_value_txt_files('results',search)

    plot_data(data, max_indices, window_size,tag)

if __name__ == "__main__":
    directory = 'values'  # Replace with your directory path
    max_indices = 10000  # Set the maximum number of indices to consider
    window_size = 1000  # Set the window size for the moving average
    main(directory, max_indices, window_size)
