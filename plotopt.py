import os
import glob
import matplotlib.pyplot as plt

# Function to read and parse data from a file
def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        print(filename)
        number_of_servers = float(lines[0].split(":")[1].strip())
        gurobi_value = float(lines[1].split(":")[1].strip())
        dp_value = float(lines[2].split(":")[1].strip())
    return number_of_servers, gurobi_value, dp_value

# Path to the directory containing the files
directory_path = 'results/optimization'

# Get a list of all files in the directory
file_paths = glob.glob(os.path.join(directory_path, '*.txt'))
# Lists to store the extracted data
number_of_servers_list = []
gurobi_values = []
dp_values = []



data = []

# Read data from each file
for file_path in file_paths:
    number_of_servers, gurobi_value, dp_value = read_data(file_path)
    data.append((number_of_servers, gurobi_value, dp_value))

# Sort the data based on the number of servers
data.sort()

# Separate the sorted data into individual lists
number_of_servers_list, gurobi_values, dp_values = zip(*data)

print('number_of_edge_servers =',list(number_of_servers_list))
print('gurobi_finish_times = ',list(gurobi_values))
print('dp_based_finish_times = ',list(dp_values))
