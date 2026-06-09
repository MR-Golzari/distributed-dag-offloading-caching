import matplotlib.pyplot as plt
import numpy as np
def read_data_from_file(file_path):
    """Reads data from a file and returns it as a list of floats."""
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data
data = read_data_from_file('results/2024_09_05__08_55_09/figs/default_prev_servers_plus_service_per_serverDQN_/values.txt')

moving_averages = {}
arr = data
window_size =200
for k in range(len(arr)):
         # Calculate the average of current window
    if (k<window_size):
        window_average = np.sum(arr[0:k+1])/(k+1)
    else:
        window_average = np.sum(arr[k-window_size:k]) / window_size
    moving_averages[k]=window_average


plt.figure()
plt.plot(range(len(data)), moving_averages.values(), '-',label='Algorithm')
#plt.plot(range(len(data)), [0.02]*len(data), '--',label='Optimal')
plt.ylim([0,.15])
plt.xlabel('Iteration')
plt.ylabel('Application finish time')
plt.title('Algorithm Convergence Over Iterations')
plt.grid()
#plt.legend()
plt.savefig('converge.pdf',dpi=300)
plt.show()
