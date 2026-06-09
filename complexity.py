# Constants
from math import log10


S = 10         # Number of cloudlets
Q = 10         # Number of services
n = 64         # Number of neurons in hidden layers
B = 1024       # Batch size
E = 1024       # Number of samples
M = 1         # Number of users
V = 10
N=10
operations_per_second = 10**9  # Assume 1 billion operations per second for CPU

# Caching Decision Algorithm Complexity
def caching_decision_complexity(S, Q):
    return Q * S + S * Q * log10(Q)  # log10(Q) ≈ 10

# DRL-based Algorithm Complexity (per user)
def drl_based_complexity(B, N, V, S, Q, n):
    """
    Calculate the complexity of the algorithm for a given task.

    Parameters:
    B (int): A parameter related to the problem.
    N (int): A parameter related to the problem.
    V (int): The size of the vertex set.
    S (int): A parameter related to the problem.
    Q (int): A parameter related to the problem.
    n (int): A parameter related to the problem.

    Returns:
    float: The calculated complexity.
    """
    term1 = B / (N * V) + 1
    term2 = (S * Q + S + 2 * Q + 3) * n + n**2 + n * S
    complexity = term1 * term2
    return complexity

# Pre-trained Neural Network Creation Complexity
def pre_trained_nn_complexity(S, Q, n, E):
    input_size = S * Q + S + 2 * Q + 3
    forward_backward_pass = input_size * n + n**2 + n * S
    return E + E * forward_backward_pass

# Time Calculation
def time_for_operations(complexity, operations_per_second):
    return complexity / operations_per_second

# Caching Decision Algorithm
caching_complexity = caching_decision_complexity(S, Q)
caching_time = time_for_operations(caching_complexity, operations_per_second)

# DRL-based Algorithm
drl_complexity = drl_based_complexity(B, N, V, S, Q, n)
drl_time = time_for_operations(drl_complexity, operations_per_second)

# Pre-trained Neural Network Creation
pre_trained_nn_complexity_value = pre_trained_nn_complexity(S, Q, n, E)
pre_trained_nn_time = time_for_operations(pre_trained_nn_complexity_value, operations_per_second)

# Print Results
print(f"Caching Decision Algorithm Complexity: {caching_complexity}")
print(f"Caching Decision Algorithm Time: {caching_time:.6f} seconds")

print(f"DRL-based Algorithm Complexity: {drl_complexity}")
print(f"DRL-based Algorithm Time: {drl_time:.6f} seconds")

print(f"Pre-trained Neural Network Creation Complexity: {pre_trained_nn_complexity_value}")
print(f"Pre-trained Neural Network Creation Time: {pre_trained_nn_time:.6f} seconds")
