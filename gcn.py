import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np

# Define the GCN Model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCNModel, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(GCNConv(input_dim, hidden_dim[0]))
        for i in range(len(hidden_dim) - 1):
            self.conv.append(GCNConv(hidden_dim[i], hidden_dim[i + 1]))
        self.conv.append(GCNConv(hidden_dim[-1], output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.conv:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return F.log_softmax(x, dim=1)

class ServerGCN(nn.Module):
    def __init__(self,numberofservices, hidden_dim, output_dim, dropout=0.5):
        super(ServerGCN, self).__init__()
        input_dim = 4 + numberofservices
        self.gcn = GCNModel(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)  # Extract features with GCN
        return x  # Outputs for each node
    
    def generate_server_features(self,graph,numberofservices):
        node_features = []
        for node, data in graph.nodes(data=True):
            server = data['server']
            services_provided = [1 if service in server.services else 0 for service in range(1,numberofservices+1)]
            features = [
                server.frequency,      # CPU frequency
                server.pos[0],         # X position
                server.pos[1],         # Y position
                len(server.task_queue),  # Length of the task queue
            ] + services_provided
            node_features.append(features)
        return torch.tensor(node_features, dtype=torch.float)

    def generate_edge_index(self,graph):
        edge_index = []
        for u, v in graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])  # Add both directions for undirected graph
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def train_multiuser_gcn_dqn(
    model, optimizers, replay_buffers, node_features, edge_index, gamma, epochs, epsilon
):
    """
    Train the shared GCN and multiple DQNs jointly.

    Args:
        model: MultiUserGCN_DQN model combining GCN and DQNs.
        optimizers: List of optimizers (one for each user's DQN).
        replay_buffers: List of replay buffers (one for each user).
        node_features: Node feature tensor (N x F).
        edge_index: Edge index tensor (2 x E).
        gamma: Discount factor for Q-learning.
        epochs: Number of training epochs.
        epsilon: Exploration rate for epsilon-greedy action selection.
    """
    model.train()  # Ensure model is in training mode

    for epoch in range(epochs):
        total_loss = 0

        # Shared GCN Forward Pass
        gcn_output, q_values_all_users = model(node_features, edge_index)

        for user_id, replay_buffer in enumerate(replay_buffers):
            optimizer = optimizers[user_id]
            dqn = model.dqns[user_id]

            # Sample transitions from the user's replay buffer
            if len(replay_buffer) < batch_size:
                continue
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert to tensors
            states = torch.stack([gcn_output[i] for i in states])  # Use GCN output as state
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.stack([gcn_output[i] for i in next_states])  # Next states from GCN
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute Q-values for the current state
            q_values_user = dqn(states)

            # Compute Q-values for the next state
            with torch.no_grad():
                q_values_next = dqn(next_states)

            # Bellman equation for target Q-values
            q_target = rewards + gamma * (1 - dones) * q_values_next.max(dim=1)[0]

            # Select Q-values corresponding to chosen actions
            q_values_selected = q_values_user.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute Q-learning loss
            loss = F.mse_loss(q_values_selected, q_target)

            # Backpropagation and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Total Loss: {total_loss:.4f}")


import torch
import torch.nn as nn

class TaskGCN(nn.Module):
    def __init__(self, numberofservices, hidden_dim, output_dim, dropout=0.5):
        super(TaskGCN, self).__init__()
        input_dim = 4 + numberofservices  # Number of features per task node
        self.gcn = GCNModel(input_dim, hidden_dim, output_dim, dropout)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)  # Extract features with GCN
        return x  # Outputs for each node




    def generate_edge_index0(self, tasks):
        edge_index = []
        # Map task_number to index in node_features list
        task_id_to_idx = {str(task.task_number): idx for idx, task in enumerate(tasks)}

        for idx, task in enumerate(tasks):
            for pred_task_number in task.predecessors:
                pred_task_number_str = str(pred_task_number)
                if pred_task_number_str in task_id_to_idx:
                    pred_idx = task_id_to_idx[pred_task_number_str]
                    # Edge from predecessor to current task
                    edge_index.append([pred_idx, idx])
                else:
                    # Handle missing predecessor if necessary
                    pass
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index
def generate_task_features(tasks, numberofservices):
    node_features = []
    for task in tasks.values():
            # Extract features from each task
            # Normalize CPU cycles and data lengths if needed
        cpu_cycle = task.cpu_cycle / 1e6  # Normalize to millions
        input_data_length = task.input_data_length  # Normalize if necessary
        num_predecessors = len(task.predecessors)
        num_successors = len(task.successors)
            # One-hot encoding of service
        service_one_hot = [0] * numberofservices
        if 1 <= task.service <= numberofservices:
            service_one_hot[task.service - 1] = 1
        else:
                # Handle invalid service ID if necessary
            pass
            # Combine features
        features = [
                cpu_cycle,
                input_data_length,
                num_predecessors,
                num_successors,
            ] + service_one_hot
        node_features.append(features)
    return torch.tensor(node_features, dtype=torch.float)
def generate_edge_index(graph):
    edge_index = []
    for u, v in graph.edges():
        edge_index.append([float(u)-1, float(v)-1])
            #edge_index.append([v, u])  # Add both directions for undirected graph
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()