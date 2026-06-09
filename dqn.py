import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_units, num_actions):
        super(MyModel, self).__init__()
        layers = []
        in_dim = input_dim
        for index, hidden_dim in enumerate(hidden_units):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())  # Activation function
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
class DQN_Agent:
    def __init__(self,num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr,epsilon,maximum_exploration):
        self.TrainNet =  DQN(num_states,num_actions,hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        self.TargetNet = DQN(num_states,num_actions,hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        self.epsilon = 0.1
        self.decay = epsilon**(1.0/maximum_exploration)
        self.min_epsilon = epsilon
    def observe(self,observation):
        exp = {'s': observation[0], 'a': observation[1], 'r': observation[2], 's2': observation[3], 'done': observation[4]}
        self.TrainNet.add_experience(exp)
    def replay(self,gcn_model=None,gcn_optimizer=None,embedding=None):
        self.TrainNet.train(self.TargetNet)
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)

class GCN_DQN_Agent:
    def __init__(self,num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr,epsilon,maximum_exploration):
        self.TrainNet =  DQN(num_states,num_actions,hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        self.TargetNet = DQN(num_states,num_actions,hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        self.epsilon = 0.1
        self.decay = epsilon**(1.0/maximum_exploration)
        self.min_epsilon = epsilon
    def observe(self,observation):
        exp = {'s': observation[0], 'a': observation[1], 'r': observation[2], 's2': observation[3], 'done': observation[4]}
        self.TrainNet.add_experience(exp)
    def replay(self,gcn_model,gcn_optimizer):
        self.TrainNet.train2(gcn_model,gcn_optimizer)
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)
class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': [],'node_features': [],'edge_index': [],'embedding_index': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
    def predict(self, inputs):
        #return self.model(np.atleast_2d(inputs.astype('float32')))
        return self.model(inputs)

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        
        # Extract batch data
        states = torch.stack([torch.tensor(self.experience['s'][i], dtype=torch.float32, device=self.device)  for i in ids])
        actions = [self.experience['a'][i] for i in ids]
        rewards = [self.experience['r'][i] for i in ids]
        next_states = torch.stack([torch.tensor(self.experience['s2'][i], dtype=torch.float32, device=self.device)  for i in ids])
        dones = [self.experience['done'][i] for i in ids]
        # Compute predicted Q-values for next states
        #value_next = TargetNet.predict(next_states).max(1)[0]
        #actual_values = rewards + self.gamma * value_next * (1 - dones)
        # Compute predicted Q-values for current states
        q_values = self.predict(states)

                # Convert actions, rewards, dones to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        selected_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        actual_values = rewards
        # Compute loss
        loss = F.mse_loss(selected_action_values, actual_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    def train2(self, gcn_model,gcn_optimizer):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)

        # Extract batch data
        actions = [self.experience['a'][i] for i in ids]
        rewards = [self.experience['r'][i] for i in ids]
        next_states = [self.experience['s2'][i] for i in ids]
        dones = [self.experience['done'][i] for i in ids]
        node_features_batch = [self.experience['node_features'][i] for i in ids]
        edge_index_batch = [self.experience['edge_index'][i] for i in ids]
        embedding_index_batch = [self.experience['embedding_index'][i] for i in ids]

        # Process states through GCN
        state_embeddings = []
        next_state_embeddings = []
        for nf, ei, eindex in zip(node_features_batch, edge_index_batch, embedding_index_batch):
            nf = nf.to(self.device)
            ei = ei.to(self.device)
            embedding = gcn_model(nf, ei)
            state_embeddings.append(embedding[eindex])
        state_embeddings = torch.stack(state_embeddings)

        # Convert actions, rewards, dones to tensors
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute predicted Q-values
        q_values_all = self.predict(state_embeddings)
        selected_action_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = F.mse_loss(selected_action_values, rewards)

        # Backpropagation
        self.optimizer.zero_grad()
        gcn_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        gcn_optimizer.step()

        return loss.item()

    def train1(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        # Extract batch data
        states = [self.experience['s'][i] for i in ids]
        actions = [self.experience['a'][i] for i in ids]
        rewards = [self.experience['r'][i] for i in ids]
        next_states = [self.experience['s2'][i] for i in ids]
        dones = [self.experience['done'][i] for i in ids]
        node_features_batch = [self.experience['node_features'][i] for i in ids]
        edge_index_batch = [self.experience['edge_index'][i] for i in ids]
            # Convert to tensors
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss
    def get_action(self, states):
        return np.argmax(self.predict(states).detach().numpy())


    def add_experience(self, exp):
        #print(f"Experience added. New experience buffer size: {len(self.experience['s'])}")
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        self.model.load_state_dict(TrainNet.model.state_dict())