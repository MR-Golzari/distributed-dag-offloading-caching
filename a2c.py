import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class MyModel(nn.Module):
    def __init__(self, num_states, hidden_units, num_actions, favored_action=None, bias_value=2.0):
        super(MyModel, self).__init__()
        layers = []
        input_dim = num_states
        
        for i, hidden_dim in enumerate(hidden_units):
            layers.append(nn.Linear(input_dim, hidden_dim))
            #layers.append(nn.ReLU())
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.policy_layer = nn.Linear(input_dim, num_actions)
        self.value_layer = nn.Linear(input_dim, 1)
        
        # Bias the policy layer if a favored action is specified
        if favored_action is not None:
            with torch.no_grad():
                self.policy_layer.bias[favored_action] = bias_value
                
    def forward(self, state, temperature):
        x = self.hidden_layers(state)
        logits = self.policy_layer(x)
        policy = torch.softmax(logits / temperature, dim=-1)
        value = self.value_layer(x)
        return policy, value

class A2C:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr, favored_action=None):
        self.num_actions = num_actions
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions, favored_action=favored_action)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = {'s': [], 'a': [], 'r': [], 's2': [], 'done': [],'node_features': [],'edge_index': [],'embedding_index': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        
        self.temperature = 1.0         # initial temperature
        self.temperature_decay = 0.9995 # decay rate
        self.min_temperature = 0.01     # minimum temperature
    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.model(state, temperature=self.temperature)
    
    def get_action(self, state):
        policy_probs, _ = self.predict(state)
        return torch.multinomial(policy_probs, 1).item()
    
    def add_experience(self, exp):
        if len(self.memory['s']) >= self.max_experiences:
            for key in self.memory.keys():
                self.memory[key].pop(0)
        for key, value in exp.items():
            self.memory[key].append(value)
    
    def decay_temperature(self):
        self.temperature = max(self.min_temperature, self.temperature * self.temperature_decay)
    def train(self):
        if len(self.memory['s']) < self.min_experiences:
            return

        ids = np.random.choice(len(self.memory['s']), self.batch_size, replace=False)
        states = torch.stack([torch.tensor(self.memory['s'][i], dtype=torch.float32) for i in ids])
        actions = torch.tensor([self.memory['a'][i] for i in ids], dtype=torch.long)
        rewards = torch.tensor([self.memory['r'][i] for i in ids], dtype=torch.float32).view(-1, 1)
        states_next = torch.stack([torch.tensor(self.memory['s2'][i], dtype=torch.float32) for i in ids])
        dones = torch.tensor([self.memory['done'][i] for i in ids], dtype=torch.float32).view(-1, 1)

        # Use temperature in forward pass
        policy_probs, values = self.predict(states)
        _, next_values = self.predict(states_next)

        advantages = rewards + self.gamma * next_values * (1 - dones) - values

        log_probs = torch.log(policy_probs + 1e-10)
        action_masks = torch.nn.functional.one_hot(actions, self.num_actions)
        policy_loss = -torch.mean((log_probs * action_masks).sum(dim=1).view(-1, 1) * advantages)
        value_loss = torch.mean(advantages.pow(2))

        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.decay_temperature()
class A2C_Agent:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr, epsilon, maximum_exploration, favored_action=None):
        self.TrainNet = A2C(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr, favored_action=favored_action)
        self.epsilon = epsilon
        self.decay = epsilon ** (1.0 / maximum_exploration)
        self.min_epsilon = epsilon
    def observe(self, observation):
        exp = {'s': observation[0], 'a': observation[1], 'r': observation[2], 's2': observation[3], 'done': observation[4]}
        self.TrainNet.add_experience(exp)
    def replay(self):
        self.TrainNet.train()
    
    def update_target_model(self):
        pass
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)