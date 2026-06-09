import numpy as np
import tensorflow as tf

class MyModel(tf.keras.models.Model):
    def __init__(self, num_states, lstm_units, num_actions):
        #super(MyModel, self).__init__(inputs=[],outputs=[])
        self.input_layer = tf.keras.layers.Input(shape=(None,num_states),name="Input_Layer")
        self.lstm_layers=[]
        index = 0
        for i in lstm_units:
            self.lstm_layers.append(tf.keras.layers.LSTM(i, activation='tanh', return_sequences=True,name=f"LSTM_Layer_{index}"))
            index+=1
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='linear',name="Output_Layer")
        super(MyModel, self).__init__(inputs=self.input_layer,outputs = self.__call__(self.input_layer))

    def __call__(self, inputs):
        z = inputs
        for layer in self.lstm_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

    def predict0(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.lstm_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output

class LSTM_Agent:
    def __init__(self, num_states, num_actions, lstm_units, gamma, max_experiences, min_experiences, batch_size, lr, epsilon, maximum_exploration):
        self.TrainNet = LSTM(num_states, num_actions, lstm_units, gamma, max_experiences, min_experiences, batch_size, lr)
        self.TargetNet = LSTM(num_states, num_actions, lstm_units, gamma, max_experiences, min_experiences, batch_size, lr)
        self.epsilon = 1
        self.decay = epsilon ** (1.0 / maximum_exploration)
        self.min_epsilon = epsilon

    def greedy_actor(self, observation):
        return self.TrainNet.get_action(observation, self.epsilon)

    def observe(self, observation):
        exp = {'s': observation[0], 'a': observation[1], 'r': observation[2], 's2': observation[3], 'done': observation[4]}
        self.TrainNet.add_experience(exp)

    def replay(self):
        self.TrainNet.train(self.TargetNet)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def update_target_model(self):
        self.TargetNet.copy_weights(self.TrainNet)

class LSTM:
    def __init__(self, num_states, num_actions, lstm_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, lstm_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):

        return tf.squeeze(self.model(np.atleast_3d(inputs.astype('float32')).reshape(inputs.shape[0],1,inputs.shape[1])))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
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

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_3d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
