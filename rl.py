# torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from environment import MEC_environment


# Model
class Model(nn.Module):
    def __init__(self, features: Sequence[int]):
        """Fully-connected Network

        Args:
            features: a list of ints like: [input_dim, 16, 16, output_dim]
        """
        super(Model, self).__init__()

        layers = []
        for i in range(len(features) - 1):
            layers.append(
                nn.Linear(
                    in_features=features[i],
                    out_features=features[i + 1],
                )
            )
            if i != len(features) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        return self.net(input)


#Building the Base Agent Class
class BaseAgent(object):
    """ The base agent class function.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        args:
            config: configuration dictionary
        """
        self.config = config

        # assert len(config['policy_layers']) > 0 # this won't allow linear models

        # environment
        self.env = gym.make(config['env_id'])
        self.gamma = config['gamma']

        # set seed
        np.random.seed(seed=config['seed'])
        self.env.seed(config['seed'])
        torch.manual_seed(config['seed'])

        # build policy model
        _policy_logits_model = Model(
            [self.env.observation_space.shape[0]] +
            config['policy_layers'] +  # note that these are only the intermediate layers
            [self.env.action_space.n],
        )
        # NOTE: by design, policy model should take *batches* of states as input.
        # self.policy_model spits out the probability of each action
        self.policy_model = nn.Sequential(
            _policy_logits_model, nn.Softmax(dim=1),
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=config['policy_learning_rate'],
        )
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)

        if config['use_baseline']:
            self.value_model = Model(
                [self.env.observation_space.shape[0]] +
                self.config['value_layers'] + [1],
            )
            self.value_optimizer = torch.optim.Adam(self.value_model.parameters(), lr=config['value_learning_rate'])

    def _make_returns(self, rewards: np.ndarray):
        """ Compute the cumulative discounted rewards at each time step

        args:
            rewards: an array of step rewards

        returns:
            returns: an array of discounted returns from that timestep onward
        """
        returns = np.zeros_like(rewards)
        returns[-1] = rewards[-1]
        for t in reversed(range(len(rewards) - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
        return returns

    # Method to implement
    def optimize_model(self, n_episodes: int) -> np.ndarray:
        """ Takes a gradient step on policy (and value) parameters using
            `n_episodes` number of episodes. You'll need to implement
            this method for each part of this problem: namely, gather a
            dataset of size `n_episodes`, approximate the gradient using
            REINFORCE, and apply it to the model parameters.

        args:
            n_episodes: number of trajectories in dataset

        returns:
            returns: the total discounted reward of each trajectory/episode.
        """

        raise NotImplementedError

    def train(self, n_episodes: int, n_iterations: int, plot: bool = True) -> Sequence[np.ndarray]:
        """ Train.
        args:
            n_episodes: number of episodes for each gradient step
            n_iterations: determine training duration
        """

        rewards = []
        for it in range(n_iterations):
            rewards.append(self.optimize_model(n_episodes))
            print(
                f'Iteration {it + 1}/{n_iterations}: rewards {round(rewards[-1].mean(), 2)} +/- {round(rewards[-1].std(), 2)}')

        if plot:
            self.plot_rewards(rewards)

        return (rewards)

    @staticmethod
    def plot_rewards(rewards: Sequence[np.ndarray], ax: Optional[Any] = None):
        # Plotting
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards[i]) for i in range(len(rewards))))),
                         columns=['Epoch', 'Reward'])
        if ax is None:
            sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');
        else:
            sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd', ax=ax);

    def evaluate(self):
        """ Evaluate and visualize a single episode.
        """

        observation = self.monitor_env.reset()
        observation = torch.tensor(observation, dtype=torch.float)[None, :]
        reward_episode = 0
        done = False

        while not done:
            probs = self.policy_model.forward(observation)
            action = torch.multinomial(probs, 1)[0]  # draw samples from dist
            observation, reward, done, info = self.monitor_env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)[None, :]
            reward_episode += reward

        self.monitor_env.close()
        show_video("./gym-results")
        print(f'Reward: {reward_episode}')


# Insert your code and run this cell
class ActorCriticAgent(BaseAgent):
    """ A2C Agent: Actor-Critic
        Here we try to FURTHER reduce the variance via bootstrapping.
    """

    def optimize_model(self, n_episodes: int):
        """ YOU NEED TO IMPLEMENT THIS METHOD

            This method is called at each training iteration and is responsible for
            (i) gathering a dataset of episodes
            (ii) computing the expectation of the policy gradient.
                 Note that you will only be computing the loss value
            In addition implement the critic network
            HINT:
                * If you've made it this far you don't need another hint!
        """
        # ======================================================================

        total_rewards = np.zeros(n_episodes, )
        observations = torch.zeros((0, 4))
        V_next = torch.zeros((0, 1))
        actions = torch.zeros((0, 1), dtype=int)
        rewards = torch.zeros((0, 1))
        for i in range(n_episodes):
            epsiode_return = 0
            episode_length = 0
            observation = self.env.reset()
            observation = torch.tensor(observation, dtype=torch.float)[None, :]
            done = False
            while not done:
                probs = self.policy_model.forward(observation)
                action = torch.multinomial(probs, 1)[0]  # draw samples from dist
                observations = torch.cat([observations, observation], 0)
                actions = torch.cat([actions, action.unsqueeze(-1)], 0)
                observation, reward, done, info = self.env.step(int(action))
                observation = torch.tensor(observation, dtype=torch.float)[None, :]
                total_rewards[i] += reward
                if done:
                    reward = 0
                    V_next = torch.cat([V_next, torch.zeros((1, 1))], 0)
                else:
                    V_next = torch.cat([V_next, self.value_model.forward(observation)], 0)
                rewards = torch.cat([rewards, torch.tensor([reward]).unsqueeze(-1)], 0)
        V = self.value_model.forward(observations)
        returns = (rewards + self.gamma * V_next).detach()
        delta = returns - V.detach()

        logprobsaction = torch.log(torch.gather(self.policy_model.forward(observations), 1, actions))

        policy_loss = -(delta * logprobsaction).mean(0)
        value_loss = ((returns - V) ** 2).sum(0)

        # ======================================================================

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return total_rewards
