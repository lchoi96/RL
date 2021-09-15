import numpy as np
import torch
import random
import collections


class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension,
                                       out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100,
                                            out_features=output_dimension)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


class Agent:

    # Function to initialise the agent
    def __init__(self):
        self.episode_length = 650
        self.num_steps_taken = 0
        self.state = None
        self.action = None

        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(),
                                          lr=0.002)
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.buffer = ReplayBuffer()
        self.num_episode = 1
        self.reached_flag = False
        self.dist_to_goal = np.inf
        self.greedy_test = False
        self.epsilon = 1
        self.dist_threshold = 0

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.reached_flag:
            if self.num_steps_taken % self.episode_length == 0:
                self.num_episode += 1
                self.num_steps_taken = 0
                return True
            return False

        if self.greedy_test and self.num_steps_taken <= 100 \
                and self.dist_to_goal < 0.03:  # successful greedy policy
            self.reached_flag = True
            self.num_steps_taken = 0
            self.num_episode += 1
            self.episode_length = 101
            return True
        elif self.greedy_test and self.num_steps_taken == 100:  # failed greedy policy
            self.num_episode += 1
            self.num_steps_taken = 0
            self.dist_to_goal = np.inf
            self.episode_length = max(round(self.episode_length * 0.94), 350)
            self.epsilon = min(1.2 * self.epsilon, 0.75)
            return True
        elif self.num_steps_taken % self.episode_length == 0:
            self.num_episode += 1
            self.num_steps_taken = 0
            self.dist_threshold = 0
            if self.num_episode >= 8:  # decay epsilon 8th episode onwards
                self.epsilon = max(0.97 * self.epsilon, 0.12)
            else:
                self.epsilon = 0.97
            return True
        return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):

        self.greedy_test = self.greedy_test_check()  # check if it's time for greedy policy

        if not self.greedy_test and self.num_episode % 4 == 0 \
                and self.num_steps_taken == 0:  # update weights
            self.update_network()

        if self.reached_flag or self.greedy_test:
            state_q_values = self.get_q_value(state)
            action = np.argmax(state_q_values)
        else:
            state_q_values = self.get_q_value(state)
            p = random.random()
            if p < self.epsilon:
                action = random.choices(range(0, 4),
                                        weights=[0.35, 0.5, 0.02, 0.5], k=1)[0]
            else:
                action = np.argmax(state_q_values)
        self.num_steps_taken += 1
        self.state = state
        self.action = action
        return self.discrete_action_to_continuous(action)

    def discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0.02, 0], dtype=np.float32)  # right
        elif discrete_action == 1:
            continuous_action = np.array([0, 0.02], dtype=np.float32)  # up
        elif discrete_action == 2:
            continuous_action = np.array([-0.02, 0], dtype=np.float32)  # left
        elif discrete_action == 3:
            continuous_action = np.array([0, -0.02], dtype=np.float32)  # down
        return continuous_action

    def set_epsilon(self, distance_to_goal):
        min_epsilon = 0.12
        steps = 100
        epsilon_delta = (self.epsilon - min_epsilon)/steps
        dist_increment = 0.1
        if self.epsilon > min_epsilon and distance_to_goal > self.dist_threshold:
            self.epsilon -= epsilon_delta
            self.dist_threshold += dist_increment

    # Function to set the next state and distance,
    # which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        self.dist_to_goal = distance_to_goal
        self.set_epsilon(distance_to_goal)
        comparison = self.state == next_state
        penalty = 0
        if comparison.all():  # penalty for hitting a wall
            penalty = 0.05
        penalty_factor = 2 ** (-0.25 * self.num_episode + 0.5) + 1
        x_penalty = (1.2 - next_state[0]) * penalty_factor
        reward = 0.8 * (1 - distance_to_goal) - penalty - x_penalty
        transition = (self.state, self.action, reward, next_state)
        self.buffer.container_append(transition)
        if self.buffer.num_samples > 500 and not self.reached_flag and not self.greedy_test:
            sample_list = self.buffer.sample(500)
            loss = self.train_q_network(sample_list)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        state_q_values = self.get_q_value(state)
        action = np.argmax(state_q_values)
        action = self.discrete_action_to_continuous(action)
        return action

    def greedy_test_check(self):
        if self.num_episode < 10:  # no greedy policy for first 10 episodes
            return False
        elif self.num_episode <= 15:
            return self.num_episode % 5 == 0
        return self.num_episode % 3 == 0

    def train_q_network(self, transition):
        self.optimiser.zero_grad()
        loss = self._calculate_loss(transition)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    # def set_lr(self, rate):
    #     self.optimiser.param_groups[0]['lr'] = max(rate, 0.001)

    def update_network(self):
        torch.save(self.q_network.state_dict(), 'weights_only.pth')
        self.target_network.load_state_dict(torch.load('weights_only.pth'))

    def _calculate_loss(self, transition):
        state = [t[0] for t in transition]
        discrete_action = [t[1] for t in transition]
        reward = [t[2] for t in transition]
        next_state = [t[3] for t in transition]

        state = torch.tensor(state).float()
        reward = torch.tensor(reward)
        discrete_action = torch.tensor(discrete_action)
        discrete_action = torch.unsqueeze(discrete_action, 1)
        next_state = torch.tensor(next_state).float()
        # Target network to calculate action with max q value of the next state
        state_q_values = self.target_network(next_state)
        state_q_values = state_q_values.detach()
        action_tensor = torch.argmax(state_q_values, dim=1)
        # calculate q values of current state
        q = self.q_network(state)
        q = torch.gather(q, 1, discrete_action)
        # calculate q values of next state using action_tensor
        second_q = self.q_network(next_state)
        second_q = torch.gather(second_q, 1,
                                action_tensor.unsqueeze(-1)).squeeze(-1)
        loss_mse = torch.nn.MSELoss()(q.squeeze(-1).float(),
                                      (reward + 0.9 * second_q).float())
        return loss_mse

    def get_q_value(self, state):
        state = torch.tensor(state).float()
        q = self.q_network(state)
        return q.detach().numpy()


class ReplayBuffer:

    def __init__(self):
        self.buffer = collections.deque(maxlen=10000)
        self.num_samples = 0

    def container_append(self, t):
        self.buffer.append(t)
        self.num_samples += 1

    def sample(self, n):
        return random.choices(self.buffer, k=n)
