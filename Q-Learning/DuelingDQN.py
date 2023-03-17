import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


class DuelNet(nn.Module):
    def __init__(self):
        super(DuelNet, self).__init__()
        self.hidden = nn.Linear(N_STATES, 50)

        self.advantage = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, N_ACTIONS)
        )

        self.value = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, x):
        x = F.relu(self.hidden(x))
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()


class DuelingDQN:
    def __init__(self):
        self.eval_net, self.target_net = DuelNet(), DuelNet()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_function = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]

        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def get_reward(self, state):
        pos, vel, ang, avel = state

        pos1 = 2.0
        ang1 = math.pi / 6

        r1 = 5 - 10 * abs(pos / pos1)
        r2 = 5 - 10 * abs(ang / ang1)

        r1 = max(r1, -5)
        r2 = max(r2, -5)

        return r1 + r2

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        samples = self.memory[sample_index, :]
        b_s = torch.FloatTensor(samples[:, :N_STATES])
        b_a = torch.LongTensor(samples[:, N_STATES:N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(samples[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(samples[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_eval_next = torch.max(self.eval_net(b_s_), 1)[1].view(BATCH_SIZE, 1)
        q_next = self.target_net(b_s_).detach()
        y_batch = b_r + GAMMA * q_next.gather(1, q_eval_next)
        loss = self.loss_function(q_eval, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = 0.001
    EPSILON = 0.9
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 100
    MEMORY_CAPACITY = 2000
    env = gym.make('CartPole-v1', render_mode='human').unwrapped
    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]
    EPOCHS = 400
    dqn = DuelingDQN()

    for i in range(EPOCHS):
        print(F'Episode: {i}')
        s, _ = env.reset()
        episode_reward_sum = 0

        while True:
            env.render()
            a = dqn.choose_action(s)
            s_, r, done, _, info = env.step(a)

            pos, vel, a, a_vel = s_  # position, velocity, angle, angular velocity
            reward = dqn.get_reward(s_)

            dqn.store_transition(s, a, reward, s_)
            episode_reward_sum += reward

            s = s_

            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()

            if done:
                print(f'episode {i}---reward_sum: {episode_reward_sum:.2f}')
                break
