import gym
import math

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 2)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = self.fc1(state)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        output = nn.functional.softmax(x)

        return output


class PolicyGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

        self.history_log_probs = []
        self.history_rewards = []

        self.gamma = 0.99

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.net(state)
        ctgr = torch.distributions.Categorical(probs)
        action = ctgr.sample()

        self.history_log_probs.append(ctgr.log_prob(action))

        return action.item()

    def choose_best_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.net(state)
        action = int(torch.argmax(probs))

        return action

    def get_reward(self, state):
        pos, vel, ang, avel = state

        pos1 = 2.0
        ang1 = math.pi / 6

        r1 = 5 - 10 * abs(pos / pos1)
        r2 = 5 - 10 * abs(ang / ang1)

        r1 = max(r1, -5)
        r2 = max(r2, -5)

        return r1 + r2

    def gg(self, state):
        pos, vel, ang, avel = state

        bad = abs(pos) > 2.0 or abs(ang) > math.pi / 4

        return bad

    def store_transition(self, reward):
        self.history_rewards.append(reward)

    def learn(self):
        # backward calculate rewards
        R = 0

        rewards = []
        for r in self.history_rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std()

        loss = 0
        for i in range(len(rewards)):
            loss += -self.history_log_probs[i] * rewards[i]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.history_log_probs.clear()
        self.history_rewards.clear()


def print_red(string):
    print('\033[0;31m', end='')
    print(string, end='')
    print('\033[0m')


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')
    state, _ = env.reset()
    model = PolicyGradient()
    learn_step = 0
    episodes = 20000

    for ep in range(episodes):
        state, _ = env.reset()

        play_step = 0
        total_rewards = 0
        print(f'\nEpisode {ep} ...')

        while True:
            env.render()

            action = model.choose_action(state)

            state, reward, done, _, info = env.step(action)
            pos, vel, a, a_vel = state  # position, velocity, angle, angular velocity

            reward = model.get_reward(state)
            if model.gg(state):
                reward += -10

            model.store_transition(reward)

            total_rewards += reward
            play_step += 1

            if play_step % 1000 == 0 or model.gg(state):
                model.learn()
                learn_step += 1
                print(f'play step {play_step} rewards {total_rewards:.2f} learn {learn_step}')

            if model.gg(state):
                break

            if play_step >= 20000:
                train_ok = True
                break
    env.close()
