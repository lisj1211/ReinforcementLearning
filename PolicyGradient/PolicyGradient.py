import gym
import numpy as np
import math

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib import animation


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


# define Policy Gradient
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


# define some functions
def print_red(string):
    print('\033[0;31m', end='')
    print(string, end='')
    print('\033[0m')


def save_gif(frames, filename):
    figure = plt.imshow(frames[0])
    plt.axis('off')

    # callback function
    def animate(i):
        figure.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save(filename, writer='pillow', fps=30)


if __name__ == '__main__':
    # create cartpole model
    env = gym.make('CartPole-v1', render_mode='human')

    # reset state of env
    state, _ = env.reset()

    # crate Policy Gradient model
    model = PolicyGradient()

    # step of learning
    learn_step = 0

    # flag of train ok
    train_ok = False
    episode = 0

    # play and train
    while not train_ok:
        state, _ = env.reset()

        play_step = 0
        total_rewards = 0

        episode += 1
        print(f'\nEpisode {episode} ...')

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

    # train ok, save model
    save_file = 'policy_gradient.ptl'
    torch.save(model, save_file)
    print_red(f'\nmodel trained ok, saved to {save_file}')

    # close env
    env.close()
