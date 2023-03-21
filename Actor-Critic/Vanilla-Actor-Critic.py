import argparse

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt


class ActorNet(nn.Module):
    def __init__(self, num_state, num_actions):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(num_state, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 20)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(hidden))
        action = self.out(hidden)
        return F.softmax(action, dim=-1)


class CriticNet(nn.Module):
    def __init__(self, num_state):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(num_state, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 20)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state):
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(hidden))
        value = self.out(hidden)
        return value


class Actor:
    def __init__(self, num_state, num_actions, lr):
        self.num_actions = num_actions
        self.actor = ActorNet(num_state, num_actions)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def choose(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state).detach().numpy()
        action = np.random.choice(np.arange(self.num_actions), p=probs)
        return action

    def learn(self, s, a, td):
        s = torch.FloatTensor(s)
        prob = self.actor(s)
        log_prob = torch.log(prob)
        actor_loss = -log_prob[a] * td
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


class Critic:
    def __init__(self, num_state, gamma, lr):
        self.gamma = gamma
        self.critic = CriticNet(num_state)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

    def learn(self, s, r, s_):
        s = torch.FloatTensor(s)
        v = self.critic(s)
        r = torch.FloatTensor([r])
        s_ = torch.FloatTensor(s_)
        target = r + self.gamma * self.critic(s_).detach()
        loss = self.loss_func(target, v)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        advantage = (target - v).detach()
        return advantage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    env = gym.make('CartPole-v1').unwrapped

    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    actor = Actor(n_states, n_actions, args.lr)
    critic = Critic(n_states, args.gamma, args.lr)

    y = []
    for i in range(args.n_episodes):
        episode_reward = 0
        s, _ = env.reset()
        while True:
            a = actor.choose(s)
            s_, r, done, _, info = env.step(a)
            if done:
                r = -50
            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)
            s = s_
            episode_reward += r
            if done:
                break
        print('Episode {:03d} | Reward:{:.03f}'.format(i, episode_reward))
        y.append(episode_reward)

    plt.plot(range(args.n_episodes), y)
    plt.show()
