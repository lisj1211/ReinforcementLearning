import argparse
from collections import namedtuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()
        self.fc = nn.Linear(state_dim, 100)
        self.mu_head = nn.Linear(100, 1)
        self.sigma_head = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.fc(x))
        mu = 2.0 * torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 8)
        self.state_value = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO:
    def __init__(self, config, state_dim):
        super(PPO, self).__init__()
        self.clip_param = config.clip_param
        self.max_grad_norm = config.max_grad_norm
        self.ppo_epoch = config.ppo_epoch
        self.buffer_capacity = config.buffer_capacity
        self.batch_size = config.batch_size
        self.gamma = config.gamma

        self.actor_net = Actor(state_dim)
        self.critic_net = Critic(state_dim)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        with torch.no_grad():
            mu, sigma = self.actor_net(state)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(-2, 2)  # 限制在Pendulum游戏的动作范围内
        return action.detach().numpy(), action_log_prob.detach().numpy()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            target_v = reward + self.gamma * self.critic_net(next_state)

        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                mu, sigma = self.actor_net(state[index])
                n = Normal(mu, sigma)
                action_log_prob = n.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])

                weighted_probs = ratio * advantage[index]
                weighted_clip_probs = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]


def main():
    parser = argparse.ArgumentParser(description='Solve the Pendulum-v1 with PPO')
    parser.add_argument('--render', atype=bool, default=False, help='render the environment')
    parser.add_argument('--episodes', atype=int, default=1000, help='train episodes')
    parser.add_argument('--ep_len', atype=int, default=200, help='number of play steps in each episode')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--clip_param', type=float, default=0.2, help='important sampling weight factor')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='clip grad norm factor')
    parser.add_argument('--ppo_epoch', type=int, default=10, help='update ppo epoch')
    parser.add_argument('--buffer_capacity', type=int, default=1000, help='buffer capacity')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--log_interval', type=int, default=10, help='interval between training status logs')
    args = parser.parse_args()

    env = gym.make('Pendulum-v1', render_mode='human').unwrapped
    state_dim = env.observation_space.shape[0]

    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])

    agent = PPO(args, state_dim)

    training_records = []
    for ep in range(args.episodes):
        score = 0
        state, _ = env.reset()
        for t in range(args.ep_len):
            if args.render:
                env.render()
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            trans = Transition(state, action, (reward + 8) / 8, action_log_prob, next_state)
            if agent.store_transition(trans):
                agent.update()
            score += reward
            state = next_state

        training_records.append(score)
        if (ep + 1) % args.log_interval == 0:
            print(f"Episode：{ep + 1}/{args.train_eps}，reward：{score:.2f}")


if __name__ == '__main__':
    main()
