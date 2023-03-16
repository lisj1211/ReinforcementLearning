import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value


class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY  # 获取transition要置入的行数
        self.memory[index, :] = transition  # 置入transition
        self.memory_counter += 1  # memory_counter自加1

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
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一定步数后更新target network
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        samples_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE, replace=False)
        samples = self.memory[samples_index, :]
        b_s = torch.FloatTensor(samples[:, :N_STATES])
        b_a = torch.LongTensor(samples[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(samples[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(samples[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = 0.01
    EPSILON = 0.9  # greedy policy
    GAMMA = 0.9  # reward discount
    TARGET_REPLACE_ITER = 100  # 目标网络更新频率
    MEMORY_CAPACITY = 2000  # 记忆库容量
    env = gym.make('CartPole-v1', render_mode='human').unwrapped
    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]
    EPOCHS = 400
    dqn = DQN()

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

            if dqn.memory_counter > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000，开始学习
                dqn.learn()

            if done:
                print(f'episode {i}---reward_sum: {episode_reward_sum:.2f}')
                break
