"""
Sarsa
Q(s, a) <-- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
"""
import random

import gym
import numpy as np


class SarsaQLearning:
    def __init__(self, env, num_steps_per_episode=200, alpha=0.1, gamma=0.5, epsilon=0.1):
        """
        :param env: environment
        :param num_steps_per_episode: update steps per episode
        :param gamma: discount rate
        :param alpha: learning rate
        :param epsilon: greedy search percent
        """
        self.env = env
        self.num_steps_per_episode = num_steps_per_episode
        self.state_num = self.env.observation_space.n
        self.action_num = self.env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.q = np.zeros((self.state_num, self.action_num))  # init q_table
        self.avg_returns = []

    def select_action(self, state):
        """choose action"""
        ran_num = random.random()
        if ran_num <= self.epsilon:  # randomly choose action
            action = random.randrange(self.action_num)
            return action

        else:
            action = self.argmax(state)  # greedily choose action
            return action

    def train(self, epochs):
        for epoch in range(epochs):
            self.env.render()  # show
            cur_state = self.env.reset()[0]
            a = self.select_action(cur_state)  # difference between vanilla and sarsa
            for i in range(self.num_steps_per_episode):
                next_s, reward, done, *_ = self.env.step(a)
                if done:
                    break
                next_a = self.select_action(next_s)
                self.update_q_table(cur_state, a, next_s, next_a, reward)
                cur_state = next_s
                a = next_a
            avg_return = self.evaluate()
            self.avg_returns.append(avg_return)
            print(f"epoch: {epoch}, avg_return: {avg_return}")

    def evaluate(self):
        q = self.q
        s = self.env.reset()[0]
        avg_return = 0.0
        for i in range(self.num_steps_per_episode):
            a = np.argmax(q[s])
            next_s, reward, done, *_ = self.env.step(a)
            avg_return += reward
            if done:
                break
            s = next_s
        return avg_return

    def update_q_table(self, s, a, next_s, next_a, r):
        """update q value"""
        s = s
        a = a
        next_s = next_s
        q_target = r + self.gamma * self.q[next_s][next_a]
        self.q[s][a] = self.q[s][a] + self.alpha * (q_target - self.q[s][a])

    def argmax(self, s):
        """choose max action prob of current state"""
        s = s
        if np.count_nonzero(self.q[s]) == 0:
            action = random.randrange(self.action_num)
        else:
            action = np.argmax(self.q[s])
        return action


if __name__ == '__main__':
    max_epoch = 1000
    environment = gym.make('CliffWalking-v0', render_mode='human')
    environment.reset()
    agent = SarsaQLearning(environment)
    agent.train(max_epoch)
    environment.close()
