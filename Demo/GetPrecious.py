"""
现有一个一维空间，宝物在该空间的最右边，寻宝人只能向左走或向右走，
通过Q-Learning训练得到最短的寻宝路径
"""
import time

import numpy as np
import pandas as pd

np.random.seed(1211)

N_STATE = 6  # the length of one-dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor
MAX_EPISODES = 15  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(n_state, actions):
    q_table = pd.DataFrame(np.zeros((n_state, len(actions))), columns=actions)
    return q_table


def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:  # random choice
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()  # greedy action
    return action_name


def get_env_feedback(state, action):
    if action == 'right':  # move right
        if state == N_STATE - 2:
            # when current state comes N_STATE - 2, next move is right, get precious and game over
            next_state = 'terminal'
            reward = 1
        else:
            next_state = state + 1
            reward = 0
    else:  # move left
        reward = 0
        if state == 0:
            next_state = state  # reach the wall
        else:
            next_state = state - 1
    return next_state, reward


def update_env(state, episode, step_counter):
    env = list('-' * (N_STATE - 1) + 'P')
    if state == 'terminal':
        interaction = f'\rEpisode {episode + 1}: total_steps = {step_counter}'
        print(interaction, end='')
        time.sleep(1)
    else:
        env[state] = 'o'
        interaction = ''.join(env)
        print('\r' + interaction, end='')
        time.sleep(FRESH_TIME)


def main():
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_count = 0
        cur_state = 0
        is_terminated = False
        update_env(cur_state, episode, step_count)
        while not is_terminated:
            action = choose_action(cur_state, q_table)
            next_state, reward = get_env_feedback(cur_state, action)
            q_predict = q_table.loc[cur_state, action]
            if next_state != 'terminal':
                q_target = reward + LAMBDA * q_table.iloc[next_state, :].max()
            else:
                q_target = reward
                is_terminated = True
            q_table.loc[cur_state, action] += ALPHA * (q_target - q_predict)
            cur_state = next_state

            step_count += 1
            update_env(cur_state, episode, step_count)
    return q_table


if __name__ == '__main__':
    q_table = main()
    print('q_table:\n', q_table)
