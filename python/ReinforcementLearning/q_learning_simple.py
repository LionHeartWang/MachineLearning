import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']

# parameters
EPSILON = 0.9
ALPHA = 0.1
LAMDA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3

def build_q_table(n_states, ACTIONS):
    table = pd.DataFrame(
        np.zeros((n_states, len(ACTIONS))),
        columns = ACTIONS,
    )
    print table
    return table

def choose_actions(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(state, action):
    if action == 'right':
        if state == N_STATES - 2:
            new_state = 'terminal'
            reward = 1
        else:
            new_state = state + 1
            reward = 0
    else:
        reward = 0
        if state == 0:
            new_state = state
        else:
            new_state = state - 1
    return new_state, reward

def update_env(state, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print '\r{}'.format(interaction),
        time.sleep(2)
        print '\r                               ',
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print '\r{}'.format(interaction),
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, q_table)
        while not is_terminated:
            action = choose_actions(state, q_table)
            new_state, reward = get_env_feedback(state, action)
            q_predict = q_table.ix[state, action]
            if new_state != 'terminal':
                q_target = reward + LAMDA * q_table.iloc[new_state, :].max()
            else:
                q_target = reward
                is_terminated = True
            q_table.ix[state, action] += ALPHA * (q_target - q_predict)
            state = new_state

            update_env(state, episode, step_counter + 1)
            step_counter += 1
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)