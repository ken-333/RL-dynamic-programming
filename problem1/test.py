
from maze_env import build_maze_env
from value_iteration import value_iteration
from simulate import simulate_trajectory

env = build_maze_env(p_stochastic=0.02)
V, policy, n_iters = value_iteration(env, gamma=0.99, theta=0.01)
states, rewards = simulate_trajectory(env, policy)

print('n_iters     :', n_iters)
print('Reached goal:', states[-1] == env['goal_state'])
print('Total reward:', round(sum(rewards), 2))
print('Steps taken :', len(states))

action_names = {0:'U', 1:'D', 2:'L', 3:'R'}
s2i = env['state_to_idx']
start = env['start_state']
print('pi(start)   :', action_names[policy[s2i[start]]])