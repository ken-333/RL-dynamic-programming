import numpy as np


def simulate_trajectory(env, policy, max_steps=400):
    """
    Sample a single trajectory from start_state under the given policy.

    Parameters
    ----------
    env      : dict  returned by build_maze_env()
    policy   : np.ndarray (N,)  action index for each state
    max_steps: int  maximum number of steps (T_max)

    Returns
    -------
    states  : list[int]  sequence of state ids visited (length ≤ max_steps+1)
    rewards : list[float]  rewards received at each step (length ≤ max_steps)
    """
    M        = env['M']
    R_full   = env['R_full']
    s2i      = env['state_to_idx']
    i2s      = env['idx_to_state']
    start    = env['start_state']
    goal     = env['goal_state']
    N        = env['N']

    # 从起点出发
    curr_id  = start # 当前状态编号，从起点50开始
    curr_idx = s2i[curr_id] # 转成矩阵下标 49

    states  = [curr_id] # 记录访问过的状态编号，初始状态已经访问了，所以先把起点编号放进去
    rewards = []   # 记录每步获得的奖励，初始为空

    for _ in range(max_steps): #最多走 max_steps 步，_ 表示不需要用到循环变量
        if curr_id == goal:  #到达终点就停止，不再继续走。
            break

        a = policy[curr_idx]                          # 查策略，得到当前状态应该执行的动作编号

        # 按转移概率采样下一个状态
        next_idx = np.random.choice(N, p=M[a][curr_idx, :]) # 按转移概率采样下一个状态   M[a][curr_idx, :]  → 当前状态执行动作a后，到各状态的概率，形状(248,)
        next_id  = i2s[next_idx]

        r = R_full[curr_idx, a, next_idx]             # 即时奖励

        states.append(next_id) # 把新状态编号加入轨迹列表
        rewards.append(r)  # 把这步奖励加入奖励列表

        curr_id  = next_id  
        curr_idx = next_idx  

    return states, rewards #返回访问过的状态编号列表和对应的奖励列表


def compute_cumulative_rewards(rewards):
    """
    Compute cumulative reward G(t) = sum of rewards from step 0 to t.

    Parameters
    ----------
    rewards : list[float]

    Returns
    -------
    G : np.ndarray  cumulative rewards at each step
    """
    return np.cumsum(rewards)  #cumsum = **cumulative sum**，累计求和。每个位置的值等于**从头加到这里**的总和


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    from maze_env import build_maze_env
    from value_iteration import value_iteration

    env = build_maze_env(p_stochastic=0.02)
    _, policy, _ = value_iteration(env, gamma=0.99, theta=0.01)

    states, rewards = simulate_trajectory(env, policy, max_steps=400)

    print(f"Trajectory length : {len(states)} states")
    print(f"Reached goal      : {states[-1] == env['goal_state']}")
    print(f"Total reward      : {sum(rewards):.2f}")
    print(f"First 10 states   : {states[:10]}")