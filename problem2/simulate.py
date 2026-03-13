import numpy as np


def simulate_episode(env, policy, steps=150, init_state_idx=None):
    """
    Sample a single length-150 trajectory under the given policy.

    Parameters
    ----------
    env            : dict  returned by build_gene_env()
    policy         : np.ndarray (N,)  action index for each state (0-indexed)
    steps          : int  trajectory length (default 150)
    init_state_idx : int or None
                     If None, sample initial state uniformly from {0,...,15}.

    Returns
    -------
    states  : list[int]   state indices visited, length = steps+1
                          (includes s_0, s_1, ..., s_150)
    activation : np.ndarray (steps,)
                 ||s_k||_1 at each step k=1,...,steps
                 (activation of s_0 is not counted per handout formula)
    """
    M          = env['M']  #从环境字典里取出需要的东西：转移矩阵、状态向量表、状态数量。
    all_states = env['all_states']   # (16, 4)
    N          = env['N']

    # Initial state
    if init_state_idx is None: #如果调用函数时没有指定初始状态，那么就从0到15中随机选择一个整数作为初始状态索引。np.random.randint(N) 会返回一个在 [0, N) 范围内的随机整数，这里就是 [0, 16) 的整数，也就是 0 到 15 之间的整数，代表16个状态中的一个。
        curr = np.random.randint(N)   # uniform over {0,...,15}     "In episode i, start with a random initial state s₀ ∈ {s¹,...,s¹⁶}"，所以默认随机选初始状态。
    else:
        curr = init_state_idx #如果调用函数时指定了初始状态索引，那么就直接使用这个索引作为初始状态。

    states     = [curr]
    activation = np.zeros(steps)

    for k in range(steps):       #k 从0到149，共150次，对应题目轨迹里的150个时间步
        a        = policy[curr]                              # action index 比如 policy[curr] = 2 就代表在当前状态 curr 下执行动作 a^3（扰动 p53）
        next_idx = np.random.choice(N, p=M[a][curr, :])     # sample s_{k+1}  从0～15中按给定概率随机抽一个数

        activation[k] = np.sum(all_states[next_idx])        # ||s_{k+1}||_1    all_states[next_idx] 取出下一状态的向量，比如 [1,0,1,1]。 np.sum() 数有几个1，也就是 ||s_{k+1}||_1。 存入activation[k]。记录的是下一状态，不记录 s0s_0 s0​，对应题目公式(4)从 k=1 开始。
        #activation 是一个长度150的数组，初始全是0
        states.append(next_idx) 
        curr = next_idx #把新状态加入轨迹列表，然后更新 curr 为新状态，准备下一轮循环。

    return states, activation #返回整个轨迹的状态索引列表和对应的激活值数组。states 包含 s0s_0 到 s150s_{150} 共151个状态索引，activation 包含 s1 到 s150s_{150} 共150个激活值，用于后续计算 Ai


def compute_AvgA(env, policy, n_episodes=75, steps=150):
    """
    Compute AvgA: average gene activation rate over n_episodes episodes.

    Each episode starts from a uniformly random initial state.

    Parameters
    ----------
    env        : dict  returned by build_gene_env()
    policy     : np.ndarray (N,)
    n_episodes : int   number of episodes (default 75)
    steps      : int   trajectory length  (default 150)

    Returns
    -------
    AvgA : float   in [0, 4]
    """
    A_list = [] #用来收集每个episode的 Ai最后对它求平均。
    for _ in range(n_episodes): #循环75次
        _, activation = simulate_episode(env, policy, steps=steps) #跑一条轨迹
        A_i = activation.mean()    # (1/150) * Σ_{k=1}^{150} ||s_k||_1
        A_list.append(A_i)

    return float(np.mean(A_list))  # (1/75) * Σ A^i


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    from gene_env import build_gene_env

    env = build_gene_env(p_noise=0.045)
    N   = env['N']

    # No-control policy: always take a^1 (action index 0)
    no_control_policy = np.zeros(N, dtype=int)

    avga_nc = compute_AvgA(env, no_control_policy)
    print(f"AvgA (no-control, p=0.045) = {avga_nc:.4f}")
    # Expected: close to 0 (system converges to s^1=[0,0,0,0] without control)

    # Single episode inspection
    states, activation = simulate_episode(env, no_control_policy, steps=150)
    print(f"First 10 states (no-control): {states[:10]}")
    print(f"Mean activation (no-control): {activation.mean():.4f}")