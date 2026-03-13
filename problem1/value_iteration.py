import numpy as np


def value_iteration(env, gamma, theta):
    """
    Matrix-form Value Iteration.

    Parameters
    ----------
    env   : dict  returned by build_maze_env()
    gamma : float discount factor
    theta : float convergence threshold

    Returns
    -------
    V          : np.ndarray (N,)  optimal state values
    pi         : np.ndarray (N,)  optimal policy (action indices)
    n_iters    : int  number of iterations performed
    """
    M    = env['M']       # list of 4 arrays, each (N, N) 取出4个转移矩阵，供后续计算Q值用
    R_sa = env['R_sa']    # (N, 4)  取出期望奖励矩阵，供后续计算Q值用
    N    = env['N']       # 248  取出状态数量248，供初始化数组用

    # ------------------------------------------------------------------
    # 1. Initialization
    #    V = 0 for all states
    # ------------------------------------------------------------------
    V = np.zeros(N)
    n_iters = 0

    while True:
        # --------------------------------------------------------------
        # 2. Value Iteration Backup
        #
        #    Q(s, a) = R_sa[:, a] + γ · M[a] @ V   for each action a
        #    V_new   = max_a Q(s, a)                row-wise max
        # --------------------------------------------------------------
        Q = np.column_stack([R_sa[:, a] + gamma * M[a] @ V
                             for a in range(4)])   # (N, 4)

        V_new = np.max(Q, axis=1)                  # (N,)

        n_iters += 1

        # --------------------------------------------------------------
        # 3. Check convergence
        #    Stop when max change across all states < theta
        # --------------------------------------------------------------
        if np.max(np.abs(V_new - V)) < theta: #如果新旧值函数之间的最大差异小于收敛阈值 theta，说明值函数已经收敛了，可以停止迭代。
            V = V_new
            break

        V = V_new

    # ------------------------------------------------------------------
    # 4. Extract optimal policy
    #    π*(s) = argmax_a Q(s, a)  using converged V
    # ------------------------------------------------------------------
    Q  = np.column_stack([R_sa[:, a] + gamma * M[a] @ V
                          for a in range(4)])
    pi = np.argmax(Q, axis=1)     #axis=1 在每一行里找最大值的位置                 # (N,)

    return V, pi, n_iters #返回最终的值函数 V、最优策略 pi 和迭代次数 n_iters


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    from maze_env import build_maze_env

    env = build_maze_env(p_stochastic=0.02)
    V, pi, n_iters = value_iteration(env, gamma=0.99, theta=0.01)

    s2i   = env['state_to_idx']
    start = env['start_state']
    goal  = env['goal_state']

    print(f"Value Iteration converged in {n_iters} iterations")
    print(f"V(start={start}) = {V[s2i[start]]:.4f}")
    print(f"V(goal ={goal})  = {V[s2i[goal]]:.4f}")

    action_names = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
    print(f"π(start) = {action_names[pi[s2i[start]]]}")