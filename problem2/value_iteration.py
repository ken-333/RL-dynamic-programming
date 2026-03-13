import numpy as np


def value_iteration(env, gamma, theta):
    """
    Matrix-form Value Iteration.

    Parameters
    ----------
    env   : dict  returned by build_gene_env()
    gamma : float discount factor
    theta : float convergence threshold

    Returns
    -------
    V          : np.ndarray (N,)  optimal state values
    pi         : np.ndarray (N,)  optimal policy (action indices)
    n_iters    : int  number of iterations performed
    """
    M      = env['M']       # list of 5 arrays, each (N, N)
    R_sa   = env['R_sa']    # (N, 5)
    N      = env['N']       # 16
    N_ACTS = env['N_ACTS']  # 5

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
                             for a in range(N_ACTS)])   # (N, N_ACTS)

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
                          for a in range(N_ACTS)])
    pi = np.argmax(Q, axis=1)     #axis=1 在每一行里找最大值的位置                 # (N,)

    return V, pi, n_iters #返回最终的值函数 V、最优策略 pi 和迭代次数 n_iters


