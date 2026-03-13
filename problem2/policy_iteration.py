import numpy as np


def policy_iteration(env, gamma, theta):
    """
    Matrix-form Policy Iteration.

    Parameters
    ----------
    env   : dict  returned by build_gene_env()
    gamma : float discount factor
    theta : float convergence threshold (unused in exact solve, kept for API consistency)

    Returns
    -------
    V              : np.ndarray (N,)  optimal state values
    pi             : np.ndarray (N,)  optimal policy (action indices)
    n_improvements : int  number of policy improvement steps performed
    """
    M      = env['M']       # list of 5 arrays, each (N, N)
    R_sa   = env['R_sa']    # (N, 5)
    N      = env['N']       # 16
    N_ACTS = env['N_ACTS']  # 5

    # ------------------------------------------------------------------
    # 1. Initialization
    #    Initial policy: action=0 (a^1, no control) for all states
    # ------------------------------------------------------------------
    pi = np.full(N, 0, dtype=int)   # use action a¹ in all states as the initial policy
    V  = np.zeros(N)

    n_improvements = 0  # 记录策略改进次数

    #评估当前策略 → 改进策略 → 评估新策略 → 改进策略 → ... → 策略不变 → 停止
    while True:  #不断循环，直到策略不再变化才 break
        # --------------------------------------------------------------
        # 2. Policy Evaluation  (exact matrix solve)
        #
        #    Construct M(π): N×N matrix where row i = M[pi[i]][i, :]
        #    Construct R^π : N   vector where element i = R_sa[i, pi[i]]
        #
        #    Solve: V^π = (I - γ M(π))^{-1} R^π
        # --------------------------------------------------------------
        M_pi = np.array([M[pi[i]][i, :] for i in range(N)])  # (N, N)，每行 i 从对应动作的 M 矩阵中取出第 i 行，组成新的矩阵 M_pi    M_pi 就是把"每个状态该用哪一行"的信息合并成一个矩阵，才能传给求解器。
        R_pi = R_sa[np.arange(N), pi]                         # (N,)   每行 i 从 R_sa 中取出第 i 行对应动作 pi[i] 的奖励，组成新的向量 R_pi
        #R_sa[np.arange(N), pi] 是一种高级索引方式，np.arange(N) 生成一个从 0 到 N-1 的数组，pi 是一个长度为 N 的数组，表示每个状态对应的动作索引。通过 R_sa[np.arange(N), pi]，我们可以同时获取每个状态对应动作的奖励值，得到一个长度为 N 的奖励向量 R_pi。


        V = np.linalg.solve(np.eye(N) - gamma * M_pi, R_pi) #V是一个长度为 N 的向量，表示在当前策略下每个状态的值函数。np.eye(N) 生成一个 N×N 的单位矩阵，gamma * M_pi 是一个 N×N 的矩阵，表示折扣后的转移概率矩阵。np.linalg.solve() 函数求解线性方程组 (I - γ M(π)) V = R^π，得到 V 的值。

        # --------------------------------------------------------------
        # 3. Policy Improvement
        #
        #    Q(s, a) = R_sa[:, a] + γ · M[a] @ V   for each action a
        #    new_pi  = argmax_a Q(s, a)             row-wise
        # --------------------------------------------------------------
        Q = np.column_stack([R_sa[:, a] + gamma * M[a] @ V
                             for a in range(N_ACTS)])   # (N, N_ACTS)
        new_pi = np.argmax(Q, axis=1)              # (N,)

        # --------------------------------------------------------------
        # 4. Check convergence
        #    Stop when policy no longer changes
        # --------------------------------------------------------------
        if np.array_equal(new_pi, pi):
            break

        n_improvements += 1  # 策略发生了变化才计数
        pi = new_pi
        

    return V, pi, n_improvements #返回每个状态的最终的值函数 V 形状 (248,)   每个状态的最优策略 pi，形状 (248,)，值为 0/1/2/3，    以及策略改进的次数 n_improvements
