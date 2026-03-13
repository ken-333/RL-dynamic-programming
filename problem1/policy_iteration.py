import numpy as np


def policy_iteration(env, gamma, theta):
    """
    Matrix-form Policy Iteration.

    Parameters
    ----------
    env   : dict  returned by build_maze_env()
    gamma : float discount factor
    theta : float convergence threshold (unused in exact solve, kept for API consistency)

    Returns
    -------
    V              : np.ndarray (N,)  optimal state values
    pi             : np.ndarray (N,)  optimal policy (action indices)
    n_improvements : int  number of policy improvement steps performed
    """
    M    = env['M']       # list of 4 arrays, each (N, N)
    R_sa = env['R_sa']    # (N, 4)
    N    = env['N']       # 248

    # ------------------------------------------------------------------
    # 1. Initialization
    #    Initial policy: action=2 (Left) for all states
    # ------------------------------------------------------------------
    pi = np.full(N, 2, dtype=int)   # all Left  创建一个长度248的数组，全部填2（代表"左"）
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
        Q = np.column_stack([R_sa[:, a] + gamma * M[a] @ V  # R_sa[:, a]→ 每个状态执行动作a的即时奖励  M[a] @ V→ 每个状态执行动作a后，下一步的期望价值   column_stack → 把4个动作的结果拼成(248,4)矩阵
                             for a in range(4)])   # (N, 4) #对每个动作 a，计算 Q(s, a) = R_sa[:, a] + γ · M[a] @ V，得到一个 (N, 4) 的矩阵 Q，其中每列对应一个动作，每行对应一个状态的 Q 值。
        #Q[i, a] = 状态i执行动作a的总期望回报
        new_pi = np.argmax(Q, axis=1)              # (N,)，对每个状态 i，找到使 Q(s, a) 最大的动作 a 的索引，得到新的策略 new_pi。np.argmax(Q, axis=1) 会返回一个长度为 N 的数组，其中每个元素是对应行（状态）的最大值所在的列索引（动作索引）。 axis=1 表示按行操作，返回每行最大值的列索引。

        # --------------------------------------------------------------
        # 4. Check convergence
        #    Stop when policy no longer changes
        # --------------------------------------------------------------
        if np.array_equal(new_pi, pi):  #如果新策略 new_pi 和旧策略 pi 完全相同，说明策略已经收敛，不再改进了，可以停止迭代。
            break #如果策略没有改变，说明已经收敛了，退出循环

        n_improvements += 1 #策略发生了变化，改进次数加1
        pi = new_pi #更新策略为新策略，继续下一轮的评估和改进
        

    return V, pi, n_improvements #返回每个状态的最终的值函数 V 形状 (248,)   每个状态的最优策略 pi，形状 (248,)，值为 0/1/2/3，    以及策略改进的次数 n_improvements


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    from maze_env import build_maze_env

    env = build_maze_env(p_stochastic=0.02)
    V, pi, n_imp = policy_iteration(env, gamma=0.99, theta=0.01)

    s2i = env['state_to_idx']
    start = env['start_state']
    goal  = env['goal_state']

    print(f"Policy Iteration converged in {n_imp} improvement steps")
    print(f"V(start={start}) = {V[s2i[start]]:.4f}")
    print(f"V(goal ={goal})  = {V[s2i[goal]]:.4f}")

    action_names = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}
    print(f"π(start) = {action_names[pi[s2i[start]]]}")