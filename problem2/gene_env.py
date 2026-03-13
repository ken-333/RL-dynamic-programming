import numpy as np

# =============================================================================
# Basic definitions
# =============================================================================

# Connectivity matrix C (4×4)
C = np.array([
    [ 0,  0, -1,  0],
    [ 1,  0, -1, -1],
    [ 0,  1,  0,  0],
    [-1,  1,  1,  0],
], dtype=float)

# All 16 states as binary vectors, shape (16, 4)
# s^1 = [0,0,0,0], s^2 = [0,0,0,1], ..., s^16 = [1,1,1,1]
# Index i corresponds to state s^{i+1} in the handout (0-indexed internally)
ALL_STATES = np.array([[int(b) for b in format(i, '04b')] for i in range(16)], dtype=float) #(16, 4) 的矩阵，每行是一个状态向量
#`format(i, '04b')` 把整数 i 转成4位二进制字符串，比如 `format(3, '04b') = '0011'`，对应状态 `[0,0,1,1]`。
# ALL_STATES[i] = binary vector for state index i
# e.g. ALL_STATES[0]  = [0,0,0,0]  ← s^1
#      ALL_STATES[15] = [1,1,1,1]  ← s^16

# All 5 actions as binary vectors, shape (5, 4)
#5个动作，每次只能扰动一个基因（或不扰动）
ALL_ACTIONS = np.array([ #5个动作，每个是4维二进制向量。a¹是什么都不做，a²～a⁵各自扰动一个基因（对应位置置1）
    [0, 0, 0, 0],  # a^1: no control
    [1, 0, 0, 0],  # a^2: perturb ATM
    [0, 1, 0, 0],  # a^3: perturb p53
    [0, 0, 1, 0],  # a^4: perturb WIP1
    [0, 0, 0, 1],  # a^5: perturb MDM2
], dtype=float)

N       = 16   # number of states
N_ACTS  = 5    # number of actions


# =============================================================================
# Helper: threshold map  v̄
#   Maps each element of a vector to 1 if > 0, else 0
# =============================================================================
def threshold(v): # 这实现了题目里的 vˉ\bar{v} vˉ 算子：大于0的元素变成1，否则变成0。注意是 严格大于0，所以0本身映射到0，负数也映射到0。
    """v̄ operator: element-wise, 1 if v[i] > 0 else 0."""
    return (v > 0).astype(float) #在 NumPy 里，对一个数组做比较运算，会返回一个布尔数组.每个元素单独判断：大于0就是 True，否则是 False.  .astype(float) 把布尔值转成浮点数, True 变成 1.0，False 变成 0.0。最终返回一个和输入 v 形状相同的数组，每个元素是 1.0 或 0.0，表示 vˉ 的结果。



# =============================================================================
# Build gene network MDP
# =============================================================================
def build_gene_env(p_noise):
    """
    Build the p53-MDM2 gene network MDP.

    Parameters
    ----------
    p_noise : float
        Bernoulli noise parameter p for each gene component.

    Returns
    -------
    env : dict  (same key structure as Problem 1's build_maze_env)
        'M'            : list of 5 arrays, each shape N×N
                         M[a][i, j] = P(s'=j | s=i, action=a)
        'R_sa'         : np.ndarray, shape N×5
                         R_sa[i, a] = Σ_j M[a][i,j] * R(s=i, a, s'=j)
        'N'            : int, 16
        'all_states'   : np.ndarray (16, 4), binary state vectors
        'all_actions'  : np.ndarray (5, 4),  binary action vectors
        'state_to_idx' : dict  (tuple of state vector) → int index
        'idx_to_state' : dict  int index → state vector (np.ndarray)
    """

    # Index mappings
    s2i = {tuple(ALL_STATES[i].astype(int)): i for i in range(N)} #s2i：状态向量 → 索引，比如 s2i[(0,0,1,1)] 返回 3
    i2s = {i: ALL_STATES[i] for i in range(N)} #索引 → 状态向量，比如 i2s[3] 返回 array([0,0,1,1])

    # ------------------------------------------------------------------
    # 1. Build M(a) — five N×N transition matrices
    #
    #    For each (i, a):
    #      expected_next = v̄(C @ s^i  ⊕  a)     (noise-free prediction)
    #      For each j:
    #        diff = || s^j - expected_next ||_1   (number of differing bits)
    #        M[a][i,j] = p^diff * (1-p)^(4-diff)
    # ------------------------------------------------------------------
    M = [np.zeros((N, N)) for _ in range(N_ACTS)] # M[a] 是一个 N×N 的矩阵，表示在动作 a 下的状态转移概率。M[a][i,j] = P(s'=j | s=i, a)。

    for a_idx in range(N_ACTS): # 遍历5个动作
        a_vec = ALL_ACTIONS[a_idx]          # shape (4,)
        for i in range(N):       # 遍历16个当前状态
            s_i = ALL_STATES[i]             # shape (4,)

            # Step 1: C @ s_i  (continuous, may have negative values)
            Cs = C @ s_i                    # shape (4,)

            # Step 2: XOR with action  →  v̄(Cs) ⊕ a
            # Note: XOR is applied AFTER the threshold map
            expected = (threshold(Cs) + a_vec) % 2   # shape (4,)
            # expected[k] = (v̄(Cs)[k] + a[k]) mod 2

            # Step 3: fill row i of M[a] 对每个 j 计算转移概率
            for j in range(N):
                s_j  = ALL_STATES[j]        # shape (4,)
                diff = int(np.sum(np.abs(s_j - expected)))   # L1 distance diff 越大（需要翻转的位数越多），概率越小
                M[a_idx][i, j] = (p_noise ** diff) * ((1 - p_noise) ** (4 - diff))

    # ------------------------------------------------------------------
    # 2. Build R(s, a, s') and R_sa
    #
    #    R(s, a, s') = 5*s'(1) + 5*s'(2) + 5*s'(3) + 5*s'(4) - ||a||_1
    #                = 5 * ||s'||_1  -  ||a||_1
    #
    #    R_sa[i, a] = Σ_j  M[a][i,j] * R(s^i, a, s^j)
    # ------------------------------------------------------------------

    # Precompute gene activation reward for each next state j: 5 * ||s^j||_1
    gene_reward = 5.0 * ALL_STATES.sum(axis=1)   # shape (N,)

    # Precompute action cost for each action: -||a||_1
    action_cost = -ALL_ACTIONS.sum(axis=1)        # shape (N_ACTS,)

    R_sa = np.zeros((N, N_ACTS))
    for a_idx in range(N_ACTS):
        # R(s^i, a, s^j) = gene_reward[j] + action_cost[a_idx]
        # R_sa[i, a] = Σ_j M[a][i,j] * (gene_reward[j] + action_cost[a])
        #            = (M[a] @ gene_reward) + action_cost[a]   (broadcast over i)
        R_sa[:, a_idx] = M[a_idx] @ gene_reward + action_cost[a_idx]

    return {
        'M'            : M,
        'R_sa'         : R_sa,
        'N'            : N,
        'N_ACTS'       : N_ACTS,
        'all_states'   : ALL_STATES,
        'all_actions'  : ALL_ACTIONS,
        'state_to_idx' : s2i,
        'idx_to_state' : i2s,
    }


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    env = build_gene_env(p_noise=0.045)
    M    = env['M']
    R_sa = env['R_sa']

    print(f"N = {env['N']},  N_ACTS = {env['N_ACTS']}")
    print(f"M[0] shape : {M[0].shape}")    # (16, 16)
    print(f"R_sa shape : {R_sa.shape}")    # (16, 5)

    # Each row of every M[a] must sum to 1
    for a in range(N_ACTS):
        row_sums = M[a].sum(axis=1)
        print(f"M[{a}] row-sum range: [{row_sums.min():.6f}, {row_sums.max():.6f}]")

    # No-control (a^1) transition: verify state s^1=[0,0,0,0]
    # Noise-free: C@[0,0,0,0]=[0,0,0,0], v̄=[0,0,0,0], XOR a^1=[0,0,0,0]
    # So expected_next = [0,0,0,0] = s^1 (index 0)
    # With p=0.045, M[0][0,0] should be (1-p)^4
    expected_00 = (1 - 0.045) ** 4
    print(f"\nM[0][0,0] = {M[0][0,0]:.6f}  (expected {expected_00:.6f})")

    # Verify against handout: noise-free, no-control, s^16=[1,1,1,1]
    # C@[1,1,1,1] = [0+0-1+0, 1+0-1-1, 0+1+0+0, -1+1+1+0] = [-1,-1,1,1]
    # v̄([-1,-1,1,1]) = [0,0,1,1]  → state index for [0,0,1,1] = 3 (s^4)
    # XOR a^1=[0,0,0,0] → expected_next = [0,0,1,1]
    # With p→0, M[0][15,3] → 1
    print(f"M[0][15,3] = {M[0][15,3]:.6f}  (noise-free limit → s^16 goes to s^4=[0,0,1,1])")

    print(f"\nR_sa (no-control, a^1) for all states:")
    print(R_sa[:, 0].round(4))