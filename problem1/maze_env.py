
import numpy as np
from matplotlib.patches import Rectangle
# =============================================================================
# State Matrix  (20×20 grid, inner 18×18 maze)
# Values:  state-id 1–248  |  0 or nan → wall / border
# Coordinate system: STATE_MATRIX[row, col],  row/col in 0–19
# =============================================================================
W = np.nan  # shorthand for wall (内部墙壁)

STATE_MATRIX = np.array([
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # row 0  外圈
    [0, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,   0],  # row 1
    [0, 214, 215, 216, 217,   W, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,   0],  # row 2
    [0, 197, 198, 199, 200,   W, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,   0],  # row 3
    [0, 193, 194,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W,   W, 195, 196,   0],  # row 4
    [0, 176, 177,   W, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,   0],  # row 5
    [0, 162, 163,   W, 164, 165,   W, 166, 167,   W, 168, 169, 170, 171, 172,   W, 173, 174, 175,   0],  # row 6
    [0, 151, 152,   W, 153, 154,   W, 155, 156,   W, 157, 158,   W,   W,   W,   W, 159, 160, 161,   0],  # row 7
    [0, 136, 137, 138, 139, 140,   W, 141, 142,   W, 143, 144, 145, 146, 147,   W, 148, 149, 150,   0],  # row 8
    [0, 121, 122, 123, 124, 125,   W, 126, 127,   W, 128, 129, 130, 131, 132,   W, 133, 134, 135,   0],  # row 9
    [0,   W,   W,   W,   W, 111,   W, 112, 113,   W,   W, 114, 115, 116, 117,   W, 118, 119, 120,   0],  # row 10
    [0,  99, 100, 101, 102, 103,   W, 104, 105, 106,   W, 107, 108,   W, 109,   W,   W,   W, 110,   0],  # row 11
    [0,  89,  90,   W,   W,   W,   W,   W,  91,  92,   W,  93,  94,   W,  95,  96,  97,   W,  98,   0],  # row 12
    [0,  75,  76,  77,  78,  79,  80,   W,  81,  82,   W,  83,  84,   W,  85,  86,  87,   W,  88,   0],  # row 13
    [0,  60,  61,  62,  63,  64,  65,   W,  66,  67,   W,  68,  69,   W,  70,  71,  72,  73,  74,   0],  # row 14
    [0,  47,  48,  49,  50,  51,  52,   W,  53,  54,  55,  56,  57,   W,   W,   W,   W,  58,  59,   0],  # row 15
    [0,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,   0],  # row 16
    [0,   W,   W,  19,  20,  21,  22,   W,   W,   W,   W,   W,   W,  23,  24,  25,  26,  27,  28,   0],  # row 17
    [0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,   0],  # row 18
    [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # row 19 外圈
], dtype=float)

# =============================================================================
# Special cells  (row, col) — maze-internal coordinates, row/col in 1–18
# =============================================================================

# Bump cells → reward component  −10  when *entered*
BUMP_CELLS = [
    (1,11),(1,12),
    (2,1),(2,2),(2,3),
    (5,1),(5,9),(5,17),
    (6,17),
    (7,2),(7,10),(7,11),(7,17),
    (8,17),
    (12,11),(12,12),
    (14,1),(14,2),
    (15,17),(15,18),
    (16,7),
]

# Oil cells → reward component  −5  when *entered*
OIL_CELLS = [
    (2,8),(2,16),
    (4,2),
    (5,6),
    (10,18),
    (15,10),
    (16,10),
    (17,14),(17,17),
    (18,7),
]

START_CELL = (15, 4)   # state id = 50
GOAL_CELL  = (3, 13)   # state id = 208

# =============================================================================
# Actions:  0=Up, 1=Down, 2=Left, 3=Right
# =============================================================================
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  #action=0（Up）→ dr=-1, dc=0（行号减1，列号不变）。
ACTION_NAMES = {0: 'U', 1: 'D', 2: 'L', 3: 'R'}

# Perpendicular action pairs  (for stochastic drift)
PERPENDICULAR = {
    0: [2, 3],   # Up    ↔ Left, Right
    1: [2, 3],   # Down  ↔ Left, Right
    2: [0, 1],   # Left  ↔ Up,   Down
    3: [0, 1],   # Right ↔ Up,   Down
}

# =============================================================================
# Low-level helpers
# =============================================================================

def rc_to_state(row, col):  #坐标转状态编号 #坐标 → 状态编号。如果是墙壁或边框返回 None。
    """(row, col) → state id, or None if wall/border."""
    val = STATE_MATRIX[row, col]
    if np.isnan(val) or val == 0:
        return None  # 墙壁 → 没有状态编号  
    return int(val)  # 有效格子 → 返回状态编号  有效格子的值就是状态编号（例如 208），转成整数后返回。STATE_MATRIX 是 float 类型，所以需要 int() 转换。

def state_to_rc(state_id):  #状态编号转坐标 状态编号 → 坐标。用 np.argwhere 在矩阵里搜索。
    """state id → (row, col)."""
    pos = np.argwhere(STATE_MATRIX == state_id) #np.argwhere 扫描整个矩阵，返回所有值等于 state_id 的格子坐标，结果是一个二维数组，例如 [[3, 13]]。
    if len(pos) == 0:
        return None
    return tuple(pos[0]) #pos[0] 取出第一个匹配结果（形如 array([3, 13])），tuple() 转换成 (3, 13) 格式返回。由于每个状态编号在矩阵里是唯一的，取 [0] 即可。

def next_cell(row, col, action): #执行动作后的下一个格子  给定当前坐标和动作，返回移动后的坐标。如果撞墙或越界，返回原坐标（原地不动）。
    """
    Apply *action* from (row, col).
    If the resulting cell is a wall or border the agent stays put.
    Returns (new_row, new_col).
    """
    dr, dc = ACTIONS[action]   # 取出位移
    nr, nc = row + dr, col + dc  # 计算移动后的新坐标：当前行/列 + 位移量。
    if nr < 1 or nr > 18 or nc < 1 or nc > 18:  # 超出边界？ 边界检查：迷宫有效区域是第 1~18 行、第 1~18 列，外圈（第 0 行/列、第 19 行/列）是边框。越界则原地不动，返回原坐标。
        return row, col                           # border → stay  # → 停在原地
    
    val = STATE_MATRIX[nr, nc]         #新坐标在边界内，但还要检查是否是内部墙壁（nan）。如果是墙，同样原地不动。
    if np.isnan(val) or val == 0:               # 撞墙？
        return row, col                           # wall   → stay  # → 停在原地
    return nr, nc                           # 正常移动

# =============================================================================
# Build state sets (called once and cached inside build_maze_env)
# =============================================================================

def build_state_sets():
    """
    Returns
    -------
    all_states  : sorted list of valid state ids  [1 … 248]
    bump_states : set of bump state ids
    oil_states  : set of oil  state ids
    start_state : state id of start cell
    goal_state  : state id of goal  cell
    """
    all_states = sorted(    #收集所有有效状态  sorted(...)将收集到的所有状态编号从小到大排序 【1, 2, ..., 248】
        int(STATE_MATRIX[r, c])
        for r in range(1, 19) for c in range(1, 19)
        if not np.isnan(STATE_MATRIX[r, c]) and STATE_MATRIX[r, c] != 0
    )
    bump_states = {rc_to_state(r, c) for (r, c) in BUMP_CELLS # 收集 bump 格子的状态编号
                   if rc_to_state(r, c) is not None} 
    oil_states  = {rc_to_state(r, c) for (r, c) in OIL_CELLS # 收集 oil 格子的状态编号
                   if rc_to_state(r, c) is not None}
    start_state = rc_to_state(*START_CELL) #*START_CELL 是解包 
    goal_state  = rc_to_state(*GOAL_CELL) 
    return all_states, bump_states, oil_states, start_state, goal_state

# =============================================================================
# Core builder — produces M(a) and R_s^a in TA matrix form
# =============================================================================

def build_maze_env(p_stochastic, bump_penalty=-10):
    """
    Build the maze MDP in the matrix form specified by the TA.

    Parameters
    ----------
    p_stochastic : float
        Stochastic drift probability (called *p* in the project handout).
    bump_penalty : float, optional
        Reward component for entering a bump cell. Default -10.
        Set to -50 for Problem 4.

    Returns  (all keys described below)
    -------
    env : dict
        'M'            : list of 4 arrays, each shape N×N
                         M[a][i, j] = P(s'=j | s=i, action=a)
        'R_sa'         : np.ndarray, shape N×4
                         R_sa[i, a] = Σ_{j} M[a][i,j] * R(s=i, a, s'=j)
                         This is the R_s^a vector used in TA's pseudo-code.
        'R_full'       : np.ndarray, shape N×4×N
                         R_full[i, a, j] = R(s=i, a, s'=j)
                         (kept for trajectory simulation)
        'N'            : int, number of valid states (248)
        'all_states'   : list[int], sorted state ids
        'state_to_idx' : dict[int, int],  state id  → matrix row/col index
        'idx_to_state' : dict[int, int],  matrix index → state id
        'start_state'  : int, state id of start cell
        'goal_state'   : int, state id of goal  cell
        'bump_states'  : set[int]
        'oil_states'   : set[int]
    """
    all_states, bump_states, oil_states, start_state, goal_state = build_state_sets()  #调用之前写好的函数，一次性获取所有基础数据

    N      = len(all_states)                                  # 248
    s2i    = {s: i for i, s in enumerate(all_states)}        # state id → idx  enumerate 给列表里的每个元素自动配一个从0开始的序号
    i2s    = {i: s for i, s in enumerate(all_states)}        # idx → state id

    # ------------------------------------------------------------------
    # 1.  Build M(a)  —  four N×N transition matrices
    # ------------------------------------------------------------------
    M = [np.zeros((N, N)) for _ in range(4)]     # M[a][i, j] 的含义：执行动作 a，从状态下标 i 转移到状态下标 j 的概率
    #创建 4 个全零的 248×248 矩阵，对应 4 个动作,每个矩阵的行/列索引对应状态编号在 all_states 中的下标。我们会填充这些矩阵来表示 MDP 的转移概率。
    for s in all_states:
        si       = s2i[s] #状态编号 → 矩阵下标
        row, col = state_to_rc(s) #状态编号 → 坐标

        # Goal is absorbing: stays there with probability 1 处理终点
        if s == goal_state:
            for a in range(4):
                M[a][si, si] = 1.0
            continue # 终点状态的转移概率：无论执行哪个动作，都以概率 1 留在原地（吸收状态）。continue 我们直接设置 M[a][si, si] = 1.0 来表示这一点，然后跳过后续的转移构建。

        for a in range(4):
            # --- main direction: prob (1 − p) ---
            mr, mc = next_cell(row, col, a)
            mi     = s2i[rc_to_state(mr, mc)] #调用 next_cell 得到主方向的落点坐标，再转成矩阵下标 mi。

            # --- two perpendicular directions: prob p/2 each ---
            pa1, pa2 = PERPENDICULAR[a]
            pr1, pc1 = next_cell(row, col, pa1)
            pr2, pc2 = next_cell(row, col, pa2)
            pi1 = s2i[rc_to_state(pr1, pc1)] 
            pi2 = s2i[rc_to_state(pr2, pc2)] #同理得到两个垂直方向的落点矩阵下标 pi1, pi2

            #填入概率
            M[a][si, mi]  += (1.0 - p_stochastic)  #+= 是为了保证撞墙时概率不被覆盖而是正确累加，确保每行概率之和始终等于 1。
            M[a][si, pi1] += p_stochastic / 2.0  #主方向的概率是 (1-p)，两个垂直方向的概率是 p/2。我们使用 += 来累加概率，因为在某些情况下（例如撞墙）可能会有多个路径导致同一个目标状态。
            M[a][si, pi2] += p_stochastic / 2.0  #M就是一个列表，包含4个矩阵，每个矩阵对应一个动作。

    # ------------------------------------------------------------------
    # 2.  Build R(s, a, s')  — immediate reward for each (s, a, s') triple
    #
    #     Reward structure (TA handout, page 2):
    #       −1      : taking any action (delay cost)
    #       −0.8    : hitting a wall  (agent stays in s)
    #       −5      : entering an oil cell
    #       −10     : entering a bump cell  (or bump_penalty)
    #       +200    : reaching goal
    #
    #     "Hitting a wall" means the INTENDED move (the direction of action a)
    #     results in the agent staying put.  Staying put due to a perpendicular
    #     drift also counts (the TA example R(4,R,4)=−1.8 confirms this).
    #     We handle this by checking geometry directly rather than inferring
    #     from s'==s (which would double-count when multiple directions bounce).
    # ------------------------------------------------------------------
    R_full = np.zeros((N, 4, N))    # R_full[si, a, sj]

    for s in all_states:
        si       = s2i[s] #状态编号 → 矩阵下标
        row, col = state_to_rc(s)

        if s == goal_state:
            # Absorbing state: no further reward
            continue

        for a in range(4):
            # Determine whether the *main* direction hits a wall
            mr, mc        = next_cell(row, col, a)
            main_hit_wall = (mr == row and mc == col) #如果主方向的落点坐标和原坐标相同，说明主方向撞墙了，main_hit_wall 就是 True；否则就是 False。

            # Determine whether each *perpendicular* direction hits a wall
            pa1, pa2      = PERPENDICULAR[a]  # 取出两个垂直动作
            pr1, pc1      = next_cell(row, col, pa1)
            pr2, pc2      = next_cell(row, col, pa2)
            perp1_hit     = (pr1 == row and pc1 == col)  # 垂直方向1是否撞墙 Ture/False
            perp2_hit     = (pr2 == row and pc2 == col)  # 垂直方向2是否撞墙

            # Map destination cells to state indices
            mi  = s2i[rc_to_state(mr,  mc)]
            pi1 = s2i[rc_to_state(pr1, pc1)]
            pi2 = s2i[rc_to_state(pr2, pc2)] #我们已经计算了主方向和两个垂直方向的落点坐标，现在将它们转换成状态编号，再转换成矩阵下标 mi, pi1, pi2。

            # --- Assign R(s, a, s') for each reachable s' ---
            # We iterate over unique destination indices with their
            # "hit-wall" flags.
            destinations = [
                (mi,  main_hit_wall), # 主方向落点，是否撞墙
                (pi1, perp1_hit),   # 垂直方向1落点，是否撞墙
                (pi2, perp2_hit),   # 垂直方向2落点，是否撞墙
            ]
            #计算每个落点的奖励
            for (sj, hit_wall) in destinations:
                sp   = i2s[sj]     # 矩阵下标 → 状态编号
                base = -1.0                        # action cost (always)

                if hit_wall:
                    base += -0.8                   # wall penalty

                if sp in oil_states:
                    base += -5.0

                if sp in bump_states:
                    base += bump_penalty           # default −10

                if sp == goal_state:
                    base += 200.0

                # Multiple (a, perp) combinations can map to the same sj;
                # take the value from whichever path was used.
                # (They give the same reward since reward depends on s'
                #  and whether a wall was hit, not on which direction drifted.)
                R_full[si, a, sj] = base

    # ------------------------------------------------------------------
    # 3.  Build R_s^a  (TA notation)
    #     R_sa[i, a] = Σ_j  M[a][i, j] * R_full[i, a, j]
    #     This is the N×1 vector the PI / VI pseudo-code multiplies by.
    # ------------------------------------------------------------------
    R_sa = np.zeros((N, 4)) #创建一个 248×4 的全零矩阵，准备填入每个状态×每个动作的期望奖励。
    for a in range(4):
        # element-wise product M[a] ⊙ R_full[:,a,:]  then sum over j
        R_sa[:, a] = np.sum(M[a] * R_full[:, a, :], axis=1) #R_sa[:, a]取出第a列的所有元素        右边算出一个 (248,) 的向量，直接整列写入 R_sa 的第 a 列，比逐行赋值高效得多。
        #M[a] 形状 (248, 248)，概率矩阵   R_full[:, a, :] 形状 (248, 248)，奖励矩阵 是从三维矩阵里**切出固定动作a的一层**

    return {
        'M'            : M,            # list of 4  N×N matrices  ← TA format
        'R_sa'         : R_sa,         # N×4  expected reward      ← TA format
        'R_full'       : R_full,       # N×4×N raw reward (for simulation)
        'N'            : N,
        'all_states'   : all_states,
        'state_to_idx' : s2i,
        'idx_to_state' : i2s,
        'start_state'  : start_state,
        'goal_state'   : goal_state,
        'bump_states'  : bump_states,
        'oil_states'   : oil_states,
    }


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    all_states, bump_states, oil_states, start_state, goal_state = \
        build_state_sets()

    print(f"Total states : {len(all_states)}")    # expected: 248
    print(f"Bump states  : {len(bump_states)}")   # expected: 21
    print(f"Oil  states  : {len(oil_states)}")    # expected: 10
    print(f"Start state  : {start_state}")        # expected: 50
    print(f"Goal  state  : {goal_state}")         # expected: 208

    env = build_maze_env(p_stochastic=0.02)

    M    = env['M'] #M 是一个列表，包含4个矩阵，每个矩阵对应一个动作，矩阵的行/列索引对应状态编号在 all_states 中的下标。M[a][i, j] 的含义：执行动作 a，从状态下标 i 转移到状态下标 j 的概率。
    R_sa = env['R_sa']

    print(f"\nM[0] shape   : {M[0].shape}")       # (248, 248) 0=Up, 1=Down, 2=Left, 3=Right
    print(f"R_sa shape   : {R_sa.shape}")          # (248, 4)

    # Each row of every M[a] must sum to 1
    for a in range(4):
        row_sums = M[a].sum(axis=1)
        print(f"M[{a}] row-sum range: "
              f"[{row_sums.min():.6f}, {row_sums.max():.6f}]")  # both ≈ 1

    # Verify start / goal positions
    print(f"\nStart ({start_state}) @ {state_to_rc(start_state)}")
    print(f"Goal  ({goal_state})  @ {state_to_rc(goal_state)}")

    # Example: reproduce TA reward examples from the handout (small grid)
    # R(4, L, 3) = -6  →  check conceptually with the actual maze
    si = env['state_to_idx'][start_state]
    print(f"\nR_sa[start, Left] = {R_sa[si, 2]:.4f}")


