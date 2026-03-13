import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrow
import seaborn as sns

from maze_env import (STATE_MATRIX, OIL_CELLS, BUMP_CELLS,
                      START_CELL, GOAL_CELL)

ACTION_ARROWS = {0: (0, -0.3),  # Up:    dy=-0.3 (seaborn y轴翻转，向上实际是dy负)
                 1: (0, 0.3),   # Down:  dy=+0.3
                 2: (-0.3, 0),  # Left:  dx=-0.3
                 3: (0.3, 0)}   # Right: dx=+0.3


# =============================================================================
# Shared helper: color special cells
# =============================================================================
#给特殊格子上色
def coloring_blocks(ax, oil_cells, bump_cells, start_cell, goal_cell):
    for (r, c) in oil_cells:
        ax.add_patch(Rectangle((c, r), 1, 1, fill=True,
                               facecolor='red', edgecolor='red', lw=0.25)) #在ax上添加一个矩形补丁，位置是(c, r)，宽度和高度都是1，填充颜色为红色，边框颜色也为红色，边框宽度为0.25。注意这里的坐标是(c, r)而不是(r, c)，因为matplotlib的坐标系中x轴对应列索引，y轴对应行索引。
    for (r, c) in bump_cells:
        ax.add_patch(Rectangle((c, r), 1, 1, fill=True,
                               facecolor='lightsalmon', edgecolor='lightsalmon', lw=0.25))
    r, c = start_cell
    ax.add_patch(Rectangle((c, r), 1, 1, fill=True,
                           facecolor='lightblue', edgecolor='lightblue', lw=0.25))
    r, c = goal_cell
    ax.add_patch(Rectangle((c, r), 1, 1, fill=True,
                           facecolor='lightgreen', edgecolor='lightgreen', lw=0.25))


# =============================================================================
# 1. Value function heatmap  (Figure 7 style)
# =============================================================================
#价值函数热力图
def plot_value_function(V, env, title='Optimal Value Function'):
    """
    Plot V(s) as a heatmap over the maze grid.
    Wall cells are shown in black, valid cells are colored by value.
    """
    s2i      = env['state_to_idx'] #状态编号 → 矩阵下标
    all_states = env['all_states'] # [1, 2, ..., 248]

    # Build a 20×20 display matrix (nan = wall/border → black)
    display = np.full((20, 20), np.nan) #创建 20×20 的矩阵，全部填 nan，墙壁和边框默认显示黑色。
    annot_mat = np.full((20, 20), np.nan)  # 标注：存 V 值用于显示数字

    for s in all_states:
        pos = np.argwhere(STATE_MATRIX == s) #遍历每个状态，用 argwhere 找到有效格子在 STATE_MATRIX 里的坐标
        if len(pos) == 0: #防御性检查，找不到坐标就跳过（正常情况不会触发）
            continue
        r, c = pos[0]    #取出坐标，r 是行索引，c 是列索引
        display[r, c]   = 1.0          # 有效格子背景统一为白色
        annot_mat[r, c] = V[s2i[s]]   # 数值用于标注

    # ── 改动2：构建标注字符串矩阵（nan格子显示空字符串）──
    annot_str = np.full((20, 20), '', dtype=object)
    for r in range(20):
        for c in range(20):
            if not np.isnan(annot_mat[r, c]):
                annot_str[r, c] = f'{annot_mat[r, c]:.1f}'

    fig, ax = plt.subplots(figsize=(10, 7.5))

    # ── 改动3：Greys colormap + vmin/vmax 锁死白色，关闭 colorbar ──
    cmap = plt.get_cmap('Greys').copy()
    cmap.set_bad('black') #把 bad color（nan格子）设置为黑色
    sns.heatmap(display,
                linewidths=0.25, linecolor='black', #格子间的线条，黑色，宽度0.25
                cbar=False,                # ← 关闭 colorbar
                cmap=cmap, vmin=0, vmax=2, # ← 有效格子固定白色
                ax=ax,
                annot=annot_str,           # ← 显示 V 值数字
                fmt='',                    # ← 用字符串矩阵时 fmt 必须为空
                annot_kws={'size': 5})

    # ── 改动4：特殊格子高亮 ──
    coloring_blocks(ax, OIL_CELLS, BUMP_CELLS, START_CELL, GOAL_CELL)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. Optimal policy plot  (Figure 8 style)
# =============================================================================

def plot_policy(policy, env, title='Optimal Policy'):
    """
    Plot the optimal policy as arrows on the maze grid.
    """
    s2i      = env['state_to_idx']
    all_states = env['all_states'] #取出状态编号→下标映射和所有状态列表

    fig, ax = plt.subplots(figsize=(10, 7.5))
    # Base maze
    display = np.full((20, 20), np.nan)
    for r in range(1, 19):
        for c in range(1, 19):
            val = STATE_MATRIX[r, c]
            if not np.isnan(val) and val != 0:
                display[r, c] = 1.0

    cmap = plt.get_cmap('Greys').copy()
    cmap.set_bad('black')
    sns.heatmap(display, linewidths=0.25, linecolor='black',
                cbar=False, cmap=cmap, vmin=0, vmax=2, ax=ax)
    
    coloring_blocks(ax, OIL_CELLS, BUMP_CELLS, START_CELL, GOAL_CELL)
    # Draw arrows
    for s in all_states:
        pos = np.argwhere(STATE_MATRIX == s)
        if len(pos) == 0:
            continue
        r, c   = pos[0]
        a      = policy[s2i[s]] #查策略，得到状态 s 应该执行的动作 a  该状态的最优动作编号
        dx, dy = ACTION_ARROWS[a]  # 动作对应的箭头位移
        # Arrow center at (c+0.5, r+0.5) in matplotlib coords
        ax.annotate('', xy=(c + 0.5 + dx, r + 0.5 + dy),
                    xytext=(c + 0.5 - dx, r + 0.5 - dy),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8)) #ax.annotate` 画箭头，从 `xytext`（起点）指向 `xy`（终点）

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. Optimal path plot  (Figure 9 style)
# =============================================================================

def plot_path(states, env, title='Optimal Path'):
    """
    Highlight a sampled trajectory on the maze grid.

    Parameters
    ----------
    states : list[int]  sequence of state ids from simulate_trajectory()
    """
    # Build a blank display matrix: 1 = valid cell, nan = wall/border
    display = np.full((20, 20), np.nan) #创建 20×20 的矩阵，全部填 nan，墙壁和边框默认显示黑色。
    for r in range(1, 19):
        for c in range(1, 19):
            val = STATE_MATRIX[r, c]
            if not np.isnan(val) and val != 0: #如果这个格子不是墙壁（nan）也不是边界（0），就是有效格子
                display[r, c] = 1.0 #有效格子显示为1.0，后续会用颜色区分路径和非路径格子

    fig, ax = plt.subplots(figsize=(10, 7.5))
    # ── 改动1：Greys colormap，有效格子白色，不需要 set_facecolor ──
    cmap = plt.get_cmap('Greys').copy() #复制一份 Greys colormap，后续修改这个副本的 bad color
    cmap.set_bad('black')  #把 bad color（nan格子）设置为黑色
    sns.heatmap(display, linewidths=0.25, linecolor='black',
                cbar=False, cmap=cmap, vmin=0, vmax=2, ax=ax) #用 seaborn 画热力图，格子间有黑色线条，关闭 colorbar，使用修改后的 cmap，vmin/vmax 锁死有效格子为白色

    # Draw arrows between consecutive states  画移动箭头
    for i in range(len(states) - 1):
        s_cur  = states[i]
        s_next = states[i + 1]
        if s_cur == s_next:          # 原地不动（撞墙）跳过
            continue
        pos_cur  = np.argwhere(STATE_MATRIX == s_cur)
        pos_next = np.argwhere(STATE_MATRIX == s_next)
        if len(pos_cur) == 0 or len(pos_next) == 0:
            continue
        r1, c1 = pos_cur[0]
        r2, c2 = pos_next[0]
        # 从格子中心画箭头到下一格子中心
        ax.annotate('', xy=(c2 + 0.5, r2 + 0.5),
                    xytext=(c1 + 0.5, r1 + 0.5),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Draw special cells on top of path
    coloring_blocks(ax, OIL_CELLS, BUMP_CELLS, START_CELL, GOAL_CELL)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. Average cumulative reward curves  (Problem 3b)
# =============================================================================

def plot_avg_cumulative_rewards(curves_dict, title='Average Cumulative Reward'):
    """
    Plot average cumulative reward curves for multiple settings.

    Parameters
    ----------
    curves_dict : dict  {label: avg_reward_array}
                  e.g. {'p=0.02': array, 'p=0.2': array, 'p=0.6': array}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, curve in curves_dict.items():
        ax.plot(curve, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Average Cumulative Reward')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Quick sanity-check
# =============================================================================
if __name__ == '__main__':
    from maze_env import build_maze_env
    from value_iteration import value_iteration
    from simulate import simulate_trajectory

    env = build_maze_env(p_stochastic=0.02)
    V, policy, _ = value_iteration(env, gamma=0.99, theta=0.01)
    states, _    = simulate_trajectory(env, policy)

    plot_value_function(V, env, title='Value Function — Base Scenario (VI)')
    plot_policy(policy, env, title='Optimal Policy — Base Scenario (VI)')
    plot_path(states, env, title='Optimal Path — Base Scenario (VI)')