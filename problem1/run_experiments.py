import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from maze_env import (build_maze_env, STATE_MATRIX,
                      OIL_CELLS, BUMP_CELLS, START_CELL, GOAL_CELL)
from policy_iteration import policy_iteration
from value_iteration import value_iteration
from simulate import simulate_trajectory, compute_cumulative_rewards
from visualize import (plot_value_function, plot_policy, plot_path,
                       plot_avg_cumulative_rewards, coloring_blocks)

# =============================================================================
# Scenarios for Problem 1 & 2
# =============================================================================
SCENARIOS = [
    {'p': 0.02, 'gamma': 0.99, 'theta': 0.01, 'name': 'Base Scenario'},
    {'p': 0.40, 'gamma': 0.99, 'theta': 0.01, 'name': 'Large Stochasticity Scenario'},
    {'p': 0.02, 'gamma': 0.40, 'theta': 0.01, 'name': 'Small Discount Factor Scenario'},
]

T_MAX = 400 


# =============================================================================
# Problem 1 & 2: PI and VI across three scenarios
# =============================================================================
def run_problem1_2():
    print("=" * 60)
    print("Problem 1 & 2: Policy Iteration vs Value Iteration")
    print("=" * 60)

    results = {}  # 收集结果，最后打印对比汇总

    for sc in SCENARIOS:
        env   = build_maze_env(p_stochastic=sc['p']) 
        s2i   = env['state_to_idx'] # 状态编号 → 矩阵下标的字典
        start = env['start_state']  # 起点状态编号，= 50
        goal  = env['goal_state']   # 终点状态编号，= 208
        name  = sc['name']    # 场景名称字符串
        g     = sc['gamma']   # 折扣因子
        th    = sc['theta']   # 收敛阈值

        print(f"\n--- Scenario: {name}  (p={sc['p']}, γ={g}, θ={th}) ---") #打印当前场景的名称和参数设置

        # ── Policy Iteration ──────────────────────────────────────
        V_pi, pol_pi, n_pi = policy_iteration(env, gamma=g, theta=th) #运行策略迭代算法，得到值函数 V_pi、策略 pol_pi 和迭代次数 n_pi
        states_pi, _ = simulate_trajectory(env, pol_pi, max_steps=T_MAX) #使用得到的策略 pol_pi 模拟一条轨迹，记录访问过的状态编号列表 states_pi 和对应的奖励列表（这里用 _ 忽略了奖励列表）
        print(f"  PI : {n_pi} improvement steps | "
              f"V(start)={V_pi[s2i[start]]:.2f} | "
              f"reached goal: {states_pi[-1] == goal} | "
              f"steps: {len(states_pi)}")

        plot_value_function(V_pi, env, title=f'Value Function — {name} (PI)')  #画三张图：价值函数热力图、策略箭头图、路径图。
        plot_policy(pol_pi, env,       title=f'Optimal Policy — {name} (PI)')
        plot_path(states_pi, env,      title=f'Optimal Path — {name} (PI)')

        # ── Value Iteration ───────────────────────────────────────
        V_vi, pol_vi, n_vi = value_iteration(env, gamma=g, theta=th)
        states_vi, _ = simulate_trajectory(env, pol_vi, max_steps=T_MAX)
        print(f"  VI : {n_vi} iterations       | "
              f"V(start)={V_vi[s2i[start]]:.2f} | "
              f"reached goal: {states_vi[-1] == goal} | "
              f"steps: {len(states_vi)}")

        plot_value_function(V_vi, env, title=f'Value Function — {name} (VI)')
        plot_policy(pol_vi, env,       title=f'Optimal Policy — {name} (VI)')
        plot_path(states_vi, env,      title=f'Optimal Path — {name} (VI)')

        # 收集结果
        results[name] = {   #把这个场景的关键数字存进 results 字典，name 是 key。
            'p': sc['p'], 'gamma': g,
            'PI_iters': n_pi, 'VI_iters': n_vi,
            'V_start_PI': V_pi[s2i[start]],
            'V_start_VI': V_vi[s2i[start]],
        }

    # ── 对比汇总 ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Scenario':<35} {'PI iters':>9} {'VI iters':>9} "
          f"{'V(start) PI':>12} {'V(start) VI':>12}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<35} {r['PI_iters']:>9} {r['VI_iters']:>9} "
              f"{r['V_start_PI']:>12.2f} {r['V_start_VI']:>12.2f}")


# =============================================================================
# Problem 3: Effect of stochasticity
# =============================================================================
def run_problem3():
    print("\n" + "=" * 60)
    print("Problem 3: Effect of Stochasticity")
    print("=" * 60)

    p_values = [0.02, 0.2, 0.6]
    gamma, theta = 0.99, 0.01
    N_TRAJ = 10
    avg_curves = {}

    for p_val in p_values:          # ← 改名为 p_val，避免和坐标变量混淆
        env  = build_maze_env(p_stochastic=p_val)
        goal = env['goal_state']

        _, policy, _ = value_iteration(env, gamma=gamma, theta=theta) # 用VI得到最优策略

        # ── 3a: Two independent trajectories on same figure ───────
        traj1, _ = simulate_trajectory(env, policy, max_steps=T_MAX)  #两条独立采样轨迹（simulate内部用np.random.choice采样）
        traj2, _ = simulate_trajectory(env, policy, max_steps=T_MAX)
        
        #构建迷宫显示矩阵
        display = np.full((20, 20), np.nan) # 20x20矩阵，初始值全是 np.nan，表示不可访问的格子。后面会把可访问的格子标记为 1.0。
        for r in range(1, 19): 
            for c in range(1, 19):
                val = STATE_MATRIX[r, c] 
                if not np.isnan(val) and val != 0:
                    display[r, c] = 1.0 
        #绘制迷宫底图
        fig, ax = plt.subplots(figsize=(10, 7.5))
        sns.heatmap(display, linewidths=0.25, linecolor='black',
                    cbar=False, cmap='Greys', vmin=0, vmax=2, ax=ax)
        ax.set_facecolor('black')

        # Trajectory 1: steelblue
        for i in range(len(traj1) - 1):
            s1, s2 = traj1[i], traj1[i+1]
            if s1 == s2: #如果前后状态编号相同，说明这个动作没有成功执行（可能是因为环境的随机性导致的），就不画箭头，直接跳过。
                continue
            pos1 = np.argwhere(STATE_MATRIX == s1) #在 STATE_MATRIX 中找到状态编号 s1 的位置，返回一个二维数组 pos1，形状是 (1, 2)，包含行列坐标。
            pos2 = np.argwhere(STATE_MATRIX == s2)
            if len(pos1) == 0 or len(pos2) == 0:
                continue
            r1, c1 = pos1[0]; r2, c2 = pos2[0] #取出起点和终点的行列坐标
            ax.annotate('', xy=(c2+0.5, r2+0.5), xytext=(c1+0.5, r1+0.5),
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

        # Trajectory 2: darkorange
        for i in range(len(traj2) - 1):
            s1, s2 = traj2[i], traj2[i+1]
            if s1 == s2:
                continue
            pos1 = np.argwhere(STATE_MATRIX == s1)
            pos2 = np.argwhere(STATE_MATRIX == s2)
            if len(pos1) == 0 or len(pos2) == 0:
                continue
            r1, c1 = pos1[0]; r2, c2 = pos2[0]
            ax.annotate('', xy=(c2+0.5, r2+0.5), xytext=(c1+0.5, r1+0.5),
                        arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2))

        coloring_blocks(ax, OIL_CELLS, BUMP_CELLS, START_CELL, GOAL_CELL)

        legend = [Line2D([0],[0], color='steelblue',  lw=2, label='Trajectory 1'),
                  Line2D([0],[0], color='darkorange', lw=2, label='Trajectory 2')]
        ax.legend(handles=legend, loc='upper right')
        ax.set_title(f'Path Overlay — p={p_val}')
        plt.tight_layout()
        plt.show()

        print(f"  p={p_val}: traj1={len(traj1)} steps, traj2={len(traj2)} steps")

        # ── 3b: Average cumulative reward over 10 trajectories ────
        all_rewards = []
        tau_max = 0
        for _ in range(N_TRAJ): #模拟10条轨迹，记录每条轨迹的奖励列表，并更新 tau_max 以记录最长的轨迹长度。
            _, rews = simulate_trajectory(env, policy, max_steps=T_MAX)
            all_rewards.append(rews)
            tau_max = max(tau_max, len(rews)) #记录最长的轨迹长度，后面用来构建 G_matrix   max(tau_max, len(rews))是指在当前的 tau_max 和新轨迹的长度 len(rews) 之间取较大值，确保 tau_max 始终是所有轨迹中最长的那个长度。

        G_matrix = np.zeros((N_TRAJ, tau_max)) #构建一个 N_TRAJ 行、tau_max 列的矩阵 G_matrix，用来存储每条轨迹在每个时间步的累计奖励。初始值全是 0。
        for i, rews in enumerate(all_rewards): #i 是轨迹编号（0~9），rews 是该条轨迹的逐步奖励列表
            G = np.cumsum(rews) #计算该条轨迹的累计奖励列表 G，长度和 rews 一样。G[t] 表示从第0步到第t步的奖励总和。
            G_matrix[i, :len(G)] = G #把有效部分填入矩阵
            if len(G) < tau_max: #如果这条轨迹比最长的轨迹短，就把剩余部分填成最后一个累计奖励值，表示在后续时间步奖励不再变化了。
                G_matrix[i, len(G):] = G[-1]

        avg_curves[f'p={p_val}'] = G_matrix.mean(axis=0) #按列求平均，得到每个时间步的平均累计奖励曲线，存入 avg_curves 字典，key 是 p 的值字符串。

    plot_avg_cumulative_rewards(avg_curves,
        title='Average Cumulative Reward — Effect of Stochasticity')


# =============================================================================
# Problem 4: Effect of bump penalty
# =============================================================================
def run_problem4():
    print("\n" + "=" * 60)
    print("Problem 4: Effect of Bump Penalty")
    print("=" * 60)

    gamma, theta = 0.99, 0.01

    env10 = build_maze_env(p_stochastic=0.02, bump_penalty=-10)
    _, pol10, _ = value_iteration(env10, gamma=gamma, theta=theta) #用VI得到最优策略,然后模拟一条轨迹，记录访问过的状态编号列表 states10 和对应的奖励列表（这里用 _ 忽略了奖励列表）。
    states10, _ = simulate_trajectory(env10, pol10, max_steps=T_MAX)  #用得到的策略 pol10 模拟一条轨迹，记录访问过的状态编号列表 states10 和对应的奖励列表（这里用 _ 忽略了奖励列表）。
    plot_policy(pol10, env10, title='Optimal Policy — Bump Penalty = -10')
    plot_path(states10, env10, title='Optimal Path — Bump Penalty = -10')

    env50 = build_maze_env(p_stochastic=0.02, bump_penalty=-50)
    _, pol50, _ = value_iteration(env50, gamma=gamma, theta=theta)
    states50, _ = simulate_trajectory(env50, pol50, max_steps=T_MAX)
    plot_policy(pol50, env50, title='Optimal Policy — Bump Penalty = -50')
    plot_path(states50, env50, title='Optimal Path — Bump Penalty = -50')

    print(f"  penalty=-10 path: {len(states10)} steps")
    print(f"  penalty=-50 path: {len(states50)} steps")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    run_problem1_2()
    run_problem3()
    run_problem4()