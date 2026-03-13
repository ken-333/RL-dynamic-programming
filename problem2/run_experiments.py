import numpy as np 
import matplotlib.pyplot as plt

from gene_env import build_gene_env, ALL_STATES, ALL_ACTIONS
from value_iteration import value_iteration
from policy_iteration import policy_iteration
from simulate import compute_AvgA

GAMMA = 0.99
THETA = 0.01 #"use matrix-form Dynamic Programming with γ = 0.99" 和 "θ = 0.01"

ACTION_NAMES = ['a1(none)', 'a2(ATM)', 'a3(p53)', 'a4(WIP1)', 'a5(MDM2)'] # 是动作的文字标签，用于打印
STATE_LABELS  = [''.join(str(int(b)) for b in ALL_STATES[i]) for i in range(16)] #把每个状态向量转成字符串，比如索引3的状态 [0,0,1,1] → '0011'，用于打印和画图
# e.g. STATE_LABELS[0]='0000', STATE_LABELS[15]='1111'


# =============================================================================
# Shared helper: print policy table
# =============================================================================
def print_policy(policy, label): #print_policy
    print(f"\n  Policy — {label}")
    print(f"  {'State':<8} {'Action':<12}")
    print(f"  {'-'*20}")
    for i in range(16):
        print(f"  {STATE_LABELS[i]:<8} {ACTION_NAMES[policy[i]]:<12}") #每行告诉你：在这个状态下，最优策略规定执行哪个动作


# =============================================================================
# Shared helper: plot value function as bar chart
# =============================================================================
def plot_value_function(V, title):
    fig, ax = plt.subplots(figsize=(10, 4)) #fig 是整张图，ax 是坐标轴
    ax.bar(STATE_LABELS, V, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('State (binary)')
    ax.set_ylabel('V*(s)')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45) #把 x 轴的状态标签旋转 45 度，避免重叠
    plt.tight_layout()
    plt.show() #tight_layout() 自动调整间距防止标签被截断，show() 弹出图像窗口。


# =============================================================================
# Shared helper: plot policy as heatmap (state × gene, colored by action)
# =============================================================================
def plot_policy_heatmap(policy, title):
    """
    One row per state, one column per gene.
    Cell color = which action is taken at that state.
    """
    fig, ax = plt.subplots(figsize=(5, 8)) #创建5×8的画布。action_colors 定义每个动作对应的颜色
    action_colors = ['white', 'steelblue', 'darkorange', 'green', 'red']
    data = np.array([[policy[i]] for i in range(16)])   # (16, 1) for color

    for i in range(16): #循环16次，每次处理一个状态。a 是该状态对应的动作索引。
        a = policy[i]
        ax.add_patch(plt.Rectangle((0, i), 1, 1,
                                   color=action_colors[a], ec='black', lw=0.5))
        ax.text(0.5, i + 0.5, ACTION_NAMES[a], ha='center', va='center', fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 16)
    ax.set_yticks(np.arange(16) + 0.5)
    ax.set_yticklabels(STATE_LABELS)
    ax.set_xticks([])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Problem 2a: p=0.045, VI, compare optimal vs no-control
# =============================================================================
def run_part_a(): 
    print("=" * 60)
    print("Part (a): p=0.045, Value Iteration")
    print("=" * 60)

    p = 0.045 #"For p=0.045, use matrix-form Value Iteration... Compare AvgA under optimal policy with AvgA under no-control policy."
    env = build_gene_env(p_noise=p)
    N   = env['N']  # 用 p=0.045p=0.045 p=0.045 构建环境，得到转移矩阵 M 和奖励矩阵 R_sa。  N=16 是状态数。

    # ── Value Iteration ───────────────────────────────────────
    V, policy_vi, n_iters = value_iteration(env, gamma=GAMMA, theta=THETA)
    print(f"\n  VI converged in {n_iters} iterations")

    # ── No-control policy ─────────────────────────────────────
    no_ctrl = np.zeros(N, dtype=int)   # always a^1

    # ── AvgA comparison ───────────────────────────────────────
    avga_opt = compute_AvgA(env, policy_vi)
    avga_nc  = compute_AvgA(env, no_ctrl)
    print(f"\n  AvgA (optimal policy) = {avga_opt:.4f}")
    print(f"  AvgA (no-control)     = {avga_nc:.4f}")

    print_policy(policy_vi, f'VI  p={p}')
    plot_value_function(V, f'Value Function — p={p} (VI)')
    plot_policy_heatmap(policy_vi, f'Optimal Policy — p={p} (VI)')

    return policy_vi, V, n_iters


# =============================================================================
# Problem 2b: p=0.18 and p=0.55, VI, compare with part (a)
# =============================================================================
def run_part_b(policy_a):
    print("\n" + "=" * 60)
    print("Part (b): p=0.18 and p=0.55, Value Iteration")
    print("=" * 60)

    results = {}
    for p in [0.18, 0.55]:
        env = build_gene_env(p_noise=p)

        V, policy_vi, n_iters = value_iteration(env, gamma=GAMMA, theta=THETA)
        avga_opt = compute_AvgA(env, policy_vi)
        avga_nc  = compute_AvgA(env, np.zeros(env['N'], dtype=int))

        print(f"\n  p={p}: VI converged in {n_iters} iterations")
        print(f"  AvgA (optimal) = {avga_opt:.4f} | AvgA (no-control) = {avga_nc:.4f}")

        # Compare policy with part (a)
        same = np.array_equal(policy_vi, policy_a)
        print(f"  Policy identical to part (a): {same}")
        if not same:
            diff_states = [STATE_LABELS[i] for i in range(16) if policy_vi[i] != policy_a[i]]
            print(f"  States with different actions: {diff_states}")

        print_policy(policy_vi, f'VI  p={p}')
        plot_value_function(V, f'Value Function — p={p} (VI)')
        plot_policy_heatmap(policy_vi, f'Optimal Policy — p={p} (VI)')

        results[p] = {'V': V, 'policy': policy_vi, 'n_iters': n_iters,
                      'avga_opt': avga_opt, 'avga_nc': avga_nc}

    return results


# =============================================================================
# Problem 2c: p=0.045, PI (initial policy = a^1 everywhere), compare with (a)
# =============================================================================
def run_part_c(policy_a, V_a):
    print("\n" + "=" * 60)
    print("Part (c): p=0.045, Policy Iteration")
    print("=" * 60)

    p   = 0.045
    env = build_gene_env(p_noise=p)
    N   = env['N']

    # PI initial policy: a^1 (index 0) for all states
    V, policy_pi, n_impr = policy_iteration(env, gamma=GAMMA, theta=THETA)

    avga_opt = compute_AvgA(env, policy_pi)
    print(f"\n  PI converged in {n_impr} improvement steps")
    print(f"  AvgA (PI optimal) = {avga_opt:.4f}")

    # Compare with VI result from part (a)
    same = np.array_equal(policy_pi, policy_a)
    print(f"\n  Policy identical to VI (part a): {same}")
    if not same:
        diff_states = [STATE_LABELS[i] for i in range(16) if policy_pi[i] != policy_a[i]]
        print(f"  States with different actions: {diff_states}")

    # Compare V values
    max_V_diff = np.max(np.abs(V - V_a))
    print(f"  Max |V_PI - V_VI| = {max_V_diff:.6f}")

    print_policy(policy_pi, f'PI  p={p}')
    plot_value_function(V, f'Value Function — p={p} (PI)')
    plot_policy_heatmap(policy_pi, f'Optimal Policy — p={p} (PI)')


# =============================================================================
# Summary table
# =============================================================================
def print_summary(policy_a, results_b, n_iters_a):
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'p':<8} {'Method':<6} {'Iters':<8} {'AvgA (opt)':<14} {'AvgA (no-ctrl)':<16}")
    print("-" * 70)
    # part a — reuse already-computed n_iters_a, only recompute AvgA
    env_a   = build_gene_env(p_noise=0.045)
    avga_a  = compute_AvgA(env_a, policy_a)
    avga_nc = compute_AvgA(env_a, np.zeros(16, dtype=int))
    print(f"{0.045:<8} {'VI':<6} {n_iters_a:<8} {avga_a:<14.4f} {avga_nc:<16.4f}")
    for p, r in results_b.items():
        print(f"{p:<8} {'VI':<6} {r['n_iters']:<8} {r['avga_opt']:<14.4f} {r['avga_nc']:<16.4f}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    policy_a, V_a, n_iters_a = run_part_a()
    results_b                 = run_part_b(policy_a)
    run_part_c(policy_a, V_a)
    print_summary(policy_a, results_b, n_iters_a)