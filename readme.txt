project2/
│
├── problem1/
│   ├── maze_env.py          # 迷宫环境（状态空间、转移矩阵、奖励）
│   ├── policy_iteration.py  # Policy Iteration 算法
│   ├── value_iteration.py   # Value Iteration 算法
│   ├── simulate.py          # 轨迹采样/模拟
│   ├── visualize.py         # 所有可视化函数
│   └── run_experiments.py   # 主运行文件（各场景实验）
│
├── problem2/
│   ├── gene_env.py          # 基因网络环境（转移矩阵、奖励）
│   ├── policy_iteration.py  # Policy Iteration（矩阵形式）
│   ├── value_iteration.py   # Value Iteration（矩阵形式）
│   ├── simulate.py          # AvgA 评估
│   ├── visualize.py         # 可视化
│   └── run_experiments.py   # 主运行文件
│
└── main.py                  # 可选：统一入口