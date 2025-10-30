#!/usr/bin/env python3
import os
import numpy as np
from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.visualizer import create_optimization_animation, visualize_initial_setup
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 创建结果管理器
results_dir = os.path.join('results', 'history_visualization')
os.makedirs(results_dir, exist_ok=True)
results_mgr = ResultsManager(results_dir)

# 加载优化历史
history_path = 'D:/3dtp/results/topo3d_20251028_165239/optimization_history.npy'
history = np.load(history_path, allow_pickle=True).item()

# 打印历史数据的信息
print("历史数据包含以下键:", list(history.keys()))
print("密度历史帧数:", len(history['density_history']))
print("迭代历史长度:", len(history['iteration_history']))
print("柔度历史长度:", len(history['compliance_history']))

# 创建力和约束数组
nelx, nely, nelz = 32, 32, 32
loads_array = np.zeros((nely, nelx, nelz))
constraints_array = np.zeros((nely, nelx, nelz))

# 创建初始设置可视化
viz_path = visualize_initial_setup(
    nelx=nelx,
    nely=nely,
    nelz=nelz,
    loads_array=loads_array,
    constraints_array=constraints_array,
    experiment_name='chair_optimization',
    logger=logger,
    results_mgr=results_mgr
)

print(f"初始设置可视化已保存到: {viz_path}")

# 创建优化动画
gif_path = create_optimization_animation(
    nelx=nelx,
    nely=nely,
    nelz=nelz,
    experiment_name='chair_optimization',
    disp_thres=0.5,
    animation_frames=100,  # 增加帧数
    animation_fps=10,  # 增加帧率
    logger=logger,
    results_mgr=results_mgr,
    history=history,
    loads_array=loads_array,
    constraints_array=constraints_array
)

if gif_path and os.path.exists(gif_path):
    print(f"动画已保存到: {gif_path}")
else:
    print("动画创建失败")