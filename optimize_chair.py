#!/usr/bin/env python3
import os
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from pytopo3d.core.optimizer import top3d
from pytopo3d.utils.assembly import build_force_vector, build_supports
from pytopo3d.visualization.display import display_3D

# 创建结果目录
result_dir = os.path.join('results', f'topo3d_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(os.path.join(result_dir, 'exports'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'visualizations'), exist_ok=True)

# 保存配置信息
config = {
    'nelx': 32,
    'nely': 32,
    'nelz': 32,
    'volfrac': 0.2,
    'penal': 3.0,
    'rmin': 1.5
}
with open(os.path.join(result_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# 加载初始密度场
initial_density = np.load('D:/3dtp/data_npy/outputyz_latent0.5.npy')
print(f"Loaded initial density field with shape: {initial_density.shape}")
print(f"Density range: {initial_density.min():.3f} to {initial_density.max():.3f}")

# 设置优化参数
nelx, nely, nelz = 32, 32, 32  # 与密度场尺寸匹配
volfrac = 0.1  # 体积分数约束
penal = 3.0  # SIMP惩罚因子
rmin = 2.0  # 滤波半径
disp_thres = 0.5  # 显示阈值

# 计算自由度数量
ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)

# 找到存在体素的最小x值
x_with_voxels = np.where(np.any(initial_density > 0.5, axis=(0, 2)))[0]
min_x = x_with_voxels[0] if len(x_with_voxels) > 0 else 0
print(f"最小x值：{min_x}")

#y_with_voxels = np.where(np.any(initial_density > 0.5, axis=(1, 2)))[0]
#min_y = y_with_voxels[0] if len(y_with_voxels) > 0 else 0
#print(f"最小y值：{min_y}")

# 定义四个支撑区域的x和z范围
#support_regions = [
#    (range(10, 13), range(10, 13)),  # 区域1: x=10-12, z=10-12
#    (range(20, 23), range(20, 23)),  # 区域2: x=20-22, z=20-22
#    (range(10, 13), range(20, 23)),  # 区域3: x=10-12, z=20-22
#    (range(20, 23), range(10, 13))   # 区域4: x=20-22, z=10-12
#]

# 找到最小y平面上所有的体素位置
#min_y_mask = (initial_density[min_y] > 0.5)
#min_y_voxels = np.where(min_y_mask)

# 找到最小x平面上所有的体素位置
min_x_mask = (initial_density[:, min_x, :] > 0.5)
min_x_voxels = np.where(min_x_mask)

# 设置支撑条件
fixed_nodes = []

# 将最小y平面上的所有体素位置添加为固定节点
#for i in range(len(min_y_voxels[0])):  # min_y_voxels[0]是x坐标，min_y_voxels[1]是z坐标
#    x = min_y_voxels[0][i]
#    z = min_y_voxels[1][i]
#    if x < nelx and z < nelz:
#        node = x + min_y * (nelx+1) + z * (nelx+1) * (nely+1)
#        fixed_nodes.append(int(node))

# 将最小x平面上的所有体素位置添加为固定节点
for i in range(len(min_x_voxels[0])):  # min_x_voxels[0]是y坐标，min_x_voxels[1]是z坐标
    y = min_x_voxels[0][i]
    z = min_x_voxels[1][i]
    if y < nely and z < nelz:
        node = min_x + y * (nelx+1) + z * (nelx+1) * (nely+1)
        fixed_nodes.append(int(node))

# 在最小y平面上为原有支撑区域添加固定节点
#for x_range, z_range in support_regions:
#    for x in x_range:
#        for z in z_range:
#            # 在最小y平面上的节点
#            node = x + min_y * (nelx+1) + z * (nelx+1) * (nely+1)
#            fixed_nodes.append(int(node))

# 设置载荷条件
force_nodes = []
forces = []

# 座面区域载荷(x=10)
seat_x = 10
seat_y_range = range(10, 21)  
seat_z_range = range(10, 21)  
seat_mask = (initial_density > 0.5) & (np.indices(initial_density.shape)[1] == seat_x)
seat_voxels = np.where(seat_mask)

for i in range(len(seat_voxels[0])):
    y = seat_voxels[0][i]
    z = seat_voxels[2][i]
    if y in seat_y_range and z in seat_z_range and seat_x < nelx:
        node = seat_x + y * (nelx+1) + z * (nelx+1) * (nely+1)
        force_nodes.append(int(node))
        # 座面区域载荷(x=10)
        forces.append((0, -1, 0))  # 向下的力

# 靠背区域载荷(z=10, x=5-20, y=10-20)
back_z = 10
back_x_range = range(5, 21)  # 5到20
back_y_range = range(10, 31)  # 10到20

# 在z=10平面上创建靠背区域
back_mask = (initial_density > 0.5) & (np.indices(initial_density.shape)[2] == back_z)
back_voxels = np.where(back_mask)

for i in range(len(back_voxels[0])):
    y = back_voxels[0][i]
    x = back_voxels[1][i]
    if x in back_x_range and y in back_y_range and back_z < nelz:
        node = x + y * (nelx+1) + back_z * (nelx+1) * (nely+1)
        force_nodes.append(int(node))
        forces.append((0, 0, -1))  # 向+z方向的力（靠背受力方向）

# 创建力向量和支撑约束
F = np.zeros((ndof, 1))
for node, force in zip(force_nodes, forces):
    F[node*3:(node+1)*3, 0] = force

# 创建固定自由度
fixeddofs = []
for node in fixed_nodes:
    fixeddofs.extend([3*node, 3*node+1, 3*node+2])
fixeddofs = np.array(fixeddofs)
freedofs = np.delete(np.arange(ndof), fixeddofs)

# 创建力场和支撑掩码
force_field = np.zeros((nely, nelx, nelz, 3))
support_mask = np.zeros((nely, nelx, nelz), dtype=bool)

# 将节点力转换为单元力场
for node, force in zip(force_nodes, forces):
    # 计算节点的(y,x,z)坐标
    k = node // ((nelx+1) * (nely+1))  # z坐标
    remainder = node % ((nelx+1) * (nely+1))
    x = remainder % (nelx+1)  # x坐标
    y = remainder // (nelx+1)  # y坐标
    
    # 如果节点在有效范围内，设置相应单元的力
    if x < nelx and y < nely and k < nelz:
        force_field[y, x, k] = force

# 将节点支撑转换为单元支撑掩码
for node in fixed_nodes:
    # 计算节点的(y,x,z)坐标
    k = node // ((nelx+1) * (nely+1))  # z坐标
    remainder = node % ((nelx+1) * (nely+1))
    x = remainder % (nelx+1)  # x坐标
    y = remainder // (nelx+1)  # y坐标
    
    # 如果节点在有效范围内，设置相应单元的支撑
    if x < nelx and y < nely and k < nelz:
        support_mask[y, x, k] = True

# 执行优化
print("Starting optimization...")
result = top3d(
    nelx=nelx,
    nely=nely,
    nelz=nelz,
    volfrac=volfrac,
    penal=penal,
    rmin=rmin,
    disp_thres=disp_thres,
    force_field=force_field,  # 使用力场
    support_mask=support_mask,  # 使用支撑掩码
    maxloop=300,  # 最大迭代次数
    use_gpu=True,  # 使用GPU加速
    save_history=True,  # 保存优化历史
    history_frequency=2  # 每2次迭代保存一次
)

# 解包结果
xPhys, history = result if isinstance(result, tuple) else (result, None)

# 获取最终的体积分数和收敛变化量
current_vol = np.mean(xPhys)
change = 0.0  # 由于无法直接获取change值，这里设置为0

# 保存优化结果
np.save('optimized_design.npy', xPhys)
print("Optimization complete. Result saved to optimized_design.npy")

# 保存优化历史
np.save(os.path.join(result_dir, 'optimization_history.npy'), history)
print(f"Optimization history saved to {os.path.join(result_dir, 'optimization_history.npy')}")

# 保存优化结果
np.save(os.path.join(result_dir, 'optimized_design.npy'), xPhys)
print(f"Optimized design saved to {os.path.join(result_dir, 'optimized_design.npy')}")

# 保存优化指标
metrics = {
    'final_objective': float(history['compliance_history'][-1]),
    'final_volume_fraction': float(current_vol),
    'iterations': len(history['compliance_history']),
    'convergence_change': float(change)
}
with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# 显示和保存优化后的设计
fig = display_3D(xPhys, disp_thres)
plt.savefig(os.path.join(result_dir, 'visualizations', 'optimized_design_with_boundary_conditions.png'))
plt.close()

print(f"\nResults saved in directory: {result_dir}")

# 可视化边界条件
support_density = np.zeros((nely, nelx, nelz))  # 使用 (y, x, z) 坐标系统

#for x_range, z_range in support_regions:
#    for x in x_range:
#        for z in z_range:
#            support_density[0, x, z] = 1.0  # 在 y=0 平面（xz平面）上添加支撑

# 使用最小x平面上的所有支撑点来生成支撑密度场
min_x_mask = (initial_density[:, min_x, :] > 0.5)
for y in range(nely):
    for z in range(nelz):
        if min_x_mask[y, z]:
            support_density[y, min_x, z] = 1.0  # 在最小x平面上设置支撑

force_density = np.zeros((nely, nelx, nelz))  # 使用 (y, x, z) 坐标系统
for node in force_nodes:
    # 从节点索引计算坐标
    k = node // ((nelx+1) * (nely+1))  # z坐标
    remainder = node % ((nelx+1) * (nely+1))
    i = remainder % (nelx+1)  # x坐标
    j = remainder // (nelx+1)  # y坐标
    
    if i < nelx and j < nely and k < nelz:
        force_density[j, i, k] = 1.0  # 使用 (y, x, z) 顺序

fig = display_3D(
    [support_density, force_density],
    thresholds=[0.5, 0.5],
    colors=['red', 'blue'],
    labels=['Support', 'Force'],
    alphas=[0.9, 0.9]
)
plt.savefig(os.path.join(result_dir, 'visualizations', 'boundary_conditions_and_obstacles.png'))
plt.close()