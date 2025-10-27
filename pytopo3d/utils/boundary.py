"""
Boundary condition utilities for 3D topology optimization.
# 3D 拓扑优化的边界条件工具

This module contains functions for handling boundary conditions such as loads and constraints.
# 本模块包含用于处理边界条件（如加载和约束）的函数。
"""

from typing import Tuple

import numpy as np


def calculate_boundary_positions(
    nelx: int, nely: int, nelz: int
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Calculate the positions of loads and constraints for visualization.
    # 计算加载和约束的可视化位置。

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        (load_positions, constraint_positions), where each position is a tuple of (x, y, z) arrays.
    """
    # Calculate load positions
    il, jl, kl = np.meshgrid([nelx], [0], np.arange(nelz + 1), indexing="ij")
    load_x = il.ravel()
    load_y = nely - jl.ravel()  # Converted to visualization coordinates 
    # 加载位置：在 x=nelx, y=0, z=0 到 z=nelz+1 的位置
    load_z = kl.ravel()

    # Calculate constraint positions
    # 约束位置：在 x=0, y=0 到 y=nely+1, z=0 到 z=nelz+1 的位置
    iif, jf, kf = np.meshgrid(
        [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
    )
    constraint_x = iif.ravel()
    constraint_y = nely - jf.ravel()  # Converted to visualization coordinates
    constraint_z = kf.ravel()

    return (load_x, load_y, load_z), (constraint_x, constraint_y, constraint_z)


def create_boundary_arrays(
    nelx: int, nely: int, nelz: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create density arrays for loads and constraints visualization.
    # 创建加载和约束的可视化密度数组。

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz)
        where positions have value 1.0 and the rest have value 0.0
    """
    # Create empty arrays for loads and constraints
    # 创建加载和约束的可视化密度数组
    loads_array = np.zeros((nely, nelx, nelz))
    constraints_array = np.zeros((nely, nelx, nelz))

    # DEPRECATED - Hardcoded default BCs, use create_bc_visualization_arrays instead
    # Set loads at the right face (x=nelx-1) at the bottom (y=nely-1) on all Z levels
    # 在所有 Z 层的右表面（x=nelx-1）底部（y=nely-1）设置载荷
    # loads_array[nely-1, nelx-1, :] = 1.0

    # Set constraints at the left face (x=0) on all Y and Z levels
    # 在所有 Y 和 Z 层的左表面（x=0）设置约束
    # constraints_array[:, 0, :] = 1.0
    print(
        "Warning: create_boundary_arrays is deprecated. Use create_bc_visualization_arrays."
    )
    return loads_array, constraints_array


def create_bc_visualization_arrays(
    nelx: int, nely: int, nelz: int, ndof: int, F: np.ndarray, fixeddof0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates visualization arrays for loads and constraints based on actual FEA boundary conditions.
    # 根据实际 FEA 边界条件创建加载和约束的可视化数组。

    Maps nodal forces and fixed DOFs to the element grid for visualization.
    # 将节点力和固定 DOF 映射到元素网格以进行可视化。

    Marks all elements adjacent to a node with applied BCs.
    # 标记所有与应用边界条件的节点相邻的元素。

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom.
    F : np.ndarray
        Global force vector (shape: ndof).
    fixeddof0 : np.ndarray
        Array of 0-based fixed degree of freedom indices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (loads_array, constraints_array), each with shape (nely, nelx, nelz).
    """
    loads_array = np.zeros((nely, nelx, nelz), dtype=float)
    constraints_array = np.zeros((nely, nelx, nelz), dtype=float)

    # --- Map Fixed DOFs to Element Constraints ---
    # 将固定 DOF 映射到元素约束
    fixed_nid0 = np.unique(fixeddof0 // 3)
    nelyp1 = nely + 1
    nelxp1_nelyp1 = (nelx + 1) * (nely + 1)

    for nid in fixed_nid0:
        # Inverse calculation for Fortran node index -> (ix, iy, iz)
        # 逆计算 Fortran 节点索引 -> (ix, iy, iz)
        iz = nid // nelxp1_nelyp1
        rem = nid % nelxp1_nelyp1
        ix = rem // nelyp1
        iy = rem % nelyp1

        # Mark adjacent elements (within grid bounds)
        # 标记所有与固定节点相邻的元素（在网格边界内）
        for elz in range(max(0, iz - 1), min(nelz, iz + 1)):
            for elx in range(max(0, ix - 1), min(nelx, ix + 1)):
                for ely in range(max(0, iy - 1), min(nely, iy + 1)):
                    if 0 <= elx < nelx and 0 <= ely < nely and 0 <= elz < nelz:
                        constraints_array[ely, elx, elz] = 1.0

    # --- Map Forces to Element Loads ---
    loaded_dof0 = np.where(F != 0)[0]
    loaded_nid0 = np.unique(loaded_dof0 // 3)

    for nid in loaded_nid0:
        # Inverse calculation for Fortran node index -> (ix, iy, iz)
        iz = nid // nelxp1_nelyp1
        rem = nid % nelxp1_nelyp1
        ix = rem // nelyp1
        iy = rem % nelyp1

        # Mark adjacent elements (within grid bounds)
        for elz in range(max(0, iz - 1), min(nelz, iz + 1)):
            for elx in range(max(0, ix - 1), min(nelx, ix + 1)):
                for ely in range(max(0, iy - 1), min(nely, iy + 1)):
                    if 0 <= elx < nelx and 0 <= ely < nely and 0 <= elz < nelz:
                        loads_array[ely, elx, elz] = 1.0

    return loads_array, constraints_array
