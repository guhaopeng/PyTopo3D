"""
Assembly utilities for 3D topology optimization.

This module contains helper functions for assembling the force vector, 
该模块包含用于组装力向量的辅助函数。
boundary conditions, and element DOF matrices. 
该模块还包含用于组装边界条件和单元自由度矩阵的辅助函数。
"""

from typing import Optional, Set, Tuple

import numpy as np


def build_force_vector(
    nelx: int,
    nely: int,
    nelz: int,
    ndof: int,
    force_field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build the global force vector F.
    构建全局力向量 F。

    If force_field is None, applies default forces to the right face (x=nelx)
    in the negative z-direction.
    如果 force_field 为 None，则在负 z 方向将默认力应用于 x=nelx 的右面。

    If force_field is provided, it distributes the forces specified for each
    element equally among its 8 corner nodes.
    如果提供了 force_field，则将每个元素指定的力均匀分布在其 8 个角节点上。

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
        # 元素在 x、y、z 方向的数量。
    ndof : int
        Total number of degrees of freedom (3 * number_of_nodes).
        # 总自由度数量 (3 * 节点数量)。
    force_field : Optional[np.ndarray], shape (nely, nelx, nelz, 3), optional    
    force_field：可选的 np.ndarray，形状为 (nely, nelx, nelz, 3)，可选参数

        A 4D array where `force_field[y, x, z, :]` is the [Fx, Fy, Fz] force
        一个 4D 数组，其中force_field[y, x, z, :]表示与网格位置 (y, x, z) 处单元相关联的 [Fx, Fy, Fz] 力向量。

        vector associated with the element at grid position (y, x, z).
        # 与网格位置 (y, x, z) 处单元相关联的 [Fx, Fy, Fz] 力向量。
        Defaults to None (use default load case).


    Returns
    -------
    np.ndarray
        Global force vector F (shape: ndof) with applied nodal loads.
        # 应用了节点力的全局力向量 F (形状: ndof)。
    """
    F = np.zeros(ndof)

    if force_field is None:
        # Default implementation - forces on nodes at x=nelx, y=0 in -z direction
        # 应用于 x=nelx, y=0 的节点的默认力，在负 z 方向。
        # Nodes on the line x = nelx, y = 0
        # 在 x=nelx, y=0 的节点上应用默认力，在负 z 方向。
        il, jl, kl = np.meshgrid(  
            [nelx],
            np.arange(nely + 1),
            [0],
            indexing="ij",  # Only y=0  
        )
        # Calculate 0-based global node indices using Fortran order
        # 使用 Fortran 顺序计算基于 0 的全局节点索引
        # nid = iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
        loadnid_0based = (
            jl.flatten()
            + il.flatten() * (nely + 1)
            + kl.flatten() * (nelx + 1) * (nely + 1)
        )
        # Calculate 0-based DOF indices for z-direction (3*nid + 2)
        #计算 z 方向的 0-based 自由度索引（3*nid + 2）
        loaddof_0based = 3 * loadnid_0based + 2
        # Apply unit force in negative z-direction
        # 在负 z 方向应用单位力
        F[loaddof_0based] = -1.0

    else:
        # Validate force_field shape
        # 验证 force_field 形状
        expected_shape = (nely, nelx, nelz, 3)
        if force_field.shape != expected_shape:
            raise ValueError(
                f"force_field has shape {force_field.shape}, "
                f"but expected {expected_shape}"
            )

        # Iterate through each element and distribute its force to its 8 nodes
        # 遍历每个元素，将其力分布到其 8 个节点上
        for elz in range(nelz):
            for elx in range(nelx):
                for ely in range(nely):
                    element_force = force_field[ely, elx, elz, :]

                    # Skip if force is zero for this element
                    # 如果该元素的力为零，则跳过
                    if not np.any(element_force):
                        continue

                    # Distribute force to the 8 corner nodes
                    # 将该元素的力分布到其 8 个角节点上
                    force_per_node = element_force / 8.0

                    # Loop over the 8 local corners (relative coordinates dx, dy, dz in {0, 1})
                    # 遍历 8 个本地角点（相对坐标 dx, dy, dz 在 {0, 1} 中）
                    for dz in [0, 1]:
                        for dx in [0, 1]:
                            for dy in [0, 1]:
                                # Global coordinates of the node
                                # 节点的全局坐标
                                ix = elx + dx
                                iy = ely + dy
                                iz = elz + dz

                                # Calculate 0-based global node index (Fortran order)
                                # 使用 Fortran 顺序计算基于 0 的全局节点索引
                                nid = (
                                    iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
                                )

                                # Calculate 0-based global DOF indices
                                # 计算基于 0 的全局自由度索引
                                dof_x = 3 * nid
                                dof_y = 3 * nid + 1
                                dof_z = 3 * nid + 2

                                # Add force contribution to the global force vector F

                                # Ensure DOFs are within bounds (although they should be
                                # if nid is calculated correctly)
                                if dof_x < ndof:
                                    F[dof_x] += force_per_node[0]
                                if dof_y < ndof:
                                    F[dof_y] += force_per_node[1]
                                if dof_z < ndof:
                                    F[dof_z] += force_per_node[2]

    return F


def build_supports(
    nelx: int,
    nely: int,
    nelz: int,
    ndof: int,
    support_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build support constraints (fixed DOFs).

    If support_mask is None, applies default supports (fixed nodes)
    # 如果 support_mask 为 None，则应用默认支持（固定节点）
    on the left face (x=0). 所有在 x=0 的节点上的自由度都将被固定。


    If support_mask is provided, it identifies elements marked as supported. 
    # 如果提供了 support_mask，则根据该 mask 标识被标记为支持的元素。

    All 8 corner nodes of these marked elements will have their DOFs fixed.
    # 所有被标记为支持的元素的 8 个角节点上的自由度都将被固定。

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.
    ndof : int
        Total number of degrees of freedom. 
    support_mask : Optional[np.ndarray], shape (nely, nelx, nelz), optional 
    # 支持掩码，用于标识哪些元素被标记为支持。
    # 如果为 None，则使用默认的左脸支持放置（x=0 的节点固定）。
        Boolean mask indicating which elements are supported.
        If None, uses default left-face support placement.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - freedofs0: 0-based array of free (unconstrained) DOFs. 
        # 自由自由度索引，基于 0 索引，包含所有未被固定的自由度。

        - fixeddof0: 0-based array of fixed (constrained) DOFs. 
        # 固定自由度索引，基于 0 索引，包含所有被固定的自由度。
    """
    fixeddof0 = np.array([], dtype=int)  # Initialize as empty array
    fixednid_0based: np.ndarray = np.array([], dtype=int)  # Ensure defined type

    if support_mask is None:
        # Default implementation - fixed DOFs on nodes of the left face (x=0)
        iif, jf, kf = np.meshgrid(
            [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
        )
        # Calculate 0-based global node indices using Fortran order
        # nid = iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
        fixednid_0based = (
            jf.flatten()
            + iif.flatten() * (nely + 1)
            + kf.flatten() * (nelx + 1) * (nely + 1)
        )

    else:
        # Apply supports based on the support_mask provided for elements
        # 基于提供的 support_mask 应用支持（固定元素的 8 个角节点）
        # Validate support_mask shape
        expected_shape = (nely, nelx, nelz)
        if support_mask.shape != expected_shape:
            raise ValueError(
                f"support_mask has shape {support_mask.shape}, "
                f"but expected {expected_shape}"
            )

        # Find elements marked for support
        # 查找被标记为支持的元素
        y_indices, x_indices, z_indices = np.where(support_mask)

        if len(y_indices) == 0:
            # Fall back to default if no supports are specified in the mask
            # 如果 mask 中没有指定支持元素，则回退到默认的左脸支持
            print(
                "Warning: Support mask is provided but is empty. "
                "Falling back to default left-face support."
            )
            iif, jf, kf = np.meshgrid(
                [0], np.arange(nely + 1), np.arange(nelz + 1), indexing="ij"
            )
            fixednid_0based = (
                jf.flatten()
                + iif.flatten() * (nely + 1)
                + kf.flatten() * (nelx + 1) * (nely + 1)
            )
        else:
            # Strategy: For each element marked in support_mask, fix all DOFs
            # 策略：对于 support_mask 中标记的每个单元，固定所有自由度

            # of its 8 corner nodes. Collect all unique fixed node indices.
            # 收集所有唯一的固定节点索引
            all_fixed_node_indices: Set[int] = set()

            for ely, elx, elz in zip(y_indices, x_indices, z_indices):
                # Loop over the 8 local corners (relative coordinates dx, dy, dz in {0, 1})
                # 遍历 8 个本地角点（相对坐标 dx, dy, dz 在 {0, 1} 中）
                for dz in [0, 1]:
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            # Global coordinates of the node
                            ix = elx + dx
                            iy = ely + dy
                            iz = elz + dz

                            # Ensure node coordinates are within the grid boundaries
                            if 0 <= iy <= nely and 0 <= ix <= nelx and 0 <= iz <= nelz:
                                # Calculate 0-based global node index (Fortran order)
                                nid = (
                                    iy + ix * (nely + 1) + iz * (nelx + 1) * (nely + 1)
                                )
                                all_fixed_node_indices.add(nid)

            fixednid_0based = np.array(list(all_fixed_node_indices), dtype=int)

    # If fixednid_0based is defined and not empty (either from default or mask)
    # 如果 fixednid_0based 已定义且不为空（无论是默认还是 mask）
    if fixednid_0based.size > 0:
        # Fix all degrees of freedom (X, Y, Z) at these nodes
        # 固定所有节点的 X、Y、Z 自由度
        # DOFs are 0-based: 3*node_idx, 3*node_idx+1, 3*node_idx+2
        # 固定所有节点的 X、Y、Z 自由度，DOF 索引为 3*node_idx, 3*node_idx+1, 3*node_idx+2
        fixeddof0 = np.concatenate(
            [
                3 * fixednid_0based,
                3 * fixednid_0based + 1,
                3 * fixednid_0based + 2,
            ]
        )
        # Ensure uniqueness and sort
        # 确保固定自由度索引唯一并排序
        fixeddof0 = np.unique(fixeddof0)
        # Ensure DOFs are within bounds
        # 确保固定自由度索引在有效范围内
        fixeddof0 = fixeddof0[(fixeddof0 >= 0) & (fixeddof0 < ndof)]

    # Determine free DOFs
    # 确定自由自由度索引
    all_dofs0 = np.arange(ndof)  # 0-based DOFs
    # Use np.isin for potentially faster set difference calculation
    # 使用 np.isin 检查哪些自由度是固定的，哪些是自由的
    is_fixed = np.isin(all_dofs0, fixeddof0, assume_unique=True)
    freedofs0 = all_dofs0[~is_fixed]

    # Return 0-based indices
    return freedofs0, fixeddof0


def build_edof(
    nelx: int, nely: int, nelz: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build element DOF mapping and global assembly indices (iK, jK).
    构建单元自由度映射矩阵 edofMat 和全局组装索引 iK, jK。

    Uses Fortran-style (column-major) node numbering convention.
    采用 Fortran 风格（列主序）节点编号约定。

    edofMat maps element number to its 24 global DOFs (8 nodes * 3 DOFs/node).
    每个元素的 24 个全局自由度（8 个节点 * 3 个自由度/节点）。

    iK, jK provide the row and column indices for assembling the global
    stiffness matrix K in sparse COO format.
    iK, jK 提供组装全局刚度矩阵 K 的行索引和列索引（COO 格式）。

    Parameters
    ----------
    nelx, nely, nelz : int
        Number of elements in x, y, z directions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - edofMat: Array (nele, 24) mapping element index to 1-based global DOFs.
        - iK: Array (nele * 576,) row indices for COO sparse matrix assembly (1-based).
        - jK: Array (nele * 576,) column indices for COO sparse matrix assembly (1-based).
    """
    # Generate node numbers for the grid in Fortran order
    # 生成 Fortran 风格（列主序）的节点编号
    nodenrs = np.arange(1, (nelx + 1) * (nely + 1) * (nelz + 1) + 1).reshape(
        (nely + 1, nelx + 1, nelz + 1), order="F"
    )
    # Get the node number for the 'bottom-front-left' corner of each element
    # 每个元素的底部前左节点编号
    # (iy=0, ix=0, iz=0 local coords relative to element)
    edofVec_node_ids = nodenrs[:-1, :-1, :-1].ravel(order="F")

    # Get the 1-based DOF index for the first DOF (x-direction) of the first node
    # 每个元素的第一个节点的 X 自由度索引（1-based）
    edofVec = (
        3 * edofVec_node_ids - 2
    )  # 3*nid - 2 -> x-dof ; 3*nid - 1 -> y-dof ; 3*nid -> z-dof

    # Define the offsets to get the 24 DOFs of an H8 element relative to the first DOF
    # 每个 H8 元素的 24 个自由度（8 个节点 * 3 个自由度/节点）
    # Node order (local): 0,1,2,3 (bottom face z=0), 4,5,6,7 (top face z=1)
    # 节点顺序（本地）：0,1,2,3（底部面 z=0）, 4,5,6,7（顶部面 z=1）
    # Local node coords (y, x, z):
    # 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0)
    # 4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    # DOFs follow node order: [node0_x, node0_y, node0_z, node1_x, ..., node7_z]
    # 每个节点的 XYZ 自由度按节点顺序排列：[node0_x, node0_y, node0_z, node1_x, ..., node7_z]

    # Offsets calculated relative to edofVec (node0_x DOF)
    dof_offsets = np.array(
        [
            0,
            1,
            2,  # Node 0 (iy=0, ix=0, iz=0)
            3 * 1 + 0,
            3 * 1 + 1,
            3 * 1 + 2,  # Node 1 (iy=1, ix=0, iz=0) Offset=3*dy=3
            3 * (nely + 1 + 1) + 0,
            3 * (nely + 1 + 1) + 1,
            3 * (nely + 1 + 1)
            + 2,  # Node 2 (iy=1, ix=1, iz=0) Offset=3*(dx*(nely+1)+dy) = 3*(nely+1+1)
            3 * (nely + 1) + 0,
            3 * (nely + 1) + 1,
            3 * (nely + 1)
            + 2,  # Node 3 (iy=0, ix=1, iz=0) Offset=3*dx*(nely+1)=3*(nely+1)
            3 * (nelx + 1) * (nely + 1) + 0,
            3 * (nelx + 1) * (nely + 1) + 1,
            3 * (nelx + 1) * (nely + 1)
            + 2,  # Node 4 (iy=0, ix=0, iz=1) Offset=3*dz*(nelx+1)*(nely+1)
            3 * (nelx + 1) * (nely + 1) + 3 * 1 + 0,
            3 * (nelx + 1) * (nely + 1) + 3 * 1 + 1,
            3 * (nelx + 1) * (nely + 1) + 3 * 1 + 2,  # Node 5 (iy=1, ix=0, iz=1)
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1 + 1) + 0,
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1 + 1) + 1,
            3 * (nelx + 1) * (nely + 1)
            + 3 * (nely + 1 + 1)
            + 2,  # Node 6 (iy=1, ix=1, iz=1)
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1) + 0,
            3 * (nelx + 1) * (nely + 1) + 3 * (nely + 1) + 1,
            3 * (nelx + 1) * (nely + 1)
            + 3 * (nely + 1)
            + 2,  # Node 7 (iy=0, ix=1, iz=1)
        ],
        dtype=int,
    )

    # Build edofMat (nele, 24) containing 1-based global DOF indices for each element
    # 每个元素的 24 个全局自由度索引（1-based）
    edofMat = edofVec[:, np.newaxis] + dof_offsets[np.newaxis, :]

    # Prepare iK, jK indices (1-based) for COO sparse matrix format
    # 组装全局刚度矩阵 K 的行索引和列索引（COO 格式）（1-based）
    
    nele = nelx * nely * nelz
    iK = np.kron(edofMat, np.ones((1, 24), dtype=int)).ravel()
    jK = np.kron(edofMat, np.ones((24, 1), dtype=int)).ravel()

    return edofMat, iK, jK
