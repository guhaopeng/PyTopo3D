"""
Geometry processing utilities for 3D topology optimization.

This module provides functions for loading and processing geometry data from STL files
该模块提供了从 STL 文件加载和处理几何数据的函数。
and creating boundary conditions.
该模块还提供了创建边界条件的函数。
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

from pytopo3d.utils.boundary import create_boundary_arrays
from pytopo3d.utils.import_design_space import stl_to_design_space
from pytopo3d.utils.obstacles import parse_obstacle_config_file
from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.runner import create_visualization


def load_geometry_data(
    nelx: int,
    nely: int,
    nelz: int,
    design_space_stl: Optional[str] = None,
    pitch: float = 1.0,
    invert_design_space: bool = False,
    obstacle_config: Optional[str] = None, # 障碍物配置文件路径
    experiment_name: str = "experiment",
    logger: Optional[logging.Logger] = None,
    results_mgr: Optional[ResultsManager] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load design space and obstacle data.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        design_space_stl: Path to STL file defining design space
        pitch: Voxelization pitch for STL
        invert_design_space: Whether to invert design space (STL represents void)
        obstacle_config: Path to obstacle configuration file
        experiment_name: Name of the experiment
        logger: Configured logger
        results_mgr: Results manager instance

    Returns:
        Tuple containing design space mask, obstacle mask, and combined obstacle mask
    """
    # Handle design space from STL if provided
    design_space_mask = None
    if design_space_stl:
        try:
            if logger:
                logger.info(f"Loading design space from STL file: {design_space_stl}")

            # Use the voxelization pitch
            if logger:
                logger.info(f"Using voxelization pitch: {pitch}")

            # Invert flag
            if invert_design_space and logger:
                logger.info("Design space will be inverted (STL represents void space)")

            # Generate design space from STL
            # Resolution is determined by the mesh and pitch
            design_space_mask = stl_to_design_space(
                design_space_stl, pitch=pitch, invert=invert_design_space
            )

            # Update nelx, nely, nelz based on the voxelized shape
            local_nely, local_nelx, local_nelz = design_space_mask.shape

            if logger:
                logger.info(
                    f"Resolution from voxelization: {local_nely}x{local_nelx}x{local_nelz}"
                )

            # Save design space mask if results_mgr is provided
            if results_mgr:
                design_space_path = os.path.join(
                    results_mgr.experiment_dir, "design_space_mask.npy"
                )
                np.save(design_space_path, design_space_mask)
                if logger:
                    logger.info(f"Design space mask saved to {design_space_path}")

                # Copy the STL file to the experiment directory
                results_mgr.copy_file(design_space_stl, "design_space.stl")
                if logger:
                    logger.debug("Copied design space STL file to experiment directory")

                # Visualize the design space mask
                visualize_design_space_mask(
                    design_space_mask, experiment_name, results_mgr, logger
                )

        except Exception as e:
            if logger:
                logger.error(f"Error loading design space from STL: {e}")
                import traceback

                logger.debug(f"STL loading error details: {traceback.format_exc()}")
            raise
    else:
        if logger:
            logger.debug("No STL design space provided, using full rectangular domain")
        design_space_mask = np.ones((nely, nelx, nelz), dtype=bool)

    # Create obstacle mask if requested
    # 如果提供了障碍物配置文件，解析并创建障碍物掩码
    
    #obstacle_mask = None # 障碍物掩码
    obstacle_mask = np.zeros((nely, nelx, nelz), dtype=bool)
    # 自定义障碍物：[y范围, x范围, z范围]
    obstacle_mask[5:15, 15:20, 5:25] = True

    # 记录硬编码的障碍物
    hardcoded_obstacle = obstacle_mask.copy()

    # Handle obstacle config file case
    # 如果提供了障碍物配置文件，解析并创建障碍物掩码
    if obstacle_config:
        try:
            shape = (nely, nelx, nelz)
            obstacle_mask = parse_obstacle_config_file(obstacle_config, shape)
            # 合并配置文件中的障碍物和硬编码的障碍物
            obstacle_mask = np.logical_or(obstacle_mask, hardcoded_obstacle)
            n_obstacle_elements = np.count_nonzero(obstacle_mask)
            if logger:
                logger.info(
                    f"Loaded {n_obstacle_elements} obstacle elements (including hardcoded obstacles)"
                )

            # Copy the obstacle config file to the experiment directory if results_mgr is provided
            if results_mgr:
                results_mgr.copy_file(obstacle_config, "obstacle_config.json")
                if logger:
                    logger.debug("Copied obstacle config file to experiment directory")

        except Exception as e:
            if logger:
                logger.error(f"Error loading obstacle configuration: {e}")
            raise
    else:
        if logger:
            n_obstacle_elements = np.count_nonzero(obstacle_mask)
            logger.info(
                f"No obstacle configuration provided, using hardcoded obstacles ({n_obstacle_elements} elements)"
            )
        # 移除这部分代码，保留之前设置的硬编码障碍物
        # if design_space_stl:
        #     obstacle_mask = np.zeros_like(design_space_mask)
        # else:
        #     obstacle_mask = np.zeros((nely, nelx, nelz), dtype=bool)

        

    # Combine design space and obstacle masks 
    # Elements outside the design space are treated as obstacles
    combined_obstacle_mask = obstacle_mask.copy()
    if design_space_mask is not None:
        # Areas outside design space (False values) become obstacles (True in obstacle mask)
        # 设计空间之外的区域（False 值）成为障碍物（在障碍物掩码中为 True）
        combined_obstacle_mask = np.logical_or(
            combined_obstacle_mask, ~design_space_mask
        )
        if logger:
            logger.info(
                f"Combined obstacle and design space masks, {np.count_nonzero(combined_obstacle_mask)} elements restricted"
            )

    return design_space_mask, obstacle_mask, combined_obstacle_mask


def visualize_design_space_mask(
    design_space_mask, experiment_name, results_mgr, logger
) -> str:
    """
    Create and save visualization of the design space mask.

    Args:
        design_space_mask: Boolean array representing the design space
        experiment_name: Name of the experiment
        results_mgr: Results manager instance
        logger: Logger instance

    Returns:
        Path to the saved visualization
    """
    if logger:
        logger.info("Creating visualization of design space mask from STL")

    # Convert boolean mask to float for visualization
    design_space_array = design_space_mask.astype(float)

    # Create visualization of just the design space
    viz_path = create_visualization(
        arrays=[design_space_array],
        thresholds=[0.5],
        colors=["green"],
        labels=["Design Space"],
        alphas=[0.7],
        experiment_name=experiment_name,
        results_mgr=results_mgr,
        filename="design_space_mask",
        title="Design Space Mask from STL",
    )

    if logger:
        logger.info(f"Design space visualization saved to {viz_path}")
    return viz_path


def create_boundary_conditions(nelx, nely, nelz) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create boundary condition arrays for loads and constraints.

    A wrapper around create_boundary_arrays with standardized output.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction

    Returns:
        Tuple containing loads array and constraints array
    """
    return create_boundary_arrays(nelx, nely, nelz)
