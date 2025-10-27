"""
Visualization utilities for 3D topology optimization.

This module provides functions for creating visualizations of topology optimization
results, boundary conditions, and creating animations.
# 该模块提供了用于创建拓扑优化结果可视化、边界条件可视化和动画的函数。
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from pytopo3d.utils.results_manager import ResultsManager
from pytopo3d.visualization.animation import save_optimization_gif
from pytopo3d.visualization.runner import create_visualization


def visualize_initial_setup(
    nelx: int,
    nely: int,
    nelz: int,
    loads_array: np.ndarray,
    constraints_array: np.ndarray,
    experiment_name: str,
    logger: Optional[logging.Logger] = None,
    results_mgr: Optional[ResultsManager] = None,
    combined_obstacle_mask: Optional[np.ndarray] = None,
) -> str:
    """
    Create and save initial visualization showing boundary conditions and obstacles.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        loads_array: Array showing load positions for visualization (nely, nelx, nelz)
        constraints_array: Array showing constraint positions for visualization (nely, nelx, nelz)
        experiment_name: Name of the experiment
        logger: Configured logger
        results_mgr: Results manager instance
        combined_obstacle_mask: Combined obstacle and design space mask

    Returns:
        Path to the saved visualization
    """
    # Create obstacle array for visualization
    obstacle_array = (
        combined_obstacle_mask.astype(float)
        if combined_obstacle_mask is not None
        else np.zeros((nely, nelx, nelz))
    )

    # Boundary condition arrays are now passed directly

    # Save configuration if results_mgr is provided
    if results_mgr:
        # Create a simple configuration dictionary
        config = {
            "nelx": nelx,
            "nely": nely,
            "nelz": nelz,
            "experiment_name": experiment_name,
        }

        results_mgr.save_config(config)
        if logger:
            logger.debug("Configuration saved")

    # Create visualization for boundary conditions, loads, constraints, and obstacle geometry
    if logger:
        logger.debug("Creating boundary condition visualization")

    # Validate shapes
    expected_shape = (nely, nelx, nelz)
    if loads_array.shape != expected_shape:
        raise ValueError(
            f"loads_array shape mismatch: expected {expected_shape}, got {loads_array.shape}"
        )
    if constraints_array.shape != expected_shape:
        raise ValueError(
            f"constraints_array shape mismatch: expected {expected_shape}, got {constraints_array.shape}"
        )
    if obstacle_array.shape != expected_shape:
        raise ValueError(
            f"obstacle_array shape mismatch: expected {expected_shape}, got {obstacle_array.shape}"
        )

    arrays_to_visualize = [obstacle_array, loads_array, constraints_array]
    thresholds = [0.5, 0.5, 0.5]
    colors = ["yellow", "blue", "red"]
    labels = ["Obstacles", "Loads", "Constraints"]
    alphas = [0.3, 0.9, 0.9]  # Make obstacles transparent

    boundary_viz_path = create_visualization(
        arrays=arrays_to_visualize,
        thresholds=thresholds,
        colors=colors,
        labels=labels,
        alphas=alphas,  # Add transparency
        experiment_name=experiment_name,
        results_mgr=results_mgr,
        filename="boundary_conditions_and_obstacles",
        title="Boundary Conditions and Obstacles",
    )
    if logger:
        logger.info(f"Boundary conditions visualization saved to {boundary_viz_path}")

    # Return only the path, as loads/constraints are passed in
    return boundary_viz_path


def visualize_final_result(
    nelx: int,
    nely: int,
    nelz: int,
    experiment_name: str,
    disp_thres: float,
    logger: Optional[logging.Logger] = None,
    results_mgr: Optional[ResultsManager] = None,
    xPhys: Optional[np.ndarray] = None,
    combined_obstacle_mask: Optional[np.ndarray] = None,
    loads_array: Optional[np.ndarray] = None,
    constraints_array: Optional[np.ndarray] = None,
) -> str:
    """
    Create and save visualization of the final optimization result.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        experiment_name: Name of the experiment
        disp_thres: Threshold for displaying elements
        logger: Configured logger
        results_mgr: Results manager instance
        xPhys: Optimized design
        combined_obstacle_mask: Combined obstacle and design space mask
        loads_array: Array showing load positions
        constraints_array: Array showing constraint positions

    Returns:
        Path to the saved visualization
    """
    # Create design_only array (optimized design without obstacles)
    design_only = xPhys.copy()
    design_only[combined_obstacle_mask] = (
        0  # Remove design elements where obstacles are
    )

    # Create combined visualization with optimized design, loads, constraints, and obstacles
    if logger:
        logger.debug("Creating combined visualization")
    obstacle_array = combined_obstacle_mask.astype(float)
    combined_arrays = [design_only, obstacle_array, loads_array, constraints_array]
    combined_thresholds = [disp_thres, 0.5, 0.5, 0.5]
    combined_colors = ["gray", "yellow", "blue", "red"]
    combined_labels = ["Optimized Design", "Obstacles", "Loads", "Constraints"]
    combined_alphas = [0.9, 0.3, 0.9, 0.9]  # Make obstacles transparent

    combined_viz_path = create_visualization(
        arrays=combined_arrays,
        thresholds=combined_thresholds,
        colors=combined_colors,
        labels=combined_labels,
        alphas=combined_alphas,  # Add transparency
        experiment_name=experiment_name,
        results_mgr=results_mgr,
        filename="optimized_design_with_boundary_conditions",
        title="Optimized Design with Boundary Conditions",
    )
    if logger:
        logger.info(f"Combined visualization saved to {combined_viz_path}")
    return combined_viz_path


def create_optimization_animation(
    nelx: int,
    nely: int,
    nelz: int,
    experiment_name: str,
    disp_thres: float,
    animation_frames: int = 50,
    animation_fps: int = 5,
    logger: Optional[logging.Logger] = None,
    results_mgr: Optional[ResultsManager] = None,
    history: Optional[Dict[str, Any]] = None,
    combined_obstacle_mask: Optional[np.ndarray] = None,
    loads_array: Optional[np.ndarray] = None,
    constraints_array: Optional[np.ndarray] = None,
) -> Optional[str]:
    """
    Create an animation of the optimization process if history was captured.

    Args:
        nelx: Number of elements in x direction
        nely: Number of elements in y direction
        nelz: Number of elements in z direction
        experiment_name: Name of the experiment
        disp_thres: Threshold for displaying elements
        animation_frames: Number of frames to include in animation
        animation_fps: Frames per second for animation
        logger: Configured logger
        results_mgr: Results manager instance
        history: Optimization history data
        combined_obstacle_mask: Combined obstacle and design space mask
        loads_array: Array showing load positions
        constraints_array: Array showing constraint positions

    Returns:
        Path to the generated GIF file, or None if animation failed
    """
    if not history:
        return None

    try:
        if logger:
            logger.info("Creating GIF visualization of optimization process...")

        # If there are more frames than we want to include, sample them
        history_frames = history["density_history"]
        history_iterations = history["iteration_history"]
        history_compliances = history["compliance_history"]

        if logger:
            logger.debug(
                f"Animation data: {len(history_frames)} density frames, "
                f"{len(history_iterations)} iterations, "
                f"{len(history_compliances)} compliance values"
            )

        if len(history_frames) > animation_frames:
            # Calculate sampling frequency
            sample_rate = max(1, len(history_frames) // animation_frames)
            if logger:
                logger.debug(f"Sampling animation frames (every {sample_rate} frames)")
        else:
            sample_rate = 1

        # Create the animation
        try:
            gif_path = save_optimization_gif(
                frames=history_frames,
                obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
                compliances=history_compliances,
                disp_thres=disp_thres,
                results_mgr=results_mgr,
                filename="optimization_animation",
                fps=animation_fps,
                every_n_iterations=sample_rate,
            )

            if os.path.exists(gif_path) and os.path.getsize(gif_path) > 0:
                if logger:
                    logger.info(f"Optimization animation saved to {gif_path}")
                return gif_path
            else:
                if logger:
                    logger.error(
                        f"Animation GIF file was not created properly or is empty: {gif_path}"
                    )
                return None
        except Exception as e:
            if logger:
                logger.error(f"Error creating animation: {e}")
            return None

    except Exception as e:
        if logger:
            logger.error(f"Error preparing animation data: {e}")
        return None
