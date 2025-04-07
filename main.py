#!/usr/bin/env python3
"""
Main entry point for the 3D topology optimization package.

This script provides a command-line interface to run the topology optimization.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pytopo3d.cli.parser import parse_args
from pytopo3d.preprocessing.geometry import load_geometry_data
from pytopo3d.runners.experiment import (
    execute_optimization,
    export_result_to_stl,
    setup_experiment,
)
from pytopo3d.utils.metrics import collect_metrics
from pytopo3d.visualization.visualizer import (
    create_optimization_animation,
    visualize_final_result,
    visualize_initial_setup,
)


def main():
    """
    Main function to run the optimization from command-line arguments.
    """
    # Parse command-line arguments
    args = parse_args()

    # Handle scaling test if requested
    if args.scaling_test:
        return run_scaling_test(args)

    try:
        # Setup experiment, logging and results manager
        logger, results_mgr = setup_experiment(
            verbose=args.verbose,
            quiet=args.quiet,
            log_level=args.log_level,
            log_file=args.log_file,
            experiment_name=getattr(args, "experiment_name", None),
            description=args.description,
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin
        )
        
        # Update args.experiment_name if it was generated in setup_experiment
        if not hasattr(args, "experiment_name") or not args.experiment_name:
            args.experiment_name = results_mgr.experiment_name

        # Load design space and obstacle data
        design_space_mask, obstacle_mask, combined_obstacle_mask = load_geometry_data(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            design_space_stl=getattr(args, "design_space_stl", None),
            pitch=getattr(args, "pitch", 1.0),
            invert_design_space=getattr(args, "invert_design_space", False),
            obstacle_config=getattr(args, "obstacle_config", None),
            experiment_name=args.experiment_name,
            logger=logger, 
            results_mgr=results_mgr
        )

        # Create and save initial visualization
        loads_array, constraints_array, _ = visualize_initial_setup(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            experiment_name=args.experiment_name,
            logger=logger, 
            results_mgr=results_mgr, 
            combined_obstacle_mask=combined_obstacle_mask
        )

        # Run the optimization with benchmarking if requested
        xPhys, history, run_time, benchmark_results = execute_optimization(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
            disp_thres=args.disp_thres,
            tolx=getattr(args, "tolx", 0.01),
            maxloop=getattr(args, "maxloop", 2000),
            create_animation=getattr(args, "create_animation", False),
            animation_frequency=getattr(args, "animation_frequency", 10),
            logger=logger,
            combined_obstacle_mask=combined_obstacle_mask,
            benchmark=getattr(args, "benchmark", False),
            save_benchmark=getattr(args, "save_benchmark", False),
            benchmark_dir=getattr(args, "benchmark_dir", "results/benchmarks"),
        )

        # Save the result to the experiment directory
        result_path = results_mgr.save_result(xPhys, "optimized_design.npy")
        logger.debug(f"Optimization result saved to {result_path}")

        # Create visualization of the final result
        visualize_final_result(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            experiment_name=args.experiment_name,
            disp_thres=args.disp_thres,
            logger=logger,
            results_mgr=results_mgr,
            xPhys=xPhys,
            combined_obstacle_mask=combined_obstacle_mask,
            loads_array=loads_array,
            constraints_array=constraints_array,
        )

        # Create animation if history was captured
        gif_path = None
        if history:
            gif_path = create_optimization_animation(
                nelx=args.nelx,
                nely=args.nely,
                nelz=args.nelz,
                experiment_name=args.experiment_name,
                disp_thres=args.disp_thres,
                animation_frames=getattr(args, "animation_frames", 50),
                animation_fps=getattr(args, "animation_fps", 5),
                logger=logger,
                results_mgr=results_mgr,
                history=history,
                combined_obstacle_mask=combined_obstacle_mask,
                loads_array=loads_array,
                constraints_array=constraints_array,
            )

        # Export result as STL if requested
        stl_exported = export_result_to_stl(
            export_stl=getattr(args, "export_stl", False),
            stl_level=getattr(args, "stl_level", 0.5),
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            logger=logger, 
            results_mgr=results_mgr, 
            result_path=result_path
        )

        # Collect and save metrics
        metrics = collect_metrics(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=args.rmin,
            disp_thres=args.disp_thres,
            tolx=getattr(args, "tolx", 0.01),
            maxloop=getattr(args, "maxloop", 2000),
            design_space_stl=getattr(args, "design_space_stl", None),
            pitch=getattr(args, "pitch", 1.0),
            obstacle_config=getattr(args, "obstacle_config", None),
            animation_fps=getattr(args, "animation_fps", 5),
            stl_level=getattr(args, "stl_level", 0.5),
            smooth_stl=getattr(args, "smooth_stl", False),
            smooth_iterations=getattr(args, "smooth_iterations", 3),
            xPhys=xPhys,
            design_space_mask=design_space_mask,
            obstacle_mask=obstacle_mask,
            combined_obstacle_mask=combined_obstacle_mask,
            run_time=run_time,
            gif_path=gif_path,
            stl_exported=stl_exported,
        )
        
        # Add benchmark results to metrics if available
        if benchmark_results:
            metrics["benchmark"] = benchmark_results
        
        results_mgr.update_metrics(metrics)
        logger.debug("Metrics updated")

        # Generate benchmark plots if requested
        if getattr(args, "generate_plots", False) and benchmark_results:
            try:
                from pytopo3d.utils.benchmarking import (
                    compare_benchmarks,
                    generate_benchmark_plots,
                )
                
                # Load MATLAB comparison data if provided
                matlab_data = None
                if args.benchmark_matlab and os.path.exists(args.benchmark_matlab):
                    try:
                        with open(args.benchmark_matlab, 'r') as f:
                            matlab_data = json.load(f)
                        logger.info(f"Loaded MATLAB benchmark data from {args.benchmark_matlab}")
                    except Exception as e:
                        logger.error(f"Error loading MATLAB benchmark data: {e}")
                
                # Generate plots directory
                plots_dir = os.path.join(args.benchmark_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                # Generate benchmark plots
                problem_size = args.nelx * args.nely * args.nelz
                single_run_data = {str(problem_size): benchmark_results}
                plot_files = generate_benchmark_plots(single_run_data, plots_dir)
                
                # Generate comparison plots if MATLAB data is available
                if matlab_data:
                    comparison_plots = compare_benchmarks(
                        single_run_data, matlab_data, 
                        os.path.join(plots_dir, "comparison")
                    )
                    plot_files.update(comparison_plots)
                
                logger.info(f"Benchmark plots generated in {plots_dir}")
                
            except ImportError as e:
                logger.error(f"Error generating benchmark plots: {e}")

        logger.info(f"Optimization complete in {run_time:.2f} seconds.")
        logger.info(f"Result saved to {result_path}")
        logger.info(f"All experiment files are in {results_mgr.experiment_dir}")

    except Exception as e:
        if "logger" in locals():
            logger.error(f"Error in main function: {e}")
            import traceback

            logger.debug(f"Error details: {traceback.format_exc()}")
        else:
            print(f"Error during initialization: {e}", file=sys.stderr)
            import traceback

            print(f"Error details: {traceback.format_exc()}", file=sys.stderr)
        return 1

    return 0


def run_scaling_test(args):
    """
    Run a scaling test with multiple problem sizes.
    
    Args:
        args: Command line arguments
    
    Returns:
        Exit code (0 for success)
    """
    # Parse scaling sizes from command line
    try:
        sizes = [int(s.strip()) for s in args.scaling_sizes.split(",")]
    except ValueError:
        print(f"Error: Invalid scaling sizes format: {args.scaling_sizes}")
        print("Expected format: comma-separated integers (e.g., 8,16,32,64)")
        return 1
    
    print(f"Running scaling test with sizes: {sizes}")
    
    # Setup benchmark directory
    benchmark_dir = args.benchmark_dir or "results/benchmarks"
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Create results dictionary
    benchmark_results = {}
    
    # Run optimization for each size
    for size in sizes:
        print(f"\n=== Running optimization with size {size} ===")
        
        # Override args with current size
        args.nelx = size
        args.nely = max(1, size // 2)  # Typical aspect ratio
        args.nelz = max(1, size // 3)  # Typical aspect ratio
        args.benchmark = True
        args.save_benchmark = True
        
        # Generate appropriate experiment name
        args.experiment_name = f"scaling_test_{size}x{args.nely}x{args.nelz}"
        
        # Setup experiment
        logger, results_mgr = setup_experiment(
            verbose=args.verbose,
            quiet=args.quiet,
            log_level=args.log_level,
            log_file=args.log_file,
            experiment_name=args.experiment_name,
            description=f"Scaling test with size {size}",
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=min(3.0, max(1.0, size / 20))  # Scale rmin with problem size
        )
        
        # Initialize obstacle/design masks
        combined_obstacle_mask = None
        
        # Run the optimization with benchmarking
        _, _, run_time, benchmark_data = execute_optimization(
            nelx=args.nelx,
            nely=args.nely,
            nelz=args.nelz,
            volfrac=args.volfrac,
            penal=args.penal,
            rmin=min(3.0, max(1.0, size / 20)),  # Scale rmin with problem size
            disp_thres=args.disp_thres,
            tolx=getattr(args, "tolx", 0.01),
            maxloop=getattr(args, "maxloop", 2000),
            create_animation=False,  # Don't create animations for scaling tests
            logger=logger,
            combined_obstacle_mask=combined_obstacle_mask,
            benchmark=True,
            save_benchmark=True,
            benchmark_dir=benchmark_dir,
        )
        
        # Store benchmark data for this size
        if benchmark_data:
            problem_size = args.nelx * args.nely * args.nelz
            benchmark_results[str(problem_size)] = benchmark_data
        
        print(f"Completed size {size} in {run_time:.2f} seconds")
    
    # Save the combined results
    combined_results_path = os.path.join(benchmark_dir, "scaling_test_results.json")
    with open(combined_results_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    print(f"Scaling test complete. Results saved to {combined_results_path}")
    
    # Generate plots from the combined results
    try:
        from pytopo3d.utils.benchmarking import generate_benchmark_plots
        
        plots_dir = os.path.join(benchmark_dir, "plots")
        plot_files = generate_benchmark_plots(benchmark_results, plots_dir)
        
        print(f"Benchmark plots generated in {plots_dir}")
        for plot_type, filepath in plot_files.items():
            print(f"  - {plot_type}: {os.path.basename(filepath)}")
            
    except ImportError as e:
        print(f"Error generating benchmark plots: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
