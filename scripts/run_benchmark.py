#!/usr/bin/env python3
"""
Benchmark script for PyTopo3D.

This script runs comprehensive benchmarks to evaluate the performance of PyTopo3D
compared to the original MATLAB implementation and generates comparative plots.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(
        description="Benchmark PyTopo3D performance and compare with MATLAB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/benchmarks",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--matlab-data",
        type=str,
        help="Path to MATLAB benchmark data (JSON format)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="8,16,32,64",
        help="Comma-separated list of problem sizes to benchmark",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum problem size (elements) to benchmark",
    )
    parser.add_argument(
        "--volfrac",
        type=float,
        default=0.3,
        help="Volume fraction for benchmark problems",
    )
    parser.add_argument(
        "--penal",
        type=float,
        default=3.0,
        help="Penalization value for SIMP",
    )
    parser.add_argument(
        "--tolx",
        type=float,
        default=0.01,
        help="Convergence tolerance",
    )
    parser.add_argument(
        "--maxloop",
        type=int,
        default=100,
        help="Maximum number of iterations (keep low for benchmarking)",
    )
    parser.add_argument(
        "--run-python",
        action="store_true",
        help="Run Python benchmarks",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    return parser.parse_args()


def load_matlab_data(matlab_data_path: str) -> Dict[str, Any]:
    """
    Load MATLAB benchmark data from a JSON file.
    
    Args:
        matlab_data_path: Path to the JSON file with MATLAB benchmark data
        
    Returns:
        Dictionary with benchmark data
    """
    try:
        with open(matlab_data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded MATLAB benchmark data from {matlab_data_path}")
        return data
    except Exception as e:
        print(f"Error loading MATLAB data from {matlab_data_path}: {e}")
        return {}


def run_python_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run Python benchmarks for specified problem sizes.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dictionary with benchmark results
    """
    # Parse sizes
    try:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
    except ValueError:
        print(f"Error: Invalid size format: {args.sizes}")
        print("Expected format: comma-separated integers (e.g., 8,16,32,64)")
        return {}
    
    # Filter by max size if specified
    if args.max_size:
        sizes = [s for s in sizes if s <= args.max_size]
    
    if not sizes:
        print("No valid sizes to benchmark")
        return {}
    
    print(f"Running Python benchmarks for sizes: {sizes}")
    
    # Run scaling test using main.py
    cmd = [
        "python", "main.py",
        "--scaling-test",
        f"--scaling-sizes={','.join(str(s) for s in sizes)}",
        f"--volfrac={args.volfrac}",
        f"--penal={args.penal}",
        f"--tolx={args.tolx}",
        f"--maxloop={args.maxloop}",
        f"--benchmark-dir={args.output_dir}"
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Load the results
        results_path = os.path.join(args.output_dir, "scaling_test_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded Python benchmark results from {results_path}")
            return results
        else:
            print(f"Error: Benchmark results file not found at {results_path}")
            return {}
            
    except subprocess.CalledProcessError as e:
        print(f"Error running Python benchmarks: {e}")
        return {}


def generate_comparison_plots(
    python_results: Dict[str, Any],
    matlab_results: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Generate comparison plots between Python and MATLAB implementations.
    
    Args:
        python_results: Python benchmark results
        matlab_results: MATLAB benchmark results
        output_dir: Directory to save plots
    """
    try:
        from pytopo3d.utils.benchmarking import (
            compare_benchmarks,
            generate_benchmark_plots,
        )
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate Python-only plots
        if python_results:
            plot_files = generate_benchmark_plots(python_results, plots_dir)
            print(f"Generated Python benchmark plots in {plots_dir}")
        
        # Generate comparison plots if both datasets are available
        if python_results and matlab_results:
            comparison_dir = os.path.join(plots_dir, "comparison")
            comparison_plots = compare_benchmarks(
                python_results, matlab_results, comparison_dir
            )
            print(f"Generated comparison plots in {comparison_dir}")
            
    except ImportError as e:
        print(f"Error importing benchmarking modules: {e}")
        print("Make sure you've activated the correct Python environment")


def main():
    """Run the benchmark script."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MATLAB data if provided
    matlab_results = {}
    if args.matlab_data:
        matlab_results = load_matlab_data(args.matlab_data)
    
    # Run Python benchmarks if requested
    python_results = {}
    if args.run_python:
        python_results = run_python_benchmarks(args)
    else:
        # Try to load existing Python results
        results_path = os.path.join(args.output_dir, "scaling_test_results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    python_results = json.load(f)
                print(f"Loaded existing Python benchmark results from {results_path}")
            except Exception as e:
                print(f"Error loading existing Python results: {e}")
    
    # Generate comparison plots
    if not args.skip_plots and (python_results or matlab_results):
        generate_comparison_plots(python_results, matlab_results, args.output_dir)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=================")
    
    if python_results:
        py_sizes = sorted([int(size) for size in python_results.keys()])
        print(f"Python benchmarks: {len(py_sizes)} sizes - {py_sizes}")
    else:
        print("No Python benchmark data available")
    
    if matlab_results:
        matlab_sizes = sorted([int(size) for size in matlab_results.keys()])
        print(f"MATLAB benchmarks: {len(matlab_sizes)} sizes - {matlab_sizes}")
    else:
        print("No MATLAB benchmark data available")
    
    if python_results and matlab_results:
        common_sizes = sorted(list(set([int(s) for s in python_results.keys()]) & 
                              set([int(s) for s in matlab_results.keys()])))
        
        if common_sizes:
            print("\nPerformance Comparison (Python/MATLAB ratio):")
            print("---------------------------------------------")
            for size in common_sizes:
                py_time = python_results[str(size)]["total_time_seconds"]
                matlab_time = matlab_results[str(size)]["total_time_seconds"]
                ratio = py_time / matlab_time
                print(f"  Size {size}: {py_time:.2f}s / {matlab_time:.2f}s = {ratio:.2f}x")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 