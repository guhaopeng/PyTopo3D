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
        
        # Handle MATLAB 'n' prefix in keys (e.g., 'n8192' -> '8192')
        # MATLAB doesn't allow numeric field names, so we add 'n' prefix
        fixed_data = {}
        for key, value in data.items():
            if key.startswith('n') and key[1:].isdigit():
                fixed_data[key[1:]] = value
            else:
                fixed_data[key] = value
        
        print(f"Loaded MATLAB benchmark data from {matlab_data_path}")
        return fixed_data
    except Exception as e:
        print(f"Error loading MATLAB data from {matlab_data_path}: {e}")
        return {}


def run_python_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run Python benchmarks for various problem sizes.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with benchmark results
    """
    sizes = [int(s) for s in args.sizes.split(",")]
    results = {}
    
    # Filter sizes by max_size if provided
    if args.max_size is not None:
        sizes = [s for s in sizes if s * (s//2) * (s//3) <= args.max_size]
    
    # Add large-scale problem sizes
    large_scale_sizes = [64, 128]
    
    # Combine regular and large-scale sizes, removing duplicates
    all_sizes = sorted(set(sizes + large_scale_sizes))
    
    print(f"Running Python benchmarks for sizes: {all_sizes}")
    
    for size in all_sizes:
        nelx = size
        nely = max(1, size // 2)
        nelz = max(1, size // 3)
        elements = nelx * nely * nelz
        
        # Check if this problem size is too large
        if args.max_size is not None and elements > args.max_size:
            print(f"Skipping size {nelx}x{nely}x{nelz} ({elements} elements) - exceeds max size {args.max_size}")
            continue
            
        # For large problems, reduce iterations and adjust tolerance to save time
        maxloop = args.maxloop
        tolx = args.tolx
        
        if elements > 100000:  # For problems with more than 100k elements
            print(f"Large problem detected ({elements} elements) - adjusting parameters")
            maxloop = min(args.maxloop, 50)  # Reduce max iterations for large problems
            tolx = max(args.tolx, 0.02)      # Use higher tolerance for large problems
            print(f"Adjusted parameters: maxloop={maxloop}, tolx={tolx}")
        
        print(f"\nRunning benchmark for size {nelx}x{nely}x{nelz} ({elements} elements)")
        
        # Run the benchmark
        output_dir = os.path.join(args.output_dir, f"size_{nelx}x{nely}x{nelz}")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            "python", "main.py",
            "--nelx", str(nelx),
            "--nely", str(nely),
            "--nelz", str(nelz),
            "--volfrac", str(args.volfrac),
            "--penal", str(args.penal),
            "--tolx", str(tolx),
            "--maxloop", str(maxloop),
            "--benchmark",
            "--save-benchmark",
            "--benchmark-dir", output_dir
        ]
        
        try:
            # Release memory before running large problems
            if elements > 100000:
                if sys.platform.startswith('win'):
                    # On Windows
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                else:
                    # On Linux/Unix/MacOS
                    import gc
                    gc.collect()
                
            # Start the subprocess
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                print(f"Error running benchmark for size {nelx}x{nely}x{nelz}:")
                print(stderr)
                continue
                
            # Load the benchmark results
            benchmark_file = os.path.join(output_dir, f"benchmark_size_{elements}_nelx{nelx}_nely{nely}_nelz{nelz}_py.json")
            if os.path.exists(benchmark_file):
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                results[str(elements)] = benchmark_data
                print(f"✓ Benchmark completed: {nelx}x{nely}x{nelz} - {benchmark_data['total_time_seconds']:.2f}s, " +
                      f"{benchmark_data.get('peak_memory_mb', 0):.2f}MB")
            else:
                print(f"✗ Benchmark file not found: {benchmark_file}")
                
        except Exception as e:
            print(f"Error running benchmark for size {nelx}x{nely}x{nelz}: {e}")
    
    # Save combined results
    if results:
        output_file = os.path.join(args.output_dir, "scaling_test_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {output_file}")
    
    return results


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