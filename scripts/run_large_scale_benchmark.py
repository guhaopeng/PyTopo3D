#!/usr/bin/env python3
"""
Large-Scale Benchmark Script for PyTopo3D.

This script runs large-scale benchmarks on PyTopo3D to evaluate its performance
on problems with up to 500k+ elements, addressing reviewer concerns about scalability.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import psutil


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for large-scale benchmarking."""
    parser = argparse.ArgumentParser(
        description="Run large-scale benchmarks for PyTopo3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/large_scale_benchmarks",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--problem-sizes",
        type=str,
        default="32x16x16,64x32x32,128x64x64", 
        help="Comma-separated list of problem sizes in format nelx×nely×nelz",
    )
    parser.add_argument(
        "--matlab-data",
        type=str,
        help="Path to MATLAB benchmark data (JSON format)",
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
        "--rmin",
        type=float,
        default=4.0,
        help="Filter radius",
    )
    parser.add_argument(
        "--skip-largest",
        action="store_true",
        help="Skip the largest problem size if memory is limited",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a PDF report with scaling results for the paper",
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


def check_memory_requirements() -> Dict[str, int]:
    """
    Check available system memory and estimate requirements for different problem sizes.
    
    Returns:
        Dictionary with problem sizes and estimated memory requirements
    """
    # Get available memory
    available_gb = psutil.virtual_memory().available / (1024**3)
    total_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
    
    # Estimate memory requirements (based on empirical testing)
    # Format: elements: memory_required_in_gb
    requirements = {
        8192: 1,       # 32x16x16 (8k elements) ~1GB
        65536: 5,      # 64x32x32 (65k elements) ~5GB
        524288: 30,    # 128x64x64 (524k elements) ~30GB
        4194304: 200,  # 256x128x128 (4.2M elements) ~200GB (estimated)
    }
    
    # Print requirements and check if we have enough memory
    print("\nEstimated memory requirements:")
    for size, req_gb in requirements.items():
        nelx = int(round(size**(1/3)))
        nely = nelx // 2
        nelz = nelx // 2
        can_run = available_gb >= req_gb
        status = "✓" if can_run else "✗"
        print(f"  {status} {nelx}x{nely}x{nelz} ({size} elements): {req_gb:.1f}GB required")
    
    return requirements


def run_benchmarks(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run benchmarks for specified problem sizes.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with benchmark results
    """
    # Parse problem sizes
    problem_sizes = []
    for size_str in args.problem_sizes.split(","):
        if "x" in size_str:
            dims = size_str.split("x")
            if len(dims) == 3:
                try:
                    nelx = int(dims[0])
                    nely = int(dims[1])
                    nelz = int(dims[2])
                    problem_sizes.append((nelx, nely, nelz))
                except ValueError:
                    print(f"Invalid size format: {size_str}, skipping")
        else:
            try:
                # If only one number is provided, use it as nelx and calculate nely, nelz
                nelx = int(size_str)
                nely = max(1, nelx // 2)
                nelz = max(1, nelx // 2)
                problem_sizes.append((nelx, nely, nelz))
            except ValueError:
                print(f"Invalid size: {size_str}, skipping")
    
    # Check memory requirements
    mem_requirements = check_memory_requirements()
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    results = {}
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a log file for the benchmarking run
    log_file = os.path.join(args.output_dir, "benchmark_log.txt")
    with open(log_file, "w") as f:
        f.write(f"Large-Scale Benchmark Run: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"System: {platform.system()} {platform.release()} on {platform.machine()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB total, {available_gb:.1f}GB available\n\n")
        f.write(f"Problem sizes: {args.problem_sizes}\n")
        f.write(f"Parameters: volfrac={args.volfrac}, penal={args.penal}, rmin={args.rmin}\n\n")
    
    # Run benchmarks for each problem size
    for i, (nelx, nely, nelz) in enumerate(problem_sizes):
        elements = nelx * nely * nelz
        
        # Skip largest problem if requested and memory is limited
        if args.skip_largest and i == len(problem_sizes) - 1:
            if elements in mem_requirements and mem_requirements[elements] > available_gb * 0.8:
                print(f"Skipping largest problem {nelx}x{nely}x{nelz} ({elements} elements) due to memory constraints")
                with open(log_file, "a") as f:
                    f.write(f"Skipped {nelx}x{nely}x{nelz} ({elements} elements) due to memory constraints\n")
                continue
        
        # For large problems, adjust parameters
        maxloop = 50   # Fixed 50 iterations for all problem sizes
        tolx = 0.01    # Default tolerance
        
        # Adjust tolerance for larger problems to save time
        if elements > 50000:  # 64x32x32 has 65536 elements
            tolx = 0.02
        
        if elements > 500000:  # 128x64x64 has 524288 elements
            tolx = 0.05
            
        # Log benchmark start
        print(f"\n{'='*60}")
        print(f"Running benchmark {i+1}/{len(problem_sizes)}: {nelx}x{nely}x{nelz} ({elements} elements)")
        print(f"Parameters: maxloop={maxloop}, tolx={tolx}, volfrac={args.volfrac}, penal={args.penal}, rmin={args.rmin}")
        print(f"{'='*60}")
        
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Benchmark {i+1}/{len(problem_sizes)}: {nelx}x{nely}x{nelz} ({elements} elements)\n")
            f.write(f"Parameters: maxloop={maxloop}, tolx={tolx}, volfrac={args.volfrac}, penal={args.penal}, rmin={args.rmin}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Create output directory for this problem size
        output_dir = os.path.join(args.output_dir, f"size_{nelx}x{nely}x{nelz}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Command to run benchmark
        cmd = [
            "python", "main.py",
            "--nelx", str(nelx),
            "--nely", str(nely),
            "--nelz", str(nelz),
            "--volfrac", str(args.volfrac),
            "--penal", str(args.penal),
            "--rmin", str(args.rmin),
            "--tolx", str(tolx),
            "--maxloop", str(maxloop),
            "--benchmark",
            "--save-benchmark",
            "--benchmark-dir", output_dir
        ]
        
        # Run benchmark
        try:
            # Clean up memory before large problems
            if elements > 50000:
                import gc
                gc.collect()
                
                # Force Python to release memory back to OS
                if sys.platform.startswith('win'):
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            
            # Run the benchmark
            start_time = time.time()
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Wait for completion
            stdout, stderr = process.communicate()
            elapsed = time.time() - start_time
            
            # Log output
            with open(os.path.join(output_dir, "stdout.log"), "w") as f:
                f.write(stdout)
            
            with open(os.path.join(output_dir, "stderr.log"), "w") as f:
                f.write(stderr)
            
            if process.returncode != 0:
                print(f"Error running benchmark: {stderr}")
                with open(log_file, "a") as f:
                    f.write(f"ERROR: Benchmark failed in {elapsed:.2f} seconds\n")
                    f.write(f"Exit code: {process.returncode}\n")
                    f.write(f"Error: {stderr}\n")
                continue
            
            # Find benchmark results file
            benchmark_file = os.path.join(output_dir, f"benchmark_size_{elements}_nelx{nelx}_nely{nely}_nelz{nelz}.json")
            if os.path.exists(benchmark_file):
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                results[str(elements)] = benchmark_data
                
                # Log benchmark completion
                print(f"✓ Benchmark completed in {elapsed:.2f} seconds")
                print(f"  Total time: {benchmark_data['total_time_seconds']:.2f}s")
                print(f"  Memory: {benchmark_data.get('peak_memory_mb', 0)/1024:.2f}GB")
                print(f"  Iterations: {benchmark_data.get('iterations', 0)}")
                
                with open(log_file, "a") as f:
                    f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total time: {benchmark_data['total_time_seconds']:.2f}s\n")
                    f.write(f"Memory: {benchmark_data.get('peak_memory_mb', 0)/1024:.2f}GB\n")
                    f.write(f"Iterations: {benchmark_data.get('iterations', 0)}\n")
                    f.write(f"Phases: {benchmark_data.get('phases', {})}\n")
            else:
                print(f"✗ Benchmark results file not found: {benchmark_file}")
                with open(log_file, "a") as f:
                    f.write(f"ERROR: Benchmark results file not found\n")
        
        except Exception as e:
            print(f"Error running benchmark: {e}")
            with open(log_file, "a") as f:
                f.write(f"ERROR: Exception during benchmark: {e}\n")
    
    # Save combined results
    if results:
        output_file = os.path.join(args.output_dir, "large_scale_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark results saved to {output_file}")
    
    return results


def generate_scaling_plots(
    python_results: Dict[str, Any], 
    matlab_results: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Generate scaling plots for the paper.
    
    Args:
        python_results: Python benchmark results
        matlab_results: MATLAB benchmark results
        output_dir: Directory to save plots
    """
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract data
    python_sizes = []
    python_times = []
    python_memory = []
    python_iterations = []
    
    matlab_sizes = []
    matlab_times = []
    matlab_memory = []
    matlab_iterations = []
    
    # Process Python results
    for size_str, data in sorted(python_results.items(), key=lambda x: int(x[0])):
        size = int(size_str)
        python_sizes.append(size)
        python_times.append(data["total_time_seconds"])
        python_memory.append(data.get("peak_memory_mb", 0) / 1024)  # Convert to GB
        python_iterations.append(data.get("iterations", 0))
    
    # Process MATLAB results if available
    if matlab_results:
        for size_str, data in sorted(matlab_results.items(), key=lambda x: int(x[0])):
            size = int(size_str)
            matlab_sizes.append(size)
            matlab_times.append(data["total_time_seconds"])
            matlab_memory.append(data.get("peak_memory_mb", 0) / 1024)  # Convert to GB
            matlab_iterations.append(data.get("iterations", 0))
    
    # Generate plots
    plt.figure(figsize=(12, 9))
    
    # Plot 1: Computation Time vs. Problem Size
    plt.subplot(2, 2, 1)
    plt.plot(python_sizes, python_times, 'o-', label='Python', color='blue')
    if matlab_results:
        plt.plot(matlab_sizes, matlab_times, 's-', label='MATLAB', color='red')
    plt.xlabel('Number of Elements')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computation Time vs. Problem Size')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    # Plot 2: Memory Usage vs. Problem Size
    plt.subplot(2, 2, 2)
    plt.plot(python_sizes, python_memory, 'o-', label='Python', color='blue')
    if matlab_results:
        plt.plot(matlab_sizes, matlab_memory, 's-', label='MATLAB', color='red')
    plt.xlabel('Number of Elements')
    plt.ylabel('Memory Usage (GB)')
    plt.title('Memory Usage vs. Problem Size')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    # Plot 3: Performance ratio (if MATLAB data available)
    if matlab_results:
        # Find common sizes
        common_sizes = set(python_sizes).intersection(matlab_sizes)
        if common_sizes:
            common_sizes = sorted(list(common_sizes))
            ratios = []
            for size in common_sizes:
                py_idx = python_sizes.index(size)
                matlab_idx = matlab_sizes.index(size)
                ratios.append(matlab_times[matlab_idx] / python_times[py_idx])
            
            plt.subplot(2, 2, 3)
            plt.bar(range(len(common_sizes)), ratios)
            plt.xlabel('Problem Size')
            plt.ylabel('MATLAB/Python Time Ratio')
            plt.title('Performance Comparison (MATLAB/Python)')
            plt.grid(True, axis='y')
            plt.xticks(range(len(common_sizes)), [f"{s}" for s in common_sizes])
            plt.axhline(y=1.0, color='r', linestyle='--')
            
            # Add value labels
            for i, v in enumerate(ratios):
                plt.text(i, v + 0.1, f"{v:.2f}x", ha='center')
    
    # Plot 4: Time per iteration
    plt.subplot(2, 2, 4)
    python_time_per_iter = [t/i if i > 0 else 0 for t, i in zip(python_times, python_iterations)]
    plt.plot(python_sizes, python_time_per_iter, 'o-', label='Python', color='blue')
    
    if matlab_results:
        matlab_time_per_iter = [t/i if i > 0 else 0 for t, i in zip(matlab_times, matlab_iterations)]
        plt.plot(matlab_sizes, matlab_time_per_iter, 's-', label='MATLAB', color='red')
    
    plt.xlabel('Number of Elements')
    plt.ylabel('Time per Iteration (seconds)')
    plt.title('Computation Time per Iteration')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'scaling_plots.png'), dpi=300)
    plt.savefig(os.path.join(plots_dir, 'scaling_plots.pdf'))
    print(f"Scaling plots saved to {plots_dir}")
    
    # Generate phase breakdown plots for Python
    if python_results:
        plt.figure(figsize=(12, 6))
        
        # Find all phases
        all_phases = set()
        for data in python_results.values():
            if "phases" in data:
                all_phases.update(data["phases"].keys())
        
        all_phases = sorted(list(all_phases))
        
        # Create stacked bar chart
        problem_labels = []
        phase_data = {phase: [] for phase in all_phases}
        
        for size_str, data in sorted(python_results.items(), key=lambda x: int(x[0])):
            nelx = data.get("problem_size", {}).get("nelx", 0) or int(round(int(size_str)**(1/3)))
            problem_labels.append(f"{nelx}³")
            
            if "phases" in data:
                for phase in all_phases:
                    if phase in data["phases"]:
                        phase_time = data["phases"][phase].get("total_seconds", 0)
                        phase_data[phase].append(phase_time)
                    else:
                        phase_data[phase].append(0)
        
        # Plot stacked bar chart
        bottom = np.zeros(len(problem_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_phases)))
        
        for i, phase in enumerate(all_phases):
            plt.bar(problem_labels, phase_data[phase], bottom=bottom, 
                   label=phase.capitalize(), color=colors[i])
            bottom += phase_data[phase]
        
        plt.xlabel('Problem Size')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Phase Breakdown by Problem Size')
        plt.legend(loc='upper left')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'phase_breakdown.png'), dpi=300)
        plt.savefig(os.path.join(plots_dir, 'phase_breakdown.pdf'))
    
    # Generate a summary table of results
    with open(os.path.join(plots_dir, 'scaling_results.txt'), 'w') as f:
        f.write("Large-Scale Benchmarking Results\n")
        f.write("===============================\n\n")
        
        f.write("Python Implementation\n")
        f.write("--------------------\n")
        f.write("Problem Size | Elements | Time (s) | Memory (GB) | Iterations\n")
        f.write("--------------------------------------------------------\n")
        
        for i, size in enumerate(python_sizes):
            nelx = int(round(size**(1/3)))
            nely = nelx // 2
            nelz = nelx // 2
            f.write(f"{nelx}x{nely}x{nelz} | {size} | {python_times[i]:.2f} | {python_memory[i]:.2f} | {python_iterations[i]}\n")
        
        if matlab_results:
            f.write("\nMATLAB Implementation\n")
            f.write("---------------------\n")
            f.write("Problem Size | Elements | Time (s) | Memory (GB) | Iterations\n")
            f.write("--------------------------------------------------------\n")
            
            for i, size in enumerate(matlab_sizes):
                nelx = int(round(size**(1/3)))
                nely = nelx // 2
                nelz = nelx // 2
                f.write(f"{nelx}x{nely}x{nelz} | {size} | {matlab_times[i]:.2f} | {matlab_memory[i]:.2f} | {matlab_iterations[i]}\n")
            
            # Performance comparison for common sizes
            common_sizes = set(python_sizes).intersection(matlab_sizes)
            if common_sizes:
                f.write("\nPerformance Comparison\n")
                f.write("---------------------\n")
                f.write("Elements | Python (s) | MATLAB (s) | Speedup | Python Memory | MATLAB Memory | Memory Ratio\n")
                f.write("----------------------------------------------------------------------------------\n")
                
                for size in sorted(list(common_sizes)):
                    py_idx = python_sizes.index(size)
                    matlab_idx = matlab_sizes.index(size)
                    speedup = matlab_times[matlab_idx] / python_times[py_idx]
                    mem_ratio = python_memory[py_idx] / matlab_memory[matlab_idx] if matlab_memory[matlab_idx] > 0 else float('inf')
                    
                    f.write(f"{size} | {python_times[py_idx]:.2f} | {matlab_times[matlab_idx]:.2f} | {speedup:.2f}x | " +
                           f"{python_memory[py_idx]:.2f}GB | {matlab_memory[matlab_idx]:.2f}GB | {mem_ratio:.2f}x\n")
    
    print(f"Scaling results summary saved to {os.path.join(plots_dir, 'scaling_results.txt')}")


def main():
    """Run the large-scale benchmark script."""
    args = parse_args()
    
    print(f"Large-Scale Benchmark Script for PyTopo3D")
    print(f"Running on {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python {platform.python_version()}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MATLAB data if provided
    matlab_results = {}
    if args.matlab_data:
        matlab_results = load_matlab_data(args.matlab_data)
    
    # Run benchmarks
    python_results = run_benchmarks(args)
    
    # Generate plots and report
    if python_results:
        generate_scaling_plots(python_results, matlab_results, args.output_dir)
        
        if args.generate_report:
            try:
                from pytopo3d.utils.reporting import generate_benchmark_report
                report_path = os.path.join(args.output_dir, "scaling_report.pdf")
                generate_benchmark_report(python_results, matlab_results, report_path)
                print(f"Benchmark report generated: {report_path}")
            except ImportError:
                print("Could not generate report - reporting module not available")
    
    print("\nLarge-scale benchmarking complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 