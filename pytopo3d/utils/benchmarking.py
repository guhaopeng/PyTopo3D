"""
Benchmarking utilities for 3D topology optimization.

This module provides functions for tracking and analyzing performance metrics
of the optimization process, including timing of different phases, memory usage,
and scalability with problem size.
"""

import json
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import psutil


class BenchmarkTracker:
    """Class for tracking performance metrics during optimization."""

    def __init__(self, track_memory: bool = True, track_detailed_timing: bool = True):
        """
        Initialize the benchmark tracker.

        Parameters
        ----------
        track_memory : bool
            Whether to track memory usage.
        track_detailed_timing : bool
            Whether to track detailed timing for different phases.
        """
        self.track_memory = track_memory
        self.track_detailed_timing = track_detailed_timing
        self.start_time = time.time()
        
        # Initialize tracking data
        self.total_time = 0.0
        self.iteration_times: List[float] = []
        self.detailed_timings: Dict[str, List[float]] = {
            "assembly": [],
            "solve": [],
            "compliance": [],
            "sensitivity": [],
            "filter": [],
            "update": [],
            "misc": []
        }
        
        # Memory tracking
        self.memory_usage: List[float] = []
        self.peak_memory = 0.0
        
        # System info
        self.system_info = self._get_system_info()
        
        # Phase timing
        self.current_phase: Optional[str] = None
        self.phase_start_time = 0.0
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count() or 0,
            "total_memory_gb": psutil.virtual_memory().total / (1024**3)
        }
        return info
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start timing a specific phase of the optimization.

        Parameters
        ----------
        phase_name : str
            Name of the phase to time.
        """
        if not self.track_detailed_timing:
            return
        
        # If we're already timing a phase, end it first
        if self.current_phase:
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_start_time = time.time()
    
    def end_phase(self) -> None:
        """End timing for the current phase."""
        if not self.track_detailed_timing or not self.current_phase:
            return
        
        elapsed = time.time() - self.phase_start_time
        
        if self.current_phase in self.detailed_timings:
            self.detailed_timings[self.current_phase].append(elapsed)
        
        self.current_phase = None
    
    def start_iteration(self) -> None:
        """Start timing a new optimization iteration."""
        if self.track_memory:
            # Record current memory usage
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024**2)  # MB
            self.memory_usage.append(mem)
            self.peak_memory = max(self.peak_memory, mem)
        
        # Reset iteration start time
        self.iteration_start_time = time.time()
    
    def end_iteration(self) -> None:
        """End timing for the current iteration."""
        iteration_time = time.time() - self.iteration_start_time
        self.iteration_times.append(iteration_time)
    
    def finalize(self) -> None:
        """Finalize the benchmarking and calculate overall statistics."""
        self.total_time = time.time() - self.start_time
        
        # If we're still timing a phase, end it
        if self.current_phase:
            self.end_phase()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of benchmark results.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing benchmark metrics.
        """
        # Ensure we've finalized
        if not self.total_time:
            self.finalize()
        
        summary = {
            "total_time_seconds": self.total_time,
            "iterations": len(self.iteration_times),
            "system_info": self.system_info,
        }
        
        if self.iteration_times:
            summary.update({
                "avg_iteration_time_seconds": np.mean(self.iteration_times),
                "max_iteration_time_seconds": np.max(self.iteration_times),
                "min_iteration_time_seconds": np.min(self.iteration_times),
            })
        
        if self.track_detailed_timing:
            phase_summary = {}
            for phase, times in self.detailed_timings.items():
                if times:
                    phase_summary[phase] = {
                        "total_seconds": np.sum(times),
                        "avg_seconds": np.mean(times),
                        "percentage": (np.sum(times) / self.total_time) * 100 if self.total_time else 0
                    }
            summary["phases"] = phase_summary
        
        if self.track_memory:
            summary.update({
                "peak_memory_mb": self.peak_memory,
                "final_memory_mb": self.memory_usage[-1] if self.memory_usage else 0,
                "avg_memory_mb": np.mean(self.memory_usage) if self.memory_usage else 0,
            })
        
        return summary
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save benchmark data to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to save the benchmark data.
        """
        summary = self.get_summary()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


def generate_benchmark_plots(benchmark_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Generate plots from benchmark data.

    Parameters
    ----------
    benchmark_data : Dict[str, Any]
        Dictionary containing benchmark data for various problem sizes.
    output_dir : str
        Directory to save the plots.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot types to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_files = {}
    
    # Extract problem sizes and corresponding metrics
    sizes = []
    total_times = []
    memory_usage = []
    
    for size, data in benchmark_data.items():
        sizes.append(int(size))
        total_times.append(data["total_time_seconds"])
        if "peak_memory_mb" in data:
            memory_usage.append(data["peak_memory_mb"])
    
    # Sort by problem size
    sorted_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in sorted_indices]
    total_times = [total_times[i] for i in sorted_indices]
    
    # Plot execution time vs problem size
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, total_times, 'o-', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Number of Elements')
    plt.ylabel('Execution Time (s)')
    plt.title('Scalability: Execution Time vs Problem Size')
    
    # Calculate and show scaling factor
    if len(sizes) > 1:
        scaling_factor = np.polyfit(np.log10(sizes), np.log10(total_times), 1)[0]
        plt.text(0.05, 0.95, f'Scaling Factor: {scaling_factor:.2f}', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    time_plot_path = os.path.join(output_dir, 'execution_time_scaling.png')
    plt.savefig(time_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files['execution_time'] = time_plot_path
    
    # Plot memory usage vs problem size if available
    if memory_usage:
        memory_usage = [memory_usage[i] for i in sorted_indices]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, memory_usage, 'o-', linewidth=2)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Number of Elements')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('Memory Usage vs Problem Size')
        
        # Calculate and show scaling factor
        if len(sizes) > 1:
            mem_scaling_factor = np.polyfit(np.log10(sizes), np.log10(memory_usage), 1)[0]
            plt.text(0.05, 0.95, f'Scaling Factor: {mem_scaling_factor:.2f}', 
                     transform=plt.gca().transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
        
        memory_plot_path = os.path.join(output_dir, 'memory_usage_scaling.png')
        plt.savefig(memory_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['memory_usage'] = memory_plot_path
    
    # If detailed phase timing is available, generate pie charts
    for size, data in benchmark_data.items():
        if "phases" in data:
            phases = data["phases"]
            labels = []
            values = []
            
            for phase, phase_data in phases.items():
                labels.append(phase)
                values.append(phase_data["total_seconds"])
            
            plt.figure(figsize=(10, 8))
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f'Time Distribution for Size {size}')
            
            pie_plot_path = os.path.join(output_dir, f'time_distribution_{size}.png')
            plt.savefig(pie_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files[f'time_distribution_{size}'] = pie_plot_path
    
    return plot_files


def compare_benchmarks(python_results: Dict[str, Any], matlab_results: Dict[str, Any], 
                       output_dir: str) -> Dict[str, str]:
    """
    Generate comparison plots between Python and MATLAB implementations.

    Parameters
    ----------
    python_results : Dict[str, Any]
        Benchmark results from Python implementation.
    matlab_results : Dict[str, Any]
        Benchmark results from MATLAB implementation.
    output_dir : str
        Directory to save the comparison plots.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot types to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_files = {}
    
    # Extract common problem sizes
    py_sizes = sorted([int(size) for size in python_results.keys()])
    matlab_sizes = sorted([int(size) for size in matlab_results.keys()])
    
    common_sizes = sorted(list(set(py_sizes).intersection(set(matlab_sizes))))
    
    if not common_sizes:
        return plot_files
    
    # Extract execution times
    py_times = [python_results[str(size)]["total_time_seconds"] for size in common_sizes]
    matlab_times = [matlab_results[str(size)]["total_time_seconds"] for size in common_sizes]
    
    # Performance ratio (Python/MATLAB)
    performance_ratio = [py/matlab for py, matlab in zip(py_times, matlab_times)]
    
    # Plot execution time comparison
    plt.figure(figsize=(10, 6))
    
    plt.plot(common_sizes, py_times, 'o-', linewidth=2, label='Python')
    plt.plot(common_sizes, matlab_times, 's-', linewidth=2, label='MATLAB')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Number of Elements')
    plt.ylabel('Execution Time (s)')
    plt.title('Python vs MATLAB: Execution Time')
    plt.legend()
    
    comparison_plot_path = os.path.join(output_dir, 'python_vs_matlab_time.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files['time_comparison'] = comparison_plot_path
    
    # Plot performance ratio
    plt.figure(figsize=(10, 6))
    
    plt.plot(common_sizes, performance_ratio, 'o-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Equal Performance')
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Number of Elements')
    plt.ylabel('Performance Ratio (Python/MATLAB)')
    plt.title('Performance Ratio: Python vs MATLAB')
    
    # Add horizontal line at y=1 to indicate equal performance
    plt.legend()
    
    ratio_plot_path = os.path.join(output_dir, 'performance_ratio.png')
    plt.savefig(ratio_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files['performance_ratio'] = ratio_plot_path
    
    return plot_files 