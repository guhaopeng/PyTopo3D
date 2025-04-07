#!/usr/bin/env python3
"""
MATLAB Benchmark Converter for PyTopo3D.

This script converts MATLAB benchmark results to the JSON format used by PyTopo3D's
benchmarking tools. It helps with integrating MATLAB performance data for comparison.

Expected MATLAB benchmark format:
- CSV file with columns: Size,Elements,Time,Memory
- Size: Size parameter (e.g., nelx for cube)
- Elements: Total number of elements
- Time: Total execution time in seconds
- Memory: Peak memory usage in MB (optional)
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert MATLAB benchmark data to PyTopo3D JSON format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to MATLAB benchmark CSV file",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        default="matlab_benchmarks.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--system-info",
        type=str,
        help="Path to JSON file with MATLAB system info (optional)",
    )
    
    return parser.parse_args()


def read_matlab_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Read MATLAB benchmark data from CSV.
    
    Args:
        csv_path: Path to CSV file with MATLAB benchmark data
        
    Returns:
        List of dictionaries with benchmark data for each size
    """
    results = []
    
    try:
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Check if headers match expected format
                if not all(key in row for key in ['Size', 'Elements', 'Time']):
                    print("Warning: CSV headers don't match expected format.")
                    print("Expected: Size, Elements, Time, Memory (optional)")
                    print(f"Found: {list(row.keys())}")
                
                result = {
                    'size': int(row['Size']),
                    'elements': int(row['Elements']),
                    'time': float(row['Time']),
                }
                
                # Add memory if available
                if 'Memory' in row and row['Memory']:
                    result['memory'] = float(row['Memory'])
                
                results.append(result)
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []
    
    return results


def read_system_info(json_path: str) -> Dict[str, Any]:
    """
    Read system information from JSON file.
    
    Args:
        json_path: Path to JSON file with system information
        
    Returns:
        Dictionary with system info or default values if not available
    """
    if not json_path or not os.path.exists(json_path):
        # Return default system info
        return {
            "platform": "MATLAB",
            "processor": "Unknown",
            "cpu_count": 0,
            "total_memory_gb": 0,
        }
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading system info file: {e}")
        # Return default system info
        return {
            "platform": "MATLAB",
            "processor": "Unknown",
            "cpu_count": 0,
            "total_memory_gb": 0,
        }


def convert_to_pytopo3d_format(
    matlab_data: List[Dict[str, Any]], 
    system_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert MATLAB benchmark data to PyTopo3D format.
    
    Args:
        matlab_data: List of dictionaries with MATLAB benchmark data
        system_info: Dictionary with system information
        
    Returns:
        Dictionary with benchmark data in PyTopo3D format
    """
    result = {}
    
    for entry in matlab_data:
        size = entry['size']
        elements = entry['elements']
        time = entry['time']
        
        size_key = str(elements)
        
        result[size_key] = {
            "total_time_seconds": time,
            "system_info": system_info,
            "matlab_size_param": size,
        }
        
        # Add memory info if available
        if 'memory' in entry:
            result[size_key]["peak_memory_mb"] = entry['memory']
    
    return result


def main():
    """Convert MATLAB benchmark data to PyTopo3D format."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    
    # Read MATLAB data
    matlab_data = read_matlab_csv(args.input_file)
    if not matlab_data:
        print("Error: No valid data found in input file")
        return 1
    
    # Read system info
    system_info = read_system_info(args.system_info)
    
    # Convert to PyTopo3D format
    result = convert_to_pytopo3d_format(matlab_data, system_info)
    
    # Save output file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Successfully converted MATLAB benchmark data to {args.output_file}")
        print(f"Found data for {len(result)} problem sizes: {sorted([int(k) for k in result.keys()])}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 