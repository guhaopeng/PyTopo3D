#!/bin/bash
# Example script to run topology optimization and export the result as an STL file
# 示例脚本，用于运行拓扑优化并将结果导出为 STL 文件

# Activate the conda environment if needed
# source ./activate_pytopo3d.sh

# Run the optimization with STL export
# 运行拓扑优化并导出 STL 文件
python main.py \
    --nelx 60 \
    --nely 20 \
    --nelz 10 \
    --volfrac 0.3 \
    --penal 3.0 \
    --rmin 3.0 \
    --tolx 0.1 \
    --maxloop 1000 \
    --export-stl \
    --stl-level 0.5 \
    --smooth-iterations 5