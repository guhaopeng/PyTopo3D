"""
Command-line argument parsing for the 3D topology optimization package.
"""

import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the topology optimization.

    Parameters
    ----------
    args : Optional[List[str]], optional
        Command line arguments, by default None (uses sys.argv[1:])

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
        参数
args : 可选 [List [str]]，可选参数命令行参数，默认为 None（使用 sys.argv [1:]）
返回
argparse.Namespace解析后的命令行参数。
    """
    
    parser = argparse.ArgumentParser(
        description="3D Topology Optimization", #创建参数解析器对象
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, #设置参数帮助信息的格式化类
    )

    # Basic parameters
    basic_group = parser.add_argument_group("Basic parameters") #创建基本参数组
    basic_group.add_argument(
        "--nelx", type=int, default=32, help="Number of elements in x direction"
    )
    basic_group.add_argument(
        "--nely", type=int, default=32, help="Number of elements in y direction"
    )
    basic_group.add_argument(
        "--nelz", type=int, default=32, help="Number of elements in z direction"
    )
    basic_group.add_argument(
        "--volfrac", type=float, default=0.3, help="Volume fraction constraint"
    )
    basic_group.add_argument(
        "--penal", type=float, default=3.0, help="Penalty parameter"
    )
    basic_group.add_argument("--rmin", type=float, default=3.0, help="Filter radius")
    basic_group.add_argument(
        "--disp_thres",
        type=float,
        default=0.5,
        help="Threshold for displaying elements in visualization",  #可视化中显示元素的阈值
    )
    basic_group.add_argument(
        "--tolx",
        type=float,
        default=0.01,
        help="Convergence tolerance on design change",  #设计变化的收敛容差
    )
    basic_group.add_argument(
        "--maxloop",
        type=int,
        default=2000,
        help="Maximum number of iterations",  #最大迭代次数
    )

    # Performance parameters
    performance_group = parser.add_argument_group("Performance parameters") # 性能参数组
    performance_group.add_argument( # 启用 GPU 加速时的默认值为 False（需要 CuPy）
        "--gpu",
        action="store_true",
        default=True,
        help="Enable GPU acceleration when available (requires CuPy)",  #当可用时启用 GPU 加速（需要 CuPy）
    )

    # Output parameters
    output_group = parser.add_argument_group("Output parameters") # 输出参数组
    output_group.add_argument( # 优化设计的输出文件名
        "--output",
        type=str,
        default="optimized_design.npy",
        help="Output filename for the optimized design",  #优化设计的输出文件名
    )
    output_group.add_argument(
        "--export-stl",
        action="store_true",
        help="Export the final optimization result as an STL file",  #导出最终优化结果为 STL 文件
    )
    output_group.add_argument(
        "--stl-level",
        type=float,
        default=0.5,
        help="Contour level for STL export (default: 0.5)",  #导出 STL 文件的轮廓级别（默认值：0.5）
    )
    output_group.add_argument(
        "--smooth-stl",
        action="store_true",
        default=True,
        help="Apply smoothing to the exported STL (default: True)",  #导出 STL 文件后是否应用平滑（默认值：True）
    )
    output_group.add_argument(
        "--smooth-iterations",
        type=int,
        default=5,
        help="Number of smoothing iterations for STL export (default: 5)",  #导出 STL 文件后应用平滑的迭代次数（默认值：5）
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom name for the experiment (optional)",  #实验的自定义名称（可选）
    )
    output_group.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of the experiment (optional)",  #实验的描述（可选）
    )

    # Animation parameters
    animation_group = parser.add_argument_group("Animation parameters")
    animation_group.add_argument( # 创建优化过程的 GIF 动画
        "--create-animation",
        action="store_true",   # 是否创建优化过程的 GIF 动画 （默认值：True）
        help="Create a GIF animation of the optimization process",  #创建优化过程的 GIF 动画
    )
    animation_group.add_argument(
        "--animation-frequency",
        type=int,
        default=10,
        help="Store every N iterations for the animation (default: 10)",  #动画中存储每 N 次迭代（默认值：10）
    )
    animation_group.add_argument(
        "--animation-frames",
        type=int,
        default=50,
        help="Target number of frames to include in the animation (default: 50)",  #动画中包含的目标帧数（默认值：50）
    )
    animation_group.add_argument(
        "--animation-fps",
        type=int,
        default=5,
        help="Frames per second in the animation (default: 5)",  #动画的帧率（默认值：5）
    )

    # Design space parameters
    design_space_group = parser.add_argument_group("Design space parameters")
    design_space_group.add_argument(
        "--design-space-stl",
        type=str,
        help="Path to an STL file defining the design space geometry",  #定义设计空间几何的 STL 文件路径
    )
    design_space_group.add_argument( # 体素化 STL 时，体素中心之间的距离（较小值创建更精细的细节）
        "--pitch", 
        type=float,
        default=1.0,
        help="Distance between voxel centers when voxelizing STL (smaller values create finer detail)",  #当体素化 STL 时，体素中心之间的距离（较小值创建更精细的细节）
    )
    design_space_group.add_argument( # 反转设计空间（将 STL 视为_void 空间而不是设计空间）
        "--invert-design-space", #
        action="store_true",
        help="Invert the design space (treat STL as void space rather than design space)",  #反转设计空间（将 STL 视为_void 空间而不是设计空间）
    )

    # Obstacle related arguments
    # 障碍物相关参数组
    obstacle_group = parser.add_argument_group("Obstacle parameters")
    obstacle_group.add_argument(
        "--obstacle-config", 
        type=str, 
        help="Path to a JSON file defining obstacles"  #定义障碍物的 JSON 文件路径
    )

    # Logging parameters
    # 日志参数组
    log_group = parser.add_argument_group("Logging parameters")
    log_group.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level", 
    )
    log_group.add_argument("--log-file", type=str, default=None, help="Log file path")
    log_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (DEBUG level)",  #启用详细输出（DEBUG 级别）
    )
    log_group.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (WARNING level)"  #抑制输出（WARNING 级别）
    )

    return parser.parse_args(args)


def generate_experiment_name(args: argparse.Namespace) -> str:
    """
    Generate an experiment name from command-line arguments. 
    从命令行参数生成实验名称。

    Parameters
    ----------
    args : argparse.Namespace   
        Command-line arguments.   

    Returns
    -------
    str
        Generated experiment name.  #生成的实验名称
    """
    if args.experiment_name:
        return args.experiment_name  # 如果提供了实验名称，则直接返回

    dims = f"{args.nelx}x{args.nely}x{args.nelz}"  # 设计空间维度（nelx x nely x nelz）

    # Include obstacle info in experiment name 
    obstacle_type = "no_obstacle" # 障碍物类型（默认值：无障碍物）
    if args.obstacle_config:
        obstacle_type = os.path.basename(args.obstacle_config).replace(".json", "")  # 从障碍物配置文件路径中提取障碍物类型（去掉.json 扩展名）

    # Include design space STL info in experiment name if provided 
    design_space = ""
    if hasattr(args, "design_space_stl") and args.design_space_stl:  # 如果提供了设计空间 STL 文件路径
        stl_name = os.path.basename(args.design_space_stl).replace(".stl", "")  # 从 STL 文件路径中提取文件名（去掉.stl 扩展名）
        pitch_info = f"_p{args.pitch}".replace(".", "p")  # 从命令行参数中提取体素化 STL 时的体素中心之间的距离（较小值创建更精细的细节）
        design_space = f"_ds_{stl_name}{pitch_info}"  # 设计空间 STL 文件名和体素中心之间的距离（用于实验名称）

    return f"{dims}_{obstacle_type}{design_space}"  # 实验名称，包含设计空间维度、障碍物类型和设计空间 STL 文件名（如果提供）


def create_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create a configuration dictionary from command-line arguments. 
    从命令行参数创建配置字典。

    Parameters
    ----------
    args : argparse.Namespace 
        Command-line arguments.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.  #配置字典
    """
    config = vars(args)  # 将命令行参数转换为字典
    config["timestamp"] = datetime.now().isoformat()  # 添加当前时间戳（ISO 格式）
    return config
