"""
拱结构荷载-变形动画生成脚本

功能：
1. 读取input.txt获取抛物线拱的矢高和跨径
2. 反算21个控制点的原始坐标
3. 根据CSV位移数据绘制拱的荷载-变形动画

作者：Hao Tang
日期：2025-04
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def read_input_file(input_path: str) -> dict:
    """
    读取input.txt文件，获取拱的几何参数

    Args:
        input_path: input.txt文件路径

    Returns:
        参数字典，包含L(跨径)、f_arch(矢高)等
    """
    params = {}
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # 按照LHS_AnsysBatch.py中的写入顺序解析
    param_names = [
        'I0', 'A11', 'B11', 'D11_eq', 'L', 'S', 'b', 'h', 'f_arch',
        'lambda_real', 'KXL', 'KYL', 'KZL', 'KXR', 'KYR', 'KZR'
    ]

    for i, name in enumerate(param_names):
        if i < len(lines):
            params[name] = float(lines[i].strip())

    return params


def calculate_original_coordinates(L: float, f_arch: float, n_points: int = 21) -> np.ndarray:
    """
    根据跨径和矢高反算抛物线拱控制点的原始坐标

    抛物线方程: y = 4f/L² × (L²/4 - x²)

    Args:
        L: 跨径 (m)
        f_arch: 矢高 (m)
        n_points: 控制点数量，默认21

    Returns:
        coords: (n_points, 2) 数组，每行为(x, y)坐标
    """
    coords = np.zeros((n_points, 2))

    for i in range(n_points):
        # x坐标：从 -L/2 到 +L/2 均匀分布
        x = -L / 2 + i * (L / (n_points - 1))
        # y坐标：抛物线方程
        y = (4 * f_arch / (L ** 2)) * ((L / 2) ** 2 - x ** 2)
        coords[i, 0] = x
        coords[i, 1] = y

    return coords


def read_displacement_data(result_dir: str, n_points: int = 21) -> tuple:
    """
    读取21个控制节点的位移CSV文件

    Args:
        result_dir: result文件夹路径
        n_points: 控制点数量

    Returns:
        (time_array, disp_data): 时间步数组和位移数组(n_steps, n_points, 2)
    """
    disp_data_list = []
    time_array = None

    # 按节点顺序读取CSV文件
    for i in range(1, n_points + 1):
        csv_path = os.path.join(result_dir, f'load-disp_{i}.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到位移文件: {csv_path}")

        df = pd.read_csv(csv_path)

        # 获取时间/荷载数组
        if time_array is None:
            time_array = df.iloc[:, 0].values  # TIME列

        # 获取位移数据 (UX, UY)
        ux = df.iloc[:, 1].values  # UX
        uy = df.iloc[:, 2].values  # UY

        # 组合位移
        disp_data_list.append(np.column_stack([ux, uy]))

    # 转换为3D数组: (n_steps, n_points, 2)
    disp_data = np.stack(disp_data_list, axis=1)

    return time_array, disp_data


def create_deformed_coordinates(original_coords: np.ndarray, disp_data: np.ndarray) -> np.ndarray:
    """
    计算变形后的坐标

    Args:
        original_coords: 原始坐标 (n_points, 2)
        disp_data: 位移数据 (n_steps, n_points, 2)

    Returns:
        deformed_coords: 变形后坐标 (n_steps, n_points, 2)
    """
    n_steps = disp_data.shape[0]
    deformed_coords = np.zeros((n_steps, original_coords.shape[0], 2))

    for step in range(n_steps):
        deformed_coords[step] = original_coords + disp_data[step]

    return deformed_coords


def draw_arch(ax, coords: np.ndarray, color: str = 'blue', linewidth: float = 2.0,
              alpha: float = 1.0, label: str = None, draw_points: bool = True):
    """
    绘制拱形状

    Args:
        ax: matplotlib轴对象
        coords: 坐标数组 (n_points, 2)
        color: 颜色
        linewidth: 线宽
        alpha: 透明度
        label: 图例标签
        draw_points: 是否绘制控制点
    """
    # 绘制拱线
    ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth,
            alpha=alpha, label=label)

    # 绘制控制点
    if draw_points:
        ax.scatter(coords[:, 0], coords[:, 1], color=color, s=30, alpha=alpha)


def create_animation(result_dir: str, output_path: str = None,
                     fps: int = 30, scale_factor: float = 1.0, max_frames: int = 200):
    """
    创建荷载-变形动画

    Args:
        result_dir: result文件夹路径（包含input.txt和CSV文件）
        output_path: 输出动画文件路径（如output.gif或output.mp4）
        fps: 帧率
        scale_factor: 位移缩放因子（用于放大显示微小位移）
        max_frames: 最大帧数（超过此数量会进行采样，避免文件过大）
    """
    # 读取参数
    input_path = os.path.join(result_dir, 'input.txt')
    params = read_input_file(input_path)

    L = params['L']
    f_arch = params['f_arch']

    print(f"拱参数: 跨径 L = {L:.4f} m, 矢高 f = {f_arch:.4f} m, 矢跨比 f/L = {f_arch/L:.4f}")

    # 计算原始坐标
    original_coords = calculate_original_coordinates(L, f_arch)
    print(f"控制点坐标已计算，共 {original_coords.shape[0]} 个点")

    # 读取位移数据
    time_array, disp_data = read_displacement_data(result_dir)
    n_steps = len(time_array)
    print(f"位移数据已读取，共 {n_steps} 个时间步")

    # 帧采样：如果帧数超过max_frames，进行均匀采样
    if n_steps > max_frames:
        sample_indices = np.linspace(0, n_steps - 1, max_frames, dtype=int)
        time_array = time_array[sample_indices]
        disp_data = disp_data[sample_indices]
        n_steps = max_frames
        print(f"已采样至 {n_steps} 帧，动画时长 {n_steps/fps:.1f} 秒")

    # 计算变形后坐标（应用缩放因子）
    disp_scaled = disp_data * scale_factor
    deformed_coords = create_deformed_coordinates(original_coords, disp_scaled)

    # 设置图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 计算坐标范围
    x_min = min(original_coords[:, 0].min(), deformed_coords[:, :, 0].min())
    x_max = max(original_coords[:, 0].max(), deformed_coords[:, :, 0].max())
    y_min = min(original_coords[:, 1].min(), deformed_coords[:, :, 1].min())
    y_max = max(original_coords[:, 1].max(), deformed_coords[:, :, 1].max())

    # 添加边界余量
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 绘制原始拱（静态背景）
    draw_arch(ax, original_coords, color='gray', linewidth=1.5, alpha=0.5,
              label='原始形状', draw_points=False)

    # 初始化变形拱
    line, = ax.plot([], [], color='blue', linewidth=2.5, label='变形形状')
    points = ax.scatter([], [], color='blue', s=50)

    # 添加荷载信息文本
    load_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', fontweight='bold')

    # 标题
    ax.set_title(f'抛物线拱荷载-变形动画 (L={L:.2f}m, f={f_arch:.2f}m)', fontsize=14)
    ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        points.set_offsets(np.empty((0, 2)))
        load_text.set_text('')
        return line, points, load_text

    def update(frame):
        # 更新变形拱数据
        coords = deformed_coords[frame]
        line.set_data(coords[:, 0], coords[:, 1])
        points.set_offsets(coords)

        # 更新荷载文本
        load_factor = time_array[frame]
        load_text.set_text(f'荷载因子: {load_factor:.4f}')

        return line, points, load_text

    # 创建动画
    anim = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                         blit=True, interval=1000/fps)

    # 保存动画
    if output_path:
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
            print(f"动画已保存为GIF: {output_path}")
        elif output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"动画已保存为MP4: {output_path}")
        else:
            # 默认保存为GIF
            anim.save(output_path + '.gif', writer='pillow', fps=fps)
            print(f"动画已保存为GIF: {output_path}.gif")
    else:
        # 显示动画
        plt.show()

    plt.close(fig)


def create_animation_with_color_gradient(result_dir: str, output_path: str = None,
                                          fps: int = 30, scale_factor: float = 1.0,
                                          max_frames: int = 200):
    """
    创建带颜色渐变的荷载-变形动画（位移越大颜色越红）

    Args:
        result_dir: result文件夹路径
        output_path: 输出动画文件路径
        fps: 帧率
        scale_factor: 位移缩放因子
        max_frames: 最大帧数（超过此数量会进行采样）
    """
    # 读取参数
    input_path = os.path.join(result_dir, 'input.txt')
    params = read_input_file(input_path)

    L = params['L']
    f_arch = params['f_arch']

    # 计算原始坐标
    original_coords = calculate_original_coordinates(L, f_arch)

    # 读取位移数据
    time_array, disp_data = read_displacement_data(result_dir)
    n_steps = len(time_array)

    # 帧采样：如果帧数超过max_frames，进行均匀采样
    if n_steps > max_frames:
        sample_indices = np.linspace(0, n_steps - 1, max_frames, dtype=int)
        time_array = time_array[sample_indices]
        disp_data = disp_data[sample_indices]
        n_steps = max_frames
        print(f"已采样至 {n_steps} 帧")

    # 计算变形后坐标
    disp_scaled = disp_data * scale_factor
    deformed_coords = create_deformed_coordinates(original_coords, disp_scaled)

    # 计算最大位移（用于颜色映射）
    max_disp = np.max(np.sqrt(disp_scaled[:, :, 0]**2 + disp_scaled[:, :, 1]**2))

    # 设置图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 计算坐标范围
    x_min = min(original_coords[:, 0].min(), deformed_coords[:, :, 0].min())
    x_max = max(original_coords[:, 0].max(), deformed_coords[:, :, 0].max())
    y_min = min(original_coords[:, 1].min(), deformed_coords[:, :, 1].min())
    y_max = max(original_coords[:, 1].max(), deformed_coords[:, :, 1].max())

    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # 绘制原始拱（静态背景）
    draw_arch(ax, original_coords, color='gray', linewidth=1.5, alpha=0.5,
              label='原始形状', draw_points=False)

    # 初始化彩色线段集合
    segments = np.zeros((len(original_coords) - 1, 2, 2))
    colors = np.zeros(len(original_coords) - 1)

    lc = LineCollection([], cmap='coolwarm', linewidths=3)
    lc.set_array(colors)
    ax.add_collection(lc)

    # 初始化点（可选）
    points = ax.scatter([], [], c=[], cmap='coolwarm', s=50, vmin=0, vmax=max_disp)

    # 添加荷载信息文本
    load_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(lc, ax=ax, label='位移大小 (m)')

    ax.set_title(f'抛物线拱荷载-变形动画 (L={L:.2f}m, f={f_arch:.2f}m)', fontsize=14)

    def init():
        lc.set_segments([])
        points.set_offsets(np.empty((0, 2)))
        load_text.set_text('')
        return lc, points, load_text

    def update(frame):
        coords = deformed_coords[frame]

        # 计算线段
        segments = np.stack([coords[:-1], coords[1:]], axis=1)
        lc.set_segments(segments)

        # 计算每个线段的位移大小（取两端平均）
        disp_mag = np.sqrt(disp_scaled[frame, :, 0]**2 + disp_scaled[frame, :, 1]**2)
        segment_disp = (disp_mag[:-1] + disp_mag[1:]) / 2
        lc.set_array(segment_disp)
        lc.set_clim(0, max_disp)

        # 更新点
        points.set_offsets(coords)
        points.set_array(disp_mag)

        # 更新荷载文本
        load_factor = time_array[frame]
        load_text.set_text(f'荷载因子: {load_factor:.4f}')

        return lc, points, load_text

    # 创建动画
    anim = FuncAnimation(fig, update, frames=n_steps, init_func=init,
                         blit=True, interval=1000/fps)

    # 保存动画
    if output_path:
        if output_path.endswith('.gif'):
            anim.save(output_path, writer='pillow', fps=fps)
            print(f"动画已保存为GIF: {output_path}")
        elif output_path.endswith('.mp4'):
            anim.save(output_path, writer='ffmpeg', fps=fps)
            print(f"动画已保存为MP4: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_load_displacement_curve(result_dir: str, node_idx: int = 11,
                                  output_path: str = None):
    """
    绘制特定节点的荷载-位移曲线

    Args:
        result_dir: result文件夹路径
        node_idx: 节点索引（1-21），默认11为拱顶节点
        output_path: 输出图片路径
    """
    # 读取参数
    input_path = os.path.join(result_dir, 'input.txt')
    params = read_input_file(input_path)

    L = params['L']
    f_arch = params['f_arch']

    # 计算原始坐标
    original_coords = calculate_original_coordinates(L, f_arch)
    node_coord = original_coords[node_idx - 1]

    # 读取位移数据
    csv_path = os.path.join(result_dir, f'load-disp_{node_idx}.csv')
    df = pd.read_csv(csv_path)

    load = df.iloc[:, 0].values
    ux = df.iloc[:, 1].values
    uy = df.iloc[:, 2].values

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # X位移
    axes[0].plot(load, ux, 'b-', linewidth=1.5)
    axes[0].set_xlabel('荷载因子', fontsize=12)
    axes[0].set_ylabel('X位移 (m)', fontsize=12)
    axes[0].set_title(f'节点{node_idx} 荷载-X位移曲线\n原始位置: ({node_coord[0]:.4f}, {node_coord[1]:.4f})')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Y位移
    axes[1].plot(load, uy, 'r-', linewidth=1.5)
    axes[1].set_xlabel('荷载因子', fontsize=12)
    axes[1].set_ylabel('Y位移 (m)', fontsize=12)
    axes[1].set_title(f'节点{node_idx} 荷载-Y位移曲线\n原始位置: ({node_coord[0]:.4f}, {node_coord[1]:.4f})')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle(f'抛物线拱 L={L:.2f}m, f={f_arch:.2f}m', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"荷载-位移曲线已保存: {output_path}")
    else:
        plt.show()

    plt.close(fig)


# 默认路径配置
DEFAULT_RESULT_DIR = r"./LHS_Arch_M1_v3\load_disp_1\result"
DEFAULT_OUTPUT_DIR = r"./results/animation2"


def main():
    parser = argparse.ArgumentParser(description='拱结构荷载-变形动画生成工具')
    parser.add_argument('--result_dir', type=str, default=DEFAULT_RESULT_DIR,
                        help=f'result文件夹路径（默认: {DEFAULT_RESULT_DIR}）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出动画文件路径（默认自动保存到results文件夹）')
    parser.add_argument('--fps', type=int, default=30,
                        help='动画帧率（默认30）')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='位移缩放因子（用于放大显示微小位移，默认1.0）')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='最大帧数（超过此数量会进行采样，默认200）')
    parser.add_argument('--mode', type=str, default='animation',
                        choices=['animation', 'color', 'curve'],
                        help='输出模式: animation(基础动画), color(颜色渐变动画), curve(荷载-位移曲线)')
    parser.add_argument('--node', type=int, default=11,
                        help='绘制荷载-位移曲线时的节点索引（1-21，默认11拱顶）')

    args = parser.parse_args()

    # 自动生成输出路径
    output_path = args.output
    if output_path is None:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        sample_name = os.path.basename(os.path.dirname(args.result_dir))
        if args.mode == 'animation':
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f'{sample_name}_animation.gif')
        elif args.mode == 'color':
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f'{sample_name}_color.gif')
        elif args.mode == 'curve':
            output_path = os.path.join(DEFAULT_OUTPUT_DIR, f'{sample_name}_node{args.node}_curve.png')
        print(f"输出将保存到: {output_path}")

    if args.mode == 'animation':
        create_animation(args.result_dir, output_path, args.fps, args.scale, args.max_frames)
    elif args.mode == 'color':
        create_animation_with_color_gradient(args.result_dir, output_path, args.fps, args.scale, args.max_frames)
    elif args.mode == 'curve':
        plot_load_displacement_curve(args.result_dir, args.node, output_path)


if __name__ == '__main__':
    main()