"""
为每个样本单独绘制综合图
左子图：荷载-位移曲线
右子图：屈曲时刻整体变形
按屈曲类型分文件夹存放
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("results")
DATA_DIR = Path(r"./LHS_Arch_M2fix_v3")

# 读取分析结果
with open(OUTPUT_DIR / "buckling_analysis_results.json", 'r', encoding='utf-8') as f:
    results = json.load(f)

def read_curve_data(sample_id):
    """读取样本的荷载位移数据"""
    folder = DATA_DIR / f"FGM_load_disp_{sample_id}" / "result"

    # 读取节点11数据
    file = folder / "load-disp_11.csv"
    with open(file, 'r') as f:
        lines = f.readlines()
    data_lines = lines[1:-1]

    loads = []
    disps = []
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            loads.append(float(parts[0]) / 1e6)  # MPa
            disps.append(-float(parts[2]))  # 正向位移
    loads = np.array(loads)
    disps = np.array(disps)

    # 找屈曲点索引
    buckling_idx = None
    for i in range(10, len(loads) - 1):
        if loads[i] > loads[i-1] and loads[i] >= loads[i+1]:
            if i + 5 < len(loads):
                subsequent_mean = np.mean(loads[i+1:i+6])
                if subsequent_mean < loads[i]:
                    buckling_idx = i
                    break
    if buckling_idx is None:
        buckling_idx = np.argmax(loads[:len(loads)//3])

    # 找极小值点（类型1）
    min_idx = None
    if buckling_idx is not None and buckling_idx + 30 < len(loads):
        for i in range(buckling_idx + 20, len(loads) - 5):
            if loads[i] < loads[i-1] and loads[i] < loads[i+1]:
                if i + 10 < len(loads):
                    if np.mean(loads[i+5:i+10]) > loads[i]:
                        if (loads[buckling_idx] - loads[i]) > loads[buckling_idx] * 0.10:
                            min_idx = i
                            break

    # 找全局最大点索引
    global_max_idx = np.argmax(loads)

    return loads, disps, buckling_idx, min_idx, global_max_idx


def read_arch_shape(sample_id, buckling_idx):
    """读取拱的整体变形数据"""
    folder = DATA_DIR / f"FGM_load_disp_{sample_id}" / "result"

    # 从input.txt获取几何参数
    with open(folder / "input.txt", 'r') as f:
        params_lines = f.readlines()
    L_arch = float(params_lines[4].strip())
    f_rise = float(params_lines[8].strip())

    # 计算初始节点位置（抛物线拱）
    n_nodes = 21
    x_init = np.array([-L_arch/2 + (i-1)*(L_arch/(n_nodes-1)) for i in range(1, n_nodes+1)])
    y_init = (4*f_rise/(L_arch**2)) * ((L_arch/2)**2 - x_init**2)

    # 读取所有节点在屈曲时刻的位移
    displacements = []
    for node_id in range(1, 22):
        file = folder / f"load-disp_{node_id}.csv"
        with open(file, 'r') as f:
            lines = f.readlines()
        data_lines = lines[1:-1]

        uxs = []
        uys = []
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                uxs.append(float(parts[1]))
                uys.append(float(parts[2]))

        idx = min(buckling_idx, len(uxs)-1)
        displacements.append({
            'ux': uxs[idx],
            'uy': uys[idx]
        })

    x_deformed = x_init + np.array([d['ux'] for d in displacements])
    y_deformed = y_init + np.array([d['uy'] for d in displacements])

    return x_init, y_init, x_deformed, y_deformed, L_arch, f_rise


def spline_interpolate(x_nodes, y_nodes, n_points=200):
    """样条插值"""
    spline = CubicSpline(x_nodes, y_nodes)
    x_smooth = np.linspace(x_nodes.min(), x_nodes.max(), n_points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def plot_sample_figure(sample_id, sample_info, buckling_type, output_folder):
    """绘制单个样本的综合图：左-荷载位移，右-整体变形"""
    loads, disps, buckling_idx, min_idx, global_max_idx = read_curve_data(sample_id)
    x_init, y_init, x_def, y_def, L_arch, f_rise = read_arch_shape(sample_id, buckling_idx)

    # 样条插值
    x_init_smooth, y_init_smooth = spline_interpolate(x_init, y_init)
    x_def_smooth, y_def_smooth = spline_interpolate(x_def, y_def)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    type_name = '完整反拱型' if buckling_type == 1 else '极限荷载平台型'
    color = 'b' if buckling_type == 1 else 'r'

    # ===== 左子图：荷载-位移曲线 =====
    axes[0].plot(disps, loads, color+'-', linewidth=1.5)

    # 标记屈曲点
    buckling_load = loads[buckling_idx]
    buckling_disp = disps[buckling_idx]
    axes[0].scatter([buckling_disp], [buckling_load], color='red', s=120, marker='*',
                    zorder=5, label=f'屈曲点 ({buckling_load:.2f} MPa)')

    # 类型1：标记极小值点和全局最大点
    if buckling_type == 1 and min_idx is not None:
        min_load = loads[min_idx]
        min_disp = disps[min_idx]
        axes[0].scatter([min_disp], [min_load], color='green', s=100, marker='o',
                        zorder=5, label=f'极小值点 ({min_load:.2f} MPa)')

        global_max_load = loads[global_max_idx]
        global_max_disp = disps[global_max_idx]
        axes[0].scatter([global_max_disp], [global_max_load], color='orange', s=80, marker='^',
                        zorder=5, label=f'全局最大 ({global_max_load:.2f} MPa)')

    axes[0].set_xlabel('跨中节点Y方向位移 (m)', fontsize=12)
    axes[0].set_ylabel('荷载 (MPa)', fontsize=12)
    axes[0].set_title('荷载-位移曲线', fontsize=14)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3)

    # ===== 右子图：整体变形 =====
    axes[1].plot(x_init_smooth, y_init_smooth, 'k--', linewidth=2, label='初始形状', alpha=0.7)
    axes[1].plot(x_def_smooth, y_def_smooth, color+'-', linewidth=2.5, label='屈曲时刻形状')
    axes[1].fill_between(x_init_smooth, y_init_smooth, y_def_smooth, alpha=0.2, color=color)

    # 标记关键节点
    axes[1].scatter(x_init[0], y_init[0], color='green', s=80, marker='s', zorder=5, label='拱脚')
    axes[1].scatter(x_init[-1], y_init[-1], color='green', s=80, marker='s', zorder=5)
    axes[1].scatter(x_def[10], y_def[10], color='red', s=100, marker='*', zorder=5, label='跨中节点')

    axes[1].set_xlabel('X坐标 (m)', fontsize=12)
    axes[1].set_ylabel('Y坐标 (m)', fontsize=12)
    axes[1].set_title(f'屈曲时刻整体变形\nL={L_arch:.3f}m, f={f_rise:.4f}m', fontsize=13)
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='datalim')

    # 总标题
    fig.suptitle(f'样本 {sample_id} - {type_name}\n屈曲荷载: {buckling_load:.2f} MPa, 矢跨比: {f_rise/L_arch:.3f}',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_folder / f'{sample_id}.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    # 清理旧文件夹并重新创建
    type1_folder = OUTPUT_DIR / "type1_figures"
    type2_folder = OUTPUT_DIR / "type2_figures"

    # 删除旧文件
    if type1_folder.exists():
        for f in type1_folder.glob("*"):
            f.unlink()
    type1_folder.mkdir(exist_ok=True)

    if type2_folder.exists():
        for f in type2_folder.glob("*"):
            f.unlink()
    type2_folder.mkdir(exist_ok=True)

    # 分类样本
    type1_samples = [r for r in results if r['buckling_type'] == 1]
    type2_samples = [r for r in results if r['buckling_type'] == 2]

    print(f"类型1（完整反拱型）样本数: {len(type1_samples)}")
    print(f"类型2（极限荷载平台型）样本数: {len(type2_samples)}")

    # 绘制类型1样本图
    print("\n绘制类型1样本综合图...")
    for i, sample in enumerate(type1_samples):
        sample_id = sample['sample_id']
        plot_sample_figure(sample_id, sample, 1, type1_folder)
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{len(type1_samples)}")
    print(f"  完成！保存至: {type1_folder}")

    # 绘制类型2样本图
    print("\n绘制类型2样本综合图...")
    for i, sample in enumerate(type2_samples):
        sample_id = sample['sample_id']
        plot_sample_figure(sample_id, sample, 2, type2_folder)
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{len(type2_samples)}")
    print(f"  完成！保存至: {type2_folder}")

    print("\n" + "="*60)
    print("绘制完成统计:")
    print(f"  type1_figures: {len(type1_samples)} 张图")
    print(f"  type2_figures: {len(type2_samples)} 张图")
    print(f"  总计: {len(results)} 张图")
    print("="*60)


if __name__ == "__main__":
    main()