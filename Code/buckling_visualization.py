"""
拱结构屈曲结果可视化脚本
生成各类分析图表
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicSpline  # 样条插值
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path("results")

# 读取分析结果
with open(OUTPUT_DIR / "buckling_analysis_results.json", 'r', encoding='utf-8') as f:
    results = json.load(f)

df = pd.DataFrame(results)

# 读取统计数据
with open(OUTPUT_DIR / "buckling_statistics.json", 'r', encoding='utf-8') as f:
    stats = json.load(f)

# ============================================================
# 图1: 荷载-位移曲线分类示例
# ============================================================
def plot_typical_curves():
    """绘制两种类型的典型荷载位移曲线"""
    # 选择类型1和类型2的典型样本
    type1_samples = df[df['buckling_type'] == 1]['sample_id'].tolist()
    type2_samples = df[df['buckling_type'] == 2]['sample_id'].tolist()

    # 选择屈曲荷载接近平均值的样本
    type1_mean = stats['type1_buckling_load_stats']['mean']
    type2_mean = stats['type2_buckling_load_stats']['mean']

    type1_df = df[df['buckling_type'] == 1]
    type2_df = df[df['buckling_type'] == 2]

    type1_idx = (type1_df['buckling_load'] - type1_mean).abs().idxmin()
    type2_idx = (type2_df['buckling_load'] - type2_mean).abs().idxmin()

    type1_sample = int(type1_df.loc[type1_idx, 'sample_id'])
    type2_sample = int(type2_df.loc[type2_idx, 'sample_id'])

    # 读取原始数据
    DATA_DIR = Path(r"./LHS_Arch_M2fix_v3")

    def read_curve(sample_id):
        folder = DATA_DIR / f"FGM_load_disp_{sample_id}" / "result"
        file = folder / "load-disp_11.csv"
        with open(file, 'r') as f:
            lines = f.readlines()
        data_lines = lines[1:-1]
        data = []
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                load = float(parts[0]) / 1e6  # 转换为MPa
                uy = -float(parts[2])  # 转为正向位移
                data.append([load, uy])
        return np.array(data)

    curve1 = read_curve(type1_sample)
    curve2 = read_curve(type2_sample)

    # 获取该样本的屈曲点和极小值点信息
    type1_info = type1_df.loc[type1_idx]
    type2_info = type2_df.loc[type2_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 类型1
    axes[0].plot(curve1[:, 1], curve1[:, 0], 'b-', linewidth=1.5)
    # 标记屈曲点（第一个局部极大值）
    # 注意：buckling_disp 已经是正值（分析脚本中 disp = -uy），直接使用
    buckling_disp1 = type1_info['buckling_disp']
    buckling_load1 = type1_info['buckling_load'] / 1e6
    axes[0].scatter([buckling_disp1], [buckling_load1], color='red', s=100,
                    marker='*', zorder=5, label=f'屈曲点({buckling_load1:.2f}MPa)')
    # 标记全局最大值点（反拱后）- 使用curve数据中的实际位置
    global_max_idx1 = int(type1_info['global_max_idx'])
    global_max_disp1 = curve1[global_max_idx1, 1]
    global_max_load1 = curve1[global_max_idx1, 0]  # 使用curve中的实际荷载值
    axes[0].scatter([global_max_disp1], [global_max_load1], color='orange', s=80,
                    marker='^', zorder=5, label=f'全局最大({global_max_load1:.2f}MPa)')
    # 标记极小值点 - 需要在原始曲线数据中找到对应的索引
    if type1_info['min_load'] is not None and type1_info['min_disp'] is not None:
        min_disp1 = type1_info['min_disp']  # 已经是正值，直接使用
        min_load1 = type1_info['min_load'] / 1e6
        axes[0].scatter([min_disp1], [min_load1], color='green', s=100,
                        marker='o', zorder=5, label=f'极小值点({min_load1:.2f}MPa)')
    axes[0].set_xlabel('跨中节点Y方向位移绝对值 (m)', fontsize=12)
    axes[0].set_ylabel('荷载 (MPa)', fontsize=12)
    axes[0].set_title(f'类型1：完整反拱型 (样本{type1_sample})', fontsize=14)
    axes[0].legend(fontsize=9, loc='best')
    axes[0].grid(True, alpha=0.3)

    # 类型2
    axes[1].plot(curve2[:, 1], curve2[:, 0], 'r-', linewidth=1.5)
    # 标记屈曲点（全局最大值）
    buckling_disp2 = type2_info['buckling_disp']  # 已经是正值，直接使用
    buckling_load2 = type2_info['buckling_load'] / 1e6
    axes[1].scatter([buckling_disp2], [buckling_load2], color='red', s=100,
                    marker='*', zorder=5, label=f'屈曲点({buckling_load2:.2f}MPa)')
    axes[1].set_xlabel('跨中节点Y方向位移绝对值 (m)', fontsize=12)
    axes[1].set_ylabel('荷载 (MPa)', fontsize=12)
    axes[1].set_title(f'类型2：极限荷载平台型 (样本{type2_sample})', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_typical_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图1已保存: fig1_typical_curves.png")

# ============================================================
# 图2: 屈曲荷载与长细比的散点图
# ============================================================
def plot_buckling_vs_slenderness():
    """屈曲荷载与长细比的关系"""
    fig, ax = plt.subplots(figsize=(10, 6))

    type1 = df[df['buckling_type'] == 1]
    type2 = df[df['buckling_type'] == 2]

    ax.scatter(type1['lambda_real'], type1['buckling_load']/1e6,
               c='blue', alpha=0.6, s=50, label=f'类型1 (n={len(type1)})')
    ax.scatter(type2['lambda_real'], type2['buckling_load']/1e6,
               c='red', alpha=0.6, s=50, label=f'类型2 (n={len(type2)})')

    ax.set_xlabel('长细比 λ', fontsize=12)
    ax.set_ylabel('屈曲荷载 (MPa)', fontsize=12)
    ax.set_title('屈曲荷载与长细比的关系', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_buckling_vs_slenderness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图2已保存: fig2_buckling_vs_slenderness.png")

# ============================================================
# 图3: 屈曲荷载与矢跨比的散点图
# ============================================================
def plot_buckling_vs_rise_span():
    """屈曲荷载与矢跨比的关系"""
    fig, ax = plt.subplots(figsize=(10, 6))

    type1 = df[df['buckling_type'] == 1]
    type2 = df[df['buckling_type'] == 2]

    ax.scatter(type1['rise_span_ratio'], type1['buckling_load']/1e6,
               c='blue', alpha=0.6, s=50, label=f'类型1 (n={len(type1)})')
    ax.scatter(type2['rise_span_ratio'], type2['buckling_load']/1e6,
               c='red', alpha=0.6, s=50, label=f'类型2 (n={len(type2)})')

    ax.set_xlabel('矢跨比 f/L', fontsize=12)
    ax.set_ylabel('屈曲荷载 (MPa)', fontsize=12)
    ax.set_title('屈曲荷载与矢跨比的关系', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_buckling_vs_rise_span.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图3已保存: fig3_buckling_vs_rise_span.png")

# ============================================================
# 图4: 两类样本的参数分布对比
# ============================================================
def plot_parameter_comparison():
    """两类样本的关键参数分布对比"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    type1 = df[df['buckling_type'] == 1]
    type2 = df[df['buckling_type'] == 2]

    # 长细比分布
    axes[0, 0].hist(type1['lambda_real'], bins=15, alpha=0.6, color='blue', label='类型1')
    axes[0, 0].hist(type2['lambda_real'], bins=15, alpha=0.6, color='red', label='类型2')
    axes[0, 0].set_xlabel('长细比 λ', fontsize=11)
    axes[0, 0].set_ylabel('样本数', fontsize=11)
    axes[0, 0].set_title('长细比分布对比', fontsize=12)
    axes[0, 0].legend(fontsize=10)

    # 矢跨比分布
    axes[0, 1].hist(type1['rise_span_ratio'], bins=15, alpha=0.6, color='blue', label='类型1')
    axes[0, 1].hist(type2['rise_span_ratio'], bins=15, alpha=0.6, color='red', label='类型2')
    axes[0, 1].set_xlabel('矢跨比 f/L', fontsize=11)
    axes[0, 1].set_ylabel('样本数', fontsize=11)
    axes[0, 1].set_title('矢跨比分布对比', fontsize=12)
    axes[0, 1].legend(fontsize=10)

    # 屈曲荷载分布
    axes[1, 0].hist(type1['buckling_load']/1e6, bins=15, alpha=0.6, color='blue', label='类型1')
    axes[1, 0].hist(type2['buckling_load']/1e6, bins=15, alpha=0.6, color='red', label='类型2')
    axes[1, 0].set_xlabel('屈曲荷载 (MPa)', fontsize=11)
    axes[1, 0].set_ylabel('样本数', fontsize=11)
    axes[1, 0].set_title('屈曲荷载分布对比', fontsize=12)
    axes[1, 0].legend(fontsize=10)

    # 等效弯曲刚度分布
    axes[1, 1].hist(type1['D11_eq'], bins=15, alpha=0.6, color='blue', label='类型1')
    axes[1, 1].hist(type2['D11_eq'], bins=15, alpha=0.6, color='red', label='类型2')
    axes[1, 1].set_xlabel('等效弯曲刚度 D11_eq', fontsize=11)
    axes[1, 1].set_ylabel('样本数', fontsize=11)
    axes[1, 1].set_title('等效弯曲刚度分布对比', fontsize=12)
    axes[1, 1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图4已保存: fig4_parameter_comparison.png")

# ============================================================
# 图5: 屈曲荷载箱线图对比
# ============================================================
def plot_boxplot_comparison():
    """屈曲荷载箱线图"""
    fig, ax = plt.subplots(figsize=(8, 6))

    type1_loads = df[df['buckling_type'] == 1]['buckling_load']/1e6
    type2_loads = df[df['buckling_type'] == 2]['buckling_load']/1e6

    box_data = [type1_loads, type2_loads]
    bp = ax.boxplot(box_data, labels=['类型1\n(完整反拱型)', '类型2\n(极限荷载平台型)'],
                    patch_artist=True)

    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax.set_ylabel('屈曲荷载 (MPa)', fontsize=12)
    ax.set_title('两类屈曲行为的屈曲荷载对比', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    ax.text(1, stats['type1_buckling_load_stats']['mean']/1e6 + 1,
            f"均值={stats['type1_buckling_load_stats']['mean']/1e6:.2f}",
            ha='center', fontsize=10)
    ax.text(2, stats['type2_buckling_load_stats']['mean']/1e6 + 1,
            f"均值={stats['type2_buckling_load_stats']['mean']/1e6:.2f}",
            ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_boxplot_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图5已保存: fig5_boxplot_comparison.png")

# ============================================================
# 图6: 类型1和类型2屈曲时的整体变形对比
# ============================================================
def plot_deformation_comparison():
    """绘制拱在屈曲时刻的整体变形，对比初始形状与屈曲形状"""
    DATA_DIR = Path(r"./LHS_Arch_M2fix_v3")

    # 选择类型1和类型2的典型样本（屈曲荷载接近平均值）
    type1_df = df[df['buckling_type'] == 1]
    type2_df = df[df['buckling_type'] == 2]

    type1_mean = stats['type1_buckling_load_stats']['mean']
    type2_mean = stats['type2_buckling_load_stats']['mean']

    type1_idx = (type1_df['buckling_load'] - type1_mean).abs().idxmin()
    type2_idx = (type2_df['buckling_load'] - type2_mean).abs().idxmin()

    type1_sample = int(type1_df.loc[type1_idx, 'sample_id'])
    type2_sample = int(type2_df.loc[type2_idx, 'sample_id'])

    type1_info = type1_df.loc[type1_idx]
    type2_info = type2_df.loc[type2_idx]

    def read_arch_shape(sample_id, buckling_load_target):
        """读取拱的初始形状和屈曲时刻的变形形状"""
        folder = DATA_DIR / f"FGM_load_disp_{sample_id}" / "result"

        # 从input.txt获取几何参数
        with open(folder / "input.txt", 'r') as f:
            params_lines = f.readlines()
        L_arch = float(params_lines[4].strip())  # 第5行是L
        f_rise = float(params_lines[8].strip())  # 第9行是f_arch

        # 计算初始节点位置（抛物线拱）
        n_nodes = 21
        x_init = np.array([-L_arch/2 + (i-1)*(L_arch/(n_nodes-1)) for i in range(1, n_nodes+1)])
        y_init = (4*f_rise/(L_arch**2)) * ((L_arch/2)**2 - x_init**2)

        # 读取所有节点的位移数据
        displacements = []
        buckling_idx_global = None

        for node_id in range(1, 22):  # 节点1到21
            file = folder / f"load-disp_{node_id}.csv"
            with open(file, 'r') as f:
                lines = f.readlines()
            data_lines = lines[1:-1]  # 去掉header和最后一行

            # 找到屈曲时刻对应的位移数据
            loads = []
            uxs = []
            uys = []
            for line in data_lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    loads.append(float(parts[0]))
                    uxs.append(float(parts[1]))
                    uys.append(float(parts[2]))

            loads = np.array(loads)
            uxs = np.array(uxs)
            uys = np.array(uys)

            # 使用节点11（跨中）来确定屈曲时刻索引
            if node_id == 11:
                # 找屈曲时刻索引（第一个局部极大值）
                for i in range(10, len(loads) - 1):
                    if loads[i] > loads[i-1] and loads[i] >= loads[i+1]:
                        if i + 5 < len(loads):
                            subsequent_mean = np.mean(loads[i+1:i+6])
                            if subsequent_mean < loads[i]:
                                buckling_idx_global = i
                                break
                if buckling_idx_global is None:
                    buckling_idx_global = 100  # 默认取一个较早的点

            displacements.append({
                'ux': uxs,
                'uy': uys
            })

        # 计算屈曲时刻的节点位置
        x_deformed = x_init + np.array([d['ux'][buckling_idx_global] for d in displacements])
        y_deformed = y_init + np.array([d['uy'][buckling_idx_global] for d in displacements])

        return x_init, y_init, x_deformed, y_deformed, L_arch, f_rise

    # 读取两个样本的拱形状
    x1_init, y1_init, x1_def, y1_def, L1, f1 = read_arch_shape(type1_sample, type1_info['buckling_load'])
    x2_init, y2_init, x2_def, y2_def, L2, f2 = read_arch_shape(type2_sample, type2_info['buckling_load'])

    # 使用样条插值生成更密集的点，使曲线更平滑
    def spline_interpolate(x_nodes, y_nodes, n_points=200):
        """使用三次样条插值生成平滑曲线"""
        # 创建样条插值函数
        spline = CubicSpline(x_nodes, y_nodes)
        # 生成更密集的x坐标
        x_smooth = np.linspace(x_nodes.min(), x_nodes.max(), n_points)
        y_smooth = spline(x_smooth)
        return x_smooth, y_smooth

    # 对初始形状和变形形状进行样条插值
    x1_init_smooth, y1_init_smooth = spline_interpolate(x1_init, y1_init)
    x1_def_smooth, y1_def_smooth = spline_interpolate(x1_def, y1_def)
    x2_init_smooth, y2_init_smooth = spline_interpolate(x2_init, y2_init)
    x2_def_smooth, y2_def_smooth = spline_interpolate(x2_def, y2_def)

    # 绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 类型1 - 使用样条插值后的平滑曲线
    axes[0].plot(x1_init_smooth, y1_init_smooth, 'k--', linewidth=2, label='初始形状', alpha=0.7)
    axes[0].plot(x1_def_smooth, y1_def_smooth, 'b-', linewidth=2.5, label='屈曲时刻形状')
    # 使用平滑曲线进行填充
    axes[0].fill_between(x1_init_smooth, y1_init_smooth, y1_def_smooth, alpha=0.2, color='blue')
    # 标记节点（使用原始节点位置）
    axes[0].scatter(x1_init[0], y1_init[0], color='green', s=80, marker='s', zorder=5, label='左拱脚')
    axes[0].scatter(x1_init[-1], y1_init[-1], color='green', s=80, marker='s', zorder=5, label='右拱脚')
    axes[0].scatter(x1_init[10], y1_init[10], color='orange', s=60, marker='o', zorder=5, label='控制点', alpha=0.6)
    axes[0].scatter(x1_def[10], y1_def[10], color='red', s=100, marker='*', zorder=5, label='跨中节点')
    axes[0].set_xlabel('X坐标 (m)', fontsize=12)
    axes[0].set_ylabel('Y坐标 (m)', fontsize=12)
    axes[0].set_title(f'类型1：完整反拱型 (样本{type1_sample})\nL={L1:.3f}m, f={f1:.4f}m, 屈曲荷载={type1_info["buckling_load"]/1e6:.2f}MPa', fontsize=13)
    axes[0].legend(fontsize=10, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='datalim')

    # 类型2 - 使用样条插值后的平滑曲线
    axes[1].plot(x2_init_smooth, y2_init_smooth, 'k--', linewidth=2, label='初始形状', alpha=0.7)
    axes[1].plot(x2_def_smooth, y2_def_smooth, 'r-', linewidth=2.5, label='屈曲时刻形状')
    # 使用平滑曲线进行填充
    axes[1].fill_between(x2_init_smooth, y2_init_smooth, y2_def_smooth, alpha=0.2, color='red')
    # 标记节点（使用原始节点位置）
    axes[1].scatter(x2_init[0], y2_init[0], color='green', s=80, marker='s', zorder=5, label='左拱脚')
    axes[1].scatter(x2_init[-1], y2_init[-1], color='green', s=80, marker='s', zorder=5, label='右拱脚')
    axes[1].scatter(x2_init[10], y2_init[10], color='orange', s=60, marker='o', zorder=5, label='控制点', alpha=0.6)
    axes[1].scatter(x2_def[10], y2_def[10], color='red', s=100, marker='*', zorder=5, label='跨中节点')
    axes[1].set_xlabel('X坐标 (m)', fontsize=12)
    axes[1].set_ylabel('Y坐标 (m)', fontsize=12)
    axes[1].set_title(f'类型2：极限荷载平台型 (样本{type2_sample})\nL={L2:.3f}m, f={f2:.4f}m, 屈曲荷载={type2_info["buckling_load"]/1e6:.2f}MPa', fontsize=13)
    axes[1].legend(fontsize=10, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_deformation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("图6已保存: fig6_deformation_comparison.png")

# 运行所有绘图函数
print("开始生成可视化图表...")
plot_typical_curves()
plot_buckling_vs_slenderness()
plot_buckling_vs_rise_span()
plot_parameter_comparison()
plot_boxplot_comparison()
plot_deformation_comparison()  # 新增：整体变形对比图
print("\n所有图表已生成完成！")