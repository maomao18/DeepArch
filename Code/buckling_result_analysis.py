"""
拱结构屈曲结果分析脚本
分析LHS样本的屈曲行为，分类并统计
v3.0更新：
- 参数格式更新为9个参数（I0, A11, B11, D11_eq, f_L, lambda_real, b_h, etaRotL, etaRotR）
- 添加截断逻辑：对有后屈曲的数据截断到屈曲点与反拱点的左三等分点
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

# 数据路径
DATA_DIR = Path(r"./LHS_Arch_M3_v3")
OUTPUT_DIR = Path("results")


def find_truncation_index(disp, buckling_idx, min_idx):
    """
    计算截断点索引

    截断规则：
    - 找屈曲点（上极值点）buckling_idx
    - 找反拱点（下极值点）min_idx
    - 截断位移 = buckling_disp + (min_disp - buckling_disp) / 3
    - 找到对应的截断索引

    Args:
        disp: 位移数组（正值）
        buckling_idx: 屈曲点索引
        min_idx: 反拱点索引

    Returns:
        truncate_idx: 截断点索引
    """
    buckling_disp = disp[buckling_idx]
    min_disp = disp[min_idx]

    # 计算截断位移（屈曲位移与反拱位移的左三等分点）
    truncate_disp = buckling_disp + (min_disp - buckling_disp) / 3

    # 在屈曲点和反拱点之间找截断索引
    # 找位移最接近truncate_disp的点
    search_range = disp[buckling_idx:min_idx+1]
    if len(search_range) == 0:
        return buckling_idx

    # 找最接近truncate_disp的索引（相对索引）
    relative_idx = np.argmin(np.abs(search_range - truncate_disp))
    truncate_idx = buckling_idx + relative_idx

    return truncate_idx


def truncate_load_disp_curve(load, uy, buckling_info):
    """
    对荷载-位移曲线进行截断处理

    情况1（有后屈曲）：截断到屈曲点与反拱点的左三等分点
    情况2（无后屈曲）：去掉最后一行不收敛数据

    Args:
        load: 荷载数组
        uy: Y位移数组（负值）
        buckling_info: classify_buckling_type返回的分类信息

    Returns:
        load_truncated: 截断后的荷载数组
        uy_truncated: 截断后的位移数组
        truncate_idx: 截断点索引（用于记录）
    """
    disp = -uy  # 转为正值

    if buckling_info['type'] == 1 and buckling_info['min_load'] is not None:
        # 情况1：有后屈曲，需要截断
        buckling_idx = buckling_info['buckling_idx']
        # 从buckling_info中获取min_idx（需要添加到classify_buckling_type的返回值中）
        # 这里需要重新计算min_idx
        min_idx_absolute = None
        for i in range(buckling_idx + 20, len(load) - 5):
            if load[i] < load[i-1] and load[i] < load[i+1]:
                if i + 10 < len(load):
                    subsequent_trend = np.mean(load[i+5:i+10])
                    if subsequent_trend > load[i]:
                        min_idx_absolute = i
                        break

        if min_idx_absolute is not None:
            truncate_idx = find_truncation_index(disp, buckling_idx, min_idx_absolute)
            load_truncated = load[:truncate_idx+1]
            uy_truncated = uy[:truncate_idx+1]
            return load_truncated, uy_truncated, truncate_idx

    # 情况2：无后屈曲，去掉最后一行
    return load[:-1], uy[:-1], len(load) - 2

def read_sample_data(sample_folder):
    """读取单个样本的荷载位移数据和参数

    v3.0更新：
    - 参数格式：I0, A11, B11, D11_eq, f_L, lambda_real, b_h, etaRotL, etaRotR
    - 添加截断处理逻辑
    """
    result_dir = sample_folder / "result"

    # 读取节点11的荷载位移数据
    load_disp_file = result_dir / "load-disp_11.csv"
    if not load_disp_file.exists():
        return None

    # 读取数据
    with open(load_disp_file, 'r') as f:
        lines = f.readlines()

    # 先读取所有数据（去掉header）
    data_lines = lines[1:]  # 去掉header

    if len(data_lines) < 10:
        return None

    # 解析数据
    data = []
    for line in data_lines:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            try:
                load = float(parts[0])  # TIME = 荷载
                ux = float(parts[1])    # X位移
                uy = float(parts[2])    # Y位移
                data.append([load, ux, uy])
            except ValueError:
                continue

    if len(data) < 10:
        return None

    data = np.array(data)
    load = data[:, 0]
    ux = data[:, 1]
    uy = data[:, 2]

    # 分类屈曲类型并获取截断信息
    buckling_info = classify_buckling_type_for_truncation(load, uy)

    # 应用截断逻辑
    load_truncated, uy_truncated, truncate_idx = truncate_load_disp_curve(load, uy, buckling_info)

    # 读取特征参数（新格式：9个参数）
    input_file = result_dir / "input.txt"
    if not input_file.exists():
        return None

    with open(input_file, 'r') as f:
        params_lines = f.readlines()

    params = {}
    param_names = ['I0', 'A11', 'B11', 'D11_eq', 'f_L', 'lambda_real', 'b_h', 'etaRotL', 'etaRotR']
    for i, name in enumerate(param_names):
        if i < len(params_lines):
            try:
                params[name] = float(params_lines[i].strip())
            except ValueError:
                params[name] = None

    return {
        'load': load_truncated,
        'ux': ux[:len(load_truncated)],  # 同步截断ux
        'uy': uy_truncated,
        'params': params,
        'buckling_info': buckling_info,
        'truncate_idx': truncate_idx
    }


def classify_buckling_type_for_truncation(load, uy):
    """
    分类屈曲类型（用于截断处理）
    返回包含buckling_idx的信息，供截断函数使用
    """
    disp = -uy  # 转为正向位移

    # 找第一个局部极大值作为屈曲点
    buckling_idx, buckling_load = find_first_local_maximum_idx(load)
    buckling_disp = disp[buckling_idx]

    # 寻找极小值点（在屈曲点之后）
    has_minimum = False
    min_idx_absolute = None
    min_load = None
    min_disp = None

    if len(load) > buckling_idx + 30:
        for i in range(buckling_idx + 20, len(load) - 5):
            if load[i] < load[i-1] and load[i] < load[i+1]:
                if i + 10 < len(load):
                    subsequent_trend = np.mean(load[i+5:i+10])
                    if subsequent_trend > load[i]:
                        if (buckling_load - load[i]) > buckling_load * 0.10:
                            has_minimum = True
                            min_idx_absolute = i
                            min_load = load[i]
                            min_disp = disp[i]
                            break

    # 分类
    if has_minimum:
        buckling_type = 1  # 完整反拱型
    else:
        buckling_type = 2  # 无极小值型

    return {
        'type': buckling_type,
        'buckling_idx': buckling_idx,
        'buckling_load': buckling_load,
        'buckling_disp': buckling_disp,
        'min_idx': min_idx_absolute,
        'min_load': min_load,
        'min_disp': min_disp,
        'max_disp': disp.max(),
        'final_load': load[-1],
        'final_disp': disp[-1],
        'global_max_load': load.max(),
        'global_max_idx': int(np.argmax(load))
    }


def find_first_local_maximum_idx(load):
    """
    找第一个局部极大值点（屈曲点）

    返回：(极大值索引, 极大值荷载)
    """
    for i in range(10, len(load) - 1):
        if load[i] > load[i-1] and load[i] >= load[i+1]:
            if i + 5 < len(load):
                subsequent_mean = np.mean(load[i+1:i+6])
                if subsequent_mean < load[i]:
                    return i, load[i]
            else:
                return i, load[i]

    max_idx = np.argmax(load)
    for i in range(max_idx):
        if load[i] > load[i+1]:
            return i, load[i]

    return max_idx, load[max_idx]


def find_first_local_maximum(load):
    """
    找第一个局部极大值点（屈曲点）
    局部极大值定义：load[i] > load[i-1] 且 load[i] >= load[i+1]

    返回：(极大值索引, 极大值荷载)
    """
    # 遍历数据找第一个局部极大值
    # 从第10个点开始，避免初始噪声
    for i in range(10, len(load) - 1):
        # 局部极大值条件：比前一个大，且比后一个大或相等
        if load[i] > load[i-1] and load[i] >= load[i+1]:
            # 进一步验证：检查后续5个点是否有明显下降趋势
            if i + 5 < len(load):
                subsequent_mean = np.mean(load[i+1:i+6])
                if subsequent_mean < load[i]:
                    return i, load[i]
            else:
                # 如果数据不够长，直接返回
                return i, load[i]

    # 如果没有找到明显的局部极大值，找最大值下降前的点
    max_idx = np.argmax(load)
    # 从前往后找第一个下降点
    for i in range(max_idx):
        if load[i] > load[i+1]:
            return i, load[i]

    return max_idx, load[max_idx]


def classify_buckling_type(load, uy):
    """
    分类屈曲类型
    类型1：有极大值和极小值点（完整反拱）
    类型2：只有极大值点，无极小值点

    注意：UY为负值（向下位移），我们分析荷载-UY的关系

    关键修正：屈曲荷载是第一个局部极大值，而非全局最大值
    因为情况一反拱后荷载可能超过屈曲点荷载
    """
    # 将UY转换为正值便于分析（位移绝对值）
    disp = -uy  # 转为正向位移

    # 找第一个局部极大值作为屈曲点
    buckling_idx, buckling_load = find_first_local_maximum(load)
    buckling_disp = disp[buckling_idx]

    # 计算切线刚度（数值微分）用于验证
    # 切线刚度 dP/dδ 在屈曲点应为零或接近零
    if buckling_idx > 0 and buckling_idx < len(load) - 1:
        tangent_stiffness_before = (load[buckling_idx] - load[buckling_idx-1]) / (disp[buckling_idx] - disp[buckling_idx-1])
        tangent_stiffness_after = (load[buckling_idx+1] - load[buckling_idx]) / (disp[buckling_idx+1] - disp[buckling_idx])

    # 寻找极小值点（在屈曲点之后）
    # 只在屈曲点之后的数据中寻找
    post_buckling_load = load[buckling_idx:]
    post_buckling_disp = disp[buckling_idx:]

    has_minimum = False
    min_load_absolute = None
    min_disp = None

    if len(post_buckling_load) > 30:
        # 找极小值点：局部极小值，且明显低于屈曲荷载
        for i in range(20, len(post_buckling_load) - 5):
            # 局部极小值：比前后都小
            if post_buckling_load[i] < post_buckling_load[i-1] and \
               post_buckling_load[i] < post_buckling_load[i+1]:
                # 检查后续是否有上升趋势（反拱特征）
                if i + 10 < len(post_buckling_load):
                    subsequent_trend = np.mean(post_buckling_load[i+5:i+10])
                    if subsequent_trend > post_buckling_load[i]:
                        # 极小值应明显低于屈曲荷载（至少低10%）
                        if (buckling_load - post_buckling_load[i]) > buckling_load * 0.10:
                            has_minimum = True
                            min_idx_absolute = buckling_idx + i
                            min_load_absolute = load[min_idx_absolute]
                            min_disp = disp[min_idx_absolute]
                            break

    # 分类
    if has_minimum:
        buckling_type = 1  # 完整反拱型
    else:
        buckling_type = 2  # 无极小值型

    return {
        'type': buckling_type,
        'buckling_load': buckling_load,
        'buckling_disp': buckling_disp,
        'min_load': min_load_absolute,
        'min_disp': min_disp,
        'max_disp': disp.max(),
        'final_load': load[-1],
        'final_disp': disp[-1],
        'global_max_load': load.max(),  # 全局最大荷载（用于对比）
        'global_max_idx': int(np.argmax(load))  # 全局最大值索引
    }


def analyze_all_samples():
    """分析所有样本"""
    results = []

    # 辅助函数：将numpy类型转换为Python原生类型
    def to_python_type(val):
        if val is None or pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        return val

    # 获取所有样本文件夹
    sample_folders = sorted(DATA_DIR.glob("FGM_load_disp_*"))

    print(f"找到 {len(sample_folders)} 个样本文件夹")

    for folder in sample_folders:
        sample_id = folder.name.split('_')[-1]

        data = read_sample_data(folder)
        if data is None:
            print(f"样本 {sample_id} 数据读取失败，跳过")
            continue

        # 分类屈曲类型
        classification = classify_buckling_type(data['load'], data['uy'])

        # 计算无量纲参数
        params = data['params']
        if params['L'] is not None and params['L'] > 0:
            rise_span_ratio = params['f_arch'] / params['L'] if params['f_arch'] else None
        else:
            rise_span_ratio = None

        result = {
            'sample_id': int(sample_id),
            'buckling_type': classification['type'],
            'buckling_load': to_python_type(classification['buckling_load']),
            'buckling_disp': to_python_type(classification['buckling_disp']),
            'min_load': to_python_type(classification['min_load']),
            'min_disp': to_python_type(classification['min_disp']),
            'max_disp': to_python_type(classification['max_disp']),
            'final_load': to_python_type(classification['final_load']),
            'final_disp': to_python_type(classification['final_disp']),
            'global_max_load': to_python_type(classification['global_max_load']),
            'global_max_idx': classification['global_max_idx'],
            'I0': to_python_type(params['I0']),
            'A11': to_python_type(params['A11']),
            'B11': to_python_type(params['B11']),
            'D11_eq': to_python_type(params['D11_eq']),
            'L': to_python_type(params['L']),
            'S': to_python_type(params['S']),
            'b': to_python_type(params['b']),
            'h': to_python_type(params['h']),
            'f_arch': to_python_type(params['f_arch']),
            'lambda_real': to_python_type(params['lambda_real']),
            'rise_span_ratio': to_python_type(rise_span_ratio)
        }
        results.append(result)

        if int(sample_id) % 10 == 0:
            print(f"已处理样本 {sample_id}")

    return results


def generate_summary(results):
    """生成统计摘要"""
    df = pd.DataFrame(results)

    # 类型统计
    type1_count = int((df['buckling_type'] == 1).sum())
    type2_count = int((df['buckling_type'] == 2).sum())
    total = int(len(df))

    # 辅助函数：将numpy类型转换为Python原生类型
    def to_python_type(val):
        if pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        if isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        return val

    summary = {
        'total_samples': total,
        'type1_count': type1_count,
        'type2_count': type2_count,
        'type1_ratio': float(type1_count / total) if total > 0 else 0,
        'type2_ratio': float(type2_count / total) if total > 0 else 0,
        'type1_buckling_load_stats': {
            'mean': to_python_type(df[df['buckling_type'] == 1]['buckling_load'].mean()),
            'std': to_python_type(df[df['buckling_type'] == 1]['buckling_load'].std()),
            'min': to_python_type(df[df['buckling_type'] == 1]['buckling_load'].min()),
            'max': to_python_type(df[df['buckling_type'] == 1]['buckling_load'].max())
        },
        'type2_buckling_load_stats': {
            'mean': to_python_type(df[df['buckling_type'] == 2]['buckling_load'].mean()),
            'std': to_python_type(df[df['buckling_type'] == 2]['buckling_load'].std()),
            'min': to_python_type(df[df['buckling_type'] == 2]['buckling_load'].min()),
            'max': to_python_type(df[df['buckling_type'] == 2]['buckling_load'].max())
        },
        'lambda_real_stats': {
            'type1_mean': to_python_type(df[df['buckling_type'] == 1]['lambda_real'].mean()),
            'type2_mean': to_python_type(df[df['buckling_type'] == 2]['lambda_real'].mean())
        },
        'rise_span_ratio_stats': {
            'type1_mean': to_python_type(df[df['buckling_type'] == 1]['rise_span_ratio'].mean()),
            'type2_mean': to_python_type(df[df['buckling_type'] == 2]['rise_span_ratio'].mean())
        }
    }

    return df, summary


def main():
    """主函数"""
    print("="*60)
    print("拱结构屈曲结果分析")
    print("="*60)

    # 分析所有样本
    results = analyze_all_samples()

    print(f"\n共分析 {len(results)} 个样本")

    # 生成摘要
    df, summary = generate_summary(results)

    # 输出结果
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    print(f"总样本数: {summary['total_samples']}")
    print(f"类型1（完整反拱型）样本数: {summary['type1_count']} ({summary['type1_ratio']*100:.1f}%)")
    print(f"类型2（无极小值型）样本数: {summary['type2_count']} ({summary['type2_ratio']*100:.1f}%)")
    print(f"\n类型1屈曲荷载统计:")
    print(f"  平均值: {summary['type1_buckling_load_stats']['mean']/1e6:.2f} MPa")
    print(f"  标准差: {summary['type1_buckling_load_stats']['std']/1e6:.2f} MPa")
    print(f"  最小值: {summary['type1_buckling_load_stats']['min']/1e6:.2f} MPa")
    print(f"  最大值: {summary['type1_buckling_load_stats']['max']/1e6:.2f} MPa")
    print(f"\n类型2屈曲荷载统计:")
    print(f"  平均值: {summary['type2_buckling_load_stats']['mean']/1e6:.2f} MPa")
    print(f"  标准差: {summary['type2_buckling_load_stats']['std']/1e6:.2f} MPa")
    print(f"  最小值: {summary['type2_buckling_load_stats']['min']/1e6:.2f} MPa")
    print(f"  最大值: {summary['type2_buckling_load_stats']['max']/1e6:.2f} MPa")

    # 保存详细结果
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(OUTPUT_DIR / "buckling_analysis_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df.to_csv(OUTPUT_DIR / "buckling_analysis_summary.csv", index=False, encoding='utf-8')

    with open(OUTPUT_DIR / "buckling_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 {OUTPUT_DIR}/ 目录")
    print("="*60)


if __name__ == "__main__":
    main()