"""
使用训练好的模型验证 Hu (2018) Fig. 6(c)(d) 工况
Verification of Hu (2018) Fig. 6(c)(d) cases using trained model
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# 添加Code路径
sys.path.insert(0, r'D:\tanghao\deeparch_review\Code')

from models.transformer_model import BucklingTransformer
from data.dataset import load_and_normalize_data
from config.model_config import ModelConfig

def prepare_input_for_model(case_info, model_type='Full_2'):
    """
    根据Hu论文的几何参数准备模型输入

    Parameters:
    -----------
    case_info : dict
        包含 f, L, boundary 等信息
    model_type : str
        模型版本 ('Full_1', 'Full_2', 'v2F' 等)
    """

    # 几何参数
    L = case_info['L']  # 跨度 (m)
    f = case_info['f']  # 矢高 (m)
    f_L = case_info['f_L']  # 矢跨比

    # 截面参数 (Hu 2018)
    D = 0.045  # 深度 45mm
    B = 0.400  # 宽度 400mm
    E = 210e9  # 弹性模量 210 GPa

    # 截面特性
    A = B * D
    I = B * D**3 / 12

    # 边界条件
    boundary = case_info['boundary']  # 'pin-ended' or 'fixed'

    # 加载类型: 均布竖向荷载
    load_type = 'uniform_vertical'

    print(f"\n准备输入 - f/L = {f_L:.4f}, L = {L:.4f}m, {boundary}")
    print(f"  截面: B={B*1000}mm x D={D*1000}mm")
    print(f"  A = {A*1e6:.2f} mm^2, I = {I*1e9:.6f} mm^4")

    # 这里需要根据您的模型实际输入格式进行调整
    # 示例：假设模型需要归一化的几何参数
    input_dict = {
        'span': L,
        'rise': f,
        'rise_span_ratio': f_L,
        'section_area': A,
        'moment_inertia': I,
        'youngs_modulus': E,
        'boundary': boundary,
        'load_type': load_type
    }

    return input_dict


def load_trained_model(model_path, device='cuda'):
    """
    加载训练好的模型
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 - {model_path}")
        return None

    print(f"\n加载模型: {model_path}")

    # 加载模型配置
    config = ModelConfig()

    # 创建模型
    model = BucklingTransformer(config)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型加载成功! Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, checkpoint


def predict_buckling_load(model, input_data, scalers):
    """
    使用模型预测屈曲载荷
    """
    # 这里需要根据您的模型实际预测流程进行实现
    # 示例代码框架：

    with torch.no_grad():
        # 1. 准备输入张量
        # input_tensor = prepare_model_input(input_data, scalers)

        # 2. 模型预测
        # prediction = model(input_tensor)

        # 3. 反归一化
        # buckling_load = denormalize_prediction(prediction, scalers)

        pass

    # 临时返回
    return {
        'buckling_load': None,
        'load_displacement_curve': None
    }


def main():
    print("=" * 70)
    print("验证 Hu (2018) Fig. 6(c)(d) - 反对称分岔屈曲载荷")
    print("=" * 70)

    # 读取工况
    cases_pin = pd.read_csv('hu2018_fig6_cases_pin.csv')
    cases_fixed = pd.read_csv('hu2018_fig6_cases_fixed.csv')

    print(f"\nPin-ended工况数量: {len(cases_pin)}")
    print(f"Fixed工况数量: {len(cases_fixed)}")

    # 选择要使用的模型
    model_name = 'Full_2'  # 或 'Full_1', 'v2F' 等
    model_path = f'./models/{model_name}/buckling_predictor_best.pth'

    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"\n警告: 模型文件不存在 - {model_path}")
        print("请检查模型路径或先训练模型")
        print("\n可用的模型:")
        model_dirs = [d for d in os.listdir('./models') if os.path.isdir(f'./models/{d}')]
        for d in model_dirs:
            model_file = f'./models/{d}/buckling_predictor_best.pth'
            if os.path.exists(model_file):
                print(f"  - {d}")
        return

    # 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # TODO: 加载模型和数据标准化器
    # model, checkpoint = load_trained_model(model_path, device)

    # 选择要验证的工况 (例如: f/L = 1/12.5, 1/9, 1/8, 1/7)
    test_ratios = [1/12.5, 1/9, 1/8, 1/7, 1/6, 1/5]

    print("\n" + "=" * 70)
    print("开始预测 - Pin-ended arches")
    print("=" * 70)

    results_pin = []

    for idx, row in cases_pin.iterrows():
        if row['f_L'] in test_ratios:
            case_info = {
                'L': row['L'],
                'f': row['f'],
                'f_L': row['f_L'],
                'lambda': row['lambda'],
                'boundary': row['boundary']
            }

            # 准备输入
            input_data = prepare_input_for_model(case_info, model_name)

            # TODO: 使用模型预测
            # prediction = predict_buckling_load(model, input_data, scalers)

            # 保存结果
            results_pin.append({
                'f_L': row['f_L'],
                'L': row['L'],
                'f': row['f'],
                'predicted_load': None,  # prediction['buckling_load']
                'boundary': 'pin-ended'
            })

    print("\n" + "=" * 70)
    print("开始预测 - Fixed arches")
    print("=" * 70)

    results_fixed = []

    for idx, row in cases_fixed.iterrows():
        if row['f_L'] in test_ratios:
            case_info = {
                'L': row['L'],
                'f': row['f'],
                'f_L': row['f_L'],
                'lambda': row['lambda'],
                'boundary': row['boundary']
            }

            # 准备输入
            input_data = prepare_input_for_model(case_info, model_name)

            # TODO: 使用模型预测
            # prediction = predict_buckling_load(model, input_data, scalers)

            # 保存结果
            results_fixed.append({
                'f_L': row['f_L'],
                'L': row['L'],
                'f': row['f'],
                'predicted_load': None,  # prediction['buckling_load']
                'boundary': 'fixed'
            })

    print("\n" + "=" * 70)
    print("预测完成")
    print("=" * 70)

    # 保存结果
    df_results_pin = pd.DataFrame(results_pin)
    df_results_fixed = pd.DataFrame(results_fixed)

    df_results_pin.to_csv('hu2018_predictions_pin.csv', index=False)
    df_results_fixed.to_csv('hu2018_predictions_fixed.csv', index=False)

    print("\n结果已保存:")
    print("  - hu2018_predictions_pin.csv")
    print("  - hu2018_predictions_fixed.csv")

    print("\n" + "=" * 70)
    print("下一步:")
    print("1. 完善 predict_buckling_load() 函数以使用实际模型")
    print("2. 从 Hu (2018) Fig. 6(c)(d) 提取数值数据")
    print("3. 创建对比图表")
    print("=" * 70)


if __name__ == '__main__':
    main()
