# -*- coding: utf-8 -*-
"""
Inference Entry Point / 推理入口点
====================================
Main script for running inference with trained models.
使用训练好的模型运行推理的主脚本。

Usage:
    python Code/inference.py --folder ./Data/Test
    python Code/inference.py --sample ./Data/Test/sample.csv
    python Code/inference.py --params "param1,param2,..."
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

warnings.filterwarnings('ignore')

# Import from new modules / 从新模块导入
from config.inference import InferenceConfig
from inference.predictor import BucklingPredictorInference


# =============================================================================
# Parameter Names / 参数名称
# =============================================================================
PARAM_NAMES = [
    "I0",      # 截面质量 / Section quality
    "A11",     # 压缩刚度 / Compression stiffness
    "B11",     # 压-弯耦合刚度 / Compression-bending coupling stiffness
    "D11",     # 弯曲刚度 / Bending stiffness
    "L",       # 跨径 / Span
    "f",       # 拱高 / Arch height
    "b",       # 拱截面宽 / Arch section width
    "h",       # 拱截面高 / Arch section height
    "S",       # 拱长 / Arch length
    "lambda",  # 长细比 / Slenderness ratio
    "KXL",     # 左X弹性支撑系数 / Left X elastic support coefficient
    "KYL",     # 左Y弹性支撑系数 / Left Y elastic support coefficient
    "KZL",     # 左转动弹性支撑系数 / Left rotation elastic support coefficient
    "KXR",     # 右X弹性支撑系数 / Right X elastic support coefficient
    "KYR",     # 右Y弹性支撑系数 / Right Y elastic support coefficient
    "KZR",     # 右转动弹性支撑系数 / Right rotation elastic support coefficient
]


def parse_fixed_parameters(param_str: str) -> np.ndarray:
    """
    Parse fixed parameters from string.
    从字符串解析固定参数。

    Args:
        param_str: Comma-separated parameter string / 逗号分隔的参数字符串

    Returns:
        Numpy array of fixed parameters / 固定参数的numpy数组
    """
    values = [float(x.strip()) for x in param_str.split(',')]
    if len(values) != 16:
        raise ValueError(
            f"Expected 16 parameters, got {len(values)} / "
            f"期望16个参数，得到{len(values)}"
        )
    return np.array(values, dtype=np.float32)


def run_batch_inference(
    predictor: BucklingPredictorInference,
    config: InferenceConfig,
    input_dir: str
) -> List[Dict[str, Any]]:
    """
    Run batch inference on folder.
    对文件夹运行批量推理。

    Args:
        predictor: Inference predictor / 推理预测器
        config: Inference configuration / 推理配置
        input_dir: Input folder path / 输入文件夹路径

    Returns:
        List of inference results / 推理结果列表
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Run prediction on folder / 对文件夹运行预测
    predictor.predict_on_folder(
        test_dir=input_dir,
        out_dir=config.output_dir,
        plot=True,
        plot_node=config.buckling_node,
        plot_axis=config.buckling_axis
    )

    return []


def main() -> None:
    """Main function / 主函数"""
    parser = argparse.ArgumentParser(
        description="Inference script for Buckling Predictor / 屈曲预测器推理脚本"
    )

    # Input options / 输入选项
    parser.add_argument(
        "--folder", type=str, default=r"./Data/Test",
        help="Path to folder containing CSV files for batch inference / 包含CSV文件的文件夹路径"
    )
    parser.add_argument(
        "--sample", type=str, default=None,
        help="Path to single sample CSV file / 单个样本CSV文件路径"
    )
    parser.add_argument(
        "--params", type=str, default=None,
        help="Comma-separated fixed parameters (16 values) / 逗号分隔的固定参数（16个值）"
    )

    # Optional arguments / 可选参数
    parser.add_argument(
        "--model", type=str, default=r"./models/FA-LSTM/seed_618/buckling_predictor_best.pth",
        help="Path to trained model / 训练好的模型路径"
    )
    parser.add_argument(
        "--output", type=str, default=r"./results/FA-LSTM/inference",
        help="Output directory / 输出目录"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation / 计算设备"
    )

    args = parser.parse_args()

    # Setup logging / 设置日志
    log_dir = Path(args.output)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inference_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info("Buckling Predictor Inference / 屈曲预测器推理")
    logging.info("=" * 60)

    # Initialize config / 初始化配置
    config = InferenceConfig(
        model_path=args.model,
        output_dir=args.output,
        device=args.device
    )

    # Load predictor / 加载预测器
    predictor = BucklingPredictorInference(config.model_path, device=config.device)

    # Run inference / 运行推理
    if args.sample:
        logging.info(f"Running inference on single sample: {args.sample}")
        result = predictor.predict_from_csv(args.sample)
        print(f"\nBuckling Load: {result['loads_pred'][-1]:.6e}")

    elif args.params:
        logging.info("Running inference with custom parameters")
        # Custom parameter inference would require initial state
        logging.warning("Custom parameter inference requires initial state, not implemented in this version")

    else:
        # Default: batch inference on folder / 默认：对文件夹进行批量推理
        logging.info(f"Running batch inference on folder: {args.folder}")
        run_batch_inference(predictor, config, args.folder)

    logging.info("=" * 60)
    logging.info("Inference completed / 推理完成")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()