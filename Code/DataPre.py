"""
Data Preprocessing Pipeline / 数据预处理流水线
===============================================

Preprocesses raw ANSYS MAPDL simulation output into training-ready CSV files.
Each sample directory (containing ``result/input.txt`` and per-node
``load-disp_*.csv`` files) is processed through:

1. Parameter loading  (input.txt  ->  9 structural parameters)
2. Displacement loading  (load-disp_*.csv  ->  per-node UX/UY histories)
3. Dimensionless normalization  (load / displacement scaled by D11, L, b)
4. Initial-state padding  (prepend t=0 with zero load/displacement)
5. Interpolation to fixed 200-step sequence
6. Post-buckling classification  ("有后屈曲" / "无后屈曲")
7. Output CSV with columns:  time_step | 9 params | load | 21 nodes x 2 disp

Uses multiprocessing for speed.  Output lands in ``./Data/processed/``.
"""

import math
import os
import hashlib
import re
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Tuple, List, Optional
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp


@dataclass
class ProcessingConfig:
    """
    数据预处理配置
    Processing configuration container.

    v3.0更新：
    - 参数格式更新为9个：I0, A11, B11, D11_eq, f_L, lambda_real, b_h, etaRotL, etaRotR
    - 固定跨径L=1，不再作为参数
    - X/Y铰接约束，仅保留转动刚度系数
    """
    NUM_NODES: int = 21
    NUM_PARAMS: int = 9                # 写入 CSV 的参数个数 / feature parameter count
    RAW_PARAM_COUNT: int = 9           # 输入 input.txt 的原始参数个数
    DISP_COMPONENTS: int = 2           # 节点自由度数量：仅 UX、UY / only UX, UY
    TARGET_LENGTH: int = 200
    MIN_TIME_STEPS: int = 200
    ADD_INITIAL_STATE: bool = True     # 是否添加荷载=0、位移=0 的初始状态
    PARAM_NAMES: List[str] = None

    def __post_init__(self):
        if self.PARAM_NAMES is None:
            self.PARAM_NAMES = [
                "截面质量I0",
                "压缩刚度A11",
                "压-弯耦合刚度B11",
                "弯曲刚度D11",
                "矢跨比f_L",
                "长细比λ",
                "宽高比b_h",
                "左转动刚度系数etaRotL",
                "右转动刚度系数etaRotR",
            ]


def generate_md5_filename(sample_dir: str) -> str:
    """根据样本目录生成稳定的 MD5 文件名 / Generate a deterministic name via MD5."""
    hash_obj = hashlib.md5(sample_dir.encode('utf-8'))
    return f"{hash_obj.hexdigest()}_FGM_{sample_dir}.csv"


def compute_feature_parameters(raw_params: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """
    由 9 个原始参数计算特征参数。

    v3.0更新：
    原始参数顺序（input.txt）：
    I0, A11, B11, D11_eq, f_L, lambda_real, b_h, etaRotL, etaRotR

    这些参数直接作为特征参数使用（无需额外计算）
    """
    if raw_params.size != config.RAW_PARAM_COUNT:
        raise ValueError(
            f"参数数量不正确，期望 {config.RAW_PARAM_COUNT} 个，实际 {raw_params.size} 个"
        )

    # 新格式参数直接使用，无需计算
    (I0, A11, B11, D11_eq, f_L, lambda_val, b_h,
     etaRotL, etaRotR) = raw_params

    feature_params = np.array(
        [
            I0, A11, B11, D11_eq,
            f_L, lambda_val, b_h,
            etaRotL, etaRotR,
        ],
        dtype=np.float64,
    )
    return feature_params


def add_initial_state(params: np.ndarray, load_dimless: np.ndarray,
                     disp_dimless: List[np.ndarray], config: ProcessingConfig
                     ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    在序列最前方添加荷载与位移的初始零状态。
    Prepend zero-load/zero-displacement state if requested.
    """
    if not config.ADD_INITIAL_STATE:
        return load_dimless, disp_dimless

    initial_load = np.array([0.0], dtype=load_dimless.dtype)
    initial_disp_list = [
        np.zeros((1, config.DISP_COMPONENTS), dtype=node_disp.dtype)
        for node_disp in disp_dimless
    ]

    load_with_initial = np.concatenate([initial_load, load_dimless])
    disp_with_initial = [
        np.concatenate([initial_disp_list[i], node_disp], axis=0)
        for i, node_disp in enumerate(disp_dimless)
    ]

    logging.debug("添加初始状态：原长度 %d → 新长度 %d",
                  len(load_dimless), len(load_with_initial))
    return load_with_initial, disp_with_initial


def uniform_sample_or_interpolate(sequence: np.ndarray,
                                  target_length: int = 200) -> np.ndarray:
    """
    使用线性插值将序列统一为固定长度。
    Resample sequences to a unified length via linear interpolation.
    """
    original_length = len(sequence)
    if original_length == target_length:
        return sequence.copy()

    x_old = np.arange(original_length, dtype=np.float64)
    x_new = np.linspace(0, original_length - 1, target_length)

    result = np.zeros((target_length, sequence.shape[1]), dtype=sequence.dtype)
    for col in range(sequence.shape[1]):
        result[:, col] = np.interp(x_new, x_old, sequence[:, col])

    return result


class ProcessingResult:
    """统计成功/失败样本数量。Track preprocessing statistics."""
    def __init__(self):
        self.success = 0
        self.fail = 0
        self.errors = []

    def add_success(self):
        self.success += 1

    def add_failure(self, error_msg: str):
        self.fail += 1
        self.errors.append(error_msg)


def load_design_parameters(input_path: Path, config: ProcessingConfig
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取 input.txt，返回原始参数与特征参数。
    Load raw parameters (16) and compute 13 feature parameters.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"参数文件不存在: {input_path}")

    raw_params = np.loadtxt(input_path, max_rows=config.RAW_PARAM_COUNT)
    if raw_params.size != config.RAW_PARAM_COUNT:
        raise ValueError(
            f"参数数量不正确，期望 {config.RAW_PARAM_COUNT} 个，实际 {raw_params.size} 个"
        )

    feature_params = compute_feature_parameters(raw_params, config)
    return raw_params, feature_params


def load_displacement_data(result_dir: Path, config: ProcessingConfig
                           ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    读取节点荷载-位移数据，并统一去除最后一行。
    Load load-displacement histories per node, discarding the last row to avoid divergence.
    """
    disp_data = []
    load_sequence = None
    time_steps = None

    for node_idx in range(1, config.NUM_NODES + 1):
        csv_path = result_dir / f"load-disp_{node_idx}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"节点 {node_idx} 的位移文件不存在: {csv_path}")

        df = pd.read_csv(csv_path, usecols=range(4))
        if df.shape[0] < 2:
            raise ValueError(f"节点 {node_idx} 的数据不足以截断尾部")
        if df.shape[1] < 4:
            raise ValueError(f"节点{node_idx}的CSV列数不足")

        data = df.values
        current_load = data[:-1, 0]          # 去掉最后一行 / drop last row
        current_disp = data[:-1, 1:3]        # 仅保留 UX、UY / only UX, UY

        if time_steps is None:
            time_steps = len(current_load)
            load_sequence = current_load
        else:
            if len(current_load) != time_steps:
                raise ValueError(
                    f"节点{node_idx}的时间步数不一致，期望{time_steps}，实际{len(current_load)}"
                )
            if not np.allclose(current_load, load_sequence, atol=1e-6):
                logging.warning("节点 %d 的荷载序列存在微小差异", node_idx)

        disp_data.append(current_disp)

    return load_sequence, disp_data


def load_section_width(result_dir: Path) -> float:
    """
    从 run.txt 中提取 b_section（截面宽度）。
    """
    run_path = result_dir / "run.txt"
    if not run_path.exists():
        raise FileNotFoundError(f"缺少 run.txt：{run_path}")
    pattern = re.compile(r"b_section\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    with run_path.open('r', encoding='GBK') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"run.txt 中未找到 b_section 参数：{run_path}")


def load_section_height(result_dir: Path) -> float:
    """
    从 run.txt 中提取 h_section（截面goadu）。
    """
    run_path = result_dir / "run.txt"
    if not run_path.exists():
        raise FileNotFoundError(f"缺少 run.txt：{run_path}")
    pattern = re.compile(r"h_section\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    with run_path.open('r', encoding='GBK') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return float(match.group(1))
    raise ValueError(f"run.txt 中未找到 h_section 参数：{run_path}")


def dimensionalize_data(load_sequence: np.ndarray,
                        disp_data: List[np.ndarray],
                        raw_params: np.ndarray,
                        section_width: float,
                        config: ProcessingConfig
                        ) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    无量纲化荷载与位移数据。

    v3.0更新：
    - L固定为1，不再从参数读取
    - raw_params格式：I0, A11, B11, D11_eq, f_L, lambda_real, b_h, etaRotL, etaRotR
    - D11 = D11_eq + B11^2/A11（还原原始弯曲刚度）
    """
    # 新格式参数索引
    A11 = raw_params[1]
    B11 = raw_params[2]
    D11_eq = raw_params[3]
    L = 1.0  # 固定跨径
    b = section_width

    # 计算原始弯曲刚度（用于荷载无量纲化）
    D11 = D11_eq + (B11 ** 2) / A11

    if any(val <= 0 for val in [D11, L, b]):
        raise ValueError(f"无效的参数值：D11={D11}, L={L}, b={b}")

    load_dimless = (load_sequence * b * (L ** 2)) / D11

    # 位移无量纲化（L=1时，位移直接使用，无需缩放）
    disp_dimless = []
    for node_disp in disp_data:
        normalized = node_disp.astype(np.float64)
        normalized[:, 0] /= L  # UX/L，L=1时为UX
        normalized[:, 1] /= L  # UY/L，L=1时为UY
        disp_dimless.append(normalized)

    return load_dimless, disp_dimless


def build_feature_sequence(feature_params: np.ndarray,
                           load_dimless: np.ndarray,
                           disp_dimless: List[np.ndarray],
                           config: ProcessingConfig) -> np.ndarray:
    """
    拼接参数、荷载及位移数据为统一特征序列。
    Build the [time_step × feature] matrix.
    """
    time_steps = len(load_dimless)
    total_features = config.NUM_PARAMS + 1 + config.NUM_NODES * config.DISP_COMPONENTS
    sequence = np.zeros((time_steps, total_features), dtype=np.float64)

    sequence[:, :config.NUM_PARAMS] = feature_params
    sequence[:, config.NUM_PARAMS] = load_dimless

    disp_start = config.NUM_PARAMS + 1
    for i, node_disp in enumerate(disp_dimless):
        start_col = disp_start + i * config.DISP_COMPONENTS
        end_col = start_col + config.DISP_COMPONENTS
        sequence[:, start_col:end_col] = node_disp

    return sequence


def get_feature_names(config: ProcessingConfig) -> List[str]:
    """
    构造列名（参数 + 荷载 + 节点位移）。
    Build human-readable column names.
    """
    feature_names = config.PARAM_NAMES.copy()
    feature_names.append("荷载")

    for node in range(1, config.NUM_NODES + 1):
        feature_names.extend([
            f"节点_{node}_x位移",
            f"节点_{node}_y位移",
        ])

    return feature_names


def classify_load_history(load_series: np.ndarray,
                          tol: float = 1e-5) -> str:
    """
    根据荷载—位移曲线分类后屈曲类型。
    Classify the curve into:
      - “无后屈曲”: only one local maximum (monotonic up then down)
      - “有后屈曲”: at least one max followed by a min (post-buckling strengthening)
    """
    if load_series.size < 3:
        return "无后屈曲"

    diffs = np.diff(load_series)
    diffs[np.abs(diffs) < tol] = 0.0

    extrema = []
    prev_sign = 0
    for idx, delta in enumerate(diffs, start=1):
        if delta > 0:
            current_sign = 1
        elif delta < 0:
            current_sign = -1
        else:
            current_sign = 0

        if prev_sign > 0 and current_sign < 0:
            extrema.append(("max", idx))
        elif prev_sign < 0 and current_sign > 0:
            extrema.append(("min", idx))

        if current_sign != 0:
            prev_sign = current_sign

    first_max_index = next((idx for kind, idx in extrema if kind == "max"), None)
    min_after_max = None
    if first_max_index is not None:
        min_after_max = next(
            (idx for kind, idx in extrema
             if kind == "min" and idx > first_max_index),
            None
        )

    if first_max_index is not None and min_after_max is not None:
        return "有后屈曲"
    return "无后屈曲"


def process_single_sample(args: Tuple[str, str, str, str]) -> Tuple[bool, str]:
    """
    单样本处理函数（供多进程调用）。
    Per-sample preprocessing entry point.
    """
    sample_dir, root_dir, no_post_dir, post_dir = args
    config = ProcessingConfig()

    try:
        sample_path = Path(root_dir) / sample_dir
        result_dir = sample_path / "result"
        if not result_dir.exists():
            raise FileNotFoundError(f"结果目录不存在: {result_dir}")

        input_path = result_dir / "input.txt"
        raw_params, feature_params = load_design_parameters(input_path, config)
        section_width = load_section_width(result_dir)  # 仅用于无量纲化
        load_sequence, disp_data = load_displacement_data(result_dir, config)
        time_steps = len(load_sequence)
        if time_steps < config.MIN_TIME_STEPS:
            raise ValueError(f"时间步数仅 {time_steps}，不足以插值")

        load_dimless, disp_dimless = dimensionalize_data(load_sequence, disp_data,
                                                         raw_params, section_width, config)
        load_with_initial, disp_with_initial = add_initial_state(
            feature_params, load_dimless, disp_dimless, config
        )

        sequence = build_feature_sequence(feature_params,
                                          load_with_initial,
                                          disp_with_initial,
                                          config)
        sequence_uniform = uniform_sample_or_interpolate(sequence,
                                                         config.TARGET_LENGTH)

        load_column = sequence_uniform[:, config.NUM_PARAMS]
        category = classify_load_history(load_column)

        feature_names = get_feature_names(config)
        df_sequence = pd.DataFrame(sequence_uniform, columns=feature_names)
        df_sequence.insert(0, "时间步", range(1, config.TARGET_LENGTH + 1))

        output_filename = generate_md5_filename(sample_dir)
        category_dir = Path(post_dir if category == "有后屈曲" else no_post_dir)
        category_dir.mkdir(parents=True, exist_ok=True)
        output_path = category_dir / output_filename
        df_sequence.to_csv(output_path, index=False, encoding='utf-8-sig')

        return True, (
            f"样本 {sample_dir} 处理成功（原始: {time_steps} → "
            f"含初始: {len(load_with_initial)} → 统一: {config.TARGET_LENGTH}），"
            f"分类: {category}"
        )

    except Exception as exc:
        return False, f"样本 {sample_dir} 处理失败: {exc}"


def preprocess_data(root_dir: str, max_workers: Optional[int] = None) -> ProcessingResult:
    """
    主预处理入口：多进程遍历样本目录。
    Main entry for preprocessing with multiprocessing support.
    """
    root_path = Path(root_dir)
    output_base = Path(r"./Data/processed")
    output_base.mkdir(exist_ok=True)

    no_post_dir = output_base / "无后屈曲"
    post_dir = output_base / "有后屈曲"
    no_post_dir.mkdir(exist_ok=True)
    post_dir.mkdir(exist_ok=True)

    log_file = output_base / "preprocessing.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )

    logging.info("预处理输出目录：%s (分类: %s / %s)",
                 output_base, no_post_dir.name, post_dir.name)

    exclude_dirs = {"preprocessed_data_csv", "preprocessed_data_200"}
    sample_dirs = [
        d.name for d in root_path.iterdir()
        if d.is_dir() and d.name not in exclude_dirs
    ]

    if not sample_dirs:
        logging.warning("未找到任何样本文件夹，请检查根目录。")
        return ProcessingResult()

    total_samples = len(sample_dirs)
    logging.info("发现 %d 个样本，开始多进程预处理...", total_samples)

    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, total_samples, 8)
        max_workers = max(1, max_workers)
    logging.info("使用 %d 个进程", max_workers)

    result = ProcessingResult()
    args_list = [
        (sample_dir, str(root_path), str(no_post_dir), str(post_dir))
        for sample_dir in sample_dirs
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_sample, args): args[0]
            for args in args_list
        }

        for future in tqdm(as_completed(futures),
                           total=total_samples,
                           desc="处理样本进度",
                           unit="样本"):
            sample_dir = futures[future]
            try:
                success, message = future.result()
                if success:
                    result.add_success()
                    logging.info(message)
                else:
                    result.add_failure(message)
                    logging.error(message)
            except Exception as exc:
                error_msg = f"样本 {sample_dir} 处理异常: {exc}"
                result.add_failure(error_msg)
                logging.error(error_msg)

    logging.info("预处理完成！成功: %d, 失败: %d", result.success, result.fail)
    logging.info("输出根目录: %s", output_base)

    if result.errors:
        logging.info("失败样本（最多列出10个）：")
        for error in result.errors[:10]:
            logging.info("  - %s", error)
        if len(result.errors) > 10:
            logging.info("  ... 其余 %d 个错误详见日志", len(result.errors) - 10)

    return result


if __name__ == "__main__":
    data_root = r"./Data/raw"
    if not Path(data_root).exists():
        print(f"错误：根目录不存在: {data_root}")
        exit(1)

    print("开始数据预处理（统一为 200 行，包含初始状态，自动分类）...")
    preprocess_result = preprocess_data(data_root, max_workers=8)
    print(f"预处理完成！成功: {preprocess_result.success} 个, "
          f"失败: {preprocess_result.fail} 个")
    print("详情请查看日志和输出文件夹。")
