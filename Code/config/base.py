# -*- coding: utf-8 -*-
"""
Base Configuration / 基础配置
=============================
Base configuration class with YAML loading support.
基础配置类，支持YAML加载。
"""

from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class BaseConfig:
    """
    Base configuration class with common functionality.
    具有通用功能的基础配置类。

    Supports:
        - YAML file loading
        - Directory auto-creation
        - Post-initialization validation
    """

    # Device configuration / 设备配置
    device: str = "cuda"  # Will be overridden by actual availability

    # Random seed / 随机种子
    seed: int = 618

    use_lstm_prior: bool = False


    def __post_init__(self) -> None:
        """Post-initialization hook / 后初始化钩子"""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            object.__setattr__(self, 'device', 'cpu')

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "BaseConfig":
        """
        Load configuration from YAML file.
        从YAML文件加载配置。

        Args:
            yaml_path: Path to YAML configuration file / YAML配置文件路径

        Returns:
            Configuration instance / 配置实例
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        # Flatten nested config / 展平嵌套配置
        flat_dict = cls._flatten_config(config_dict)

        return cls(**flat_dict)

    @staticmethod
    def _flatten_config(config_dict: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Flatten nested configuration dictionary.
        展平嵌套的配置字典。

        Args:
            config_dict: Nested configuration dictionary / 嵌套配置字典
            parent_key: Parent key for recursion / 递归用的父键

        Returns:
            Flattened dictionary / 展平后的字典
        """
        items = []
        for k, v in config_dict.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(BaseConfig._flatten_config(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        将配置转换为字典。

        Returns:
            Dictionary representation / 字典表示
        """
        return asdict(self)

    def update(self, **kwargs) -> None:
        """
        Update configuration with keyword arguments.
        使用关键字参数更新配置。

        Args:
            **kwargs: Key-value pairs to update / 要更新的键值对
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                object.__setattr__(self, key, value)

    def ensure_dirs(self, *paths: str) -> None:
        """
        Ensure directories exist for given paths.
        确保给定路径的目录存在。

        Args:
            *paths: Directory paths to create / 要创建的目录路径
        """
        for path in paths:
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)