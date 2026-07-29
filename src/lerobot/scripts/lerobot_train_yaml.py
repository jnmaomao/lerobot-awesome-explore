#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Train a policy with YAML config support.

This wrapper converts YAML configs to JSON before calling the original train script.

Usage:
    python -m lerobot.scripts.lerobot_train_yaml --config_path=examples/nexarm/train.yaml
    python -m lerobot.scripts.lerobot_train_yaml --config_path=examples/nexarm/train.yaml --batch_size=16

Requires: pip install 'lerobot[training]' PyYAML
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# 添加 PyYAML 依赖检查
try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required for YAML config support. Install with: pip install PyYAML"
    )

from lerobot.scripts.lerobot_train import train
from lerobot.configs import parser


def yaml_to_dict(yaml_path: Path) -> dict:
    """读取 YAML 文件并返回字典"""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, cli_args: list[str]) -> dict:
    """
    合并 CLI 参数覆盖到配置字典

    支持 --key=value 和 --key value 两种格式
    """
    i = 0
    while i < len(cli_args):
        arg = cli_args[i]

        if arg.startswith("--"):
            key = arg[2:]  # 去掉 --

            if "=" in key:
                # --key=value 格式
                key, value = key.split("=", 1)
            else:
                # --key value 格式
                i += 1
                if i >= len(cli_args):
                    raise ValueError(f"参数 {arg} 缺少值")
                value = cli_args[i]

            # 解析值类型
            parsed_value = parse_value(value)

            # 设置嵌套键值
            set_nested(config, key, parsed_value)

        i += 1

    return config


def parse_value(value: str):
    """解析字符串值的类型"""
    # 布尔值
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # None
    if value.lower() in ("null", "none"):
        return None

    # 整数
    try:
        return int(value)
    except ValueError:
        pass

    # 浮点数
    try:
        return float(value)
    except ValueError:
        pass

    # JSON 列表/字典
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # 字符串
    return value


def set_nested(d: dict, key: str, value):
    """
    设置嵌套字典的值

    例如: set_nested(d, "dataset.repo_id", "test")
         等同于 d["dataset"]["repo_id"] = "test"
    """
    keys = key.split(".")
    current = d

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    current[keys[-1]] = value


def main():
    """主入口 - 解析 YAML 并转换为 JSON，然后调用原始 train()"""
    # 自定义参数解析
    arg_parser = argparse.ArgumentParser(
        description="使用 YAML 配置文件训练策略（自动转换为 JSON）",
        add_help=False,  # 禁用自动 help，让原始脚本处理
    )
    arg_parser.add_argument(
        "--config_path",
        type=str,
        help="YAML 或 JSON 配置文件路径"
    )
    arg_parser.add_argument(
        "--help-yaml",
        action="store_true",
        help="显示 YAML 训练脚本帮助信息"
    )
    arg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印转换后的 JSON 配置，不实际运行训练"
    )

    # 解析已知参数
    args, remaining_args = arg_parser.parse_known_args()

    # Dry-run 模式
    if args.dry_run and args.config_path:
        config_path = Path(args.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        if config_path.suffix in ['.yaml', '.yml']:
            config_dict = yaml_to_dict(config_path)
            if remaining_args:
                config_dict = merge_cli_overrides(config_dict, remaining_args)
        else:
            with open(config_path) as f:
                config_dict = json.load(f)

        print("=" * 60)
        print("转换后的 JSON 配置:")
        print("=" * 60)
        print(json.dumps(config_dict, indent=2, ensure_ascii=False))
        return

    if args.help_yaml:
        print(__doc__)
        print("\n示例:")
        print("  # 使用 YAML 配置训练")
        print("  python -m lerobot.scripts.lerobot_train_yaml --config_path=examples/nexarm/train.yaml")
        print()
        print("  # 使用 YAML 配置并覆盖部分参数")
        print("  python -m lerobot.scripts.lerobot_train_yaml --config_path=examples/nexarm/train.yaml --batch_size=16")
        sys.exit(0)

    if not args.config_path:
        # 如果没有提供 config_path，让原始 train() 处理
        logging.warning("未提供 --config_path，将使用默认配置或 CLI 参数")
        train()
        return

    config_path = Path(args.config_path)

    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 如果是 JSON 文件，直接调用原始 train()
    if config_path.suffix not in ['.yaml', '.yml']:
        logging.info(f"使用 JSON 配置: {config_path}")
        sys.argv = ["lerobot_train", "--config_path", str(config_path)] + remaining_args
        train()
        return

    # YAML 文件处理
    logging.info(f"读取 YAML 配置: {config_path}")
    config_dict = yaml_to_dict(config_path)

    # 合并 CLI 覆盖参数
    if remaining_args:
        config_dict = merge_cli_overrides(config_dict, remaining_args)
        logging.info(f"应用 CLI 覆盖参数: {remaining_args}")

    # 创建临时 JSON 文件
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.json',
        delete=False
    ) as temp_file:
        json.dump(config_dict, temp_file, indent=2)
        temp_json_path = Path(temp_file.name)

    try:
        logging.info(f"创建临时 JSON 配置: {temp_json_path}")

        # 修改 sys.argv 让原始 train() 使用临时 JSON 文件
        original_argv = sys.argv
        sys.argv = ["lerobot_train", "--config_path", str(temp_json_path)]

        try:
            train()
        finally:
            sys.argv = original_argv

    finally:
        # 清理临时文件
        temp_json_path.unlink(missing_ok=True)
        logging.info("清理临时文件完成")


if __name__ == "__main__":
    main()
