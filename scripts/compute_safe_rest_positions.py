#!/usr/bin/env python3
"""
计算训练数据集的 safe_rest_positions（首帧关节位置统计量）

用途:
    推理前机器人需要从 safe position (2048) 移动到训练起始位，
    确保起始状态在训练分布内，避免首帧 action chunk 输出错误方向。
    本脚本从数据集中提取每个 episode 的首帧 observation.state，
    计算均值 / 中位数等统计量，输出可直接粘贴到 YAML 配置的 safe_rest_positions 字段。

用法:
    python scripts/compute_safe_rest_positions.py <dataset_path>

示例:
    python scripts/compute_safe_rest_positions.py datasets/nexarm_smolvla_merged/train
    python scripts/compute_safe_rest_positions.py datasets/nexarm_act/train

输出:
    - 各关节首帧统计表（mean, median, std, min, max, p5, p95）
    - YAML safe_rest_positions（mean 版本，可直接复制）
    - YAML safe_rest_positions（median 版本，可直接复制）
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def load_dataset(dataset_path: str) -> tuple[np.ndarray, list[str]]:
    """加载数据集，返回首帧状态矩阵和关节名称列表。

    优先使用 parquet 直接读取（更快），回退到 LeRobotDataset API。
    """
    dataset_path = Path(dataset_path)

    # --- 尝试 parquet 直接读取 ---
    data_dir = dataset_path / "data"
    if data_dir.exists():
        import pyarrow.parquet as pq

        tables = []
        for chunk_dir in sorted(data_dir.iterdir()):
            if chunk_dir.is_dir():
                for pq_file in sorted(chunk_dir.glob("*.parquet")):
                    tables.append(pq.read_table(pq_file))

        if not tables:
            print(f"错误: 在 {data_dir} 下未找到 parquet 文件", file=sys.stderr)
            sys.exit(1)

        import pyarrow as pa
        combined = pa.concat_tables(tables)
        df = combined.to_pandas()

        # 读取关节名称（从 info.json）
        info_path = dataset_path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            joint_names = info["features"]["observation.state"]["names"]
        else:
            joint_names = JOINT_NAMES

        # 提取首帧
        first_frames = df.groupby("episode_index").first()
        states = np.stack(first_frames["observation.state"].values)

        return states, joint_names

    # --- 回退到 LeRobotDataset API ---
    # dataset_path 可能是 root/repo_id 形式，尝试推断
    repo_id = dataset_path.name
    root = str(dataset_path.parent)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(repo_id=repo_id, root=root)
    except Exception:
        # 也许 dataset_path 就是 root，repo_id 需要从 info.json 推断
        info_path = dataset_path / "meta" / "info.json"
        if info_path.exists():
            # 这种情况下 dataset_path 本身就是数据集目录
            # LeRobotDataset 需要 repo_id 和 root 分开
            parent = dataset_path.parent
            repo_id = dataset_path.name
            # 尝试 parent 作为 root
            try:
                ds = LeRobotDataset(repo_id=f"local/{repo_id}", root=str(parent))
            except Exception:
                print(f"错误: 无法使用 LeRobotDataset 加载 {dataset_path}", file=sys.stderr)
                print("请确保数据集目录结构正确，或安装 lerobot 包。", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"错误: 无法识别数据集格式: {dataset_path}", file=sys.stderr)
            sys.exit(1)

    # 从 LeRobotDataset 提取首帧
    episode_indices = ds.episode_data_index["from"].tolist()
    states_list = []
    for ep_idx, start_idx in enumerate(episode_indices):
        frame = ds[start_idx]
        state = frame["observation.state"].numpy()
        states_list.append(state)

    states = np.stack(states_list)

    # 获取关节名称
    info_path = Path(ds.root) / ds.repo_id / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        joint_names = info["features"]["observation.state"]["names"]
    else:
        joint_names = JOINT_NAMES

    return states, joint_names


def compute_statistics(states: np.ndarray, joint_names: list[str]) -> dict:
    """计算首帧状态的各项统计量。"""
    n_joints = states.shape[1]
    stats = {}
    for i, name in enumerate(joint_names):
        col = states[:, i]
        stats[name] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p5": float(np.percentile(col, 5)),
            "p95": float(np.percentile(col, 95)),
        }
    return stats


def print_results(stats: dict, n_episodes: int, joint_names: list[str]):
    """格式化输出结果。"""

    # === 统计表 ===
    print(f"\n{'='*80}")
    print(f"  Safe Rest Positions — 首帧关节位置统计 (共 {n_episodes} 个 episode)")
    print(f"{'='*80}")
    print()
    header = f"{'Joint':<20s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'P5':>8s} {'P95':>8s}"
    print(header)
    print("-" * len(header))
    for name in joint_names:
        s = stats[name]
        print(
            f"{name:<20s} "
            f"{s['mean']:8.0f} {s['median']:8.0f} {s['std']:8.0f} "
            f"{s['min']:8.0f} {s['max']:8.0f} "
            f"{s['p5']:8.0f} {s['p95']:8.0f}"
        )

    # === YAML 输出 ===
    print(f"\n{'='*80}")
    print("  YAML safe_rest_positions (mean — 推荐)")
    print(f"{'='*80}")
    print()
    print("  safe_rest_positions:")
    for name in joint_names:
        print(f"    {name}: {int(round(stats[name]['mean']))}")

    print(f"\n{'='*80}")
    print("  YAML safe_rest_positions (median)")
    print(f"{'='*80}")
    print()
    print("  safe_rest_positions:")
    for name in joint_names:
        print(f"    {name}: {int(round(stats[name]['median']))}")

    # === 单行复制（mean）===
    mean_vals = [int(round(stats[name]["mean"])) for name in joint_names]
    print(f"\n{'='*80}")
    print("  一行复制 (mean):")
    print(f"{'='*80}")
    print(f"  {mean_vals}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="计算训练数据集首帧关节位置统计量，生成 safe_rest_positions 配置"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集路径 (如 datasets/nexarm_smolvla_merged/train)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="同时输出 JSON 格式的统计结果",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "median"],
        default="mean",
        help="YAML 默认使用的统计量 (默认: mean)",
    )

    args = parser.parse_args()

    print(f"加载数据集: {args.dataset_path}")
    states, joint_names = load_dataset(args.dataset_path)
    n_episodes, n_joints = states.shape
    print(f"  Episode 数量: {n_episodes}")
    print(f"  关节数量: {n_joints}")

    stats = compute_statistics(states, joint_names)
    print_results(stats, n_episodes, joint_names)

    if args.json:
        output = {
            "dataset_path": str(args.dataset_path),
            "n_episodes": n_episodes,
            "n_joints": n_joints,
            "joint_names": joint_names,
            "statistics": stats,
            "safe_rest_positions_mean": {
                name: int(round(stats[name]["mean"])) for name in joint_names
            },
            "safe_rest_positions_median": {
                name: int(round(stats[name]["median"])) for name in joint_names
            },
        }
        json_path = Path(args.dataset_path) / "safe_rest_positions.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"JSON 结果已保存到: {json_path}")


if __name__ == "__main__":
    main()
