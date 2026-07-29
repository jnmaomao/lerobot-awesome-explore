#!/usr/bin/env python3
"""
数据集质量评估脚本（基于 LeRobot 内置 API）

用法:
    python scripts/evaluate_dataset.py <repo_id> <root>

示例:
    python scripts/evaluate_dataset.py local/nexarm_merged datasets/nexarm_merged
    python scripts/evaluate_dataset.py local/0715_10 datasets/0715_10
"""

import json
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def evaluate_dataset(repo_id: str, root: str):
    """使用 LeRobot 内置 API 评估数据集质量"""

    # ===== 加载数据集 =====
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    stats_path = Path(root) / ds.repo_id / "meta" / "stats.json" if Path(root).is_dir() else Path(root) / "meta" / "stats.json"
    # 尝试定位 stats.json
    for candidate in [Path(root) / "meta" / "stats.json", Path(root) / ds.repo_id / "meta" / "stats.json"]:
        if candidate.exists():
            stats_path = candidate
            break

    with open(stats_path) as f:
        stats = json.load(f)

    # ===== 基本信息 =====
    total_episodes = ds.num_episodes
    total_frames = len(ds)
    fps = ds.fps
    avg_frames = total_frames / total_episodes
    avg_duration = avg_frames / fps
    total_minutes = total_frames / fps / 60

    # 获取关节名称
    joint_names = ds.meta.features["action"]["names"]

    # ===== Episode 长度分布 =====
    episode_lengths = np.array([ep["length"] for ep in ds.meta.episodes])

    # ===== 输出报告 =====
    print("=" * 80)
    print("数据集质量评估报告")
    print("=" * 80)
    print(f"\n数据集: {repo_id}")
    print(f"路径: {root}")

    # ----- 1. 基本信息 -----
    print(f"\n{'='*60}")
    print("基本信息")
    print(f"{'='*60}")
    print(f"  总 Episode 数:  {total_episodes}")
    print(f"  总帧数:          {total_frames}")
    print(f"  FPS:             {fps}")
    print(f"  平均每Episode:   {avg_frames:.1f} 帧 ({avg_duration:.2f} 秒)")
    print(f"  总时长:          {total_minutes:.2f} 分钟")

    # ----- 2. Episode 长度分布 -----
    print(f"\n{'='*60}")
    print("Episode 长度分布")
    print(f"{'='*60}")
    print(f"  平均: {np.mean(episode_lengths):.1f} 帧 ({np.mean(episode_lengths)/fps:.2f}s)")
    print(f"  最短: {np.min(episode_lengths)} 帧 ({np.min(episode_lengths)/fps:.2f}s)")
    print(f"  最长: {np.max(episode_lengths)} 帧 ({np.max(episode_lengths)/fps:.2f}s)")
    print(f"  中位数: {np.median(episode_lengths):.0f} 帧")
    print(f"  标准差: {np.std(episode_lengths):.1f} 帧")

    # ----- 3. 动作空间统计 -----
    print(f"\n{'='*60}")
    print("动作空间统计")
    print(f"{'='*60}")
    print(f"{'关节':<20} {'最小值':>8} {'最大值':>8} {'均值':>10} {'标准差':>10} {'运动范围':>10}")
    print("-" * 70)
    for i, name in enumerate(joint_names):
        min_v = stats["action"]["min"][i]
        max_v = stats["action"]["max"][i]
        mean_v = stats["action"]["mean"][i]
        std_v = stats["action"]["std"][i]
        range_v = max_v - min_v
        print(f"{name:<20} {min_v:>8.0f} {max_v:>8.0f} {mean_v:>10.1f} {std_v:>10.1f} {range_v:>10.0f}")

    # ----- 4. 状态空间统计 -----
    print(f"\n{'='*60}")
    print("状态空间统计")
    print(f"{'='*60}")
    print(f"{'关节':<20} {'最小值':>8} {'最大值':>8} {'均值':>10} {'标准差':>10}")
    print("-" * 60)
    for i, name in enumerate(joint_names):
        min_v = stats["observation.state"]["min"][i]
        max_v = stats["observation.state"]["max"][i]
        mean_v = stats["observation.state"]["mean"][i]
        std_v = stats["observation.state"]["std"][i]
        print(f"{name:<20} {min_v:>8.0f} {max_v:>8.0f} {mean_v:>10.1f} {std_v:>10.1f}")

    # ----- 5. 状态-动作一致性 -----
    print(f"\n{'='*60}")
    print("状态-动作一致性")
    print(f"{'='*60}")
    for i, name in enumerate(joint_names):
        diff = abs(stats["action"]["mean"][i] - stats["observation.state"]["mean"][i])
        s = "✅" if diff <= 100 else "⚠️"
        print(f"  {s} {name}: 差异 = {diff:.1f}")

    # ----- 6. 数据多样性 (CV) -----
    print(f"\n{'='*60}")
    print("数据多样性 (CV = std/mean)")
    print(f"{'='*60}")
    for i, name in enumerate(joint_names):
        mean = stats["action"]["mean"][i]
        std = stats["action"]["std"][i]
        if mean > 0:
            cv = std / mean
            if cv < 0.05:
                tag = "运动较少"
            elif cv > 0.3:
                tag = "运动剧烈"
            else:
                tag = "适中"
            print(f"  {name}: CV = {cv:.3f} ({tag})")

    # ----- 7. 视频文件检查 -----
    print(f"\n{'='*60}")
    print("视频文件检查")
    print(f"{'='*60}")
    video_path = Path(root) / "videos"
    if video_path.exists():
        for vk in sorted(video_path.iterdir()):
            if vk.is_dir():
                files = list(vk.rglob("*.mp4"))
                size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                print(f"  ✅ {vk.name}: {len(files)} 个视频, {size_mb:.2f} MB")
    else:
        print("  ⚠️ 未找到视频目录")

    # ===== 质量评估 =====
    print(f"\n{'='*80}")
    print("数据质量评估")
    print(f"{'='*80}")

    issues = []

    # Episode 数量
    if total_episodes < 10:
        issues.append(f"Episode 数量不足: {total_episodes} (建议至少 10)")
    elif total_episodes < 50:
        issues.append(f"Episode 数量中等: {total_episodes} (建议 50+)")
    else:
        print(f"  ✅ Episode 数量充足: {total_episodes}")

    # Episode 时长
    if avg_duration < 5:
        issues.append(f"Episode 时长过短: {avg_duration:.2f}s (建议 > 5s)")
    elif avg_duration > 30:
        issues.append(f"Episode 时长过长: {avg_duration:.2f}s (可能导致训练困难)")
    else:
        print(f"  ✅ Episode 时长合理: {avg_duration:.2f}s")

    # 动作空间质量
    print("\n  动作空间质量:")
    for i, name in enumerate(joint_names):
        range_val = stats["action"]["max"][i] - stats["action"]["min"][i]
        std_val = stats["action"]["std"][i]
        if range_val < 100:
            issues.append(f"{name} 运动范围过小: {range_val:.0f}")
            print(f"    ⚠️  {name}: 运动范围={range_val:.0f} (过小)")
        elif std_val > 1000:
            issues.append(f"{name} 标准差过大: {std_val:.1f}")
            print(f"    ⚠️  {name}: 标准差={std_val:.1f} (过大)")
        else:
            print(f"    ✅ {name}: 运动范围={range_val:.0f}, 标准差={std_val:.1f}")

    # 状态-动作一致性
    print("\n  状态-动作一致性:")
    for i, name in enumerate(joint_names):
        diff = abs(stats["action"]["mean"][i] - stats["observation.state"]["mean"][i])
        if diff > 100:
            issues.append(f"{name} 状态-动作差异大: {diff:.1f}")
            print(f"    ⚠️  {name}: 差异={diff:.1f}")
        else:
            print(f"    ✅ {name}: 差异={diff:.1f}")

    # 问题汇总
    print(f"\n{'='*60}")
    print("问题汇总")
    print(f"{'='*60}")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. ⚠️  {issue}")
    else:
        print("  ✅ 未发现明显数据质量问题")

    # ===== 训练建议 =====
    print(f"\n{'='*80}")
    print("训练建议")
    print(f"{'='*80}")

    if total_episodes >= 50:
        epochs = 10
        note = "数据充足"
    elif total_episodes >= 20:
        epochs = 15
        note = "数据中等"
    else:
        epochs = 20
        note = "数据较少，需更多 epoch"

    batch_size = 8
    steps_per_epoch = int(np.ceil(total_frames / batch_size))
    total_steps = epochs * steps_per_epoch

    print(f"\n  Epochs: {epochs} ({note})")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")

    print(f"\n  推荐命令:")
    print(f"  lerobot-train \\")
    print(f"    --dataset.repo_id={repo_id} \\")
    print(f"    --dataset.root={root} \\")
    print(f"    --policy.type=act \\")
    print(f"    --steps={total_steps} \\")
    print(f"    --policy.scheduler_decay_steps={total_steps} \\")
    print(f"    --batch_size={batch_size}")

    if total_episodes < 50:
        need = 50 - total_episodes
        est_min = need * avg_frames / fps / 60
        print(f"\n  ⚠️  建议补充 {need} 个 episodes (约 {est_min:.1f} 分钟数据)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python scripts/evaluate_dataset.py <repo_id> <root>")
        print("示例: python scripts/evaluate_dataset.py local/nexarm_merged datasets/nexarm_merged")
        sys.exit(1)
    evaluate_dataset(sys.argv[1], sys.argv[2])
