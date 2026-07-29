#!/usr/bin/env python3
"""
SmolVLA 离线推理验证脚本
使用厂商提供的推理流程和处理器

与 scripts/simple_eval_act.py 的区别：
  - SmolVLA 是 VLA 模型，需要语言任务描述（task）作为输入
  - 预处理器包含 TokenizerProcessorStep（语言分词）
  - 需要离线模式加载 VLM processor（设置 HF_HUB_OFFLINE=1）
"""

import os

# 离线模式：避免 HuggingFace Hub 网络请求（VLM processor 已缓存）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors


def evaluate(checkpoint_path="outputs/train/nexarm_smolvla_finetune/checkpoints/015000/pretrained_model",
             dataset_name="local/nexarm_smolvla_clean",
             dataset_root="datasets/nexarm_smolvla/clean",
             num_samples=100):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print(f"加载模型: {checkpoint_path}")
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # 创建预处理器和后处理器（自动加载归一化统计）
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_path,
        device=device
    )

    print("✓ 已加载预处理器和后处理器")

    # 加载数据集
    print(f"\n加载数据集: {dataset_name}")
    dataset = LeRobotDataset(repo_id=dataset_name, root=dataset_root)
    print(f"数据集大小: {len(dataset)} 帧")

    # 加载stats用于计算相对误差
    stats_path = Path(dataset.root) / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    action_min = np.array(stats["action"]["min"])
    action_max = np.array(stats["action"]["max"])
    action_range = action_max - action_min

    # 采样评估
    print(f"\n评估 {num_samples} 个样本...")
    errors = []
    relative_errors = []

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    for i in tqdm(range(0, min(num_samples, len(dataset)))):
        sample = dataset[i]

        # 构造观测（包含 task 语言描述，SmolVLA 必须）
        obs = {"task": sample["task"]}
        for key in sample.keys():
            if key.startswith("observation."):
                obs[key] = sample[key].unsqueeze(0).to(device)

        # 应用预处理器
        obs_processed = preprocessor(obs)

        # 推理
        with torch.no_grad():
            action = policy.select_action(obs_processed)

        # 应用后处理器（反归一化）
        action_unnorm = postprocessor(action)

        # 计算误差
        pred = action_unnorm.squeeze().cpu().numpy()
        gt = sample["action"].numpy()

        error = np.abs(pred - gt)
        rel_error = error / action_range

        errors.append(error)
        relative_errors.append(rel_error)

    # 统计
    errors = np.array(errors)
    relative_errors = np.array(relative_errors)

    mae = np.mean(errors)
    rel_mae = np.mean(relative_errors)

    print(f"\n评估结果:")
    print(f"  平均绝对误差: {mae:.2f}")
    print(f"  平均相对误差: {rel_mae*100:.2f}%")
    print(f"  最大相对误差: {np.max(relative_errors)*100:.2f}%")

    # 各关节误差
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    print(f"\n各关节相对误差:")
    for i, name in enumerate(joint_names):
        joint_rel_mae = np.mean(relative_errors[:, i])
        joint_rel_max = np.max(relative_errors[:, i])
        print(f"  {name:15s}: {joint_rel_mae*100:5.2f}% (最大: {joint_rel_max*100:5.2f}%)")

    # 建议
    if rel_mae > 0.25:
        print(f"\n⛔ 禁止部署 - 平均相对误差过大 ({rel_mae*100:.2f}% > 25%)")
        print("建议: 增加训练数据或调整训练参数")
    elif rel_mae > 0.15:
        print(f"\n⚠️  不建议部署 - 平均相对误差较大 ({rel_mae*100:.2f}% > 15%)")
        print("建议: 在安全环境下低速测试")
    else:
        print(f"\n✅ 可以部署 - 平均相对误差可接受 ({rel_mae*100:.2f}%)")
        print("建议: 先在安全环境下测试")

    return rel_mae


if __name__ == "__main__":
    evaluate(num_samples=200)
