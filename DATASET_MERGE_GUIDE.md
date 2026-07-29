# LeRobot 数据集合并与质量评估操作手册

> 本手册提供完整的步骤和命令，可直接复制执行。适用于分批采集数据的合并、验证和质量评估。

---

## 目录

- [准备工作](#准备工作)
- [步骤一：合并数据集](#步骤一合并数据集)
- [步骤二：重新计算统计信息](#步骤二重新计算统计信息)
- [步骤三：验证合并结果](#步骤三验证合并结果)
- [步骤四：数据质量评估](#步骤四数据质量评估)
- [步骤五：异常值处理](#步骤五异常值处理)
- [步骤六：训练参数建议](#步骤六训练参数建议)
- [快速参考](#快速参考)
- [常见问题](#常见问题)

---

## 准备工作

### 确认数据集路径

```bash
ls -la datasets/
```

### 检查单个数据集

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/dataset_name \
    --root datasets/dataset_name \
    --operation.type info
```

---

## 步骤一：合并数据集

### 命令模板

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id local/merged_dataset \
    --new_root datasets/merged_dataset \
    --operation.type merge \
    --operation.repo_ids "[local/dataset1, local/dataset2, ...]" \
    --operation.roots "[datasets/dataset1, datasets/dataset2, ...]"
```

### 实际示例

合并 4 个数据集：`0715_10`, `0715_20`, `0715_30`, `0715_40`

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id local/nexarm_merged \
    --new_root datasets/nexarm_merged \
    --operation.type merge \
    --operation.repo_ids "[local/0715_10, local/0715_20, local/0715_30, local/0715_40]" \
    --operation.roots "[datasets/0715_10, datasets/0715_20, datasets/0715_30, datasets/0715_40]"
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--new_repo_id` | 合并后的数据集 ID | `local/nexarm_merged` |
| `--new_root` | 合并后的存储路径 | `datasets/nexarm_merged` |
| `--operation.type` | 操作类型 | `merge` |
| `--operation.repo_ids` | 数据集 ID 列表（JSON 数组） | `"[local/ds1, local/ds2]"` |
| `--operation.roots` | 数据集路径列表（JSON 数组） | `"[datasets/ds1, datasets/ds2]"` |

### ✅ 成功标志

输出显示正确的 Episode 数和帧数：
```
INFO Episodes: 40, Frames: 22773
```

---

## 步骤二：重新计算统计信息

### ⚠️ 此步骤不可跳过！

合并后必须重算统计信息，否则训练时归一化会出错。

### 命令模板

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/merged_dataset \
    --root datasets/merged_dataset \
    --operation.type recompute_stats \
    --operation.skip_image_video true \
    --operation.overwrite true
```

### 实际示例

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_merged \
    --root datasets/nexarm_merged \
    --operation.type recompute_stats \
    --operation.skip_image_video true \
    --operation.overwrite true
```

### 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--operation.type` | 操作类型 | `recompute_stats` |
| `--operation.skip_image_video` | 跳过图像统计（只算 action/state） | `true` |
| `--operation.overwrite` | 覆盖旧统计文件 | `true` |

### ✅ 成功标志

```
INFO Stats recomputed successfully
```

### 为什么必须重算

- 合并后需要基于所有数据重新计算全局统计量（min/max/mean/std）
- 训练时数据归一化依赖这些统计信息
- 不重算会导致归一化错误，影响训练效果

---

## 步骤三：验证合并结果

### 命令模板

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/merged_dataset \
    --root datasets/merged_dataset \
    --operation.type info \
    --operation.show_features true
```

### 实际示例

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_merged \
    --root datasets/nexarm_merged \
    --operation.type info \
    --operation.show_features true
```

### 检查清单

| 检查项 | 验证方法 | 预期结果 |
|--------|----------|----------|
| Episode 数量 | 应等于各数据集 Episode 之和 | 10+10+10+10=40 |
| 总帧数 | 应等于各数据集帧数之和 | 各帧数之和 |
| FPS | 应与原数据集一致 | 30 |
| 特征结构 | 包含 action, state, images | 完整无缺失 |

---

## 步骤四：数据质量评估

使用项目提供的 `evaluate_dataset.py` 脚本（基于 LeRobot 内置 API）：

### 执行命令

```bash
python evaluate_dataset.py local/nexarm_merged datasets/nexarm_merged
```

### 评估内容

脚本自动完成以下评估：

| 评估维度 | 内容 | 判定标准 |
|---------|------|---------|
| Episode 数量 | 总数是否充足 | <10 不足 / 10-50 中等 / 50+ 充足 |
| Episode 时长 | 平均时长是否合理 | <5s 过短 / 5-30s 合理 / >30s 过长 |
| Episode 长度分布 | 最短/最长/标准差 | 标准差/均值 < 0.2 为一致 |
| 动作空间质量 | 各关节运动范围和标准差 | 范围>100 且 标准差<1000 |
| 状态-动作一致性 | 均值差异 | 差异 < 100 |
| 数据多样性 | 变异系数 CV=std/mean | 0.05-0.3 适中 |
| 视频文件完整性 | 视频数量和大小 | 文件存在且大小合理 |

### 输出示例

```
================================================================================
数据集质量评估报告
================================================================================

============================================================
基本信息
============================================================
  总 Episode 数:  40
  总帧数:          22773
  FPS:             30
  平均每Episode:   569.3 帧 (18.98 秒)
  总时长:          12.65 分钟

============================================================
动作空间统计
============================================================
关节                        最小值      最大值         均值        标准差       运动范围
----------------------------------------------------------------------
shoulder_pan.pos         1726     2185     1997.6      112.6        459
...

============================================================
数据质量评估
================================================================================
  ✅ Episode 时长合理: 18.98s

  动作空间质量:
    ✅ shoulder_pan.pos: 运动范围=459, 标准差=112.6
    ...

============================================================
问题汇总
============================================================
  1. ⚠️  Episode 数量中等: 40 (建议 50+)

================================================================================
训练建议
================================================================================
  Epochs: 15 (数据中等)
  Total steps: 42705

  推荐命令:
  lerobot-train \
    --dataset.repo_id=local/nexarm_merged \
    --dataset.root=datasets/nexarm_merged \
    --policy.type=act \
    --steps=42705 \
    --policy.scheduler_decay_steps=42705 \
    --batch_size=8
```

---

## 步骤五：异常值处理

### 问题背景

串口通信丢包或舵机未就绪时，状态空间（observation.state）会读取到 **12-bit 编码器极限值**（0 或 4095），这些不是舵机的真实位置，不应出现在训练数据中。

### 异常类型与处理策略

| 异常类型 | 特征 | 影响 | 处理策略 |
|---------|------|------|---------|
| **首帧初始化异常** | Episode 第 0 帧状态全为边界值 | 模型学到错误的初始状态感知 | **删除整个 Episode** |
| **中间帧通信丢包** | 单帧多关节同时出现极限值，action 正常 | 归一化范围被拉宽 | 占比 < 0.1% 可忽略；否则用前帧值替换 |

### 检测异常 Episode

使用以下脚本检测状态空间中的编码器极限值异常：

```python
import pandas as pd
import numpy as np
import glob

root = "datasets/merged_dataset"
parquet_files = sorted(glob.glob(f"{root}/data/*/*.parquet"))
dfs = [pd.read_parquet(f) for f in parquet_files]
df = pd.concat(dfs, ignore_index=True)

# 检测异常帧：状态值 < 500 或 > 3800（编码器极限值区域）
states = np.stack(df['observation.state'].values)
anomaly_mask = (states < 500) | (states > 3800)
anomaly_frames = anomaly_mask.any(axis=1)

# 找出异常 Episode
ep_indices = df['episode_index'].values
anom_episodes = sorted(np.unique(ep_indices[anomaly_frames]))

# 区分首帧异常和中间帧异常
for ep in anom_episodes:
    ep_mask = ep_indices == ep
    ep_anom = anomaly_frames[ep_mask]
    first_anom_frame = np.where(ep_anom)[0]
    if 0 in first_anom_frame:
        print(f"Episode {ep}: 首帧初始化异常 → 建议删除")
    else:
        print(f"Episode {ep}: 帧 {first_anom_frame} 通信丢包 → 可忽略或替换")
```

### 删除异常 Episode

使用 `lerobot-edit-dataset` 的 `delete_episodes` 操作，**保留原始数据，输出新数据集**：

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/merged_dataset \
    --root datasets/merged_dataset \
    --new_repo_id local/clean_dataset \
    --new_root datasets/clean_dataset \
    --operation.type delete_episodes \
    --operation.episode_indices "[30]"
```

### 实际示例

删除 Episode 30（首帧状态全为编码器边界值，舵机未就绪）：

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_smolvla_merged \
    --root datasets/nexarm_smolvla/merged \
    --new_repo_id local/nexarm_smolvla_clean \
    --new_root datasets/nexarm_smolvla/clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[30]"
```

### 删除后必须重算统计信息

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/clean_dataset \
    --root datasets/clean_dataset \
    --operation.type recompute_stats \
    --operation.skip_image_video true \
    --operation.overwrite true
```

### ✅ 成功标志

```
INFO Created new dataset with 49 episodes
INFO Episodes: 49, Frames: 29303
INFO Stats recomputed successfully
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--repo_id` | 源数据集 ID | `local/nexarm_smolvla_merged` |
| `--root` | 源数据集路径 | `datasets/nexarm_smolvla/merged` |
| `--new_repo_id` | 新数据集 ID | `local/nexarm_smolvla_clean` |
| `--new_root` | 新数据集路径 | `datasets/nexarm_smolvla/clean` |
| `--operation.type` | 操作类型 | `delete_episodes` |
| `--operation.episode_indices` | 要删除的 Episode 索引（JSON 数组） | `"[30]"` 或 `"[0, 4, 7]"` |

### ⚠️ 注意事项

1. **Episode 索引基于合并后的数据集**，不是原始子数据集的索引
2. **删除多个 Episode 时**，索引是删除前的原始索引（不是递减后的索引）
3. **原始数据集不会被修改**，新数据集是独立副本
4. 删除后必须重新计算统计信息（步骤二）

---

## 步骤六：训练参数建议

质量评估脚本会自动计算并输出推荐训练命令。也可手动计算：

### 计算方法

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset(repo_id="local/nexarm_merged", root="datasets/nexarm_merged")

# Epochs 根据数据量调整
if ds.num_episodes >= 50: epochs = 10
elif ds.num_episodes >= 20: epochs = 15
else: epochs = 20

batch_size = 8
total_steps = epochs * int(np.ceil(len(ds) / batch_size))
```

### 训练参数推荐值

| 数据量 | Epochs | 说明 |
|--------|--------|------|
| < 20 | 20 | 数据较少，需要更多训练 |
| 20-50 | 15 | 数据中等 |
| ≥ 50 | 10 | 数据充足 |

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `batch_size` | 8 | GPU 内存足够时可增大 |
| `n_action_steps` | 100 | ACT 默认值 |
| `chunk_size` | 100 | ACT 默认值 |
| `n_obs_steps` | 1 | 单帧观察 |

---

## 快速参考

### 完整操作流程（复制即用）

```bash
# ==================== 步骤1：合并数据集 ====================
python -m lerobot.scripts.lerobot_edit_dataset \
    --new_repo_id local/nexarm_merged \
    --new_root datasets/nexarm_merged \
    --operation.type merge \
    --operation.repo_ids "[local/0715_10, local/0715_20, local/0715_30, local/0715_40]" \
    --operation.roots "[datasets/0715_10, datasets/0715_20, datasets/0715_30, datasets/0715_40]"

# ==================== 步骤2：重算统计（必须！） ====================
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_merged \
    --root datasets/nexarm_merged \
    --operation.type recompute_stats \
    --operation.skip_image_video true \
    --operation.overwrite true

# ==================== 步骤3：验证结果 ====================
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_merged \
    --root datasets/nexarm_merged \
    --operation.type info \
    --operation.show_features true

# ==================== 步骤4：质量评估 ====================
python evaluate_dataset.py local/nexarm_merged datasets/nexarm_merged

# ==================== 步骤5：异常值处理（如有需要） ====================
# 检测异常 Episode 后，删除首帧初始化异常的 Episode
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_merged \
    --root datasets/nexarm_merged \
    --new_repo_id local/nexarm_clean \
    --new_root datasets/nexarm_clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[30]"

# 删除后重算统计（必须！）
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/nexarm_clean \
    --root datasets/nexarm_clean \
    --operation.type recompute_stats \
    --operation.skip_image_video true \
    --operation.overwrite true

# ==================== 步骤6：开始训练 ====================
lerobot-train \
  --dataset.repo_id=local/nexarm_clean \
  --dataset.root=datasets/nexarm_clean \
  --policy.type=act \
  --steps=54945 \
  --policy.scheduler_decay_steps=54945 \
  --batch_size=8
```

### 关键检查清单

- [ ] **步骤1**：合并后显示正确的 Episode 数和帧数
- [ ] **步骤2**：显示 `Stats recomputed successfully`
- [ ] **步骤3**：验证 Episode 数 = 各数据集之和
- [ ] **步骤4**：数据质量无警告或警告已理解
- [ ] **步骤5**：异常 Episode 已删除，统计信息已重算
- [ ] **步骤6**：训练参数已根据数据量调整

---

## 常见问题

### Q1: 合并时提示路径错误？

确保 `repo_ids` 和 `roots` 参数是 JSON 数组格式（带引号）：

```bash
# ✅ 正确
--operation.repo_ids "[local/ds1, local/ds2]"
--operation.roots "[datasets/ds1, datasets/ds2]"

# ❌ 错误（缺少引号）
--operation.repo_ids [local/ds1, local/ds2]
```

### Q2: 合并后统计信息未更新？

跳过了步骤2。必须执行 `recompute_stats`，否则训练归一化会出错。

### Q3: 训练时报归一化错误？

原因：合并后未重算统计信息。解决：重新执行步骤2，确认 `stats.json` 已更新，重新开始训练。

### Q4: Episode 数量验证失败？

```bash
# 分别查看各数据集 Episode 数
for dir in datasets/0715_*; do
    echo "$dir:"; cat $dir/meta/info.json | grep "total_episodes"
done

# 查看合并后 Episode 数
cat datasets/nexarm_merged/meta/info.json | grep "total_episodes"
```

### Q5: 如何判断数据质量是否合格？

**质量标准**：
- ✅ Episode 数量 ≥ 50（理想）
- ✅ Episode 时长 5-30 秒
- ✅ 各关节运动范围 > 100
- ✅ 状态-动作差异 < 100
- ✅ 变异系数 0.05-0.3（适中）

**处理建议**：
- 数据不足（< 50 episodes）：增加 epochs 到 15-20
- Episode 过短/过长：检查任务设计
- 运动范围小：增加场景多样性

### Q6: 状态空间出现 0 或 4095 等编码器极限值？

**原因**：串口通信丢包或舵机未就绪时，编码器返回默认极限值（12-bit 范围 0-4095），不是真实位置。

**判断方法**：
- 同一帧多个关节同时出现极限值，但 action 值正常 → 通信丢包
- Episode 第 0 帧状态全为极限值 → 舵机初始化未就绪

**处理方式**：
- 首帧初始化异常 → 删除整个 Episode（见步骤五）
- 中间帧偶发丢包，占比 < 0.1% → 可忽略
- 中间帧丢包占比 > 0.1% → 用前帧状态值替换（前向填充）

### Q7: 删除 Episode 后训练报错？

删除 Episode 后必须重新计算统计信息：

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id local/clean_dataset \
    --root datasets/clean_dataset \
    --operation.type recompute_stats \
    --operation.skip_image_video true \
    --operation.overwrite true
```

---

## 附录：lerobot_edit_dataset 操作类型速查

| 操作类型 | 说明 | 常用参数 |
|---------|------|---------|
| `merge` | 合并多个数据集 | `--new_repo_id`, `--new_root`, `--operation.repo_ids`, `--operation.roots` |
| `recompute_stats` | 重算统计信息 | `--operation.skip_image_video`, `--operation.overwrite` |
| `info` | 查看数据集信息 | `--operation.show_features` |
| `split` | 分割数据集 | `--operation.splits` |
| `delete_episodes` | 删除指定 episodes | `--operation.episode_indices` |
| `remove_feature` | 移除特征 | `--operation.feature_names` |
| `modify_tasks` | 修改任务标签 | `--operation.new_task`, `--operation.episode_tasks` |
| `convert_image_to_video` | 图像转视频 | `--operation.vcodec`, `--operation.crf` |

---

**文档版本**: 2026-07-22 (v2, 新增异常值处理流程)
**适用范围**: LeRobot v3.0+
