# NexArm 6-DOF 机械臂 — LeRobot 完整使用指南

NexArm 是一款基于 ESP32 + AT32F421 协处理器的高性能 6 自由度桌面机械臂，采用 HX-30HM 高性能串行总线舵机，具备 12 位精度与 1 Mbps 高速通信，响应迅速、扭矩强劲。本文档涵盖从硬件连接、遥操作测试、数据采集、模型训练到策略推理的完整流程。

---

## 目录

- [硬件概述](#硬件概述)
- [环境安装](#环境安装)
- [第一步：查找串口](#第一步查找串口)
- [第二步：查找摄像头](#第二步查找摄像头)
- [第三步：遥操作测试](#第三步遥操作测试)
- [第四步：采集数据集](#第四步采集数据集)
  - [录制状态灯（可选）](#录制状态灯可选)
- [第五步：训练策略模型](#第五步训练策略模型)
- [第六步：策略推理部署](#第六步策略推理部署)
  - [Pi0 int8 量化部署（RTX 5060 8GB）](#pi0-int8-量化部署rtx-5060-8gb2025-08-新增)
- [代码架构说明](#代码架构说明)
- [PR 提交指南](#pr-提交指南)
- [常见问题排查](#常见问题排查)

---

## 硬件概述

### 硬件组成

| 组件 | 说明 |
|------|------|
| **主臂 (Leader)** | ESP32 开发板，直接驱动 6 个 HX-30HM 舵机。用于遥操作，操作者自由拖动此臂 |
| **从臂 (Follower)** | ESP32 开发板 + AT32F421 协处理器，驱动 6 个 HX-30HM 舵机。镜像主臂动作或执行策略输出 |
| **舵机** | HX-30HM 高性能串行总线舵机，12 位精度 (0–4095)，1 Mbps 波特率，高扭矩、快响应 |
| **摄像头** | 2 个 USB 摄像头 — "front"（工作区俯视）和 "wrist"（末端执行器近景），640×480 @ 30 FPS |

### 关节布局 (6 DOF)

| 关节编号 | 名称 | 说明 |
|----------|------|------|
| 1 | `shoulder_pan` | 底座旋转 |
| 2 | `shoulder_lift` | 主臂/从臂之间镜像映射 (4096 − pos) |
| 3 | `elbow_flex` | 肘关节 |
| 4 | `wrist_flex` | 腕部俯仰 |
| 5 | `wrist_roll` | 腕部旋转 |
| 6 | `gripper` | 夹爪开合，映射范围 [1195, 2833] |

### 通信协议

NexArm 使用自定义 CommProtocol 通过 USB 串口通信：

```
帧格式: [0xFF][0xFF][ID][LEN][CMD][ARGS...][CHECKSUM]
```

| 命令 | 功能 | 方向 |
|------|------|------|
| CMD 68 | 进入/退出 LeRobot 桥接模式（仅从臂） | 主机 → 从臂 |
| CMD 96 | 读取 6 个舵机位置（回复：12 字节） | 主机 → 设备 → 主机 |
| CMD 97 | 写入 6 个舵机位置（12 字节，无回复） | 主机 → 设备 |
| CMD 98 | 使能/失能力矩 | 主机 → 设备 |

---

## 环境安装

### 1. 创建环境并安装

推荐使用 conda 创建独立环境：

```bash
conda create -n nexarm python=3.12 -y
conda activate nexarm

# 安装 lerobot（editable 模式，包含记录/回放/遥操作/串口通信等核心依赖）
pip install -e ".[core_scripts]"
```

如果需要遥操作时实时可视化（rerun/foxglove 界面），`core_scripts` 已包含 viz 依赖，无需额外安装。

> **注意：** `core_scripts` 包含 dataset + hardware + viz，涵盖 NexArm 所需的全部功能（pyserial 串口通信、视频编解码、可视化）。也可以单独安装 `pip install -e "."` 然后按需添加 extras（如 `pip install -e ".[dataset,hardware]"` 最小安装不含可视化）。
>
> **可视化说明：** 不安装 viz 也可以正常使用遥操作、采集、训练和推理，只是没有实时画面显示。配置文件中已默认关闭 `display_data`。`core_scripts` 已包含 rerun 和 foxglove 可视化依赖，无需额外安装。

也可以使用 venv 代替 conda：

```bash
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[core_scripts]"
```

### 2. 验证安装

```bash
python -c "import lerobot.rollout; print('OK')"
```

### 3. 连接硬件

- 将 **从臂 (Follower/Slave)** 的 ESP32 通过 USB 连接到电脑
- 将 **主臂 (Leader/Master)** 的 ESP32 通过 USB 连接到电脑
- 插入两个 USB 摄像头（front + wrist）

### 平台兼容性

| 平台 | 状态 | 说明 |
|------|------|------|
| Windows 10/11 | 已验证 | CH340 驱动需安装，串口格式 COM19 |
| Ubuntu 20.04+ | 已验证 | 串口格式 /dev/ttyUSB0，需 dialout 权限 |
| macOS | 应兼容 | 串口格式 /dev/tty.usbserial-xxx |

---

## 第一步：查找串口

确定主臂和从臂各自对应哪个串口。

```bash
python -m lerobot.scripts.lerobot_find_port
```

Windows 上的典型输出：

| 端口 | 设备 |
|------|------|
| COM18 | 主臂 (Leader/Master) ESP32 |
| COM19 | 从臂 (Follower/Slave) ESP32 |

Linux 上通常是 `/dev/ttyUSB0` 和 `/dev/ttyUSB1`。

> **提示：** 如果无法确定哪个是主臂哪个是从臂，可以只插一个试试，或查看设备管理器中的 USB 设备描述。

---

## 第二步：查找摄像头

确定哪个摄像头索引对应"前方"和"腕部"。

```bash
python -m lerobot.scripts.lerobot_find_cameras opencv
```

或者手动扫描并拍照确认：

```python
import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"cam_{i}.png", frame)
            print(f"摄像头 {i}: 可用")
        cap.release()
```

查看保存的图片，确定：
- **front**（前方摄像头）：拍到整个工作区的俯视画面
- **wrist**（腕部摄像头）：拍到末端执行器/夹爪的近景

> **注意：** 摄像头索引会随 USB 插拔顺序变化。每次重新插拔后建议重新扫描确认。

---

## 第三步：遥操作测试

验证主臂-从臂的遥操作链路是否正常。主臂力矩关闭，操作者自由拖动主臂，从臂实时镜像跟随。

### 方式一：使用配置文件

先根据你的实际情况修改 `examples/nexarm_act/teleoperate.yaml` 中的串口和摄像头索引，然后运行：

```bash
python -m lerobot.scripts.lerobot_teleoperate --config_path=examples/nexarm_act/teleoperate.yaml
```

### 方式二：使用命令行参数

```bash
python -m lerobot.scripts.lerobot_teleoperate --robot.type=nexarm_follower --robot.port=COM19 --teleop.type=nexarm_leader --teleop.port=COM18 --fps=30
```

### 验证要点

- 拖动主臂时从臂应该实时跟随
- 各个关节方向应一致（shoulder_lift 自动镜像）
- 夹爪开合应该正常映射

> **提示：** 如果遥操作感觉卡顿，检查是否有其他程序（如 Arduino IDE 串口监视器）占用了 COM 端口。

---

## 第四步：采集数据集

通过遥操作录制示范数据，用于后续模仿学习训练。操作者拖动主臂执行任务，从臂跟随执行，同时录制摄像头画面和关节位置。

### 运行采集

先修改 `examples/nexarm_act/record.yaml` 中的配置（串口、摄像头、数据集名称等），然后运行：

```bash
python -m lerobot.scripts.lerobot_record --config_path=examples/nexarm_act/record.yaml
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset.repo_id` | `local/nexarm_pick` | 数据集名称（本地保存） |
| `dataset.num_episodes` | 50 | 录制的 episode 数量 |
| `dataset.episode_time_s` | 10 | 每个 episode 的时长（秒） |
| `dataset.reset_time_s` | 10 | episode 之间用于重置场景的等待时间 |
| `dataset.fps` | 30 | 录制帧率 |
| `dataset.push_to_hub` | false | 是否上传到 HuggingFace Hub |
| `lamp_port` | None | 状态灯串口（如 `COM4`），None = 禁用灯 |

### 采集流程

1. 脚本启动后自动连接双臂和摄像头
2. 每个 episode：
   - 按 Enter 开始录制
   - 拖动主臂执行任务（如抓取物体）
   - 到达 `episode_time_s` 后自动结束录制
   - 在 `reset_time_s` 内将物体重置到初始位置
3. 全部 episode 录完后，数据集保存到本地并可选上传到 HuggingFace Hub

### 录制状态灯（可选）

录制时可连接一个 ESP32 + NeoPixel 环形灯，实时显示录制进度：

| 灯色 | 含义 |
|------|------|
| 🔴 红色 | 正在录制 episode |
| 🟠 橙色 | episode 间重置场景 |
| 🟢 绿色 | 全部 episode 录制完成 |

**查找灯的串口：**

> **依赖：** 需要 `pyserial`（已包含在 `pip install -e ".[core_scripts]"` 中；最小安装用户需 `pip install pyserial`）。

```bash
# 方式一：CLI 命令（推荐）
lerobot-find-lamp-port

# 方式二：直接运行模块
python -m lerobot.utils.find_lamp_port
```

按提示操作：先记录当前端口列表，然后拔掉灯的 USB，按 Enter，消失的端口即灯的串口。

**采集时启用灯：**

```bash
python -m lerobot.scripts.lerobot_record --config_path=examples/nexarm_act/record.yaml --lamp_port=COM4
```

或在 yaml 配置文件中添加：

```yaml
lamp_port: COM4    # Windows
# lamp_port: /dev/ttyUSB2  # Linux
```

> **技术细节：** 灯通信在后台 daemon 线程中执行，不会阻塞主录制循环。进度更新节流至 ~1 Hz，不会淹没串口。灯通信失败时自动静默，不影响录制。依赖 `pyserial`（NexArm 安装已包含）。

### 仅本地保存（不上传）

```bash
python -m lerobot.scripts.lerobot_record --config_path=examples/nexarm_act/record.yaml --dataset.repo_id=local/nexarm_pick --dataset.push_to_hub=false
```

### 数据质量建议

- **至少录制 50 个 episode** 才能获得可用的训练效果
- 每次录制保持动作一致性（相同的起始位置、相似的运动轨迹）
- 确保摄像头画面清晰、光照稳定
- 录制完成后可用 `lerobot_dataset_viz` 查看数据集质量

---

## 第五步：训练策略模型

使用 ACT（Action Chunking with Transformers）策略在采集的数据集上训练。

### 安装训练依赖

训练需要额外的依赖（accelerate、wandb 等），安装方式：

```bash
pip install -e ".[core_scripts,training]"
```

### 运行训练

```bash
python -m lerobot.scripts.lerobot_train --dataset.repo_id=local/nexarm_pick --policy.type=act --output_dir=outputs/train/nexarm_act --batch_size=32 --steps=100000 --save_freq=25000
```

> **注意：** 不建议使用 `--config_path=examples/nexarm_act/train.yaml` 加 `--dataset.repo_id` 覆盖的方式，可能会报 JSON 格式错误。直接使用命令行参数即可。

### 训练建议

| 项目 | 建议值 | 说明 |
|------|--------|------|
| 训练设备 | CUDA GPU | 推荐 RTX 3090 或以上 |
| 示范 episode 数 | 50+ | 越多越好，至少 50 个 |
| 训练步数 | 100,000 | 根据 loss 收敛情况调整 |
| 批量大小 | 32 | 默认值，显存不足可降到 16 |
| 保存频率 | 25,000 | 每 25k 步保存一个检查点 |

> **关于 episode 时长：** 每个 episode 可以是 10-30 秒。ACT 训练时从 episode 中随机采样起始帧，取接下来 chunk_size=100 帧（3.3秒）作为预测目标。episode 长度不影响 chunk_size 设置。关键是每个 episode 内包含完整的任务动作（如完整的一次抓取-放置）。

### 训练输出

训练完成后，模型检查点保存在：

```
outputs/train/nexarm_act/checkpoints/last/pretrained_model/
├── config.json                    # 策略配置
├── model.safetensors              # 模型权重
├── policy_preprocessor.json       # 输入预处理流水线
├── policy_preprocessor_step_3_normalizer_processor.safetensors
├── policy_postprocessor.json      # 输出后处理流水线
├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
└── train_config.json              # 训练配置记录
```

---

## 第六步：策略推理部署

将训练好的策略部署到真实机器人上。从臂自主执行模型预测的动作，无需主臂。

### 运行推理

```bash
python -m lerobot.scripts.lerobot_rollout --config_path=examples/nexarm_act/inference.yaml --policy.path=outputs/train/nexarm_act/checkpoints/last/pretrained_model
```

### 重要说明

- **不需要传 `--teleop` 参数** — 策略替代人类操作者
- 数据集名称必须以 `rollout_` 开头（配置文件中已设置为 `rollout_nexarm_pick`）
- 每次推理会自动在 repo_id 后追加时间戳，生成唯一目录，不会冲突
- 推理结果会录制成数据集，方便后续分析评估
- 配置文件中 `device: cuda`，有 GPU 时使用 GPU 推理；无 GPU 时系统自动退回 CPU
- 推理结束后机械臂会回到初始位置并保持力矩（不掉电）

### 推理性能参考

| 设备 | 大约帧率 | 说明 |
|------|----------|------|
| NVIDIA GPU | 30 Hz | 满速实时控制 |
| CPU (i7/i5) | 20–30 Hz | ACT 策略利用 action chunking，模型每 100 帧推理一次，其余帧从队列弹出，CPU 也能实时 |

> **ACT 策略在 CPU 上可以正常实时运行。** chunk_size=100 意味着模型推理（~700ms）只在队列为空时触发，其余 99 帧只需 ~1ms 弹出动作，整体帧率可达 20-30 Hz。

### 不同策略模式

| 策略 | 命令 | 说明 |
|------|------|------|
| `base` | `--strategy.type=base` | 仅推理，不录制 |
| `sentry` | `--strategy.type=sentry` | 连续录制 + 自动保存（推荐用于评估） |
| `highlight` | `--strategy.type=highlight` | 环形缓冲区，按键保存精彩片段 |
| `dagger` | `--strategy.type=dagger` | 人机协作，需要主臂 |

### 推理安全机制（2024-07 新增）

为解决串口通信丢包导致的异常状态值问题，新增三层安全保护：

#### 1. 状态异常检测与恢复

- **检测范围：** `[100, 3995]` (排除边界附近 100 个单位)
- **丢包识别：** 值 ≤ 100 或 ≥ 3995 视为丢包异常
- **自动恢复：** 用上一帧合法状态替代，避免异常值输入模型
- **连续告警：** 连续 5 帧异常触发 ERROR 日志，提示检查硬件

#### 2. 动作相对幅度限制

- **限制幅度：** 单步最大移动 500 单位 (~12% 全范围 [0, 4095])
- **防止突变：** 避免策略输出异常值导致舵机突然大幅跳转
- **性能无损：** 安全检查使用缓存位置，不增加串口读取开销

#### 3. 配置化参数（软限制）

在 `inference.yaml` 中可自定义：

```yaml
robot:
  type: nexarm_follower
  state_safety_margin: 100      # 安全范围边界，默认 100
  max_consecutive_anomalies: 5  # 连续异常阈值，默认 5 帧
  max_relative_target: 500      # 最大相对动作幅度，默认 500
```

**禁用安全机制：**

```yaml
robot:
  state_safety_margin: 0         # 禁用状态异常检测
  max_relative_target: 4095      # 禁用动作幅度限制
```

#### 性能影响

| 操作 | 原始 | 新增 | 总开销 |
|------|------|------|--------|
| get_observation() | ~15-20ms | ~0.01ms | ~15-20ms |
| send_action() | ~20-30ms | ~0.01ms | ~20-30ms |
| **控制频率** | 30 Hz | - | **30 Hz（不变）** |

**详细文档：** 参见 [`SAFETY_MECHANISMS_SUMMARY.md`](./SAFETY_MECHANISMS_SUMMARY.md)

#### 4. 首帧阻尼器（2025-08 新增）

**问题背景：** 策略推理启动时，首次前向传播耗时较长（SmolVLA ~556ms，ACT ~700ms），期间机器人完全静止。推理完成后首个 action 与当前位置的偏差可达 300-500 raw units，视觉上表现为明显的"抽搐"。现有的 `ensure_safe_goal_position`（max_relative_target=500）不认为此偏差异常，因为安全系统没有时间感知——556ms 空闲后的 500 单位跳跃与 33ms 间的 500 单位跳跃被同等对待。

**检测机制（双重条件，杜绝误触发）：**

1. **时间间隙** `gap_since_obs > 0.2s`：从上次 `get_observation()` 到本次 `send_action()` 的间隔超过 200ms。正常推理帧间隔约 33-40ms，遥控模式约 7ms，推理慢帧约 180ms — 只有首次推理后的首帧才会超过 200ms
2. **位置偏差** `max_delta > 50`：目标位置与当前位置的最大关节偏差超过 50 raw units。排除静止或微小调整场景

两个条件**同时满足**时才激活阻尼器。

**阻尼行为：**

- 激活后，执行因子从 **0.1** 线性递增至 **1.0**（15 帧内）
- 帧 0：只执行 10% 的偏移（如 500 单位跳跃 → 只移动 50 单位）
- 帧 7：执行 57%
- 帧 14：执行 97%
- 帧 15+：100% 全速执行
- 阻尼器位于 `ensure_safe_goal_position` 之前，两层安全机制串联生效

**为什么不会"永远慢一拍"：** SmolVLA 输出绝对位置（非相对增量），阻尼器修改的只是发送给舵机的目标，不影响策略队列。下一帧 `get_observation()` 读取真实位置，`ensure_safe_goal_position()` 基于真实位置重新计算。机器人约 0.5s 后完全追上策略轨迹，对 30s 推理任务影响 <2%。

**不误触发的场景验证：**

| 场景 | gap_since_obs | max_delta | 阻尼触发？ |
|---|---|---|---|
| Sync 推理首帧 | ~556ms ✅ | ~300-500 ✅ | ✅ 正确触发 |
| Sync 后续帧 | ~40ms | — | ❌ |
| Teleoperate 遥控 | ~7ms | — | ❌ |
| Record 录制 | ~7ms | — | ❌ |
| Replay 回放 | ~7ms | — | ❌ |
| DAgger 遥控修正 | ~7ms | — | ❌ |
| return_to_initial_position | ~30ms | — | ❌ |
| move_to_start_position | ~30ms | — | ❌ |
| follower_smooth_move_to | ~33ms | — | ❌ |
| 推理 5.5Hz 慢帧 | ~180ms | — | ❌ |
| connect() 内部 | 不经过 send_action | — | N/A |

**可配置参数：**

```yaml
robot:
  type: nexarm_follower
  # 首帧阻尼器参数
  first_frame_gap_threshold_s: 0.2   # 激活阈值：观测-动作时间间隙（秒）
  first_frame_min_delta: 50          # 激活阈值：最小位置偏差（raw units）
  first_frame_warmup_frames: 15      # 阻尼持续帧数（15帧 ≈ 0.5s @30fps）
```

**禁用阻尼器：**

```yaml
robot:
  first_frame_gap_threshold_s: 999   # 设置极大值，永不触发
```

**性能影响：**

| 操作 | 原始 | 新增 | 总开销 |
|------|------|------|--------|
| send_action() | ~5-10ms（串口I/O） | ~0.002ms（阻尼计算） | ~5-10ms（不变） |
| 控制频率 | 30 Hz | — | 30 Hz（不变） |

**RTC 兼容性说明：** 当前阻尼器基于时间间隙检测，对 Sync 推理引擎有效。RTC 引擎的首帧问题更复杂（ActionQueue 的 delay 机制会丢弃 chunk 前几步），需要独立的阻尼方案（在 `send_next_action` 或 `robot_wrapper` 层面实现）。RTC 的阻尼器不在本方案的范围内，后续按需开发。

**合并分支说明：** 首帧阻尼器所有代码均在 NexArm 专属文件内（`nexarm_follower.py` + `config_nexarm_follower.py`），不修改 LeRobot 上游核心代码，合并官方更新时零冲突风险。

---

### Pi0 int8 量化部署（RTX 5060 8GB，2025-08 新增）

**问题背景：** Pi0 (flow) 在 4090 上训练产物 `model.safetensors` 约 **8.3GB**，这是混合精度：bf16 语言塔+action expert（~6.7GB）+ fp32 视觉塔/layernorm/proj（~1.6GB，由 `to_bfloat16_for_selected_params` 强制保留）。RTX 5060 Laptop 总显存 ~8GB，扣 Xorg 后原始权重直接加载 **必然 OOM**——连权重都放不下，更别提 flow 去噪激活。

> 对比：ACT/SmolVLA 权重小，5060 可直接加载；Pi0-FAST 7.2GB 勉强能跑；**Pi0 flow 8.3GB 必须量化**。

**方案选型（路径 A：运行时 weight-only 量化）：**

| 路径 | 在哪量化 | 精度 | 复杂度 | 采用 |
|------|---------|------|--------|------|
| A. 运行时 torchao int8wo | 5060 加载后 | 与离线一致（weight-only RTN 与硬件无关） | 低，不改 checkpoint | ✅ |
| B. 4090 离线量化产物 | 4090 | 同 A | 高，需自定义 int4/int8 序列化 loader | ❌ |
| C. 4090 GPTQ/AWQ 激活感知 | 4090 | 更好 | 很高，换工具链 | 备选（仅 A 精度不达标时） |

> 关键事实：weight-only RTN 量化（torchao `int8_weight_only`）是确定性操作，只看权重数值分布，**不需要校准数据/前向 pass**。因此在 4090 上量化和在 5060/CPU 上量化得到的 int8 权重数值完全相同——"更好硬件→更好量化"是误解，只对激活感知量化（GPTQ/AWQ）才成立。

**实现（3 处改动 + 1 个测试文件，均在 NexArm 专属范围内）：**

1. `src/lerobot/policies/pi0/configuration_pi0.py`：`PI0Config` 加 `quantization: str = "none"`（取值 `none`|`int8wo`）+ `__post_init__` 校验。
2. `src/lerobot/policies/pi0/modeling_pi0.py`：
   - `__init__`：开启量化时跳过 `self.model.to(config.device)`，让模型留在 CPU，使 `from_pretrained` 能把 8.3GB 权重加载进 CPU RAM（直接上 cuda 会 OOM）。
   - 新增 `_apply_weight_only_quantization()`：用 `torchao.quantization.quantize_` + `Int8WeightOnlyConfig()`，**逐模块**量化 bf16 Linear（252 层），跳过 `vision_tower`/`multi_modal_projector`/`layernorm`/`model.norm`/`embed_tokens`/`lm_head`/`action_*_proj`/`state_proj`/`action_time_mlp`。
   - `from_pretrained`：`load_state_dict` 后调用量化，再 `model.to(device)`。量化+搬运放在 `try/except` 之外，让错误暴露。
3. `test_load_pi0_int8_5060.py`：基于 `test_load_5060.py` 改写，验证量化层数/峰值 VRAM/数值正确性/延迟吞吐。
4. 原文件已备份为 `*.quant_backup_<时间戳>`。

> torchao 0.18 API 变更：废弃了 callable `filter_fn` 形式的 `quantize_`，改为逐模块 `quantize_(linear, Int8WeightOnlyConfig())`。

**环境搭建关键点：**

```bash
# 1. 安装量化依赖（唯一必需包，配 torch 2.11）
pip install torchao==0.18.0
```

```bash
# 2. 【必须】修复 diffusers 0.35.2 + torchao 0.18 的 logger 加载顺序 bug
#    否则 import lerobot.policies 时 NameError（任何策略 import 都崩，非仅 Pi0）
#    症状: name 'logger' is not defined @ diffusers/quantizers/torchao/torchao_quantizer.py
#    原因: torchao 0.18 移除了 NF4Tensor，触发 diffusers 的 except 分支，
#          而该分支引用的 logger 定义在调用之后。
#    修复: 把 logger = logging.get_logger(__name__) 提到 _update_torch_safe_globals() 之前。
#    已对环境内该文件打补丁并备份为 .bak_<时间戳>。升级 diffusers (>=0.36) 也可解决。
```

```bash
# 3. 可忽略的 torchao 警告（不影响 int8 量化）：
#    - _C_cutlass_90a.abi3.so 加载失败 → Hopper(sm_90a) 专用，Blackwell 用不到
#    - _C_mxfp8.cpython-310-...so 加载失败 → wheel 里混进的 py3.10 编译产物，与 py3.12 不符
#    - "Unable to import torchao Tensor objects" → 同上，仅影响 torchao 序列化 checkpoint 加载，本方案不涉及
```

```bash
# 4. int4 暂不可用：torchao 0.18 的 Int4WeightOnlyConfig 在 sm_120(Blackwell) 上
#    tinygemm kernel 报 Requires mslk >= 1.0.0，需额外装 mslk，稳定性未知。
#    int8 已足够（峰值 6.29GB），故默认走 int8。
```

**验证命令与实测结果：**

```bash
cd lerobot-awesome-explore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
python test_load_pi0_int8_5060.py
```

| 指标 | 实测值 | 判定 |
|------|--------|------|
| 量化层数 | 252/422 bf16 Linear | 视觉塔/layernorm 正确跳过 |
| 量化后权重 | Int8Tensor 4.27GB + bf16 2.45GB + fp32 1.56GB | — |
| **峰值 VRAM** | **6.29 GB** | PASS（8GB 卡留 1.7GB 余量） |
| 单 chunk 生成 (p95) | 400 ms | 10 步去噪，~40ms/step |
| **可持续控制 fps** | **125 fps** | 远超 30fps 控制需求 |
| int8 相对误差 | ~0.5% | 对 VLA 推理无实质影响 |

**真实推理部署命令（5060）：**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
python -m lerobot.scripts.lerobot_rollout \
  --config_path=examples/nexarm_pi0/inference_base.yaml \
  --policy.path=outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model \
  --policy.quantization=int8wo \
  --policy.gradient_checkpointing=false --policy.use_amp=false --use_torch_compile=false
```

> `--policy.quantization=int8wo` 是 5060 必加项（`inference_base.yaml` 的 policy 段在 `--policy.path` 传入时被忽略，必须 CLI）。4090/48GB 等大显存机器可省略，走 `quantization=none` 原始 bf16 路径。

**质量验证建议：** int8 量化后首次部署建议对比 bf16（在大显存机器上）的 rollout 成功率。简单任务（放方块入杯）通常无感；若复杂任务精度下降明显，再考虑路径 C（4090 上 GPTQ/AWQ 激活感知量化）。

---

## 代码架构说明

### 文件结构

```
src/lerobot/
├── motors/nexarm/                          # 串口驱动层
│   ├── __init__.py                         # 导出 NexArmMotorsBus
│   └── nexarm.py                           # CommProtocol 帧构建/解析、
│                                           #   位置读写、力矩控制、桥接模式
├── robots/nexarm_follower/                 # 从臂机器人
│   ├── __init__.py                         # 导出 NexArmFollower, NexArmFollowerConfig
│   ├── config_nexarm_follower.py           # RobotConfig 子类（端口、摄像头、波特率）
│   └── nexarm_follower.py                  # Robot 子类（连接、观测、动作执行）
└── teleoperators/nexarm_leader/            # 主臂遥操作
    ├── __init__.py                         # 导出 NexArmLeader, NexArmLeaderConfig
    ├── config_nexarm_leader.py             # TeleoperatorConfig 子类（端口、波特率）
    └── nexarm_leader.py                    # Teleoperator 子类（读取动作、主从映射）
├── utils/
│   ├── status_lamp.py                      # 录制状态灯控制器（ESP32 + NeoPixel）
│   └── find_lamp_port.py                   # 热插拔检测灯的串口
```

### 修改的已有文件

| 文件 | 修改内容 |
|------|----------|
| `src/lerobot/robots/utils.py` | 在 `make_robot_from_config()` 中添加 `nexarm_follower` 分支 |
| `src/lerobot/teleoperators/utils.py` | 在 `make_teleoperator_from_config()` 中添加 `nexarm_leader` 分支 |
| `src/lerobot/scripts/lerobot_rollout.py` | 添加 `nexarm_follower` 和 `nexarm_leader` import |
| `src/lerobot/scripts/lerobot_record.py` | 添加 `nexarm_follower` 和 `nexarm_leader` import |
| `src/lerobot/scripts/lerobot_teleoperate.py` | 添加 `nexarm_follower` 和 `nexarm_leader` import |
| `src/lerobot/scripts/lerobot_replay.py` | 添加 `nexarm_follower` import |
| `src/lerobot/scripts/lerobot_calibrate.py` | 添加 `nexarm_follower` 和 `nexarm_leader` import |
| `src/lerobot/scripts/lerobot_find_joint_limits.py` | 添加 `nexarm_follower` 和 `nexarm_leader` import |
| `pyproject.toml` | 添加 `lerobot-find-lamp-port` 入口点 |
| `src/lerobot/utils/status_lamp.py` | **新增** — 录制状态灯控制器（ESP32 + NeoPixel 环形灯） |
| `src/lerobot/utils/find_lamp_port.py` | **新增** — 热插拔检测灯的串口 |
| `src/lerobot/scripts/lerobot_record.py` | 添加 `--lamp_port` 参数、`on_progress` 回调、灯初始化与清理 |

### 数据流向

```
遥操作模式:
  主臂 → read_positions() → map_leader_to_follower() → write_positions() → 从臂

策略推理模式:
  摄像头 + 从臂关节位置 → ACT 模型推理 → 预测动作 → write_positions() → 从臂
```

---

## PR 提交指南

### 步骤 1：Fork 并克隆

```bash
# Fork huggingface/lerobot 到你自己的 GitHub 账号
git clone https://github.com/你的用户名/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
git fetch upstream
git checkout -b add-nexarm-support upstream/main
```

### 步骤 2：复制 NexArm 文件

将本仓库中的 NexArm 相关文件复制到你 fork 的仓库中：

```bash
# 从本仓库复制
cp -r src/lerobot/motors/nexarm        <你的lerobot>/src/lerobot/motors/
cp -r src/lerobot/robots/nexarm_follower   <你的lerobot>/src/lerobot/robots/
cp -r src/lerobot/teleoperators/nexarm_leader  <你的lerobot>/src/lerobot/teleoperators/
cp -r examples/nexarm_act              <你的lerobot>/examples/
cp -r examples/nexarm_SmolVLA          <你的lerobot>/examples/
```

### 步骤 3：修改已有文件

参考本仓库中已修改的文件，对你 fork 的仓库做同样的修改：

1. `src/lerobot/robots/utils.py` — 添加 `nexarm_follower` 分支
2. `src/lerobot/teleoperators/utils.py` — 添加 `nexarm_leader` 分支
3. `src/lerobot/scripts/lerobot_rollout.py` — 添加 `nexarm_follower` 和 `nexarm_leader` import
4. `src/lerobot/scripts/lerobot_record.py` — 添加 `nexarm_follower` 和 `nexarm_leader` import
5. `src/lerobot/scripts/lerobot_teleoperate.py` — 添加 `nexarm_follower` 和 `nexarm_leader` import
6. `src/lerobot/scripts/lerobot_replay.py` — 添加 `nexarm_follower` import
7. `src/lerobot/scripts/lerobot_calibrate.py` — 添加 `nexarm_follower` 和 `nexarm_leader` import
8. `src/lerobot/scripts/lerobot_find_joint_limits.py` — 添加 `nexarm_follower` 和 `nexarm_leader` import
9. `pyproject.toml` — 添加 `nexarm` 依赖项

### 步骤 4：安装并测试

```bash
pip install -e ".[core_scripts,dev]"

# 代码风格检查
pre-commit install
pre-commit run --all-files

# 运行测试
pytest tests/ -v
```

### 步骤 5：提交 PR

```bash
git add -A
git commit -m "Add NexArm 6-DOF robot arm support (follower + leader)"
git push origin add-nexarm-support
```

在 GitHub 上创建 Pull Request，标题和描述建议：

- **标题：** `Add NexArm 6-DOF robot arm support`
- **描述：** 包含硬件简介、测试结果截图（遥操作、训练曲线、推理视频链接）

---

## 常见问题排查

### 串口权限不足 (Linux)

```bash
sudo usermod -a -G dialout $USER
# 需要注销并重新登录
```

### 找不到摄像头

- 运行 `lerobot_find_cameras` 扫描可用索引
- Windows 上关闭其他占用摄像头的程序（OBS、浏览器等）
- 重新插拔 USB 后摄像头索引可能会变

### 遥操作时从臂不动

1. 检查从臂的 COM 端口是否正确
2. 确认从臂 ESP32 固件支持 CMD 68（LeRobot 桥接模式）
3. 尝试重新给从臂上电

### 采集数据时主臂报 TimeoutError

```
TimeoutError: No position reply from NexArm
```

**原因：** 主臂 ESP32 固件在 LeRobot 模式下仍会每秒通过 Serial 打印调试信息（`[WARN]`、`[POS]`），这些文本会污染协议帧，导致 PC 端解析超时。

**解决方案（已内置）：** 驱动程序 `nexarm.py` 的 `read_positions()` 已内置 3 次重试机制，单次超时后会自动重发 CMD 96。绝大多数情况下重试即可恢复。

**彻底修复（固件侧）：** 在主臂固件 `Nex_Arm.ino` 的 `loop()` 中，将 LeRobot 模式下的 `Serial.printf` 调试输出注释掉或改用 `Serial2`：

```cpp
// 在 lerobotMode == true 的分支中，删除或注释以下行：
// Serial.printf("[WARN] ID%d read fail...");
// Serial.printf("[POS] %d %d %d %d %d %d\n", ...);
```

修改后重新烧录固件，即可完全消除此问题。

### 推理时出现状态异常告警

```
WARNING: State anomaly detected (frame 1): [shoulder_pan=50, elbow_flex=4095].
Using last valid state for recovery. Safe range: [100, 3995]

ERROR: CRITICAL: 5 consecutive anomalous frames detected!
```

**原因：** 串口通信丢包，舵机返回边界值（0 或 4095），已超出安全范围 `[100, 3995]`。

**单帧异常：** 系统自动用上一帧合法状态替代，模型输入正常，无需干预。

**连续异常：** 触发 ERROR 日志，需检查：
1. 串口线缆是否松动
2. USB 端口是否过载（多个 USB 设备共享总线）
3. 舵机是否有硬件故障

**调整阈值：** 在配置文件中修改：

```yaml
robot:
  state_safety_margin: 150      # 更严格的安全范围 [150, 3945]
  max_consecutive_anomalies: 3  # 更早触发告警
```

**禁用检测：** `state_safety_margin: 0`（不推荐）

### 训练 loss 不下降

- 确保有足够的示范 episode（建议 50+ 个）
- 检查数据集中的摄像头画面是否正常（非黑屏/模糊）
- 尝试提高学习率（如 `--policy.optimizer_lr=1e-4`）

### Pi0 在 5060 上推理 OOM / import lerobot 报 NameError

- **OOM（CUDA out of memory）**：5060 8GB 无法直接加载 8.3GB 的 Pi0 权重，必须加 `--policy.quantization=int8wo`。详见 [Pi0 int8 量化部署](#pi0-int8-量化部署rtx-5060-8gb2025-08-新增)。
- **import 时 `NameError: name 'logger' is not defined`**：diffusers 0.35.2 + torchao 0.18 的已知 bug，需打补丁（把 `logger` 定义提前）。见量化部署章节「环境搭建关键点」。
- **torchao 警告 `_C_cutlass_90a`/`_C_mxfp8` 加载失败**：可忽略，不影响 int8 量化核心。
- **`quantize_` 报 "Passing a generic Callable is no longer recommended"**：torchao 0.18 废弃了 filter_fn 形式，代码已改为逐模块量化。若自行修改请用 `quantize_(linear, Int8WeightOnlyConfig())`。

### 推理时机械臂犹犹豫豫/动作幅度很小

这通常是模型"坍缩到均值"——输出的动作接近训练数据的平均值，没有学到完整的时序轨迹。排查和调整方法：

**1. 检查训练数据质量**

```bash
# 查看数据集统计，确认 observation.state 和 action 的值范围接近
python -c "
from safetensors.torch import load_file
stats = load_file('outputs/train/nexarm_act/checkpoints/last/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors')
print('state mean:', stats['observation.state.mean'].tolist())
print('action mean:', stats['action.mean'].tolist())
# 两者应该很接近（差距 < 100），否则说明采集时位置反馈有问题
"
```

**2. 增大 batch_size**

batch_size 越大，VAE 的 latent space 学得越稳定，模型越不容易坍缩。

```bash
--batch_size=32    # 默认值，显存不足可用 16
--batch_size=64    # 如果显存够（24GB+），效果更好
```

**3. 调整 kl_weight**

kl_weight 控制 VAE 正则化强度。太大会让模型输出过于保守（接近均值），太小会让动作不连贯。

```bash
--policy.kl_weight=10.0   # 默认值
--policy.kl_weight=5.0    # 如果动作太保守，尝试降低
--policy.kl_weight=1.0    # 进一步降低，让模型更大胆
```

**4. 提高学习率**

默认 1e-5 比较保守，可以适当提高：

```bash
--policy.optimizer_lr=5e-5     # 适度提高
--policy.optimizer_lr=1e-4     # 更激进，注意观察 loss 是否震荡
```

**5. 增加训练步数**

如果 loss 还在下降但还没收敛：

```bash
--steps=200000    # 从 100k 增加到 200k
```

**6. 增加训练数据**

50 个 episode 是下限。如果任务复杂（多步骤、多物体），建议：

```bash
--dataset.num_episodes=100    # 或更多
```

**7. 确保采集数据动作一致**

- 每个 episode 都应包含完整的任务动作（伸手→抓取→抬起→放置）
- 起始位置尽量一致
- 避免 episode 中有大量空闲/等待时间
- 动作速度保持均匀，不要忽快忽慢

### CPU 推理帧率低于预期

ACT 策略使用 action chunking（chunk_size=100），正常情况下 CPU 也能达到 20-30 Hz。如果帧率异常低：

- 确认 `inference.yaml` 中设置了 `device: cpu`
- 检查是否有其他程序占用 CPU（如后台编码、杀毒软件）
- 双摄像头采集本身需要 ~45ms，这是正常开销

### 串口通信校验和错误

主臂固件存在一个已知的校验和 bug（`tx_packet_complete` 使用了 `rx_packet.elements.length` 而非 `tx_packet.elements.length`）。驱动程序同时接受正确和错误的校验和以保持兼容性。
