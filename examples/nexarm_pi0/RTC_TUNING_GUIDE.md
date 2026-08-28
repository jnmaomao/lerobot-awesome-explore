# Pi0 RTC 推理调参指南（NexArm / RTX 5060 8GB + int8）

> 适用：`examples/nexarm_pi0/inference_base.yaml` + `--inference.type=rtc`
> 背景：Pi0 (flow) 在 5060 8GB 上用 torchao int8 weight-only 量化部署，单 chunk ~0.30–0.40s。
> 状态：**latency_tracker 冷启动尖峰 bug 已修复**（2026-08-28），`delay` 现反映稳态 p95。

本指南解释 RTC（Real-Time Chunking）推理的三个核心时间参数，以及为什么 **`inference_delay < execution_horizon`** 是流畅运行的硬性条件，并给出针对本机的调参决策树。

---

## 0. TL;DR — 直接抄命令

经实测验证流畅的推荐配置（10 步去噪，动作清晰自信）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m lerobot.scripts.lerobot_rollout \
  --config_path=examples/nexarm_pi0/inference_base.yaml \
  --policy.path=outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model \
  --policy.quantization=int8wo --policy.gradient_checkpointing=false --policy.use_amp=false \
  --use_torch_compile=false \
  --inference.type=rtc \
  --policy.num_inference_steps=10 \
  --interpolation_multiplier=3 \
  --inference.rtc.execution_horizon=18
```

核心心法：**让 `execution_horizon - inference_delay ≥ 6`**（blend 宽度）。
10 步去噪 `delay≈12` → `horizon=18` → blend=6。详见第 5 节速查表。

---

## 1. 三个时间参数的含义

先建立时间轴。控制环 30Hz，所以 **一个 step = 1/30 ≈ 0.0333s**。
Pi0 一次生成 `chunk_size=50` 步动作，覆盖 `50 × 0.0333 ≈ 1.67s` 的未来。

```
chunk 内步序:  0  1  2  3  4  5  6  7  8  9 10 11 12 ... 49
              ←-- 已经/正在执行 --→ ←-- 还没执行的未来 --→
```

### `inference_delay`（推理延迟，单位 step）

后台线程算一个新 chunk 要花 `latency` 秒。这段时间主线程没闲着，一直在从**旧** chunk 的队列里 pop 动作执行。等新 chunk 算好时，机器人已经往前走了 `ceil(latency / 0.0333)` 步。这个数就是 `inference_delay`——「我算这一下，世界已经前进了几步」。

- 实测：单 chunk ~0.30s（4 步去噪）→ `inference_delay = ceil(0.30 / 0.0333) = 9`
- 单 chunk ~0.37s（10 步去噪）→ `inference_delay = 12`

> **修复后**：决定 blend 区的 `delay` 用 `latency_tracker.p95()`（100 样本滑窗，见第 6 节），
> 与日志里 `Indexes diff ... real_delay=N` 的 `N` **一致**。修复前两者不同——日志里的 `real_delay`
> 是队列跳过值，`delay` 是被冷启动尖峰毒死的全程 max，这才是旧版卡顿的根因。

### `execution_horizon`（执行/混合窗口，单位 step，默认 10）

RTC 约定：新 chunk 的前 `execution_horizon` 步要和旧 chunk 的对应段「对齐」，超过这个范围才允许新 chunk 自由发挥。这是「信任旧计划多远」的距离。

### blend zone（混合区）= `[inference_delay, execution_horizon)`

宽度 = `execution_horizon - inference_delay`。这是新旧计划**平滑过渡**的唯一区域。

---

## 2. RTC 重锚定到底在干什么

RTC 的核心：每来一个新 chunk，不要硬替换，而是用一组**权重**把新 chunk 的前段拉向旧 chunk 没执行完的尾巴（leftover），让两段动作无缝衔接。

权重数组（长度 50，见 `src/lerobot/policies/rtc/modeling_rtc.py:251-298`）：

| 步序区间 | 权重 | 含义 |
|---|---|---|
| `[0, inference_delay)` | **1.0** | 硬复制旧 leftover。这些步「在推理期间已被执行或即将执行」，强制新 chunk 与旧计划一致，保证不回跳 |
| `[inference_delay, execution_horizon)` | **1.0 → 0.0 线性衰减** | 混合区：越往后越相信新计划，平滑过渡 |
| `[execution_horizon, 50)` | **0.0** | 完全自由：新 chunk 按模型自己的预测走 |

直观图（`execution_horizon=18`, `delay=12` 为例，blend=6）：

```
权重 1.0 ─┐
          │────────────┐  ← blend 斜坡（6 步）
          │            ╲
权重 0.0 ─┴──────────────╲──────────────────
步序:    0 ──── 12 ──── 18 ─────────────── 50
         ←硬复制旧计划→ ←混合→  ←新计划自由→
```

作用机制（`modeling_rtc.py:217`）：`err = (leftover - new_plan) * weights`，把这个误差加回新计划。
权重=1 → 新计划被拉到等于 leftover（零误差）；权重=0 → 不拉（自由）。
**blend 斜坡的存在让速度连续**，没有跳变。

---

## 3. 为什么 `inference_delay < execution_horizon` 是硬性条件

blend 区宽度 = `execution_horizon - inference_delay`。只有它 > 0，斜坡才存在。

### 反例：blend=0（悬崖）

当 `inference_delay ≥ execution_horizon` 时，代码里 `start = min(delay, end) = end`
（`modeling_rtc.py:252`），blend 宽度归零，`_linweights` 返回空张量（`:277`）。
权重退化成悬崖：

```
权重 1.0 ─┐
          │──────────┐  ← 悬崖！无斜坡
          │          │
权重 0.0 ─┴──────────┴──────────────────
步序:    0 ──────── H ──────────────── 50
         ←硬复制旧计划→ ←新计划突然自由→
```

第 H 步处**权重从 1.0 直接掉到 0.0**——新旧计划硬切换。
而 `_replace_actions_queue` 跳过 `real_delay` 步后恢复执行，**正好落在悬崖之后的自由区**
→ 每 ~0.6s 一次速度突变 = 感受到的「电机频繁起停卡顿」。

> 修复前这个坑极易触发：`delay` 被冷启动尖峰永久卡在 20，而默认 `horizon=10`，
> blend 一直是 0。修复后 `delay` 反映稳态 p95，才让调参真正生效。

### 弱例：blend=3（陡峭）

10 步去噪 `delay=12`, `horizon=15` → blend=3。斜坡存在但太陡，自信动作在 3 步内硬切进来
→ 体感「动作激进、加速突然」。**这正是 `horizon=15` 不够、需抬到 18 的原因。**

### 良例：blend=6（推荐）

10 步去噪 `delay=12`, `horizon=18` → blend=6。6 步斜坡给自信动作一个软着陆区 → 体感流畅。

---

## 4. 怎么调参（核心公式）

记住一句话：**让 `execution_horizon - inference_delay ≥ 6`。**

blend 区越宽，过渡越柔；但 `execution_horizon` 也不能无限大——它代表「用旧计划约束新计划多远」，
太大意味着用过时的旧动作约束太久，机器人会迟钝、跟不上场景变化。经验上限 ~25（约 chunk 一半）。

### 调参决策树

1. **先看诊断日志的 `delay(p95)`**（每 ~1s 打一条 `RTC diag: ... delay(p95)=N horizon=H blend=B`，
   见第 7 节）。这是真正生效的 `delay`，不是日志里的 `real_delay`（修复前两者不同，修复后一致）。
   忽略首帧冷启动的大值（warmup 已丢弃）。
2. **目标**：`execution_horizon ≥ delay(p95) + 6`。
3. **两条路**：
   - **降 `delay`**（让模型更快）：`--policy.num_inference_steps` 从 10→4→2。
     但瓶颈常在循环开销（观测构建 + preprocessor + 去噪 + postprocessor），不是单纯去噪步数，
     这条收益有上限；且 4 步去噪动作偏保守带噪、10 步更自信——是质量权衡，不只看 delay。
   - **升 `execution_horizon`**（给 blend 更多空间）：通常是更有效的杠杆。
     `delay=12` → 设 `execution_horizon=18`，blend = 6。

### 去噪步数怎么选

| 去噪步数 | 动作特性 | 适用 |
|---|---|---|
| 10 | 充分去噪，轨迹清晰自信，但需更宽 blend 软着陆 | 追求动作质量（**推荐**） |
| 4 | 欠采样，动作偏保守带噪 | 追求响应快、不易突变 |
| 2 | 更快但质量下降明显 | 仅极简任务 |

> 「10 步比 4 步激进」不是 bug：10 步动作更自信，在窄 blend 下显得突然；
> 抬 horizon 给足斜坡后即变流畅。不要用降去噪步数来掩盖 blend 不足。

---

## 5. 参数速查表（fps=30, chunk_size=50，修复后实测）

| `num_inference_steps` | 实测 `delay(p95)` | 推荐 `execution_horizon` | blend 宽度 | 动作特性 |
|---|---|---|---|---|
| 10（**推荐**） | 12 | **18** | 6 | 清晰自信 |
| 4 | 9 | 15~17 | 6~8 | 保守带噪 |
| 2 | ~7 | 13 | 6 | 快但糙 |

- 想要**动作质量**（轨迹清晰、自信）→ 10 步 + horizon=18
- 想要**响应快、动作保守**（不易突变）→ 4 步 + horizon=15~17
- **不建议** horizon > 25：用旧计划约束太久，机器人迟钝、跟不上场景

---

## 6. latency_tracker 冷启动尖峰 bug（已修复）

### 修复前的问题

`delay` 来自 `LatencyTracker.max()`——**单调不减的全程最大值**
（`src/lerobot/policies/rtc/latency_tracker.py`）。原代码只有 `torch.compile` warmup 才 reset，
而 8GB 卡上 `compile` 是关的，于是 warmup 根本不执行，tracker 从不 reset。

后果：第一帧冷启动（cudnn autotuning，实测 ~0.67s vs 稳态 ~0.37s）把 `max` 永久钉在 0.67s
→ `delay = ceil(0.67/0.0333) = 20` 全程不变 → 默认 `horizon=10` 下 blend 恒为 0。
**日志里 `real_delay=9/12` 是队列跳过值，与决定 blend 的 `delay=20` 是两个数**——
对着 `real_delay` 调 horizon 调错了对象，怎么调都塌缩。

偶发卡顿（USB 抖、GC、相机卡顿）也会把 `max` 抬高且永不下降，表现是「跑着跑着又变卡」。

### 修复（`src/lerobot/rollout/inference/rtc.py`，2026-08-28）

1. `delay` 改用 `latency_tracker.p95()`（100 样本滑窗）代替全程 `max()`，忽略孤立尖峰。
2. 新增 `_RTC_LATENCY_WARMUP=3`：前 3 帧冷启动推理**不进** p95 窗口（warmup 期间持续 reset tracker），
   且**独立于 torch.compile**——compile 关闭时也会执行。
3. `delay` 现与日志 `real_delay` 一致，调参所见即所得。

备份：`rtc.py.bak_20260828_160415`。

### 仍需注意

p95 滑窗抗孤立尖峰，但**持续**慢速（如散热降频、显存碎片）仍会逐步抬高 p95。
若 `RTC diag` 行的 `delay(p95)` 随时间明显爬升（>2），说明有持续性减速，检查散热/后台进程，
或临时调大 `--inference.rtc.execution_horizon` 留余量。

---

## 7. 诊断日志（修复后新增）

修复后每 ~1s（30 帧）打一条 INFO：

```
RTC diag: latency=0.37s delay(p95)=12 horizon=15 blend=3 queue=38 real_delay=12
```

| 字段 | 含义 | 健康值 |
|---|---|---|
| `latency` | 本次推理墙钟 | 4 步 ~0.30s / 10 步 ~0.37s |
| `delay(p95)` | 决定 blend 区起点的延迟（滑窗 p95） | 应与 `real_delay` 一致 |
| `horizon` | `execution_horizon` 配置值 | — |
| `blend` | `horizon - delay(p95)`，斜坡宽度 | **≥ 6** |
| `queue` | 动作队列长度 | 充足（>阈值 30 不停算即说明跟得上） |
| `real_delay` | 队列跳过步数（本次墙钟） | 与 `delay(p95)` 一致 |

另有 `Latency warmup complete (3 cold-start inferences discarded)` 确认冷启动已丢弃。
DEBUG 级别有逐帧详情：`LOGLEVEL=DEBUG` 前缀可看。

> `Indexes diff is not equal to real delay` warning 仍会出现（off-by-1，`ceil` 保守取整导致），
> 属良性，不影响控制。

---

## 8. 为什么 sync 模式反而显得顺（对比基准）

sync 走 Pi0 内部 `_action_queue`：**整段 50 步一口气平滑播完**再重新推理，中间没有任何
重锚定/队列替换（`src/lerobot/policies/pi0/modeling_pi0.py:1137-1142`）。
代价是每 1.67s 卡一次 ~0.37s（日志里的 `Record loop running slower (2.8 Hz)` 警告），
但两段卡顿之间是一条完整平滑轨迹。

RTC 每 ~0.6s 就用新 chunk 替换一次队列；当 `inference_delay ≥ execution_horizon`（blend=0）时，
替换是硬跳变 → 持续微抖，体感比 sync 的「长平滑 + 偶尔卡」更差。
**只有把 blend 区调出足够余量（≥6），RTC 的「永不阻塞主线程」优势才会显现，且整体优于 sync。**
