# fork 补齐 v0.6.1 同步状态与交接清单

> 生成日期：2026-11-17
> 基线事实：本仓库从上游 main 基点 `95211b98`（PR #4139 之后）切出，落后上游 v0.6.1 tag **96 个提交**（75 个涉及 src/lerobot）。pyproject 中版本号 `0.6.1` 不等于工作树等于 v0.6.1 tag。
> 上游 tag 已在本地 git refs 中（`v0.6.1`、`v0.7.0`），服务器上若没有请先：`git remote add upstream https://github.com/huggingface/lerobot.git && git fetch upstream tag v0.6.1 v0.7.0`

---

## 一、已完成（已提交：`e0a3d3db`，8 文件 +228/−86）

| 文件 | 内容 | 冲突处理 |
|---|---|---|
| pyproject.toml | draccus≥0.11.6、hub≥1.6、datasets≥4.8、pytest<10、setuptools<82、mypy overrides | 自动合并 |
| policies/pi0/modeling_pi0.py | `supports_rtc()`（#4228 配套）；本地 int8 量化保留 | 自动合并 |
| rollout/inference/rtc.py | `supports_rtc_inference()` 能力探测；本地冷启动修复（`_RTC_LATENCY_WARMUP`、p95）保留 | 插入点错开，自动合并 |
| rollout/configs.py | Episodic/DAgger `smooth_handover` 开关；本地 `move_to_start_position` 保留 | 自动合并 |
| utils/utils.py | `LeRobotFormatter` 重写 + say() TTS 容错；本地 log_say ANSI 着色保留 | 自动合并 |
| cameras/opencv/camera_opencv.py | #4187：connect() try/except + `_cleanup_resources` + `_reset_connection_settings`；本地 `CAP_PROP_BUFFERSIZE=1` 手工移入新 try 块；本地 FPS 容错保留 | 手工调和（1 处） |
| scripts/lerobot_rollout.py | 上游 lekiwi 与本地 nexarm_follower import 并存 | 手工调和（插入-插入） |
| rollout/context.py | 手工三方合并（`git merge-file`）：上游 RTC 门控移入本地 api/else 分支内；torch.compile 采用上游 `_wrap_predict_action_chunk_with_torch_compile()` + 回退传播，条件保留本地 `not is_api and policy is not None`；本地 api 跳过加载/`_move_to_position`/`_verify_position`/传感器预热全保留 | 手工深合并（2 处冲突） |

**静态校验已通过**：全部文件 `py_compile` ✓、13 项内容标记在场 ✓、0 冲突残留 ✓。

## 二、未完成（追平 tag 的剩余纠缠点）

> **2026-11-17 更新：以下各项已全部完成**，整批 merge 提交为 `99a75d5f`（分支 `sync/v0.6.1-full`）。

1. **`policies/pi0_fast/modeling_pi0_fast.py`** —— 已决定**保持本地补丁版本**（自回归解码提前终止 + tokenizer fallback）。整批 merge 时对该文件取 `ours`。注意：上游 #4275 是同主题优化且保护 RNG 序列（本地版不分 temperature、会扰动随机采样路径），未来可重评。 ✅ 已按 ours 保留（对 v0.6.1 差异 +46/−23）
2. **`processor/tokenizer_processor.py` + `types.py → lerobot_types.py` 模块改名** —— 上游该文件唯一改动是 import 改名，而模块改名本身也在缺失提交里，**必须随整批 merge 做**（单独应用会断 import）。整批 merge 时需同步处理本地 `_fast_action_tokenizer_compat.py` 的引用。 ✅ 改名已随 merge 应用；compat shim 不引用 `lerobot.types`（核实无需改动）；fork 侧唯一断链点 `nexarm_follower.py:34` 已改为 `lerobot.lerobot_types` 并入 merge 提交
3. **`rollout/inference/factory.py` + `__init__.py`** —— 本地自研推理 API（msgpack 边端推理）注册，上游无对应；整批 merge 时保留本地版并把上游新条目（如有）并入注册表。 ✅ 已核实：本地 api 注册完整保留，上游 base→tag 无新增引擎条目
4. **其余 ~130 个无本地改动的文件** —— 整批 merge 时 git 会自动处理。 ✅ 双侧重叠面实际为 10 文件（4 冲突手工解决 + 6 自动合并逐一核验），`strategies/core.py` 上游 base→tag 未触碰、无合并输入

**实际操作记录**（已完成）：

```bash
git checkout -b sync/v0.6.1-full
git merge v0.6.1        # 冲突 4 文件 5 块，全按上述规则解决后提交 99a75d5f
# pi0_fast 冲突取 ours：git checkout --ours src/lerobot/policies/pi0_fast/modeling_pi0_fast.py
```

## 三、待测试（真实运行环境执行，一项都不能跳）

> **2026-11-17 更新：T1–T5 已在 Linux 服务器（conda env `nexarm_pico`，draccus 升级 0.10.0→0.11.6）执行通过；T6 依赖真机硬件（串口/双摄像头/PICO 均未连接），已完成其可自动化子项，真机流程待接上硬件后执行。**

按依赖顺序：

- [x] **T1. draccus 兼容性**：✅ **PASS**。`pip install -e .`（draccus 0.11.6、pytest<10.0.0 已装），`lerobot-record --config_path examples/nexarm_pi0fast/record.yaml` 冒烟：完整配置树（nexarm_follower 嵌套 cameras/teleop/dataset）解析成功并推进到 `teleop.connect()`，因 `/dev/ttyUSB1` 不存在干净报错（预期终点）。`pip check` 无新增冲突（仅存量 gym-xarm/mujoco 无关冲突）。
- [x] **T2. camera 合并语义**：✅ **PASS 16/16**（真实 ASUS 摄像头 /dev/video0）。正常 connect ✓；本地 `CAP_PROP_BUFFERSIZE=1` 在上游 try 块内生效 ✓；本地"采纳原生帧率"（请求 10 → 采纳 15）✓；disconnect 释放+停线程 ✓（上游契约：reset 属 connect 失败路径，disconnect 不回滚）；失败 connect（不可达 fps=120）→ `_cleanup_resources` 释放 + `_reset_connection_settings` 把 fps 回滚到 `config.fps` ✓；无设备/线程泄漏、重连成功 ✓。
- [x] **T3. context.py 合并语义（api 路径）**：✅ **PASS 11/11**（stub 机器人 monkeypatch，`_load_pretrained_policy` 注入"被调用即失败"探针）。不触发本地 torch.compile ✓；`use_torch_compile=True` 原样透传 RuntimeContext（无 False 回写）+ `policy_config.compile_model` 戳记给远端 ✓；策略权重不本地加载（policy=None）✓；pre/post processors 跳过 ✓；`ApiInferenceEngine` 持有机器人侧 action 键序 ✓。
- [x] **T4. context.py 合并语义（本地路径）**：✅ **PASS 7/7**。`use_torch_compile=True` + backend=eager：`_wrap_predict_action_chunk_with_torch_compile()` 真实接线成功（predict_action_chunk 被替换）、`cfg.use_torch_compile=True` 向下游传播 ✓；backend 不可达（torch 2.11 在 wrap 时即抛 `InvalidBackend`，走真实 except 分支）：`cfg.use_torch_compile` 回退 False 并经 RuntimeContext 传播、策略方法未被替换 ✓。
- [x] **T5. RTC 门控**：✅ **PASS 15/15**。不支持 RTC 的 policy 带 RTC 配置 → ValueError 且文案来自 #4228 新门控（含 `--inference.type=sync` 提示）✓；`supports_rtc()=True` 但 `predict_action_chunk` 缺 RTC 形参 → 签名 bind 检查同样拦截 ✓；RTC-capable policy → 构建成功、`rtc_config` 挂载、`init_rtc_processor` 调用一次 ✓；本地冷启动修复未回归：`_RTC_LATENCY_WARMUP=3` 在场、空 tracker p95()=0.0（delay=0 安全路径）、p95 滑窗忽略孤立尖峰（max 会钉死）、100 样本窗口滚动淘汰 ✓。
- [x] **T6. 全量回归**：✅ **PASS**（2026-11-17 16:02 真机执行）。`lerobot-record --config_path=examples/nexarm_pi0fast/record.yaml`：draccus 0.11.6 解析完整配置树 ✓；leader(/dev/ttyUSB1)+follower(/dev/ttyUSB0) 双总线连接、warm-up 一次通过 6 关节全有效 ✓；front(video6)+wrist(video4) 双相机取流（wrist 原生 60fps 容错告警符合本地设计）✓；episode 0 录制 35s、1046 帧样本、双相机 x264 编码落盘（标准 LeRobotDataset 布局：data/chunk-000/*.parquet + meta + videos）✓；Esc 干净断开退出 ✓。合并后 record 全链路（解析→主从控制→取流→编码→落盘）验证通过。
  - 已完成子项：log_say ANSI 三色着色 ✓、say() TTS 容错（FileNotFoundError→warning 不抛出）✓、`LeRobotFormatter`（init_logging 嵌套类）格式化 ✓、move_to_start_position（context.py:416-443）与传感器预热（context.py:401-406）结构在位 ✓；`tests/test_rollout.py` 27/27 ✓、`tests/cameras/test_opencv.py + tests/utils` 75 passed/1 skipped ✓。

## 四、背景与关联

- 完整架构决策与 0.7.0 评估见 [lerobot-bs-console-design.md](./lerobot-bs-console-design.md)（§6.1 升级冲突面、§4.4 WebSocket 传输基线）。
- 结论回顾：**B/S 控制台基于 0.6.1 基线开发，零硬缺口**；0.7.0 升级暂缓（硬件层重建 + async_inference 删除 + policy API 变更三重命中 fork 定制），触发条件见设计文档 §6.1。
- 关键数字（若需向上游汇报或评估再同步）：本地真实修改 25 个上游文件；与 0.6.1→0.7.0 变更重叠 19 个；0.7.0 升级必重做面 = rollout/inference 重定位（3 文件）+ pi0 系补丁移植。
