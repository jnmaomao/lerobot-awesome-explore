# LeRobot 推理 API 服务部署指南(Pi0 / Pi0-FAST)

把模型推理的重计算部分(Pi0 flow 的 10 步 flow-matching 前向 / Pi0-FAST 的自回归 token 解码,均过 PaliGemma 3B)从本地机器人主机卸载到一台更强大的推理服务器(如 4090/3090)。本地 LeRobot 继续负责机械臂连接、图像/关节数据采集、控制循环、安全限位与首帧阻尼;通过 HTTP/msgpack 调用本服务获取动作 chunk。

服务器按 checkpoint `config.json` 的 `type` 字段自动 dispatch 加载哪个 policy 类(当前支持 `pi0` / `pi0_fast`),加载路径分别照搬已验证的部署脚本。新增 chunk-based policy 只需在 `server/policy_server.py` 的 `_POLICY_REGISTRY` 追加一行。

本指南说明服务器与客户端的部署、配置、协议与验证。

---

## 目录

- [架构概览](#架构概览)
- [部署拓扑](#部署拓扑)
- [服务器部署](#服务器部署)
- [客户端部署](#客户端部署)
- [API 协议](#api-协议)
- [配置详解](#配置详解)
- [验证](#验证)
- [性能与调优](#性能与调优)
- [故障排查](#故障排查)

---

## 架构概览

Pi0 是 **chunk-based** policy:一次前向生成 50 步动作(`chunk_size = n_action_steps = 50`),内部用 `deque` 缓存,队列空才跑完整前向。

本方案利用这一特性,把网络 RPC 频率压到极低:

```
本地机器人主机 (30fps 控制环)              推理服务器 (4090)
┌─────────────────────────────┐         ┌──────────────────────────┐
│ robot.get_observation()      │         │  PI0Policy (GPU)          │
│   ↓ 图像 JPEG + state + task │  HTTP   │  preprocessor             │
│   ↓ msgpack (~100KB)         │───────→ │  predict_action_chunk     │
│                              │  RPC    │  postprocessor            │
│   ← chunk (50,6) float32     │←─────── │   (~0.1-0.4s / chunk)     │
│   ↓ deque 缓存逐帧 pop       │         └──────────────────────────┘
│   ↓ ActionInterpolator 插值   │
│   ↓ send_action (安全限位)    │
└─────────────────────────────┘
```

**关键数据**:
- RPC 频率:约 **0.6 次/秒**(@30fps,每 1.67s 一个 chunk)
- 上行:~100KB(JPEG 双目图像)
- 下行:~1.2KB(chunk `(50,6)` float32)
- 延迟掩盖:chunk 提供 1.67s 动作缓冲,RPC 的 ~0.1-0.4s 前向延迟由 `ActionInterpolator` + 队列剩余动作自然吸收

### 数据切分边界

| 阶段 | 归宿 | 说明 |
|---|---|---|
| 采集图像 / 关节 | 本地 | `robot.get_observation()` |
| JPEG 编码图像 | 本地 | uint8 `(480,640,3)` → JPEG bytes |
| **RPC 上行** | 网络 | msgpack: images + state + task |
| JPEG 解码 + `prepare_observation_for_inference` | 服务器 | numpy→tensor、加 batch 维、[0,1] |
| `preprocessor` (tokenize + 状态归一化) | 服务器 | 复用 `PolicyProcessorPipeline` |
| **`predict_action_chunk`** (10 步 flow 前向) | **服务器** | **唯一重活** |
| `postprocessor` (动作反归一化) | 服务器 | 复用 `PolicyProcessorPipeline` |
| **RPC 下行** | 网络 | chunk `(50,6)` float32 |
| chunk 队列 + pop | 本地 | 客户端 `deque(maxlen=50)` |
| 插值 + 安全限位 + 写舵机 | 本地 | `nexarm_follower.send_action` |

> 阶段一原则:**服务器全包**——预处理、模型、后处理都在服务器,客户端零模型依赖、零 stats 依赖。服务器加载路径照搬已验证的 `scripts/test_load_pi0_int8_5060.py`(Pi0 flow,`PI0Policy`),模型代码零修改。

---

## 部署拓扑

```
┌─────────────────────────┐        LAN / 跨机        ┌──────────────────────────┐
│  本地机器人主机           │◄──────────────────────────►│  推理服务器 (4090)         │
│  - nexarm_follower       │   POST /predict (msgpack)   │  - server/policy_server.py│
│  - 双目 OpenCV 相机       │   ← chunk (50,6)            │  - PI0Policy on GPU       │
│  - LeRobot (client)      │                             │  - checkpoint 权重         │
│  - 无需 GPU              │                             │  - cuda                   │
└─────────────────────────┘                             └──────────────────────────┘
```

两台机器通过局域网通信。服务器需有 GPU;客户端无需 GPU。

---

## 服务器部署

### 1. 安装依赖

服务器需要完整的 lerobot 训练依赖(提供 `PI0Policy` / processor / utils),外加 API 服务依赖:

```bash
# 在推理服务器上(如 4090)
cd lerobot-awesome-explore_api_service

# lerobot 本体(含 torch / transformers / torchao 等)
pip install -e ".[core_scripts,training]"

# API 服务依赖
pip install -r server/requirements.txt
# 等价于: pip install fastapi uvicorn[standard] msgpack opencv-python
```

> **torchao 版本**:若使用 `int8wo` 量化路径(小显存机器),需 `torchao==0.18.0` 配 torch 2.11,且 diffusers 0.35.2 有一个 logger 加载顺序 bug 需打补丁——见 `README_NEXARM.md`「Pi0 int8 量化部署」章节与 `scripts/check_patch_diffusers_torchao.py`。4090 大显存走 `quantization=none` 则无需 torchao。

### 2. 准备 checkpoint

将训练产物放到服务器可访问路径(本地路径即可,无需上传 Hub):

```
outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model/
├── config.json                          # 模型配置
├── model.safetensors                    # 权重(8.3GB,bf16+fp32 混合)
├── policy_preprocessor.json             # 预处理器配置
├── policy_preprocessor_step_6_normalizer_processor.safetensors   # 归一化 stats
├── policy_postprocessor.json            # 后处理器配置
└── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

> checkpoint 目录必须完整——服务器需加载模型权重 + pre/post processor + 归一化 stats。

### 3. 启动服务器

```bash
# Pi0 flow — 4090 大显存走原始 bf16(默认 quantization=none)
python server/policy_server.py \
    --ckpt outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model \
    --host 0.0.0.0 --port 8000

# Pi0 flow — 小显存机器(如 5060 8GB)开 int8 weight-only 量化
python server/policy_server.py \
    --ckpt outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model \
    --host 0.0.0.0 --port 8000 \
    --quantization int8wo

# Pi0-FAST — 自回归解码,走 bf16(不支持量化;4090 24GB 无压力,5060 峰值 7.2GB 勉强可跑)
python server/policy_server.py \
    --ckpt outputs/train/nexarm_pi0fast/checkpoints/040000/pretrained_model \
    --host 0.0.0.0 --port 8000
```

> Pi0-FAST 依赖解码提前终止优化(见 `README_NEXARM.md`「Pi0-FAST 自回归解码提前终止优化」章节),否则单 chunk 耗时 ~6.7s(4090)不可用;优化后 4090 ~0.5s、5060 ~0.65s。未优化版本会跑满 256 步解码循环浪费 95% 计算。

启动后日志会显示加载过程(config → 权重 → 量化 → pre/post processor)。int8 量化 252 层约需 60s。就绪后输出:

```
Policy server READY
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. CLI 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ckpt` | (必填) | `pretrained_model` 目录路径(由 config.json 的 `type` 决定加载 pi0 还是 pi0_fast) |
| `--host` | `0.0.0.0` | 监听地址;跨机部署用 `0.0.0.0` |
| `--port` | `8000` | 监听端口 |
| `--quantization` | `none` | `none`=原始 bf16(大显存);`int8wo`=torchao int8 weight-only(小显存)。**仅 pi0(flow)支持;pi0_fast 会忽略并走 bf16** |
| `--num_inference_steps` | (用 config) | 覆盖 flow 去噪步数(降速可设 4,简单任务通常够)。**仅 pi0(flow)生效;pi0_fast 忽略** |
| `--log_level` | `info` | 日志级别 |

### 5. 服务器环境变量(小显存机器推荐)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 抑制显存碎片
export HF_HUB_OFFLINE=1                                    # 强制走本地 HF 缓存
# 需预缓存: google/paligemma-3b-pt-224(PaliGemma tokenizer)
```

### 6. 健康检查

```bash
curl http://<server-ip>:8000/health
# {"ready":true,"chunk_size":50,"n_action_steps":50,"action_dim":6,
#  "num_inference_steps":10,"quantization":"none"}
```

客户端启动时会自动探活(`ApiInferenceEngine.start()`)。

---

## 客户端部署

### 1. 安装依赖

客户端不需要训练依赖,只需核心脚本 + msgpack:

```bash
# 在本地机器人主机上
cd lerobot-awesome-explore_api_service
pip install -e ".[core_scripts]"
pip install msgpack          # 唯一新增依赖(纯 Python,419KB)
```

> `requests` 和 `opencv-python` 已是 lerobot 依赖,无需额外安装。客户端**不需要 GPU**、不需要 torchao。

### 2. 准备 policy 元数据

客户端用 `--policy.path` 仅读取 checkpoint 的 `config.json`(获取 `action_feature_names` / `chunk_size` 等元数据用于 action 重排),**不加载 `model.safetensors` 权重**。两种方式:

- **方式 A(推荐)**:把整个 checkpoint 目录拷到客户端(权重文件不会被读取,但放着无害)。
- **方式 B(最省)**:只拷 `config.json` 到客户端某目录,`--policy.path` 指向它。

### 3. 配置客户端

编辑客户端配置(Pi0 flow 用 `examples/nexarm_pi0/inference_api_client.yaml`,Pi0-FAST 用 `examples/nexarm_pi0fast/inference_api_client.yaml`):

```yaml
inference:
  type: api
  server_url: http://<4090-服务器IP>:8000   # ← 改成你的服务器地址
  timeout: 5.0
  n_action_steps: 50                         # 与 policy config 一致(Pi0-FAST 为 10)

device: cpu          # 客户端无 GPU
task: "Put blue cube in cup"   # 必须与训练时 single_task 一致
```

> **`n_action_steps` 必须与服务器 checkpoint 的 chunk_size 一致**(Pi0 flow=50,Pi0-FAST=10)。不一致会导致动作截断或重复。

`robot` / `cameras` / `safety` / `strategy` / `fps` 段与对应 `inference_base.yaml` 完全相同——本地控制逻辑零改动。

### 4. 启动客户端

```bash
# 先确认服务器已就绪
curl http://<server-ip>:8000/health

# 启动推理(Pi0 flow)
python -m lerobot.scripts.lerobot_rollout \
    --config_path=examples/nexarm_pi0/inference_api_client.yaml \
    --policy.path=outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model

# 启动推理(Pi0-FAST)
python -m lerobot.scripts.lerobot_rollout \
    --config_path=examples/nexarm_pi0fast/inference_api_client.yaml \
    --policy.path=outputs/train/nexarm_pi0fast/checkpoints/040000/pretrained_model
```

客户端启动后会:
1. 读 `config.json` 元数据(不加载权重)
2. 连接机械臂 + 相机
3. 移动到 `safe_rest_position`
4. 构造 `ApiInferenceEngine`,探活服务器
5. 进入 30fps 控制环:队列空时 RPC 拉 chunk,逐帧 pop + 插值 + 发舵机

---

## API 协议

### `POST /predict`

**请求**:
- `Content-Type: application/x-msgpack`
- Body 为 msgpack 编码:

```python
{
    "images": {
        "front": <JPEG bytes>,   # 相机名 → JPEG 编码的图像
        "wrist": <JPEG bytes>,   # 键名需与 policy image_features 对应
    },
    "state": [2082.0, 2370.0, 1664.0, 2505.0, 2054.0, 1690.0],  # 原始舵机值 list
    "task":  "Put blue cube in cup"
}
```

**响应**:
- `Content-Type: application/x-msgpack`
- Body 为 msgpack 编码:

```python
{
    "chunk":  <bytes>,   # float32 原始字节,长度 = 50*6*4 = 1200
    "shape":  [50, 6]    # [n_action_steps, action_dim]
}
```

客户端还原:`np.frombuffer(result["chunk"], dtype=np.float32).reshape(result["shape"])`

### 图像颜色顺序

- OpenCV 相机默认 `color_mode=RGB`(模型在 RGB 上训练)。
- 客户端用 `cv2.imencode` 编码 RGB 数组,服务器用 `cv2.imdecode` 解码——按通道位置无损往返,**无需 BGR/RGB 翻转**。
- JPEG 质量:客户端默认 `jpeg_quality=90`(可在 `ApiInferenceEngine` 构造时调)。

### `GET /health`

```json
{"ready": true, "chunk_size": 50, "n_action_steps": 50, "action_dim": 6,
 "num_inference_steps": 10, "quantization": "none"}
```

### 并发模型

- uvicorn **单 worker**(`workers=1`):模型前向占满 GPU,多 worker 会复制显存导致 OOM。
- 服务器用 `threading.Lock` 串行化 GPU 前向,并发请求排队(不丢请求)。
- 客户端控制环本就是串行单连接,无并发压力。

---

## 配置详解

### 服务器侧关键开关

| 开关 | 何时用 | 效果 |
|---|---|---|
| `--quantization none` | 4090 / 大显存(默认) | 原始 bf16,峰值 ~6-7GB,精度最高 |
| `--quantization int8wo` | 5060 8GB / 小显存 | int8 weight-only,峰值 ~3.4GB,相对误差 ~0.5% |
| `--num_inference_steps 4` | 需更快推理 | 10→4 步去噪,~2.5x 加速,简单任务通常够 |

### 客户端配置字段(`inference.api.*`)

```yaml
inference:
  type: api
  server_url: http://192.168.1.100:8000
  timeout: 5.0            # RPC 超时(秒);弱网络可调大
  n_action_steps: 50       # 必须与 policy config 的 n_action_steps 一致
```

> `n_action_steps` 用于客户端 chunk 队列长度。若与服务器实际 chunk_size 不一致,会出现动作截断或重复——务必对齐。

### 本地控制逻辑(与 `inference_base.yaml` 完全相同)

- `state_safety_margin: 100` — 绝对限位 `[100, 3995]`
- `max_relative_target: 500` — 单 tick 相对位移上限
- `first_frame_gap_threshold_s: 0.2` / `first_frame_min_delta: 50` / `first_frame_warmup_frames: 15` — 首帧阻尼,防止 RPC 首帧延迟导致动作突变
- `move_to_start_position: true` — 推理前缓慢移动到训练起始位

---

## 验证

### 1. 服务器数值正确性(对照本地推理)

用与 `scripts/test_load_pi0_int8_5060.py:87-92` 相同的 dummy 输入(zeros + task),分别走服务器 RPC 和本地 `predict_action_chunk + postprocessor`,比较输出:

```bash
# 服务器侧:启动服务后发 dummy 请求
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/x-msgpack" \
    --data-binary @dummy.msgpack -o resp.msgpack

# 对照脚本:见 scripts/test_load_pi0_int8_5060.py(本地路径)
```

**预期**:服务器与本地输出统计一致(均值相对误差 < 5%)。注意 pi0 flow 推理本身因随机噪声采样具有非确定性——本地两次推理自身差异约 8.9%——服务器-vs-本地差异应落在该范围内。

### 2. 客户端引擎单元行为

`ApiInferenceEngine` 的 chunk 队列逻辑:
- 第 1 次 `get_action`:触发 RPC #1,填满 50 步队列,pop 第 1 步
- 第 2~50 次:不触发 RPC,逐次 pop
- 第 51 次:队列空,触发 RPC #2
- `reset()`:清空队列,下次立即 RPC

### 3. 端到端(真实机械臂)

```bash
# 服务器
python server/policy_server.py --ckpt <ckpt> --host 0.0.0.0

# 客户端
lerobot-rollout --config_path=examples/nexarm_pi0/inference_api_client.yaml \
    --policy.path=<ckpt>
```

**验证清单**:
- [ ] 机械臂从 `safe_rest_position` 起步
- [ ] 首帧阻尼生效(无突变痉挛)
- [ ] 30fps 控制环稳定不丢帧(日志无 "running slower than target FPS" 警告)
- [ ] 动作平滑、符合任务预期
- [ ] 对比同一 checkpoint 本地推理(`inference_base.yaml`)的动作轨迹应高度一致

### 4. 延迟 / 带宽实测

客户端日志(DEBUG 级)会打印每次 RPC:

```
RPC #1: 2 cameras, 98.3KB up, chunk [50, 6], RTT 412ms
```

**预期**:上行 < 200KB,RTT < 150ms(LAN + 4090),控制环不因 RPC 阻塞丢帧。

---

## 性能与调优

### RPC 频率与控制 fps 的关系

```
可持续控制 fps = n_action_steps / 单次 RPC 耗时
```

- n_action_steps = 50,RPC 耗时 0.4s → 可持续 125fps(远超 30fps 目标)
- 即使 RPC 耗时 1.7s,仍可维持 30fps

**结论**:chunk-based 特性使 API 化对网络延迟高度鲁棒,4090 LAN 下毫无压力。

### 降低上行带宽

若网络受限(如 WiFi):
- 降低 JPEG 质量:`ApiInferenceEngine(..., jpeg_quality=75)`(默认 90)
- 降低相机分辨率:配置 `width: 320, height: 240`(注意:需与训练数据分辨率分布一致,谨慎)

### 加速服务器推理

- 降 `--num_inference_steps 4`(10→4,~2.5x,简单任务通常够)
- 4090 大显存用 `--quantization none`(避免量化开销,精度最高)
- 关 `gradient_checkpointing`(服务器默认已关)

### 显存与速度对照(真实数据实测)

**Pi0 flow(chunk_size=50,10 步去噪前向)** — 参考 `test_load_pi0_int8_5060.py`:

| 配置 | 峰值 VRAM | 单 chunk p95 | 可持续 fps |
|---|---|---|---|
| 4090 + `none` (bf16) | ~6-7GB | ~80-150ms | 300+ |
| 5060 8GB + `int8wo` | ~3.4GB | ~400ms | 125 |

**Pi0-FAST(chunk_size=10,自回归解码,需解码提前终止优化)**:

| 配置 | 峰值 VRAM | 单 chunk p95 | 可持续 fps | 10Hz | 30Hz |
|---|---|---|---|---|---|
| 4090 + bf16 | 7.20GB | 527ms | 22.9 Hz | ✅ | ⚠️ 略不足 |
| 5060 8GB + bf16 | 7.20GB | 645ms | 15.5 Hz | ✅ | ❌ |
| (优化前 4090) | 7.20GB | ~6700ms | 1.5 Hz | ❌ | ❌ |

> Pi0-FAST 30Hz 略不足是自回归架构固有特性(逐 token 串行前向,有效 token ~13 个 × 单步 ~30ms),非优化遗留问题。chunk=10 缓冲仅 0.33s,对网络抖动比 Pi0 flow(1.67s 缓冲)更敏感。真机部署建议以实际动作平滑度为准。

---

## 故障排查

### 服务器启动 OOM

```
torch.OutOfMemoryError: CUDA out of memory
```
- 4090 应走 `--quantization none`(峰值 ~6-7GB,24GB 显存无压力)
- 小显存机器必须 `--quantization int8wo`
- 检查是否有其他进程占用 GPU:`nvidia-smi`
- 确保设了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### 客户端连接服务器超时

```
requests.exceptions.ConnectionError / Timeout
```
- 确认服务器 `--host 0.0.0.0`(不是 `127.0.0.1`)
- 检查防火墙:`curl http://<server-ip>:8000/health` 应返回 JSON
- 调大 `inference.api.timeout`(默认 5s)
- 弱网络可降低 JPEG 质量减小包体

### 客户端报 `no image keys found in obs_frame`

`ApiInferenceEngine._encode_obs` 依赖观测键前缀 `observation.images.`。确认:
- `--policy.path` 指向的 `config.json` 与服务器 checkpoint 一致(相机名匹配)
- 相机在 yaml 中正确定义

### 动作错位 / 关节顺序不对

`ordered_action_keys` 来自机器人硬件的 `action_features`(`.pos` 关节)。若动作发送到错误关节:
- 确认客户端 `config.json` 的 `action_feature_names` 与机械臂关节一致
- `inference.api.n_action_steps` 必须与服务器 `chunk_size` 一致

### 服务器 vs 本地动作轨迹差异大

- pi0 flow 推理有随机性(本地两次自身差异 ~9%),**小幅差异正常**
- 若大幅偏离:确认服务器与客户端用**同一 checkpoint**、同一 `num_inference_steps`、同一 `task` 字符串
- `quantization` 差异(int8 vs bf16)引入 ~0.5% 相对误差,对轨迹无实质影响

### torchao / diffusers 报错(仅 int8 路径)

```
NameError: name 'logger' is not defined  (diffusers 0.35.2 bug)
```
运行 `python scripts/check_patch_diffusers_torchao.py --apply` 打补丁,详见 `README_NEXARM.md`。

### 控制环丢帧

日志出现 `running slower than X Hz than target FPS`:
- 通常是相机 FPS 跟不上(USB 带宽),与 API 化无关
- 或网络 RTT 过大——检查 LAN 延迟,考虑降低 `num_inference_steps`
- chunk 缓冲期间控制环不应阻塞,若阻塞说明 RPC 同步阻塞了控制线程(确认 `ApiInferenceEngine.get_action` 在队列非空时不发 RPC)

---

## 相关文件

| 文件 | 说明 |
|---|---|
| `server/policy_server.py` | FastAPI 推理服务主体(按 config.json type dispatch pi0/pi0_fast) |
| `server/requirements.txt` | 服务器依赖 |
| `server/DEPLOY.md` | 本指南 |
| `src/lerobot/rollout/inference/api.py` | 客户端 `ApiInferenceEngine`(policy 无关) |
| `src/lerobot/rollout/inference/factory.py` | 引擎注册与 dispatch |
| `src/lerobot/rollout/context.py` | api 模式跳过本地权重加载 |
| `src/lerobot/policies/pi0_fast/modeling_pi0_fast.py` | Pi0-FAST 解码提前终止优化(备份 `.bak_20260828_111900`) |
| `examples/nexarm_pi0/inference_api_client.yaml` | Pi0 flow 客户端配置示例 |
| `examples/nexarm_pi0fast/inference_api_client.yaml` | Pi0-FAST 客户端配置示例(n_action_steps=10) |
| `scripts/test_load_pi0_int8_5060.py` | Pi0 flow 本地推理参照脚本(加载路径同源) |
| `scripts/test_load_5060.py` | Pi0-FAST 本地推理参照脚本(加载路径同源) |

---

## 阶段一边界(不做的事)

- **不动模型代码**:`modeling_pi0.py` / `_preprocess_images` / `sample_actions` 零修改
- **不前移预处理到客户端**:阶段二才考虑(当前带宽非瓶颈)
- **不做并发批处理**:单 worker 串行前向
- **不做相对动作 / RTC**:api 引擎走 `predict_action_chunk`,与 sync 同属非 RTC 路径
