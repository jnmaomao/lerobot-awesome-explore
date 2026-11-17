# LeRobot 之上的 B/S 采集控制台：技术选型与架构设计

## 一、选型结论

Foxglove 应当被换掉，但替换物不是"自研看板"，而是 **Rerun 的官方 Web Viewer**——它就是 `lerobot-record` 弹出的那个窗口编译成 WASM 的版本，摄像头画面与舵机曲线一个像素都不差，只是跑在浏览器里。

**版本基线：钉在 v0.6.1。** 这条决策先于其他所有选型，因为它决定后面每一条事实的来源版本。本地 fork 基于 v0.6.1 且带大量电机层与推理侧定制，而 v0.7.0 的破坏性变更恰好集中命中这些定制点。把两版逐文件比对之后，结论比预想的更干净：

| 控制台依赖的能力 | v0.6.1 | v0.7.0 |
|---|---|---|
| `RecordConfig` 全字段与必填性 | ✓ 与 0.7.0 **逐字段一致**【1】 | ✓ |
| `DatasetRecordConfig` 全字段与默认值 | ✓ 与 0.7.0 **逐字节相同**【2】 | ✓ |
| `init_visualization` / `init_rerun` 的 rerun-vs-foxglove 语义 | ✓ 一致【3】【4】 | ✓ |
| `record_loop` 的 `events` 字典与三类事件 | ✓ 一致【1】【5】 | ✓ |
| draccus YAML/JSON 配置文件路径 | ✓ 一致【6】 | ✓ |
| `rerun-sdk>=0.24.0,<0.34.0`、`foxglove-sdk>=0.25.1,<0.26.0` | ✓ **完全相同的 pin**【7】 | ✓ |
| `lerobot-annotate`、`steerable_pipeline`、`lerobot-eval`、`lerobot-rollout` | ✓ 全有【8】 | ✓ |
| `transport/zenoh.py`（presence 令牌、背压） | ✗ **不存在**【8】 | ✓ |
| `env_server/`（会话语义参照） | ✗ **不存在**【8】 | ✓ |
| `remote_inference/`、LanceDB 数据集后端 | ✗ 不存在【8】 | ✓ |

**控制台要用的能力，v0.6.1 已经全部具备；而 v0.7.0 新增的三块（Zenoh、env_server、remote_inference）与本地 fork 的冲突面（电机层重建、reBot B601 的 `gripper_mit_kp/kd` 移除、policy API 变更、训练栈重建）高度重叠。** 换基线的成本因此接近于零。

这带来一个必须写进架构的推论：**控制台要按"版本无关的隔离层"来设计**，否则将来升到 0.7.0 就是重写而不是升级。4.6 节给出具体规则。

这个结论与上一轮报告里"实时反馈直接用 LeRobot 原生通道"的建议不冲突，但要把话说准：LeRobot v0.6.1 与 v0.7.0 的原生通道都只有两个后端（`rerun` 与 `foxglove`），两版一致【3】，其中只有 Rerun 后端支持**把数据推送到远端服务器**（`--display_ip` / `--display_port`），而 Foxglove 后端的 `ip` 语义是"绑定哪个网卡供本机客户端连接"【1】【3】。换句话说，Rerun 才是那个天然适配中心化 B/S 的后端。

三块需求的归属也随之清晰：

| 需求 | 归属 | 需要写多少代码 |
|---|---|---|
| 浏览器里看实时相机 + 舵机曲线 | 中心 Rerun Server + `web-viewer-react` 嵌入 | 部署配置，几乎零业务代码 |
| 网页下发采集任务、开始/结束 | 自研控制面，调用 LeRobot 的 `record(cfg)` | 中心服务 + 设备 Agent 两端 |
| 后续标注复用同一界面 | 自研控制台的第三个模块 + LeRobot 标注管线 | 标注表单与审核状态机 |

## 二、三个决定性事实

**其一，LeRobot 的录制进程已经支持"把画面发到远处"。** `RecordConfig` 里有一组字段：`display_data`、`display_mode`（`"rerun"` 或 `"foxglove"`）、`display_ip`、`display_port`、`display_compressed_images`、`resume`【1】。文档字符串明确写着：对于 `rerun` 后端，`ip`/`port` 指向一个**可选的远端 Rerun 服务器**；对于 `foxglove` 后端，`ip` 是要绑定的网卡、`port` 是端口【1】【3】。于是这一条命令就完成了"采集 + 远程可视化"：

```
lerobot-record --display_data=true --display_mode=rerun \
               --display_ip=<中心服务器> --display_port=9876 \
               --robot.type=... --dataset.repo_id=...
```

还有一个容易忽略但很关键的细节：LeRobot 在 `display_ip` 与 `display_port` 同时存在时，会**自动打开图像压缩**【1】。上游自己把"远程传输要走 JPEG"写进了默认逻辑，这说明带宽是真实约束，不是理论担忧。

**其二，采集会话可以被程序驱动，而不必解析 stdout。** `lerobot/scripts/lerobot_record.py` 里 `record(cfg)` 与 `record_loop(...)` 都是模块级函数【1】，因此设备 Agent 可以直接调用它们，不需要去 grep 子进程输出。

但**"构造一个 `RecordConfig` 对象直接调用"这句话需要修正**。`RecordConfig` 的 `robot: RobotConfig` 与 `dataset: DatasetRecordConfig` 两个字段**没有默认值、是必填项**，而 `teleop` 虽声明为可选，`__post_init__` 里一旦为空就直接抛 `ValueError`【1】。更关键的是这些嵌套类型都是 draccus 的多态 dataclass（`ChoiceRegistry` + `--robot.type=...` 的选择机制）【6】——手工搭出 `So101FollowerConfig` 加相机配置加 `DatasetRecordConfig` 这棵树是可行的，但会绕过 draccus 负责的 CLI/YAML 合并、类型校验与插件发现。

**正确的做法是让 draccus 干活**：LeRobot 的配置解析器支持 YAML 与 JSON 配置文件路径，命令行参数可覆盖文件中的值【6】——v0.6.1 与 v0.7.0 在这一点上同构，所以任务下发链路在两版上写法完全相同。因此任务下发应当是——**控制台生成一份任务 YAML，设备 Agent 用 draccus 把它解析成 `RecordConfig`**（或直接以 `--config_path` 交给 CLI），而不是在 Python 里手搓配置树。`RecordConfig.resume` 字段【1】正好对应上一轮报告里"episode 级 checkpoint + 续采"的协议要求。

**其三，LeRobot v0.7.0 自带一套传输与会话协议。** `src/lerobot/transport/` 下有 `services.proto`（仅用于 RL 的 LearnerService）、`zenoh.py`（Zenoh 传输实现）与 `utils.py`；`src/lerobot/env_server/` 下有 `client.py`、`server.py`、`protocol.py`，对应脚本 `lerobot_env_server.py`【9】【10】【11】【12】。其中 `zenoh.py` 提供 `ZenohConfig`（JSON5 配置文件）、有界订阅者 `BoundedSubscriber`（带背压）、`PresenceEvent` / `PresenceToken`（**设备在线/离线的存活性令牌**）、以及基于 query 的请求-响应【9】。

`env_server` 的会话语义尤其值得借鉴，因为它已经把"远程控制一个运行中的东西"这件事的边界都想清楚了：

- `EnvClient.describe()` 做能力发现，版本或部署不匹配直接报出可操作的错误信息【10】
- `open()` 开启会话并返回 `session_id`，`request("reset"|"step")` 驱动，`prefix()` 给出本次会话的键前缀【10】
- 键路径有清晰的层级：`lerobot/env/v1/deployments/{deployment}/instances/{instance}/sessions/{session}`【12】
- 每次 `reset` 递增 generation，早于该 generation 的请求被判为陈旧并拒绝；模拟器一旦失败，会话会**锁死**并拒绝后续所有操作【11】
- `EnvServerConfig` 有 `max_sessions` 与 `max_envs` 两个并发闸门，并且有"客户端多久不出现就判定会话过期"的超时【11】
- 协议版本号 `PROTOCOL_VERSION = 1`，默认单消息上限 32 MiB，相机帧在回复中以 JPEG 形式批量返回【12】

需要说清一个边界：`env_server` 面向的是**仿真环境**（`EnvServerConfig` 要求 `sim.type` 与 `deployment` 非空【11】）。它的语义模型可以照抄，真机的"采集会话"仍然要自己实现——这与上一轮报告"运营控制层必须自研"的结论一致，但现在的自研是有明确契约可抄的，而不是从零设计。

## 三、实时监视层

### 3.1 为什么是 Rerun Web Viewer 而不是 Foxglove

Rerun 的 Viewer 有原生桌面版与浏览器版两种形态，二者是同一套代码：原生版用 wgpu + egui，浏览器版把整个 Viewer 编译成 WebAssembly，交付物是一个瘦 HTML、一个 `.wasm` blob 和一个自动生成的 JS 桥接文件，由 `re_web_viewer_server` 通过 HTTP 提供【13】。浏览器版可以用 gRPC-Web 读取流式日志数据，也可以拖放加载 `.rrd` 文件【13】【14】。

选它而不是 Foxglove 的理由是三条具体的，不是"更轻"这种空话：

其一，**UI 完全一致**。你要的就是 `lerobot-record` 弹出的那个窗口。Rerun 后端下 LeRobot 写的本来就是同一套实体路径与视图，改成 Web Viewer 等于零视觉改造成本；Foxglove 则是另一套 topic 树 + 面板的调试器范式，要重新适配成"采集监视"的信息结构。

其二，**不需要改 LeRobot 的录制代码**。后端选择是运行时的 `--display_mode` 字符串，`init_visualization` / `log_visualization_data` / `shutdown_visualization` 三个函数做后端无关的分派，调用方从不直接 `if rerun / else foxglove`【3】。

其三，**可以嵌进你自己的控制台**。Rerun 官方提供 `web-viewer` 与 `web-viewer-react` 两个 NPM 包，把 Web Viewer 作为组件嵌进网页应用【15】。这意味着左边的实时监视面板和右边的任务表单可以是同一个 React 应用里的两个区域，而不是一个 iframe 套一个壳。

### 3.2 中心 Rerun Server 的部署形态

Rerun 的 CLI 本身就是服务器。三种相关形态【16】：

```
rerun --serve-web        # 同时托管 HTTP Web Viewer 与 gRPC 服务器（默认 9876）
rerun --serve-grpc       # 只托管 gRPC 服务器，SDK 连入后数据被缓冲，供后来连上的 Viewer 消费
rerun --connect <url>    # 不起服务器，改为连接已有服务器
```

Python 侧对应 `rr.serve_grpc()` 与 `rr.serve_web_viewer(connect_to=server_uri)`【17】。官方示例的写法就是这两行连用，然后 `rr.log(...)` 持续灌数据【17】。

因此中心侧部署一次 `rerun --serve-web`，设备侧的 LeRobot 进程用 `--display_ip` / `--display_port` 指过来，浏览器访问 Web Viewer 的 HTTP 地址即可。

三个部署参数是必答题：

- `--bind 0.0.0.0`：否则 gRPC 只监听回环，设备连不上【16】
- `--cors-allow-origin`：自有域名下的 Web Viewer 需要它，官方文档特别说明"自定义域名的 Viewer 需要它"【16】
- `--server-memory-limit`（默认 1 GiB）与 `--memory-limit`：超限时**丢弃最旧数据**而非拒绝连接，这正是长时采集想要的降级行为【16】

另外，浏览器访问 Web Viewer 走的是 **gRPC-Web over HTTP**，不是裸 gRPC【14】。所以中心服务器对外暴露的是 HTTP 端口，鉴权、TLS、访问控制都在这一层用反向代理做——不要把 9876 直接开到公网。

### 3.3 多设备如何区分：app_id 共享，recording_id 区分

这一点此前写错过，必须纠正。LeRobot 的 `init_rerun(session_name=...)` 由 `lerobot_record.py` 调用时传入的 `session_name` 是**硬编码的 `"recording"`**【4】【1】，而 `RecordConfig` 并不暴露 app_id 字段【1】。所以：

- **所有设备共享同一个 application_id**，无法通过配置改成 `arm-03` 之类
- 区分不同设备的是 **recording_id**
- 这恰好不是坏事：Rerun 的 blueprint 按 application_id 存储，**同 app_id 的不同 recording 共享同一份视图布局**【18】。管理员从采集员 A 切到 B 时，相机面板与曲线面板位置完全一致，不需要重新摆

recording_id 的生成规则需要讲清楚，因为它是多机方案的地基。Rerun Python SDK 的 `get_recording_id()` 文档写明：**默认 recording_id 基于 `multiprocessing.current_process().authkey`**，`multiprocessing` 派生的所有子进程共享同一个默认值【19】。推论有三条：

1. 两台设备各自独立启动的 `lerobot-record` 进程，authkey 不同，recording_id 不同，因此在同一个代理下是两个独立 recording，不会互相覆盖
2. **同一台机器上由同一个父进程用 `multiprocessing` 派生的多个子进程，recording_id 相同，两路流会合并成一条录制**——这是本方案最容易踩的坑
3. 因此设备 Agent 启动采集进程**必须用 `subprocess`（独立 exec），不能用 `multiprocessing`**

另外，Rerun 源码里留着一条注释 `TODO(#3793): defaulting recording_id to authkey should be opt-in`【19】，说明上游认为这个默认行为本身是有争议的、未来可能变成可选。这意味着**不要把 recording_id 的默认行为当作稳定契约**：显式指定 recording_id 才是稳的，而 LeRobot 目前没有暴露这个参数——这是唯一一处真正需要考虑打补丁的地方，且只在"同一台机器并发多路采集"时才必要。

### 3.4 性能与版本约束

Web Viewer 的硬限制必须提前知道【15】：

- 跑在 **32 位 Wasm** 上，实践中内存约 **2 GiB** 上限
- **单线程**，整体比原生版慢
- 浏览器端优先 WebGPU，不可用时回退 WebGL，**回退路径的功能集更受限**

LeRobot 的依赖固定是 `rerun-sdk>=0.24.0,<0.34.0`，`foxglove-sdk>=0.25.1,<0.26.0`【7】——**这两个 pin 在 v0.6.1 与 v0.7.0 上完全相同**，所以 Rerun Web Viewer 的版本约束在换基线时不受影响。（`remote` 额外依赖里的 `eclipse-zenoh` 与 `msgpack` 是 v0.7.0 才有的，v0.6.1 没有，见 4.4 节。）`pip install rerun-sdk` 会自带 `rerun` 二进制，所以**必须用与 SDK 同版本的那份 CLI 来起中心服务器**，不能把系统里的 Rerun 单独升到社区最新小版本，否则 gRPC 协议与 Web 资源都对不上。

结论是：**4 路 1080p@30fps 的实时监视很可能超出 Web Viewer 的舒适区**。LeRobot 在远程模式下会自动开 JPEG 压缩【1】，说明带宽已经在被约束。真到那一步，处理方式是分层的——监视面板降分辨率与帧率用于"看着没问题"，原始数据仍按采集设置落盘；或者对相机单独走 MJPEG/WebRTC 面板，曲线仍走 Rerun。

## 四、采集任务控制面

### 4.1 设备侧 Agent

每台机械臂的工控机上跑一个常驻 Agent，承担两件事：

1. 持有 LeRobot 进程（`record(cfg)` 的调用方），负责把配置变成一次采集会话
2. 与中心服务保持长连接，接收任务指令、回传状态

`record()` 是阻塞函数，所以会话应当跑在**独立子进程**而非主线程——这样一次采集中途结束不会污染 Agent 的事件循环，且子进程崩溃不影响 Agent 存活。启动方式必须是 `subprocess` 而不是 `multiprocessing`，理由见 3.3 节关于 recording_id 的说明【19】。episode 级续采则用 `RecordConfig.resume`【1】。

**网页能控制到什么粒度，需要先讲清。** v0.6.1 与 v0.7.0 的 `record_loop` 签名里都有一个 `events` 字典参数（0.6.1 标注为 `dict`，0.7.0 细化为 `dict[str, bool]`），两版行为一致【1】【5】，控制循环通过读这个字典来响应三类事件：`exit_early`（提前结束当前阶段）、`rerecord_episode`（丢弃并重录当前 episode）、`stop_recording`（结束整个录制）【5】。问题在于这个字典由 `init_keyboard_listener()` 产生，而 `record()` 在自己内部创建它【5】——也就是说**用 CLI 跑的时候，事件源是键盘监听器，不是任何外部 API**。

由此得到控制粒度的结论：

| 粒度 | 网页能否远程操作 | 实现方式 |
|---|---|---|
| 开始任务 / 结束任务 / 取消任务 | **能** | 管理子进程生命周期（启动、发送终止信号、回收） |
| 中断当前 episode 并重录 | **不能直接做到** | CLI 下事件源是键盘；需自行编排 `record_loop`（复制 `record()` 的会话编排逻辑，0.6.1 约一百八十行、0.7.0 约两百行）后接管 events 字典 |
| 提前退出当前 phase | 同上 | 同上 |

**建议第一版把 episode 级控制留在采集终端本地**——采集员人就站在机械臂旁边，键盘触发重录本来就顺手。把 episode 级控制搬到网页需要复制 LeRobot 的会话编排逻辑，那是一次实打实的 fork 成本，与"尽量不改上游"的整体策略冲突。等中心调度跑顺了再评估。

### 4.2 中心服务与接口

中心服务用 FastAPI + WebSocket。LeRobot 自己的 `phone` 额外依赖里已经出现 `fastapi<1.0`【7】，说明在这个代码基上起 Web 服务不算引入异构技术栈。

接口面照抄 `env_server` 的语义模型，但换成采集域的对象：

| 接口 | 语义 | 对应 env_server 的概念 |
|---|---|---|
| `GET /devices` | 设备台账、在线状态、能力描述 | `describe()` + presence 令牌【9】【10】 |
| `POST /tasks` | 创建任务模板（任务名、目标 episode 数、fps、机械臂与相机配置） | `EnvServerConfig` 的 deployment 与限额【11】 |
| `POST /devices/{id}/sessions` | 下发采集任务，返回 session_id | `open()`【10】 |
| `POST /sessions/{id}/stop` | 结束采集，触发 episode 收尾与上传 | `_Session.close()`【11】 |
| `GET /sessions/{id}/status` | episode 计数、实际帧率、码率、掉帧、标定快照版本 | `StepResult` 的观测回传【12】 |
| `WS /stream` | 状态与事件推送 | — |

其中"采集会话"这一层是自研的——`env_server` 的会话跑的是仿真 env，采集会话跑的是真机 + 数据集写入。上表右列的语义模型照抄，实现另起。

**右列的 env_server 概念取自 v0.7.0，v0.6.1 里没有这个模块**【8】。它在这里是**语义参照物而不是依赖**：会话 id、generation 防陈旧、失败锁死、并发闸门这四套语义值得照抄，但实现要自己写。控制面因此不因基线是 0.6.1 而缺失任何东西——这一点在 4.4 节还会再遇到一次。

### 4.3 采集任务模板的字段模型

`POST /tasks` 创建的任务模板不该另造一套字段——LeRobot 的 `DatasetRecordConfig` 就是现成的 schema，控制台只需把它暴露成表单，设备侧再包一层机器人与遥操作配置【2】。

| 字段 | 源码默认值 | 控制台暴露 | 说明 |
|---|---|---|---|
| `repo_id` | `""` | 是 | 数据集仓库 ID。**创建时会被自动追加 `_YYYYMMDD_HHMMSS` 时间戳**【2】【1】 |
| `single_task` | `""` | 是 | 任务文本，成为每条 episode 的 task 字段 |
| `root` | `None` | 是 | 本地落盘路径；为 None 时落到 `$HF_LEROBOT_HOME/<repo_id>`【2】 |
| `fps` | 30 | 是 | 采集帧率，也是后续全部时序换算的基准 |
| `episode_time_s` | 60 | 是 | 单条 episode 最长时长 |
| `reset_time_s` | 60 | 是 | 回位阶段时长。回位阶段**不写入数据集**，但会占用实际采集时间【2】 |
| `num_episodes` | 50 | 是 | 目标条数 |
| `video` | True | 否 | 是否存视频 |
| `push_to_hub` | **True** | 是（模板默认改为 False） | 源码默认值是 True。设备侧直连 Hub 往往不可达或不合规，任务模板应当默认关闭、由中心统一上传【2】 |
| `private` | `None` | 是 | 为 None 时沿用 Hub 上的组织默认值 |
| `tags` | `None` | 是 | 数据集标签，中心侧批量治理用 |
| `num_image_writer_processes` | 0 | 是 | 图像写盘子进程数。官方建议以"每相机 4 线程 + 0 进程"起手【2】 |
| `num_image_writer_threads_per_camera` | 4 | 是 | 单相机写盘线程数。过多会阻塞主线程导致遥操作帧率不稳【2】 |
| `video_encoding_batch_size` | 1 | 是 | 批量编码条数；1 表示立即编码（默认行为），调高用内存换吞吐【2】 |
| `rgb_encoder` / `depth_encoder` | 默认工厂配置 | 是 | 视频编码器设置（codec / 质量 / GOP），可覆盖如 `--dataset.rgb_encoder.vcodec=h264`【2】 |
| `streaming_encoding` | False | **是** | 开启后采集时实时编码，不再先写 PNG 再转码，**`save_episode()` 接近瞬时**【2】 |
| `encoder_queue_maxsize` | 30 | 是 | 编码队列上限，30fps 下约 1 秒缓冲，编码跟不上时提供背压【2】 |
| `encoder_threads` | `None` | 是 | 每个编码器实例的线程数，None 为自动 |
| `no_stamp` | False | 是 | 不给 `repo_id` 追加时间戳，用于自管版本命名（配合后续的 `lerobot-edit-dataset merge`）【2】 |

有三个默认值需要在任务模板里被覆盖掉，否则会踩坑：

第一，**`push_to_hub` 源码默认是 True**。不显式关闭，设备端每采完一批就尝试往 Hugging Face Hub 推，网络策略不允许就直接失败，且产物散落在各人账号下。任务模板应当默认 `push_to_hub: false`，上传交给中心。

第二，**`reset_time_s` 源码默认是 60 秒**。每个 episode 之后都有一段不写数据的回位时间，这段时间是真实成本，任务排期与"一天能采多少条"要按 `episode_time_s + reset_time_s` 算，而不是只按 60 秒算。

第三，**`repo_id` 会在创建时被自动加时间戳**【1】【2】。`stamp_repo_id()` 只在数据集创建时调用，续采路径走的是 `LeRobotDataset.resume`，不会二次加戳——所以"episode 级 checkpoint + 续采"不会把数据集名冲掉。但如果希望数据集名由中心自己管理（版本号、批次号），要显式设 `no_stamp: true`，否则名字里会多一段不可控的时间戳。

`streaming_encoding` 那一条值得单独强调：它把"采集时先写 PNG、episode 结束再转码"变成"采集时直接编码"。上一轮报告强调 episode 级 checkpoint 对数据可恢复性的意义——如果 episode 结束时还要等一段转码，checkpoint 的"立即可校验"就不成立。**这一项应当在 POC 第 1 周就测，不要留到后期。**

### 4.4 传输层选型

**基线上没有 Zenoh，这是换基线带来的唯一实质影响。** `transport/zenoh.py` 是 v0.7.0 才有的模块；v0.6.1 的 `transport/` 目录只有 `services.proto`、`services_pb2*.py` 与 `utils.py`【8】。所以第一版的传输层必须自己写两件 Zenoh 本来提供的事：

- **在线检测**：设备 Agent 周期上报心跳，中心按 `last_seen` 判定失联。`PresenceToken` / `PresenceEvent` 的语义值得照抄——**"设备掉线"与"任务失败"必须能区分**，这是中心服务最难自己写对的一块【9】
- **背压**：状态下行与事件上行速率不同，用有界队列加丢弃策略即可，不需要分布式消息中间件的语义

如果只在内网单站点运行，用 WebSocket + HTTP 就够。**升到 v0.7.0 之后再把这一层换成 Zenoh**——它在跨站点场景下确实能省掉中心的公网入口（scouting / router 模式）。届时设备 Agent 与中心之间的契约不变，只是底层换实现，这也正是把传输层收敛在一个接口后面的理由。

倾向是：**设备 Agent 与中心之间用 WebSocket/HTTP（0.6.1 基线），浏览器与中心之间同样用 WebSocket/HTTP**，两段解耦；Zenoh 作为 v0.7.0 之后的可选替换，而不是前置条件。

### 4.5 故障语义必须先定义

### 4.6 版本无关的隔离层：控制台不 import lerobot 内部模块

既然基线是 v0.6.1、而升级回 0.7.0 的窗口迟早会打开，控制面就不该把自己焊死在哪一版的内部结构上。落地规则有三条：

1. **控制台与设备 Agent 是独立包**，放在 `src/lerobot` 之外。设备 Agent 通过 `subprocess` 启动 `lerobot-record`，用 draccus 配置文件对接，**不 import `lerobot.*`**
2. **唯一例外集中在一个适配模块**：`RecordConfig` / `DatasetRecordConfig` 的类型引用与字段映射全部收口到一个文件。两版的这两个类字段集完全一致【1】【2】，所以这个适配模块在 0.6.1 与 0.7.0 上是同一份代码；将来若 `motor_family`（reBot B601 的 `gripper_mit_kp/kd` 已在 0.7.0 被移除，见 6.1 节）之类字段真的分叉，只改这一个文件
3. **CLI 与配置文件是对接面，不是 API**。`--display_mode`、`--display_ip`、`--display_port`、`--config_path`、`--resume` 在两版之间是稳定的——这正是把控制面做成"配置生成器 + 进程管理器"而不是"函数调用者"的原因

收益很直接：将来升级到 0.7.0 时控制台代码不动，只在设备镜像里换 lerobot 版本并跑一次回归。**控制台是版本无关的隔离层，lerobot 是可替换的后端。**

三件容易被合并、但必须区分的事件：

1. **设备失联**：presence 令牌过期。会话按 env_server 的超时策略关闭，上传已完成的 episode
2. **采集失败**：进程退出码非零或会话抛出。保留失败前已完成的 episode
3. **主动结束**：操作员点"结束"。正常收尾并上传

上一轮报告里"舵机故障会被 LeRobot 当成 episode 结束信号而中止整个会话"【引自前序报告】在这个架构下同样成立，所以"结束"与"中止"在语义上必须能被区分，否则操作员会误以为任务完成了。

## 五、标注复用同一界面

控制台外壳统一为五个模块：设备台账、采集任务、数据集、作业、标注。共享同一套数据集模型与权限。

LeRobot 侧现成的件比想象中多，而且 v0.6.1 与 v0.7.0 在这块完全一致（`lerobot-annotate`、`steerable_pipeline` 的提示词集、`jobs/annotate.py` 两版都在）【8】：

- `lerobot-annotate` 支持 `--root` 或 `--repo_id` 输入，产出到 `--new_repo_id`，完成后推送 Hub、生成数据集卡片、打版本 tag【20】
- `src/lerobot/annotations/steerable_pipeline/` 是**可干预的标注流水线**，带 `AnnotationPipelineConfig`（`--vlm` / `--plan` / `--job` 等嵌套配置）与 `prompts/plan_subtask_relabel.txt` 这类子任务重标注提示词
- `src/lerobot/jobs/annotate.py` 提供 `submit_annotate_to_hf()`：把标注作业提交到 HF Jobs 的 GPU pod（自带 vLLM 镜像），并可跟随日志直到终态，或用 `--job.detach` 立即返回；`--job.target=local` 防止 pod 自我递归派发【21】

这意味着**标注作业的"提交—跟日志—看结果"闭环，LeRobot 已经在 HF Jobs 上实现过一遍**。控制台要做的不是重新发明作业模型，而是把这套语义搬到自有的队列与界面上。

数据集回看也建议走 Rerun：Rerun 的数据层把 LeRobot 作为可摄取格式，有专门的 `re_lerobot` crate 负责 LeRobot 数据集到 chunk 的加载【22】。这样**实时监视与离线回看是同一套 UI、同一套时间轴语义**，标注时在时间轴上定位帧、在右侧表单写标注，两边共享 frame index。

真正需要自研的标注部分是：帧与子任务段的切分、动作/事件标签的录入、审核状态机（待标/已标/复核/驳回）、多人分工与冲突处理。Rerun 负责"看得见"，表单与状态机要自己写。

## 六、风险与取舍

**版本耦合是长期成本，但风险的方向变了。** `rerun-sdk` 被固定在 `>=0.24.0,<0.34.0`【7】——好消息是这个 pin 在 v0.6.1 与 v0.7.0 上完全相同，所以 Rerun 侧不构成额外的版本风险。真正的风险来自另一头：LeRobot 自身在 0.7.0 做了大规模重构（电机层、policy API、训练栈、传输层），而本地 fork 的定制正落在这些面上。因此约束要改成两条：

- **Rerun 版本锁进镜像 tag**，CI 里加一条"`lerobot` 与 `rerun` 版本一致性"断言——这条不变
- **LeRobot 版本不做向上追踪**。基线钉在 v0.6.1，升级只在满足明确条件时触发（需要 0.7.0 的新 policy 或官方 remote 推理；本地电机层已迁到新总线；上游发布破坏性变更收窄的补丁版）。在此之前，任何"顺手升一下"的诱惑都要挡住

**Web Viewer 的内存与线程限制是硬边界。** 2 GiB、单线程【15】意味着高分辨率多相机会卡。这意味着要么在监视侧降规格，要么承认监视面板不能替代离线分析。

**中心化 Rerun Server 是单点。** 它挂了，所有在采设备的数据流就断了——但采集本身不该中断，LeRobot 的录制写盘与可视化是解耦的（`log_visualization_data` 失败不应影响写盘）。这一点需要在封装 `record()` 时保证：可视化异常降级为"本轮无监视"，而不是终止采集。

**不要 fork LeRobot。** 保持 pip 依赖关系、把 fork 面控制在最小，是因为 v0.7.0 这种破坏性版本节奏下，fork 的合并成本会迅速超过自研。需要改的是自己的 Agent 与控制层，不是上游。

### 6.1 升级到 v0.7.0 的实测冲突面

这一节的作用是让"什么时候升"这个决定有据可依，而不是靠感觉。以下四行是能从上游仓库直接测出来的，已逐项核对：

| 0.7.0 的破坏性变更 | 上游实测证据 | 对本地 fork 的含义 |
|---|---|---|
| **reBot B601 夹爪增益字段被移除** | `gripper_mit_kp` / `gripper_mit_kd` 在 v0.6.1 的 `rebot_b601_follower.py` 与其 config 中各出现 1 次，在 v0.7.0 中**命中数为 0**；v0.7.0 新增 `motor_family.py`，以 `MotorFamily`（DM / RS）与 `ArmControlMode`（MIT / POS_VEL）替代【23】【24】 | **直接 API 断裂**：任何使用这两个字段的代码都必须重写 |
| **Feetech / Dynamixel 换到 rustypot** | v0.7.0 的 pyproject 新增 `rustypot-dep = ["rustypot>=1.13.1,<2.0.0"]` 与 `serial-motors` extra，`feetech` / `dynamixel` 两个 extra 由 vendor SDK（`feetech-servo-sdk`、`dynamixel-sdk`）改为 `lerobot[serial-motors]`；注释明写 "rustypot drives Feetech and Dynamixel motors"【25】 | 本地自研的两套串口电机层无法直接沿用 |
| **vendor 协议层与控制表被移除** | `motors/motors_bus.py` 由自研 pyserial + `PortHandler`/`PacketHandler` 协议层改为走 rustypot；`feetech.py` 从 19643 字节缩到 1969 字节、`dynamixel.py` 从 11337 缩到 1898 字节；`feetech/tables.py`、`dynamixel/tables.py`、`encoding_utils.py` 三个文件在 v0.7.0 中不存在【25】 | 控制表与编码工具的本地改动全部要迁移 |
| **`async_inference` 整体删除** | v0.6.1 有 `async_inference/` 6 个文件，v0.7.0 为 0；取而代之的是新增的 `remote_inference/` 38 个文件与 `transport/zenoh.py`【8】 | 自研边端推理 API 与上游新方案功能重叠但接口不同 |

另有三行涉及本地 fork 内部状态与 GPU 环境（policy API 的 `PolicyOutput` / dtype 统一、训练栈 wandb→trackio 与 FSDP1→FSDP2、RTC 修复与本地尖峰补丁的冲突），**无法从上游仓库单独判定**，需要在本地 fork 上跑 `git diff` 才能确认影响面。这三行也是"不升级"的真实分量所在，建议在决定升级前先量化一次。

**结论**：前四行已足以说明升级不是"合并几个 PR"，而是硬件层与推理层的重做适配；后三行决定的是重做的具体规模。因此升级窗口应当等本地电机层迁到新总线之后，再评估剩余三行。

这个策略目前有一个明确的例外窗口，且应当被承认：如果要做"网页远程重录当前 episode"，就得自己接管 `record_loop`，而 `record()` 的会话编排逻辑有两百行上下（0.6.1 约一百八十行、0.7.0 约两百行；包含创建数据集、连接机器人、初始化处理器、逐条 episode 的记录相与回位相、收尾与上传）【5】——复制这部分等于承担一份持续的上游合并负担。**建议第一版不做这个功能，把 episode 级控制留在采集终端本地**，等它真的成为瓶颈时再决定是复制这 200 行，还是向上游提一个"外部注入 events"的 PR。后者更干净但要等版本窗口。

**网络暴露面。** Rerun gRPC 没有内建鉴权。自建域名必须走反向代理 + TLS + 自己的鉴权。跨站点场景不要把 gRPC 端口暴露出去：0.6.1 基线上用中心侧的反向代理与 VPN/专线收敛入口；升到 0.7.0 之后再改用 Zenoh 的 router 模式（见 4.4 节）。

## 七、POC 验证清单

第 0 周，五个假设，五个可证伪的实验：

1. **单臂远程可见性**：中心起 `rerun --serve-web`，设备跑 `lerobot-record --display_mode=rerun --display_ip --display_port`，浏览器里是否出现与本地窗口一致的相机画面与舵机曲线
2. **多臂路由**：两台设备（同一 app_id、不同 recording_id）同时连同一中心 server，浏览器能否在两者间切换且各自独立；**并确认两者的 recording_id 确实不同**（同机启动需注意 3.3 节的 authkey 陷阱）
3. **嵌入可行性**：`web-viewer-react` 能否与自有表单面板并排布局（而非 iframe 嵌套），交互与滚动是否协调
4. **性能边界**：4 路 1080p@30fps 下的实际帧率、浏览器内存占用、CJK 字体与 canvas 的兼容性——据此决定监视面板是否要转 MJPEG/WebRTC 路线
5. **落盘延迟**：`streaming_encoding` 开与关两种模式下，`save_episode()` 的实际耗时、CPU 占用与掉帧率对比——据此确定 episode 级 checkpoint 能否"立即可校验"

第 1 周，把 YAML 任务配置 → draccus 解析 → `record(cfg)` 这一条链包成设备 Agent 的会话接口，验证开始/结束/取消三条路径与 episode 级续采。

**贯穿全程的一条约束**：控制台代码只允许 import 适配模块里收口的那几个类型，其余一律走 CLI 与配置文件。这条约束的验收方式是**双版本回归**——同一份任务 YAML 在 v0.6.1 与 v0.7.0 两个容器里各跑一次，都必须成功采集到 episode。跑通这条，才说明控制台真的与版本无关，而不是恰好在 0.6.1 上能用。

## 八、明确不做

- **不自研实时曲线看板**。多路相机编码、时间轴对齐、拖动回放、内存淘汰这四件事 Rerun 已经解决，自研的产出是一个更差的版本。除非第 4 个实验证明 Web Viewer 扛不住你的相机数量。
- **不 fork、不集成 Foxglove 平台**。Foxglove 适合通用机器人调试，不适合做产品化的采集操作台。
- **不自研遥操作协议**。`lerobot-teleoperate` 与 rollout 策略先够用；真要跨公网做低延迟人工介入，再评估 WebRTC 或直接用 `remote_inference` 的传输层。
- **不在没有 lint 基线前开放多来源数据集合并**。上一轮报告的结论不变：先跑通时间戳与标定快照的完整性校验。

## 九、多采集员与管理员监看

### 9.1 Rerun 代理是广播模型，没有按用户隔离的会话

这是整节的前提。Rerun 官方文档对 `--serve-web` / `--serve-grpc` / `rr.serve_grpc()` 的描述是：代理会把收到的每一条消息转发给每一个已连接的 Viewer，**它没有 per-user session，所有用户看到的是同一条流**；代理还会缓冲历史消息，使得晚连接的 Viewer 能拿到回放，缓冲上限由 `--server-memory-limit` 控制，默认 1 GiB，满了丢最旧的。文档对这种用法的定位是"适合直播演示或共享大屏，**不适合给每个用户各自的数据**"【26】。

因此"管理员切换查看各采集员实时画面"天然成立，而"采集员只看自己那一路"在单代理下**不成立**。Rerun 给出的"每人一份数据"路径是走文件链接或 catalog server 链接，而不是走实时代理【26】。

### 9.2 管理员切换能力的来源

四条事实叠加，管理员场景不需要写一行 Rerun 代码：

其一，LeRobot 的 `init_rerun` 调用 `rr.init(session_name)`，**没有显式指定 recording_id**【4】。Python SDK 的默认 recording_id 基于 `multiprocessing.current_process().authkey`，因此**独立启动**的采集进程各有不同的 recording_id，同一 application_id 下可以并存多个 recording【19】【18】。两台设备因此天然是两个独立 recording，并存于同一个代理之下，而不是互相覆盖——前提是它们不是从同一个 `multiprocessing` 父进程派生的（见 3.3 节）。

其二，`lerobot-record` 传入的 `session_name` 是 `"recording"`，所以所有设备共享同一个 application_id。Rerun 规定 blueprint 按 application_id 存储，**同 app_id 的不同 recording 共享同一份视图布局**【18】。管理员从采集员 A 切到 B 时，相机面板与曲线面板的位置完全一致，不需要重新摆。

其三，Web Viewer 是同一套 UI 编译的 WASM，因此原生 Viewer 的录制选择能力在浏览器里同样存在，而且**可以由代码驱动而不必让操作员去点面板**。`@rerun-io/web-viewer` 暴露的 `WebViewer` 类除 `start` / `open` / `close` / `stop` 之外，还有 `get_state()`（当前打开的录制、时间线与游标）、`get_active_recording_id()` 与 `set_active_recording_id()`（切换当前录制）、`set_location()`、`set_time()` 与 `set_playing()`（驱动时间轴）、`on("recording_open")` 等事件回调、`save_recording()`（把管理员正在看的那条导出为 .rrd）、`take_screenshot()`，以及 `list_commands()` 与 `run_command()`（直接驱动 Viewer 的 UI 命令）【27】。

其四，Rerun 的 Viewer URL 提供两套定位形式：`about:recording?app_id=&recording_id=` 指向 Viewer 已打开的录制，`rerun+http://host/entry/` 指向服务器上的条目【28】。其五，`@rerun-io/web-viewer-react` 提供编程接口：`viewer.start(url | url[] | null, parentElement)` 打开，`viewer.open(url)` 与 `viewer.close(url)` 增删录制，`viewer.stop()` 释放资源【29】。所以"任务列表点某一项 → 右侧面板切到那一路"可以完全自研，不必让操作员去点 Rerun 内部的 UI。

### 9.3 采集员隔离的三条路径

**路径 1：全部连同一个中心代理。** 零开发量，管理员能力完整，但采集员能看到别人的流。只能作为内网 POC，生产环境不可接受。

**路径 2：每台设备一个独立代理，中心做路由表（推荐）。** 设备上跑 `rerun --serve-web`（各占一个端口），LeRobot 用 `--display_ip=127.0.0.1 --display_port=<本机端口>`；中心维护 `任务 ID → 设备代理地址` 的映射。采集员只连自己那台的代理，隔离是网络拓扑保证的，不需要读 Rerun 协议。管理员控制台用 `viewer.open("rerun+http://<设备代理地址>/proxy")` 按需切到任意一路【29】。

这条路还顺手解决了一个更麻烦的问题：**单代理方案需要知道每路的 recording_id 才能在代码里区分，而设备代理方案只需要知道设备地址**。LeRobot 的 `init_rerun` 不返回 recording_id【4】，所以路径 1 想程序化区分各路并不方便，路径 2 则是天然可路由的。

代价是端口分配与路由表要自研，且设备侧要能被中心访问（内网直连，或用 LeRobot 自带的 Zenoh 传输打通跨站点）。

**路径 3：中心自研 gRPC-Web 路由层，按 recording_id 分流。** 唯一能同时做到"单中心代理 + 每人一份视图"的方案，但需要实现 Rerun 的 gRPC-Web 转发与按录制分流，涉及 `re_protos` 与 `x-rerun-*` header 约定，成本最高，除非确有"必须单点接入"的硬约束，否则不建议先行。

### 9.4 带宽的三段拆解

把"带宽"拆成三段，才知道哪一段能被独立代理方案省掉：

| 段 | 单中心代理 | 每设备独立代理 |
|---|---|---|
| 设备 → 网络上行（画面送出） | N 路 | N 路，**没省** |
| 采集员浏览器 ← 设备 | 走中心，N 路下行 | 同 LAN 或本机，**归零** |
| 管理员浏览器 ← 设备 | 走中心，1 路下行（一次看一路） | 1 路下行 |

所以独立代理方案省掉的是**重复消费的那一段**——N 个采集员各自从中心拉一遍全码率视频。上行那一段不省，因为管理员要看实时，设备就得以持续速率把画面送出去。

量级参考：LeRobot 在指定远端 `display_ip` 与 `display_port` 时会自动开启图像压缩【3】，四路 1080p@30fps 的 JPEG 流每路仍在个位数 Mbps，单台设备上行约 8–20 Mbps。十台设备就是 80–200 Mbps 的车队出口。这是"管理员要实时"这一需求本身的成本，不是架构 inefficiency。

### 9.5 降级到"已保存数据"——Rerun 官方就是这么解的

Rerun 官方文档在讲多用户看不同录制时，给出的替代路径正是**文件链接**与 **catalog server 链接**，而不是实时代理【26】。也就是说"管理员看已保存数据"不是妥协，是这条技术路线上的正解。

具体有两种载体：

**载体 A：管理员直接看 LeRobotDataset（推荐）。** Rerun 内置 LeRobot importer，v2.1 与 v3.0 两种目录布局都支持，查看方式有三种：命令行 `rerun path/to/lerobot_dataset`、把目录拖进 Viewer、或 SDK 的 `rr.log_file_from_path(dir)`【30】。它的数据模型映射对管理员极其友好——**每个 episode 变成一个独立 recording（recording id 形如 `episode_1`），因此在 Viewer 里天然可以逐 episode 切换**；每个 feature 变成一个实体，`/observation.state` 是带轴名元数据的 Scalars，`/observation.images.*` 是 EncodedImage，`/task` 是 TextDocument；v3 独有的 `subtask_index` 会落成 `/subtask` 实体上的 TextDocument【30】——这一条对后续标注复用是直接可用的。

**载体 B：转成 .rrd 后挂 catalog server。** `rr.experimental.LeRobotReader` 逐 episode 惰性读取并写出每 episode 一个 `.rrd`，`recording_id` 即 segment id，再用 `rerun server -d my_dataset=rrd` 起本地目录，管理员用 `rerun+http://host/entry/<id>` 定位【31】。代价是多一道转换，且有 ffmpeg 依赖——LeRobot 默认的 AV1 无 B 帧可直接拷贝，H.264/H.265 带 B 帧需要重编码，v3.0 中途起始的 episode 连首个 GOP 都要重编码，ffmpeg 必须在 PATH 上【31】。

建议选 A。管理员要审的就是训练真源，`.rrd` 是二次产物，两者可能不一致；A 还少一道转换。

**载体 A 有一个现成的服务端实现，之前被漏掉了。** `lerobot-dataset-viz` 自带"远端模式"：`--mode distant` 会在数据所在机器上起一个服务，其他机器用 `rerun rerun+http://IP:<grpc-port>/proxy` 连接；它还有 `--web-port`（默认 9090，即 Web Viewer 端口）与 `--grpc-port`（默认 9876），以及 `--display-mode rerun|foxglove`【32】。也就是说"中心管理员看已保存数据"这件事，在 0.6.1 上**不需要自己转 .rrd、不需要写 Rerun 代码**，只要在快照目录上跑一条 `lerobot-dataset-viz --mode distant`，管理员的浏览器指过去即可。

但要注意它与 `lerobot-record` 是**两条独立的可视化分派路径**：dataset-viz 自己做 `--mode local|distant` 的分派、自己连 `rerun+http://IP:9876/proxy`，并不复用 `init_visualization`【32】。参数命名也不同（`--display-mode` 连字符 vs `--display_mode` 下划线）。这两点都要在控制台里各写一个适配函数，不要假设它们共享同一套开关。

### 9.6 快照是硬约束，不是优化

**不能把浏览器指向一个正在写入的 LeRobot 数据集目录。** 数据集的 meta 与 parquet 在采集中持续变动，importer 需要一个静止的目录。因此中心必须提供快照语义：

设备端在 episode 级 checkpoint 之后，把已完成的 episode 增量同步到对象存储或中心；中心为每个任务维护一份只读快照目录；管理员打开的是快照，不是活目录。

这一步与上一轮报告里的 episode 级 checkpoint 是同一件事的两种收益——它既保证"已完成 episode 完整可校验"，也顺带提供了"管理员可读的静止目录"。

### 9.7 同步语义要分三层

"可以接受不同步"这句话需要拆开，否则会掩盖真问题：

1. 实时流与磁盘不同步——正常现象，Rerun 是异步流
2. 管理员看到的比采集员看到的滞后——这是降级的代价，可接受
3. **管理员看到的与最终训练用的数据集不一致**——这才是真问题

第三层的防护是：管理员只读最终数据集的快照，且快照的构造过程与训练读取的是同一份产物。任何"另存一份给管理员看"的旁路都会引入不一致风险。

### 9.8 状态流与视频流分离

一条折中路径能把带宽再压一个数量级：设备 Agent 持续发布低成本状态——episode 计数、实际帧率、掉帧率、标定快照版本、一张 1 fps 的低分辨率缩略图，总量在几十 kbps 量级；管理员默认看到的是任务卡片级别的"进行到哪、是否正常"，点开某一路时才付出视频上行。舵机曲线本身很便宜（6 关节 × 100 Hz 浮点仅几 kbps），可以常驻推送。

这样带宽从"每人每台一条视频流"变成"每人每台一条状态流 + 按需的一条视频流"。

### 9.9 对上游改动的影响

一个之前说重的结论需要修正：**在降级方案下，LeRobot 的 `init_rerun` 完全不需要改。** 设备只需把 `--display_ip` 指向自己的本地代理（`127.0.0.1`），采集员在同 LAN 或本机看，管理员走已保存数据路径，全程单 sink。只有当管理员确实需要实时看某一路时，才需要在设备侧把数据流分叉成"本地 + 远端"两路——那时才需要 `set_sinks()` 与 `get_global_data_recording()` 的改法【33】，且它是一个可选项而非前置条件。

### 9.10 小结

管理员能力：开箱可用，且不需要自研 Rerun 代码。采集员隔离：单代理做不到，需要按设备分代理或自研分流层。带宽：降级到"看已保存数据"即可绕开实时视频上行；坚持实时则需付出车队出口代价，并可选用本地/中心分叉——那是可选项而非前置条件。

## 参考资料

1. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/scripts/lerobot_record.py`（`record_loop` 的 `events` 字典（0.6.1 标注为 `dict`，0.7.0 细化为 `dict[str, bool]`）与 `exit_early` / `rerecord_episode` / `stop_recording`；`init_keyboard_listener()` 在 `record()` 内部自建 events 故无法外部注入；`record()` 会话编排 0.6.1 约一百八十行、0.7.0 约两百行）、`record_loop` 的 events 字典与 exit_early/rerecord_episode/stop_recording、`init_keyboard_listener()` 在 `record()` 内部自建 events——**两版逐字段一致**）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_record.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/scripts/lerobot_record.py
2. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/configs/dataset.py`（`DatasetRecordConfig` 全字段与默认值（push_to_hub=True、reset_time_s=60、streaming_encoding、encoder_queue_maxsize=30、no_stamp）——**两版逐字节相同**）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/dataset.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/dataset.py
3. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/utils/visualization_utils.py`（`VISUALIZATION_MODES = ("rerun","foxglove")` 与 `init_visualization` 的 rerun=远端 / foxglove=绑网卡语义——**两版一致**（0.7.0 仅在日志侧新增 policy prediction））https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/utils/visualization_utils.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/utils/visualization_utils.py
4. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/utils/rerun_visualization.py`（`init_rerun` 行为——`rr.init(session_name)` 不指定 recording_id，ip+port 存在则 `rr.connect_grpc(url="rerun+http://{ip}:{port}/proxy")`，否则 `rr.spawn()`——**两版一致**）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/utils/rerun_visualization.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/utils/rerun_visualization.py
5. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/scripts/lerobot_record.py`（`record_loop` 的 `events` 字典（0.6.1 标注为 `dict`，0.7.0 细化为 `dict[str, bool]`）与 `exit_early` / `rerecord_episode` / `stop_recording`；`init_keyboard_listener()` 在 `record()` 内部自建 events 故无法外部注入；`record()` 会话编排 0.6.1 约一百八十行、0.7.0 约两百行）、`record_loop` 的 events 字典与 exit_early/rerecord_episode/stop_recording、`init_keyboard_listener()` 在 `record()` 内部自建 events——**两版逐字段一致**）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_record.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/scripts/lerobot_record.py
6. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/configs/parser.py`（draccus 多态配置解析；YAML/JSON 配置文件路径；命令行参数覆盖文件中的值——**两版同构**）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/configs/parser.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/configs/parser.py
7. LeRobot v0.6.1 ⇄ v0.7.0 · `pyproject.toml`（`rerun-sdk>=0.24.0,<0.34.0` 与 `foxglove-sdk>=0.25.1,<0.26.0`——**两版 pin 完全相同**；`phone` extra 均含 `fastapi<1.0`；0.7.0 另有 `remote` extra 的 eclipse-zenoh 与 msgpack）https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml https://github.com/huggingface/lerobot/blob/v0.7.0/pyproject.toml
8. LeRobot v0.6.1 与 v0.7.0 的文件面对照（0.6.1 有 `async_inference/` 6 个文件，无 `env_server/`、无 `transport/zenoh.py`、无 `remote_inference/`、无 LanceDB 数据集后端；0.7.0 全部相反新增）https://data.jsdelivr.com/v1/packages/gh/huggingface/lerobot@v0.6.1?structure=flat https://data.jsdelivr.com/v1/packages/gh/huggingface/lerobot@v0.7.0?structure=flat
9. LeRobot v0.7.0 · `src/lerobot/transport/zenoh.py`（`ZenohConfig`、`BoundedSubscriber`、`PresenceEvent` / `PresenceToken`、`PendingQuery`）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/transport/zenoh.py
10. LeRobot v0.7.0 · `src/lerobot/env_server/client.py`（`EnvClient.describe()` / `open()` / `request()` / `prefix()`）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/env_server/client.py
11. LeRobot v0.7.0 · `src/lerobot/env_server/server.py`（`EnvServerConfig` 的 `max_sessions` / `max_envs`、`_Session` 的 generation 防陈旧与失败锁死）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/env_server/server.py
12. LeRobot v0.7.0 · `src/lerobot/env_server/protocol.py`（`PROTOCOL_VERSION = 1`、键前缀层级、JPEG 回复、32 MiB 载荷上限）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/env_server/protocol.py
13. Rerun · ARCHITECTURE.md（Web Viewer 编译为 WASM，`re_web_viewer_server` 经 HTTP 提供 HTML/wasm/JS 桥接）https://github.com/rerun-io/rerun/blob/main/ARCHITECTURE.md
14. Rerun · Use multiple native viewers（Web Viewer 通过 gRPC-Web over HTTP 连接 gRPC 服务器）https://rerun.io/docs/howto/visualization/multiple-viewers
15. Rerun · How does Rerun work?（Web Viewer 为 32 位 Wasm、约 2 GiB 内存、单线程、慢于原生；WebGPU 回退 WebGL；`web-viewer` 与 `web-viewer-react` NPM 包）https://rerun.io/docs/pr-3109/concepts/how-does-rerun-work
16. Rerun · CLI reference（`--serve-web`、`--serve-grpc`、`--connect`、`--cors-allow-origin`、`--bind`、`--memory-limit`、`--server-memory-limit`）https://rerun.io/docs/reference/cli
17. Rerun · Operating Modes（`rr.serve_grpc()`、`rr.serve_web_viewer(connect_to=...)`、`rr.init(application_id)` 的作用）https://rerun.io/docs/0.28.1/reference/sdk/operating-modes
18. Rerun · Recordings（application_id 与 recording_id 的职责划分；每次启动生成随机 recording_id；同 application_id 的不同 recording 共享 blueprint）https://rerun.io/docs/0.28.0/concepts/apps-and-recordings
19. Rerun SDK 0.28.1 · `rerun_py/rerun_sdk/rerun/recording_stream.py` · `get_recording_id()`（默认 recording_id 基于 `multiprocessing.current_process().authkey`，`multiprocessing` 派生的子进程共享同一默认值；源码 TODO(#3793) 指出该默认行为应改为可选）https://github.com/rerun-io/rerun/tree/0.28.1/rerun_py/rerun_sdk/rerun
20. LeRobot v0.7.0 · `src/lerobot/scripts/lerobot_annotate.py`（`--root` / `--repo_id` → `--new_repo_id`，推送、生成卡片、打 tag）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/scripts/lerobot_annotate.py
21. LeRobot v0.7.0 · `src/lerobot/jobs/annotate.py`（`submit_annotate_to_hf()`、vLLM pod、`--job.detach` / `--job.target=local`）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/jobs/annotate.py
22. Rerun · GitHub 仓库（摄取格式含 LeRobot，`re_lerobot` crate 负责 LeRobot 数据集转 chunk；Apache-2.0 与 MIT 双许可）https://github.com/rerun-io/rerun
23. LeRobot v0.6.1 · `src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py` 与 `config_rebot_b601_follower.py`（`gripper_mit_kp` / `gripper_mit_kd` 在两个文件中各存在 1 处）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/robots/rebot_b601_follower/config_rebot_b601_follower.py
24. LeRobot v0.7.0 · `src/lerobot/robots/rebot_b601_follower/`（`rebot_b601_follower.py` 与 `config_rebot_b601_follower.py` 中 `gripper_mit_kp` / `gripper_mit_kd` 命中数为 0；新增 `motor_family.py` 定义 `MotorFamily` = DM / RS 与 `ArmControlMode` = MIT / POS_VEL）https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/rebot_b601_follower.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/config_rebot_b601_follower.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/robots/rebot_b601_follower/motor_family.py
25. LeRobot v0.6.1 ⇄ v0.7.0 · `src/lerobot/motors/` 与 `pyproject.toml`（v0.7.0 新增 `rustypot-dep = ["rustypot>=1.13.1,<2.0.0"]` 与 `serial-motors` extra，feetech / dynamixel extra 改指 rustypot，注释称 "rustypot drives Feetech and Dynamixel motors"；`motors_bus.py` 由自研 pyserial 协议层改走 rustypot；`feetech.py` 19643→1969 字节、`dynamixel.py` 11337→1898 字节；`feetech/tables.py`、`dynamixel/tables.py`、`encoding_utils.py` 在 v0.7.0 中不存在）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/motors_bus.py https://github.com/huggingface/lerobot/blob/v0.7.0/src/lerobot/motors/motors_bus.py https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml https://github.com/huggingface/lerobot/blob/v0.7.0/pyproject.toml
26. Rerun · Share recordings with many users through the web Viewer（代理广播、无 per-user session、`--server-memory-limit` 默认 1 GiB；文件链接与 catalog server 两条替代路径；`rerun server --cors-allow-origin`）https://rerun.io/docs/howto/integrations/host-web-viewer
27. Rerun · Web Viewer API 参考（`WebViewer` 的 `get_state` / `get_active_recording_id` / `set_active_recording_id` / `set_time` / `set_playing` / `on("recording_open")` / `save_recording` / `take_screenshot` / `list_commands` / `run_command` / `open` / `close` / `start` / `stop`）https://ref.rerun.io/docs/js/web-viewer/classes/WebViewer
28. Rerun · Viewer URLs（`about:recording?app_id=&recording_id=`、`rerun+http://host/entry/`、web viewer 的 `?url=` 参数形式）https://rerun.io/docs/reference/viewer/urls
29. Rerun · Embed Rerun in Web pages（`@rerun-io/web-viewer` 与 `@rerun-io/web-viewer-react`；`viewer.start()` / `viewer.open()` / `viewer.close()` / `viewer.stop()`；iframe 形式）https://rerun.io/docs/howto/embed-rerun-viewer
30. Rerun · Working with LeRobot datasets（v2.1 与 v3.0 布局均支持；`rerun <dir>` / 拖放 / `rr.log_file_from_path`；每 episode 一个独立 recording；feature 到实体的映射含 `/task` 与 `/subtask`；时间轴由 `frame_index` 或 `timestamp` 推导；数据集 fps 不存入录制，需从 `meta/info.json` 换算）https://rerun.io/docs/howto/logging-and-ingestion/lerobot
31. Rerun · rerun-lerobot 技能说明（`rr.experimental.LeRobotReader` 逐 episode 转 .rrd；`recording_id` 即 segment id；episode id 需零填充；`rerun server -d my_dataset=rrd`；AV1 可直拷而 H.264/H.265 需 ffmpeg 重编码）https://github.com/rerun-io/rerun/tree/main/skills/rerun-lerobot
32. LeRobot v0.6.1 · `src/lerobot/scripts/lerobot_dataset_viz.py`（`--mode local|distant`、`--web-port` 默认 9090 即 Web Viewer 端口、`--grpc-port` 默认 9876、`--display-mode rerun|foxglove`；自带独立分派路径，不复用 `init_visualization`）https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/scripts/lerobot_dataset_viz.py
33. Rerun SDK 0.28.1 · `rerun_py/rerun_sdk/rerun/recording_stream.py` 与 `rerun/__init__.py`（`RecordingStream.set_sinks()` / `connect_grpc()` / `spawn()`；导出 `get_global_data_recording`）https://github.com/rerun-io/rerun/tree/0.28.1/rerun_py/rerun_sdk/rerun
