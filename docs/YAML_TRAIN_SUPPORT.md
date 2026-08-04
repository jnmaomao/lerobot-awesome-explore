# YAML 配置训练支持

## 问题背景

`lerobot-train` 原本只支持 JSON 配置文件，而 `lerobot-record` 支持 YAML。为了统一配置格式，我们添加了 YAML 配置支持。

## 使用方法

### 方式 1：直接使用 YAML 配置训练

```bash
python -m lerobot.scripts.lerobot_train_yaml --config_path=examples/nexarm_act/train.yaml
```

### 方式 2：YAML 配置 + CLI 参数覆盖

```bash
python -m lerobot.scripts.lerobot_train_yaml \
  --config_path=examples/nexarm_act/train.yaml \
  --batch_size=16 \
  --steps=50000
```

### 方式 3：预览转换后的 JSON 配置（dry-run）

```bash
python -m lerobot.scripts.lerobot_train_yaml \
  --config_path=examples/nexarm_act/train.yaml \
  --dry-run
```

## 实现原理

1. 读取 YAML 配置文件
2. 自动转换为临时 JSON 文件
3. 调用原始 `lerobot-train` 命令
4. 训练完成后自动清理临时文件

## 配置示例

### YAML 配置（推荐）

`examples/nexarm_act/train.yaml`:
```yaml
dataset:
  repo_id: local/nexarm_merged
  root: /path/to/dataset

policy:
  type: act

output_dir: outputs/train/nexarm_act
job_name: nexarm_act
batch_size: 8
steps: 100000
```

### JSON 配置（仍然支持）

```bash
python -m lerobot.scripts.lerobot_train_yaml --config_path=examples/nexarm_act/train.json
```

## CLI 参数格式支持

支持两种格式：

```bash
# 格式 1: --key=value
--batch_size=16

# 格式 2: --key value
--batch_size 16
```

## 类型自动转换

CLI 参数会自动转换为正确的类型：

- `true/false` → 布尔值
- `123` → 整数
- `3.14` → 浮点数
- `[1,2,3]` → 列表（JSON 格式）
- 其他 → 字符串

## 与 record 命令对比

| 命令 | 配置格式 | 说明 |
|------|---------|------|
| `lerobot-record` | YAML/JSON | 原生支持 YAML |
| `lerobot-train` | JSON only | 仅支持 JSON |
| `lerobot_train_yaml` | YAML/JSON | **新增：支持 YAML** |

## 注意事项

1. 需要 `PyYAML` 依赖：`pip install PyYAML`
2. 临时 JSON 文件在训练完成后自动删除
3. CLI 参数优先级高于 YAML 配置
