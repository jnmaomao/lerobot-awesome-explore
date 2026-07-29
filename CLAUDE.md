# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. This fork adds support for **NexArm 6-DOF robot arms**.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
# Install with uv (recommended for contributors)
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything

# Or install with pip (for NexArm users)
pip install -e ".[nexarm]"                  # NexArm support
pip install -e ".[nexarm,viz]"              # With visualization
pip install -e ".[nexarm,training]"         # With training deps

# Pull test artifacts
git lfs install && git lfs pull
```

## Key Commands

```bash
# Testing
uv run pytest tests -svv --maxfail=10       # All tests
DEVICE=cuda make test-end-to-end            # E2E tests (GPU)
pytest -sv tests/test_specific.py           # Single test file

# Linting & Formatting
pre-commit run --all-files                  # Ruff, typos, bandit, etc.
pre-commit install                          # Auto-run on commit

# CLI Entry Points (mapped in pyproject.toml [project.scripts])
lerobot-train --policy.type=act --dataset.repo_id=...
lerobot-eval --policy.path=... --env.type=...
lerobot-record --robot.type=nexarm_follower ...
lerobot-teleoperate --robot.type=nexarm_follower --teleop.type=nexarm_leader ...
lerobot-find-port                           # Discover USB ports
lerobot-find-cameras opencv                 # Discover cameras
lerobot-find-lamp-port                      # Discover LED lamp serial port
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.). Each script has a `main()` function called via pyproject.toml entry points.
- **`configs/`** — Dataclass configs parsed by draccus. `TrainPipelineConfig` is the top-level config for training. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy (ACT, Diffusion, SmolVLA, etc.) in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers. Each robot/teleop has its own subdir with config + implementation.
- **`rollout/`** — Strategy patterns for inference (`base`, `sentry`, `highlight`, `dagger`).
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.
- **`utils/status_lamp.py`** — Recording status lamp (ESP32 + NeoPixel ring). Thread-safe, non-blocking serial I/O. Active when `--lamp_port` is set.
- **`utils/find_lamp_port.py`** — Hot-plug detection for the LED lamp's serial port. CLI: `lerobot-find-lamp-port`.

## NexArm-Specific Architecture

This fork adds NexArm 6-DOF robot arm support:

```
src/lerobot/
├── motors/nexarm/           # Serial protocol driver (CommProtocol)
│   └── nexarm.py            # Frame building/parsing, position read/write, torque control
├── robots/nexarm_follower/  # Follower arm (executes actions)
│   ├── config_nexarm_follower.py
│   └── nexarm_follower.py
└── teleoperators/nexarm_leader/  # Leader arm (teleoperation)
    ├── config_nexarm_leader.py
    └── nexarm_leader.py
```

**Key files modified for NexArm:**
- `src/lerobot/robots/utils.py` — Added `nexarm_follower` branch in `make_robot_from_config()`
- `src/lerobot/teleoperators/utils.py` — Added `nexarm_leader` branch in `make_teleoperator_from_config()`
- `src/lerobot/scripts/*.py` — Added imports for NexArm classes

**NexArm configuration files:** `examples/nexarm/{teleoperate,record,inference,train}.yaml`

## Config System (draccus)

Configs are Python dataclasses. CLI args override config fields. Example:

```bash
# Use config file
lerobot-train --config_path=examples/nexarm/train.yaml

# Override via CLI
lerobot-train --dataset.repo_id=local/my_dataset --policy.type=act --steps=50000
```

**Important:** For training, prefer direct CLI args over `--config_path` + overrides to avoid JSON parsing issues.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml`, `full_tests.yml`, `security.yml`, `release.yml`.
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides.
- **`examples/`** — End-user configs organized by robot (e.g., `examples/nexarm/`).
- **`docker/`** — Dockerfiles for benchmarks (LIBERO, MetaWorld, RoboCasa, etc.).
- **`benchmarks/`** — Performance benchmarking scripts.

## Coding Conventions

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy.
- **Video decoding**: datasets store observations as video files. Tests need ffmpeg installed.
- **Prioritize `uv run`** to execute Python commands (not raw `python` or `pip`).

## Policy Training Guidelines

- **Epochs vs Steps**: Think in epochs first (5–10 typical), then convert to steps.
  ```
  total_frames = episodes × fps × episode_time_s
  steps_per_epoch = ceil(total_frames / batch_size)
  total_steps = epochs × steps_per_epoch
  ```
- **Scale scheduler with steps**: Set `--policy.scheduler_decay_steps` ≈ `--steps` for short runs.
- **Batch size**: Prefer larger batches. Use gradient accumulation if GPU memory limited.
- **SmolVLA**: Unfreeze vision encoder (`--policy.freeze_vision_encoder=false`) for better performance on specialized tasks.

## Common Workflows

### Adding a new robot

1. Create `src/lerobot/robots/my_robot/` with `config_my_robot.py` and `my_robot.py`
2. Inherit from `Robot` class, implement `connect()`, `disconnect()`, `get_observation()`, `send_action()`
3. Add to `src/lerobot/robots/utils.py` factory function
4. Add imports to relevant scripts (`lerobot_record.py`, `lerobot_teleoperate.py`, etc.)
5. Create example configs in `examples/my_robot/`

### Adding a new policy

1. Create `src/lerobot/policies/my_policy/` with config and implementation
2. Inherit from `PreTrainedPolicy`, implement `forward()`, `select_action()`, `reset()`
3. Register with `@register_subclass("my_policy")` decorator on config class
4. Add to `src/lerobot/policies/factory.py`
5. Add optional dependencies to `pyproject.toml` if needed

## Further Reference

- **User-facing guide**: [`AGENT_GUIDE.md`](./AGENT_GUIDE.md) — SO-101 setup, data collection, training, evaluation
- **NexArm guide**: [`README_NEXARM.md`](./README_NEXARM.md) — Complete NexArm usage from hardware to deployment
- **Contributor guide**: [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- **AI usage policy**: [`AI_POLICY.md`](./AI_POLICY.md)
