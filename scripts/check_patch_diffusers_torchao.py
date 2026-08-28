#!/usr/bin/env python3
"""
检测并修复 diffusers + torchao 的 logger 加载顺序 bug。

Bug 背景
--------
diffusers 0.35.2 / 0.36.0 的 `quantizers/torchao/torchao_quantizer.py` 把

    logger = logging.get_logger(__name__)

定义在模块级调用 `_update_torch_safe_globals()` **之后**。当 torchao >= 0.18
移除了 `NF4Tensor` 时，`_update_torch_safe_globals` 内的 `try/except` 走到
except 分支，而该分支引用 `logger` —— 此时 `logger` 尚未定义，于是抛

    NameError: name 'logger' is not defined

这一 NameError 在 `import diffusers` 时就触发，进而通过
`lerobot.policies.factory → groot → diffusers` 链让**任何** `import lerobot.policies`
都崩，不只是 Pi0。

上游状态（2026-08 核实）
------------------------
已在 diffusers >= 0.37.0 修复：
  commit 1cdb8723b8 "fix torchao quantizer for new torchao versions (#12901)"
  2025-12-30 合入。修法 = 把 `logger` 提前到模块顶部，与本脚本所打补丁一致。
  版本勘误：0.36.0 仍带 bug；首个修复版本是 0.37.0。

本脚本做的事
------------
1. 定位环境内的 `torchao_quantizer.py`（按 `import diffusers` 解析真实路径，
   兼容 venv / conda / editable install）。
2. 检测是否需要打补丁：`logger =` 是否出现在模块级
   `_update_torch_safe_globals()` 调用**之前**。已提前则视为无需补丁
   （含 diffusers >= 0.37.0 的官方修复，以及本脚本此前打过的补丁）。
3. 需要时：先备份为 `.bak_<时间戳>`（不覆盖已存在的备份），再在
   `from ..base import DiffusersQuantizer` 之后插入 `logger` 提前定义。
4. 幂等：重复运行不会重复插入；已打补丁或上游已修时直接退出。

用法
----
    python scripts/check_patch_diffusers_torchao.py          # 仅检测，不改文件
    python scripts/check_patch_diffusers_torchao.py --apply   # 检测并打补丁
    python scripts/check_patch_diffusers_torchao.py --apply --force  # 跳过已备份检查

退出码：0 = 无需补丁 / 已修复；1 = 需要补丁（--apply 后为已修复）；2 = 运行出错。

注意：本脚本只改环境内第三方包文件，不改仓库源码。属环境级补丁，与
`README_NEXARM.md`「Pi0 int8 量化部署 → 环境搭建关键点」对应。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import sys
import traceback
from pathlib import Path

PATCH_MARKER = "env-patch — diffusers"
TARGET_FILENAME = "torchao_quantizer.py"
TARGET_REL = Path("quantizers") / "torchao" / TARGET_FILENAME

# 提前插入的 logger 定义块（与上游 1cdb8723b8 修法一致）。
LOGGER_BLOCK = (
    "from ..base import DiffusersQuantizer\n\n"
    "# NOTE: env-patch — diffusers defines `logger` AFTER `_update_torch_safe_globals()`\n"
    "# is invoked at module load. When torchao>=0.18 removes NF4Tensor, the except branch\n"
    "# runs before `logger` exists and raises NameError, breaking any import of diffusers.\n"
    "# Define it here (early) so the except branch is safe. The late definition below is a\n"
    "# harmless reassignment.\n"
    "logger = logging.get_logger(__name__)\n"
)


def locate_target() -> Path | None:
    """返回环境内 diffusers 的 torchao_quantizer.py 真实路径。"""
    try:
        import diffusers  # noqa: F401
    except Exception as e:
        print(f"[ERR] 无法 import diffusers：{e}", file=sys.stderr)
        return None
    # diffusers 是命名空间包；用 __file__ 不可靠，改用 spec.submodule_search_locations。
    import diffusers as _d

    roots = []
    spec = getattr(_d, "__spec__", None)
    if spec is not None:
        locs = getattr(spec, "submodule_search_locations", None) or []
        roots.extend(locs)
    if not roots and getattr(_d, "__file__", None):
        roots.append(str(Path(_d.__file__).resolve().parent))
    for root in roots:
        cand = Path(root) / TARGET_REL
        if cand.is_file():
            return cand.resolve()
    return None


def find_line(text: str, pattern: str) -> int | None:
    """返回第一个匹配 pattern 的行号（1-based），无则 None。"""
    for i, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return i
    return None


def needs_patch(src: str) -> tuple[bool, str]:
    """判断是否需要打补丁。

    需要补丁的条件：模块级 `_update_torch_safe_globals()` 调用存在，
    且 `logger = logging.get_logger` 定义在其**之后**（或文件里完全没有 logger 定义）。
    已打补丁（含本脚本标记或上游 0.37+）则返回 False。
    """
    # 本脚本此前打过补丁的标记 —— 直接判为已修复。
    if PATCH_MARKER in src:
        return False, "已存在本脚本的补丁标记（env-patch），无需再打。"

    call_line = find_line(src, "_update_torch_safe_globals()")
    # 排除函数定义行（def _update_torch_safe_globals()），只要模块级裸调用。
    # 裸调用形如 4 空格缩进或不缩进的 `_update_torch_safe_globals()`。
    module_call_re = re.compile(r"^\s{0,4}_update_torch_safe_globals\(\)\s*$", re.M)
    mcall = module_call_re.search(src)
    if not mcall:
        return False, "未找到模块级 `_update_torch_safe_globals()` 调用，文件结构与已知 bug 不符，跳过。"

    logger_lines = [i for i, ln in enumerate(src.splitlines(), start=1)
                    if "logger = logging.get_logger" in ln]
    if not logger_lines:
        # 没有 logger 定义——无法用本补丁修复，可能是重写过的新版本。
        return False, "文件中未找到 `logger = logging.get_logger` 定义，跳过（可能版本已重构）。"

    call_lineno = src[:mcall.start()].count("\n") + 1
    first_logger = logger_lines[0]
    if first_logger < call_lineno:
        return False, (f"logger 已定义在调用之前 (logger@L{first_logger} < call@L{call_lineno})，"
                       f"上游已修复（diffusers >= 0.37.0），无需补丁。")
    return True, (f"需要补丁：logger@L{first_logger} 在调用@L{call_lineno} 之后，"
                  f"torchao>=0.18 会触发 except 分支 NameError。")


def apply_patch(target: Path, src: str, force: bool) -> str:
    """执行补丁，返回人类可读的结果说明。"""
    anchor = "from ..base import DiffusersQuantizer\n"
    idx = src.find(anchor)
    if idx == -1:
        raise RuntimeError("未找到锚点 `from ..base import DiffusersQuantizer`，无法安全插入补丁。")

    # 备份（不覆盖已有备份）。
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = target.with_name(f"{target.name}.bak_{ts}")
    if bak.exists() and not force:
        # 复用最近一个备份而不覆盖。
        existing = sorted(target.parent.glob(f"{target.name}.bak_*"))
        bak = existing[-1]
        bak_note = f"（已有备份 {bak.name}，未重复备份）"
    else:
        shutil.copy2(target, bak)
        bak_note = f"（已备份为 {bak.name}）"

    # 在锚点之后插入 logger 块。锚点行末已有 \n，直接拼接。
    new_src = src[:idx] + LOGGER_BLOCK + src[idx + len(anchor):]
    # 注意：原锚点 `from ..base import DiffusersQuantizer\n` 被 LOGGER_BLOCK 末尾
    # 的 `from ..base import DiffusersQuantizer\n` 替代，保持语义不变。
    target.write_text(new_src)
    return f"已打补丁{bak_note}。"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="检测到需要补丁时自动打补丁（默认仅检测）")
    ap.add_argument("--force", action="store_true", help="即使已有备份也允许新建备份（配合 --apply）")
    args = ap.parse_args()

    target = locate_target()
    if target is None:
        print("[ERR] 找不到 diffusers 的 torchao_quantizer.py。请确认 diffusers 已安装。", file=sys.stderr)
        return 2

    print(f"[INFO] 目标文件: {target}")
    try:
        src = target.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERR] 读取失败: {e}", file=sys.stderr)
        return 2

    try:
        need, reason = needs_patch(src)
    except Exception:
        print("[ERR] 检测时异常:\n" + traceback.format_exc(), file=sys.stderr)
        return 2

    print(f"[INFO] {reason}")

    if not need:
        print("[OK] 无需补丁。")
        return 0

    if not args.apply:
        print("[HINT] 需要补丁但未加 --apply，仅检测不改文件。加 --apply 自动修复。")
        return 1

    try:
        msg = apply_patch(target, src, args.force)
    except Exception as e:
        print(f"[ERR] 打补丁失败: {e}\n" + traceback.format_exc(), file=sys.stderr)
        return 2
    print(f"[DONE] {msg}")

    # 复检
    src2 = target.read_text(encoding="utf-8")
    need2, reason2 = needs_patch(src2)
    if need2:
        print(f"[WARN] 补丁后复检仍判定需要补丁：{reason2}", file=sys.stderr)
        return 2
    print(f"[OK] 复检通过：{reason2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
