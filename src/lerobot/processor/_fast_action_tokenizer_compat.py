#!/usr/bin/env python
#
# Compatibility shim for loading the FAST action tokenizer under transformers 5.x.
#
# This file is a local patch (not part of upstream lerobot). It exists so that
# `ActionTokenizerProcessorStep` can keep its original `AutoProcessor.from_pretrained`
# call byte-for-byte identical (minimising merge conflicts with future upstream
# updates) and only fall back to this manual construction when the standard loader
# fails.
#
# Background:
#   transformers 5.x `ProcessorMixin.from_pretrained` recursively loads the
#   sub-components declared in `attributes` via `AutoTokenizer.from_pretrained`,
#   which requires a `tokenizer.json` to be present in the repo. The
#   `physical-intelligence/fast` repo only ships `processing_action_tokenizer.py`
#   and `processor_config.json` (the BPE vocab is injected after training), so the
#   standard loader fails. `lerobot/fast-action-tokenizer` ships the trained
#   `bpe_tokenizer/tokenizer.json`. Here we build `UniversalActionProcessor`
#   manually from those files.
#
# Behaviour:
#   * Prefer the repo named by `action_tokenizer_name` (respects user config).
#   * If that repo has no `bpe_tokenizer/tokenizer.json`, fall back to the
#     official pre-trained `lerobot/fast-action-tokenizer`.

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any


def load_fast_action_processor(
    action_tokenizer_name: str, trust_remote_code: bool = True
) -> Any:
    """Build a ``UniversalActionProcessor`` manually for transformers 5.x.

    Args:
        action_tokenizer_name: Hub repo id configured by the user. Used first;
            falls back to ``lerobot/fast-action-tokenizer`` if it lacks a trained
            BPE vocab.
        trust_remote_code: Kept for API symmetry; the remote code is always
            executed here by construction.

    Returns:
        An instantiated ``UniversalActionProcessor``.
    """
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    # Prefer the configured repo; fall back to the official pre-trained one if it
    # doesn't ship a bpe_tokenizer/tokenizer.json (e.g. physical-intelligence/fast).
    repo = action_tokenizer_name
    snapshot_path = snapshot_download(repo)
    bpe_json = os.path.join(snapshot_path, "bpe_tokenizer", "tokenizer.json")
    if not os.path.exists(bpe_json):
        repo = "lerobot/fast-action-tokenizer"
        snapshot_path = snapshot_download(repo)
        bpe_json = os.path.join(snapshot_path, "bpe_tokenizer", "tokenizer.json")

    # Load the remote processing_action_tokenizer.py defining UniversalActionProcessor.
    spec = importlib.util.spec_from_file_location(
        "processing_action_tokenizer",
        os.path.join(snapshot_path, "processing_action_tokenizer.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with open(os.path.join(snapshot_path, "processor_config.json")) as f:
        cfg = json.load(f)

    tokenizer_obj = Tokenizer.from_file(bpe_json)
    bpe = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj, clean_up_tokenization_spaces=False
    )
    return module.UniversalActionProcessor(
        bpe_tokenizer=bpe,
        scale=cfg.get("scale", 10),
        vocab_size=cfg.get("vocab_size", 2048),
        min_token=cfg.get("min_token", -354),
    )
