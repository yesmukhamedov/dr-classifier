#!/usr/bin/env python3
"""Deep-merge YAML configs into a single file for Kaggle runs.

``run_experiment.py`` (and the helper scripts) accept only a single
``--config``. Rather than patch every CLI to support layered configs, this
helper merges ``default.yaml`` with one or more override files (later files
win) plus optional dotted ``KEY=VALUE`` overrides from the command line, and
writes the result to a single merged YAML that every command then consumes.

Usage:
    python kaggle/merge_config.py \
        --base configs/default.yaml \
        --override kaggle/kaggle_paths.yaml \
        --out configs/kaggle_merged.yaml \
        paths.eyepacs=/kaggle/working/eyepacs

Dotted overrides are parsed as YAML scalars, so
``training.batch_size=8`` becomes an int and ``subset.enabled=false`` a bool.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``; later values win.

    Args:
        base: Base mapping (not mutated).
        override: Mapping whose values take precedence.

    Returns:
        A new merged dict. Nested dicts are merged key-by-key; any other
        value type in ``override`` replaces the base value outright.
    """
    import copy

    merged = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def apply_dotted_override(config: dict[str, Any], dotted: str) -> None:
    """Apply a single ``a.b.c=value`` override in place.

    The value is parsed as a YAML scalar so types are preserved
    (``8`` → int, ``false`` → bool, ``null`` → None, ``1.5`` → float).

    Args:
        config: Config dict to mutate.
        dotted: String of the form ``dotted.key=value``.

    Raises:
        ValueError: If ``dotted`` does not contain ``=``.
    """
    if "=" not in dotted:
        raise ValueError(f"Override '{dotted}' must be of the form key.path=value")
    key_path, raw_value = dotted.split("=", 1)
    value = yaml.safe_load(raw_value)

    keys = key_path.split(".")
    node: dict[str, Any] = config
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep-merge YAML configs.")
    parser.add_argument("--base", required=True, type=Path, help="Base config YAML.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        type=Path,
        help="Override YAML(s); may be repeated. Later files win.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output merged YAML path.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional dotted overrides, e.g. paths.eyepacs=/kaggle/working/eyepacs",
    )
    args = parser.parse_args()

    with open(args.base, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    for override_path in args.override:
        with open(override_path, "r") as f:
            override = yaml.safe_load(f) or {}
        config = deep_merge(config, override)

    for dotted in args.overrides:
        apply_dotted_override(config, dotted)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Merged config written to {args.out}")
    print(f"  paths.eyepacs    = {config.get('paths', {}).get('eyepacs')}")
    print(f"  paths.output_dir = {config.get('paths', {}).get('output_dir')}")
    print(f"  training.batch_size  = {config.get('training', {}).get('batch_size')}")
    print(f"  training.num_workers = {config.get('training', {}).get('num_workers')}")


if __name__ == "__main__":
    main()
