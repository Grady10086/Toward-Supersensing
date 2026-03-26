#!/usr/bin/env python3
"""Unified Cambrian-W evaluation wrapper for this release."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DATA_DIR = REPO_ROOT / "lmms_eval" / "tasks" / "cambw" / "data"

MODEL_CONFIGS = {
    "qwen3_vl_8b": {"model": "qwen3_vl", "model_args": "pretrained=Qwen/Qwen3-VL-8B-Instruct"},
    "qwen2_vl_7b": {"model": "qwen2_vl", "model_args": "pretrained=Qwen/Qwen2-VL-7B-Instruct"},
    "qwen2_5_vl_7b": {"model": "qwen_vl", "model_args": "pretrained=Qwen/Qwen2.5-VL-7B-Instruct"},
    "llava_onevision_7b": {"model": "llava_onevision", "model_args": "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov"},
    "internvl2_5_8b": {"model": "internvl2", "model_args": "pretrained=OpenGVLab/InternVL2_5-8B"},
}

TASK_MAP = {
    "part1": "cambw_part1_long",
    "part2_3": "cambw_part2_3_short",
}


def require_path(value: str | None, env_name: str) -> str:
    if value:
        return value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    raise SystemExit(f"Missing required path. Provide the CLI flag or set {env_name}.")


def ensure_data_prepared(part: str, benchmark_dir: str | None, video_root: str | None) -> None:
    jsonl_file = TASK_DATA_DIR / ("part1_long.jsonl" if part == "part1" else "part2_3_short.jsonl")
    if jsonl_file.exists():
        print(f"[INFO] task data already exists: {jsonl_file}")
        return

    prepare_script = REPO_ROOT / "tools" / "cambw" / "prepare_data.py"
    cmd = [
        sys.executable,
        str(prepare_script),
        "--benchmark-dir",
        require_path(benchmark_dir, "CAMBW_BENCH_DIR"),
        "--data-root",
        require_path(video_root, "CAMBW_VIDEO_ROOT"),
        "--out-dir",
        str(TASK_DATA_DIR),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def resolve_model(model: str, model_args: str | None) -> tuple[str, str]:
    if model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model]
        return config["model"], config["model_args"] if model_args is None else model_args
    if model_args is None:
        raise SystemExit(f"Unknown model '{model}'. Pass --model_args explicitly for arbitrary lmms-eval adapters.")
    return model, model_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["part1", "part2_3", "all"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default=None)
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_num_frames", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()

    parts = ["part1", "part2_3"] if args.part == "all" else [args.part]
    lmms_model, lmms_model_args = resolve_model(args.model, args.model_args)

    extra_args = []
    if args.fps is not None:
        extra_args.append(f"fps={args.fps}")
    if args.max_num_frames is not None:
        extra_args.append(f"max_num_frames={args.max_num_frames}")
    if extra_args:
        lmms_model_args = lmms_model_args + "," + ",".join(extra_args) if lmms_model_args else ",".join(extra_args)

    task_names = []
    for part in parts:
        ensure_data_prepared(part, args.benchmark_dir, args.video_root)
        task_names.append(TASK_MAP[part])

    output_path = args.output_path
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(REPO_ROOT / "results" / f"{'_'.join(parts)}_{args.model}_{stamp}")

    cmd = [
        sys.executable,
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        lmms_model,
        "--model_args",
        lmms_model_args,
        "--tasks",
        ",".join(task_names),
        "--batch_size",
        str(args.batch_size),
        "--output_path",
        output_path,
        "--log_samples",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
