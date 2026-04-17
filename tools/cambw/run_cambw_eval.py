#!/usr/bin/env python3
"""Unified Cambrian-W evaluation wrapper for this release."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DATA_DIR = REPO_ROOT / "lmms_eval" / "tasks" / "cambw" / "data"
BENCHMARK_VERSION_ROOT = REPO_ROOT / "lmms_eval" / "tasks" / "cambw" / "benchmarks"

MODEL_CONFIGS = {
    "qwen3_vl_8b": {"model": "qwen3_vl", "model_args": "pretrained=Qwen/Qwen3-VL-8B-Instruct"},
    "qwen2_vl_7b": {"model": "qwen2_vl", "model_args": "pretrained=Qwen/Qwen2-VL-7B-Instruct"},
    "qwen2_5_vl_7b": {"model": "qwen2_5_vl", "model_args": "pretrained=Qwen/Qwen2.5-VL-7B-Instruct"},
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


def materialize_data_dir(source_dir: Path) -> None:
    TASK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in ["part1_long.jsonl", "part2_3_short.jsonl"]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, TASK_DATA_DIR / name)
            print(f"[INFO] materialized benchmark data: {src} -> {TASK_DATA_DIR / name}")


def jsonl_has_video_path(data_dir: Path) -> bool:
    for name in ["part1_long.jsonl", "part2_3_short.jsonl"]:
        path = data_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    return False
                return bool(item.get("video_path"))
    return False


def materialize_bench_version(bench_version: str, video_root: str | None) -> None:
    bench_root = BENCHMARK_VERSION_ROOT / bench_version
    data_dir = bench_root / "data"
    if data_dir.exists() and jsonl_has_video_path(data_dir):
        materialize_data_dir(data_dir)
        return

    source_dir = bench_root / "source"
    if not source_dir.exists():
        raise SystemExit(f"Benchmark version '{bench_version}' has no usable data/ or source/ directory.")

    prepare_script = REPO_ROOT / "tools" / "cambw" / "prepare_data.py"
    cmd = [
        sys.executable,
        str(prepare_script),
        "--benchmark-dir",
        str(source_dir),
        "--data-root",
        require_path(video_root, "CAMBW_VIDEO_ROOT"),
        "--out-dir",
        str(TASK_DATA_DIR),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


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
    parser.add_argument("--bench-version", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_num_frames", type=int, default=None)
    parser.add_argument("--video_protocol", choices=["full_fps1", "uniform_max_frames"], default="full_fps1")
    parser.add_argument("--protocol_fps", type=float, default=1.0)
    parser.add_argument("--protocol_max_frames", type=int, default=128)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--force_simple", action="store_true")
    args = parser.parse_args()

    parts = ["part1", "part2_3"] if args.part == "all" else [args.part]
    lmms_model, lmms_model_args = resolve_model(args.model, args.model_args)

    if args.data_dir:
        materialize_data_dir(Path(args.data_dir))
    elif args.bench_version:
        materialize_bench_version(args.bench_version, args.video_root)

    extra_args = []
    if args.fps is not None:
        extra_args.append(f"fps={args.fps}")
    if args.max_num_frames is not None:
        extra_args.append(f"max_num_frames={args.max_num_frames}")
    if extra_args:
        lmms_model_args = lmms_model_args + "," + ",".join(extra_args) if lmms_model_args else ",".join(extra_args)

    task_names = []
    for part in parts:
        if not args.data_dir and not args.bench_version:
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
    if args.force_simple:
        cmd.append("--force_simple")

    child_env = os.environ.copy()
    child_env.setdefault("DECORD_EOF_RETRY_MAX", "65536")
    child_env["CAMBW_VIDEO_PROTOCOL"] = args.video_protocol
    if args.video_protocol == "full_fps1":
        child_env["CAMBW_VIDEO_FPS"] = str(args.protocol_fps)
        child_env.pop("CAMBW_VIDEO_MAX_FRAMES", None)
    else:
        child_env["CAMBW_VIDEO_MAX_FRAMES"] = str(args.protocol_max_frames)
        child_env.pop("CAMBW_VIDEO_FPS", None)

    print("[ENV]", f"CAMBW_VIDEO_PROTOCOL={child_env['CAMBW_VIDEO_PROTOCOL']}", f"CAMBW_VIDEO_FPS={child_env.get('CAMBW_VIDEO_FPS', '-')}", f"CAMBW_VIDEO_MAX_FRAMES={child_env.get('CAMBW_VIDEO_MAX_FRAMES', '-')}")
    print("[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=child_env)


if __name__ == "__main__":
    main()
