#!/usr/bin/env python3
"""Generate Cambrian-W JSONL task files for lmms-eval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "lmms_eval" / "tasks" / "cambw" / "data"
SOURCE_FOLDER_TO_DATA_SUBDIRS = {
    "new_long_video/corrected_json_2": ["long_video_persp", "new_long_video_persp"],
    "top20merge/corrected_json": ["top20merge_0207_persp"],
    "long_video/corrected_json_2": ["long_video_persp"],
    "top20merge_full/corrected_json_2": ["top20merge_0207_persp"],
}


def resolve_video_path(video: Dict[str, Any], data_root: str) -> Optional[str]:
    video_path = video.get("video_path") or video.get("video")
    if video_path:
        video_path = str(video_path).strip()
        if os.path.isabs(video_path):
            for old_prefix in [
                "/lustre/fs12/portfolios/nvr/projects/nvr_av_end2endav/users/ymingli/projects/xty/cambw/data",
                "/lustre/fsw/portfolios/nvr/users/ymingli/projects/xty/cambw/data",
                "/data",
                "/path/to/data",
            ]:
                if video_path.startswith(old_prefix):
                    video_path = os.path.join(data_root, video_path[len(old_prefix):].lstrip("/"))
                    break
        else:
            video_path = os.path.join(data_root, video_path)
        return video_path

    video_name = video.get("video_name")
    source_folder = video.get("source_folder")
    if not video_name or not source_folder:
        return None

    subdirs = SOURCE_FOLDER_TO_DATA_SUBDIRS.get(source_folder)
    if not subdirs:
        return None

    base = video_name if video_name.endswith(".mp4") else f"{video_name}.mp4"
    for subdir in subdirs:
        subdir_abs = os.path.join(data_root, subdir)
        if os.path.isdir(subdir_abs):
            return os.path.join(subdir_abs, base)
    return os.path.join(data_root, subdirs[0], base)


def task_has_valid_checkpoints(task: Dict[str, Any]) -> bool:
    return any(cp.get("answer") is not None for cp in task.get("checkpoints", []))


def flatten_part(bench_path: Path, data_root: str) -> List[Dict[str, Any]]:
    with bench_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Dict[str, Any]] = []
    for video in data.get("videos") or []:
        video_path = resolve_video_path(video, data_root)
        if not video_path:
            continue
        video_name = video.get("video_name", "")
        tasks = video.get("tasks") or []
        for ti, task in enumerate(tasks):
            if not task_has_valid_checkpoints(task):
                continue
            task_type = task.get("task_type", "")
            if task.get("variant"):
                task_type = f"{task_type}_{task['variant']}"
            for cpi, checkpoint in enumerate(task.get("checkpoints", [])):
                if checkpoint.get("answer") is None:
                    continue
                doc = {
                    "doc_id": f"{video_name}|{ti}|{task_type}|{cpi}",
                    "video_name": video_name,
                    "video_path": video_path,
                    "task_type": task_type,
                    "question": task.get("question", ""),
                    "answer": str(checkpoint["answer"]),
                    "frame_indices": None,
                    "options": checkpoint.get("options"),
                    "subset_concepts": task.get("subset_concepts"),
                }
                if "frame_recall" in task_type and checkpoint.get("frames"):
                    doc["frame_indices"] = [int(frame["frame_idx"]) for frame in checkpoint["frames"] if "frame_idx" in frame]
                docs.append(doc)
    return docs


def summarize_missing_videos(jsonl_path: Path) -> None:
    missing = 0
    total = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            video_path = item.get("video_path")
            if not video_path or not os.path.exists(video_path):
                missing += 1
    print(f"[video-check] {jsonl_path.name}: missing {missing}/{total}")


def require_path(value: Optional[str], env_name: str) -> str:
    if value:
        return value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    raise SystemExit(f"Missing required path. Provide the matching CLI flag or set {env_name}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    benchmark_dir = Path(require_path(args.benchmark_dir, "CAMBW_BENCH_DIR"))
    data_root = require_path(args.data_root, "CAMBW_VIDEO_ROOT")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    part1_file = benchmark_dir / "part1_long_videos_-_dual_format_appearance.json"
    if part1_file.exists():
        docs1 = flatten_part(part1_file, data_root)
        out1 = out_dir / "part1_long.jsonl"
        with out1.open("w", encoding="utf-8") as f:
            for item in docs1:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"part1_long: {len(docs1)} docs -> {out1}")
        summarize_missing_videos(out1)
    else:
        print(f"Skip part1: not found {part1_file}")

    part2_file = benchmark_dir / "part2_short_videos_-_place_&_motion.json"
    part3_candidates = [
        benchmark_dir / "part3_short_videos_-_objects_with_dual_format_fixed_choices.json",
        benchmark_dir / "part3_short_videos_-_objects_with_dual_format.json",
    ]
    part3_file = next((p for p in part3_candidates if p.exists()), None)

    docs2_3: List[Dict[str, Any]] = []
    if part2_file.exists():
        docs2_3.extend(flatten_part(part2_file, data_root))
        print(f"part2: {len(docs2_3)} docs")
    if part3_file is not None:
        before = len(docs2_3)
        docs2_3.extend(flatten_part(part3_file, data_root))
        print(f"part3 ({part3_file.name}): +{len(docs2_3) - before} docs")
    else:
        print(f"Skip part3: not found in {[p.name for p in part3_candidates]}")

    if docs2_3:
        out2_3 = out_dir / "part2_3_short.jsonl"
        with out2_3.open("w", encoding="utf-8") as f:
            for item in docs2_3:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"part2_3_short: {len(docs2_3)} docs -> {out2_3}")
        summarize_missing_videos(out2_3)


if __name__ == "__main__":
    main()
