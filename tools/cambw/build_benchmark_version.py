#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import quote
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download

from lmms_eval.tasks.cambw.identity import compute_doc_uid, compute_eval_uid
DEFAULT_OUT_ROOT = REPO_ROOT / "lmms_eval" / "tasks" / "cambw" / "benchmarks"
DEFAULT_PART1_CORRECTED = "https://huggingface.co/datasets/Torwnexial/ready4label/resolve/main/new_long_video/corrected_json_3"
DEFAULT_PART23_CORRECTED = "https://huggingface.co/datasets/Torwnexial/ready4label/resolve/main/top20merge_full/corrected_json_4"
PART1_FILE = "part1_long_videos_-_dual_format_appearance.json"
PART2_FILE = "part2_short_videos_-_place_&_motion.json"
PART3_FILE = "part3_short_videos_-_objects_with_dual_format.json"
HF_DATASET_PREFIX = "https://huggingface.co/datasets/Torwnexial/ready4label/resolve/main/"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _cache_key(base: str) -> str:
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return digest


def _download_json(base: str, video_name: str, cache_dir: Path) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(base) / f"{video_name}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return load_json(cache_path)
    url = base.rstrip("/") + "/" + quote(f"{video_name}.json")
    if base.startswith(HF_DATASET_PREFIX):
        relative_dir = base[len(HF_DATASET_PREFIX) :].strip("/")
        last_error: Exception | None = None
        for _ in range(3):
            try:
                local_path = hf_hub_download(
                    repo_id="Torwnexial/ready4label",
                    repo_type="dataset",
                    filename=f"{relative_dir}/{video_name}.json",
                    local_dir=str(cache_path.parent),
                    local_dir_use_symlinks=False,
                )
                payload = load_json(Path(local_path))
                dump_json(cache_path, payload)
                return payload
            except Exception as exc:  # pragma: no cover - transient hub client failures
                last_error = exc
        try:
            with urlopen(url, timeout=60) as r:
                payload = json.load(r)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch corrected JSON for {video_name} from {url}") from (last_error or exc)
        dump_json(cache_path, payload)
        return payload
    try:
        with urlopen(url, timeout=60) as r:
            payload = json.load(r)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch corrected JSON for {video_name} from {url}") from exc
    dump_json(cache_path, payload)
    return payload


def load_corrected_json(source_roots: List[str], source_labels: List[str], video_name: str, cache_dir: Path) -> Tuple[Dict[str, Any], str]:
    if len(source_roots) != len(source_labels):
        raise ValueError("source_roots and source_labels must have the same length")
    last_error: Exception | None = None
    for source_root, source_label in zip(source_roots, source_labels):
        local_root = Path(source_root)
        try:
            if local_root.exists():
                candidate = local_root / f"{video_name}.json"
                if candidate.exists():
                    return load_json(candidate), source_label
                raise FileNotFoundError(candidate)
            return _download_json(source_root, video_name, cache_dir), source_label
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
            continue
    raise RuntimeError(f"Failed to load corrected JSON for {video_name} from any of: {source_roots}") from last_error


def try_load_corrected_json(source_roots: List[str], source_labels: List[str], video_name: str, cache_dir: Path) -> Tuple[Dict[str, Any], str] | None:
    try:
        return load_corrected_json(source_roots, source_labels, video_name, cache_dir)
    except Exception:
        return None


def swap_left_right(video_name: str) -> str:
    if video_name.endswith("_left"):
        return video_name[:-5] + "_right"
    if video_name.endswith("_right"):
        return video_name[:-6] + "_left"
    return video_name


def clip_count_map(raw: Dict[str, Any], concept: str) -> Dict[int, int]:
    counts = raw.get("concept_obj_counts_per_clip", {}).get(concept, {})
    out: Dict[int, int] = {}
    if isinstance(counts, dict):
        for key, value in counts.items():
            try:
                out[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
    return out


def cumulative_count(counts: Dict[int, int], clip_idx: int) -> int:
    return sum(value for key, value in counts.items() if key <= clip_idx)


def first_seen_clip(counts: Dict[int, int]) -> int | None:
    positives = [clip for clip, value in counts.items() if value > 0]
    return min(positives) if positives else None


def last_seen_clip_up_to(counts: Dict[int, int], clip_idx: int) -> int | None:
    positives = [clip for clip, value in counts.items() if clip <= clip_idx and value > 0]
    return max(positives) if positives else None


def build_permutations(order: List[str]) -> List[List[str]]:
    if len(order) < 2:
        return [list(order)]
    perms: List[List[str]] = []
    seen = set()

    def add(seq: Iterable[str]) -> None:
        seq = tuple(seq)
        if seq not in seen:
            seen.add(seq)
            perms.append(list(seq))

    add(order)
    add(list(reversed(order)))
    if len(order) >= 2:
        add(order[1:] + order[:1])
    if len(order) >= 3:
        add(order[2:] + order[:2])
    for perm in itertools.permutations(order, len(order)):
        add(perm)
        if len(perms) >= 4:
            break
    return perms[:4]


def pick_correct_index(correct_order: List[str], checkpoint: Dict[str, Any], task_type: str, video_name: str, num_choices: int) -> int:
    h = hashlib.sha256()
    key = "|".join(correct_order)
    key += f"|{task_type}|{video_name}|{checkpoint.get('clip_idx', 0)}|{checkpoint.get('checkpoint', 0)}|{checkpoint.get('elapsed_minutes', 0)}"
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest()[:2], "big") % max(num_choices, 1)


def render_options(permutations: List[List[str]], idx_correct: int) -> Tuple[List[str], str]:
    letters = ["A", "B", "C", "D"]
    if not permutations:
        return [], "A"
    arranged: List[List[str]] = [None] * len(permutations)  # type: ignore[assignment]
    arranged[idx_correct] = permutations[0]
    distractors = permutations[1:]
    d_idx = 0
    for idx in range(len(arranged)):
        if arranged[idx] is not None:
            continue
        arranged[idx] = distractors[d_idx]
        d_idx += 1
    options = [f"{letters[idx]}) {' → '.join(seq)}" for idx, seq in enumerate(arranged)]
    return options, letters[idx_correct]


def parse_option_sequence(option: str) -> List[str]:
    if not option:
        return []
    text = option.strip()
    if ")" in text:
        text = text.split(")", 1)[1].strip()
    return [part.strip() for part in text.split("→") if part.strip()]


def normalize_option_text(option: str) -> str:
    return " → ".join(parse_option_sequence(option))


def compute_first_order(raw: Dict[str, Any], subset_concepts: List[str], clip_idx: int) -> Tuple[List[str], List[str]]:
    raw_order = raw.get("concept_first_appearance_order") or []
    raw_rank = {concept: idx for idx, concept in enumerate(raw_order)}
    counts_by_concept = {concept: clip_count_map(raw, concept) for concept in subset_concepts}
    first_by_concept = {concept: first_seen_clip(counts) for concept, counts in counts_by_concept.items()}
    seen = [concept for concept in subset_concepts if first_by_concept[concept] is not None and first_by_concept[concept] <= clip_idx]
    actual = sorted(
        seen,
        key=lambda concept: (
            first_by_concept[concept],
            raw_rank.get(concept, 10**9),
            concept,
        ),
    )
    return seen, actual


def compute_last_order(raw: Dict[str, Any], subset_concepts: List[str], clip_idx: int) -> Tuple[List[str], List[str]]:
    raw_order = raw.get("concept_last_appearance_order") or []
    raw_rank = {concept: idx for idx, concept in enumerate(raw_order)}
    counts_by_concept = {concept: clip_count_map(raw, concept) for concept in subset_concepts}
    last_by_concept = {concept: last_seen_clip_up_to(counts, clip_idx) for concept, counts in counts_by_concept.items()}
    seen = [concept for concept in subset_concepts if last_by_concept[concept] is not None]
    actual = sorted(
        seen,
        key=lambda concept: (
            last_by_concept[concept],
            raw_rank.get(concept, 10**9),
            concept,
        ),
    )
    return seen, actual


def compute_global_appearance_order(raw: Dict[str, Any], subset_concepts: List[str], is_first: bool) -> List[str]:
    raw_order = raw.get("concept_first_appearance_order" if is_first else "concept_last_appearance_order") or []
    raw_rank = {concept: idx for idx, concept in enumerate(raw_order)}
    counts_by_concept = {concept: clip_count_map(raw, concept) for concept in subset_concepts}

    if is_first:
        clip_pos = {concept: first_seen_clip(counts) for concept, counts in counts_by_concept.items()}
    else:
        clip_pos = {
            concept: last_seen_clip_up_to(counts, max(counts) if counts else -1)
            for concept, counts in counts_by_concept.items()
        }

    ordered = [concept for concept in raw_order if concept in subset_concepts]
    remaining = [concept for concept in subset_concepts if concept not in ordered]
    ordered.extend(
        sorted(
            remaining,
            key=lambda concept: (
                clip_pos[concept] is None,
                clip_pos[concept] if clip_pos[concept] is not None else 10**9,
                concept,
            ),
        )
    )
    return ordered


def encode_direct_answer(subset_concepts: List[str], order: List[str]) -> str:
    label_map = {concept: chr(ord("A") + idx) for idx, concept in enumerate(subset_concepts)}
    return "".join(label_map[concept] for concept in order if concept in label_map)


def patch_counting_task(task: Dict[str, Any], raw: Dict[str, Any]) -> None:
    concept = task.get("concept")
    if not concept:
        return
    counts = clip_count_map(raw, concept)
    for checkpoint in task.get("checkpoints") or []:
        clip_idx = int(checkpoint.get("clip_idx", 0))
        value = cumulative_count(counts, clip_idx)
        if value > 0:
            checkpoint["answer"] = value
            checkpoint.pop("note", None)
        else:
            checkpoint["answer"] = None
            checkpoint["note"] = "concept not yet appeared"


def patch_appearance_task(task: Dict[str, Any], raw: Dict[str, Any], video_name: str) -> None:
    subset_concepts = list(task.get("subset_concepts") or [])
    task_type = str(task.get("task_type") or "")
    is_first = task_type.startswith("first_appearance")
    global_order = compute_global_appearance_order(raw, subset_concepts, is_first)

    for checkpoint in task.get("checkpoints") or []:
        clip_idx = int(checkpoint.get("clip_idx", 0))
        seen, checkpoint_order = (
            compute_first_order(raw, subset_concepts, clip_idx)
            if is_first
            else compute_last_order(raw, subset_concepts, clip_idx)
        )
        actual_order = global_order
        checkpoint["concepts_seen"] = seen
        checkpoint["correct_order"] = actual_order
        checkpoint["original_order"] = checkpoint_order
        checkpoint["checkpoint_order"] = checkpoint_order

        if len(actual_order) != len(subset_concepts) or len(actual_order) < 2:
            checkpoint["answer"] = None
            if task_type.endswith("_choice"):
                checkpoint["options"] = []
            continue

        if task_type.endswith("_choice"):
            existing_options = list(checkpoint.get("options") or [])
            normalized_options = [parse_option_sequence(option) for option in existing_options]
            normalized_text = [normalize_option_text(option) for option in existing_options]
            correct_text = " → ".join(actual_order)
            if (
                len(normalized_options) == 4
                and all(len(option) == len(actual_order) for option in normalized_options)
                and
                actual_order in normalized_options
                and len(normalized_text) == len(set(normalized_text))
                and normalized_text.count(correct_text) == 1
            ):
                checkpoint["answer"] = chr(ord("A") + normalized_options.index(actual_order))
            else:
                permutations = build_permutations(actual_order)
                idx_correct = pick_correct_index(actual_order, checkpoint, task_type, video_name, len(permutations))
                options, answer = render_options(permutations, idx_correct)
                checkpoint["options"] = options
                checkpoint["answer"] = answer
        else:
            checkpoint["answer"] = encode_direct_answer(subset_concepts, actual_order)


def _task_signature(task: Dict[str, Any]) -> str:
    return json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def merge_duplicate_videos(data: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    out = copy.deepcopy(data)
    videos = out.get("videos") or []
    merged_videos: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}
    task_sigs: Dict[str, set[str]] = {}
    duplicate_videos = 0
    duplicate_tasks = 0
    for video in videos:
        name = video.get("video_name", "")
        if name not in by_name:
            by_name[name] = video
            merged_videos.append(video)
            task_sigs[name] = {_task_signature(task) for task in video.get("tasks") or []}
            continue
        duplicate_videos += 1
        existing = by_name[name]
        seen = task_sigs[name]
        for task in video.get("tasks") or []:
            signature = _task_signature(task)
            if signature in seen:
                duplicate_tasks += 1
                continue
            existing.setdefault("tasks", []).append(task)
            seen.add(signature)
    out["videos"] = merged_videos
    return out, duplicate_videos, duplicate_tasks


def flatten_part(data: Dict[str, Any], bench_version: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    seen_doc_uids = set()
    seen_doc_ids = set()
    for video in data.get("videos") or []:
        video_name = video.get("video_name", "")
        for task_index, task in enumerate(video.get("tasks") or []):
            task_type = task.get("task_type", "")
            if task.get("variant"):
                task_type = f"{task_type}_{task['variant']}"
            source_video_name = task.get("source_video_name") or video_name
            source_folder = task.get("source_folder") or video.get("source_folder")
            for checkpoint_index, checkpoint in enumerate(task.get("checkpoints") or []):
                if checkpoint.get("answer") is None:
                    continue
                doc = {
                    "bench_version": bench_version,
                    "video_name": video_name,
                    "source_video_name": source_video_name,
                    "source_folder": source_folder,
                    "task_type": task_type,
                    "question": task.get("question", ""),
                    "answer": str(checkpoint["answer"]),
                    "frame_indices": None,
                    "options": checkpoint.get("options"),
                    "subset_concepts": task.get("subset_concepts"),
                }
                if "frame_recall" in task_type and checkpoint.get("frames"):
                    doc["frame_indices"] = [int(frame["frame_idx"]) for frame in checkpoint["frames"] if "frame_idx" in frame]
                doc["eval_uid"] = compute_eval_uid(doc)
                doc["doc_uid"] = compute_doc_uid(doc)
                if doc["doc_uid"] in seen_doc_uids:
                    continue
                base_doc_id = f"{video_name}|{task_index}|{task_type}|{checkpoint_index}"
                doc_id = base_doc_id
                if doc_id in seen_doc_ids:
                    doc_id = f"{base_doc_id}|{doc['eval_uid'][:8]}"
                dedupe_idx = 2
                while doc_id in seen_doc_ids:
                    doc_id = f"{base_doc_id}|{doc['eval_uid'][:8]}_{dedupe_idx}"
                    dedupe_idx += 1
                doc["doc_id"] = doc_id
                seen_doc_ids.add(doc_id)
                seen_doc_uids.add(doc["doc_uid"])
                docs.append(doc)
    return docs


def write_jsonl(path: Path, docs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def patch_part1(data: Dict[str, Any], part1_root: str, cache_dir: Path) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    part1_cache = cache_dir / "part1"
    part1_roots = [part1_root]
    part1_labels = ["new_long_video/corrected_json_3"]
    affected = {
        "object_counting",
        "first_appearance_recall_choice",
        "first_appearance_recall_direct",
        "last_appearance_recall_choice",
        "last_appearance_recall_direct",
    }
    kept_videos = []
    dropped_videos = []
    for video in out.get("videos") or []:
        video_name = video.get("video_name", "")
        needs_corrected = any(task.get("task_type") in affected for task in video.get("tasks") or [])
        resolved_video_source = video.get("source_folder")
        if needs_corrected:
            source_video_name = swap_left_right(video_name)
            loaded = try_load_corrected_json(part1_roots, part1_labels, source_video_name, part1_cache)
            if loaded is None:
                dropped_videos.append(video_name)
                continue
            raw, resolved_label = loaded
            resolved_video_source = resolved_label
            for task in video.get("tasks") or []:
                if task.get("task_type") not in affected:
                    continue
                task["source_video_name"] = source_video_name
                task["source_folder"] = resolved_label
                if task.get("task_type") == "object_counting":
                    patch_counting_task(task, raw)
                else:
                    patch_appearance_task(task, raw, source_video_name)
        if resolved_video_source:
            video["source_folder"] = resolved_video_source
        kept_videos.append(video)
    out["videos"] = kept_videos
    meta = out.setdefault("metadata", {})
    meta["version"] = "2.0"
    meta["source_override"] = "new_long_video/corrected_json_3"
    meta["left_right_swapped_task_types"] = sorted(affected)
    meta["dropped_videos_missing_source"] = dropped_videos
    return out


def patch_part3(data: Dict[str, Any], part3_root: str, cache_dir: Path) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    part3_cache = cache_dir / "part3"
    part3_roots = [part3_root]
    part3_labels = ["top20merge_full/corrected_json_4"]
    kept_videos = []
    dropped_videos = []
    for video in out.get("videos") or []:
        video_name = video.get("video_name", "")
        loaded = try_load_corrected_json(part3_roots, part3_labels, video_name, part3_cache)
        if loaded is None:
            dropped_videos.append(video_name)
            continue
        raw, resolved_label = loaded
        video["source_folder"] = resolved_label
        for task in video.get("tasks") or []:
            task["source_folder"] = resolved_label
            if task.get("task_type") == "object_counting":
                patch_counting_task(task, raw)
            else:
                patch_appearance_task(task, raw, video_name)
        kept_videos.append(video)
    out["videos"] = kept_videos
    meta = out.setdefault("metadata", {})
    meta["version"] = "2.0"
    meta["source_override"] = "top20merge_full/corrected_json_4"
    meta["dropped_videos_missing_source"] = dropped_videos
    return out


def summarize_docs(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, int] = {}
    for doc in docs:
        by_task[doc["task_type"]] = by_task.get(doc["task_type"], 0) + 1
    return {"num_docs": len(docs), "task_counts": by_task}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corrected Cambrian-W benchmark version with stable identities.")
    parser.add_argument("--current-benchmark-dir", required=True)
    parser.add_argument("--bench-version", required=True)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--part1-corrected-root", default=DEFAULT_PART1_CORRECTED)
    parser.add_argument("--part23-corrected-root", default=DEFAULT_PART23_CORRECTED)
    args = parser.parse_args()

    current_dir = Path(args.current_benchmark_dir)
    version_dir = Path(args.out_root) / args.bench_version
    source_dir = version_dir / "source"
    data_dir = version_dir / "data"
    cache_dir = REPO_ROOT / ".cache" / "cambw_benchmark_build" / args.bench_version

    if version_dir.exists():
        shutil.rmtree(version_dir)

    part1 = load_json(current_dir / PART1_FILE)
    part2 = load_json(current_dir / PART2_FILE)
    part3 = load_json(current_dir / PART3_FILE)

    new_part1 = patch_part1(part1, args.part1_corrected_root, cache_dir)
    new_part2, merged_videos, merged_tasks = merge_duplicate_videos(part2)
    new_part3 = patch_part3(part3, args.part23_corrected_root, cache_dir)

    dump_json(source_dir / PART1_FILE, new_part1)
    dump_json(source_dir / PART2_FILE, new_part2)
    dump_json(source_dir / PART3_FILE, new_part3)

    part1_docs = flatten_part(new_part1, args.bench_version)
    part2_docs = flatten_part(new_part2, args.bench_version)
    part3_docs = flatten_part(new_part3, args.bench_version)
    part23_docs = part2_docs + part3_docs

    write_jsonl(data_dir / "part1_long.jsonl", part1_docs)
    write_jsonl(data_dir / "part2_3_short.jsonl", part23_docs)

    manifest = {
        "bench_version": args.bench_version,
        "current_benchmark_dir": str(current_dir),
        "part1_corrected_root": args.part1_corrected_root,
        "part23_corrected_root": args.part23_corrected_root,
        "part1": summarize_docs(part1_docs),
        "part2": summarize_docs(part2_docs),
        "part3": summarize_docs(part3_docs),
        "part2_3": summarize_docs(part23_docs),
        "part2_merge": {
            "merged_duplicate_videos": merged_videos,
            "removed_duplicate_tasks": merged_tasks,
        },
    }
    dump_json(version_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
