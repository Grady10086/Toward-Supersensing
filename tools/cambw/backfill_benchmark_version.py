#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.tasks.cambw.identity import compute_doc_uid, compute_eval_uid


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_options(options: Any) -> List[str]:
    if not options:
        return []
    return [str(x).strip() for x in options]


def parse_sequence(text: str) -> str:
    text = text.upper().replace("\u2192", "").replace("->", "").replace(" ", "").replace(",", "")
    match = re.search(r"[ABCD]{2,4}", text)
    return match.group(0) if match else ""


def extract_letter(text: str) -> str:
    text = text.upper().strip()
    patterns = [
        r"ANSWER\s*(?:IS|:)?\s*[\"']?([A-D])[\"']?",
        r"\bOPTION\s*([A-D])\b",
        r"\bCHOICE\s*([A-D])\b",
        r"\(([A-D])\)",
        r"^[\s\[\(\{]*([A-D])[\s\]\)\}\. ]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    match = re.search(r"\b([A-D])[\.\)]", text)
    if match:
        return match.group(1)
    for letter in ["A", "B", "C", "D"]:
        if letter in text:
            return letter
    return "A"


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    text = (results[0] or "").strip() if results else ""
    task_type = (doc.get("task_type") or "").lower()
    gt = doc.get("answer")
    out = {
        "doc_id": doc.get("doc_id"),
        "eval_uid": doc.get("eval_uid") or compute_eval_uid(doc),
        "doc_uid": doc.get("doc_uid") or compute_doc_uid(doc),
        "gt": gt,
        "pred_raw": text,
        "task_type": doc.get("task_type") or "unknown",
    }
    if "direct" in task_type and "appearance" in task_type:
        pred = parse_sequence(text)
        out["pred"] = pred
        out["correct"] = pred == gt
        out["accuracy"] = 1.0 if pred == gt else 0.0
        return out
    if doc.get("options") is not None or "choice" in task_type or "recall" in task_type or "motion" in task_type:
        pred = extract_letter(text)
        out["pred"] = pred
        out["correct"] = pred == gt
        out["accuracy"] = 1.0 if pred == gt else 0.0
        return out
    if "count" in task_type:
        nums = re.findall(r"\d+", text)
        pred_val = int(nums[0]) if nums else 0
        gt_val = gt if isinstance(gt, int) else int(gt)
        mra = max(0.0, 1.0 - abs(pred_val - gt_val) / max(gt_val, 1))
        out["pred"] = pred_val
        out["mra"] = mra
        out["accuracy"] = 1.0 if abs(pred_val - gt_val) / max(gt_val, 1) <= 0.1 else 0.0
        return out
    return out


def aggregate_accuracy(results: List[Dict[str, Any]]) -> float:
    vals = [_as_float_or_none(item.get("accuracy")) for item in results]
    vals = [value for value in vals if value is not None]
    return 0.0 if not vals else sum(vals) / len(vals) * 100.0


def aggregate_mra(results: List[Dict[str, Any]]) -> float:
    vals = [_as_float_or_none(item.get("mra")) for item in results]
    vals = [value for value in vals if value is not None]
    return 0.0 if not vals else sum(vals) / len(vals) * 100.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_identities(doc: Dict[str, Any]) -> Dict[str, Any]:
    doc = dict(doc)
    doc.setdefault("eval_uid", compute_eval_uid(doc))
    doc.setdefault("doc_uid", compute_doc_uid(doc))
    return doc


def load_doc_map(data_dir: Path, file_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    docs = [ensure_identities(doc) for doc in load_jsonl(data_dir / file_name)]
    by_doc_id: Dict[str, List[Dict[str, Any]]] = {}
    for doc in docs:
        by_doc_id.setdefault(doc["doc_id"], []).append(doc)
    by_eval_uid = {doc["eval_uid"]: doc for doc in docs}
    return docs, by_doc_id, by_eval_uid


def rescore_sample(sample: Dict[str, Any], new_doc: Dict[str, Any]) -> Dict[str, Any]:
    rescored = dict(sample)
    metrics = process_results(new_doc, [sample.get("pred_raw") or ""])
    rescored.update(metrics)
    rescored["doc_id"] = new_doc.get("doc_id")
    rescored["target"] = new_doc.get("answer")
    rescored["bench_version"] = new_doc.get("bench_version")
    rescored["video_name"] = new_doc.get("video_name")
    rescored["source_video_name"] = new_doc.get("source_video_name") or new_doc.get("video_name")
    rescored["source_folder"] = new_doc.get("source_folder")
    return rescored


def _sample_question(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("doc"), dict):
        return str(sample["doc"].get("question") or "")
    return str(sample.get("question") or "")


def _sample_options(sample: Dict[str, Any]) -> List[str]:
    if isinstance(sample.get("doc"), dict):
        return _normalize_options(sample["doc"].get("options"))
    return _normalize_options(sample.get("options"))


def select_old_doc(candidates: List[Dict[str, Any]], sample: Dict[str, Any]) -> Dict[str, Any] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    target = sample.get("target")
    if target is not None:
        matched = [doc for doc in candidates if str(doc.get("answer")) == str(target)]
        if len(matched) == 1:
            return matched[0]
        if matched:
            candidates = matched

    question = _sample_question(sample)
    if question:
        matched = [doc for doc in candidates if str(doc.get("question") or "") == question]
        if len(matched) == 1:
            return matched[0]
        if matched:
            candidates = matched

    sample_options = _sample_options(sample)
    if sample_options:
        matched = [doc for doc in candidates if _normalize_options(doc.get("options")) == sample_options]
        if len(matched) == 1:
            return matched[0]
        if matched:
            candidates = matched

    sample_video_name = sample.get("video_name")
    if sample_video_name:
        matched = [doc for doc in candidates if doc.get("video_name") == sample_video_name]
        if len(matched) == 1:
            return matched[0]
        if matched:
            candidates = matched

    if len(candidates) == 1:
        return candidates[0]
    return None


def prepare_delta(args: argparse.Namespace) -> None:
    old_docs, old_by_doc_id, _ = load_doc_map(Path(args.old_data_dir), args.file_name)
    new_docs, _, new_by_eval_uid = load_doc_map(Path(args.new_data_dir), args.file_name)
    old_samples = load_jsonl(Path(args.old_samples))

    reusable: List[Dict[str, Any]] = []
    reusable_eval_uids = set()
    exact_reuse = 0
    rescored_reuse = 0
    ambiguous_old_doc_ids = 0

    for sample in old_samples:
        old_doc = select_old_doc(old_by_doc_id.get(sample.get("doc_id"), []), sample)
        if old_doc is None:
            if sample.get("doc_id") in old_by_doc_id and len(old_by_doc_id[sample["doc_id"]]) > 1:
                ambiguous_old_doc_ids += 1
            continue
        new_doc = new_by_eval_uid.get(old_doc["eval_uid"])
        if new_doc is None:
            continue
        reusable_eval_uids.add(new_doc["eval_uid"])
        rescored = rescore_sample(sample, new_doc)
        reusable.append(rescored)
        if old_doc["doc_uid"] == new_doc["doc_uid"]:
            exact_reuse += 1
        else:
            rescored_reuse += 1

    delta_docs = [doc for doc in new_docs if doc["eval_uid"] not in reusable_eval_uids]

    out_dir = Path(args.out_dir)
    runner_ready_path = out_dir / args.file_name
    part_name = "part1" if args.file_name == "part1_long.jsonl" else "part2_3"
    write_jsonl(out_dir / "reused_samples.jsonl", reusable)
    write_jsonl(out_dir / "delta_data.jsonl", delta_docs)
    write_jsonl(runner_ready_path, delta_docs)
    (out_dir / "how_to_run.txt").write_text(
        "\n".join(
            [
                f"delta_docs={len(delta_docs)}",
                f"reused_exact={exact_reuse}",
                f"reused_rescored={rescored_reuse}",
                "",
                f"Runner-ready delta file: {runner_ready_path.name}",
                "",
                "Minimal path:",
                f"1. python3 tools/cambw/run_cambw_eval.py --part {part_name} --data-dir {out_dir} --model <model_key_or_adapter> [other args]",
                f"2. python3 tools/cambw/backfill_benchmark_version.py finalize --reused-samples {out_dir / 'reused_samples.jsonl'} --delta-samples <new_samples_jsonl> --out-dir <finalize_out_dir>",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "file_name": args.file_name,
        "old_docs": len(old_docs),
        "new_docs": len(new_docs),
        "reused_exact": exact_reuse,
        "reused_rescored": rescored_reuse,
        "delta_docs": len(delta_docs),
        "ambiguous_old_doc_ids": ambiguous_old_doc_ids,
        "runner_ready_file": str(runner_ready_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def aggregate_by_task(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("task_type") or "unknown", []).append(row)
    out: Dict[str, Dict[str, float]] = {}
    for task_type, items in grouped.items():
        out[task_type] = {
            "count": float(len(items)),
            "accuracy": aggregate_accuracy(items),
            "mra": aggregate_mra(items),
        }
    return out


def finalize(args: argparse.Namespace) -> None:
    reused = load_jsonl(Path(args.reused_samples))
    delta = load_jsonl(Path(args.delta_samples))
    merged = reused + delta
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "merged_samples.jsonl", merged)
    summary = {
        "count": len(merged),
        "accuracy": aggregate_accuracy(merged),
        "mra": aggregate_mra(merged),
        "by_task": aggregate_by_task(merged),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare delta backfill or finalize merged samples for a corrected benchmark version.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--old-data-dir", required=True)
    prepare.add_argument("--new-data-dir", required=True)
    prepare.add_argument("--old-samples", required=True)
    prepare.add_argument("--file-name", required=True, choices=["part1_long.jsonl", "part2_3_short.jsonl"])
    prepare.add_argument("--out-dir", required=True)
    prepare.set_defaults(func=prepare_delta)

    finalize_p = sub.add_parser("finalize")
    finalize_p.add_argument("--reused-samples", required=True)
    finalize_p.add_argument("--delta-samples", required=True)
    finalize_p.add_argument("--out-dir", required=True)
    finalize_p.set_defaults(func=finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
