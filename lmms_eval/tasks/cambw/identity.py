from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List


def _normalize_options(options: Any) -> List[str]:
    if not options:
        return []
    return [str(x).strip() for x in options]


def _normalize_subset(subset: Any) -> List[str]:
    if not subset:
        return []
    return [str(x).strip() for x in subset]


def _normalize_frames(frame_indices: Any) -> List[int]:
    if not frame_indices:
        return []
    out: List[int] = []
    for value in frame_indices:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def build_eval_identity_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that define the model-facing input.

    This intentionally excludes the ground-truth answer so old samples can be
    rescored under a corrected benchmark without forcing a rerun.
    """

    return {
        "task_type": doc.get("task_type"),
        "video_name": doc.get("video_name"),
        "source_video_name": doc.get("source_video_name") or doc.get("video_name"),
        "question": doc.get("question", ""),
        "options": _normalize_options(doc.get("options")),
        "subset_concepts": _normalize_subset(doc.get("subset_concepts")),
        "frame_indices": _normalize_frames(doc.get("frame_indices")),
    }


def build_doc_identity_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    payload = build_eval_identity_payload(doc)
    payload["answer"] = str(doc.get("answer"))
    return payload


def _stable_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_eval_uid(doc: Dict[str, Any]) -> str:
    return _stable_hash(build_eval_identity_payload(doc))


def compute_doc_uid(doc: Dict[str, Any]) -> str:
    return _stable_hash(build_doc_identity_payload(doc))
