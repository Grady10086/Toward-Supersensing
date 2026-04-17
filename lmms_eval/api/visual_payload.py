from __future__ import annotations

from typing import Any, Dict, List, Sequence

from PIL import Image

VISUAL_PAYLOAD_MARKER = "_lmms_visual_payload"
KIND_VIDEO_PATH = "video_path"
KIND_IMAGE_SEQUENCE = "image_sequence"
KIND_LEGACY_ITEMS = "legacy_items"


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def is_visual_payload(value: Any) -> bool:
    return isinstance(value, dict) and value.get(VISUAL_PAYLOAD_MARKER) is True


def make_video_path_payload(video_path: str, **metadata: Any) -> Dict[str, Any]:
    return {
        VISUAL_PAYLOAD_MARKER: True,
        "kind": KIND_VIDEO_PATH,
        "video_path": video_path,
        "metadata": metadata,
    }


def make_image_sequence_payload(images: Sequence[Image.Image], **metadata: Any) -> Dict[str, Any]:
    return {
        VISUAL_PAYLOAD_MARKER: True,
        "kind": KIND_IMAGE_SEQUENCE,
        "images": list(images),
        "metadata": metadata,
    }


def normalize_visual_payloads(visuals: Any) -> List[Dict[str, Any]]:
    entries = _ensure_list(visuals)
    if not entries:
        return []

    if all(is_visual_payload(entry) for entry in entries):
        return list(entries)

    if all(isinstance(entry, Image.Image) for entry in entries):
        return [make_image_sequence_payload(entries)]

    if len(entries) == 1 and isinstance(entries[0], str):
        return [make_video_path_payload(entries[0])]

    return [
        {
            VISUAL_PAYLOAD_MARKER: True,
            "kind": KIND_LEGACY_ITEMS,
            "items": entries,
            "metadata": {},
        }
    ]


def flatten_visual_inputs(visuals: Any) -> List[Any]:
    flat: List[Any] = []
    for payload in normalize_visual_payloads(visuals):
        kind = payload.get("kind")
        if kind == KIND_VIDEO_PATH:
            flat.append(payload["video_path"])
        elif kind == KIND_IMAGE_SEQUENCE:
            flat.extend(payload.get("images", []))
        elif kind == KIND_LEGACY_ITEMS:
            flat.extend(payload.get("items", []))
        else:
            raise ValueError(f"Unsupported visual payload kind: {kind}")
    return flat
