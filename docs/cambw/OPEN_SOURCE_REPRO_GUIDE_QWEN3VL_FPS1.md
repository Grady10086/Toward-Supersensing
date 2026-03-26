# Qwen3-VL on Cambrian-W Repro Guide

This document describes the validated public repro path for evaluating `Qwen/Qwen3-VL-8B-Instruct` on Cambrian-W inside `lmms-eval`.

## Scope

Supported tasks:

- `cambw_part1_long`
- `cambw_part2_3_short`

Validated behavior:

- `fps=1` long-video evaluation for `qwen3_vl`
- official `lmms-eval` outputs remain unchanged
- optional realtime sidecar logging via `LMMS_EVAL_STREAM_LOG`
- `frame_recall` is materialized at the task layer as `12` context frames plus `4` checkpoint frames

## Data Prerequisites

This repository does **not** include:

- benchmark JSON files
- raw benchmark videos

You need to provide:

- `CAMBW_BENCH_DIR=/path/to/cambw_annotations`
- `CAMBW_VIDEO_ROOT=/path/to/cambw_videos`

Then generate task JSONL:

```bash
python tools/cambw/prepare_data.py
```

Generated files:

- `lmms_eval/tasks/cambw/data/part1_long.jsonl`
- `lmms_eval/tasks/cambw/data/part2_3_short.jsonl`

## Validated Package Versions

```bash
python -m pip install -r requirements-cambw-qwen3vl.lock.txt
```

Expected validation:

```bash
python - <<'PY'
import transformers
print('transformers', transformers.__version__)
print('has_Qwen3VLForConditionalGeneration', hasattr(transformers, 'Qwen3VLForConditionalGeneration'))
PY
```

Expected output:

- `transformers 4.57.6`
- `has_Qwen3VLForConditionalGeneration True`

## Minimal Smoke Test

```bash
export HF_HOME=/path/to/hf_cache
export LMMS_EVAL_STREAM_LOG=/path/to/output/stream_samples.jsonl
accelerate launch --num_processes 1 -m lmms_eval eval   --model qwen3_vl   --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,fps=1,min_pixels=100352,max_pixels=200704"   --tasks cambw_part1_long   --batch_size 1   --limit 1   --output_path /path/to/output   --log_samples
```

Expected artifacts:

- `stream_samples.jsonl`
- `Qwen__Qwen3-VL-8B-Instruct/*_results.json`
- `Qwen__Qwen3-VL-8B-Instruct/*_samples_cambw_part1_long.jsonl`

## Full Evaluation

```bash
export HF_HOME=/path/to/hf_cache
export LMMS_EVAL_STREAM_LOG=/path/to/output/stream_samples.jsonl
accelerate launch --num_processes 1 -m lmms_eval eval   --model qwen3_vl   --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,fps=1,min_pixels=100352,max_pixels=200704"   --tasks cambw_part1_long,cambw_part2_3_short   --batch_size 1   --output_path /path/to/output   --log_samples
```

## Parameter Notes

- `fps=1`: target temporal sampling rate for long videos
- `max_num_frames`: optional cap after enabling `fps`
- `min_pixels` / `max_pixels`: per-frame pixel range
- `LMMS_EVAL_STREAM_LOG`: optional sidecar log, does not change official `lmms-eval` outputs

## Common Issues

### `ImportError: Qwen3VLForConditionalGeneration`

Use the lockfile versions from this release.

### `AssertionError: max_pixels >= min_pixels`

Ensure `max_pixels >= min_pixels`.

### Long videos are slow

This is expected for `fps=1`. Mitigations:

- lower `max_pixels`
- set `max_num_frames`
- run `--limit 1` first
- keep `LMMS_EVAL_STREAM_LOG` enabled for visibility

## Release Checklist

- no benchmark JSONL with private absolute paths committed
- no raw videos committed
- no result directories committed
- `README.md` and this guide match the actual commands
- `frame_recall` task behavior is task-layer, not model-specific
- `stream_samples.jsonl` is documented as sidecar output only
