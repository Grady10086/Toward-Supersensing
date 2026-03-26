# Toward Supersensing

Toward Supersensing is a public `lmms-eval`-based release for evaluating Cambrian-W with a unified task interface across long and short videos.

This repository keeps the standard `lmms-eval` evaluation flow and official output schema while adding Cambrian-W task definitions and the validated `Qwen3-VL` long-video path.

## What This Release Includes

- Native `lmms-eval` tasks:
  - `cambw_part1_long`
  - `cambw_part2_3_short`
- Long-video support for `qwen3_vl` with `fps=1`
- No implicit fallback to `32` frames when `fps` is enabled and `max_num_frames` is unset
- Optional realtime sidecar logging via `LMMS_EVAL_STREAM_LOG`
- Task-layer `frame_recall` materialization as:
  - `12` uniformly sampled context frames
  - `4` checkpoint frames (`Frame A` to `Frame D`)
- Official `lmms-eval` outputs preserved:
  - `*_results.json`
  - `*_samples_*.jsonl`

## Benchmark Split

- `cambw_part1_long`: `40` long videos, `603` evaluation docs
- `cambw_part2_3_short`: `434` short videos, `11751` evaluation docs

## Important Notes

- Benchmark JSON files and raw videos are **not** included in this repository.
- Generate task JSONL locally with `tools/cambw/prepare_data.py`.
- This release was validated against upstream `lmms-eval` commit `88b23e2`.

## Quick Start

```bash
python -m pip install -e .
python -m pip install -r requirements-cambw-qwen3vl.lock.txt
```

Prepare task data:

```bash
export CAMBW_BENCH_DIR=/path/to/cambw_annotations
export CAMBW_VIDEO_ROOT=/path/to/cambw_videos
python tools/cambw/prepare_data.py
```

Smoke test `Qwen3-VL` on long video with `fps=1`:

```bash
export HF_HOME=/path/to/hf_cache
export LMMS_EVAL_STREAM_LOG=/path/to/output/stream_samples.jsonl
accelerate launch --num_processes 1 -m lmms_eval eval   --model qwen3_vl   --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,fps=1,min_pixels=100352,max_pixels=200704"   --tasks cambw_part1_long   --batch_size 1   --limit 1   --output_path /path/to/output   --log_samples
```

Use the convenience wrapper:

```bash
python tools/cambw/run_cambw_eval.py   --part part1   --model qwen3_vl   --model_args "pretrained=Qwen/Qwen3-VL-8B-Instruct,min_pixels=100352,max_pixels=200704"   --fps 1
```

## Repository Guide

- [Cambrian-W Repro Guide](docs/cambw/OPEN_SOURCE_REPRO_GUIDE_QWEN3VL_FPS1.md)
- [Cambrian-W Task Data Notes](lmms_eval/tasks/cambw/data/README.md)
- [Qwen3-VL Lockfile](requirements-cambw-qwen3vl.lock.txt)

## Upstream Attribution

This release is built on top of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). The upstream project structure and license are preserved here, with Cambrian-W specific integration layered on top.
