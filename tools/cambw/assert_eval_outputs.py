#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", required=True)
args = parser.parse_args()
root = Path(args.output_path)
results = list(root.rglob("*_results.json"))
samples = list(root.rglob("*_samples_*.jsonl"))
if not results or not samples:
    print(f"[ASSERT] missing official outputs under {root}")
    print(f"[ASSERT] results={len(results)} samples={len(samples)}")
    raise SystemExit(1)
print(f"[ASSERT] ok results={len(results)} samples={len(samples)} under {root}")
