#!/usr/bin/env python3
"""Smoke-test exported BGE-reranker-v2-m3 ONNX directly, no server.

Loads model_quantized.onnx, tokenizes two (query, doc) pairs — one relevant,
one unrelated — and asserts the relevant pair gets a higher score.

Run: python3 scripts/smoke_reranker_bge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = Path("/home/krolik/deploy/krolik-server/models/bge-reranker-v2-m3")


def main() -> int:
    onnx_path = MODEL_DIR / "model_quantized.onnx"
    if not onnx_path.exists():
        # Fall back to non-quantized if quantization step failed.
        onnx_path = MODEL_DIR / "model.onnx"
    print(f"loading {onnx_path}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    pairs = [
        ("what is a cat", "a cat is a small domestic feline mammal"),
        ("what is a cat", "the price of oil dropped yesterday"),
    ]
    enc = tok(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True, truncation=True, max_length=512, return_tensors="np",
    )
    input_names = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in enc.items() if k in input_names}
    out = sess.run(None, feeds)
    logits = np.asarray(out[0]).reshape(-1)
    print(f"scores: relevant={logits[0]:.3f} unrelated={logits[1]:.3f}")
    if logits[0] <= logits[1]:
        print("FAIL: relevant pair should outscore unrelated pair", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
