#!/usr/bin/env python3
"""scripts/equivalence_static_coderank.py — the NO-REINDEX gate.

Embeds a sample corpus through BOTH the dynamic int8 ONNX graph and the
new static-shape graph (model_quantized_static_b<N>_s<M>.onnx), applying
the SAME post-processing the Rust serving path uses
(crate::pool::mean_pool_normalize: average only positions where
attention_mask > 0, then L2-normalize), and asserts per-vector
equivalence:

    cosine(dynamic_i, static_i) >= 0.99999   (default)
    max |dynamic_i - static_i|   <= 1e-5      (default)

WHY THIS IS LOAD-BEARING
  The static graph is re-exported from source and re-quantized. If its
  outputs drift from the shipped dynamic path, every vector already in
  the go-code corpus becomes inconsistent with new ones → a full 57-repo
  reindex is forced (cheap → expensive). This gate BLOCKS promotion.
  See docs/plans/2026-06-12-coderank-static-shape.md Scenario 3 / ADR-001.

WHAT IT MIRRORS FROM THE RUST SERVING PATH
  - seq padding: model.rs pads each batch to round_up_seq_len(max, cap)
    and zero-fills the mask over padded positions. Here the static graph
    has a FIXED seq (e.g. 512); we right-pad ids with pad_id and mask
    with 0 up to that fixed length, exactly as the future static serving
    code (CL-5) will.
  - batch padding + slice-back: the static graph has a FIXED batch (e.g.
    32). We pad the batch up to that with dummy rows, run, then SLICE the
    output back to the real row count — the equivalence is asserted only
    over the real rows. (CL-5 does the same slice in Rust.)
  - mean-pool + normalize: averages only mask>0 positions, then L2.
    Byte-for-byte the model.rs:374-395 guarantee.

This runs offline (no embed-server process); both graphs are loaded
directly via onnxruntime so the dynamic and static paths can be compared
on identical inputs in one process. Run AFTER export_static_coderank.sh
and BEFORE rsync'ing anything to pillow.

Usage:
  python3 scripts/equivalence_static_coderank.py \
    --dynamic <dir>/model_int8.onnx \
    --static  ./code-rank-embed-static/model_quantized_static_b32_s512.onnx \
    --tokenizer nomic-ai/CodeRankEmbed

  # custom corpus / thresholds / shape:
  python3 scripts/equivalence_static_coderank.py \
    --dynamic d.onnx --static s.onnx --tokenizer nomic-ai/CodeRankEmbed \
    --corpus my_symbols.txt --min-cosine 0.99999 --max-abs 1e-5 \
    --static-batch 32 --static-seq 512

Exit codes:
  0  — all sampled vectors pass both thresholds (NO REINDEX needed).
  1  — at least one vector fails (REINDEX FORCED — report loudly).
  2  — setup/usage error (missing file, shape mismatch, etc.).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("FATAL: onnxruntime not installed. `pip install onnxruntime`.", file=sys.stderr)
    sys.exit(2)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("FATAL: transformers not installed. `pip install transformers`.", file=sys.stderr)
    sys.exit(2)


# A default code-shaped corpus spanning short → long symbols so the gate
# exercises padded-short cases AND the seq=512 hot path AND the b=4 tail.
# Override with --corpus <file> (one symbol per line) for the ≥500-symbol
# run the plan's Scenario 3 mandates before a real promote.
DEFAULT_CORPUS = [
    "func main() {}",
    'package main\nimport "fmt"\nfunc main() { fmt.Println("hello") }',
    "def add(a, b):\n    return a + b",
    "class Foo:\n    def __init__(self, x):\n        self.x = x\n    def bar(self):\n        return self.x * 2",
    "// Round n up to the next power of two, capped at cap.\nfn round_up_seq_len(n: usize, cap: usize) -> usize {\n    if n <= 1 { return 1; }\n    n.next_power_of_two().min(cap)\n}",
    "SELECT id, name FROM users WHERE active = true ORDER BY created_at DESC LIMIT 100;",
    "x",
    "const STATIC_POOL_SIZE_PER_SHAPE: usize = 1;",
    (
        "// A long synthetic symbol to exercise the seq=512 hot path.\n"
        + "fn process_batch(items: &[Item]) -> Result<Vec<Output>, Error> {\n"
        + "".join(
            f"    let v{i} = items.get({i}).map(|it| it.transform()).unwrap_or_default();\n"
            for i in range(60)
        )
        + "    Ok(vec![])\n}\n"
    ),
    "import numpy as np\nfrom transformers import AutoTokenizer\n# embed two graphs and compare",
]


def mean_pool_normalize(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Average token vectors over mask>0 positions, then L2-normalize.

    Mirrors crate::pool::mean_pool_normalize exactly: padded positions
    (mask == 0) contribute neither to the mean numerator nor to the
    token count, so static-seq padding is numerically invisible.

    last_hidden: (B, S, H) float32
    mask:        (B, S)    int (1 = real token, 0 = pad)
    returns:     (B, H)    float32, each row L2-normalized
    """
    mask_f = mask.astype(np.float32)[:, :, None]  # (B, S, 1)
    summed = (last_hidden * mask_f).sum(axis=1)  # (B, H)
    counts = np.clip(mask_f.sum(axis=1), 1e-9, None)  # (B, 1) — avoid /0
    mean = summed / counts  # (B, H)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mean / norms


def get_input_names(sess: ort.InferenceSession) -> list[str]:
    return [i.name for i in sess.get_inputs()]


def first_output_name(sess: ort.InferenceSession) -> str:
    # feature-extraction graphs emit last_hidden_state as the primary
    # output; some exports name it differently — take the first.
    return sess.get_outputs()[0].name


def run_dynamic(sess: ort.InferenceSession, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run the dynamic graph at the natural (real_batch, real_seq) shape."""
    feed = {"input_ids": ids, "attention_mask": mask}
    names = set(get_input_names(sess))
    feed = {k: v for k, v in feed.items() if k in names}
    out = sess.run([first_output_name(sess)], feed)[0]
    return mean_pool_normalize(out, mask)


def run_static(
    sess: ort.InferenceSession,
    ids: np.ndarray,
    mask: np.ndarray,
    static_b: int,
    static_s: int,
    pad_id: int,
) -> np.ndarray:
    """Replicate the Rust static serving path: pad batch UP to static_b
    and seq UP to static_s, run the fixed-shape graph, slice rows back to
    the real batch count.

    Asserts the real shape fits the bucket (real_b <= static_b,
    real_s <= static_s) — exactly the routing precondition CL-5 enforces.
    """
    real_b, real_s = ids.shape
    if real_b > static_b or real_s > static_s:
        raise ValueError(
            f"real shape ({real_b},{real_s}) exceeds static bucket "
            f"({static_b},{static_s}) — routing would not select this bucket"
        )
    padded_ids = np.full((static_b, static_s), pad_id, dtype=ids.dtype)
    padded_mask = np.zeros((static_b, static_s), dtype=mask.dtype)
    padded_ids[:real_b, :real_s] = ids
    padded_mask[:real_b, :real_s] = mask

    feed = {"input_ids": padded_ids, "attention_mask": padded_mask}
    names = set(get_input_names(sess))
    feed = {k: v for k, v in feed.items() if k in names}
    out = sess.run([first_output_name(sess)], feed)[0]  # (static_b, static_s, H)

    pooled = mean_pool_normalize(out, padded_mask)  # (static_b, H)
    return pooled[:real_b]  # slice the batch dim back — the dummy rows are dropped


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (an * bn).sum(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dynamic", required=True, help="path to the dynamic int8 ONNX graph (model_int8.onnx)")
    ap.add_argument("--static", required=True, help="path to model_quantized_static_b<N>_s<M>.onnx")
    ap.add_argument("--tokenizer", default="nomic-ai/CodeRankEmbed", help="HF tokenizer id or path")
    ap.add_argument("--corpus", help="file with one symbol per line; default = built-in code-shaped set")
    ap.add_argument("--static-batch", type=int, default=32)
    ap.add_argument("--static-seq", type=int, default=512)
    ap.add_argument("--min-cosine", type=float, default=0.99999)
    ap.add_argument("--max-abs", type=float, default=1e-5)
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    args = ap.parse_args()

    for p in (args.dynamic, args.static):
        if not Path(p).exists():
            print(f"FATAL: ONNX file not found: {p}", file=sys.stderr)
            return 2

    if args.corpus:
        corpus = [ln.rstrip("\n") for ln in Path(args.corpus).read_text().splitlines() if ln.strip()]
        if len(corpus) < 500:
            print(
                f"WARNING: corpus has {len(corpus)} symbols; the plan's Scenario 3 "
                f"mandates >=500 spanning all seq buckets + the b=4 tail before a real promote.",
                file=sys.stderr,
            )
    else:
        corpus = DEFAULT_CORPUS
        print(
            "NOTE: using the built-in default corpus (smoke set). For the PROMOTE gate, "
            "pass --corpus with >=500 real symbols (plan Scenario 3).",
            file=sys.stderr,
        )

    print(f"loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    print(f"loading dynamic graph: {args.dynamic}")
    dyn = ort.InferenceSession(args.dynamic, providers=["CPUExecutionProvider"])
    print(f"loading static graph:  {args.static} (bucket b={args.static_batch} s={args.static_seq})")
    stat = ort.InferenceSession(args.static, providers=["CPUExecutionProvider"])

    # Tokenize once; truncate to the static seq cap so every symbol fits
    # the bucket (the Rust path caps at self.max_len before bucketing).
    enc = tok(
        corpus,
        padding=False,
        truncation=True,
        max_length=args.static_seq,
        return_tensors=None,
    )

    worst_cos = 1.0
    worst_abs = 0.0
    fails: list[tuple[int, float, float]] = []
    n = len(corpus)

    # Embed one symbol at a time so the dynamic path runs at its natural
    # per-symbol shape and the static path pads a real_b=1 row up to the
    # bucket — this is the strictest per-vector comparison. (Batched runs
    # are equivalent because mean-pool is per-row; per-row keeps the diff
    # attributable to a specific symbol.)
    for i in range(n):
        ids_1 = np.array([enc["input_ids"][i]], dtype=np.int64)
        mask_1 = np.array([enc["attention_mask"][i]], dtype=np.int64)

        v_dyn = run_dynamic(dyn, ids_1, mask_1)[0]
        v_stat = run_static(stat, ids_1, mask_1, args.static_batch, args.static_seq, pad_id)[0]

        cos = float(cosine_rows(v_dyn[None, :], v_stat[None, :])[0])
        mabs = float(np.max(np.abs(v_dyn - v_stat)))
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, mabs)
        if cos < args.min_cosine or mabs > args.max_abs:
            fails.append((i, cos, mabs))

    print()
    print("============================================================")
    print(f"EQUIVALENCE RESULT  ({n} symbols, bucket b{args.static_batch}_s{args.static_seq})")
    print(f"  worst cosine     : {worst_cos:.8f}   (gate >= {args.min_cosine})")
    print(f"  worst |abs delta|: {worst_abs:.3e}   (gate <= {args.max_abs:.1e})")
    print("============================================================")

    if fails:
        print(f"\nFAILED — {len(fails)} of {n} vectors drift beyond the gate:", file=sys.stderr)
        for i, cos, mabs in fails[:20]:
            preview = corpus[i][:60].replace("\n", "\\n")
            print(f"  [{i}] cos={cos:.8f} max_abs={mabs:.3e}  «{preview}»", file=sys.stderr)
        if len(fails) > 20:
            print(f"  ... and {len(fails) - 20} more", file=sys.stderr)
        print(
            "\n*** REINDEX FORCED *** — the static graph is NOT numerically\n"
            "equivalent to the dynamic path. Do NOT promote without a full\n"
            "go-code reindex. Investigate the quant recipe / export drift\n"
            "(plan ADR-001) before shipping.",
            file=sys.stderr,
        )
        return 1

    print("\nPASS — static path is numerically equivalent to dynamic (NO REINDEX needed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
