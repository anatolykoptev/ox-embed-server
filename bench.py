# /home/krolik/src/embed-server/bench.py
"""Benchmark harness for embed-server.

Supports `/v1/embeddings` (default) and `/v1/rerank` endpoints, with
configurable concurrency, batch shape, and output format.

For ModernBERT vs gte-multi head-to-head: see `docs/plans/2026-05-01-modernbert-optimization.md`
Phase 2 — recommended sweep mirrors the real memdb-go consumer:
`--kind rerank --docs-per-req {5,20,50} --size {short,medium,long}`.
"""
import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request


# ---------------------------------------------------------------------
# Test fixtures. Three sizes per text/code kind so callers can sweep
# document length, which dominates cross-encoder cost (O(seq_len^2) for
# the global-attention block on ModernBERT, O(N) on the local-attention
# blocks). The `long` fixture reaches ModernBERT's 256-token cap so we
# can measure how the model behaves at saturation, where current
# production traffic actually lives.
# ---------------------------------------------------------------------

RUSSIAN_SHORT = "Санкт-Петербург — культурная столица России."
RUSSIAN_MEDIUM = (
    "Санкт-Петербург — город федерального значения и культурная столица России. "
    "Основан Петром I в 1703 году как крепость Санкт-Питер-Бурх на Заячьем острове. "
    "Город отличается уникальной архитектурой, многочисленными каналами и мостами, а также богатым "
    "культурным наследием, включая Эрмитаж, Петропавловскую крепость и Казанский собор."
) * 2
RUSSIAN_LONG = RUSSIAN_MEDIUM * 4  # ~400 tokens — saturates max_len=256 with truncation
CODE_SHORT = 'func main() { fmt.Println("hello") }'
CODE_MEDIUM = (
    'package main\nimport "fmt"\n'
    'func calculate(x, y int) int {\n    result := x*y + 42\n    if result > 100 { return result }\n    return 0\n}\n'
    'func main() {\n    for i := 0; i < 100; i++ { fmt.Println(calculate(i, i+1)) }\n}\n'
) * 2
CODE_LONG = CODE_MEDIUM * 4

# Realistic memdb-go-shaped query: short, declarative, English. Phase 2
# bench should NOT use a paragraph-length query — that's not what the
# /v1/rerank handler ever sees in production.
RERANK_QUERY = "what does the function do"

FIXTURES = {
    ("text", "short"): RUSSIAN_SHORT,
    ("text", "medium"): RUSSIAN_MEDIUM,
    ("text", "long"): RUSSIAN_LONG,
    ("code", "short"): CODE_SHORT,
    ("code", "medium"): CODE_MEDIUM,
    ("code", "long"): CODE_LONG,
    # rerank uses the same text/code corpus for documents — no separate
    # rerank-only snippet, since the cross-encoder receives whatever
    # candidates the retrieval stage upstream produced.
    ("rerank", "short"): RUSSIAN_SHORT,
    ("rerank", "medium"): RUSSIAN_MEDIUM,
    ("rerank", "long"): RUSSIAN_LONG,
}


# ---------------------------------------------------------------------
# HTTP client. Two payload shapes; one transport.
# ---------------------------------------------------------------------

def _post(url: str, payload: dict) -> float:
    """Issue one POST, return wall-clock latency in milliseconds."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return (time.monotonic() - t0) * 1000.0


def _fire_embed(url: str, model: str, texts: list[str]) -> float:
    return _post(url, {"input": texts, "model": model})


def _fire_rerank(url: str, model: str, query: str, docs: list[str]) -> float:
    return _post(url, {"model": model, "query": query, "documents": docs})


# ---------------------------------------------------------------------
# Scenario runner.
# ---------------------------------------------------------------------

def _percentile(latencies_sorted: list[float], p: float) -> float:
    """Index-based percentile (linear interpolation not needed at our N)."""
    idx = max(0, min(len(latencies_sorted) - 1, int(len(latencies_sorted) * p)))
    return latencies_sorted[idx]


def run(fire_fn, concurrency: int, iterations: int) -> dict:
    """Run `fire_fn` `iterations` times at `concurrency`, return summary."""
    # 3-call warmup absorbs JIT, cache, and ORT arena-allocation spikes
    # so the first measured call is not pathological.
    for _ in range(3):
        fire_fn()
    latencies: list[float] = []
    wall_start = time.monotonic()
    if concurrency == 1:
        for _ in range(iterations):
            latencies.append(fire_fn())
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(fire_fn) for _ in range(iterations)]
            for f in cf.as_completed(futures):
                latencies.append(f.result())
    wall_ms = (time.monotonic() - wall_start) * 1000.0
    latencies.sort()
    n = len(latencies)
    return {
        "concurrency": concurrency,
        "iterations": iterations,
        "wall_ms": round(wall_ms, 1),
        "throughput_req_per_s": round(n / (wall_ms / 1000.0), 2),
        "p50_ms": round(_percentile(latencies, 0.50), 1),
        "p90_ms": round(_percentile(latencies, 0.90), 1),
        "p99_ms": round(latencies[-1] if n < 100 else _percentile(latencies, 0.99), 1),
        "avg_ms": round(statistics.mean(latencies), 1),
        "worst_ms": round(latencies[-1], 1),
    }


# ---------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------

def _print_table(results: list[dict]) -> None:
    print("| conc | iters | wall_ms | tps   | p50  | p90   | p99   | worst |")
    print("|------|-------|---------|-------|------|-------|-------|-------|")
    for r in results:
        print(
            f"| {r['concurrency']:>4} | {r['iterations']:>5} | "
            f"{r['wall_ms']:>7.1f} | {r['throughput_req_per_s']:>5.2f} | "
            f"{r['p50_ms']:>4.0f} | {r['p90_ms']:>5.0f} | "
            f"{r['p99_ms']:>5.0f} | {r['worst_ms']:>5.0f} |"
        )


def _parse_scenarios(spec: str) -> list[tuple[int, int]]:
    """Parse `1x20,4x40,...` into [(1, 20), (4, 40), ...]."""
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        c, _, i = chunk.strip().partition("x")
        out.append((int(c), int(i)))
    return out


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------

DEFAULT_SCENARIOS_EMBED = "1x20,4x40,10x40,20x40,40x40"
# Rerank-specific defaults mirror the real memdb-go consumer pattern:
# `MAX_CONCURRENT_RERANK_REQUESTS=4` server-side cap → benching above c=4
# only measures 429 backoff. D7 sub-query fanout produces 3-5 parallel
# calls per chat turn, so c=1/4/10 covers the operational envelope.
DEFAULT_SCENARIOS_RERANK = "1x20,4x40,10x40"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kind",
        choices=["text", "code", "rerank"],
        default="text",
        help="text/code → POST /v1/embeddings; rerank → POST /v1/rerank",
    )
    p.add_argument("--size", choices=["short", "medium", "long"], default="medium")
    p.add_argument(
        "--url",
        default=None,
        help="Defaults: /v1/embeddings for text|code, /v1/rerank for rerank.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Defaults: multilingual-e5-large for text, jina-code-v2 for code, gte-multi-rerank for rerank.",
    )
    p.add_argument(
        "--texts-per-req",
        type=int,
        default=8,
        help="Texts per /v1/embeddings request. Ignored for --kind rerank.",
    )
    p.add_argument(
        "--docs-per-req",
        type=int,
        default=5,
        help="Documents per /v1/rerank request. Ignored for --kind text|code. "
        "Default 5 mirrors memdb-go's typical D7 sub-query candidate count.",
    )
    p.add_argument(
        "--scenarios",
        default=None,
        help='Comma-separated "concurrency x iterations" pairs, e.g. "1x20,4x40". '
        "Defaults differ by --kind (rerank caps at c=10 since prod semaphore is 4).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON document instead of the markdown table. Useful for diffing.",
    )
    args = p.parse_args()

    snippet = FIXTURES[(args.kind, args.size)]

    if args.kind == "rerank":
        url = args.url or "http://localhost:8082/v1/rerank"
        model = args.model or "gte-multi-rerank"
        docs = [snippet] * args.docs_per_req
        fire_fn = lambda: _fire_rerank(url, model, RERANK_QUERY, docs)
        scenarios_spec = args.scenarios or DEFAULT_SCENARIOS_RERANK
        request_shape = {
            "endpoint": "/v1/rerank",
            "docs_per_req": args.docs_per_req,
            "query_chars": len(RERANK_QUERY),
            "doc_chars": len(snippet),
        }
    else:
        url = args.url or "http://localhost:8082/v1/embeddings"
        model = args.model or (
            "multilingual-e5-large" if args.kind == "text" else "jina-code-v2"
        )
        texts = [snippet] * args.texts_per_req
        fire_fn = lambda: _fire_embed(url, model, texts)
        scenarios_spec = args.scenarios or DEFAULT_SCENARIOS_EMBED
        request_shape = {
            "endpoint": "/v1/embeddings",
            "texts_per_req": args.texts_per_req,
            "text_chars": len(snippet),
        }

    scenarios = _parse_scenarios(scenarios_spec)
    results = [run(fire_fn, c, i) for c, i in scenarios]

    if args.json:
        print(
            json.dumps(
                {
                    "kind": args.kind,
                    "size": args.size,
                    "url": url,
                    "model": model,
                    "request_shape": request_shape,
                    "scenarios": results,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        _print_table(results)


if __name__ == "__main__":
    main()
