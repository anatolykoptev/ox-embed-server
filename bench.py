# /home/krolik/src/embed-server/bench.py
"""Benchmark harness for embed-server. Runs sequential + concurrent workloads."""
import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request


RUSSIAN_SHORT = "Санкт-Петербург — культурная столица России."
RUSSIAN_MEDIUM = (
    "Санкт-Петербург — город федерального значения и культурная столица России. "
    "Основан Петром I в 1703 году как крепость Санкт-Питер-Бурх на Заячьем острове. "
    "Город отличается уникальной архитектурой, многочисленными каналами и мостами, а также богатым "
    "культурным наследием, включая Эрмитаж, Петропавловскую крепость и Казанский собор."
) * 2
CODE_SHORT = 'func main() { fmt.Println("hello") }'
CODE_MEDIUM = (
    'package main\nimport "fmt"\n'
    'func calculate(x, y int) int {\n    result := x*y + 42\n    if result > 100 { return result }\n    return 0\n}\n'
    'func main() {\n    for i := 0; i < 100; i++ { fmt.Println(calculate(i, i+1)) }\n}\n'
) * 2


def _fire_one(url: str, model: str, texts: list[str]) -> float:
    data = json.dumps({"input": texts, "model": model}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return (time.monotonic() - t0) * 1000.0


def run(url: str, model: str, texts: list[str], concurrency: int, iterations: int) -> dict:
    for _ in range(3):
        _fire_one(url, model, texts)
    latencies = []
    wall_start = time.monotonic()
    if concurrency == 1:
        for _ in range(iterations):
            latencies.append(_fire_one(url, model, texts))
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [ex.submit(_fire_one, url, model, texts) for _ in range(iterations)]
            for f in cf.as_completed(futures):
                latencies.append(f.result())
    wall_ms = (time.monotonic() - wall_start) * 1000.0
    latencies.sort()
    n = len(latencies)
    return {
        "concurrency": concurrency, "iterations": iterations,
        "wall_ms": round(wall_ms, 1),
        "throughput_req_per_s": round(n / (wall_ms / 1000.0), 2),
        "p50_ms": round(latencies[n // 2], 1),
        "p90_ms": round(latencies[int(n * 0.9)], 1),
        "p99_ms": round(latencies[int(n * 0.99)], 1),
        "avg_ms": round(statistics.mean(latencies), 1),
        "worst_ms": round(latencies[-1], 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8082/v1/embeddings")
    p.add_argument("--model", default="multilingual-e5-large")
    p.add_argument("--size", choices=["short", "medium"], default="medium")
    p.add_argument("--kind", choices=["text", "code"], default="text")
    p.add_argument("--texts-per-req", type=int, default=8)
    args = p.parse_args()
    snippet = {
        ("text", "short"): RUSSIAN_SHORT, ("text", "medium"): RUSSIAN_MEDIUM,
        ("code", "short"): CODE_SHORT, ("code", "medium"): CODE_MEDIUM,
    }[(args.kind, args.size)]
    texts = [snippet] * args.texts_per_req
    scenarios = [(1, 20), (4, 40), (10, 40), (20, 40), (40, 40)]
    results = [run(args.url, args.model, texts, c, i) for c, i in scenarios]
    print(f"| conc | iters | wall_ms | tps   | p50  | p90   | p99   | worst |")
    print(f"|------|-------|---------|-------|------|-------|-------|-------|")
    for r in results:
        print(
            f"| {r['concurrency']:>4} | {r['iterations']:>5} | "
            f"{r['wall_ms']:>7.1f} | {r['throughput_req_per_s']:>5.2f} | "
            f"{r['p50_ms']:>4.0f} | {r['p90_ms']:>5.0f} | "
            f"{r['p99_ms']:>5.0f} | {r['worst_ms']:>5.0f} |"
        )


if __name__ == "__main__":
    main()
