"""Benchmark three serving modes on the held-out test split.

  1. Pure Frontier Model            (frontier sees raw text only)
  2. Pure Local Fine-Tuned + CNN    (direct mode, no fallback)
  3. ModelRoute Enterprise          (agentic RAG + CNN + dynamic hybrid router)

Metrics: Accuracy (intent+priority+department exact match), Macro-F1 (intent), p95 latency,
total tokens, cost per 1,000 requests, fallback rate and quality retention vs. the frontier.
Outputs: artifacts/reports/benchmark.{json,md}, docs/data/benchmark.json (website) and
artifacts/reports/resume_bullets.md (filled with *measured* numbers only).

Usage:
    python -m evaluation.benchmark --limit 300 --threshold 0.85
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import numpy as np

import config
from agentic_workflow.agent_router import build_messages, parse_decision
from router.hybrid_router import FrontierLLM, HybridRouter, frontier_cost

FIELDS = ("intent", "priority", "target_department")


def exact_match(decision: dict[str, str] | None, row: dict[str, Any]) -> bool:
    return bool(decision) and all(decision.get(f) == row[f] for f in FIELDS)


def macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def summarize(name: str, rows: list[dict[str, Any]], outcomes: list[tuple[dict | None, dict]]) -> dict[str, Any]:
    decisions = [d for d, _ in outcomes]
    tel = [t for _, t in outcomes]
    n = max(1, len(rows))
    return {
        "mode": name, "n": len(rows),
        "accuracy": 100.0 * sum(exact_match(d, r) for d, r in zip(decisions, rows)) / n,
        "macro_f1": 100.0 * macro_f1([r["intent"] for r in rows],
                                     [(d or {}).get("intent", "none") for d in decisions], config.INTENTS),
        "p95_latency_ms": float(np.percentile([t["latency_ms"] for t in tel], 95)) if tel else 0.0,
        "total_tokens": int(sum(t["tokens_in"] + t["tokens_out"] for t in tel)),
        "cost_per_1k_usd": 1000.0 * sum(t["cost_usd"] for t in tel) / n,
        "fallback_rate_pct": 100.0 * sum(t["route"] == "frontier" for t in tel) / n,
    }


def run_mode(name: str, rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], tuple[dict | None, dict]]):
    from tqdm import tqdm

    outcomes = []
    for row in tqdm(rows, desc=name):
        try:
            outcomes.append(fn(row))
        except Exception as exc:  # count a failed request as an incorrect, zero-cost miss
            print(f"[{name}] {row['id']} failed: {exc}")
            outcomes.append((None, {"route": "local", "latency_ms": 0.0, "tokens_in": 0, "tokens_out": 0,
                                    "cost_usd": 0.0}))
    return summarize(name, rows, outcomes)


def pct_change(new: float, old: float) -> float:
    return 100.0 * (1.0 - new / old) if old else 0.0


def resume_block(stats: list[dict[str, Any]], threshold: float) -> str:
    frontier, local, route = stats
    cost_cut = pct_change(route["cost_per_1k_usd"], frontier["cost_per_1k_usd"])
    lat_cut = pct_change(route["p95_latency_ms"], frontier["p95_latency_ms"])
    lat_phrase = (f"cut p95 latency by {lat_cut:.0f}%" if lat_cut >= 0
                  else f"kept p95 latency within {abs(lat_cut):.0f}% of the frontier baseline")
    return "\n".join([
        f"**{config.PROJECT_NAME}** | {config.AUTHOR_TAGLINE}",
        f"- Built a multimodal ticket-routing platform (4-bit QLoRA LLM + PyTorch CNN + agentic hybrid RAG) that "
        f"cut cost per 1,000 requests by {cost_cut:.0f}% vs. a pure frontier model "
        f"(${frontier['cost_per_1k_usd']:.2f} -> ${route['cost_per_1k_usd']:.2f}).",
        f"- Designed a confidence router (text log-prob + CNN + RAG relevance, threshold {threshold:.2f}) that served "
        f"{100 - route['fallback_rate_pct']:.0f}% of traffic locally, {lat_phrase}.",
        f"- Retained {100.0 * route['accuracy'] / frontier['accuracy'] if frontier['accuracy'] else 0:.0f}% of frontier "
        f"accuracy ({route['accuracy']:.1f}% vs {frontier['accuracy']:.1f}%, macro-F1 {route['macro_f1']:.1f}) "
        f"on {route['n']} held-out tickets.",
        "- Shipped FastAPI + Docker (vLLM, ChromaDB) with per-request latency and cost telemetry.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="default: calibrated or 0.85")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    from tabulate import tabulate

    config.ensure_dirs()
    config.DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.TEST_PATH if args.split == "test" else config.VAL_PATH
    rows = config.read_jsonl(path)[: args.limit or None]
    router = HybridRouter.from_config(args.threshold)
    frontier = router.frontier or FrontierLLM.try_create()
    if frontier is None:
        raise SystemExit("set OPENAI_API_KEY, ANTHROPIC_API_KEY or GROQ_API_KEY (and MR_FRONTIER_PROVIDER)")
    threshold = router.threshold

    def image_of(row: dict[str, Any]) -> str | None:
        return str(config.resolve_data_path(row["image_path"])) if row["image_path"] else None

    def pure_frontier(row):
        msgs = build_messages(row["ticket_text"], row["customer_tier"], "not available", "- (none provided)")
        res = frontier.generate(msgs)
        tel = {"route": "frontier", "latency_ms": res.latency_ms, "tokens_in": res.tokens_in,
               "tokens_out": res.tokens_out, "cost_usd": frontier_cost(frontier.model, res.tokens_in, res.tokens_out)}
        return parse_decision(res.text), tel

    def local_cnn(row):
        r = router.route(row["ticket_text"], row["customer_tier"], image_of(row), mode="direct", allow_fallback=False)
        return r.decision, r.telemetry

    def modelroute(row):
        r = router.route(row["ticket_text"], row["customer_tier"], image_of(row), mode="agent",
                         threshold=threshold, allow_fallback=True)
        return r.decision, r.telemetry

    stats = [run_mode("Pure Frontier Model", rows, pure_frontier),
             run_mode("Pure Local Fine-Tuned + CNN", rows, local_cnn),
             run_mode("ModelRoute Enterprise", rows, modelroute)]
    base_acc = stats[0]["accuracy"] or 1.0
    for s in stats:
        s["quality_retention_pct"] = 100.0 * s["accuracy"] / base_acc

    table = [[s["mode"], f"{s['accuracy']:.1f}", f"{s['macro_f1']:.1f}", f"{s['p95_latency_ms']:.0f}",
              s["total_tokens"], f"{s['cost_per_1k_usd']:.3f}", f"{s['fallback_rate_pct']:.1f}",
              f"{s['quality_retention_pct']:.1f}"] for s in stats]
    headers = ["Mode", "Accuracy %", "Macro-F1", "p95 ms", "Total tokens", "$/1k req", "Fallback %", "Retention %"]
    md = tabulate(table, headers=headers, tablefmt="github")
    print("\n" + md)

    payload = {"status": "measured", "split": args.split, "n": len(rows), "threshold": threshold,
               "frontier_model": frontier.model, "author": config.AUTHOR, "modes": stats}
    (config.REPORT_DIR / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (config.REPORT_DIR / "benchmark.md").write_text(md + "\n", encoding="utf-8")
    (config.DOCS_DATA_DIR / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    bullets = resume_block(stats, threshold)
    (config.REPORT_DIR / "resume_bullets.md").write_text(bullets + "\n", encoding="utf-8")
    print("\nResume Bullet Block (measured):\n" + bullets)


if __name__ == "__main__":
    main()
