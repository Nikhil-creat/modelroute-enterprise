"""Pick the router threshold on the validation split.

Local (agent) predictions and frontier predictions are collected once per ticket and cached;
every threshold on the grid is then evaluated offline: ticket i falls back when
C_total_i < threshold. The chosen threshold is the cheapest one whose quality retention
(accuracy / frontier-only accuracy) meets ``--min-retention``; it is written to
artifacts/reports/threshold.json, which the router loads automatically.

Usage:
    python -m evaluation.calibrate_threshold --min-retention 98
"""
from __future__ import annotations

import argparse
import csv
import json
from typing import Any

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

import config
from agentic_workflow.agent_router import build_messages, parse_decision
from evaluation.benchmark import exact_match, macro_f1
from router.hybrid_router import FrontierLLM, HybridRouter, frontier_cost, local_cost

CACHE_PATH = config.REPORT_DIR / "calibration_cache.jsonl"


def collect(rows: list[dict[str, Any]], router: HybridRouter, frontier: FrontierLLM) -> list[dict[str, Any]]:
    cache = {r["id"]: r for r in config.read_jsonl(CACHE_PATH)} if CACHE_PATH.exists() else {}
    with open(CACHE_PATH, "a", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="collect"):
            if row["id"] in cache:
                continue
            image = str(config.resolve_data_path(row["image_path"])) if row["image_path"] else None
            local = router.route(row["ticket_text"], row["customer_tier"], image, mode="agent",
                                 threshold=0.0, allow_fallback=False)
            rec: dict[str, Any] = {
                "id": row["id"], "c_total": local.confidence.c_total, "local": local.decision,
                "local_latency": local.telemetry["latency_ms"],
                "local_cost": local_cost(local.agent.tokens_in, local.agent.tokens_out),
                "frontier": None, "frontier_latency": 0.0, "frontier_cost": 0.0}
            try:
                res = frontier.generate(local.agent.messages)
                rec.update(frontier=parse_decision(res.text), frontier_latency=res.latency_ms,
                           frontier_cost=frontier_cost(frontier.model, res.tokens_in, res.tokens_out))
            except Exception as exc:
                print(f"frontier failed for {row['id']}: {exc}")
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            cache[row["id"]] = rec
    return [cache[r["id"]] for r in rows]


def pure_frontier_accuracy(rows, recs) -> float:
    return 100.0 * sum(exact_match(r["frontier"], row) for r, row in zip(recs, rows)) / max(1, len(rows))


def evaluate(rows, recs, threshold: float, frontier_acc: float) -> dict[str, Any]:
    decisions, cost, latency, fallbacks = [], [], [], 0
    for rec in recs:
        use_frontier = rec["c_total"] < threshold and rec["frontier"] is not None
        decisions.append(rec["frontier"] if use_frontier else rec["local"])
        cost.append(rec["local_cost"] + (rec["frontier_cost"] if use_frontier else 0.0))
        latency.append(rec["local_latency"] + (rec["frontier_latency"] if use_frontier else 0.0))
        fallbacks += int(use_frontier)
    n = max(1, len(rows))
    acc = 100.0 * sum(exact_match(d, r) for d, r in zip(decisions, rows)) / n
    return {"threshold": threshold, "accuracy": acc,
            "macro_f1": 100.0 * macro_f1([r["intent"] for r in rows],
                                         [(d or {}).get("intent", "none") for d in decisions], config.INTENTS),
            "fallback_rate_pct": 100.0 * fallbacks / n, "cost_per_1k_usd": 1000.0 * sum(cost) / n,
            "p95_latency_ms": float(np.percentile(latency, 95)),
            "quality_retention_pct": 100.0 * acc / frontier_acc if frontier_acc else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-retention", type=float, default=98.0, help="minimum quality retention %%")
    args = parser.parse_args()

    config.ensure_dirs()
    rows = config.read_jsonl(config.VAL_PATH)[: args.limit or None]
    router = HybridRouter.from_config(threshold=0.0)
    frontier = router.frontier or FrontierLLM.try_create()
    if frontier is None:
        raise SystemExit("set OPENAI_API_KEY, ANTHROPIC_API_KEY or GROQ_API_KEY (and MR_FRONTIER_PROVIDER)")
    recs = collect(rows, router, frontier)

    frontier_acc = pure_frontier_accuracy(rows, recs)
    grid = [evaluate(rows, recs, t, frontier_acc) for t in config.THRESHOLD_GRID + [1.01]]
    headers = ["threshold", "accuracy", "macro_f1", "fallback_rate_pct", "cost_per_1k_usd", "p95_latency_ms",
               "quality_retention_pct"]
    print(tabulate([[round(g[h], 3) for h in headers] for g in grid], headers=headers, tablefmt="github"))
    print(f"\nfrontier-only accuracy: {frontier_acc:.2f}%")

    feasible = [g for g in grid if g["quality_retention_pct"] >= args.min_retention]
    best = min(feasible, key=lambda g: g["cost_per_1k_usd"]) if feasible else max(grid, key=lambda g: g["accuracy"])
    if not feasible:
        print(f"no threshold reached {args.min_retention}% retention; using the most accurate point")
    print(f"\nchosen threshold: {best['threshold']}")

    with open(config.REPORT_DIR / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows({h: g[h] for h in headers} for g in grid)
    config.THRESHOLD_FILE.write_text(json.dumps(
        {"threshold": min(best["threshold"], 1.0), "min_retention": args.min_retention, "split": "val",
         "n": len(rows), "frontier_only_accuracy": frontier_acc, "metrics": best}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
