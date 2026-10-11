"""Stratified train/val/test split plus chat-format SFT files for QLoRA.

Reads  : data/tickets.jsonl, data/kb/knowledge_base.jsonl
Writes : data/splits/{train,val,test}.jsonl
         data/splits/{sft_train,sft_val}.jsonl   (messages format for TRL SFTTrainer)
         data/splits/split_report.json

The SFT user prompt matches the one used at inference (config.USER_PROMPT_TEMPLATE): the
KB context lists the gold article(s) plus one random distractor so the model learns to
ignore irrelevant context. The image diagnosis line uses the ground-truth visual label with
a simulated CNN confidence; at serving time it is replaced by the real CNN output.

Usage:
    python -m data_pipeline.split_data
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from typing import Any, Callable

import config


def stratified_split(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str],
                     ratios: tuple[float, float, float], seed: int):
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        rng.shuffle(group)
        n_train = int(round(len(group) * ratios[0]))
        n_val = int(round(len(group) * ratios[1]))
        train += group[:n_train]
        val += group[n_train:n_train + n_val]
        test += group[n_train + n_val:]
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def build_kb_context(row: dict[str, Any], kb_by_id: dict[str, dict[str, Any]], rng: random.Random) -> str:
    ids = list(row["knowledge_base_ids"])
    distractors = [k for k in kb_by_id if k not in ids]
    ids.append(rng.choice(distractors))
    rng.shuffle(ids)
    return "\n".join(f"- [{kb_id}] {kb_by_id[kb_id]['title']}" for kb_id in ids)


def sft_example(row: dict[str, Any], kb_by_id: dict[str, dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    if row["visual_label"]:
        visual = f"{row['visual_label']} (confidence {rng.uniform(0.90, 0.99):.2f})"
    else:
        visual = "none (no attachment)"
    user = config.USER_PROMPT_TEMPLATE.format(
        tier=row["customer_tier"], visual=visual,
        kb_context=build_kb_context(row, kb_by_id, rng), ticket_text=row["ticket_text"])
    target = {"intent": row["intent"], "priority": row["priority"],
              "target_department": row["target_department"]}
    return {
        "id": row["id"],
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(target)},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--ratios", type=float, nargs=3, default=list(config.SPLIT_RATIOS),
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        parser.error("ratios must sum to 1.0")

    config.ensure_dirs()
    tickets = config.read_jsonl(config.TICKETS_PATH)
    kb_by_id = {a["kb_id"]: a for a in config.read_jsonl(config.KB_PATH)}
    train, val, test = stratified_split(tickets, lambda r: r["intent"], tuple(args.ratios), args.seed)

    config.write_jsonl(config.TRAIN_PATH, train)
    config.write_jsonl(config.VAL_PATH, val)
    config.write_jsonl(config.TEST_PATH, test)

    rng = random.Random(args.seed + 1)
    config.write_jsonl(config.SFT_TRAIN_PATH, (sft_example(r, kb_by_id, rng) for r in train))
    config.write_jsonl(config.SFT_VAL_PATH, (sft_example(r, kb_by_id, rng) for r in val))

    report = {
        "total": len(tickets), "train": len(train), "val": len(val), "test": len(test),
        "intent_distribution": {
            name: dict(sorted(Counter(r["intent"] for r in part).items()))
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "hard_tickets": {name: sum(r["difficulty"] == "hard" for r in part)
                         for name, part in (("train", train), ("val", val), ("test", test))},
    }
    (config.SPLIT_DIR / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("total", "train", "val", "test", "hard_tickets")}, indent=2))


if __name__ == "__main__":
    main()
