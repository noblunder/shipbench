#!/usr/bin/env python3
"""Validation diagnostics for frontier predictions (per GPT review feedback).

Detects 5 prediction quality issues:
1. Empty prediction (reasoning consumed all tokens)
2. Refusal / apology / "cannot determine" (model declined)
3. Parse failure (no numeric value extractable)
4. Out-of-range value (e.g., B4 m² > 10^4 = sanity-fail)
5. Truncation (finish_reason='length')

Usage:
  python scripts/validate_predictions.py --pred outputs/frontier_eval/gpt-5_main.jsonl

Saves quality report + flags items needing review.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


# Refusal / apology / "cannot determine" patterns
REFUSAL_PATTERNS = [
    r"\bi\s+(?:cannot|can'?t|am unable|don'?t (?:know|have))\b",
    r"\bunable to\b",
    r"\bcannot determine\b",
    r"\bnot (?:enough|sufficient) (?:information|data|context)\b",
    r"\bsorry,?\s+but\b",
    r"\bI apologize\b",
    r"\bI don'?t see\b",
    r"\bno (?:information|data) (?:provided|available)\b",
    r"\bplease provide\b",
    r"\bcould you (?:provide|share|clarify)\b",
    r"\bthe image (?:does not|doesn'?t) (?:show|contain)\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)

# Numeric extraction (matches eval_v3_unit_aware.py)
NUM_UNIT_RE = re.compile(
    r"(-?\d+(?:[\.,]\d+)?(?:\s*[eE][+-]?\d+)?)\s*"
    r"(m\^[23]|m[²³]|m[23]|mm\^[23]|mm[²³]|mm[23]|mm|m|cm\^[23]|cm[²³]|cm[23])?"
)

# Sanity ranges for values (GT-derived from ShipBench)
SANITY_RANGES = {
    "B1_plate_thickness":   (5, 100),       # mm; reasonable plate thicknesses
    "B2_stiffener_size":    (50, 1500),     # mm; stiffener heights
    "B3_cargo_capacity_v1": (5000, 500000), # m^3; per-hold cargo volume
    "B3_cargo_capacity_v3": (5000, 500000),
    "B4_section_area_v1":   (0.1, 50),      # m^2; named-plate half-section area
    "B4_section_area_v3_cot": (0.1, 50),
    "C3_bulkhead_position":     (1000, 500000),  # mm; ship length range
    "C3_bulkhead_position_v3":  (1000, 500000),
}


def classify(pred: str, task: str, finish_reason: str | None = None) -> dict:
    """Classify a prediction into quality tiers."""
    pred = (pred or "").strip()

    flags = {
        "empty": False,
        "refusal": False,
        "parse_fail": False,
        "out_of_range": False,
        "truncated": False,
        "ok": False,
    }

    if not pred:
        flags["empty"] = True
        return flags

    if finish_reason == "length":
        flags["truncated"] = True
        # may still be partially usable — fall through to other checks

    # Refusal detection
    if REFUSAL_RE.search(pred):
        flags["refusal"] = True
        return flags

    # Parse: extract last number+unit (smart, mirrors eval_v3_unit_aware.py)
    matches = list(NUM_UNIT_RE.finditer(pred))
    last_with_unit = None
    for m in matches:
        if m.group(2):
            last_with_unit = m
    last_match = last_with_unit if last_with_unit else (matches[-1] if matches else None)

    if not last_match:
        flags["parse_fail"] = True
        return flags

    # Value sanity (only for numeric tasks with ranges)
    if task in SANITY_RANGES:
        try:
            val = float(last_match.group(1).replace(",", ""))
            lo, hi = SANITY_RANGES[task]
            if not (lo <= val <= hi):
                flags["out_of_range"] = True
        except ValueError:
            flags["parse_fail"] = True

    # If no failure flags set, it's OK
    if not (flags["empty"] or flags["refusal"] or flags["parse_fail"] or flags["out_of_range"]):
        flags["ok"] = True

    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output", default=None, help="Save full report JSON")
    args = ap.parse_args()

    preds = [json.loads(l) for l in open(args.pred)]
    n_total = len(preds)

    counts = Counter()
    by_task = {}
    flagged_items = []
    completion_tokens = []
    reasoning_tokens = []

    for p in preds:
        task = p.get("task", "?")
        flags = classify(p.get("prediction", ""), task, p.get("finish_reason"))
        primary_flag = next(
            (k for k in ["empty", "refusal", "parse_fail", "out_of_range", "truncated", "ok"]
             if flags[k]), "unknown")

        counts[primary_flag] += 1
        by_task.setdefault(task, Counter())[primary_flag] += 1

        if not flags["ok"]:
            flagged_items.append({
                "qa_id": p["qa_id"],
                "task": task,
                "prediction": p["prediction"][:200],
                "flags": [k for k, v in flags.items() if v and k != "ok"],
                "finish_reason": p.get("finish_reason"),
                "reasoning_tokens": p.get("reasoning_tokens"),
                "completion_tokens": p.get("completion_tokens"),
            })

        if p.get("completion_tokens"):
            completion_tokens.append(p["completion_tokens"])
        if p.get("reasoning_tokens"):
            reasoning_tokens.append(p["reasoning_tokens"])

    # Summary
    print(f"=== Validation report: {Path(args.pred).name} ===")
    print(f"Total predictions: {n_total}")
    print()
    print(f"{'Status':>16s} | count | %")
    print("-" * 35)
    for status in ["ok", "empty", "refusal", "parse_fail", "out_of_range", "truncated"]:
        n = counts.get(status, 0)
        pct = 100 * n / n_total if n_total else 0
        print(f"{status:>16s} | {n:5d} | {pct:5.1f}")

    print()
    print(f"{'Task':30s} | OK | empty | refusal | parse_fail | out_of_range | truncated")
    print("-" * 105)
    for task in sorted(by_task):
        c = by_task[task]
        print(f"{task:30s} | {c.get('ok',0):2d} | {c.get('empty',0):5d} | {c.get('refusal',0):7d} | {c.get('parse_fail',0):10d} | {c.get('out_of_range',0):12d} | {c.get('truncated',0):9d}")

    print()
    if completion_tokens:
        print(f"Completion tokens: avg={sum(completion_tokens)/len(completion_tokens):.0f}, max={max(completion_tokens)}")
    if reasoning_tokens:
        print(f"Reasoning tokens: avg={sum(reasoning_tokens)/len(reasoning_tokens):.0f}, max={max(reasoning_tokens)}")

    if flagged_items:
        print(f"\nFlagged items (sample of first 10):")
        for it in flagged_items[:10]:
            print(f"  {it['qa_id']} ({it['task']}): {it['flags']}")
            print(f"    pred={it['prediction'][:100]!r}")

    if args.output:
        report = {
            "n_total": n_total,
            "counts": dict(counts),
            "by_task": {t: dict(c) for t, c in by_task.items()},
            "completion_tokens_avg": sum(completion_tokens)/len(completion_tokens) if completion_tokens else None,
            "reasoning_tokens_avg": sum(reasoning_tokens)/len(reasoning_tokens) if reasoning_tokens else None,
            "flagged_items": flagged_items,
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"\nReport saved → {args.output}")


if __name__ == "__main__":
    main()
