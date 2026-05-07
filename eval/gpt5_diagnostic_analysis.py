#!/usr/bin/env python3
"""GPT-5 per-task classification + Pitfall 10 diagnostic.

Reads the GPT-5 paired-with-Opus jsonl produced by launch_gpt5_paired.sh,
classifies EACH of the 9 tasks as either:

  - MAIN     (refusal_rate ≤ 20%): include in main accuracy table, paired with Opus
  - DIAGNOSTIC (refusal_rate > 20%): exclude from main; report refusal pattern only

Per task, computes:
  - refusal/empty/parse_fail/oor/ok counts
  - Wilson 95% CI on each rate
  - on the 'ok' (non-refusal, parseable, in-range) subset:
      * GPT-5 accuracy with Clopper-Pearson 95% CI
      * paired vs Opus accuracy (if --opus provided): McNemar p, matched-pair OR
  - average reasoning_tokens / completion_tokens
  - paper-ready text snippets

Usage:
  python scripts/gpt5_diagnostic_analysis.py \
      --pred outputs/frontier_eval/gpt-5_main_paired.jsonl \
      --opus outputs/frontier_eval/claude_opus_main.jsonl \
      --output outputs/main_eval/gpt5_per_task_classification.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


# Mirror validate_predictions.py / paired_frontier_analysis.py
NUM_UNIT_RE = re.compile(
    r"(-?\d+(?:[\.,]\d+)?(?:\s*[eE][+-]?\d+)?)\s*"
    r"(m\^[23]|m[²³]|m[23]|mm\^[23]|mm[²³]|mm[23]|mm|m|cm\^[23]|cm[²³]|cm[23])?"
)
REFUSAL_PATTERNS = [
    r"\bi\s+(?:cannot|can'?t|am unable|don'?t (?:know|have))\b",
    r"\bunable to\b",
    r"\bcannot determine\b",
    r"\bnot (?:enough|sufficient) (?:information|data|context)\b",
    r"\bsorry,?\s+but\b",
    r"\bI apologize\b",
    r"\bplease provide\b",
    r"\bcould you (?:provide|share|clarify)\b",
    r"\bhigher[- ]resolution\b",
    r"\bdimensioned image\b",
    r"\bthe image (?:does not|doesn'?t) (?:show|contain)\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)

SANITY_RANGES = {
    "B1_plate_thickness":   (5, 100),
    "B2_stiffener_size":    (50, 1500),
    "B3_cargo_capacity_v1": (5000, 500000),
    "B3_cargo_capacity_v3": (5000, 500000),
    "B4_section_area_v1":   (0.1, 50),
    "B4_section_area_v3_cot": (0.1, 50),
    "C3_bulkhead_position":     (1000, 500000),
    "C3_bulkhead_position_v3":  (1000, 500000),
}
NUMERIC_TASKS = {**SANITY_RANGES, **{
    "B3_cargo_capacity_v1": (5000, 500000),
}}
MCQ_TASKS = {"A1_shiptype", "A1_shiptype_section_only", "A2_stiffener_type",
             "C1_compartment_locate", "C2_compartment_boundary"}

# Tolerance % used by the unit-aware grader (mirrors paired_frontier_analysis.NUMERIC_TASKS)
TOLERANCE_PCT = {
    "B1_plate_thickness": 5.0, "B2_stiffener_size": 5.0,
    "B3_cargo_capacity_v1": 10.0, "B3_cargo_capacity_v3": 10.0,
    "B4_section_area_v1": 10.0, "B4_section_area_v3_cot": 10.0,
    "C3_bulkhead_position": 10.0, "C3_bulkhead_position_v3": 10.0,
}

# Per-task threshold for "useful for main accuracy table" classification.
# refusal+empty+parse_fail+oor (= bad outcomes) > REFUSAL_THRESHOLD ⇒ DIAGNOSTIC only.
#
# 20% pre-specified rationale (paper text):
#   "We pre-specify a 20% bad-output threshold: above this level, accuracy is
#    dominated by interface compliance rather than task performance, so the task
#    is reported diagnostically rather than as a main accuracy estimate."
#
# Constant committed before GPT-5.5 main inference launch on 2026-05-07.
REFUSAL_THRESHOLD = 0.20


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    except ImportError:
        lo, hi = wilson_ci(k, n, alpha)
    return lo, hi


def extract_value(pred: str):
    """Last number+unit, fall back to last number."""
    matches = list(NUM_UNIT_RE.finditer(pred))
    last_with_unit = next((m for m in reversed(matches) if m.group(2)), None)
    last = last_with_unit if last_with_unit else (matches[-1] if matches else None)
    if not last:
        return None, None
    try:
        v = float(last.group(1).replace(",", ""))
    except ValueError:
        return None, None
    return v, (last.group(2) or "").strip().lower()


def grade_pred(pred: str, gt_item: dict) -> int | None:
    """Return 1=correct, 0=wrong, None=unparseable. Mirrors paired_frontier_analysis."""
    task = gt_item.get("task")
    pred = (pred or "").strip()
    if task in MCQ_TASKS:
        m = re.search(r"\b([A-F])\b", pred.upper())
        if not m:
            return None
        return int(m.group(1) == gt_item["answer"].strip().upper())
    if task in NUMERIC_TASKS:
        v, _ = extract_value(pred)
        if v is None:
            return None
        gt_val = float(gt_item["metadata"]["value"])
        tol = gt_item["metadata"].get("tolerance_pct", TOLERANCE_PCT.get(task, 10.0))
        rel_err = abs(v - gt_val) / max(abs(gt_val), 1e-9)
        return int(rel_err <= tol / 100.0)
    return None


def classify_outcome(pred: str, task: str, finish_reason: str | None = None) -> str:
    """Bucket each prediction. Returns one of: empty, refusal, parse_fail, oor, ok."""
    pred = (pred or "").strip()
    if not pred:
        return "empty"
    if REFUSAL_RE.search(pred):
        return "refusal"
    if task in MCQ_TASKS:
        return "ok" if re.search(r"\b[A-F]\b", pred.upper()) else "parse_fail"
    v, _ = extract_value(pred)
    if v is None:
        return "parse_fail"
    if task in SANITY_RANGES:
        lo, hi = SANITY_RANGES[task]
        if not (lo <= v <= hi):
            return "oor"
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="GPT-5 prediction JSONL")
    ap.add_argument("--opus", default=None, help="Opus prediction JSONL (optional, for paired comparison)")
    ap.add_argument("--gt-files", nargs="*", default=[
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_main_eval.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_main_eval_opus_paired.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_B3_v3_clarified.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_B4_v3_cot.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_C3_v3_clarified.jsonl",
    ])
    ap.add_argument("--output", default=None, help="Output JSON (per-task classification)")
    ap.add_argument("--threshold", type=float, default=REFUSAL_THRESHOLD,
                    help="Refusal-rate threshold for MAIN vs DIAGNOSTIC classification (default 0.20)")
    args = ap.parse_args()

    # Load GT items
    gt_by_id = {}
    for f in args.gt_files:
        if Path(f).exists():
            for line in open(f):
                d = json.loads(line)
                gt_by_id.setdefault(d["qa_id"], d)

    preds = [json.loads(l) for l in open(args.pred)]
    opus_by_id = {}
    if args.opus and Path(args.opus).exists():
        for l in open(args.opus):
            d = json.loads(l)
            opus_by_id[d["qa_id"]] = d

    # Per-task tally
    by_task = defaultdict(list)
    for p in preds:
        by_task[p["task"]].append(p)

    results = {
        "meta": {
            "pred_file": args.pred,
            "opus_file": args.opus,
            "n_total": len(preds),
            "tasks": sorted(by_task.keys()),
            "refusal_threshold": args.threshold,
        },
        "per_task": {},
        "summary": {"main_table_tasks": [], "diagnostic_only_tasks": []},
    }

    print(f"=== GPT-5 per-task classification ===\n")
    print(f"{'task':30s} | n   | ok %        | refusal %     | bad %       | gpt acc      | opus acc     | role")
    print("-" * 130)

    for task, items in sorted(by_task.items()):
        n = len(items)
        bucket = Counter()
        ok_qa_ids = []
        ok_gpt_correct = []  # 1/0
        rt_list, ct_list = [], []

        for p in items:
            f = classify_outcome(p["prediction"], task, p.get("finish_reason"))
            bucket[f] += 1
            if f == "ok":
                gt = gt_by_id.get(p["qa_id"])
                if gt is not None:
                    g = grade_pred(p["prediction"], gt)
                    if g is not None:
                        ok_qa_ids.append(p["qa_id"])
                        ok_gpt_correct.append(g)
            if p.get("reasoning_tokens"):
                rt_list.append(p["reasoning_tokens"])
            if p.get("completion_tokens"):
                ct_list.append(p["completion_tokens"])

        n_ok       = bucket["ok"]
        n_refusal  = bucket["refusal"]
        n_empty    = bucket["empty"]
        n_pfail    = bucket["parse_fail"]
        n_oor      = bucket["oor"]
        n_bad      = n_refusal + n_empty + n_pfail + n_oor

        ok_rate       = n_ok / n
        refusal_rate  = n_refusal / n
        bad_rate      = n_bad / n

        ok_lo, ok_hi          = wilson_ci(n_ok, n)
        refusal_lo, refusal_hi = wilson_ci(n_refusal, n)
        bad_lo, bad_hi        = wilson_ci(n_bad, n)

        # GPT accuracy on the 'ok' subset
        gpt_acc = None
        gpt_acc_ci = (None, None)
        if ok_gpt_correct:
            n_correct = sum(ok_gpt_correct)
            gpt_acc = n_correct / len(ok_gpt_correct)
            gpt_acc_ci = clopper_pearson_ci(n_correct, len(ok_gpt_correct))

        # Paired Opus comparison on the same ok_qa_ids subset
        paired = None
        if opus_by_id and ok_qa_ids:
            both_correct = a_only = b_only = both_wrong = 0
            opus_n = 0
            for qa_id, gpt_c in zip(ok_qa_ids, ok_gpt_correct):
                op = opus_by_id.get(qa_id)
                if op is None:
                    continue
                gt = gt_by_id.get(qa_id)
                if gt is None:
                    continue
                opus_c = grade_pred(op["prediction"], gt)
                if opus_c is None:
                    continue
                opus_n += 1
                if opus_c == 1 and gpt_c == 1: both_correct += 1
                elif opus_c == 1 and gpt_c == 0: a_only += 1
                elif opus_c == 0 and gpt_c == 1: b_only += 1
                else: both_wrong += 1
            if opus_n > 0:
                opus_acc = (both_correct + a_only) / opus_n
                opus_acc_ci = clopper_pearson_ci(both_correct + a_only, opus_n)
                # McNemar exact binomial when b+c < 25, else chi² with continuity correction
                bc = a_only + b_only
                p_val = method = None
                if bc == 0:
                    p_val = 1.0; method = "no_disagreement"
                elif bc < 25:
                    from math import comb
                    k = max(a_only, b_only)
                    p_val = min(2 * sum(comb(bc, i) for i in range(k, bc + 1)) / (2 ** bc), 1.0)
                    method = "exact_binomial"
                else:
                    chi2 = (abs(a_only - b_only) - 1) ** 2 / bc
                    from math import erf
                    z = math.sqrt(chi2)
                    p_val = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(2))))
                    method = "chi2_continuity_corrected"
                mp_or = (b_only / a_only) if a_only > 0 else (float('inf') if b_only > 0 else None)
                paired = {
                    "n_paired_ok": opus_n,
                    "opus_acc": round(opus_acc * 100, 2),
                    "opus_acc_95ci": [round(opus_acc_ci[0] * 100, 2), round(opus_acc_ci[1] * 100, 2)],
                    "confusion": {"both_correct": both_correct, "opus_only": a_only,
                                  "gpt_only": b_only, "both_wrong": both_wrong},
                    "mcnemar_p": round(p_val, 5),
                    "mcnemar_method": method,
                    "mp_OR": ("inf" if mp_or == float('inf') else
                              (round(mp_or, 3) if mp_or is not None else None)),
                }

        # Classify role + paper main-table cell value (per GPT review: dash convention)
        role = "MAIN" if bad_rate <= args.threshold else "DIAGNOSTIC"
        if role == "MAIN" and gpt_acc is not None:
            main_table_cell = f"{gpt_acc * 100:.1f}"
        else:
            # Em-dash for DIAGNOSTIC tasks; appendix carries the refusal characterization
            main_table_cell = "—"

        if role == "MAIN":
            results["summary"]["main_table_tasks"].append(task)
        else:
            results["summary"]["diagnostic_only_tasks"].append(task)

        results["per_task"][task] = {
            "n": n,
            "role": role,
            "main_table_cell": main_table_cell,  # paper Table 3: accuracy% or "—"
            "outcomes": {
                "ok": n_ok, "refusal": n_refusal, "empty": n_empty,
                "parse_fail": n_pfail, "oor": n_oor,
            },
            "rates_pct": {
                "ok": round(ok_rate * 100, 2),
                "refusal": round(refusal_rate * 100, 2),
                "bad_total": round(bad_rate * 100, 2),
            },
            "wilson_95ci_pct": {
                "ok":       [round(ok_lo * 100, 2), round(ok_hi * 100, 2)],
                "refusal":  [round(refusal_lo * 100, 2), round(refusal_hi * 100, 2)],
                "bad_total":[round(bad_lo * 100, 2), round(bad_hi * 100, 2)],
            },
            "gpt_acc_on_ok_subset": (None if gpt_acc is None else {
                "value_pct": round(gpt_acc * 100, 2),
                "n": len(ok_gpt_correct),
                "clopper_pearson_95ci_pct": [round(gpt_acc_ci[0] * 100, 2),
                                              round(gpt_acc_ci[1] * 100, 2)],
            }),
            "paired_opus_on_ok_subset": paired,
            "tokens": {
                "avg_reasoning": round(sum(rt_list) / len(rt_list), 1) if rt_list else None,
                "avg_completion": round(sum(ct_list) / len(ct_list), 1) if ct_list else None,
                "max_reasoning": max(rt_list) if rt_list else None,
            },
        }

        ok_str  = f"{ok_rate*100:5.1f} [{ok_lo*100:4.1f},{ok_hi*100:4.1f}]"
        ref_str = f"{refusal_rate*100:5.1f} [{refusal_lo*100:4.1f},{refusal_hi*100:4.1f}]"
        bad_str = f"{bad_rate*100:5.1f} [{bad_lo*100:4.1f},{bad_hi*100:4.1f}]"
        gpt_str = (f"{gpt_acc*100:5.1f} [{gpt_acc_ci[0]*100:4.1f},{gpt_acc_ci[1]*100:4.1f}]"
                   if gpt_acc is not None else "n/a")
        opus_str = (f"{paired['opus_acc']:5.1f} [{paired['opus_acc_95ci'][0]:4.1f},{paired['opus_acc_95ci'][1]:4.1f}]"
                    if paired else "n/a")
        print(f"{task:30s} | {n:3d} | {ok_str} | {ref_str} | {bad_str} | {gpt_str} | {opus_str} | {role}")

    print()
    main_tasks = results["summary"]["main_table_tasks"]
    diag_tasks = results["summary"]["diagnostic_only_tasks"]
    print(f"MAIN table tasks ({len(main_tasks)}):       {main_tasks}")
    print(f"DIAGNOSTIC-only tasks ({len(diag_tasks)}): {diag_tasks}")

    # Paper-ready text
    print("\n=== Paper-ready Pitfall 10 paragraph ===\n")
    high_refusal = [(t, results["per_task"][t]["rates_pct"]["refusal"],
                     results["per_task"][t]["wilson_95ci_pct"]["refusal"])
                    for t in diag_tasks]
    if high_refusal:
        parts = ", ".join(f"{t.split('_')[0]}: {r:.0f}% [CI {ci[0]:.0f},{ci[1]:.0f}]"
                          for t, r, ci in high_refusal)
        print(f"In our pre-specified GPT-5 paired evaluation (n=200/task, reasoning_effort=medium,")
        print(f"detail=high, identical prompts to Opus 4.7), GPT-5 refused precision-related tasks at")
        print(f"substantially higher rates ({parts}) than discrete-classification tasks. We therefore")
        print(f"report Opus 4.7 (and Gemini, when applicable) as the main frontier baselines and")
        print(f"document GPT-5 separately as a Pitfall 10 diagnostic of frontier-evaluation fragility.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
