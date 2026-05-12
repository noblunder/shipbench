#!/usr/bin/env python3
"""Paired multi-frontier analysis on Tier B results.

Pre-registered analyses (per GPT/Claude review feedback):
1. Paired accuracy difference (GPT-5 - Opus) per task
2. Paired bootstrap 95% CI (1000 resamples, seed=42)
3. McNemar χ² test (binary correctness)
4. Cohen's h effect size for proportions
5. Confusion: both-correct / Opus-only / GPT-only / both-wrong counts
6. Per-task disagreement rate
7. Per-ship-type breakdown for C3 / B4

Run AFTER Tier B inference completes:
  python scripts/paired_frontier_analysis.py \
    --opus outputs/frontier_eval/claude_opus_main.jsonl \
    --gpt5 outputs/frontier_eval/gpt-5_main.jsonl \
    --output outputs/main_eval/paired_frontier_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# Re-use the smart numeric extractor from eval_v3_unit_aware.py
NUM_UNIT_RE = re.compile(
    r"(-?\d+(?:[\.,]\d+)?(?:\s*[eE][+-]?\d+)?)\s*"
    r"(m\^[23]|m[²³]|m[23]|mm\^[23]|mm[²³]|mm[23]|mm|m|cm\^[23]|cm[²³]|cm[23])?"
)
FINAL_MARKER_RE = re.compile(
    r"(final\s*answer|the\s*answer\s*is|answer\s*[:=]|\\boxed\{)", re.I
)
UNIT_ALIASES = {
    "m^3": "m^3", "m³": "m^3", "m3": "m^3",
    "m^2": "m^2", "m²": "m^2", "m2": "m^2",
    "mm": "mm", "m": "m", "cm^2": "cm^2", "cm^3": "cm^3",
}


def extract_value_unit(s: str):
    s = s.strip()
    markers = list(FINAL_MARKER_RE.finditer(s))
    if markers:
        tail = s[markers[-1].end():]
        m = NUM_UNIT_RE.search(tail)
        if m:
            return _to_pair(m)
    last = None
    for m in NUM_UNIT_RE.finditer(s):
        if m.group(2):
            last = m
    if last:
        return _to_pair(last)
    m = NUM_UNIT_RE.search(s)
    return _to_pair(m) if m else (None, None)


def _to_pair(m):
    val_s, unit_raw = m.group(1), (m.group(2) or "").strip().lower()
    val_s = val_s.replace(",", "")
    try:
        val = float(val_s)
    except ValueError:
        return None, None
    unit = UNIT_ALIASES.get(unit_raw, unit_raw if unit_raw else None)
    return val, unit


# Task-aware grading
NUMERIC_TASKS = {
    "B1_plate_thickness": ("mm", 5.0),
    "B2_stiffener_size": ("mm", 5.0),
    "B3_cargo_capacity_v1": ("m^3", 10.0),
    "B3_cargo_capacity_v3": ("m^3", 10.0),
    "B4_section_area_v1": ("m^2", 10.0),
    "B4_section_area_v3_cot": ("m^2", 10.0),
    "C3_bulkhead_position": ("mm", 10.0),
    "C3_bulkhead_position_v3": ("mm", 10.0),
}
MCQ_TASKS = {"A1_shiptype", "A1_shiptype_section_only", "A2_stiffener_type",
             "C1_compartment_locate", "C2_compartment_boundary"}


def grade_pred(pred: str, gt_item: dict, strict_unit: bool = True) -> int:
    """Return 1 if correct, 0 otherwise."""
    task = gt_item.get("task")
    if task in MCQ_TASKS:
        # MCQ: extract first letter A-F
        m = re.search(r"\b([A-F])\b", pred.strip().upper())
        return int(m is not None and m.group(1) == gt_item["answer"].strip().upper())

    if task in NUMERIC_TASKS:
        gt_unit, default_tol = NUMERIC_TASKS[task]
        gt_val = float(gt_item["metadata"]["value"])
        gt_unit_meta = gt_item["metadata"].get("unit", gt_unit)
        tol = gt_item["metadata"].get("tolerance_pct", default_tol)
        val, unit = extract_value_unit(pred)
        if val is None:
            return 0
        # Lenient unit policy: unit-less numeric predictions are accepted as the
        # task's canonical unit (e.g., B1/B2 prompts ask for mm, model emits "12.0").
        # Strict unit only fails on EXPLICITLY MISMATCHED units (e.g., "12 cm" when mm expected).
        if strict_unit and unit is not None and unit != gt_unit_meta:
            return 0
        rel_err = abs(val - gt_val) / max(abs(gt_val), 1e-9)
        return int(rel_err <= tol / 100.0)

    return 0


def mcnemar_test(b: int, c: int) -> tuple[float, float, str]:
    """McNemar test — uses exact binomial when b+c < 25 (per GPT review).

    b = GPT correct, Opus wrong
    c = Opus correct, GPT wrong
    Returns (statistic, p_value, method)

    Per Edwards (1948) recommendation and GPT review feedback:
    chi-square approximation is unreliable when b+c < 25.
    Exact binomial test = Pr(X >= max(b,c)) under H0: p=0.5, n=b+c.
    """
    if b + c == 0:
        return 0.0, 1.0, "no_disagreement"

    if b + c < 25:
        # Exact binomial test (more accurate for small disagreement counts)
        # Two-sided p = 2 * P(X >= max(b,c) | n=b+c, p=0.5)
        from math import comb
        n = b + c
        k = max(b, c)
        # p-value = 2 * sum from k to n of binomial(n, i, 0.5)
        tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
        p = min(2 * tail, 1.0)
        statistic = float(max(b, c))
        method = "exact_binomial"
    else:
        # Chi-square with continuity correction (Edwards correction)
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        from math import erf
        z = math.sqrt(chi2)
        p = 2 * (1 - 0.5 * (1 + erf(z / math.sqrt(2))))
        statistic = chi2
        method = "chi2_continuity_corrected"

    return statistic, p, method


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions (NOT paired-aware)."""
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def matched_pair_odds_ratio(b: int, c: int) -> float | None:
    """Matched-pair odds ratio — directly interpretable for paired binary outcomes.

    OR = b / c
    Per GPT review feedback: more directly interpretable than Cohen's h for paired data.
    Example: b=80, c=20 → OR=4.0 = "GPT-5 4x more likely than Opus to be uniquely correct"
    """
    if c == 0:
        return float('inf') if b > 0 else None
    return b / c


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact 95% CI for binomial proportion.

    Critical for rare-event reporting (e.g., B4 0/200 needs upper bound).
    Per GPT review: report this for sparse-success cells in main table.
    """
    if n == 0:
        return 0.0, 1.0
    from math import isnan
    try:
        from scipy.stats import beta
        if k == 0:
            lo = 0.0
        else:
            lo = beta.ppf(alpha / 2, k, n - k + 1)
        if k == n:
            hi = 1.0
        else:
            hi = beta.ppf(1 - alpha / 2, k + 1, n - k)
    except ImportError:
        # Fallback: Wilson score interval (less exact but no scipy dep)
        from math import sqrt
        z = 1.96
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2*n)) / denom
        margin = z * sqrt(phat * (1-phat) / n + z**2 / (4*n**2)) / denom
        lo = max(0, center - margin)
        hi = min(1, center + margin)
    return lo, hi


def holm_correction(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction for multiple testing.

    Returns dict {task: corrected_p}. Use when reporting all 9 tasks.
    Primary tests (B3/B4/C3) should be reported uncorrected if pre-specified.
    """
    sorted_pvals = sorted(p_values.items(), key=lambda x: x[1])
    n = len(sorted_pvals)
    corrected = {}
    for rank, (task, p) in enumerate(sorted_pvals):
        # Holm: multiply by (n - rank), cap at 1.0
        corrected[task] = min(p * (n - rank), 1.0)
    return corrected


def paired_bootstrap_ci(pairs: list[tuple[int, int]], n_resamples: int = 1000, seed: int = 42):
    """Compute paired-bootstrap 95% CI for delta = mean(b_acc) - mean(a_acc)."""
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_resamples):
        idx = rng.choice(len(pairs), len(pairs), replace=True)
        sample = [pairs[i] for i in idx]
        d = (np.mean([s[1] for s in sample]) - np.mean([s[0] for s in sample])) * 100
        deltas.append(d)
    point = (np.mean([p[1] for p in pairs]) - np.mean([p[0] for p in pairs])) * 100
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return point, ci_low, ci_high


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opus", required=True, help="Opus predictions JSONL")
    ap.add_argument("--gpt5", required=True, help="GPT-5 predictions JSONL")
    ap.add_argument("--output", required=True, help="Output JSON")
    ap.add_argument("--name-a", default="Opus")
    ap.add_argument("--name-b", default="GPT-5")
    args = ap.parse_args()

    # Load predictions
    opus_preds = [json.loads(l) for l in open(args.opus)]
    gpt_preds = [json.loads(l) for l in open(args.gpt5)]

    opus_by_id = {p["qa_id"]: p for p in opus_preds}
    gpt_by_id = {p["qa_id"]: p for p in gpt_preds}

    # Load all GT files (combine main + v3)
    gt_files = [
        "<SHIPBENCH_ROOT>/data/shipbench/task_main_eval.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench/task_main_eval_opus_paired.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench/task_A1_shiptype_section_only.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench/task_A1_v2_opus_paired.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench/task_B3_v3_clarified.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench/task_B4_v3_cot.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench/task_C3_v3_clarified.jsonl",
    ]
    gt_by_id = {}
    for f in gt_files:
        if Path(f).exists():
            for line in open(f):
                d = json.loads(line)
                gt_by_id[d["qa_id"]] = d

    # Find paired qa_ids
    paired_ids = sorted(set(opus_by_id) & set(gpt_by_id))
    print(f"Paired qa_ids: {len(paired_ids)}")

    # Group by task
    by_task = defaultdict(list)
    for qa_id in paired_ids:
        gt = gt_by_id.get(qa_id)
        if gt is None:
            continue
        by_task[gt["task"]].append(qa_id)

    # Run analyses per task
    results = {"meta": {
        "name_a": args.name_a, "name_b": args.name_b,
        "n_paired": len(paired_ids), "tasks": list(by_task.keys()),
        "pre_specified_analyses": [
            "paired_acc_a", "paired_acc_b",
            "clopper_pearson_95ci_a", "clopper_pearson_95ci_b",
            "paired_delta_pp", "paired_bootstrap_ci_95",
            "mcnemar_p (exact_binomial if b+c<25 else chi2_continuity_corrected)",
            "matched_pair_OR (paired-aware effect size)",
            "cohens_h (secondary, NOT paired-aware)",
            "confusion", "disagreement_pattern", "ship_type_breakdown",
            "holm_corrected_p (for exploratory tasks)",
        ],
    }, "per_task": {}}

    for task, qa_ids in sorted(by_task.items()):
        pairs = []  # (a_correct, b_correct)
        ship_breakdown = defaultdict(lambda: [0, 0, 0])  # [a_correct, b_correct, n]
        for qa_id in qa_ids:
            gt = gt_by_id[qa_id]
            a_pred = opus_by_id[qa_id]["prediction"]
            b_pred = gpt_by_id[qa_id]["prediction"]
            a_correct = grade_pred(a_pred, gt)
            b_correct = grade_pred(b_pred, gt)
            pairs.append((a_correct, b_correct))
            ship = gt.get("ship_type", "?")
            ship_breakdown[ship][0] += a_correct
            ship_breakdown[ship][1] += b_correct
            ship_breakdown[ship][2] += 1

        n = len(pairs)
        a_correct_n = sum(p[0] for p in pairs)
        b_correct_n = sum(p[1] for p in pairs)
        a_acc = a_correct_n / n * 100
        b_acc = b_correct_n / n * 100
        delta = b_acc - a_acc

        # Clopper-Pearson 95% CI on per-model accuracies (rare-event aware)
        cp_a_lo, cp_a_hi = clopper_pearson_ci(a_correct_n, n)
        cp_b_lo, cp_b_hi = clopper_pearson_ci(b_correct_n, n)

        # Confusion: (a, b) ∈ {(0,0), (0,1), (1,0), (1,1)}
        both_wrong = sum(1 for p in pairs if p == (0, 0))
        a_only = sum(1 for p in pairs if p == (1, 0))
        b_only = sum(1 for p in pairs if p == (0, 1))
        both_correct = sum(1 for p in pairs if p == (1, 1))

        # McNemar (auto-selects exact binomial vs chi-square based on b+c)
        statistic, p_val, mcnemar_method = mcnemar_test(b=b_only, c=a_only)
        # Cohen's h (NOT paired-aware; for interpretability only — appendix)
        h = cohens_h(b_acc / 100, a_acc / 100)
        # Matched-pair odds ratio (paired-aware, directly interpretable)
        mp_or = matched_pair_odds_ratio(b=b_only, c=a_only)
        # Bootstrap CI
        point, ci_low, ci_high = paired_bootstrap_ci(pairs)

        # Disagreement structure interpretation (per GPT review)
        bc_total = b_only + a_only
        if bc_total == 0:
            disagree_pattern = "no_disagreement"
        elif bc_total < 0.05 * n:
            disagree_pattern = "low_disagreement_models_aligned"
        elif b_only >= 2 * a_only and b_only > 5:
            disagree_pattern = f"{args.name_b}_dominates"
        elif a_only >= 2 * b_only and a_only > 5:
            disagree_pattern = f"{args.name_a}_dominates"
        else:
            disagree_pattern = "task_tradeoff_models_make_different_errors"

        results["per_task"][task] = {
            "n": n,
            f"acc_{args.name_a}": round(a_acc, 2),
            f"acc_{args.name_b}": round(b_acc, 2),
            f"clopper_pearson_95ci_{args.name_a}": [round(cp_a_lo * 100, 2), round(cp_a_hi * 100, 2)],
            f"clopper_pearson_95ci_{args.name_b}": [round(cp_b_lo * 100, 2), round(cp_b_hi * 100, 2)],
            "delta_pp": round(delta, 2),
            "bootstrap_95ci": [round(ci_low, 2), round(ci_high, 2)],
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "mcnemar_method": mcnemar_method,  # exact_binomial vs chi2_continuity_corrected
            "mcnemar_statistic": round(statistic, 3),
            "mcnemar_p": round(p_val, 4),
            "mcnemar_significant": p_val < 0.05,
            "matched_pair_OR": round(mp_or, 3) if mp_or is not None and mp_or != float('inf') else (None if mp_or is None else "inf"),
            "cohens_h": round(h, 3),
            "confusion": {
                "both_correct": both_correct,
                f"{args.name_a}_only_correct": a_only,
                f"{args.name_b}_only_correct": b_only,
                "both_wrong": both_wrong,
            },
            "disagreement_rate_pct": round((a_only + b_only) / n * 100, 2),
            "disagreement_pattern": disagree_pattern,
            "ship_type_breakdown": {
                ship: {
                    f"acc_{args.name_a}": round(d[0]/d[2]*100, 1) if d[2] else None,
                    f"acc_{args.name_b}": round(d[1]/d[2]*100, 1) if d[2] else None,
                    "n": d[2],
                } for ship, d in ship_breakdown.items()
            },
        }

    # Multiple testing — Holm correction for ALL 9 tasks (for appendix)
    # Primary tests (B3/B4/C3) should be reported uncorrected if pre-specified.
    p_values = {t: r["mcnemar_p"] for t, r in results["per_task"].items()}
    corrected = holm_correction(p_values)
    for t, p_corr in corrected.items():
        results["per_task"][t]["mcnemar_p_holm"] = round(p_corr, 4)
        results["per_task"][t]["mcnemar_significant_holm"] = p_corr < 0.05

    # Mark primary vs exploratory
    PRIMARY_TASKS = {"B3_cargo_capacity_v1", "B3_cargo_capacity_v3",
                     "B4_section_area_v1", "B4_section_area_v3_cot",
                     "C3_bulkhead_position", "C3_bulkhead_position_v3"}
    for t in results["per_task"]:
        results["per_task"][t]["test_role"] = "primary" if t in PRIMARY_TASKS else "exploratory"

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved → {args.output}")

    # Print summary
    print("\n=== Paired analysis summary ===")
    header = f'{"task":33s} | n   | role | {args.name_a:>5s} (CP-CI)        | {args.name_b:>5s} (CP-CI)        | Δpp   | bootstrap CI | b/c (mp_OR)    | McNemar p (method)        | pattern'
    print(header)
    print("-" * len(header))
    for task, r in sorted(results["per_task"].items()):
        a_acc = r[f"acc_{args.name_a}"]
        b_acc = r[f"acc_{args.name_b}"]
        cp_a = r[f"clopper_pearson_95ci_{args.name_a}"]
        cp_b = r[f"clopper_pearson_95ci_{args.name_b}"]
        delta = r["delta_pp"]
        ci_low, ci_high = r["bootstrap_95ci"]
        p = r["mcnemar_p"]
        method = r["mcnemar_method"]
        b_count = r["confusion"][f"{args.name_b}_only_correct"]
        c_count = r["confusion"][f"{args.name_a}_only_correct"]
        mp_or = r["matched_pair_OR"]
        sig = "*" if r["mcnemar_significant"] else " "
        role = r["test_role"][:4]
        bc_str = f"{b_count}/{c_count}"
        or_str = f" OR={mp_or}" if mp_or is not None else ""
        method_short = "exact" if "exact" in method else "chi2"
        a_str = f"{a_acc:5.1f} [{cp_a[0]:4.1f},{cp_a[1]:5.1f}]"
        b_str = f"{b_acc:5.1f} [{cp_b[0]:4.1f},{cp_b[1]:5.1f}]"
        print(f'{task:33s} | {r["n"]:3d} | {role:4s} | {a_str} | {b_str} | {delta:+5.1f} | [{ci_low:+4.1f},{ci_high:+5.1f}] | {bc_str:>5s}{or_str:>10s} | p={p:7.4f}{sig} ({method_short}) | {r["disagreement_pattern"]}')


if __name__ == "__main__":
    main()
