#!/usr/bin/env python3
"""
ShipBench main eval — 9 sub-tasks × 594 items = 5,346 headline.

Reads ground truth directly from task_main_eval.jsonl (each item already has
`answer` + `answer_type` + `metadata`). Dispatches scoring by answer_type:
    mcq_letter / letter   → first-letter exact match (case-insensitive)
    numeric                → |pred - gt| / |gt| <= tol
    numeric_with_unit      → same + canonical unit match

Per-task tolerances follow paper §4 Table 1:
    B1, B2 ±5%;  B3, B4, C3 ±10%;  MCQ tasks raw accuracy.

Outputs JSON + console summary including per-task accuracy, 95% bootstrap CI
(1,000 resamples, seed 42), per-ship-type breakdown, and median relative error
(MRE) for numeric tasks. Format-compliance rate (parseable predictions) reported
per task.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("<SHIPBENCH_ROOT>")
DEFAULT_GT = ROOT / "data" / "shipbench" / "task_main_eval.jsonl"

TASK_TOL = {
    "B1_plate_thickness": 0.05,
    "B2_stiffener_size": 0.05,
    "B3_cargo_capacity_v1": 0.10,
    "B4_section_area_v1": 0.10,
    "C3_bulkhead_position": 0.10,
}

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
UNIT_RE = re.compile(r"(m\s*\^?\s*[23]|mm|cm|m)\b", re.IGNORECASE)
LETTER_BOUNDARY_RE = re.compile(r"(?:^|[^A-Za-z])([A-Z])(?:[^A-Za-z]|$)")


def parse_numeric(text: str) -> float | None:
    if not isinstance(text, str):
        return None
    m = NUM_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def normalize_unit(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("³", "^3").replace("²", "^2")
    m = UNIT_RE.search(t)
    if not m:
        return ""
    raw = m.group(1).lower().replace(" ", "").replace("^", "")
    return {"m3": "m^3", "m2": "m^2"}.get(raw, raw)


def parse_letter(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    m = LETTER_BOUNDARY_RE.search(" " + s + " ")
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*([A-Za-z])", s)
    return m.group(1).upper() if m else ""


def grade(pred: str, gt: dict) -> tuple[bool, bool]:
    """Return (is_correct, is_format_compliant)."""
    atype = gt.get("answer_type", "")
    answer = gt["answer"]
    # Per-item tolerance override (e.g., LPGC ±15% for B3 v2 due to Type A
    # tank length-direction mismatch). Falls back to TASK_TOL.
    item_tol = gt.get("metadata", {}).get("tolerance_pct")
    if item_tol is not None:
        tol = float(item_tol) / 100.0
    else:
        tol = TASK_TOL.get(gt["task"], 0.05)

    if atype in ("mcq_letter", "letter"):
        pl = parse_letter(pred)
        gl = parse_letter(str(answer))
        return (pl == gl and pl != "", bool(pl))

    if atype == "numeric":
        pv = parse_numeric(pred)
        gv = parse_numeric(str(answer))
        if gv is None:
            return (False, pv is not None)
        if pv is None:
            return (False, False)
        if abs(gv) < 1e-9:
            return (abs(pv) < 1e-9, True)
        return (abs(pv - gv) / abs(gv) <= tol, True)

    if atype == "numeric_with_unit":
        pv = parse_numeric(pred)
        gv = parse_numeric(str(answer))
        pu = normalize_unit(pred)
        gu = normalize_unit(str(answer))
        if gv is None:
            return (False, pv is not None)
        if pv is None:
            return (False, False)
        unit_ok = (not gu) or (pu == gu)
        if not unit_ok:
            return (False, True)
        if abs(gv) < 1e-9:
            return (abs(pv) < 1e-9, True)
        return (abs(pv - gv) / abs(gv) <= tol, True)

    return (False, False)


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42, conf: float = 0.95) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randint(0, n - 1)] for _ in range(n)) / n)
    means.sort()
    lo_i = max(0, int(n_boot * (1 - conf) / 2))
    hi_i = min(n_boot - 1, int(n_boot * (1 + conf) / 2))
    return (means[lo_i], means[hi_i])


def evaluate(pred_path: Path, gt_path: Path) -> dict:
    gt: dict[str, dict] = {}
    with open(gt_path) as f:
        for line in f:
            o = json.loads(line)
            gt[o["qa_id"]] = o

    preds: dict[str, str] = {}
    with open(pred_path) as f:
        for line in f:
            o = json.loads(line)
            preds[o["qa_id"]] = o.get("prediction", "")

    by_task_correct: dict[str, list[int]] = defaultdict(list)
    by_task_format: dict[str, list[int]] = defaultdict(list)
    by_task_relerr: dict[str, list[float]] = defaultdict(list)
    by_task_ship_correct: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    matched = 0
    for qa_id, gt_item in gt.items():
        if qa_id not in preds:
            continue
        matched += 1
        pred = preds[qa_id]
        task = gt_item["task"]
        ship = gt_item.get("ship_type", "?")
        is_corr, is_fmt = grade(pred, gt_item)
        by_task_correct[task].append(int(is_corr))
        by_task_format[task].append(int(is_fmt))
        by_task_ship_correct[task][ship].append(int(is_corr))

        if gt_item.get("answer_type", "").startswith("numeric"):
            pv = parse_numeric(pred)
            gv = parse_numeric(str(gt_item["answer"]))
            if pv is not None and gv is not None and abs(gv) > 1e-9:
                by_task_relerr[task].append(abs(pv - gv) / abs(gv))

    summary = {
        "predictions_path": str(pred_path),
        "ground_truth_path": str(gt_path),
        "n_matched": matched,
        "n_gt": len(gt),
        "per_task": {},
    }
    for task in sorted(by_task_correct):
        items = by_task_correct[task]
        fmts = by_task_format[task]
        n = len(items)
        acc = sum(items) / n
        ci_lo, ci_hi = bootstrap_ci([float(x) for x in items])
        per_ship = {}
        for ship, ship_items in sorted(by_task_ship_correct[task].items()):
            sn = len(ship_items)
            per_ship[ship] = {
                "n": sn,
                "acc": round(100 * (sum(ship_items) / sn if sn else 0), 1),
            }
        entry = {
            "n": n,
            "acc": round(100 * acc, 2),
            "ci_low": round(100 * ci_lo, 2),
            "ci_high": round(100 * ci_hi, 2),
            "format_compliance": round(100 * sum(fmts) / n, 1),
            "per_ship": per_ship,
        }
        if by_task_relerr[task]:
            entry["mre_pct"] = round(100 * statistics.median(by_task_relerr[task]), 1)
        summary["per_task"][task] = entry
    return summary


def main():
    ap = argparse.ArgumentParser(description="ShipBench main 9-task evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--ground-truth", default=str(DEFAULT_GT))
    args = ap.parse_args()

    summary = evaluate(Path(args.predictions), Path(args.ground_truth))

    print(f"\n=== {Path(args.predictions).name} ===")
    print(f"Matched: {summary['n_matched']}/{summary['n_gt']}")
    header = f"{'task':<28} {'n':>5} {'acc%':>8} {'95% CI':>20} {'fmt%':>6} {'MRE%':>7}"
    print(header)
    print("-" * len(header))
    for task, s in summary["per_task"].items():
        ci = f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]"
        mre = f"{s['mre_pct']}" if "mre_pct" in s else "—"
        print(f"{task:<28} {s['n']:>5} {s['acc']:>8.2f} {ci:>20} {s['format_compliance']:>6.1f} {mre:>7}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
