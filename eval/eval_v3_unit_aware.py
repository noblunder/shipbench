#!/usr/bin/env python3
"""
Unit-aware grader for v3 numeric tasks.
Marks an answer correct iff:
  (1) the parsed numeric value is within tolerance of GT, AND
  (2) the unit string in the prediction matches the expected unit.

Expected units: m^3, m^2, mm, etc.  Aliases handled (m³ → m^3, m² → m^2, m3 → m^3).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path("<SHIPBENCH_ROOT>")
V2 = ROOT / "data" / "shipbench"

# Task → ground-truth file (3-tier IDs).
# B3 = cargo-capacity (7 variants), B4 = section-area (5 variants), C3 = bulkhead-position.
TASK_FILES = {
    "B3_cargo_capacity_v0a":  V2 / "task_B3_cargo_capacity_v0a.jsonl",
    "B3_cargo_capacity_v0b":  V2 / "task_B3_cargo_capacity_v0b.jsonl",
    "B3_cargo_capacity_v1":   V2 / "task_B3_cargo_capacity_v1.jsonl",
    "B3_cargo_capacity_v2a":  V2 / "task_B3_cargo_capacity_v2a.jsonl",
    "B3_cargo_capacity_v2b":  V2 / "task_B3_cargo_capacity_v2b.jsonl",
    "B3_cargo_capacity_v3":   V2 / "task_B3_cargo_capacity_v3.jsonl",
    "B3_cargo_capacity_v4":   V2 / "task_B3_cargo_capacity_v4.jsonl",
    "B4_section_area_v1":     V2 / "task_B4_section_area_v1.jsonl",
    "B4_section_area_v2a":    V2 / "task_B4_section_area_v2a.jsonl",
    "B4_section_area_v2b":    V2 / "task_B4_section_area_v2b.jsonl",
    "B4_section_area_v3":     V2 / "task_B4_section_area_v3.jsonl",
    "B4_section_area_v4":     V2 / "task_B4_section_area_v4.jsonl",
    "C3_bulkhead_position":   V2 / "task_C3_bulkhead_position.jsonl",
    # v3 prompt-evolution variants (HOLD N + CD clause / CoT trigger / coord-fix + AP)
    "B3_cargo_capacity_v3":   V2 / "task_B3_v3_clarified.jsonl",
    "B4_section_area_v3_cot": V2 / "task_B4_v3_cot.jsonl",
    "C3_bulkhead_position_v3": V2 / "task_C3_v3_clarified.jsonl",
    # v2 clarified prompts (precursors to v3)
    "B3_cargo_capacity_v2_clarified": V2 / "task_B3_v2_clarified.jsonl",
    "B4_section_area_v2_clarified":   V2 / "task_B4_v2_clarified.jsonl",
    "C3_bulkhead_position_v2":        V2 / "task_C3_v2_clarified.jsonl",
}

# Expected units (canonical form)
TASK_EXPECTED_UNIT = {
    "B3_cargo_capacity_v0a":  "m^3", "B3_cargo_capacity_v0b": "m^3",
    "B3_cargo_capacity_v1":   "m^3", "B3_cargo_capacity_v2a": "m^3",
    "B3_cargo_capacity_v2b":  "m^3", "B3_cargo_capacity_v3":  "m^3",
    "B3_cargo_capacity_v4":   "m^3",
    "B4_section_area_v1":     "m^2", "B4_section_area_v2a":  "m^2",
    "B4_section_area_v2b":    "m^2", "B4_section_area_v3":   "m^2",
    "B4_section_area_v4":     "m^2",
    "C3_bulkhead_position":   "mm",
    "B4_section_area_v3_cot": "m^2",
    "C3_bulkhead_position_v3": "mm",
    "B3_cargo_capacity_v2_clarified": "m^3",
    "B4_section_area_v2_clarified":   "m^2",
    "C3_bulkhead_position_v2":        "mm",
}

# Unit aliases mapped to canonical
UNIT_ALIASES = {
    "m^3": "m^3", "m³": "m^3", "m3": "m^3", "cubic meter": "m^3", "cubic meters": "m^3",
    "m^2": "m^2", "m²": "m^2", "m2": "m^2", "square meter": "m^2", "square meters": "m^2",
    "mm^2": "mm^2", "mm²": "mm^2", "mm2": "mm^2",
    "mm^3": "mm^3", "mm³": "mm^3", "mm3": "mm^3",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm",
    "m": "m", "meter": "m", "meters": "m",
}

# Unit conversion factors to canonical (for value normalisation)
UNIT_TO_CANON = {
    ("mm^3", "m^3"):  1e-9,
    ("mm^2", "m^2"):  1e-6,
    ("cm^3", "m^3"):  1e-6,
    ("cm^2", "m^2"):  1e-4,
}


_NUM_UNIT_RE = re.compile(
    r"(-?\d+(?:[\.,]\d+)?(?:\s*[eE][+-]?\d+)?)\s*"
    r"(m\^[23]|m[²³]|m[23]|mm\^[23]|mm[²³]|mm[23]|mm|m|cm\^[23]|cm[²³]|cm[23])?"
)
_FINAL_MARKER_RE = re.compile(
    r"(final\s*answer|the\s*answer\s*is|answer\s*[:=]|\\boxed\{)",
    re.IGNORECASE,
)


def parse_value_unit(s: str):
    """Return (value, canonical_unit) or (None, None).

    Strategy for CoT outputs: prefer the value+unit pair that follows the LAST
    "Final answer:" / "\\boxed{" marker. Fall back to LAST match in the whole
    text (better than first for CoT). Fall back to first only if no number found.
    """
    s_clean = s.strip()
    # Strip prefixes
    s_clean = re.sub(r"^(answer|the answer is|approximately|about)[\s:=]*", "",
                     s_clean, flags=re.IGNORECASE)

    # 1. Final-answer marker: parse the substring after the LAST marker
    markers = list(_FINAL_MARKER_RE.finditer(s_clean))
    if markers:
        tail = s_clean[markers[-1].end():]
        m = _NUM_UNIT_RE.search(tail)
        if m:
            return _to_pair(m)

    # 2. LAST number+unit pair anywhere in text
    last = None
    for m in _NUM_UNIT_RE.finditer(s_clean):
        # Only accept matches that have a unit, OR keep the last numeric-only one as fallback
        if m.group(2):
            last = m
    if last is not None:
        return _to_pair(last)

    # 3. Fallback: very first numeric (no unit)
    m = _NUM_UNIT_RE.search(s_clean)
    return _to_pair(m) if m else (None, None)


def _to_pair(m):
    val_str, unit_raw = m.group(1), (m.group(2) or "").strip().lower()
    val_str = val_str.replace(",", "")
    try: val = float(val_str)
    except: return None, None
    unit = UNIT_ALIASES.get(unit_raw, unit_raw if unit_raw else None)
    return val, unit


def grade_numeric_with_unit(pred_text, gt_value, gt_unit, tol_pct,
                             strict_unit=True):
    val, unit = parse_value_unit(pred_text)
    out = {"parsed_value": val, "parsed_unit": unit, "gt": gt_value, "gt_unit": gt_unit,
           "rel_err": None, "format_ok": 0, "unit_ok": 0, "value_ok": 0, "correct": 0}
    if val is None:
        return out
    out["format_ok"] = 1
    # Convert if non-canonical unit recognized
    converted = val
    if unit and unit != gt_unit:
        factor = UNIT_TO_CANON.get((unit, gt_unit))
        if factor is not None:
            converted = val * factor
    out["unit_ok"] = int(unit == gt_unit) if unit else 0
    rel = abs(converted - gt_value) / max(abs(gt_value), 1e-9)
    out["rel_err"] = rel
    out["value_ok"] = int(rel <= tol_pct / 100.0)
    if strict_unit:
        out["correct"] = int(out["unit_ok"] == 1 and out["value_ok"] == 1)
    else:
        # Lenient: missing-unit accepted if magnitude is right
        out["correct"] = int(out["value_ok"] == 1)
    return out


def load_gt():
    out = {}
    for task, path in TASK_FILES.items():
        if not path.exists(): continue
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            d = json.loads(line)
            out[d["qa_id"]] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--lenient", action="store_true",
                    help="Accept missing unit if magnitude is right")
    args = ap.parse_args()

    gt = load_gt()
    preds = [json.loads(l) for l in Path(args.pred).read_text().splitlines() if l.strip()]
    print(f"GT items: {len(gt)}    Predictions: {len(preds)}")

    by_task = defaultdict(list)
    for p in preds:
        g = gt.get(p["qa_id"])
        if g is None: continue
        task = g["task"]
        gt_val = float(g["metadata"]["value"])
        gt_unit = g["metadata"].get("unit") or TASK_EXPECTED_UNIT.get(task)
        tol = g["metadata"].get("tolerance_pct", 10.0)
        ev = grade_numeric_with_unit(p.get("prediction", ""), gt_val, gt_unit, tol,
                                       strict_unit=not args.lenient)
        ev["ship_type"] = g["ship_type"]
        ev["qa_id"] = p["qa_id"]
        by_task[task].append(ev)

    out = {"model": Path(args.pred).stem, "tasks": {}}
    for task, evs in by_task.items():
        n = len(evs)
        n_correct = sum(e["correct"] for e in evs)
        n_unit_ok = sum(e["unit_ok"] for e in evs)
        n_val_ok = sum(e["value_ok"] for e in evs)
        n_fmt = sum(e["format_ok"] for e in evs)
        rels = [e["rel_err"] for e in evs if e["rel_err"] is not None]
        rec = {
            "n": n,
            "accuracy": round(n_correct / n * 100, 2),
            "unit_compliance": round(n_unit_ok / n * 100, 2),
            "value_within_tol": round(n_val_ok / n * 100, 2),
            "format_compliance": round(n_fmt / n * 100, 2),
            "median_rel_err_pct": round(statistics.median(rels) * 100, 2) if rels else None,
        }
        # Per-ship
        per_ship = defaultdict(lambda: [0, 0])
        for e in evs:
            per_ship[e["ship_type"]][0] += e["correct"]
            per_ship[e["ship_type"]][1] += 1
        rec["per_ship"] = {s: round(c/n*100, 1) if n else None
                           for s, (c, n) in per_ship.items()}
        out["tasks"][task] = rec

    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
