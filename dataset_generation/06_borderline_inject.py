#!/usr/bin/env python3
"""
Phase 0.2.F — Borderline Case Injection.

For each ship type, takes ~50-100 existing candidates and perturbs their key
rule-sensitive parameter to sit exactly at threshold ± U(-0.1, +0.1) m.
These borderline samples are the seeds for counterfactual QA in Paper 1.

Key parameters per ship (from PARAM_RANGES.md §7.1):
  Tanker: DS (req=2.0), DB (req=2.0)
  VLCC:   DS, DB
  BULKC:  DB (req=B/20), L (req=150)
  CNTR:   DS (req=0.04·B)
  LNGC:   DS (req=B/15), DB (req=B/15)
  LPGC:   DB (req=B/15)

Usage:
    python scripts/06_borderline_inject.py
    python scripts/06_borderline_inject.py --count 50 --ships LPGC
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "data_generator"
sys.path.insert(0, str(GEN_DIR))

CANDIDATES_DIR = ROOT / "data" / "candidates_R1"
STRATIFIED_DIR = CANDIDATES_DIR / "stratified"
SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]

# Import the eval functions from the candidate generator
sys.path.insert(0, str(ROOT / "scripts"))


def _compute_threshold(ship_type: str, gi: dict, param_name: str) -> float | None:
    """Compute the rule threshold value for a given parameter on this candidate."""
    B = gi.get("B_m", 0)
    L = gi.get("L_m", 0)

    if ship_type in ("Tanker", "VLCC"):
        if param_name in ("DS", "DB"):
            return 2.0  # DWT-based DS/DB threshold caps at 2.0 for large tankers
    elif ship_type == "BULKC":
        if param_name == "DB":
            return max(0.76, B / 20.0)
        if param_name == "L":
            return 150.0
    elif ship_type == "CNTR":
        if param_name == "DS":
            return 0.04 * B  # DS ≥ 0.04·B for hatch_ratio check
    elif ship_type == "LNGC":
        if param_name in ("DS", "DB"):
            return max(0.76, B / 15.0)
    elif ship_type == "LPGC":
        if param_name == "DB":
            return max(0.76, B / 15.0)
    return None


# Map of ship_type → list of (param_name, json_key) for injection
INJECTION_PARAMS = {
    "Tanker": [("DS", "doubleSide_m"), ("DB", "doubleBottom_m")],
    "VLCC":   [("DS", "doubleSide_m"), ("DB", "doubleBottom_m")],
    "BULKC":  [("DB", "doubleBottom_m")],
    "CNTR":   [("DS", "doubleSide_m")],
    "LNGC":   [("DS", "doubleSide_m"), ("DB", "doubleBottom_m")],
    "LPGC":   [("DB", "doubleBottom_m")],
}


def load_fps_selected(ship: str) -> list[dict]:
    """Load FPS-selected candidates (or fall back to stratified)."""
    manifest = STRATIFIED_DIR / "fps_selected.jsonl"
    if manifest.exists():
        selected_ids = set()
        with open(manifest) as f:
            for line in f:
                d = json.loads(line)
                if d["ship_type"] == ship:
                    selected_ids.add(d["candidate_id"])
        # Load the actual JSONs
        candidates = []
        for stratum_dir in sorted((STRATIFIED_DIR / ship).iterdir()):
            if not stratum_dir.is_dir() or stratum_dir.name == "section_png":
                continue
            for jf in sorted(stratum_dir.glob(f"{ship}-*.json")):
                if jf.stem in selected_ids:
                    candidates.append(json.load(open(jf)))
        return candidates
    else:
        # Fall back to all stratified
        candidates = []
        for stratum_dir in sorted((STRATIFIED_DIR / ship).iterdir()):
            if not stratum_dir.is_dir() or stratum_dir.name == "section_png":
                continue
            for jf in sorted(stratum_dir.glob(f"{ship}-*.json")):
                candidates.append(json.load(open(jf)))
        return candidates


def inject_borderline(
    ship_type: str,
    candidates: list[dict],
    count: int,
    rng: random.Random,
) -> list[dict]:
    """Generate borderline candidates by snapping key parameters to threshold ± U(-0.1, +0.1)."""
    params = INJECTION_PARAMS[ship_type]
    injected = []
    # Pick source candidates to perturb
    sources = rng.sample(candidates, min(count * 2, len(candidates)))

    for src in sources:
        if len(injected) >= count:
            break
        gi = copy.deepcopy(src["generator_inputs"])
        param_name, json_key = rng.choice(params)
        threshold = _compute_threshold(ship_type, gi, param_name)
        if threshold is None:
            continue
        # Snap to threshold ± uniform(-0.1, +0.1)
        perturbation = rng.uniform(-0.1, 0.1)
        new_val = round(threshold + perturbation, 2)
        # Clamp to positive
        if new_val < 0.1:
            new_val = 0.1
        gi[json_key] = new_val

        borderline = {
            "candidate_id": f"{ship_type}-BL-{len(injected):04d}",
            "ship_type": ship_type,
            "regime": "borderline",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_candidate": src["candidate_id"],
            "injection": {
                "param": param_name,
                "json_key": json_key,
                "original_value": src["generator_inputs"][json_key],
                "threshold": round(threshold, 4),
                "perturbed_value": new_val,
                "perturbation": round(perturbation, 4),
            },
            "generator_inputs": gi,
        }

        # Re-evaluate rules with perturbed params
        try:
            from scripts_00_generate_candidates_eval import re_evaluate
            kr_eval = re_evaluate(ship_type, gi)
            borderline["kr_eval"] = kr_eval
            borderline["kr_summary"] = _summarize(kr_eval)
        except Exception:
            # If re-eval not available, mark as needs-eval
            borderline["kr_eval"] = None
            borderline["kr_summary"] = {"needs_eval": True}

        injected.append(borderline)

    return injected


def _summarize(kr_eval):
    if kr_eval is None:
        return {"needs_eval": True}
    checks = kr_eval.get("auto_checks") or kr_eval.get("checks", [])
    counts = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
    for c in checks:
        st = c.get("status", "undetermined")
        counts[st] = counts.get(st, 0) + 1
    if counts["fail"] > 0:
        counts["overall"] = "fail"
    elif counts["undetermined"] > 0 or counts["not_modeled"] > 0:
        counts["overall"] = "partial"
    else:
        counts["overall"] = "pass"
    return counts


def main():
    parser = argparse.ArgumentParser(description="Phase 0.2.F: Borderline injection")
    parser.add_argument("--count", type=int, default=75,
                        help="Borderline candidates per ship (default: 75)")
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ships = args.ships or SHIPS
    rng = random.Random(args.seed)

    out_dir = CANDIDATES_DIR / "borderline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0.2.F borderline injection")
    print(f"  Count per ship: {args.count}")
    print()

    total = 0
    for ship in ships:
        candidates = load_fps_selected(ship)
        if not candidates:
            print(f"  {ship}: no candidates found, skipping")
            continue

        injected = inject_borderline(ship, candidates, args.count, rng)
        ship_dir = out_dir / ship
        ship_dir.mkdir(parents=True, exist_ok=True)

        for bl in injected:
            path = ship_dir / f"{bl['candidate_id']}.json"
            with open(path, "w") as f:
                json.dump(bl, f, indent=2, ensure_ascii=False)

        n_pass = sum(1 for bl in injected if bl.get("kr_summary", {}).get("overall") != "fail")
        n_fail = sum(1 for bl in injected if bl.get("kr_summary", {}).get("overall") == "fail")

        print(f"  {ship:6s}: {len(injected)} borderline injected "
              f"(pass/partial={n_pass}, fail={n_fail})")
        total += len(injected)

    # Summary
    with open(out_dir / "borderline_summary.json", "w") as f:
        json.dump({
            "total": total, "count_per_ship": args.count,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    print(f"\n  Total: {total} borderline candidates → {out_dir}")


if __name__ == "__main__":
    main()
