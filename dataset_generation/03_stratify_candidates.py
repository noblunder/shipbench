#!/usr/bin/env python3
"""
Phase 0.2.D — Rule State Stratification.

Reads 20K lightweight candidate JSONs from data/candidates_R1/<ship>/<regime>/,
assigns each to a stratum based on fail count, and selects a balanced preliminary
set (~1.5× final target) for downstream visual diversity filtering (Phase 0.2.E).

Strata (adapted from workflow — original 4-tier collapsed because no 3+ fail
candidates exist and undetermined is fixed at 1-2):

  A: fail=0  (all checked rules pass; some undetermined)     → 50%
  B: fail=1  (one rule fails; useful for counterfactual QA)  → 35%
  C: fail≥2  (multiple fails; hard cases)                    → 15%

Output:
  data/candidates_R1/stratified/  — symlinks or copied JSONs grouped by stratum
  data/candidates_R1/stratification_report.json — per-ship/stratum counts

Usage:
    python scripts/03_stratify_candidates.py                  # default 1.5× = ~9K
    python scripts/03_stratify_candidates.py --final-target 6000  # aim for 6K×1.5=9K
    python scripts/03_stratify_candidates.py --smoke 100      # small test
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = ROOT / "data" / "candidates_R1"

# Target strata fractions
STRATA = {
    "A_no_fail":    {"predicate": lambda nf: nf == 0, "fraction": 0.50, "label": "fail=0"},
    "B_one_fail":   {"predicate": lambda nf: nf == 1, "fraction": 0.35, "label": "fail=1"},
    "C_multi_fail": {"predicate": lambda nf: nf >= 2, "fraction": 0.15, "label": "fail≥2"},
}

SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]


def load_candidates(ship: str) -> list[dict]:
    """Load all candidate JSONs for a ship type."""
    ship_dir = CANDIDATES_DIR / ship
    candidates = []
    for regime_dir in ship_dir.iterdir():
        if not regime_dir.is_dir() or regime_dir.name == "stratified":
            continue
        for jf in sorted(regime_dir.glob(f"{ship}-*.json")):
            d = json.load(open(jf))
            d["_path"] = str(jf)
            candidates.append(d)
    return candidates


def assign_stratum(candidate: dict) -> str:
    """Determine which stratum a candidate belongs to."""
    n_fail = candidate["kr_summary"]["fail"]
    for stratum_id, cfg in STRATA.items():
        if cfg["predicate"](n_fail):
            return stratum_id
    return "C_multi_fail"  # fallback


def stratify_ship(
    ship: str,
    per_ship_target: int,
    oversample: float,
    rng: random.Random,
) -> dict:
    """Select balanced candidates for one ship.

    Returns dict with per-stratum lists of selected candidate dicts.
    """
    candidates = load_candidates(ship)
    # Group by stratum
    by_stratum = defaultdict(list)
    for c in candidates:
        by_stratum[assign_stratum(c)].append(c)

    selection = {}
    stats = {}

    for stratum_id, cfg in STRATA.items():
        pool = by_stratum.get(stratum_id, [])
        desired = int(per_ship_target * oversample * cfg["fraction"])
        actual = min(desired, len(pool))
        if actual < len(pool):
            selected = rng.sample(pool, actual)
        else:
            selected = list(pool)
        selection[stratum_id] = selected
        stats[stratum_id] = {
            "label": cfg["label"],
            "pool_size": len(pool),
            "desired": desired,
            "selected": actual,
        }

    return {"selection": selection, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Phase 0.2.D: Stratify candidates")
    parser.add_argument("--final-target", type=int, default=6000,
                        help="Final dataset target (after FPS in 0.2.E). Default 6000.")
    parser.add_argument("--oversample", type=float, default=1.5,
                        help="Oversample factor (select this × final_target). Default 1.5.")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke test: select N total across all ships")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.smoke:
        per_ship_target = args.smoke // len(SHIPS)
        oversample = 1.0
    else:
        per_ship_target = args.final_target // len(SHIPS)
        oversample = args.oversample

    out_dir = CANDIDATES_DIR / "stratified"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print(f"Phase 0.2.D stratification")
    print(f"  Final target:     {args.final_target}")
    print(f"  Per-ship target:  {per_ship_target}")
    print(f"  Oversample:       {oversample}×")
    print(f"  Selection size:   ~{int(per_ship_target * oversample * len(SHIPS))}")
    print()

    report = {"per_ship": {}, "strata_totals": defaultdict(int)}
    all_selected = []

    for ship in SHIPS:
        result = stratify_ship(ship, per_ship_target, oversample, rng)
        report["per_ship"][ship] = result["stats"]

        ship_total = 0
        for stratum_id, selected in result["selection"].items():
            stratum_dir = out_dir / ship / stratum_id
            stratum_dir.mkdir(parents=True, exist_ok=True)
            for c in selected:
                # Copy candidate JSON to stratified dir
                src = Path(c["_path"])
                dst = stratum_dir / src.name
                shutil.copy2(src, dst)
                all_selected.append(c)
            n = len(selected)
            ship_total += n
            report["strata_totals"][stratum_id] += n

        stats_str = "  ".join(
            f"{sid}={result['stats'][sid]['selected']}/{result['stats'][sid]['pool_size']}"
            for sid in STRATA
        )
        print(f"  {ship:6s}: {ship_total:5d} selected  ({stats_str})")

    grand_total = sum(report["strata_totals"].values())
    report["grand_total"] = grand_total

    # Write report
    with open(CANDIDATES_DIR / "stratification_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    # Write manifest of selected candidate IDs
    manifest = [{"candidate_id": c["candidate_id"], "ship_type": c["ship_type"],
                 "regime": c["regime"], "stratum": assign_stratum(c),
                 "fail_count": c["kr_summary"]["fail"]}
                for c in all_selected]
    with open(out_dir / "selected_manifest.jsonl", "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")

    print(f"\n  Grand total: {grand_total} candidates → {out_dir}")
    print(f"  Strata: " + "  ".join(f"{k}={v}" for k, v in sorted(report["strata_totals"].items())))


if __name__ == "__main__":
    main()
