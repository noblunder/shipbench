#!/usr/bin/env python3
"""
Phase 0.2.G — Topology Variation Labeling.

Tags FPS-selected CNTR and LNGC candidates with topology variant labels
based on their structural parameter values. These labels are used in
Paper 1 for topology-aware evaluation and in the VLM training for
structural category recognition.

CNTR topology variants:
  - girder_tier: "2-tier" (G2 < 0.8) vs "3-tier" (G2 ≥ 0.8)
    → outer girder close to inner hull = 3-tier bench girder arrangement
  - hatch_width: "narrow" (DS/B > 0.06), "standard" (0.04 < DS/B ≤ 0.06), "wide" (DS/B ≤ 0.04)
    → determines hatch opening width relative to beam

LNGC topology variants:
  - trunk_deck: "narrow" (CT < 0.4), "standard" (0.4 ≤ CT ≤ 0.7), "wide" (CT > 0.7)
    → trunk deck camber determines visual trunk width
  - inner_slope: "steep" (inner_slope_deg ≥ 50°), "standard" (< 50°)
    → inner hull slope angle for membrane tank containment

Output: adds topology_variant field to each candidate's JSON.

Usage:
    python scripts/07_topology_label.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATIFIED_DIR = ROOT / "data" / "candidates_R1" / "stratified"
FPS_MANIFEST = STRATIFIED_DIR / "fps_selected.jsonl"


def classify_cntr(gi: dict) -> dict:
    """Classify CNTR topology variant."""
    B = gi.get("B_m", 50)
    DS = gi.get("doubleSide_m", 2.0)
    G2 = gi.get("girder2_ratio", 0.8)

    girder_tier = "3-tier" if G2 >= 0.8 else "2-tier"

    ds_ratio = DS / B if B > 0 else 0
    if ds_ratio > 0.06:
        hatch_width = "narrow"
    elif ds_ratio > 0.04:
        hatch_width = "standard"
    else:
        hatch_width = "wide"

    return {
        "girder_tier": girder_tier,
        "hatch_width": hatch_width,
        "label": f"{girder_tier}_{hatch_width}",
    }


def classify_lngc(gi: dict) -> dict:
    """Classify LNGC topology variant."""
    CT = gi.get("camberTrunk_m", 0.5)
    inner_slope = gi.get("inner_slope_deg", 45.0)

    if CT is None:
        CT = 0.5
    if CT < 0.4:
        trunk_deck = "narrow"
    elif CT <= 0.7:
        trunk_deck = "standard"
    else:
        trunk_deck = "wide"

    slope_cat = "steep" if inner_slope is not None and inner_slope >= 50 else "standard"

    return {
        "trunk_deck": trunk_deck,
        "inner_slope": slope_cat,
        "label": f"{trunk_deck}_{slope_cat}",
    }


CLASSIFIERS = {
    "CNTR": classify_cntr,
    "LNGC": classify_lngc,
}


def main():
    # Load FPS selected candidate IDs
    fps_ids = {}
    with open(FPS_MANIFEST) as f:
        for line in f:
            d = json.loads(line)
            fps_ids.setdefault(d["ship_type"], set()).add(d["candidate_id"])

    for ship_type, classifier in CLASSIFIERS.items():
        ship_dir = STRATIFIED_DIR / ship_type
        if not ship_dir.exists():
            print(f"SKIP {ship_type}: no dir")
            continue

        selected = fps_ids.get(ship_type, set())
        labeled = 0
        dist = Counter()

        for stratum_dir in sorted(ship_dir.iterdir()):
            if not stratum_dir.is_dir() or stratum_dir.name == "section_png":
                continue
            for jf in sorted(stratum_dir.glob(f"{ship_type}-*.json")):
                if jf.stem not in selected:
                    continue
                candidate = json.load(open(jf))
                gi = candidate["generator_inputs"]
                topo = classifier(gi)
                candidate["topology_variant"] = topo
                with open(jf, "w") as f:
                    json.dump(candidate, f, indent=2, ensure_ascii=False)
                dist[topo["label"]] += 1
                labeled += 1

        print(f"{ship_type}: {labeled} candidates labeled")
        for label, count in sorted(dist.items()):
            print(f"  {label}: {count}")
        print()


if __name__ == "__main__":
    main()
