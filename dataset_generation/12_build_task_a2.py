#!/usr/bin/env python3
"""
Phase 1.2.A2 — Multi-view cross-reference QA builder for ShipBench.

Generates QA items that REQUIRE both section view + compart/compart3d views
to answer correctly. This is the unique "multi-view rule-grounded" angle that
differentiates ShipBench from generic VLM benchmarks.

Sub-tasks:
  A2.dim    — Cross-view dimension extraction (L, HL, L/B ratio)
              Section alone gives B,D; compart alone gives L,HL,n_hold.
              The model MUST reconcile both to answer.

  A2.extent — Feature extent across views ("DB seen in section runs along which
              compartments in compart view?")
              Per ship-type: tanker DB → cargo holds + ER; LNGC DB → holds only.

  A2.count  — Multi-view consistency ("How many cargo holds; verify section
              shows the cargo tank cross-section.")

Output: data/shipbench3d/task_A2.jsonl

Usage:
    python scripts/12_build_task_a2.py
    python scripts/12_build_task_a2.py --ships LPGC --limit 5
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed_R1"
QA_DIR = ROOT / "data" / "shipbench3d"
SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]

SHIP_DISPLAY = {
    "Tanker": "oil tanker", "VLCC": "VLCC",
    "BULKC": "bulk carrier", "CNTR": "container ship",
    "LNGC": "LNG carrier", "LPGC": "LPG carrier",
}

# ═══════════════════════════════════════════════════════════════════════
# Per-ship-type structural truths (what the multi-view should match)
# ═══════════════════════════════════════════════════════════════════════
# These encode what the parametric generator actually builds. Used as ground
# truth for A2.extent questions.

DB_EXTENT = {  # Where double bottom runs along the ship
    "Tanker": "cargo_holds_and_engine_room",
    "VLCC":   "cargo_holds_and_engine_room",
    "BULKC":  "cargo_holds_only",
    "CNTR":   "cargo_holds_and_engine_room",
    "LNGC":   "cargo_holds_only",   # cofferdams have own DB
    "LPGC":   "cargo_holds_only",
}

DB_EXTENT_DISPLAY = {
    "cargo_holds_only":         "cargo holds only",
    "cargo_holds_and_engine_room": "cargo holds and engine room",
    "all_compartments":         "all compartments including AFT and FWD",
}

DS_EXTENT = {  # Double-side (or its equivalent)
    "Tanker": "cargo_holds_and_engine_room",
    "VLCC":   "cargo_holds_and_engine_room",
    "BULKC":  "not_modeled",   # single-side hull
    "CNTR":   "cargo_holds_and_engine_room",
    "LNGC":   "cargo_holds_only",
    "LPGC":   "cargo_holds_only",
}


# ═══════════════════════════════════════════════════════════════════════
# QA generators — A2.dim (cross-view dimension)
# ═══════════════════════════════════════════════════════════════════════

DIM_TEMPLATES_L = [
    "Examining both views together, what is the ship's overall length L in meters?",
    "Based on the cross-section view (image 1) and the compartment view (image 2), "
    "what is the total length L of this {ship} in meters?",
    "Combine information from the two views to determine the overall length L (in m).",
]

DIM_TEMPLATES_HL = [
    "From the compartment view, identify the hold length HL. Verify the section view "
    "shows the corresponding cross-section. What is HL in meters?",
    "Looking at both views, what is the cargo hold length HL in meters?",
]

DIM_TEMPLATES_RATIO = [
    "Using B from the section view and L from the compartment view, what is the L/B ratio? "
    "Round to one decimal place.",
    "Compute the length-to-breadth (L/B) ratio using both views. Express to 1 decimal.",
]


def gen_a2_dim(cid: str, ship: str, gi: dict, rng: random.Random,
               compart_view: str = "compart_png") -> list[dict]:
    """Cross-view dimension extraction — model needs both views.
    compart_view: "compart_png" (longitudinal layout) or "compart3d_png" (3D)
    """
    qas = []
    L_m = float(gi.get("L_m", 0))
    B_m = float(gi.get("B_m", 0))
    HL_m = float(gi.get("HL_m", 0))
    if L_m <= 0 or B_m <= 0:
        return qas

    # A2-dim L
    q = rng.choice(DIM_TEMPLATES_L).format(ship=SHIP_DISPLAY[ship])
    qas.append({
        "qa_id": f"A2dim-{cid}-L",
        "task": "A2", "subtask": "A2.dim", "ship_type": ship, "candidate_id": cid,
        "question": q,
        "answer": str(int(L_m)) if L_m == int(L_m) else f"{L_m:.1f}",
        "answer_type": "numeric",
        "images": ["section_png", compart_view],
        "difficulty": "medium",
        "metadata": {"target": "L_m", "value": L_m, "unit": "m",
                     "tolerance_pct": 5.0},
    })

    # A2-dim HL
    if HL_m > 0:
        q = rng.choice(DIM_TEMPLATES_HL)
        qas.append({
            "qa_id": f"A2dim-{cid}-HL",
            "task": "A2", "subtask": "A2.dim", "ship_type": ship, "candidate_id": cid,
            "question": q,
            "answer": f"{HL_m:.1f}",
            "answer_type": "numeric",
            "images": ["section_png", compart_view],
            "difficulty": "medium",
            "metadata": {"target": "HL_m", "value": HL_m, "unit": "m",
                         "tolerance_pct": 5.0},
        })

    # A2-dim L/B ratio
    ratio = L_m / B_m
    q = rng.choice(DIM_TEMPLATES_RATIO)
    qas.append({
        "qa_id": f"A2dim-{cid}-LB",
        "task": "A2", "subtask": "A2.dim", "ship_type": ship, "candidate_id": cid,
        "question": q,
        "answer": f"{ratio:.1f}",
        "answer_type": "numeric",
        "images": ["section_png", compart_view],
        "difficulty": "hard",
        "metadata": {"target": "L_over_B", "value": ratio, "unit": "ratio",
                     "tolerance_pct": 7.0},
    })
    return qas


# ═══════════════════════════════════════════════════════════════════════
# QA generators — A2.extent
# ═══════════════════════════════════════════════════════════════════════

EXTENT_OPTIONS = [
    ("cargo_holds_only",          "cargo holds only"),
    ("cargo_holds_and_engine_room", "cargo holds and engine room"),
    ("all_compartments",          "all compartments including AFT and FWD"),
    ("not_modeled",               "this feature is not modeled in this ship"),
]

EXTENT_TEMPLATES_DB = [
    "The section view shows a double bottom (the layer between Bottom_Shell and IBTM). "
    "Looking at the compartment view, along which longitudinal segments does this "
    "double bottom run? Choose: A) cargo holds only, B) cargo holds and engine room, "
    "C) all compartments, D) not modeled.",

    "Identify the double bottom in the section drawing. Using the compartment view, "
    "determine which segments contain the double bottom. Options: A) cargo holds only, "
    "B) cargo holds + engine room, C) all compartments, D) not modeled.",
]

EXTENT_TEMPLATES_DS = [
    "The section shows a double-side structure (Side_Shell + IHull). From the "
    "compartment view, which segments contain this double-side? "
    "A) cargo holds only, B) cargo holds and engine room, C) all compartments, "
    "D) not modeled in this ship.",
]


def gen_a2_extent(cid: str, ship: str, gi: dict, rng: random.Random,
                  compart_view: str = "compart_png") -> list[dict]:
    """Feature extent across section ↔ compart views."""
    qas = []
    label_to_letter = {"cargo_holds_only": "A", "cargo_holds_and_engine_room": "B",
                       "all_compartments": "C", "not_modeled": "D"}

    # Double bottom extent
    db_label = DB_EXTENT.get(ship)
    if db_label:
        ans_letter = label_to_letter[db_label]
        q = rng.choice(EXTENT_TEMPLATES_DB)
        qas.append({
            "qa_id": f"A2ext-{cid}-DB",
            "task": "A2", "subtask": "A2.extent", "ship_type": ship,
            "candidate_id": cid,
            "question": q,
            "answer": ans_letter,
            "answer_type": "multiple_choice",
            "images": ["section_png", compart_view],
            "difficulty": "medium",
            "metadata": {"feature": "double_bottom",
                         "extent_label": db_label,
                         "ship_specific": True},
        })

    # Double side extent
    ds_label = DS_EXTENT.get(ship)
    if ds_label:
        ans_letter = label_to_letter[ds_label]
        q = rng.choice(EXTENT_TEMPLATES_DS)
        qas.append({
            "qa_id": f"A2ext-{cid}-DS",
            "task": "A2", "subtask": "A2.extent", "ship_type": ship,
            "candidate_id": cid,
            "question": q,
            "answer": ans_letter,
            "answer_type": "multiple_choice",
            "images": ["section_png", compart_view],
            "difficulty": "medium",
            "metadata": {"feature": "double_side",
                         "extent_label": ds_label,
                         "ship_specific": True},
        })

    return qas


# ═══════════════════════════════════════════════════════════════════════
# QA generators — A2.count
# ═══════════════════════════════════════════════════════════════════════

COUNT_TEMPLATES = [
    "Count the cargo holds visible in the compartment view. The section view shows the "
    "cross-section of one such hold. How many cargo holds does this {ship} have?",
    "Looking at the compartment view, count the cargo holds. Confirm the section view "
    "shows a cargo tank cross-section. Report the number of cargo holds.",
]


def gen_a2_count(cid: str, ship: str, gi: dict, rng: random.Random,
                  compart_view: str = "compart_png") -> list[dict]:
    n_hold = gi.get("number_of_hold")
    if not n_hold:
        return []
    q = rng.choice(COUNT_TEMPLATES).format(ship=SHIP_DISPLAY[ship])
    return [{
        "qa_id": f"A2cnt-{cid}",
        "task": "A2", "subtask": "A2.count", "ship_type": ship, "candidate_id": cid,
        "question": q,
        "answer": str(int(n_hold)),
        "answer_type": "numeric",
        "images": ["section_png", compart_view],
        "difficulty": "easy",
        "metadata": {"target": "number_of_hold", "value": int(n_hold)},
    }]


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compart-view", choices=["compart_png", "compart3d_png", "both"],
                        default="both",
                        help="Use 2D longitudinal, 3D perspective, or alternating both")
    args = parser.parse_args()

    ships = args.ships or SHIPS
    rng = random.Random(args.seed)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_qas = []
    by_subtask = {"A2.dim": 0, "A2.extent": 0, "A2.count": 0}
    by_ship = {}

    for ship in ships:
        json_dir = PROCESSED / ship / "json"
        if not json_dir.exists(): continue

        files = sorted(json_dir.glob("*.json"))
        if args.limit:
            files = files[:args.limit]

        ship_qas = []
        for jf in files:
            try:
                d = json.load(open(jf))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            # Use JSON file stem as the canonical id (the new dataset doesn't fill
            # candidate_id; assets are named {stem}.png / {stem}_Compart.png / {stem}_Compart3D.png).
            cid = jf.stem
            gi  = d.get("generator_inputs", {})
            if not gi: continue

            # Choose compart view per-candidate (alternating gives diversity)
            if args.compart_view == "both":
                view = "compart_png" if (rng.random() < 0.5) else "compart3d_png"
            else:
                view = args.compart_view

            # Resolve actual file names per ship-type rendering convention
            view_suffix = "_Compart" if view == "compart_png" else "_Compart3D"
            sec_png = PROCESSED / ship / "section_png" / f"{cid}.png"
            view_png = PROCESSED / ship / view / f"{cid}{view_suffix}.png"
            if not (sec_png.exists() and view_png.exists()):
                # Fallback to the other view
                alt = "compart3d_png" if view == "compart_png" else "compart_png"
                alt_suffix = "_Compart3D" if alt == "compart3d_png" else "_Compart"
                alt_png = PROCESSED / ship / alt / f"{cid}{alt_suffix}.png"
                if sec_png.exists() and alt_png.exists():
                    view = alt; view_png = alt_png
                else:
                    continue

            ship_qas.extend(gen_a2_dim(cid, ship, gi, rng, view))
            ship_qas.extend(gen_a2_extent(cid, ship, gi, rng, view))
            ship_qas.extend(gen_a2_count(cid, ship, gi, rng, view))

        all_qas.extend(ship_qas)
        by_ship[ship] = len(ship_qas)
        for q in ship_qas:
            by_subtask[q["subtask"]] = by_subtask.get(q["subtask"], 0) + 1
        print(f"  {ship}: {len(files)} candidates → {len(ship_qas)} A2 items")

    # Write output
    out_path = QA_DIR / "task_A2.jsonl"
    rng.shuffle(all_qas)
    with open(out_path, "w") as f:
        for q in all_qas:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0

    print(f"\nDone. {len(all_qas)} A2 items in {elapsed:.1f}s")
    print(f"  By sub-task: {by_subtask}")
    print(f"  By ship:    {by_ship}")
    print(f"  Output:     {out_path}")


if __name__ == "__main__":
    main()
