#!/usr/bin/env python3
"""
Phase 1.2 — QA Auto-Generator for ShipBench.

Generates Task A (Multi-View VQA) and Task B (Rule-Grounded Reasoning) QA items
from candidate JSONs in data/processed/.

Task A sub-tasks:
  A1: Member presence (yes/no) — "Does this section show a [member]?"
  A2: Member correspondence — deferred (requires multi-view, elev/3D not rendered yet)
  A3: Dimension extraction — "What is the [dimension] in [unit]?"
  A4: Hotspot location — "Where is the [hotspot]? Give (y,z) in mm."

Task B sub-tasks:
  B1: Compliance check — "Does this comply with [rule]?"
  B2: Counterfactual — "If [param] were [value], would [check] pass?"
  B3: Rule citation — "Why does [check] fail? Which rule applies?"

Usage:
    python scripts/10_build_qa.py                       # full
    python scripts/10_build_qa.py --ships LPGC --limit 5  # test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
QA_DIR = ROOT / "data" / "shipbench"
SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]

# ═════════════════════════════════════════════════════════════════════
# MEMBER NAMES per ship type (present in all valid sections)
# ═════════════════════════════════════════════════════════════════════
EXPECTED_MEMBERS = {
    "Tanker": {
        "always": ["Upper_Deck", "Side_Shell", "Bottom_Shell", "IHull", "IBTM",
                    "Bilge", "In_Girder", "Out_Girder", "Str1", "Str2", "Str3"],
        "sometimes": ["LBHD"],   # only when LB>0
        "never": ["Trunk_Deck", "Cargo_Tank", "Hatch_Coaming"],
    },
    "VLCC": {
        "always": ["Upper_Deck", "Side_Shell", "Bottom_Shell", "IHull", "IBTM",
                    "Bilge", "In_Girder", "Out_Girder", "LBHD",
                    "Str1", "Str2", "Str3"],
        "sometimes": [],
        "never": ["Trunk_Deck", "Cargo_Tank", "Hatch_Coaming"],
    },
    "BULKC": {
        "always": ["Upper_Deck", "Side_Shell", "Bottom_Shell", "IBTM",
                    "Bilge", "Hopper_Tank_SWT", "Top_SWT", "In_Girder", "Out_Girder",
                    "Str1", "Str2"],
        "sometimes": [],
        "never": ["IHull", "LBHD", "Cargo_Tank", "Trunk_Deck"],
    },
    "CNTR": {
        "always": ["Upper_Deck", "Side_Shell", "Bottom_Shell", "IHull", "IBTM",
                    "Bilge", "In_Girder", "Out_Girder", "Hatch_Coaming",
                    "Str1", "Str2", "Str3"],
        "sometimes": [],
        "never": ["LBHD", "Cargo_Tank", "Trunk_Deck"],
    },
    "LNGC": {
        "always": ["Upper_Deck", "Side_Shell", "Bottom_Shell", "IHull", "IBTM",
                    "Bilge", "Trunk_Deck", "InnerDeck_Slant",
                    "In_Girder", "Out_Girder", "Str1", "Str2", "Str3"],
        "sometimes": [],
        "never": ["LBHD", "Hatch_Coaming"],
    },
    "LPGC": {
        "always": ["Upper_Deck", "Side_Shell", "Bottom_Shell", "IBTM",
                    "Bilge", "Hopper_Tank_SWT", "Top_SWT", "Cargo_Tank",
                    "In_Girder", "Out_Girder", "Str1", "Str2"],
        "sometimes": [],
        "never": ["IHull", "LBHD", "Trunk_Deck", "Hatch_Coaming"],
    },
}

# Human-readable member names
MEMBER_DISPLAY = {
    "Upper_Deck": "upper deck", "Side_Shell": "side shell",
    "Bottom_Shell": "bottom shell", "IHull": "inner hull (double side)",
    "IBTM": "inner bottom (double bottom)", "Bilge": "bilge curve",
    "In_Girder": "inner girder", "Out_Girder": "outer girder",
    "LBHD": "longitudinal bulkhead", "Str1": "first stringer",
    "Str2": "second stringer", "Str3": "third stringer",
    "Hopper_Tank_SWT": "hopper side water tank", "Top_SWT": "topside water tank",
    "Cargo_Tank": "cargo tank boundary", "Trunk_Deck": "trunk deck",
    "InnerDeck_Slant": "inner deck slant (membrane containment)",
    "Hatch_Coaming": "hatch coaming",
}

# ═════════════════════════════════════════════════════════════════════
# DIMENSION QUESTIONS
# ═════════════════════════════════════════════════════════════════════
DIMENSION_QS = [
    {"field": "B_m", "q": "What is the ship breadth (B) in meters?", "unit": "m", "tol": 0.05},
    {"field": "D_m", "q": "What is the ship depth (D) in meters?", "unit": "m", "tol": 0.05},
    {"field": "doubleBottom_m", "q": "What is the double bottom height in meters?",
     "unit": "m", "tol": 0.05, "alias": ["DB"]},
    {"field": "doubleSide_m", "q": "What is the double side width in meters?",
     "unit": "m", "tol": 0.05, "ships": ["Tanker", "VLCC", "CNTR", "LNGC"]},
    {"field": "bilgeRadius_m", "q": "What is the bilge radius in meters?", "unit": "m", "tol": 0.1},
    {"field": "camberUpper_m", "q": "What is the upper deck camber height in meters?",
     "unit": "m", "tol": 0.1},
    {"field": "number_of_hold", "q": "How many cargo holds does this ship have?",
     "unit": "count", "tol": 0.0, "answer_type": "integer"},
    {"field": "L_m", "q": "What is the overall ship length in meters?",
     "unit": "m", "tol": 0.05, "views": ["compart_png"]},
]

# ═════════════════════════════════════════════════════════════════════
# RULE CHECK display names
# ═════════════════════════════════════════════════════════════════════
CHECK_DISPLAY = {
    "oil_tanker_scope": "CSR-H scope applicability (L ≥ 150 m)",
    "typical_midship_arrangement": "typical midship section arrangement",
    "double_bottom_height": "minimum double bottom height",
    "double_side_width": "minimum double side width",
    "double_side_clearance": "double side clearance",
    "double_bottom_framing": "double bottom framing system",
    "double_side_framing": "double side framing system",
    "weld_joint_detail": "weld joint detail",
    "cntr_scope": "KR Pt14 scope",
    "hatch_opening_ratio": "hatch opening ratio (b_hatch/B ≤ 0.92)",
    "torsional_stiffness": "torsional stiffness (warping analysis)",
    "hatch_coaming_height": "minimum hatch coaming height",
    "lngc_scope": "IGC scope",
    "membrane_tank": "membrane containment system",
    "cofferdam_req": "cofferdam between cargo holds",
    "tank_inboard_clearance": "IGC 2.4.1 tank inboard clearance",
    "inner_hull_slope": "inner hull slope angle",
    "lpgc_scope": "IGC scope",
    "independent_tank": "independent tank type",
    "cargo_tank_clearance": "cargo tank clearance",
    "hopper_slope_angle": "hopper slope angle",
    "tswt_arrangement": "topside water tank arrangement",
    "longitudinal_framing": "longitudinal framing system",
    "bulk_carrier_scope": "CSR-H scope (bulk carrier)",
    "hopper_tank_angle": "hopper tank angle",
}


# ═════════════════════════════════════════════════════════════════════
# QA GENERATORS
# ═════════════════════════════════════════════════════════════════════

def gen_a1_member_presence(cid: str, ship_type: str, gi: dict, rng: random.Random) -> list[dict]:
    """Task A1: Member presence yes/no questions."""
    qas = []
    info = EXPECTED_MEMBERS.get(ship_type, {})
    qa_idx = 0

    # "always" members → answer=yes
    for m in info.get("always", []):
        display = MEMBER_DISPLAY.get(m, m)
        qas.append({
            "qa_id": f"A1-{cid}-{qa_idx:03d}",
            "task": "A1", "ship_type": ship_type, "candidate_id": cid,
            "question": f"Does this section drawing show a {display}?",
            "answer": "yes", "answer_type": "binary",
            "images": ["section_png"],
            "difficulty": "easy",
            "metadata": {"member": m, "expected_presence": "always"},
        })
        qa_idx += 1

    # "never" members → answer=no (sample 2-3)
    never = info.get("never", [])
    for m in rng.sample(never, min(3, len(never))):
        display = MEMBER_DISPLAY.get(m, m)
        qas.append({
            "qa_id": f"A1-{cid}-{qa_idx:03d}",
            "task": "A1", "ship_type": ship_type, "candidate_id": cid,
            "question": f"Does this section drawing show a {display}?",
            "answer": "no", "answer_type": "binary",
            "images": ["section_png"],
            "difficulty": "easy",
            "metadata": {"member": m, "expected_presence": "never"},
        })
        qa_idx += 1

    # "sometimes" members → answer depends on params
    for m in info.get("sometimes", []):
        display = MEMBER_DISPLAY.get(m, m)
        if m == "LBHD":
            answer = "yes" if gi.get("lbhd_ratio", 0.0) > 0.0 else "no"
        else:
            answer = "unknown"
        if answer != "unknown":
            qas.append({
                "qa_id": f"A1-{cid}-{qa_idx:03d}",
                "task": "A1", "ship_type": ship_type, "candidate_id": cid,
                "question": f"Does this section drawing show a {display}?",
                "answer": answer, "answer_type": "binary",
                "images": ["section_png"],
                "difficulty": "medium",
                "metadata": {"member": m, "expected_presence": "conditional",
                             "condition": f"lbhd_ratio={gi.get('lbhd_ratio', 0.0)}"},
            })
            qa_idx += 1

    return qas


def gen_a3_dimension(cid: str, ship_type: str, gi: dict, rng: random.Random) -> list[dict]:
    """Task A3: Dimension extraction questions."""
    qas = []
    qa_idx = 0

    for dq in DIMENSION_QS:
        ships_allowed = dq.get("ships")
        if ships_allowed and ship_type not in ships_allowed:
            continue
        views = dq.get("views", ["section_png"])
        val = gi.get(dq["field"])
        if val is None:
            continue
        answer = str(int(val)) if dq.get("answer_type") == "integer" else str(round(val, 2))

        qas.append({
            "qa_id": f"A3-{cid}-{qa_idx:03d}",
            "task": "A3", "ship_type": ship_type, "candidate_id": cid,
            "question": dq["q"],
            "answer": answer,
            "answer_type": dq.get("answer_type", "numeric"),
            "unit": dq["unit"],
            "tolerance": dq["tol"],
            "images": views,
            "difficulty": "medium" if dq["field"] in ("L_m", "number_of_hold") else "easy",
            "metadata": {"source_field": f"generator_inputs.{dq['field']}"},
        })
        qa_idx += 1

    return qas


def gen_a4_hotspot(cid: str, ship_type: str, kr_eval: dict, rng: random.Random) -> list[dict]:
    """Task A4: Hotspot location questions."""
    qas = []
    hotspots = kr_eval.get("detail_hotspots", [])
    qa_idx = 0

    for hs in hotspots:
        pt = hs.get("point_mm")
        if pt is None or not isinstance(pt, list) or len(pt) < 2:
            continue
        y_mm, z_mm = pt[0], pt[1]
        hid = hs["hotspot_id"]
        display = hid.replace("_", " ")

        qas.append({
            "qa_id": f"A4-{cid}-{qa_idx:03d}",
            "task": "A4", "ship_type": ship_type, "candidate_id": cid,
            "question": f"Where is the {display} in this section drawing? "
                        f"Give the coordinates as (y_mm, z_mm) from the centerline baseline.",
            "answer": f"({y_mm}, {z_mm})",
            "answer_type": "coordinate_mm",
            "tolerance_mm": 500,
            "images": ["section_png"],
            "difficulty": "hard",
            "metadata": {"hotspot_id": hid, "rule_ref": hs.get("rule_ref", ""),
                         "y_mm": y_mm, "z_mm": z_mm},
        })
        qa_idx += 1

    return qas


def gen_b1_compliance(cid: str, ship_type: str, kr_eval: dict, rng: random.Random) -> list[dict]:
    """Task B1: Compliance check questions."""
    qas = []
    checks = kr_eval.get("auto_checks", [])
    qa_idx = 0

    for c in checks:
        check_id = c["check_id"]
        status = c["status"]
        if status not in ("pass", "fail"):
            continue  # skip undetermined/not_modeled — answer is ambiguous

        display = CHECK_DISPLAY.get(check_id, check_id.replace("_", " "))
        rule_ref = c.get("rule_ref", "")

        # Build question variants
        if rule_ref:
            q = f"Does this {ship_type} design comply with {rule_ref} ({display})?"
        else:
            q = f"Does this {ship_type} design satisfy the {display} requirement?"

        qas.append({
            "qa_id": f"B1-{cid}-{qa_idx:03d}",
            "task": "B1", "ship_type": ship_type, "candidate_id": cid,
            "question": q,
            "answer": status,
            "answer_type": "status",
            "images": ["section_png"],
            "difficulty": "easy" if status == "pass" else "medium",
            "metadata": {
                "check_id": check_id, "rule_ref": rule_ref,
                "actual": c.get("actual"), "required": c.get("required"),
            },
        })
        qa_idx += 1

    return qas


def gen_b2_counterfactual(cid: str, ship_type: str, gi: dict, kr_eval: dict,
                          rng: random.Random) -> list[dict]:
    """Task B2: Counterfactual rule reasoning.
    Pick a check and ask "what if param were X?"
    """
    qas = []
    checks = kr_eval.get("auto_checks", [])
    qa_idx = 0

    # Map check_id to the param it depends on
    param_map = {
        "double_bottom_height": ("doubleBottom_m", "DB"),
        "double_side_width": ("doubleSide_m", "DS"),
        "hatch_opening_ratio": ("doubleSide_m", "DS"),
        "tank_inboard_clearance": ("doubleSide_m", "DS"),
    }

    for c in checks:
        check_id = c["check_id"]
        if check_id not in param_map:
            continue
        status = c["status"]
        if status not in ("pass", "fail"):
            continue

        json_key, short_name = param_map[check_id]
        current_val = gi.get(json_key)
        required = c.get("required")
        if current_val is None or required is None:
            continue

        # Extract threshold
        if isinstance(required, dict):
            threshold = required.get("min_m") or required.get("min_mm")
            if threshold and isinstance(threshold, (int, float)):
                threshold = float(threshold)
            else:
                continue
        elif isinstance(required, (int, float)):
            threshold = float(required)
        else:
            continue

        # Generate counterfactual value
        if status == "pass":
            # Ask about reducing below threshold
            cf_val = round(threshold - 0.3, 1)
            expected_status = "fail"
        else:
            # Ask about increasing above threshold
            cf_val = round(threshold + 0.3, 1)
            expected_status = "pass"

        display = CHECK_DISPLAY.get(check_id, check_id.replace("_", " "))
        unit = "m" if "mm" not in json_key else "mm"

        qas.append({
            "qa_id": f"B2-{cid}-{qa_idx:03d}",
            "task": "B2", "ship_type": ship_type, "candidate_id": cid,
            "question": f"The current {short_name} is {current_val} {unit}. "
                        f"If it were changed to {cf_val} {unit}, would the "
                        f"{display} check pass or fail?",
            "answer": expected_status,
            "answer_type": "status",
            "images": ["section_png"],
            "difficulty": "hard",
            "metadata": {
                "check_id": check_id,
                "original_value": current_val,
                "counterfactual_value": cf_val,
                "threshold": threshold,
                "original_status": status,
                "expected_status": expected_status,
            },
        })
        qa_idx += 1

    return qas


def gen_b3_citation(cid: str, ship_type: str, kr_eval: dict, rng: random.Random) -> list[dict]:
    """Task B3: Rule citation for failing checks."""
    qas = []
    checks = kr_eval.get("auto_checks", [])
    qa_idx = 0

    for c in checks:
        if c["status"] != "fail":
            continue
        check_id = c["check_id"]
        rule_ref = c.get("rule_ref", "")
        display = CHECK_DISPLAY.get(check_id, check_id.replace("_", " "))
        actual = c.get("actual")
        required = c.get("required")

        # Build explanation
        if isinstance(required, dict):
            req_str = ", ".join(f"{k}={v}" for k, v in required.items())
        else:
            req_str = str(required)

        explanation = f"Actual={actual}, Required={req_str}"
        if rule_ref:
            explanation = f"{rule_ref}: {explanation}"

        qas.append({
            "qa_id": f"B3-{cid}-{qa_idx:03d}",
            "task": "B3", "ship_type": ship_type, "candidate_id": cid,
            "question": f"This design fails the {display} check. "
                        f"Which classification rule applies and what is the requirement?",
            "answer": explanation,
            "answer_type": "free_text",
            "images": ["section_png"],
            "difficulty": "hard",
            "metadata": {
                "check_id": check_id, "rule_ref": rule_ref,
                "actual": actual, "required": required,
            },
        })
        qa_idx += 1

    return qas


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def process_candidate(candidate: dict, rng: random.Random) -> list[dict]:
    """Generate all QA items for one candidate."""
    cid = candidate.get("candidate_id") or candidate.get("sample_id", "UNKNOWN")
    ship = candidate["ship_type"]
    gi = candidate["generator_inputs"]
    kr = candidate.get("csr") or candidate.get("kr_eval") or {}

    qas = []
    qas.extend(gen_a1_member_presence(cid, ship, gi, rng))
    qas.extend(gen_a3_dimension(cid, ship, gi, rng))
    qas.extend(gen_a4_hotspot(cid, ship, kr, rng))
    qas.extend(gen_b1_compliance(cid, ship, kr, rng))
    qas.extend(gen_b2_counterfactual(cid, ship, gi, kr, rng))
    qas.extend(gen_b3_citation(cid, ship, kr, rng))
    return qas


def main():
    parser = argparse.ArgumentParser(description="Phase 1.2: Generate ShipBench QA items")
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ships = args.ships or SHIPS
    rng = random.Random(args.seed)
    t0 = time.time()

    QA_DIR.mkdir(parents=True, exist_ok=True)

    all_qas = []
    task_counts = {}

    for ship in ships:
        json_dir = PROCESSED / ship / "json"
        if not json_dir.exists():
            continue

        # Only process new candidate JSONs (have candidate_id field)
        files = sorted(json_dir.glob(f"{ship}-*.json"))
        if args.limit:
            files = files[:args.limit]

        ship_qas = []
        for jf in files:
            d = json.load(open(jf))
            qas = process_candidate(d, rng)
            ship_qas.extend(qas)

        all_qas.extend(ship_qas)

        # Count per task
        for qa in ship_qas:
            task = qa["task"]
            task_counts.setdefault(task, {}).setdefault(ship, 0)
            task_counts[task][ship] += 1

        n = len(ship_qas)
        print(f"  {ship:6s}: {len(files):4d} candidates → {n:5d} QA items")

    # ── Subsample to target budget ──────────────────────────────────
    # Target: Task A ~3000, Task B ~2000 (README spec)
    TASK_BUDGET = {
        "A1": 750,   # member presence
        "A3": 1500,  # dimension extraction (richest task)
        "A4": 750,   # hotspot location
        "B1": 1200,  # compliance check
        "B2": 500,   # counterfactual
        "B3": 300,   # rule citation (fewer failing checks)
    }

    by_task_raw = {}
    for qa in all_qas:
        by_task_raw.setdefault(qa["task"], []).append(qa)

    by_task = {}
    total_sampled = 0
    print(f"\n  Subsampling to budget:")
    for task in sorted(by_task_raw):
        pool = by_task_raw[task]
        budget = TASK_BUDGET.get(task, len(pool))
        if len(pool) > budget:
            # Stratified subsample: equal per ship type
            by_ship = {}
            for qa in pool:
                by_ship.setdefault(qa["ship_type"], []).append(qa)
            per_ship = max(1, budget // len(by_ship))
            sampled = []
            for ship_name in sorted(by_ship):
                ship_pool = by_ship[ship_name]
                rng.shuffle(ship_pool)
                sampled.extend(ship_pool[:per_ship])
            # Fill remainder from leftovers
            remaining = budget - len(sampled)
            if remaining > 0:
                leftover = [q for q in pool if q not in sampled]
                rng.shuffle(leftover)
                sampled.extend(leftover[:remaining])
            rng.shuffle(sampled)
            by_task[task] = sampled
        else:
            by_task[task] = pool
        print(f"    {task}: {len(pool):6d} raw → {len(by_task[task]):5d} sampled (budget {budget})")
        total_sampled += len(by_task[task])

    all_qas = []
    for task in sorted(by_task):
        all_qas.extend(by_task[task])

    # Write per-task JSONL files
    for task, qas in sorted(by_task.items()):
        path = QA_DIR / f"task_{task}.jsonl"
        with open(path, "w") as f:
            for qa in qas:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"  {task}: {len(qas):5d} items → {path.name}")

    # Write combined file
    combined = QA_DIR / "all_qa.jsonl"
    with open(combined, "w") as f:
        for qa in all_qas:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # Stats
    stats = {
        "total": len(all_qas),
        "per_task": {t: sum(s.values()) for t, s in task_counts.items()},
        "per_task_per_ship": task_counts,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(QA_DIR / "qa_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Total: {len(all_qas)} QA items in {elapsed:.1f}s")
    print(f"  Per-task: {stats['per_task']}")


if __name__ == "__main__":
    main()
