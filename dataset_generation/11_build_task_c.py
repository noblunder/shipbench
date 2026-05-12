#!/usr/bin/env python3
"""
Phase 1.2.C — Task C (Design Generation) QA builder for ShipBench.

C1: Spec-to-params — NL design spec → generator_inputs JSON
C2: Compliance-aware — NL spec + "must comply with X" → compliant JSON

Reads from data/processed/<ship>/json/ and generates QA items.

Usage:
    python scripts/11_build_task_c.py
    python scripts/11_build_task_c.py --ships CNTR --limit 5
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
# HUMAN-READABLE PARAM NAMES
# ═════════════════════════════════════════════════════════════════════
PARAM_DISPLAY = {
    "L_m": ("length overall", "m"),
    "B_m": ("breadth", "m"),
    "D_m": ("depth", "m"),
    "HL_m": ("hold length", "m"),
    "doubleSide_m": ("double side width", "m"),
    "doubleBottom_m": ("double bottom height", "m"),
    "bilgeRadius_m": ("bilge radius", "m"),
    "camberUpper_m": ("upper deck camber", "m"),
    "camberTrunk_m": ("trunk deck camber", "m"),
    "lbhd_ratio": ("longitudinal bulkhead position ratio", ""),
    "girder1_ratio": ("inner girder position ratio", ""),
    "girder2_ratio": ("outer girder position ratio", ""),
    "girder0_ratio": ("centerline girder position ratio", ""),
    "str1_ratio": ("upper stringer position ratio", ""),
    "str2_ratio": ("middle stringer position ratio", ""),
    "str3_ratio": ("lower stringer position ratio", ""),
    "number_of_hold": ("number of cargo holds", ""),
    "number_of_cofferdam": ("number of cofferdams", ""),
    "tswt_ext_deg": ("topside wing tank angle", "degrees"),
    "ds_from_side_m": ("double side offset from shell", "m"),
    "girder_y_m": ("girder offset from centerline", "m"),
    "outgir_ratio": ("outer girder height ratio", ""),
    "inner_slope_deg": ("inner hull slope angle", "degrees"),
    "gap_tswt_m": ("topside wing tank gap", "m"),
    "gap_hopper_m": ("hopper tank gap", "m"),
    "strClearance_m": ("stringer clearance", "m"),
}

# Ship type display names for NL generation
SHIP_DISPLAY = {
    "Tanker": "oil tanker",
    "VLCC": "VLCC (Very Large Crude Carrier)",
    "BULKC": "bulk carrier",
    "CNTR": "container ship",
    "LNGC": "LNG carrier",
    "LPGC": "LPG carrier",
}

# Key dimensions (always mentioned in NL spec)
KEY_PARAMS = {
    "Tanker": ["L_m", "B_m", "D_m", "doubleSide_m", "doubleBottom_m", "number_of_hold"],
    "VLCC": ["L_m", "B_m", "D_m", "doubleSide_m", "doubleBottom_m", "number_of_hold"],
    "BULKC": ["L_m", "B_m", "D_m", "doubleBottom_m", "number_of_hold"],
    "CNTR": ["L_m", "B_m", "D_m", "doubleSide_m", "doubleBottom_m", "number_of_hold"],
    "LNGC": ["L_m", "B_m", "D_m", "doubleSide_m", "doubleBottom_m", "number_of_hold"],
    "LPGC": ["L_m", "B_m", "D_m", "doubleBottom_m", "number_of_hold"],
}

# Rules that can be used as compliance constraints for C2
COMPLIANCE_RULES = {
    "Tanker": [
        ("double_bottom_height", "Pt1.Ch2.Sec3[2.3.1]", "minimum double bottom height"),
        ("double_side_width", "Pt1.Ch2.Sec3[3.1]", "minimum double side width"),
    ],
    "VLCC": [
        ("double_bottom_height", "Pt1.Ch2.Sec3[2.3.1]", "minimum double bottom height"),
        ("double_side_width", "Pt1.Ch2.Sec3[3.1]", "minimum double side width"),
    ],
    "BULKC": [
        ("double_bottom_height", "Pt1.Ch2.Sec3[2.3.1]", "minimum double bottom height"),
        ("bilge_radius", "Pt1.Ch2.Sec3[2.5]", "minimum bilge radius"),
    ],
    "CNTR": [
        ("double_bottom_height", "Pt1.Ch2.Sec3[2.3.1]", "minimum double bottom height"),
        ("hatch_opening_ratio", "Pt1.Ch2.Sec1[3.2]", "maximum hatch opening ratio"),
    ],
    "LNGC": [
        ("double_bottom_height", "Pt15.Ch2.Sec3[4.1]", "minimum double bottom height"),
        ("double_side_width", "Pt15.Ch2.Sec3[3.1]", "minimum double side width"),
        ("trunk_clearance", "Pt15.Ch2.Sec3[5.1]", "trunk deck clearance"),
    ],
    "LPGC": [
        ("double_bottom_height", "Pt1.Ch2.Sec3[2.3.1]", "minimum double bottom height"),
        ("tank_inboard_clearance", "Pt1.Ch2.Sec3[3.2]", "tank inboard clearance"),
    ],
}


# ═════════════════════════════════════════════════════════════════════
# NL SPEC GENERATORS
# ═════════════════════════════════════════════════════════════════════

def format_val(key: str, val) -> str:
    """Format a parameter value for NL text."""
    display, unit = PARAM_DISPLAY.get(key, (key, ""))
    if isinstance(val, float):
        val_str = f"{val:.1f}" if val != int(val) else f"{int(val)}"
    else:
        val_str = str(val)
    if unit:
        return f"{display} of {val_str} {unit}"
    return f"{display} of {val_str}"


def gen_nl_spec(ship_type: str, gi: dict, rng: random.Random,
                detail_level: str = "full") -> str:
    """Generate a natural language design specification from generator_inputs.

    detail_level:
      - "full": all params mentioned
      - "key_only": only key dimensions (for harder questions)
      - "partial": key + random subset of remaining
    """
    ship_name = SHIP_DISPLAY[ship_type]
    key_params = KEY_PARAMS[ship_type]

    # Build key dimension sentence
    key_parts = []
    for k in key_params:
        if k in gi:
            key_parts.append(format_val(k, gi[k]))

    # Remaining params
    remaining = [k for k in gi if k not in key_params]

    if detail_level == "key_only":
        detail_parts = []
    elif detail_level == "partial":
        n_extra = rng.randint(2, min(4, len(remaining)))
        chosen = rng.sample(remaining, n_extra)
        detail_parts = [format_val(k, gi[k]) for k in chosen]
    else:  # full
        detail_parts = [format_val(k, gi[k]) for k in remaining]

    # Assemble NL spec with template variation
    templates = [
        "Design a {ship} midship section with {key_dims}.",
        "Generate the structural parameters for a {ship} with {key_dims}.",
        "Create a midship cross-section design for a {ship} having {key_dims}.",
    ]
    t = rng.choice(templates)
    spec = t.format(ship=ship_name, key_dims=", ".join(key_parts))

    if detail_parts:
        spec += " Additional parameters: " + ", ".join(detail_parts) + "."

    return spec


def gen_c1(cid: str, ship_type: str, gi: dict, rng: random.Random) -> list[dict]:
    """Task C1: NL spec → generator_inputs JSON."""
    qas = []

    # Easy: full spec (all params in NL, model just reformats to JSON)
    spec_full = gen_nl_spec(ship_type, gi, rng, "full")
    qas.append({
        "qa_id": f"C1-{cid}-000",
        "task": "C1", "ship_type": ship_type, "candidate_id": cid,
        "question": spec_full + "\n\nReturn the generator_inputs as a JSON object.",
        "answer": json.dumps(gi, ensure_ascii=False),
        "answer_type": "json",
        "images": [],
        "difficulty": "easy",
        "metadata": {"detail_level": "full", "n_params": len(gi)},
    })

    # Medium: partial spec (some params omitted — model must infer reasonable defaults)
    spec_partial = gen_nl_spec(ship_type, gi, rng, "partial")
    qas.append({
        "qa_id": f"C1-{cid}-001",
        "task": "C1", "ship_type": ship_type, "candidate_id": cid,
        "question": spec_partial + "\n\nReturn the complete generator_inputs as a JSON object, "
                    "inferring reasonable values for any unspecified parameters.",
        "answer": json.dumps(gi, ensure_ascii=False),
        "answer_type": "json",
        "images": [],
        "difficulty": "medium",
        "metadata": {"detail_level": "partial", "n_params": len(gi)},
    })

    return qas


def gen_c2(cid: str, ship_type: str, gi: dict, csr: dict,
           rng: random.Random) -> list[dict]:
    """Task C2: Compliance-aware generation.
    NL spec + compliance constraint → JSON that satisfies the rule.
    """
    qas = []
    rules = COMPLIANCE_RULES.get(ship_type, [])
    if not rules:
        return qas

    checks = csr.get("auto_checks", [])
    check_map = {c["check_id"]: c for c in checks}

    for check_id, rule_ref, rule_desc in rules:
        c = check_map.get(check_id)
        if c is None or c["status"] not in ("pass", "fail"):
            continue

        # Build NL spec (key params only — harder)
        spec = gen_nl_spec(ship_type, gi, rng, "key_only")
        constraint = f"The design must comply with {rule_ref} ({rule_desc})."

        qas.append({
            "qa_id": f"C2-{cid}-{len(qas):03d}",
            "task": "C2", "ship_type": ship_type, "candidate_id": cid,
            "question": spec + " " + constraint +
                        "\n\nReturn compliant generator_inputs as a JSON object.",
            "answer": json.dumps(gi, ensure_ascii=False),
            "answer_type": "json",
            "images": [],
            "difficulty": "hard",
            "metadata": {
                "check_id": check_id, "rule_ref": rule_ref,
                "original_status": c["status"],
                "required": c.get("required"),
            },
        })

    return qas


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 1.2.C: Generate Task C QA items")
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ships = args.ships or SHIPS
    rng = random.Random(args.seed)
    t0 = time.time()

    QA_DIR.mkdir(parents=True, exist_ok=True)

    all_qas = []

    for ship in ships:
        json_dir = PROCESSED / ship / "json"
        if not json_dir.exists():
            continue

        files = sorted(json_dir.glob(f"{ship}-*.json"))
        if args.limit:
            files = files[:args.limit]

        ship_qas = []
        for jf in files:
            d = json.load(open(jf))
            cid = d.get("candidate_id") or d.get("sample_id", "UNKNOWN")
            gi = d["generator_inputs"]
            csr = d.get("csr") or d.get("kr_eval") or {}

            ship_qas.extend(gen_c1(cid, ship, gi, rng))
            ship_qas.extend(gen_c2(cid, ship, gi, csr, rng))

        all_qas.extend(ship_qas)
        print(f"  {ship:6s}: {len(files):4d} candidates → {len(ship_qas):5d} QA items")

    # Subsample to budget: C1 ~300, C2 ~200
    TASK_BUDGET = {"C1": 300, "C2": 200}

    by_task_raw = {}
    for qa in all_qas:
        by_task_raw.setdefault(qa["task"], []).append(qa)

    by_task = {}
    print(f"\n  Subsampling to budget:")
    for task in sorted(by_task_raw):
        pool = by_task_raw[task]
        budget = TASK_BUDGET.get(task, len(pool))
        if len(pool) > budget:
            # Stratified: equal per ship, then by difficulty
            by_ship = {}
            for qa in pool:
                by_ship.setdefault(qa["ship_type"], []).append(qa)
            per_ship = max(1, budget // len(by_ship))
            sampled = []
            for ship_name in sorted(by_ship):
                ship_pool = by_ship[ship_name]
                rng.shuffle(ship_pool)
                sampled.extend(ship_pool[:per_ship])
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

    # Write per-task JSONL
    total = 0
    for task, qas in sorted(by_task.items()):
        path = QA_DIR / f"task_{task}.jsonl"
        with open(path, "w") as f:
            for qa in qas:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")
        print(f"  {task}: {len(qas):5d} items → {path.name}")
        total += len(qas)

    # Append to combined file
    combined = QA_DIR / "all_qa.jsonl"
    with open(combined, "a") as f:
        for task in sorted(by_task):
            for qa in by_task[task]:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # Update stats
    stats_path = QA_DIR / "qa_stats.json"
    if stats_path.exists():
        stats = json.load(open(stats_path))
    else:
        stats = {"total": 0, "per_task": {}}

    for task, qas in by_task.items():
        stats["per_task"][task] = len(qas)
    stats["total"] = sum(stats["per_task"].values())
    stats["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Task C total: {total} QA items in {elapsed:.1f}s")
    print(f"  Updated stats: {stats['per_task']}")
    print(f"  Grand total: {stats['total']}")


if __name__ == "__main__":
    main()
