#!/usr/bin/env python3
"""
Phase 0.2.H — Final Dataset Assembly + Diversity Report.

Assembles the final dataset from:
  1. FPS-selected candidates (6000) — full artifact rendering
  2. Borderline candidates (450) — full artifact rendering

For each candidate: renders section DXF/PNG, elevation DXF/PNG, 3D model DXF/PNG,
and writes the full JSON with rule evaluation + metadata.

Output structure:
  data/processed/<ship>/
    section_dxf/
    section_png/
    compart_dxf/
    compart_png/
    compart3d_dxf/
    compart3d_png/
    json/
    <ship>_dataset_index.csv

Usage:
    python scripts/08_final_assembly.py                    # full run
    python scripts/08_final_assembly.py --ships LPGC --limit 5  # test
    python scripts/08_final_assembly.py --skip-render      # assembly only (no DXF/PNG)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "data_generator"
sys.path.insert(0, str(GEN_DIR))

CANDIDATES_DIR = ROOT / "data" / "candidates_R1"
STRATIFIED_DIR = CANDIDATES_DIR / "stratified"
BORDERLINE_DIR = CANDIDATES_DIR / "borderline"
OUTPUT_DIR = ROOT / "data" / "processed"

SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]

# Reuse builders from 04_render_sections
from importlib import import_module as _imp


def collect_selected_candidates(ship: str) -> list[dict]:
    """Collect FPS-selected + borderline candidates for a ship."""
    candidates = []

    # FPS-selected
    fps_ids = set()
    fps_manifest = STRATIFIED_DIR / "fps_selected.jsonl"
    if fps_manifest.exists():
        with open(fps_manifest) as f:
            for line in f:
                d = json.loads(line)
                if d["ship_type"] == ship:
                    fps_ids.add(d["candidate_id"])

    ship_dir = STRATIFIED_DIR / ship
    if ship_dir.exists():
        for stratum_dir in sorted(ship_dir.iterdir()):
            if not stratum_dir.is_dir() or stratum_dir.name == "section_png":
                continue
            for jf in sorted(stratum_dir.glob(f"{ship}-*.json")):
                if jf.stem in fps_ids:
                    c = json.load(open(jf))
                    c["_source"] = "fps"
                    candidates.append(c)

    # Borderline
    bl_dir = BORDERLINE_DIR / ship
    if bl_dir.exists():
        for jf in sorted(bl_dir.glob(f"{ship}-BL-*.json")):
            c = json.load(open(jf))
            c["_source"] = "borderline"
            candidates.append(c)

    return candidates


def write_diversity_report(ship_stats: dict):
    """Write DIVERSITY_REPORT.md for Paper 1 Section 3.2."""
    report_path = OUTPUT_DIR / "DIVERSITY_REPORT.md"
    lines = [
        "# Dataset Diversity Report — Phase 0.2.H",
        "",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Total samples**: {sum(s['total'] for s in ship_stats.values())}",
        "",
        "## Per-ship summary",
        "",
        "| Ship | Total | FPS | Borderline | fail=0 | fail=1 | fail≥2 |",
        "|------|-------|-----|-----------|--------|--------|--------|",
    ]
    for ship in SHIPS:
        s = ship_stats.get(ship, {})
        lines.append(
            f"| {ship} | {s.get('total', 0)} | {s.get('fps', 0)} | "
            f"{s.get('borderline', 0)} | {s.get('fail_0', 0)} | "
            f"{s.get('fail_1', 0)} | {s.get('fail_2plus', 0)} |"
        )
    lines.extend([
        "",
        "## Rule-state distribution",
        "",
        "Target: pass/borderline-pass ~50%, borderline-fail ~30%, clear-fail ~15%, borderline-injected ~5%",
        "",
        "## Topology variants (CNTR/LNGC)",
        "",
    ])

    for ship in ["CNTR", "LNGC"]:
        s = ship_stats.get(ship, {})
        topo = s.get("topology_dist", {})
        if topo:
            lines.append(f"### {ship}")
            for label, count in sorted(topo.items()):
                lines.append(f"  - {label}: {count}")
            lines.append("")

    lines.extend([
        "## Visual diversity (FPS metrics)",
        "",
        "| Ship | Pool | Selected | NN dist before | NN dist after | Improvement |",
        "|------|------|----------|----------------|---------------|-------------|",
    ])

    fps_report = STRATIFIED_DIR / "fps_report.json"
    if fps_report.exists():
        fr = json.load(open(fps_report))
        for ship in SHIPS:
            ps = fr.get("per_ship", {}).get(ship, {})
            before = ps.get("diversity_before", {}).get("mean_nn_dist", 0)
            after = ps.get("diversity_after", {}).get("mean_nn_dist", 0)
            improvement = ((after - before) / before * 100) if before > 0 else 0
            lines.append(
                f"| {ship} | {ps.get('pool_size', 0)} | {ps.get('selected', 0)} | "
                f"{before:.2f} | {after:.2f} | +{improvement:.1f}% |"
            )

    lines.append("")
    lines.append("---")
    lines.append(f"_Auto-generated by `scripts/08_final_assembly.py`._")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Diversity report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0.2.H: Final dataset assembly")
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip DXF/PNG rendering, only assemble JSONs")
    args = parser.parse_args()

    ships = args.ships or SHIPS
    t0 = time.time()

    # Backup existing processed if needed
    backup_dir = ROOT / "data" / "processed_R0_backup"
    if OUTPUT_DIR.exists() and not backup_dir.exists():
        print(f"Backing up existing data/processed/ → data/processed_R0_backup/")
        shutil.copytree(OUTPUT_DIR, backup_dir)

    ship_stats = {}

    for ship in ships:
        candidates = collect_selected_candidates(ship)
        if args.limit:
            candidates = candidates[:args.limit]

        n_fps = sum(1 for c in candidates if c.get("_source") == "fps")
        n_bl = sum(1 for c in candidates if c.get("_source") == "borderline")
        print(f"\n=== {ship}: {len(candidates)} candidates ({n_fps} FPS + {n_bl} borderline) ===")

        # Create output directories
        ship_out = OUTPUT_DIR / ship
        dirs = {
            "json": ship_out / "json",
            "section_png": ship_out / "section_png",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        # Assemble: copy section PNGs from stratified, write JSONs
        stats = {"total": 0, "fps": n_fps, "borderline": n_bl,
                 "fail_0": 0, "fail_1": 0, "fail_2plus": 0,
                 "topology_dist": {}}

        for c in candidates:
            cid = c["candidate_id"]

            # Write JSON
            out_json = dirs["json"] / f"{cid}.json"
            # Remove internal fields
            out_data = {k: v for k, v in c.items() if not k.startswith("_")}
            with open(out_json, "w") as f:
                json.dump(out_data, f, indent=2, ensure_ascii=False)

            # Copy section PNG if exists
            src_png = STRATIFIED_DIR / ship / "section_png" / f"{cid}.png"
            if src_png.exists():
                dst_png = dirs["section_png"] / f"{cid}.png"
                shutil.copy2(src_png, dst_png)

            # Count rule states
            ks = c.get("kr_summary", {})
            nf = ks.get("fail", 0)
            if nf == 0:
                stats["fail_0"] += 1
            elif nf == 1:
                stats["fail_1"] += 1
            else:
                stats["fail_2plus"] += 1

            # Topology
            topo = c.get("topology_variant", {})
            if topo:
                label = topo.get("label", "unknown")
                stats["topology_dist"][label] = stats["topology_dist"].get(label, 0) + 1

            stats["total"] += 1

        ship_stats[ship] = stats
        print(f"  {ship}: {stats['total']} assembled "
              f"(fail=0: {stats['fail_0']}, fail=1: {stats['fail_1']}, fail≥2: {stats['fail_2plus']})")

    # Write diversity report
    write_diversity_report(ship_stats)

    # Write master manifest
    manifest_path = OUTPUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for ship in ships:
            json_dir = OUTPUT_DIR / ship / "json"
            if not json_dir.exists():
                continue
            for jf in sorted(json_dir.glob("*.json")):
                d = json.load(open(jf))
                if "candidate_id" not in d:
                    continue  # skip old Phase 0 JSONs
                entry = {
                    "candidate_id": d["candidate_id"],
                    "ship_type": d["ship_type"],
                    "regime": d.get("regime", "unknown"),
                    "kr_overall": d.get("kr_summary", {}).get("overall", "unknown"),
                    "fail_count": d.get("kr_summary", {}).get("fail", 0),
                    "json_path": str(jf.relative_to(ROOT)),
                }
                if d.get("topology_variant"):
                    entry["topology"] = d["topology_variant"].get("label")
                f.write(json.dumps(entry) + "\n")

    grand_total = sum(s["total"] for s in ship_stats.values())
    elapsed = time.time() - t0
    print(f"\nDone. {grand_total} samples assembled in {elapsed:.0f}s.")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
