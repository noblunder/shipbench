#!/usr/bin/env python3
"""
Phase 0.2.E (step 1) — Render section PNGs for stratified candidates.

Reads candidate JSONs from data/candidates_R1/stratified/<ship>/<stratum>/,
instantiates ship geometry, exports section DXF→PNG via each ship's DXFExporter.
Section PNGs are saved alongside candidate JSONs for DINOv2 embedding.

Usage:
    python scripts/04_render_sections.py                   # full run
    python scripts/04_render_sections.py --ships LPGC --limit 10  # test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "data_generator"
sys.path.insert(0, str(GEN_DIR))

STRATIFIED_DIR = ROOT / "data" / "candidates_R1" / "stratified"
SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]

# Module cache
_MODULES = {}

def _get_module(ship_type: str):
    if ship_type in _MODULES:
        return _MODULES[ship_type]
    import importlib
    mod = importlib.import_module(f"{ship_type}_Data_generation")
    _MODULES[ship_type] = mod
    return mod


# ── Ship builders: candidate JSON → ship object ──

def _build_tanker(gi: dict):
    mod = _get_module("Tanker")
    B, D = gi["B_m"], gi["D_m"]
    ship = mod.Tanker(
        L=gi["L_m"], B=B, D=D,
        d_ds=gi["doubleSide_m"], d_db=gi["doubleBottom_m"],
        d_hgir=1.0, h_camber=gi["camberUpper_m"],
        y_lbhd=gi.get("lbhd_ratio", 0.0) * (B / 2.0),
        y_1gir=gi.get("girder1_ratio", 0.4) * (B / 2.0),
        y_2gir=gi.get("girder2_ratio", 0.7) * (B / 2.0),
        z_3str=gi.get("str3_ratio", 0.3) * D,
        z_2str=gi.get("str2_ratio", 0.5) * D,
        z_1str=gi.get("str1_ratio", 0.7) * D,
        r_bilge=gi["bilgeRadius_m"],
    )
    return mod, ship


def _build_vlcc(gi: dict):
    mod = _get_module("VLCC")
    B, D = gi["B_m"], gi["D_m"]
    ship = mod.VLCC(
        L=gi["L_m"], B=B, D=D,
        d_ds=gi["doubleSide_m"], d_db=gi["doubleBottom_m"],
        d_hgir=1.5, h_camber=gi["camberUpper_m"],
        y_lbhd=gi.get("lbhd_ratio", 0.4) * (B / 2.0),
        y_1gir=gi.get("girder1_ratio", 0.4) * (B / 2.0),
        y_2gir=gi.get("girder2_ratio", 0.7) * (B / 2.0),
        z_3str=gi.get("str3_ratio", 0.3) * D,
        z_2str=gi.get("str2_ratio", 0.5) * D,
        z_1str=gi.get("str1_ratio", 0.7) * D,
        r_bilge=gi["bilgeRadius_m"],
    )
    return mod, ship


def _build_bulkc(gi: dict):
    mod = _get_module("BULKC")
    ship = mod.BULKC(
        L=gi["L_m"], B=gi["B_m"], D=gi["D_m"],
        DB=gi["doubleBottom_m"], R=gi["bilgeRadius_m"],
        camber=gi["camberUpper_m"],
        y_girder=gi.get("girder_y_m", 2.2),
        y_og_ratio=gi.get("outgir_ratio", 0.75),
        tswt_ext_deg=gi.get("tswt_ext_deg", 120.0),
        ds_from_side=gi.get("ds_from_side_m", 0.65),
        str_clear=gi.get("strClearance_m", 0.3),
        s1_ratio=gi.get("str1_ratio", 0.75),
        s2_ratio=gi.get("str2_ratio", 0.5),
    )
    return mod, ship


def _build_cntr(gi: dict):
    mod = _get_module("CNTR")
    B, D = gi["B_m"], gi["D_m"]
    ship = mod.CNTR(
        L=gi["L_m"], B=B, D=D,
        d_ds=gi["doubleSide_m"], d_db=gi["doubleBottom_m"],
        h_camber=gi["camberUpper_m"],
        y_1gir=gi.get("girder1_ratio", 0.15) * (B / 2.0),
        y_2gir=gi.get("girder2_ratio", 0.8) * (B / 2.0),
        z_3str=gi.get("str3_ratio", 0.18) * D,
        z_2str=gi.get("str2_ratio", 0.5) * D,
        z_1str=gi.get("str1_ratio", 0.7) * D,
        r_bilge=gi["bilgeRadius_m"],
    )
    return mod, ship


def _build_lngc(gi: dict):
    mod = _get_module("LNGC")
    B, D = gi["B_m"], gi["D_m"]
    ship = mod.LNGC(
        L=gi["L_m"], B=B, D=D,
        d_ds=gi["doubleSide_m"], d_db=gi["doubleBottom_m"],
        h_camber=gi["camberUpper_m"],
        y_0gir=gi.get("girder0_ratio", 0.0) * (B / 2.0),
        y_1gir=gi.get("girder1_ratio", 0.25) * (B / 2.0),
        y_2gir=gi.get("girder2_ratio", 0.6) * (B / 2.0),
        z_3str=gi.get("str3_ratio", 0.35) * D,
        z_2str=gi.get("str2_ratio", 0.6) * D,
        z_1str=gi.get("str1_ratio", 0.85) * D,
        r_bilge=gi["bilgeRadius_m"],
        h_camber_trunk=gi.get("camberTrunk_m", 0.5),
    )
    return mod, ship


def _build_lpgc(gi: dict):
    mod = _get_module("LPGC")
    ship = mod.LPGC(
        L=gi["L_m"], B=gi["B_m"], D=gi["D_m"],
        DB=gi["doubleBottom_m"], R=gi["bilgeRadius_m"],
        camber=gi["camberUpper_m"],
        y_girder=gi.get("girder_y_m", 1.7),
        y_og_ratio=gi.get("outgir_ratio", 0.75),
        tswt_ext_deg=gi.get("tswt_ext_deg", 120.0),
        gap_tswt=gi.get("gap_tswt_m", 0.7),
        gap_hopper=gi.get("gap_hopper_m", 0.7),
        str_clear=gi.get("strClearance_m", 0.3),
        s1_ratio=gi.get("str1_ratio", 0.75),
        s2_ratio=gi.get("str2_ratio", 0.5),
    )
    return mod, ship


BUILDERS = {
    "Tanker": _build_tanker,
    "VLCC":   _build_vlcc,
    "BULKC":  _build_bulkc,
    "CNTR":   _build_cntr,
    "LNGC":   _build_lngc,
    "LPGC":   _build_lpgc,
}

# Exporter class names per ship (some use DXFExporter, some DXFExporterMM)
EXPORTER_NAMES = {
    "Tanker": "DXFExporterMM",
    "VLCC":   "DXFExporterMM",
    "BULKC":  "DXFExporter",
    "CNTR":   "DXFExporterMM",
    "LNGC":   "DXFExporterMM",
    "LPGC":   "DXFExporter",
}


def render_section_png(
    ship_type: str,
    candidate: dict,
    png_out_dir: Path,
    dpi: int = 220,
) -> str | None:
    """Render a section PNG for a candidate. Returns PNG path or None on failure."""
    gi = candidate["generator_inputs"]
    cid = candidate["candidate_id"]

    try:
        mod, ship = BUILDERS[ship_type](gi)
    except Exception as e:
        print(f"  WARN: {cid} build failed: {e}")
        return None

    ExporterClass = getattr(mod, EXPORTER_NAMES[ship_type])

    # Create exporter with minimal params
    kwargs = {"ship": ship, "text_height": 250, "offset": 300}

    # Some exporters need additional args
    HL = gi.get("HL_m", 30.0)
    n_hold = gi.get("number_of_hold", 5)
    try:
        exp = ExporterClass(
            ship,
            text_height=250, offset=300,
            hold_length_m=HL,
            number_of_hold=n_hold,
        )
    except TypeError:
        # Fallback: try without hold params
        try:
            exp = ExporterClass(ship, text_height=250, offset=300)
        except Exception as e:
            print(f"  WARN: {cid} exporter init failed: {e}")
            return None

    # Export to temp DXF, then produce PNG
    png_out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".dxf", dir=str(png_out_dir), delete=False) as tmp:
        dxf_path = tmp.name

    final_png = str(png_out_dir / f"{cid}.png")

    # Detect export signature: some exporters use png_out_dir (directory),
    # VLCC uses png_path (file path)
    import inspect
    sig = inspect.signature(exp.export)
    try:
        if "png_out_dir" in sig.parameters:
            result = exp.export(
                save_as=dxf_path,
                png_out_dir=str(png_out_dir),
                png_dpi=dpi,
            )
        elif "png_path" in sig.parameters:
            result = exp.export(
                save_as=dxf_path,
                png_path=final_png,
                png_dpi=dpi,
            )
        else:
            result = exp.export(save_as=dxf_path)
        # Handle both 2-tuple (qc, png_path) and 3-tuple (qc, png_path, stats)
        if len(result) == 3:
            qc, png_path, stats = result
        elif len(result) == 2:
            qc, png_path = result
        else:
            qc, png_path = result[0], None
    except Exception as e:
        print(f"  WARN: {cid} export failed: {e}")
        if os.path.exists(dxf_path):
            os.unlink(dxf_path)
        return None

    # Rename PNG to candidate ID if needed
    if png_path and os.path.exists(png_path) and png_path != final_png:
        os.replace(png_path, final_png)
    elif not png_path or not os.path.exists(final_png):
        final_png = None

    # Clean up temp DXF
    if os.path.exists(dxf_path):
        os.unlink(dxf_path)

    return final_png


def main():
    parser = argparse.ArgumentParser(description="Render section PNGs for stratified candidates")
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit per ship (0 = all)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="PNG DPI (lower = faster, default 150)")
    args = parser.parse_args()

    ships = args.ships or SHIPS
    t0 = time.time()

    for ship in ships:
        ship_dir = STRATIFIED_DIR / ship
        if not ship_dir.exists():
            print(f"SKIP {ship}: no stratified dir")
            continue

        # Collect all candidate JSONs across strata
        json_files = []
        for stratum_dir in sorted(ship_dir.iterdir()):
            if not stratum_dir.is_dir():
                continue
            for jf in sorted(stratum_dir.glob(f"{ship}-*.json")):
                json_files.append((stratum_dir, jf))

        if args.limit:
            json_files = json_files[:args.limit]

        png_dir = STRATIFIED_DIR / ship / "section_png"
        print(f"=== {ship}: {len(json_files)} candidates → {png_dir} ===")

        ok = 0
        fail = 0
        for i, (stratum_dir, jf) in enumerate(json_files):
            candidate = json.load(open(jf))
            png = render_section_png(ship, candidate, png_dir, dpi=args.dpi)
            if png:
                ok += 1
            else:
                fail += 1
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  [{ship}] {i+1}/{len(json_files)} rendered "
                      f"({ok} ok, {fail} fail, {elapsed:.0f}s)")

        print(f"  {ship}: {ok} rendered, {fail} failed ({time.time()-t0:.0f}s total)")

    print(f"\nDone in {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
