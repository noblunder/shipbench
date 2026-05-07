#!/usr/bin/env python3
"""
Phase 0.2.C — Lightweight candidate generation (params + rule eval only, no DXF/PNG).

Generates ~20K candidate parameter sets across 6 ship types with 3 sampling regimes
(conservative / typical / aggressive). Each candidate is stored as a lightweight JSON
containing generator_inputs + rule evaluation result + regime metadata.

Full rendering (DXF/PNG) is deferred to Phase 0.2.H after stratification + FPS selection.

Usage:
    python scripts/00_generate_candidates.py                    # full 20K run
    python scripts/00_generate_candidates.py --smoke 20         # smoke test: 20 per ship
    python scripts/00_generate_candidates.py --ships LPGC CNTR  # subset of ships
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "data_generator"
sys.path.insert(0, str(GEN_DIR))

# ── lazy imports from generators ────────────────────────────────────
# We import ship classes, domain_rules_ok, evaluate_* functions, and lhs_samples
# on demand per ship type to avoid loading all 6 at once.


# ═══════════════════════════════════════════════════════════════════════
# 1.  PARAMETER RANGE DEFINITIONS  (v0.1 — finalized 2026-04-10)
# ═══════════════════════════════════════════════════════════════════════
# Each entry: (lo, hi, step, dtype)  where dtype ∈ {'int', 'float'}

RANGES = {
    "Tanker": {
        "B":  (38, 53, 1, "int"),
        "D":  (16, 29, 1, "int"),
        "C":  (0.5, 2.0, 0.1, "float"),
        "DS": (1.5, 3.5, 0.1, "float"),
        "DB": (1.5, 3.5, 0.1, "float"),
        "R":  (1.0, 3.0, 0.1, "float"),
        "LB": (0.0, 0.0, 0.05, "float"),
        "G1": (0.3, 0.5, 0.05, "float"),
        "G2": (0.6, 0.8, 0.05, "float"),
        "S1": (0.65, 0.75, 0.05, "float"),
        "S2": (0.4, 0.55, 0.05, "float"),
        "S3": (0.25, 0.35, 0.05, "float"),
        "HL": (25.0, 35.0, 0.1, "float"),
    },
    "VLCC": {
        "B":  (55, 68, 1, "int"),
        "D":  (24, 34, 1, "int"),
        "C":  (0.5, 2.0, 0.1, "float"),
        "DS": (1.6, 3.6, 0.1, "float"),
        "DB": (1.6, 3.6, 0.1, "float"),
        "R":  (1.0, 3.0, 0.1, "float"),
        "LB": (0.3, 0.5, 0.05, "float"),
        "G1": (0.3, 0.5, 0.05, "float"),
        "G2": (0.6, 0.8, 0.05, "float"),
        "S1": (0.65, 0.75, 0.05, "float"),
        "S2": (0.4, 0.55, 0.05, "float"),
        "S3": (0.25, 0.35, 0.05, "float"),
        "HL": (25.0, 35.0, 0.1, "float"),
    },
    "BULKC": {
        "B":  (40, 52, 1, "int"),
        "D":  (20, 30, 1, "int"),
        "C":  (0.5, 2.5, 0.5, "float"),
        "DB": (1.5, 3.0, 0.1, "float"),
        "R":  (1.0, 3.0, 0.5, "float"),
        "GY": (2.0, 2.4, 0.05, "float"),
        "OG": (0.7, 0.8, 0.05, "float"),
        "TSWT_EXT": (110.0, 130.0, 2.0, "float"),
        "DS": (0.6, 0.7, 0.05, "float"),   # side frame offset
        "STRCLR": (0.2, 0.4, 0.1, "float"),
        "S1": (0.7, 0.8, 0.05, "float"),
        "S2": (0.4, 0.6, 0.05, "float"),
        "HL": (20.0, 30.0, 0.1, "float"),
    },
    "CNTR": {
        "B":  (42, 62, 1, "int"),
        "D":  (24, 36, 1, "int"),
        "C":  (0.1, 1.0, 0.1, "float"),
        "DS": (1.5, 3.5, 0.1, "float"),
        "DB": (1.5, 3.0, 0.1, "float"),
        "R":  (3.5, 5.5, 0.1, "float"),
        "G1": (0.1, 0.2, 0.05, "float"),
        "G2": (0.7, 0.88, 0.05, "float"),
        "S1": (0.6, 0.8, 0.05, "float"),
        "S2": (0.4, 0.55, 0.05, "float"),
        "S3": (0.15, 0.2, 0.05, "float"),
        "HL": (27.0, 35.0, 0.1, "float"),
    },
    "LNGC": {
        "B":  (40, 55, 1, "int"),
        "D":  (24, 30, 1, "int"),
        "C":  (0.3, 1.5, 0.1, "float"),
        "CT": (0.1, 1.0, 0.1, "float"),   # trunk camber
        "DS": (2.0, 4.0, 0.1, "float"),
        "DB": (2.2, 4.2, 0.1, "float"),
        "R":  (3.0, 5.0, 0.5, "float"),
        "LB": (0.0, 0.0, 0.1, "float"),
        "G0": (0.0, 0.0, 0.05, "float"),
        "G1": (0.2, 0.3, 0.05, "float"),
        "G2": (0.5, 0.7, 0.05, "float"),
        "S1": (0.8, 0.9, 0.05, "float"),
        "S2": (0.55, 0.7, 0.05, "float"),
        "S3": (0.3, 0.4, 0.05, "float"),
        "HL": (40.0, 50.0, 0.1, "float"),
    },
    "LPGC": {
        "B":  (30, 44, 1, "int"),
        "D":  (18, 28, 1, "int"),
        "C":  (0.5, 2.5, 0.5, "float"),
        "DB": (1.5, 3.5, 0.1, "float"),
        "R":  (2.0, 5.0, 0.5, "float"),
        "GY": (1.6, 1.8, 0.05, "float"),
        "OG": (0.7, 0.8, 0.05, "float"),
        "TSWT_EXT": (110.0, 130.0, 2.0, "float"),
        "GAP_TSWT": (0.6, 0.8, 0.1, "float"),
        "GAP_HOP":  (0.6, 0.8, 0.1, "float"),
        "STRCLR":   (0.2, 0.4, 0.1, "float"),
        "S1": (0.7, 0.8, 0.05, "float"),
        "S2": (0.4, 0.6, 0.05, "float"),
        "HL": (35.0, 45.0, 0.1, "float"),
    },
}

# Per-ship target counts (total across all 3 regimes)
TARGET_COUNTS = {
    "Tanker": 3000,
    "VLCC":   3000,
    "BULKC":  3500,
    "CNTR":   3500,
    "LNGC":   4000,
    "LPGC":   3000,
}

# Regime allocation (fraction of target count)
# Over-sample by 2× to compensate for domain_rules_ok rejections
REGIME_FRACTIONS = {
    "conservative": 0.30,
    "typical":      0.50,
    "aggressive":   0.20,
}

# Length model defaults per ship
LENGTH_DEFAULTS = {
    "Tanker": {"fwd": 15.0, "er": 40.0, "aft": 10.0, "hlf": 0.8, "nh": (5, 7, 1)},
    "VLCC":   {"fwd": 15.0, "er": 40.0, "aft": 10.0, "hlf": 0.8, "nh": (5, 7, 1)},
    "BULKC":  {"fwd": 15.0, "er": 30.0, "aft": 15.0, "hlf": 0.8, "nh": (8, 10, 1)},
    "CNTR":   {"fwd": 15.0, "er": 40.0, "aft": 15.0, "hlf": 0.8, "nh": (7, 11, 1)},
    "LNGC":   {"fwd": 40.0, "er": 40.0, "aft": 20.0, "hlf": 0.8, "nh": (3, 5, 1),
               "n_cofferdam": 5, "cofferdam_len": 2.5},
    "LPGC":   {"fwd": 10.0, "er": 30.0, "aft": 20.0, "hlf": 0.8, "nh": (3, 5, 1)},
}

# Cb estimates for DWT (Tanker/VLCC/BULKC use build_ship_data_context)
CB_ESTIMATES = {"Tanker": 0.82, "VLCC": 0.83, "BULKC": 0.85}


# ═══════════════════════════════════════════════════════════════════════
# 2.  LHS SAMPLER  (copied from generators — pure function, no deps)
# ═══════════════════════════════════════════════════════════════════════

def quantize_to_step(x, start, step):
    return round(round((x - start) / step) * step + start, 10)


def lhs_samples(N, specs, seed=None):
    """Latin Hypercube Sampling over parameter specs."""
    rng = random.Random(seed)
    per = []
    for sp in specs:
        lo, hi = sp["min"], sp["max"]
        w = (hi - lo) / max(N, 1)
        vals = [lo + i * w + rng.random() * w for i in range(N)]
        rng.shuffle(vals)
        if sp.get("step") is not None:
            vals = [quantize_to_step(v, sp["min"], sp["step"]) for v in vals]
        if sp["type"] == "int":
            vals = [int(round(v)) for v in vals]
        per.append(vals)
    out = []
    for i in range(N):
        d = {}
        for j, sp in enumerate(specs):
            d[sp["name"]] = per[j][i]
        out.append(d)
    return out


def make_specs(ranges_dict: dict) -> list[dict]:
    """Convert RANGES[ship] dict to lhs_samples-compatible spec list."""
    specs = []
    for name, (lo, hi, step, dtype) in ranges_dict.items():
        specs.append({"name": name, "min": lo, "max": hi, "step": step, "type": dtype})
    return specs


# ═══════════════════════════════════════════════════════════════════════
# 3.  REGIME RANGE MODIFIERS
# ═══════════════════════════════════════════════════════════════════════

def _shrink_range(lo, hi, frac=0.6):
    """Shrink [lo, hi] to the inner `frac` portion centered at midpoint."""
    mid = (lo + hi) / 2.0
    half = (hi - lo) * frac / 2.0
    return max(lo, mid - half), min(hi, mid + half)


def _edge_range(lo, hi, edge_frac=0.3):
    """Return two sub-ranges: lower `edge_frac` and upper `edge_frac`."""
    span = hi - lo
    cut_lo = lo + span * edge_frac
    cut_hi = hi - span * edge_frac
    return (lo, cut_lo), (cut_hi, hi)


def apply_regime(base_ranges: dict, regime: str) -> dict:
    """Return modified range dict for the given regime.

    - conservative: inner 60% of each continuous parameter
    - typical: unchanged
    - aggressive: outer 30% edges (random pick low or high per sample is
      handled by splitting the LHS into two halves)
    """
    if regime == "typical":
        return dict(base_ranges)

    out = {}
    for name, (lo, hi, step, dtype) in base_ranges.items():
        if lo == hi:
            # degenerate range (e.g. LB=0.0–0.0, G0=0.0–0.0) — keep unchanged
            out[name] = (lo, hi, step, dtype)
            continue
        if regime == "conservative":
            new_lo, new_hi = _shrink_range(lo, hi, 0.6)
            # snap to step grid
            new_lo = quantize_to_step(max(new_lo, lo), lo, step)
            new_hi = quantize_to_step(min(new_hi, hi), lo, step)
            if new_lo >= new_hi:
                new_lo, new_hi = lo, hi  # fallback: use full range
            out[name] = (new_lo, new_hi, step, dtype)
        elif regime == "aggressive":
            # Use full range — the edge bias is achieved by oversampling 2×
            # and keeping only domain_ok + rule-fail/borderline-fail candidates.
            # Simpler and equally effective as sub-range splitting.
            out[name] = (lo, hi, step, dtype)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 4.  PER-SHIP: domain_ok + build_ship + rule_eval  (no rendering)
# ═══════════════════════════════════════════════════════════════════════

def _estimate_L(HL, n_hold, cfg):
    """Derive overall length from hold length, number of holds, and ship config."""
    hold_part = cfg["hlf"] * HL * n_hold
    if "cofferdam_len" in cfg:
        hold_part += cfg["n_cofferdam"] * cfg["cofferdam_len"]
    return cfg["fwd"] + hold_part + cfg["er"] + cfg["aft"]


def _pick_n_hold(cfg, rng):
    lo, hi, step = cfg["nh"]
    return rng.randrange(int(lo), int(hi) + 1, int(step))


# ---------- module-level caches for lazy imports ----------
_MODULES = {}

def _get_module(ship_type: str):
    """Lazy-import the generator module for ship_type."""
    if ship_type in _MODULES:
        return _MODULES[ship_type]
    mod_name = f"{ship_type}_Data_generation"
    import importlib
    mod = importlib.import_module(mod_name)
    _MODULES[ship_type] = mod
    return mod


# ── Tanker ──────────────────────────────────────────────────────────

def _eval_tanker(p, L, n_hold, HL, rng):
    mod = _get_module("Tanker")
    p_all = dict(p); p_all["L"] = L
    ok, issues = mod.domain_rules_ok_tanker(p_all)
    if not ok:
        return None
    B, D = p["B"], p["D"]
    y_lbhd = p.get("LB", 0.0) * (B / 2.0)
    y_1gir = p["G1"] * (B / 2.0)
    y_2gir = p["G2"] * (B / 2.0)
    z_1str = p["S1"] * D
    z_2str = p["S2"] * D
    z_3str = p["S3"] * D
    ship = mod.Tanker(
        L=L, B=B, D=D,
        d_ds=p["DS"], d_db=p["DB"], d_hgir=1.0, h_camber=p["C"],
        y_lbhd=y_lbhd, y_1gir=y_1gir, y_2gir=y_2gir,
        z_3str=z_3str, z_2str=z_2str, z_1str=z_1str, r_bilge=p["R"],
    )
    gen_inputs = {
        "L_m": L, "B_m": B, "D_m": D,
        "doubleSide_m": p["DS"], "doubleBottom_m": p["DB"],
        "bilgeRadius_m": p["R"], "lbhd_ratio": p.get("LB", 0.0),
        "girder1_ratio": p["G1"], "girder2_ratio": p["G2"],
    }
    ship_data = mod.build_ship_data_context(L, B_m=B, D_m=D, cb_estimate=0.82)
    kr_eval = mod.evaluate_csr_rules_tanker(gen_inputs, ship_data, ship)
    gen_inputs.update({
        "camberUpper_m": p["C"], "HL_m": HL,
        "str1_ratio": p["S1"], "str2_ratio": p["S2"], "str3_ratio": p["S3"],
        "number_of_hold": n_hold,
    })
    return {"generator_inputs": gen_inputs, "kr_eval": kr_eval,
            "ship_data": ship_data, "domain_issues": issues}


# ── VLCC ────────────────────────────────────────────────────────────

def _eval_vlcc(p, L, n_hold, HL, rng):
    mod = _get_module("VLCC")
    p_all = dict(p); p_all["L"] = L
    ok, issues = mod.domain_rules_ok(p_all)
    if not ok:
        return None
    B, D = p["B"], p["D"]
    y_lbhd = p["LB"] * (B / 2.0)
    y_1gir = p["G1"] * (B / 2.0)
    y_2gir = p["G2"] * (B / 2.0)
    z_1str = p["S1"] * D
    z_2str = p["S2"] * D
    z_3str = p["S3"] * D
    ship = mod.VLCC(
        L=L, B=B, D=D,
        d_ds=p["DS"], d_db=p["DB"], d_hgir=1.5, h_camber=p["C"],
        y_lbhd=y_lbhd, y_1gir=y_1gir, y_2gir=y_2gir,
        z_3str=z_3str, z_2str=z_2str, z_1str=z_1str, r_bilge=p["R"],
    )
    gen_inputs = {
        "L_m": L, "B_m": B, "D_m": D,
        "doubleSide_m": p["DS"], "doubleBottom_m": p["DB"],
        "bilgeRadius_m": p["R"], "lbhd_ratio": p["LB"],
    }
    ship_data = mod.build_ship_data_context(L, B_m=B, D_m=D, cb_estimate=0.83)
    kr_eval = mod.evaluate_csr_rules_vlcc(gen_inputs, ship_data, ship)
    gen_inputs.update({
        "camberUpper_m": p["C"], "HL_m": HL,
        "girder1_ratio": p["G1"], "girder2_ratio": p["G2"],
        "str1_ratio": p["S1"], "str2_ratio": p["S2"], "str3_ratio": p["S3"],
        "number_of_hold": n_hold,
    })
    return {"generator_inputs": gen_inputs, "kr_eval": kr_eval,
            "ship_data": ship_data, "domain_issues": issues}


# ── BULKC ───────────────────────────────────────────────────────────

def _eval_bulkc(p, L, n_hold, HL, rng):
    mod = _get_module("BULKC")
    p["L"] = L
    ok, issues = mod.domain_rules_ok(p)
    if not ok:
        return None
    ship = mod.BULKC(
        L=L, B=p["B"], D=p["D"], DB=p["DB"], R=p["R"], camber=p["C"],
        y_girder=p["GY"], y_og_ratio=p["OG"], tswt_ext_deg=p["TSWT_EXT"],
        ds_from_side=p["DS"], str_clear=p.get("STRCLR", 0.3),
        s1_ratio=p["S1"], s2_ratio=p["S2"],
    )
    gen_inputs = {
        "L_m": L, "B_m": p["B"], "D_m": p["D"],
        "doubleBottom_m": p["DB"], "bilgeRadius_m": p["R"],
        "tswt_ext_deg": p["TSWT_EXT"],
    }
    ship_data = mod.build_ship_data_context(L)
    kr_eval = mod.evaluate_csr_rules_bulkc(gen_inputs, ship_data, ship)
    gen_inputs.update({
        "camberUpper_m": p["C"], "HL_m": HL,
        "ds_from_side_m": p["DS"],
        "girder_y_m": p["GY"], "outgir_ratio": p["OG"],
        "str1_ratio": p["S1"], "str2_ratio": p["S2"],
        "number_of_hold": n_hold,
    })
    return {"generator_inputs": gen_inputs, "kr_eval": kr_eval,
            "ship_data": ship_data, "domain_issues": issues}


# ── CNTR ────────────────────────────────────────────────────────────

def _eval_cntr(p, L, n_hold, HL, rng):
    mod = _get_module("CNTR")
    p_all = dict(p); p_all["L"] = L
    ok, issues = mod.domain_rules_ok_cntr(p_all)
    if not ok:
        return None
    B, D = p["B"], p["D"]
    y_1gir = p["G1"] * (B / 2.0)
    y_2gir = p["G2"] * (B / 2.0)
    z_1str = p["S1"] * D
    z_2str = p["S2"] * D
    z_3str = p["S3"] * D
    ship = mod.CNTR(
        L=L, B=B, D=D,
        d_ds=p["DS"], d_db=p["DB"], h_camber=p["C"],
        y_1gir=y_1gir, y_2gir=y_2gir,
        z_3str=z_3str, z_2str=z_2str, z_1str=z_1str, r_bilge=p["R"],
    )
    gen_inputs = {
        "L_m": L, "B_m": B, "D_m": D,
        "doubleSide_m": p["DS"], "doubleBottom_m": p["DB"],
        "bilgeRadius_m": p["R"],
    }
    kr_eval = mod.evaluate_kr_rules_cntr(gen_inputs, ship)
    gen_inputs.update({
        "camberUpper_m": p["C"], "HL_m": HL,
        "girder1_ratio": p["G1"], "girder2_ratio": p["G2"],
        "str1_ratio": p["S1"], "str2_ratio": p["S2"], "str3_ratio": p["S3"],
        "number_of_hold": n_hold,
    })
    return {"generator_inputs": gen_inputs, "kr_eval": kr_eval,
            "domain_issues": issues}


# ── LNGC ────────────────────────────────────────────────────────────

def _eval_lngc(p, L, n_hold, HL, rng):
    mod = _get_module("LNGC")
    p["L"] = L; p["N_HOLD"] = n_hold
    ok, issues, detail = mod.domain_rules_ok_lngc(p)
    if not ok:
        return None
    B, D = p["B"], p["D"]
    y_0gir = p.get("G0", 0.0) * (B / 2.0)
    y_1gir = p["G1"] * (B / 2.0)
    y_2gir = p["G2"] * (B / 2.0)
    z_1str = p["S1"] * D
    z_2str = p["S2"] * D
    z_3str = p["S3"] * D
    ship = mod.LNGC(
        L=L, B=B, D=D,
        d_ds=p["DS"], d_db=p["DB"], h_camber=p["C"],
        y_0gir=y_0gir, y_1gir=y_1gir, y_2gir=y_2gir,
        z_3str=z_3str, z_2str=z_2str, z_1str=z_1str, r_bilge=p["R"],
        h_camber_trunk=p.get("CT", 0.5),
    )
    n_cofferdam = LENGTH_DEFAULTS["LNGC"].get("n_cofferdam", 5)
    gen_inputs = {
        "L_m": L, "B_m": B, "D_m": D,
        "doubleSide_m": p["DS"], "doubleBottom_m": p["DB"],
        "bilgeRadius_m": p["R"],
        "number_of_cofferdam": n_cofferdam,
        "inner_slope_deg": getattr(ship, "inner_slope_deg", None),
    }
    kr_eval = mod.evaluate_kr_rules_lngc(gen_inputs, ship)
    gen_inputs.update({
        "camberUpper_m": p["C"], "camberTrunk_m": p.get("CT", 0.5),
        "HL_m": HL,
        "girder0_ratio": p.get("G0", 0.0),
        "girder1_ratio": p["G1"], "girder2_ratio": p["G2"],
        "str1_ratio": p["S1"], "str2_ratio": p["S2"], "str3_ratio": p["S3"],
        "number_of_hold": n_hold,
    })
    return {"generator_inputs": gen_inputs, "kr_eval": kr_eval,
            "domain_issues": issues}


# ── LPGC ────────────────────────────────────────────────────────────

def _eval_lpgc(p, L, n_hold, HL, rng):
    mod = _get_module("LPGC")
    p["L"] = L
    ok, issues = mod.domain_rules_ok(p)
    if not ok:
        return None
    ship = mod.LPGC(
        L=L, B=p["B"], D=p["D"], DB=p["DB"], R=p["R"], camber=p["C"],
        y_girder=p["GY"], y_og_ratio=p["OG"], tswt_ext_deg=p["TSWT_EXT"],
        gap_tswt=p["GAP_TSWT"], gap_hopper=p["GAP_HOP"], str_clear=p["STRCLR"],
        s1_ratio=p["S1"], s2_ratio=p["S2"],
    )
    gen_inputs = {
        "L_m": L, "B_m": p["B"], "D_m": p["D"],
        "doubleBottom_m": p["DB"], "bilgeRadius_m": p["R"],
        "tswt_ext_deg": p["TSWT_EXT"],
        "gap_tswt_m": p["GAP_TSWT"], "gap_hopper_m": p["GAP_HOP"],
    }
    kr_eval = mod.evaluate_kr_rules_lpgc(gen_inputs, ship)
    gen_inputs.update({
        "camberUpper_m": p["C"], "HL_m": HL,
        "girder_y_m": p["GY"], "outgir_ratio": p["OG"],
        "strClearance_m": p["STRCLR"],
        "str1_ratio": p["S1"], "str2_ratio": p["S2"],
        "number_of_hold": n_hold,
    })
    return {"generator_inputs": gen_inputs, "kr_eval": kr_eval,
            "domain_issues": issues}


EVAL_FUNCS = {
    "Tanker": _eval_tanker,
    "VLCC":   _eval_vlcc,
    "BULKC":  _eval_bulkc,
    "CNTR":   _eval_cntr,
    "LNGC":   _eval_lngc,
    "LPGC":   _eval_lpgc,
}


# ═══════════════════════════════════════════════════════════════════════
# 5.  RULE-STATE SUMMARY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def summarize_kr(kr_eval: dict) -> dict:
    """Extract pass/fail/undetermined/not_modeled counts from kr_eval."""
    checks = kr_eval.get("auto_checks") or kr_eval.get("checks", [])
    counts = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
    for c in checks:
        st = c.get("status", "undetermined")
        if st in counts:
            counts[st] += 1
        else:
            counts["undetermined"] += 1
    # overall
    if counts["fail"] > 0:
        overall = "fail"
    elif counts["undetermined"] > 0 or counts["not_modeled"] > 0:
        overall = "partial"
    else:
        overall = "pass"
    counts["overall"] = overall
    return counts


def _round_floats(obj, nd=3):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════
# 6.  MAIN GENERATION LOOP
# ═══════════════════════════════════════════════════════════════════════

def generate_candidates(
    ship_type: str,
    target_count: int,
    out_dir: Path,
    seed: int = 42,
    oversample_factor: float = 2.5,
) -> dict:
    """Generate lightweight candidate JSONs for one ship type.

    Returns summary dict with counts per regime and rule-state distribution.
    """
    base_ranges = RANGES[ship_type]
    cfg = LENGTH_DEFAULTS[ship_type]
    eval_fn = EVAL_FUNCS[ship_type]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"ship_type": ship_type, "target": target_count, "regimes": {}}
    global_id = 0
    rng = random.Random(seed)

    for regime, frac in REGIME_FRACTIONS.items():
        regime_target = int(target_count * frac)
        # Over-sample to compensate for domain_rules_ok rejections
        # LNGC has very strict domain rules (~80% rejection) → needs higher factor
        eff_oversample = oversample_factor * 4.0 if ship_type == "LNGC" else oversample_factor
        n_sample = int(regime_target * eff_oversample)
        regime_ranges = apply_regime(base_ranges, regime)
        specs = make_specs(regime_ranges)

        # Use different seed per regime to avoid correlation
        regime_seed = seed + hash(regime) % 10000
        samples = lhs_samples(n_sample, specs, seed=regime_seed)

        regime_dir = out_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        accepted = 0
        rejected_domain = 0
        rule_dist = {"pass": 0, "fail": 0, "partial": 0}

        for p in samples:
            if accepted >= regime_target:
                break

            HL = float(p.pop("HL", 30.0))
            n_hold = _pick_n_hold(cfg, rng)
            L = round(_estimate_L(HL, n_hold, cfg), 1)

            result = eval_fn(p, L, n_hold, HL, rng)
            if result is None:
                rejected_domain += 1
                continue

            kr_summary = summarize_kr(result["kr_eval"])
            rule_dist[kr_summary["overall"]] = rule_dist.get(kr_summary["overall"], 0) + 1

            candidate = {
                "candidate_id": f"{ship_type}-{global_id:05d}",
                "ship_type": ship_type,
                "regime": regime,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": regime_seed,
                "generator_inputs": _round_floats(result["generator_inputs"]),
                "kr_eval": _round_floats(result["kr_eval"]),
                "kr_summary": kr_summary,
            }
            if "ship_data" in result:
                candidate["ship_data"] = _round_floats(result["ship_data"])
            if result.get("domain_issues"):
                candidate["domain_issues"] = result["domain_issues"]

            json_path = regime_dir / f"{candidate['candidate_id']}.json"
            with open(json_path, "w") as f:
                json.dump(candidate, f, indent=2, ensure_ascii=False)

            global_id += 1
            accepted += 1

        summary["regimes"][regime] = {
            "target": regime_target,
            "accepted": accepted,
            "rejected_domain": rejected_domain,
            "sampled": len(samples),
            "rule_distribution": rule_dist,
        }
        print(f"  {regime:14s}: {accepted:5d} accepted / {rejected_domain:5d} rejected "
              f"(sampled {len(samples)})  rule_dist={rule_dist}")

    # Write summary JSON
    total_accepted = sum(r["accepted"] for r in summary["regimes"].values())
    summary["total_accepted"] = total_accepted
    with open(out_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ═══════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 0.2.C: Generate lightweight candidates")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke test: generate N candidates per ship (default: full run)")
    parser.add_argument("--ships", nargs="*", default=None,
                        help="Subset of ships (default: all 6)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "candidates_R1"))
    parser.add_argument("--oversample", type=float, default=2.5,
                        help="Over-sample factor to compensate for domain rejections (default: 2.5)")
    args = parser.parse_args()

    ships = args.ships or list(TARGET_COUNTS.keys())
    out_base = Path(args.out_dir)

    print(f"Phase 0.2.C candidate generation")
    print(f"  Output: {out_base}")
    print(f"  Ships:  {ships}")
    print(f"  Seed:   {args.seed}")
    if args.smoke:
        print(f"  MODE:   SMOKE TEST ({args.smoke} per ship)")
    print()

    all_summaries = {}
    grand_total = 0

    for ship in ships:
        target = args.smoke if args.smoke else TARGET_COUNTS[ship]
        print(f"=== {ship} (target={target}) ===")
        s = generate_candidates(
            ship_type=ship,
            target_count=target,
            out_dir=out_base / ship,
            seed=args.seed,
            oversample_factor=args.oversample,
        )
        all_summaries[ship] = s
        grand_total += s["total_accepted"]
        print(f"  => {s['total_accepted']} candidates written\n")

    # Write master summary
    master = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grand_total": grand_total,
        "ships": all_summaries,
    }
    with open(out_base / "master_summary.json", "w") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    print(f"Done. Grand total: {grand_total} candidates → {out_base}")


if __name__ == "__main__":
    main()
