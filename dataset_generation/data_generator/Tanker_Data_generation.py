# =========================================
#   TANKER Midship Generator (LNGC-style meta, HL-based capacity)
#   - Drawing visuals kept as-is
#   - JSON/CSV/filename aligned to LNGC style (+K prefix rename)
#   - Hold Length (HL) sampled; capacity uses HL
#   - Estimated ship length (L) can use HL model (configurable)
# =========================================

import os, csv, json, time, random
from math import sin, cos, atan2, pi, hypot, degrees

import ezdxf

# ===============================
# PNG 렌더링용 (옵셔널)
# ===============================

try:
    import matplotlib
    matplotlib.use("Agg")  # GUI 없이 렌더
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MATPLOT_OK = True
except Exception:
    _MATPLOT_OK = False
    plt = None

# ================================
# CSR Standard Info — IACS CSR-H 2024
# ================================
CSR_STANDARD_INFO = {
    "title": "Common Structural Rules for Bulk Carriers and Oil Tankers",
    "edition": "01 JAN 2024",
    "short_name": "IACS CSR-H",
    "effective_from": "2024-07-01",
    "source_file": "CSR-H-01-JAN-2024.pdf",
}

# ================================
# CSR Rule Registry — Oil Tanker (Double Hull)
# IACS CSR-H 2024, Part 1
# ================================
CSR_RULE_REGISTRY_OT = {
    "oil_tanker_scope":           {"rule_ref": "Pt1.Ch1.Sec1[1.3.1]",               "title": "CSR-H scope — oil tanker applicability",    "level": "scope"},
    "typical_midship_arrangement":{"rule_ref": "Pt1.Ch2.Sec1[1.1]",                 "title": "Double-hull oil tanker arrangement",         "level": "arrangement"},
    "double_bottom_height":       {"rule_ref": "Pt1.Ch2.Sec3[2.3.1]",               "title": "Minimum double bottom height",               "level": "arrangement"},
    "double_side_width":          {"rule_ref": "Pt1.Ch2.Sec3[3.1.1]",               "title": "Minimum double side width",                  "level": "arrangement"},
    "double_side_clearance":      {"rule_ref": "Pt1.Ch2.Sec3[3.2.2]",               "title": "Minimum clearance in double side",           "level": "arrangement"},
    "double_bottom_framing":      {"rule_ref": "Pt1.Ch3.Sec2[2.1]",                 "title": "Double bottom longitudinal framing",         "level": "arrangement"},
    "double_side_framing":        {"rule_ref": "Pt1.Ch3.Sec3[2.1]",                 "title": "Double side longitudinal framing",           "level": "arrangement"},
    "weld_joint_detail":          {"rule_ref": "Pt1.Ch12.Sec3",                      "title": "Weld joint detail requirements",             "level": "detail_design"},
    "upper_hopper_knuckle":       {"rule_ref": "Pt1.Ch3.Sec6[2.2.1]+Pt1.Ch9.Sec6[4]", "title": "Upper hopper knuckle fatigue",           "level": "detail_design"},
    "lower_hopper_knuckle":       {"rule_ref": "Pt1.Ch9.Sec6[4]",                   "title": "Lower hopper knuckle fatigue",               "level": "detail_design"},
    "horizontal_stringer_heel":   {"rule_ref": "Pt1.Ch9.Sec6[5]",                   "title": "Horizontal stringer heel fatigue",           "level": "detail_design"},
    "bulkhead_stool_connection":  {"rule_ref": "Pt1.Ch9.Sec7",                      "title": "Transverse BHD stool connection",            "level": "detail_design"},
}


# ── Ship type identifier for hull-form renderer ──
_SHIP_TYPE = 'Tanker'

def _ot_rule_meta(check_id):
    r = CSR_RULE_REGISTRY_OT.get(check_id, {})
    return r.get("rule_ref", ""), r.get("title", check_id), r.get("level", "")

def make_csr_check_ot(check_id, status, *, inputs=None, actual=None, required=None,
                      unit=None, notes=None):
    rule_ref, title, level = _ot_rule_meta(check_id)
    out = {"check_id": check_id, "rule_ref": rule_ref, "title": title,
           "level": level, "status": status}
    if inputs is not None:   out["inputs"] = inputs
    if actual is not None:   out["actual"] = actual
    if required is not None: out["required"] = required
    if unit is not None:     out["unit"] = unit
    if notes is not None:    out["notes"] = notes
    return out

def estimate_dwt_t(L_m, B_m, D_m, *, cb=0.82, t_over_d=0.70, lightship_frac=0.13):
    """Estimate deadweight (DWT, t) from principal particulars.

    Empirical proxy used for CSR-H Pt1.Ch2.Sec3[3.1.1] double-side width when
    the dataset's parametric generator does not specify draught/lightship.

    Δ = CB · L · B · T · ρ_sw      (kt; ρ_sw = 1.025 t/m³)
    DWT ≈ Δ · (1 − lightship_frac)

    Defaults match typical Aframax/Suezmax oil tanker hull form (CB ≈ 0.82,
    T/D ≈ 0.70, lightship ≈ 13% of displacement). VLCC override: cb=0.83.
    """
    rho_sw = 1.025
    T_m = t_over_d * float(D_m)
    displacement_t = float(cb) * float(L_m) * float(B_m) * T_m * rho_sw
    return float(displacement_t * (1.0 - float(lightship_frac)))

def build_ship_data_context(L_m, ship_data_defaults=None, *, B_m=None, D_m=None,
                            cb_estimate=0.82):
    """Build ship-level metadata used in CSR checks (DWT, LLL, framing system).

    If `DWT_t` is not provided in defaults but `B_m` and `D_m` are supplied,
    estimate DWT via :func:`estimate_dwt_t` and tag the basis as
    "estimated_from_LBD" so downstream checks can record the assumption.
    """
    d = dict(ship_data_defaults or {})
    LLL_m = d.get("LLL_m") or L_m
    LLL_basis = "provided" if d.get("LLL_m") else "proxy_from_L_m"
    dwt = d.get("DWT_t")
    dwt_basis = "provided"
    if dwt is None and B_m is not None and D_m is not None:
        dwt = estimate_dwt_t(L_m, B_m, D_m, cb=cb_estimate)
        dwt_basis = "estimated_from_LBD"
    return {
        "ship_type": d.get("ship_type", "oil_tanker"),
        "DWT_t": dwt,
        "DWT_basis": dwt_basis,
        "LLL_m": float(LLL_m),
        "LLL_basis": LLL_basis,
        "framing_system": d.get("framing_system", "longitudinal"),
    }

def build_generator_constraints_summary(generator_inputs, ship, issues):
    """Build enriched generator_constraints block tracking inactive/clamped params."""
    inactive = []
    overrides = []
    suppressed = []

    lb = generator_inputs.get("lbhd_ratio", generator_inputs.get("LB_ratio", 0.0))
    if lb == 0.0:
        inactive.append({
            "parameter": "lbhd_ratio",
            "value": 0.0,
            "reason": "Standard tanker has no centerline longitudinal bulkhead; LBHD fixed at CL (y=0).",
        })

    req = generator_inputs.get("requested_out_girder_y_m")
    eff = generator_inputs.get("effective_out_girder_y_m")
    if req is not None and eff is not None and abs(req - eff) > 1e-4:
        overrides.append({
            "parameter": "girder2_ratio (Out_Girder)",
            "requested_value_m": round(req, 4),
            "effective_value_m": round(eff, 4),
            "reason": "Out_Girder clamped inboard of bilge toe clearance.",
        })

    has_str2 = "Str2" in (getattr(ship, "members", None) or {})
    if not has_str2:
        suppressed.append({"member": "Str2",     "reason": "Vertical gap |z_Str1 - z_Str3| < 6.0 m; mid stringer suppressed."})
        suppressed.append({"member": "HGirder2", "reason": "Suppressed together with Str2."})

    feature_flags = {
        "has_Str2": has_str2,
        "ballast_schema_variant": "three_upper_wing_ballast_tiers" if has_str2 else "merged_upper_mid_lower_wing_ballast",
    }
    return {
        "status": "pass" if not issues else "issues",
        "issues": issues,
        "inactive_parameters": inactive,
        "parameter_overrides": overrides,
        "suppressed_members": suppressed,
        "feature_flags": feature_flags,
    }

def _seg_intersection(seg_a, seg_b):
    """Return (y,z) mm intersection of two (p1,p2) segments, or None."""
    if seg_a is None or seg_b is None:
        return None
    (ay1, az1), (ay2, az2) = seg_a
    (by1, bz1), (by2, bz2) = seg_b
    day, daz = ay2 - ay1, az2 - az1
    dby, dbz = by2 - by1, bz2 - bz1
    denom = day * dbz - daz * dby
    if abs(denom) < 1e-9:
        return None
    dy = by1 - ay1; dz = bz1 - az1
    t = (dy * dbz - dz * dby) / denom
    return (round(ay1 + t * day, 3), round(az1 + t * daz, 3))

def evaluate_csr_rules_tanker(generator_inputs, ship_data, ship):
    """
    Comprehensive CSR-H 2024 evaluation for double-hull oil tanker.
    Correct formulae:
      DB >= max(1.0, min(B/15, 2.0))   [Pt1.Ch2.Sec3[2.3.1]]
      DS >= min(0.5 + DWT/20000, 2.0)  [Pt1.Ch2.Sec3[3.1.1]]  — DWT-based
    4 states: pass / fail / undetermined / not_modeled
    """
    checks = []
    assumptions = []

    if ship_data.get("LLL_basis") == "proxy_from_L_m":
        assumptions.append("LLL_m not provided; L_m used as proxy for scope screening.")
    if ship_data.get("DWT_t") is None:
        assumptions.append("DWT_t not available; double-side width rule is undetermined.")
    if ship_data.get("framing_system") == "longitudinal":
        assumptions.append("Framing system assumed longitudinal.")

    # 1. Scope check
    LLL_m = float(ship_data.get("LLL_m", generator_inputs.get("L_m", 0)))
    checks.append(make_csr_check_ot(
        "oil_tanker_scope", "pass" if LLL_m >= 150.0 else "fail",
        inputs={"LLL_m": round(LLL_m, 3), "LLL_basis": ship_data.get("LLL_basis")},
        actual=round(LLL_m, 3), required={"min_m": 150.0}, unit="m",
        notes="Scope applicability screening check.",
    ))

    # 2. Arrangement check
    mbrs = getattr(ship, "members", None) or {}
    arr = {
        "single_deck":   "Upper_Deck"  in mbrs,
        "double_side":   all(n in mbrs for n in ("Side_Shell", "IHull")),
        "double_bottom": all(n in mbrs for n in ("Bottom_Shell", "IBTM")),
    }
    checks.append(make_csr_check_ot(
        "typical_midship_arrangement", "pass" if all(arr.values()) else "fail",
        inputs=arr, actual=arr,
        required={"single_deck": True, "double_side": True, "double_bottom": True},
        notes="Confirms modeled section follows double-hull tanker arrangement.",
    ))

    # 3. Double bottom height — max(1.0, min(B/15, 2.0))
    B_m  = float(generator_inputs.get("B_m", ship.B))
    DB_m = float(generator_inputs.get("doubleBottom_m", generator_inputs.get("DB_m", ship.d_db)))
    required_db = max(1.0, min(B_m / 15.0, 2.0))
    checks.append(make_csr_check_ot(
        "double_bottom_height", "pass" if DB_m >= required_db - 1e-9 else "fail",
        inputs={"B_m": round(B_m, 3), "DB_m": round(DB_m, 3)},
        actual=round(DB_m, 3), required={"min_m": round(required_db, 4)}, unit="m",
    ))

    # 4. Double side width — min(0.5 + DWT/20000, 2.0)
    DS_m  = float(generator_inputs.get("doubleSide_m", generator_inputs.get("DS_m", ship.d_ds)))
    DWT_t = ship_data.get("DWT_t")
    if DWT_t is None:
        checks.append(make_csr_check_ot(
            "double_side_width", "undetermined",
            inputs={"DS_m": round(DS_m, 3), "DWT_t": None},
            actual=round(DS_m, 3), unit="m",
            notes="DWT_t required to compute minimum double-side width.",
        ))
    else:
        required_ds = min(0.5 + float(DWT_t) / 20000.0, 2.0)
        checks.append(make_csr_check_ot(
            "double_side_width", "pass" if DS_m >= required_ds - 1e-9 else "fail",
            inputs={"DS_m": round(DS_m, 3), "DWT_t": round(float(DWT_t), 1)},
            actual=round(DS_m, 3), required={"min_m": round(required_ds, 4)}, unit="m",
        ))

    # 5. Double side clearance (framing-based: 800 mm longitudinal, 600 mm transverse)
    framing = str(ship_data.get("framing_system") or "").strip().lower()
    req_clear_mm = 800.0 if framing == "longitudinal" else (600.0 if framing == "transverse" else None)
    gross_mm = DS_m * 1000.0
    if req_clear_mm is None:
        clear_st, clear_nt = "undetermined", "framing_system must be declared."
    elif gross_mm < req_clear_mm - 1e-9:
        clear_st, clear_nt = "fail", "Gross DS width smaller than required clearance; compliance impossible."
    else:
        clear_st, clear_nt = "undetermined", "Only gross-width screen performed; stiffener dims not modeled."
    checks.append(make_csr_check_ot(
        "double_side_clearance", clear_st,
        inputs={"DS_m": round(DS_m, 3), "framing_system": framing},
        actual={"gross_width_mm": round(gross_mm, 1)},
        required={"min_clearance_mm": req_clear_mm}, unit="mm", notes=clear_nt,
    ))

    # 6. Double bottom framing (L > 120 m)
    L_m = float(generator_inputs.get("L_m", ship.L))
    if L_m > 120.0:
        if framing == "longitudinal":
            db_st, db_nt = "pass", "Longitudinal framing declared."
        elif framing:
            db_st, db_nt = "fail", "For L > 120 m, longitudinal framing required."
        else:
            db_st, db_nt = "undetermined", "framing_system not declared."
    else:
        db_st, db_nt = "not_modeled", "Rule applies only for L > 120 m."
    checks.append(make_csr_check_ot("double_bottom_framing", db_st,
        inputs={"L_m": round(L_m, 3), "framing_system": framing}, notes=db_nt))

    # 7. Double side framing preference
    if framing == "longitudinal":
        ds_st, ds_nt = "pass", "Longitudinal framing declared."
    elif framing:
        ds_st, ds_nt = "undetermined", "Alternative framing needs special class consideration."
    else:
        ds_st, ds_nt = "undetermined", "framing_system not declared."
    checks.append(make_csr_check_ot("double_side_framing", ds_st,
        inputs={"framing_system": framing}, notes=ds_nt))

    # 8. Weld joint detail (not modeled)
    checks.append(make_csr_check_ot("weld_joint_detail", "undetermined",
        inputs={"weld_type": None},
        notes="Generator does not create fabrication/weld metadata.",
    ))

    # --- CSR Hotspots ---
    segs = ship.seg_dict()
    hotspots = []

    upper_pt = _seg_intersection(segs.get("Hopper"), segs.get("IHull"))
    hotspots.append({
        "hotspot_id": "upper_hopper_knuckle",
        "rule_ref": CSR_RULE_REGISTRY_OT["upper_hopper_knuckle"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_OT["upper_hopper_knuckle"]["title"],
        "availability": "modeled" if upper_pt is not None else "not_modeled",
        "point_mm": upper_pt, "related_members": ["Hopper", "IHull"],
        "csr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "bracket_geometry", "weld_penetration_type"],
        "description": "Fatigue-sensitive upper hopper knuckle intersection.",
    })

    lower_pt = _seg_intersection(segs.get("Hopper"), segs.get("IBTM"))
    hotspots.append({
        "hotspot_id": "lower_hopper_knuckle",
        "rule_ref": CSR_RULE_REGISTRY_OT["lower_hopper_knuckle"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_OT["lower_hopper_knuckle"]["title"],
        "availability": "modeled" if lower_pt is not None else "not_modeled",
        "point_mm": lower_pt, "related_members": ["Hopper", "IBTM", "Out_Girder"],
        "csr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "girder_connection_detail", "weld_penetration_type"],
        "description": "Fatigue-sensitive lower hopper knuckle intersection.",
    })

    for str_name in ("Str1", "Str2", "Str3"):
        if str_name not in mbrs:
            continue
        pt = _seg_intersection(segs.get(str_name), segs.get("IHull"))
        hotspots.append({
            "hotspot_id": f"horizontal_stringer_heel_{str_name}",
            "rule_ref": CSR_RULE_REGISTRY_OT["horizontal_stringer_heel"]["rule_ref"],
            "title": f"{CSR_RULE_REGISTRY_OT['horizontal_stringer_heel']['title']} ({str_name})",
            "availability": "modeled" if pt is not None else "not_modeled",
            "point_mm": pt, "related_members": [str_name, "IHull"],
            "csr_evaluation_status": "undetermined",
            "required_additional_inputs": ["plate_thickness_mm", "heel_bracket_geometry", "weld_penetration_type"],
            "description": f"Fatigue-sensitive stringer heel for {str_name}.",
        })

    hotspots.append({
        "hotspot_id": "bulkhead_stool_connection",
        "rule_ref": CSR_RULE_REGISTRY_OT["bulkhead_stool_connection"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_OT["bulkhead_stool_connection"]["title"],
        "availability": "not_modeled", "point_mm": None, "related_members": [],
        "csr_evaluation_status": "not_modeled",
        "required_additional_inputs": ["bulkhead_type", "stool_geometry", "local_FE_geometry"],
        "description": "Transverse bulkhead stool not modeled parametrically.",
    })

    counts = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
    for c in checks:
        s = c.get("status")
        if s in counts: counts[s] += 1
    overall = "fail" if counts["fail"] > 0 else ("partial" if counts["undetermined"] + counts["not_modeled"] > 0 else "pass")

    return {
        "standard": CSR_STANDARD_INFO,
        "ship_type": ship_data.get("ship_type", "oil_tanker"),
        "ship_data_basis": ship_data,
        "assumptions": assumptions,
        "auto_checks": checks,
        "detail_hotspots": hotspots,
        "needs_additional_input": [
            {"check_id": c["check_id"], "rule_ref": c.get("rule_ref"), "notes": c.get("notes")}
            for c in checks if c.get("status") in ("undetermined", "not_modeled")
        ],
        "summary": {
            "check_counts": counts,
            "hotspot_counts": {
                "modeled":     sum(1 for h in hotspots if h.get("availability") == "modeled"),
                "not_modeled": sum(1 for h in hotspots if h.get("availability") == "not_modeled"),
            },
            "overall_arrangement_status": overall,
        },
    }


# ===============================
# 공통 유틸
# ===============================
EPS = 1e-9

def float_range(start, stop, step):
    """포함형 구간 등분점 (부동소수 안전 라운딩)"""
    n0 = int(round(start / step))
    n1 = int(round(stop / step))
    for k in range(n0, n1 + 1):
        yield round(k * step, 10)

def fmt_token(val, nd=1):
    """파일명 안전용 토큰 (소수점 -> 'p')"""
    s = f"{val:.{nd}f}"
    # 소수점이 있을 때(nd>0)만 뒷자리 0과 점을 제거
    if nd > 0 and '.' in s:
         s = s.rstrip('0').rstrip('.')
    return s.replace('.', 'p')

def build_filename(base_dir, L, B, D, camber, d_ds, d_db, r_bilge,
                   lbhd_r, g1_r, g2_r, s1_r, s2_r, s3_r):
    name = (
        f"Tanker_L{fmt_token(L, 0)}_"
        f"B{fmt_token(B)}_D{fmt_token(D)}_"
        f"C{fmt_token(camber)}_DS{fmt_token(d_ds)}_DB{fmt_token(d_db)}_R{fmt_token(r_bilge)}_"
        f"LB{fmt_token(lbhd_r)}_G1{fmt_token(g1_r)}_G2{fmt_token(g2_r)}_"
        f"S1{fmt_token(s1_r)}_S2{fmt_token(s2_r)}_S3{fmt_token(s3_r)}.dxf"
    )
    return os.path.join(base_dir, name)

def unravel_index(idx, dims):
    out = []
    for base in reversed(dims):
        out.append(idx % base)
        idx //= base
    return list(reversed(out))

def quantize_to_step(x, start, step):
    return round(round((x - start) / step) * step + start, 10)

def lhs_samples(N, specs, seed=None):
    """BULKC와 동일 구조의 간단 LHS"""
    rng = random.Random(seed)
    per_dim_bins = []
    for sp in specs:
        lo, hi = sp['min'], sp['max']
        width = (hi - lo) / max(N, 1)
        vals = [lo + i * width + rng.random() * width for i in range(N)] if N > 0 else []
        rng.shuffle(vals)
        if sp.get('step') is not None:
            vals = [quantize_to_step(v, sp['min'], sp['step']) for v in vals]
        if sp['type'] == 'int':
            vals = [int(round(v)) for v in vals]
        per_dim_bins.append(vals)

    out = []
    for i in range(N):
        item = {}
        for d, sp in enumerate(specs):
            item[sp['name']] = per_dim_bins[d][i]
        out.append(item)
    return out

def clean_multiline_label(label: str) -> str:
    return (label or "").replace("\\P", " ").replace("  ", " ").strip()

def expand_abbrev(token: str) -> str:
    return clean_multiline_label(token or "")

def _round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_round_floats(obj, 3), f, ensure_ascii=False, indent=2)

def append_csv(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if first:
            w.writeheader()
        w.writerow(row)


# ===============================
# Length / Hold-length utilities
# ===============================
def _estimate_length(HL: float,
                     *,
                     fwd_len: float,
                     er_len: float,
                     aft_len: float,
                     # CHANGED: hold1_factor/number_of_hold_excl_first → hold_len_factor/number_of_hold
                     hold_len_factor: float,
                     number_of_hold: int) -> float:
    """
    L = fwd_len + (hold_len_factor * HL * number_of_hold) + er_len + aft_len
    """
    hold_part = hold_len_factor * HL * number_of_hold
    return float(fwd_len + hold_part + er_len + aft_len)


def sample_HL(N: int, hl_spec: tuple[float, float, float]) -> list[float]:
    """
    등간격 샘플 + 소량 난수 미세 분산(파일 간 중복 감소)
    hl_spec: (min, max, step)
    """
    lo, hi, step = hl_spec
    vals = []
    x = lo
    while x <= hi + 1e-9 and len(vals) < N:
        vals.append(round(x, 3))
        x += step
    if len(vals) >= N:
        return vals[:N]
    out = []
    rng = random.Random(1234)
    i = 0
    while len(out) < N:
        base = vals[i % len(vals)]
        jitter = rng.uniform(-0.25 * step, 0.25 * step)
        out.append(round(max(lo, min(hi, base + jitter)), 3))
        i += 1
    return out

# ===============================
# Longitudinal Layout Helper
#  - 0에서 +방향으로 AFT, ER, HOLD N..1, FWD
# ===============================
def build_longitudinal_layout(L_m: float,
                              HL_m: float,
                              number_of_hold: int,
                              fwd_len_m: float,
                              er_len_m: float,
                              aft_len_m: float,
                              hold_len_factor: float):
    """
    ship 길이 L_m와 길이 모델을 이용해
    AFT / ER / HOLD N..1 / FWD 구간의 x 위치(mm)를 계산.

    반환:
      {
        'L_m': ...,
        'HL_m': ...,
        'hold_seg_m': ...,
        'fwd_len_m': ...,
        'er_len_m': ...,
        'aft_len_m': ...,
        'segments': [
          {'name': 'AFT', 'x0_mm': ..., 'x1_mm': ...},
          {'name': 'ER', ...},
          {'name': 'HOLD N', ...},
          ...
          {'name': 'HOLD 1', ...},
          {'name': 'FWD', ...},
        ],
        'bulkheads_mm': [x0, x1, x2, ...]  # 구간 경계 위치(mm)
      }
    """
    if number_of_hold <= 0:
        raise ValueError("number_of_hold must be >= 1")

    hold_seg_m = hold_len_factor * HL_m

    model_L = aft_len_m + er_len_m + number_of_hold * hold_seg_m + fwd_len_m

    # L과 길이 모델 오차가 있을 경우 FWD 길이에 흡수 (수 mm 수준 정리)
    fwd_adj_m = fwd_len_m + (L_m - model_L)
    if fwd_adj_m < 0:
        # 혹시라도 음수가 되면 0으로 클램프
        fwd_adj_m = max(0.0, fwd_len_m)

    scale = 1000.0  # m -> mm
    segs = []
    bulkheads = []

    x = 0.0

    # AFT
    x0 = x
    x1 = x0 + aft_len_m * scale
    segs.append({'name': 'AFT', 'x0_mm': x0, 'x1_mm': x1})
    bulkheads.append(x0)
    x = x1

    # ER
    x0 = x
    x1 = x0 + er_len_m * scale
    segs.append({'name': 'ER', 'x0_mm': x0, 'x1_mm': x1})
    bulkheads.append(x0)
    x = x1

    # HOLD N .. HOLD 1 (AFT에서 FWD 순서대로)
    for k in range(number_of_hold):
        idx_from_aft = number_of_hold - k  # aftmost: N, fwdmost: 1
        name = f"HOLD {idx_from_aft}"
        x0 = x
        x1 = x0 + hold_seg_m * scale
        segs.append({'name': name, 'x0_mm': x0, 'x1_mm': x1})
        bulkheads.append(x0)
        x = x1

    # FWD
    x0 = x
    x1 = x0 + fwd_adj_m * scale
    segs.append({'name': 'FWD', 'x0_mm': x0, 'x1_mm': x1})
    bulkheads.append(x0)
    bulkheads.append(x1)  # 최종 끝 위치

    return {
        'L_m': L_m,
        'HL_m': HL_m,
        'hold_seg_m': hold_seg_m,
        'fwd_len_m': fwd_adj_m,
        'er_len_m': er_len_m,
        'aft_len_m': aft_len_m,
        'segments': segs,
        'bulkheads_mm': bulkheads,
    }


# ===============================
# 단면 좌표 모델 (Tanker)
# ===============================
class Tanker:
    def __init__(self, L, B, D,
                 d_ds, d_db, d_hgir, h_camber,
                 y_lbhd, y_1gir, y_2gir,
                 z_3str, z_2str, z_1str, r_bilge,
                 outg_clear=0.8):   # bilge toe로부터 최소 이격(m)

        self.L = L; self.B = B; self.D = D
        self.r_bilge = r_bilge; self.h_camber = h_camber
        self.d_ds = d_ds; self.d_db = d_db
        self.outg_clear = max(0.0, float(outg_clear))

        # 입력값(초기)
        self.y_lbhd = 0.0
        self.y_1gir = y_1gir
        self.y_ihull = self.B/2 - self.d_ds
        self.z_1str = z_1str
        self.z_2str = z_2str
        self.z_3str = z_3str

        # Out_Girder 안전 위치 보정
        y_bilge_toe = self.B/2 - self.r_bilge
        y2_safe_max = y_bilge_toe - self.outg_clear
        y2_clamped = min(y_2gir, y2_safe_max)
        if y2_clamped < 0.0:
            y2_clamped = 0.0
        self.y_2gir = y2_clamped

        # ---- 하부 멤버 정의 (mm) ----
        Ls = self.L * 500  # 기존 스케일 유지
        B2_mm = (self.B/2) * 1000

        self.memb_btm  = [[Ls, Ls], [0, (self.B/2 - self.r_bilge)*1000], [0, 0]]
        self.memb_side = [[Ls, Ls], [self.B*500, self.B*500], [self.r_bilge*1000, self.D*1000]]
        self.memb_deck = [[Ls, Ls], [0, self.B*500], [self.z_deck(0)*1000, self.D*1000]]

        self.memb_lbhd = [[Ls, Ls], [0.0, 0.0], [self.d_db*1000, self.z_deck(0.0)*1000]]
        self.memb_ibtm = [[Ls, Ls], [0, self.y_2gir*1000], [self.d_db*1000, self.d_db*1000]]
        self.memb_ihull= [[Ls, Ls], [self.y_ihull*1000, self.y_ihull*1000], [self.z_3str*1000, self.z_deck(self.y_ihull)*1000]]
        self.memb_hopp = [[Ls, Ls], [self.y_2gir*1000, self.y_ihull*1000], [self.d_db*1000, self.z_3str*1000]]

        self.memb_0gir = [[Ls, Ls], [0, 0], [0, self.d_db*1000]]                # CL_Girder
        self.memb_2gir = [[Ls, Ls], [self.y_2gir*1000, self.y_2gir*1000], [0, self.d_db*1000]]  # Out_Girder

        # ---- HGirders (수평 거더: 1,2,3) ----
        self.memb_hgir1 = [[Ls, Ls], [0.0, d_hgir * 1000], [self.z_1str * 1000, self.z_1str * 1000]]
        self.memb_hgir2 = [[Ls, Ls], [0.0, d_hgir * 1000], [self.z_2str * 1000, self.z_2str * 1000]]
        self.memb_hgir3 = [[Ls, Ls], [0.0, d_hgir * 1000], [self.z_3str * 1000, self.z_3str * 1000]]

        # ---- Stringers (수평 보강재: 1,2,3) ----
        self.memb_1str = [[Ls, Ls], [self.y_ihull*1000, B2_mm], [self.z_1str*1000, self.z_1str*1000]]
        self.memb_3str = [[Ls, Ls], [self.y_ihull*1000, B2_mm], [self.z_3str*1000, self.z_3str*1000]]

        # NEW: Str1과 Str3 사이 간격 체크(단위 m). 6m 미만이면 Str2/HGirder2 삭제
        if abs(self.z_1str - self.z_3str) >= 6.0:
            self.memb_2str = [[Ls, Ls], [self.y_ihull*1000, B2_mm], [self.z_2str*1000, self.z_2str*1000]]
        else:
            self.memb_2str = None
            self.memb_hgir2 = None  # <- Str2가 없으면 HGirder2도 제거



        self.members = {
            "Bottom_Shell": self.memb_btm,
            "Side_Shell":   self.memb_side,
            "Upper_Deck":   self.memb_deck,
            "LBHD":         self.memb_lbhd,
            "IBTM":         self.memb_ibtm,
            "IHull":        self.memb_ihull,
            "Hopper":       self.memb_hopp,
            "CL_Girder":    self.memb_0gir,
            "Out_Girder":   self.memb_2gir,
            "HGirder1":     self.memb_hgir1,
            "HGirder3":     self.memb_hgir3,
            "Str1":         self.memb_1str,
            "Str3":         self.memb_3str,
        }
        if self.memb_hgir2 is not None:
            self.members["HGirder2"] = self.memb_hgir2
        if self.memb_2str is not None:
            self.members["Str2"] = self.memb_2str


    def z_deck(self, y):
        """캠버 적용된 상갑판 높이"""
        return -(self.h_camber / (self.B/2)) * y + (self.D + self.h_camber)

    def seg_dict(self):
        """(y,z) 두 점 튜플로 구성된 선분 사전 반환"""
        def seg(m):
            return ((float(m[1][0]), float(m[2][0])), (float(m[1][1]), float(m[2][1])))
        return {name: seg(m) for name, m in self.members.items()}


# ===============================
#   Stiffener type / scantling config
# ===============================
_STF_CFG = {
    # (stf_type, flange_half_mm, web_h_mm)  ← web_h from _SCANTLING_TABLE
    "Upper_Deck":   ("T",  75, 350),  # 350 x 12 + 150 x 20 F.B(T)
    "Bottom_Shell": ("T",  75, 400),  # 400 x 14 + 150 x 22 F.B(T)
    "Side_Shell":   ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)
    "IHull":        ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)  (Inner Hull)
    "IBTM":         ("T",  75, 380),  # 380 x 14 + 150 x 20 F.B(T)  (Inner Bottom)
    "Hopper":       ("IA", 90, 280),  # 280 x 10 + 90 x 14 I.A
    "Str3":         ("IA", 90, 280),  # 280 x 10 + 90 x 14 I.A  (Str3 Hopper)
    "CL_Girder":    ("FB",  0, 200),  # 200 x 12 F.B
    "Out_Girder":   ("FB",  0, 150),  # 150 x 10 F.B
    # Not in table — reasonable defaults:
    "LBHD":         ("FB",  0, 200),
    "Str1":         ("IA", 50, 200),
    "Str2":         ("IA", 50, 200),
}

_STF_TYPE_LEGEND = {
    "F.B":    "Flat Bar — web only, no flange",
    "I.A":    "Inverted Angle — web + one-side flange (L-shape)",
    "F.B(T)": "Built-up T-bar — web + both-side flanges (T-shape)",
}

_SCANTLING_TABLE = [
    ("MEMBER",         "PLATE (mm)", "STIFFENER"),
    ("Upper Deck",     "14.0",       "350 x 12 + 150 x 20 F.B(T)"),
    ("Bottom Shell",   "18.0",       "400 x 14 + 150 x 22 F.B(T)"),
    ("Side Shell",     "15.0",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Inner Hull",     "14.0",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Inner Bottom",   "16.0",       "380 x 14 + 150 x 20 F.B(T)"),
    ("Hopper",         "12.0",       "280 x 10 + 90 x 14 I.A"),
    ("Str3 (Hopper)",  "13.0",       "280 x 10 + 90 x 14 I.A"),
    ("CL Girder",      "12.0",       "200 x 12 F.B"),
    ("Out Girder",     "11.0",       "150 x 10 F.B"),
]

# ===============================
# DXF Exporter (+ LNGC-style 메타 수집/용량/K토큰)
# ===============================
class DXFExporterMM:
    def __init__(self, ship: Tanker, text_height=250, offset=300,
                 stf_min=700, stf_max=1000, stf_target=850, stf_len=400, edge_clear=10,
                 # CHANGED: hold1_factor/number_of_hold_excl_first → hold_len_factor/hold_vol_factor/number_of_hold
                 hold_length_m=None,
                 hold_len_factor: float = 1.0,     # NEW
                 hold_vol_factor: float = 0.7,     # NEW
                 number_of_hold: int | None = None,# NEW
                 tank_length_m=None):
        self.ship = ship
        self.text_height = text_height
        self.offset = offset
        self.doc = ezdxf.new(setup=True)
        self.msp = self.doc.modelspace()

        self.stf_min = stf_min
        self.stf_max = stf_max
        self.stf_target = stf_target
        self.stf_len = stf_len
        self.edge_clear = edge_clear

        # CHANGED: 저장값 전환
        self.hold_length_m = hold_length_m
        self.hold_len_factor = float(hold_len_factor)
        self.hold_vol_factor = float(hold_vol_factor)
        self.number_of_hold = int(number_of_hold) if number_of_hold is not None else None

        self.tank_length_m = tank_length_m if tank_length_m is not None else hold_length_m

        # 내부 수집기 ...
        self._labels = []
        self._stf_stats = {}
        self._compartment_data = []
        self._intersections = []
        self.placed_label_polys = []
        self.bilge_bottom_end = None
        self.bilge_side_start = None


        # stiffener 방향 표
        self.STF_DIR = {
            "Upper_Deck":   (0.0, -1.0),
            "Bottom_Shell": (0.0, +1.0),
            "IBTM":         (0.0, -1.0),
            "Side_Shell":   (-1.0, 0.0),  # inboard
            "IHull":        (+1.0, 0.0),
            "LBHD":         (+1.0, 0.0),
            "CL_Girder":    (+1.0, 0.0),
            "Out_Girder":   (-1.0, 0.0),
            "Str1":         (0.0, -1.0),
            "Str2":         (0.0, -1.0),
            "Str3":         (0.0, -1.0),
        }

    # ---------- 기하/라벨 보조 ----------
    @staticmethod
    def _rotated_corners(cx, cz, w, h, angle):
        ca, sa = cos(angle), sin(angle)
        hw, hh = w/2, h/2
        return [
            (cx + hw*ca - hh*sa, cz + hw*sa + hh*ca),
            (cx - hw*ca - hh*sa, cz - hw*sa + hh*ca),
            (cx - hw*ca + hh*sa, cz - hw*sa - hh*ca),
            (cx + hw*ca + hh*sa, cz + hw*sa - hh*ca),
        ]

    @staticmethod
    def _polygons_overlap(poly1, poly2):
        def axes(poly):
            for i in range(4):
                x1, y1 = poly[i]
                x2, y2 = poly[(i+1)%4]
                yield (-(y2-y1), x2-x1)

        def project(poly, axis):
            ax, ay = axis
            L = hypot(ax, ay)
            if L == 0: return 0, 0
            ax/=L; ay/=L
            dots = [p[0]*ax + p[1]*ay for p in poly]
            return min(dots), max(dots)

        for axis in list(axes(poly1)) + list(axes(poly2)):
            a1, a2 = project(poly1, axis)
            b1, b2 = project(poly2, axis)
            if a2 < b1 or b2 < a1:
                return False
        return True

    @staticmethod
    def _line_intersection(p1, p2, p3, p4):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < EPS:
            return None
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
        # 세그먼트 내 포함 여부는 사용하는 쪽에서 체크
        return (px, py)

    @staticmethod
    def _seg_dir_len(p1, p2):
        dy = p2[0]-p1[0]; dz = p2[1]-p1[1]
        L = hypot(dy, dz)
        if L < EPS: return (0.0, 0.0, 0.0)
        return (dy/L, dz/L, L)

    @staticmethod
    def _project_t(p1, p2, p):
        uy, uz, L = DXFExporterMM._seg_dir_len(p1, p2)
        if L < EPS: return 0.0
        return (p[0]-p1[0])*uy + (p[1]-p1[1])*uz

    def _ensure_layer(self, name, color):
        layer = self.doc.layers.get(name) if name in self.doc.layers else self.doc.layers.add(name)
        layer.dxf.color = color

    def draw_layers(self):
        def L(n, c):
            if n not in self.doc.layers: self.doc.layers.add(n).dxf.color = c
        L("Members", 3); L("Label", 1); L("Compartment", 6); L("Bilge", 3)
        L("Stiffeners (Longi)", 4)   # cyan — longitudinal stiffeners
        L("Stiffeners (Trans)", 30)  # orange — transverse indicators
        L("Center", 8)
        L("Scantling", 252)          # dark gray — scantling table

    def _add_label_record(self, name, pos, rot_deg, layer="Label"):
        self._labels.append({
            'name': name,
            'full_name': expand_abbrev(clean_multiline_label(name)),
            'pos_mm': (float(round(pos[0],3)), float(round(pos[1],3))),
            'rotation_deg': float(rot_deg),
            'layer': layer
        })

    # ---- Title & Specs  ----
    def draw_title_and_specs(self, title: str = "ORDINARY SECTION (STBD)"):
        # CL에서의 상갑판 z(mm) 추출
        try:
            z_deck_cl = float(self.ship.memb_deck[2][0])  # at CL, mm
        except Exception:
            z_deck_cl = 0.0

        base_z = z_deck_cl + 5000.0  # 제목을 갑판 위로 띄움 (VLCC와 동일)
        center_y = 0.0  # 중심선(C.L.) 기준 정렬

        def put_line(text, dy_mult, size_mult=1.0):
            char_h = self.text_height * size_mult
            ty = base_z - self.text_height * dy_mult
            t = self.msp.add_mtext(text, dxfattribs={'char_height': char_h, 'layer': 'Label'})
            t.dxf.insert = (center_y, ty)
            t.dxf.attachment_point = 5
            t.dxf.rotation = 0
            self._add_label_record(text, (center_y, ty), 0.0, "Label")

        # Title
        put_line(title, dy_mult=-0.0, size_mult=1.5)

        # BREADTH, DEPTH only — section drawing should not include
        # longitudinal info (NUMBER OF HOLD / HOLD LENGTH / SHIP LENGTH)
        # which belongs to the compartment-arrangement view.
        put_line(f"BREADTH = {float(self.ship.B):.1f} m", dy_mult=2.6)
        put_line(f"DEPTH = {float(self.ship.D):.1f} m",   dy_mult=4.4)

    # ---------- 도면(시각) ----------
    def draw_centerline(self):
        upper_deck_z = self.ship.memb_deck[2][0]
        cl_top_z = upper_deck_z + 500

        line = self.msp.add_line((0, 0), (0, cl_top_z), dxfattribs={'layer': 'Center'})
        try:
            line.dxf.linetype = "CENTER"
            line.dxf.ltscale = 200
        except Exception:
            pass

        t = self.msp.add_mtext("C.L.", dxfattribs={'char_height': 250, 'layer': 'Label'})
        t.dxf.insert = (-500, cl_top_z + 300)
        t.dxf.rotation = 90.0
        t.dxf.attachment_point = 5
        self._add_label_record("C.L.", (-500, cl_top_z + 300), 90.0, "Label")

    def draw_line_mm(self, y_coords, z_coords, label=None, side="+", offset=None,
                     rotation_mode="parallel", normal_policy="auto_up", record_label=True):
        if offset is None:
            offset = self.offset
        side_sign = 1 if side == "+" else -1

        y1, z1 = y_coords[0], z_coords[0]
        y2, z2 = y_coords[1], z_coords[1]
        self.msp.add_line((y1, z1), (y2, z2), dxfattribs={'layer': 'Members'})

        if not label:
            return

        dy, dz = (y2 - y1), (z2 - z1)
        angle_rad = atan2(dz, dy)
        angle_deg = degrees(angle_rad)

        t_len = hypot(dy, dz)
        if t_len < EPS:
            return
        ty, tz = dy / t_len, dz / t_len

        n1y, n1z = -tz, ty
        n2y, n2z = tz, -ty

        def choose_normal(policy):
            cands = [(n1y, n1z), (n2y, n2z)]
            if policy == "out_y+":
                return max(cands, key=lambda n: n[0])
            if policy == "out_y-":
                return min(cands, key=lambda n: n[0])
            return max(cands, key=lambda n: n[1])  # auto_up

        ny, nz = choose_normal(normal_policy)
        ny *= side_sign; nz *= side_sign

        my, mz = (y1 + y2) / 2, (z1 + z2) / 2
        base_ly, base_lz = my + offset * ny, mz + offset * nz

        if rotation_mode == "horizontal":
            text_rot = 0.0
            rot_rad = 0.0
        elif rotation_mode == "vertical":
            text_rot = 90.0
            rot_rad = pi/2
        else:
            readable_deg = angle_deg
            if readable_deg > 90 or readable_deg < -90:
                readable_deg += 180
            text_rot = readable_deg
            rot_rad = atan2(dz, dy)
            if text_rot != angle_deg:
                rot_rad += pi if text_rot - angle_deg in (180, -180) else 0.0

        width = max(1, len(label)) * self.text_height * 0.6
        height = self.text_height

        def overlaps_any(poly):
            for prev in self.placed_label_polys:
                if self._polygons_overlap(poly, prev):
                    return True
            return False

        ly, lz = base_ly, base_lz
        curr_poly = self._rotated_corners(ly, lz, width, height, rot_rad)

        step, max_steps = self.text_height, 120
        if overlaps_any(curr_poly):
            for k in range(1, max_steps + 1):
                for sign in (1, -1):
                    ly_p = base_ly + sign * k * step * ty
                    lz_p = base_lz + sign * k * step * tz
                    poly_p = self._rotated_corners(ly_p, lz_p, width, height, rot_rad)
                    if not overlaps_any(poly_p):
                        ly, lz = ly_p, lz_p
                        curr_poly = poly_p
                        break
                else:
                    continue
                break

        txt = self.msp.add_mtext(label, dxfattribs={'char_height': self.text_height, 'layer': 'Label'})
        txt.dxf.insert = (ly, lz)
        txt.dxf.attachment_point = 5
        txt.dxf.rotation = text_rot

        self.placed_label_polys.append(curr_poly)
        if record_label:
            self._add_label_record(label, (ly, lz), text_rot, "Label")

    def draw_members(self):
        label_prefs = {
            "Upper_Deck":   {"side": "+", "offset": 500, "rotation": "parallel", "normal": "auto_up"},
            "Bottom_Shell": {"side": "-", "offset": 500, "rotation": "parallel", "normal": "auto_up"},
            "Side_Shell":   {"side": "+", "offset": 500, "rotation": "parallel", "normal": "out_y+"},
            "CL_Girder":    {"side": "+", "offset": 400, "rotation": "parallel", "normal": "auto_up"},
            "Out_Girder":   {"side": "-", "offset": 350, "rotation": "parallel", "normal": "auto_up"},
        }
        for name, memb in self.ship.members.items():
            y_coords = memb[1]; z_coords = memb[2]
            pref = label_prefs.get(name, {"side": "+", "offset": self.offset, "rotation": "parallel", "normal": "auto_up"})
            self.draw_line_mm(y_coords, z_coords, label=name,
                              side=pref["side"], offset=pref["offset"],
                              rotation_mode=pref["rotation"], normal_policy=pref["normal"])

    def draw_bilge_curve(self):
        R = self.ship.r_bilge * 1000
        bottom_end = (self.ship.memb_btm[1][1], self.ship.memb_btm[2][1])
        side_start = (self.ship.memb_side[1][0], self.ship.memb_side[2][0])
        self.bilge_bottom_end = bottom_end
        self.bilge_side_start = side_start

        cy = self.ship.B * 1000 / 2 - R
        cz = R

        start_angle = atan2(bottom_end[1] - cz, bottom_end[0] - cy)
        end_angle   = atan2(side_start[1] - cz, side_start[0] - cy)
        start_deg = start_angle * 180.0 / pi; end_deg = end_angle * 180.0 / pi

        self.msp.add_arc(center=(cy, cz), radius=R, start_angle=start_deg, end_angle=end_deg, dxfattribs={'layer': 'Bilge'})
        mid_angle = (start_angle + end_angle) / 2
        label_x = cy + (R + 300) * cos(mid_angle); label_z = cz + (R + 300) * sin(mid_angle)
        lab = self.msp.add_mtext("Bilge", dxfattribs={'char_height': self.text_height, 'layer': 'Label'})
        lab.set_location((label_x, label_z), rotation=0)
        self._add_label_record("Bilge", (label_x, label_z), 0.0, "Label")

    # ---- 구획(Compartment) 폴리곤/면적/중심(BULKC식 수집) ----
    @staticmethod
    def _poly_area_perimeter(verts):
        n = len(verts)
        if n < 3:
            return 0.0, 0.0
        area2 = 0.0
        per = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area2 += x1 * y2 - x2 * y1
            per += hypot(x2 - x1, y2 - y1)
        return abs(area2) * 0.5, per

    @staticmethod
    def _poly_centroid(pts):
        A = 0.0; Cx = 0.0; Cy = 0.0; n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
            cross = x1 * y2 - x2 * y1
            A += cross; Cx += (x1 + x2) * cross; Cy += (y1 + y2) * cross
        A *= 0.5
        if abs(A) < 1e-9:
            mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
            return (mx, my)
        return (Cx / (6 * A), Cy / (6 * A))

    def draw_compartments(self):
        # 레이어 준비
        if "Compartment" not in self.doc.layers:
            self.doc.layers.add("Compartment").dxf.color = 6

        segs = self.ship.seg_dict()
        # Bilge 연결점 추가 (호퍼/사이드/바텀 경계용)
        if self.bilge_bottom_end and self.bilge_side_start:
            segs["Bilge"] = (self.bilge_bottom_end, self.bilge_side_start)

        # LNGC식: CL 세그먼트 추가(면적 경계용)
        deck_top_cl = self.ship.z_deck(0) * 1000.0
        segs["CL"] = ((0.0, 0.0), (0.0, deck_top_cl))

        # --- Str2 존재 여부에 따라 컴파트먼트 동적 정의 ---
        has_str2 = ("Str2" in segs)

        if has_str2:
            # 기존 정의 그대로 (Str2가 있을 때)
            comps = {
                "Cargo tank": [
                    ("Upper_Deck","IHull"),
                    ("IHull","Hopper"),
                    ("Hopper","IBTM"),
                    ("IBTM","LBHD"),
                    ("LBHD","Upper_Deck")
                ],
                "Ballast\\Ptank 1": [
                    ("Upper_Deck","Side_Shell"),
                    ("Side_Shell","Str1"),
                    ("Str1","IHull"),
                    ("IHull","Upper_Deck")
                ],
                "Ballast\\Ptank 2": [
                    ("Str1","Side_Shell"),
                    ("Side_Shell","Str2"),
                    ("Str2","IHull"),
                    ("IHull","Str1")
                ],
                "Ballast\\Ptank 3": [
                    ("Str2","Side_Shell"),
                    ("Side_Shell","Str3"),
                    ("Str3","IHull"),
                    ("IHull","Str2")
                ],
                "Ballast tank 4": [
                    ("Str3","Side_Shell"),
                    ("Side_Shell","Bilge"),
                    ("Bilge","Bottom_Shell"),
                    ("Bottom_Shell","Out_Girder"),
                    ("Out_Girder","Hopper"),
                    ("Hopper","Str3")
                ],
                "Ballast tank 5": [
                    ("IBTM","Out_Girder"),
                    ("Out_Girder","Bottom_Shell"),
                    ("Bottom_Shell","CL_Girder"),
                    ("CL_Girder","IBTM")
                ],
            }
        else:
            # Str2가 없을 때: Str1~Str3 사이의 공간을 'Ballast tank 2'로 통합
            # (Ballast tank 3는 생성하지 않음)
            comps = {
                "Cargo tank": [
                    ("Upper_Deck","IHull"),
                    ("IHull","Hopper"),
                    ("Hopper","IBTM"),
                    ("IBTM","LBHD"),
                    ("LBHD","Upper_Deck")
                ],
                "Ballast\\Ptank 1": [
                    ("Upper_Deck","Side_Shell"),
                    ("Side_Shell","Str1"),
                    ("Str1","IHull"),
                    ("IHull","Upper_Deck")
                ],
                "Ballast\\Ptank 2": [
                    ("Str1","Side_Shell"),
                    ("Side_Shell","Str3"),
                    ("Str3","IHull"),
                    ("IHull","Str1")
                ],
                "Ballast tank 3": [
                    ("Str3","Side_Shell"),
                    ("Side_Shell","Bilge"),
                    ("Bilge","Bottom_Shell"),
                    ("Bottom_Shell","Out_Girder"),
                    ("Out_Girder","Hopper"),
                    ("Hopper","Str3")
                ],
                "Ballast tank 4": [
                    ("IBTM","Out_Girder"),
                    ("Out_Girder","Bottom_Shell"),
                    ("Bottom_Shell","CL_Girder"),
                    ("CL_Girder","IBTM")
                ],
            }


        def add_comp(name, edges):
            verts = []
            for a, b in edges:
                if a not in segs or b not in segs:
                    return
                p1, p2 = segs[a]; q1, q2 = segs[b]
                ip = self._line_intersection(p1, p2, q1, q2)
                if ip is None:
                    return
                verts.append((float(ip[0]), float(ip[1])))
            if len(verts) < 3:
                return
            cx, cy = self._poly_centroid(verts)
            # Cargo tank 라벨은 draw_scantling_table()에서 갑판↔테이블 사이에 배치
            if "cargo" not in clean_multiline_label(name).lower():
                t = self.msp.add_mtext(name, dxfattribs={'char_height': self.text_height, 'layer': 'Compartment'})
                t.dxf.insert = (cx, cy); t.dxf.attachment_point = 5; t.dxf.rotation = 0

            # 메타 축적
            area_mm2, per_all = self._poly_area_perimeter(verts)
            self._compartment_data.append({
                "raw_label": name,
                "clean_label": clean_multiline_label(name),
                "centroid_mm": (round(cx, 3), round(cy, 3)),
                "vertices_mm": [(round(v[0], 3), round(v[1], 3)) for v in verts],
                "area_mm2": round(area_mm2, 3),
                "area_m2": round(area_mm2 / 1e6, 6),
                "perimeter_mm_excl_CL": round(per_all, 3),  # 단순 표기
                "edges_used": edges,
            })

        for cname, edges in comps.items():
            add_comp(cname, edges)

        # 교차점 전체 수집
        keys = sorted(segs.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                p1, p2 = segs[k1]; q1, q2 = segs[k2]
                ip = self._line_intersection(p1, p2, q1, q2)
                # 세그먼트 범위 내 체크
                if ip is not None:
                    # 투영 범위 확인
                    def in_range(a,b,p):
                        uy, uz, L = self._seg_dir_len(a,b)
                        if L < EPS: return False
                        t = self._project_t(a,b,p)
                        return -1e-6 <= t <= L + 1e-6
                    if in_range(p1,p2,ip) and in_range(q1,q2,ip):
                        self._intersections.append({"a": k1, "b": k2, "point_mm": (round(ip[0],3), round(ip[1],3))})

    # ========= STIFFENER =========
    def _choose_n_for_spacing(self, L):
        eff = L - 2*self.edge_clear
        if eff < self.stf_min:
            return 0, None
        n_max = int(eff // self.stf_min)
        best = (0, None)
        for n in range(1, n_max+1):
            s = eff/(n+1)
            if self.stf_min <= s <= self.stf_max:
                if best[0] == 0 or abs(s - self.stf_target) < abs((best[1] or s) - self.stf_target):
                    best = (n, s)
        return best

    def _split_by_intersections(self, name, seg_dict_for_split):
        p1, p2 = seg_dict_for_split[name]
        uy, uz, L = self._seg_dir_len(p1, p2)
        if L < EPS: return []
        ts = [0.0, L]
        for other, (q1, q2) in seg_dict_for_split.items():
            if other == name: continue
            ip = self._line_intersection(p1, p2, q1, q2)
            if ip is None: continue
            t_self = self._project_t(p1, p2, ip)
            if -1e-6 <= t_self <= L + 1e-6:
                vq = self._seg_dir_len(q1, q2)
                t_other = self._project_t(q1, q2, ip)
                if -1e-6 <= t_other <= vq[2] + 1e-6:
                    ts.append(max(0.0, min(L, t_self)))
        ts = sorted(set(round(t, 6) for t in ts))
        pieces = []
        for a, b in zip(ts[:-1], ts[1:]):
            if b - a > 1e-3:
                s = (p1[0] + uy*a, p1[1] + uz*a)
                e = (p1[0] + uy*b, p1[1] + uz*b)
                pieces.append((s, e))
        return pieces

    def _even_points_on_piece(self, s, e):
        uy, uz, L = self._seg_dir_len(s, e)
        if L < 1e-6: return []
        n, spacing = self._choose_n_for_spacing(L)
        if n <= 0: return []
        pts = []
        t0 = self.edge_clear
        for i in range(1, n+1):
            t = t0 + spacing*i
            if t >= L - self.edge_clear + 1e-6:
                break
            pts.append((s[0] + uy*t, s[1] + uz*t))
        return pts

    def _draw_tick(self, base, nvec, layer="Stiffeners (Longi)"):
        ny, nz = nvec
        L = hypot(ny, nz)
        if L < EPS: return
        ny, nz = ny/L, nz/L
        p2 = (base[0] + ny*self.stf_len, base[1] + nz*self.stf_len)
        self.msp.add_line(base, p2, dxfattribs={'layer': layer})

    def _draw_stiffener_shape(self, base, nvec, along_vec, stf_type, web_len, flange_half, layer):
        """Draw typed stiffener cross-section: FB / IA / T"""
        ny, nz = nvec
        Ln = hypot(ny, nz)
        if Ln < 1e-9: return
        ny /= Ln; nz /= Ln
        ay, az = along_vec
        La = hypot(ay, az)
        if La > 1e-9: ay /= La; az /= La
        else: ay, az = 1.0, 0.0
        tip = (base[0] + ny * web_len, base[1] + nz * web_len)
        self.msp.add_line(base, tip, dxfattribs={'layer': layer})
        if stf_type == 'IA':
            fe = (tip[0] + ay * flange_half, tip[1] + az * flange_half)
            self.msp.add_line(tip, fe, dxfattribs={'layer': layer})
        elif stf_type == 'T':
            f1 = (tip[0] - ay * flange_half, tip[1] - az * flange_half)
            f2 = (tip[0] + ay * flange_half, tip[1] + az * flange_half)
            self.msp.add_line(f1, f2, dxfattribs={'layer': layer})

    def draw_stiffeners(self):
        _longi_layer = "Stiffeners (Longi)"

        segs_all = self.ship.seg_dict()
        gen_names = ["Upper_Deck", "Bottom_Shell", "IBTM", "Side_Shell", "IHull", "Hopper",
                     "LBHD", "CL_Girder", "Out_Girder", "Str1", "Str2", "Str3"]

        for name in gen_names:
            if name not in segs_all:
                continue

            pieces = self._split_by_intersections(name, segs_all)
            if not pieces:
                continue

            stf_type, flange_half, web_h = _STF_CFG.get(name, ("FB", 0, 400))
            tick_count = 0
            for s, e in pieces:
                uy, uz, _ = self._seg_dir_len(s, e)
                if name == "Upper_Deck":
                    n1 = (-uz, uy)
                    n2 = (uz, -uy)
                    nvec = n1 if n1[1] < 0 else n2
                elif name == "Hopper":
                    n1 = (-uz, uy)
                    n2 = (uz, -uy)
                    nvec = n1 if n1[0] >= n2[0] else n2
                else:
                    nvec = self.STF_DIR.get(name, (-1.0, 0.0))

                for p in self._even_points_on_piece(s, e):
                    self._draw_stiffener_shape(p, nvec, (uy, uz), stf_type, web_h, flange_half, _longi_layer)
                    tick_count += 1

            if tick_count > 0:
                self._stf_stats[name] = self._stf_stats.get(name, 0) + tick_count

        # ---- Bilge-end anchor stiffeners (BULKC-style: 100mm inboard on members) ----
        BILGE_END_OFFSET = 100.0
        if self.bilge_bottom_end is not None and self.bilge_side_start is not None:
            def _seg_dir_tanker(p1, p2):
                dy = p2[0]-p1[0]; dz = p2[1]-p1[1]; L = hypot(dy, dz)
                return (dy/L, dz/L) if L > 1e-9 else (1.0, 0.0)
            if "Bottom_Shell" in segs_all:
                p1, p2 = segs_all["Bottom_Shell"]
                uy_f, uz_f = _seg_dir_tanker(p1, p2)
                be = self.bilge_bottom_end
                base_be = (be[0] - uy_f * BILGE_END_OFFSET, be[1] - uz_f * BILGE_END_OFFSET)
                nv_be = self.STF_DIR.get("Bottom_Shell", (0.0, 1.0))
                st, fh, wh = _STF_CFG.get("Bottom_Shell", ("T", 75, 400))
                self._draw_stiffener_shape(base_be, nv_be, (uy_f, uz_f), st, wh, fh, _longi_layer)
                self._stf_stats["Bottom_Shell"] = self._stf_stats.get("Bottom_Shell", 0) + 1
            if "Side_Shell" in segs_all:
                p1, p2 = segs_all["Side_Shell"]
                uy_f, uz_f = _seg_dir_tanker(p1, p2)
                bs = self.bilge_side_start
                base_bs = (bs[0] + uy_f * BILGE_END_OFFSET, bs[1] + uz_f * BILGE_END_OFFSET)
                nv_bs = self.STF_DIR.get("Side_Shell", (-1.0, 0.0))
                st, fh, wh = _STF_CFG.get("Side_Shell", ("T", 65, 300))
                self._draw_stiffener_shape(base_bs, nv_bs, (uy_f, uz_f), st, wh, fh, _longi_layer)
                self._stf_stats["Side_Shell"] = self._stf_stats.get("Side_Shell", 0) + 1


    # ---------- 스캔틀링 표 ----------
    def draw_scantling_table(self):
        layer = "Scantling"
        txt_h = 180.0; txt_h_hdr = 200.0
        col_w = [3600.0, 1900.0, 6800.0]; row_h = 700.0
        rows = _SCANTLING_TABLE
        n_rows = len(rows); total_w = sum(col_w); total_h = n_rows * row_h

        ch_cy, ch_cz = None, None
        for c in self._compartment_data:
            if "cargo" in c.get("clean_label", "").lower():
                ch_cy, ch_cz = c["centroid_mm"]; break
        if ch_cy is None:
            B = self.ship.B * 1000.0
            DB = getattr(self.ship, 'DB', 0) * 1000.0
            D = self.ship.D * 1000.0
            ch_cy = B / 4.0
            ch_cz = (DB + D) / 2.0

        ay = max(ch_cy - total_w / 2.0, 200.0)
        az = ch_cz + total_h / 2.0

        pts = [(ay, az), (ay + total_w, az), (ay + total_w, az - total_h), (ay, az - total_h), (ay, az)]
        self.msp.add_lwpolyline(pts, dxfattribs={'layer': layer, 'closed': False})

        x = ay
        for w in col_w[:-1]:
            x += w
            self.msp.add_line((x, az), (x, az - total_h), dxfattribs={'layer': layer})
        for r in range(1, n_rows):
            zy = az - r * row_h
            self.msp.add_line((ay, zy), (ay + total_w, zy), dxfattribs={'layer': layer})

        for r, row in enumerate(rows):
            zy = az - r * row_h - row_h / 2.0
            th = txt_h_hdr if r == 0 else txt_h
            x0 = ay
            for c_idx, (cell, w) in enumerate(zip(row, col_w)):
                cx_ = x0 + w / 2.0
                self.msp.add_mtext(
                    cell,
                    dxfattribs={'layer': layer, 'char_height': th,
                                'attachment_point': 5,
                                'insert': (cx_, zy)}
                )
                x0 += w

        # 제목 (BULKC 스타일: 왼쪽 정렬)
        title = self.msp.add_mtext("SCANTLING TABLE (LONGITUDINALS)",
            dxfattribs={'layer': layer, 'char_height': txt_h_hdr + 30})
        title.dxf.insert = (ay, az + 380.0)
        title.dxf.attachment_point = 4; title.dxf.rotation = 0

        # Cargo Tank 라벨 — Compartment 레이어, 갑판과 테이블 사이 중앙 (BULKC 스타일)
        z_deck_top = self.ship.z_deck(0) * 1000.0
        ch_label = self.msp.add_mtext("Cargo Tank",
            dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
        ch_label.dxf.insert = (ch_cy, (az + z_deck_top) / 2.0)
        ch_label.dxf.attachment_point = 5; ch_label.dxf.rotation = 0

    # ===============================
    # PNG 공통 렌더링 유틸
    # ===============================
    @staticmethod
    def render_dxf_to_png(doc, path_png, dpi=220, bgcolor="white", debug=True):
        """주어진 ezdxf Document를 PNG로 렌더링"""
        if not _MATPLOT_OK:
            if debug:
                print("[PNG] matplotlib/ezdxf drawing 불가(_MATPLOT_OK=False)")
            return False

        fig = None
        try:
            face = mcolors.to_rgba(bgcolor)
            fig = plt.figure(figsize=(12, 12), dpi=dpi, facecolor=face)
            ax = fig.add_axes([0, 0, 1, 1], facecolor=face)

            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(doc.modelspace())

            ax.set_aspect("equal")
            ax.set_axis_off()

            fig.savefig(path_png, dpi=dpi, facecolor=face,
                        bbox_inches="tight", pad_inches=0)

            ok = os.path.isfile(path_png) and os.path.getsize(path_png) > 0
            if not ok and debug:
                print("[PNG] 파일 저장 실패(파일 없음/0바이트)")
            return ok
        except Exception as e:
            if debug:
                print(f"[PNG] 렌더 실패: {e}")
            return False
        finally:
            try:
                if fig is not None:
                    plt.close(fig)
                if plt is not None:
                    plt.close("all")
            except Exception:
                pass

    # ---------- PNG 저장 ----------
    def save_png(self, path_png, dpi=220, bgcolor="white", debug=True):
        return self.render_dxf_to_png(self.doc, path_png, dpi=dpi,
                                      bgcolor=bgcolor, debug=debug)


    # ---------- LNGC-style 메타 빌드 ----------
    def _build_export_stats(self, qc, png_path, final_dxf_path):
        ship = self.ship
        HL = float(self.hold_length_m) if self.hold_length_m is not None else None
        TL = float(self.tank_length_m) if self.tank_length_m is not None else HL

        # --- 1) member geometry ---
        member_props = {}
        for name, (p1, p2) in ship.seg_dict().items():
            (y1, z1), (y2, z2) = p1, p2
            Lmm = hypot(y2 - y1, z2 - z1)
            ang_deg = degrees(atan2(z2 - z1, y2 - y1)) if Lmm > EPS else 0.0
            member_props[name] = {
                "full_name": expand_abbrev(name),
                "endpoints_mm": [(round(y1, 3), round(z1, 3)), (round(y2, 3), round(z2, 3))],
                "length_mm": round(Lmm, 3),
                "length_m": round(Lmm / 1000.0, 6),
                "slope_deg": round(ang_deg, 6),
            }

        # --- 2) member areas (HL 기준) ---
        member_areas = {}
        if HL is not None:
            for nm, prop in member_props.items():
                length_m = prop["length_m"]
                area_half = length_m * HL
                area_full = area_half * 2.0
                member_areas[nm] = {
                    "area_m2_half": round(area_half, 6),
                    "area_m2_full": round(area_full, 6),
                }

        # --- 3) compartments(폴리곤) & volumes (HL 기준) ---
        comp_items = list(self._compartment_data)
        comp_vols = []
        groups_half = {"Void (STBD)": 0.0, "W.B.T (STBD)": 0.0, "Cargo tank (STBD)": 0.0, "Pipe duct (STBD)": 0.0}

        def cname(meta):
            return clean_multiline_label(meta["raw_label"])

        for c in comp_items:
            A = float(c["area_m2"])
            if HL is None:
                continue
            vol_half = A * HL
            vol_full = vol_half * 2.0
            nm = cname(c).lower()
            comp_vols.append({
                "name": cname(c),
                "volume_m3_half": round(vol_half, 6),
                "volume_m3_full": round(vol_full, 6),
            })
            if "cargo" in nm:
                groups_half["Cargo tank (STBD)"] += vol_half
            elif "ballast" in nm:
                groups_half["W.B.T (STBD)"] += vol_half
            elif "pipe" in nm:
                groups_half["Pipe duct (STBD)"] += vol_half
            else:
                groups_half["Void (STBD)"] += vol_half

        groups_full = {k.replace("(STBD)", "(FULL)"): round(v * 2.0, 6) for k, v in groups_half.items()}

        # --- 4) cargo capacity token (FULL per hold → ship total) ---
        cargo_list_full = [v["volume_m3_full"] for v in comp_vols if v["name"].lower().startswith("cargo")]
        cargo_per_hold_full = float(sum(cargo_list_full)) if cargo_list_full else None

        total_cargo_full = None
        cargo_token_k = None
        # FWD-most hold volume reduced by hold_vol_factor; remaining holds at full volume
        if (self.number_of_hold is not None) and (cargo_per_hold_full is not None) and (cargo_per_hold_full > 0.0):
            total_cargo_full = ((self.number_of_hold - 1) * cargo_per_hold_full) + (cargo_per_hold_full * self.hold_vol_factor)
            cargo_token_k = f"{int(round(total_cargo_full / 1000.0))}K"

        # --- 5) layers / bbox / labels / stiffeners ---
        layer_counts = {}
        try:
            for e in self.msp:
                ly = e.dxf.layer if hasattr(e, "dxf") and hasattr(e.dxf, "layer") else "UNKNOWN"
                layer_counts[ly] = layer_counts.get(ly, 0) + 1
        except Exception:
            pass

        ys, zs = [], []
        for mp in member_props.values():
            ys += [mp["endpoints_mm"][0][0], mp["endpoints_mm"][1][0]]
            zs += [mp["endpoints_mm"][0][1], mp["endpoints_mm"][1][1]]
        for c in comp_items:
            for (y, z) in c["vertices_mm"]:
                ys.append(y);
                zs.append(z)
        bbox = {'min_y_mm': round(min(ys), 3), 'max_y_mm': round(max(ys), 3),
                'min_z_mm': round(min(zs), 3), 'max_z_mm': round(max(zs), 3)} if ys and zs else None

        stiffeners_total = sum(self._stf_stats.values()) if self._stf_stats else 0
        stiffeners = {
            'per_member': dict(sorted(self._stf_stats.items())),
            'total': stiffeners_total,
            'rules': {
                'min_spacing_mm': self.stf_min,
                'max_spacing_mm': self.stf_max,
                'target_spacing_mm': self.stf_target,
                'tick_length_mm': self.stf_len,
                'edge_clear_mm': self.edge_clear,
            }
        }

        # --- 6) legend & conventions (LPGC와 동일 키) ---
        doc_conventions = {
            "units": {"lengths": {"drawing": "mm", "model": "m"}, "area": "m^2", "volume": "m^3"},
            "coordinate_system": {
                "axes": "y(horizontal, +outboard), z(vertical, +up)",
                "origin": "Centerline keel point at (0,0) in this section drawing",
                "section": "Midship transverse section (2D, y–z plane)"
            },
            "drawing_conventions": {
                "deck_camber": "Upper_Deck is cambered; z = D + camber - (camber/(B/2))*y",
                "labels_multiline": "\\P are line breaks in CAD text",
                "members": "Member lines are given as two points in mm in (y,z)"
            },
            "params": {
                "descriptions": {
                    'L_m': 'Ship length used (fixed or estimated), meters',
                    'B_m': 'Moulded breadth (B), meters',
                    'D_m': 'Moulded depth (D), meters',
                    'HL_m': 'Hold length per hold (HL), meters',
                    'camberUpper_m': 'Upper deck camber at CL, meters',
                    'doubleBottom_m': 'Inner bottom height (DB), meters',
                    'bilgeRadius_m': 'Bilge radius (R), meters',
                },
                "units": {
                    'L_m': 'm', 'B_m': 'm', 'D_m': 'm', 'HL_m': 'm',
                    'camberUpper_m': 'm', 'doubleBottom_m': 'm', 'bilgeRadius_m': 'm',
                },
                "symbols": {'B_m': 'B', 'D_m': 'D', 'L_m': 'L', 'HL_m': 'HL', 'camberUpper_m': 'C',
                            'doubleBottom_m': 'DB', 'bilgeRadius_m': 'R'}
            }
        }

        drawing_meta = {
            'layers': layer_counts,
            'bbox_mm': bbox,
            'labels': {'count': len(self._labels), 'items': self._labels},
            'stiffeners': stiffeners,
            'intersections': self._intersections,
            'qc': {'label_overlaps': qc.get('label_overlaps', 0), 'labels_ok': qc.get('ok', True)},
            'files': {'dxf': final_dxf_path, 'png': png_path}
        }

        export_stats = {
            'hold': {
                'length_m': HL,
                'tank_length_m': TL,
                'number_of_hold': self.number_of_hold,
                'hold_len_factor': self.hold_len_factor,
                'hold_vol_factor': self.hold_vol_factor,
                'length_basis_note': 'Member areas & compartment volumes use HL; total cargo uses (NoHold × per-hold FULL) × hold_vol_factor.'
            },
            'members': {'geometry': member_props, 'areas': member_areas},
            'compartments': {
                'items': self._compartment_data,
                'volumes': {
                    'items': comp_vols,
                    'groups_half': {k: round(v, 6) for k, v in groups_half.items()},
                    'groups_full': groups_full,
                    'cargo_per_hold_full_m3': round(cargo_per_hold_full, 6) if cargo_per_hold_full is not None else None,
                    'cargo_total_full_m3': round(total_cargo_full, 6) if total_cargo_full is not None else None,
                    'cargo_capacity_token': cargo_token_k,
                },
                'count': len(self._compartment_data),
                'total_area_m2_half': round(sum(c["area_m2"] for c in self._compartment_data),
                                            6) if self._compartment_data else 0.0,
                'total_area_m2_full': round(2.0 * sum(c["area_m2"] for c in self._compartment_data),
                                            6) if self._compartment_data else 0.0,
            },
            'drawing': drawing_meta,
            'domain': {
                'legend': {k: expand_abbrev(k) for k in self.ship.seg_dict().keys()},
                'registry_version': "1.0",
                'conventions': doc_conventions,
                'rule_refs': {
                    'stiffener_rules': f"{self.stf_min} ≤ spacing ≤ {self.stf_max} (target {self.stf_target}), edge clear {self.edge_clear} mm",
                },
                'stiffener_types': _STF_TYPE_LEGEND,
                'scantling_table': [
                    {'member': r[0], 'plate_mm': r[1], 'stiffener': r[2]}
                    for r in _SCANTLING_TABLE[1:]
                ],
            }
        }
        return export_stats

    # ---------- 전체 Export (파일 저장 + PNG + 메타 산출 + K-접두 리네임) ----------
    def export(self, save_as=None, dxf_version='R2018', png_out_dir=None, png_dpi=220):
        qc = {'label_overlaps': -1, 'ok': False}
        png_path = None
        final_dxf_path = save_as

        # 도면 그리기 + 저장
        self.doc = ezdxf.new(setup=True, dxfversion=dxf_version)
        self.msp = self.doc.modelspace()
        self.placed_label_polys = []
        self._labels.clear();
        self._stf_stats.clear();
        self._compartment_data.clear();
        self._intersections.clear()

        # 레이어
        self.draw_layers()

        # 드로잉
        self.draw_centerline()
        self.draw_title_and_specs(title="ORDINARY SECTION (STBD)")
        self.draw_members()
        self.draw_bilge_curve()
        self.draw_compartments()
        self.draw_stiffeners()
        self.draw_scantling_table()

        # QC
        label_overlap_cnt = 0
        for i in range(len(self.placed_label_polys)):
            for j in range(i + 1, len(self.placed_label_polys)):
                if self._polygons_overlap(self.placed_label_polys[i], self.placed_label_polys[j]):
                    label_overlap_cnt += 1
        qc = {'label_overlaps': label_overlap_cnt, 'ok': (label_overlap_cnt == 0)}

        # DXF 저장
        if save_as:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            self.doc.saveas(save_as)
            final_dxf_path = save_as

        # PNG 저장
        if png_out_dir:
            os.makedirs(png_out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(save_as or "tanker"))[0]
            candidate = os.path.join(png_out_dir, base + ".png")
            if self.save_png(candidate, dpi=png_dpi, bgcolor="white", debug=True):
                png_path = candidate

        # export_stats 생성 및 반환
        stats = self._build_export_stats(qc, png_path, final_dxf_path)
        return qc, png_path, stats


# ===============================
# 도메인 규칙 기반 필터링 (Tanker) — (원 코드 유지/호출 가정)
#  - 이 블록은 기존 정의가 상단에 이미 있다면 중복 정의하지 않아도 됩니다.
# ===============================
def domain_rules_ok_tanker(params):
    L = params['L']; B = params['B']; D = params['D']
    C = params['C']; DS = params['DS']; DB = params['DB']; R = params['R']
    LB = params['LB']; G1 = params['G1']; G2 = params['G2']
    S1 = params['S1']; S2 = params['S2']; S3 = params['S3']

    issues = []
    if not (B > 0 and D > 0 and DS > 0 and DB > 0 and R > 0):
        issues.append("PositiveDims")
    if DS >= B/2:
        issues.append("DS_too_large_vs_B")
    if DB >= D/2:
        issues.append("DB_too_large_vs_D")
    if C < 0: issues.append("Camber_negative")
    if C > 0.05 * B: issues.append("Camber_over_0p05B")
    if C > 0.10 * D: issues.append("Camber_over_0p10D")
    if R >= (B/2 - DS) - 0.1: issues.append("BilgeR_exceeds_inner_hull_clearance")
    if R >= (D - DB) - 0.1: issues.append("BilgeR_exceeds_depth_clearance")
    if not (0 <= LB < 1.0): issues.append("LB_ratio_range")
    if not (0 < G1 < G2 < 1.0): issues.append("Girder_ratio_order_Tanker")

    y_ihull = B/2 - DS; y_2gir = G2 * (B/2)
    if y_2gir >= y_ihull - 0.5: issues.append("G2_too_close_to_inner_hull")

    if (S3 * D) <= DB + 0.5: issues.append("Hopper_low_height")
    if (G2 * (B/2)) >= (B/2 - DS) - 0.5: issues.append("Hopper_low_width")
    if (D - S1 * D) < 0.5: issues.append("Str1_too_close_to_deck")

    y_bilge_toe = B/2 - R
    if (G2 * (B/2)) > (y_bilge_toe - 0.8):
        issues.append("Out_Girder_too_outboard_vs_bilge_toe")

    return len(issues) == 0, issues

# ===============================
# Segment display name helper
# ===============================
def _seg_display_name(name: str) -> str:
    n = name.strip()
    if n == 'AFT':
        return 'AFT End'
    elif n == 'ER':
        return 'Engine Room'
    elif n == 'FWD':
        return 'FWD End'
    elif n.upper().startswith('HOLD'):
        num = n[4:].replace('_', '').strip()
        return f'Hold {num}' if num else 'Hold'
    return n

# ===============================
# Center Line Elevation Generator
# ===============================
def create_compartment_arrangement_drawing(
    dxf_path: str,
    layout: dict,
    D_m: float,
    camber_m: float,
    DB_m: float,
    text_height: float = 250,
    png_dir: str | None = None,
    png_dpi: int = 220,
):
    """
    Center line elevation 도면 생성.
    - x: 선박 길이 방향(mm), 0 기준 + 방향 AFT->ER->HOLD N..1->FWD
    - z: 상하(mm)
    """
    os.makedirs(os.path.dirname(dxf_path), exist_ok=True)
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    # 레이어
    for name, color in [
        ("Hull", 3),
        ("Center", 8),
        ("Bulkhead", 2),
        ("Label", 1),
    ]:
        layer = doc.layers.get(name) if name in doc.layers else doc.layers.add(name)
        layer.dxf.color = color

    # 전체 길이(mm)
    segs = layout['segments']
    x_end = segs[-1]['x1_mm']

    # 높이들
    deck_z_mm = (D_m + camber_m) * 1000.0  # CL에서 갑판 캠버 포함
    db_z_mm = DB_m * 1000.0

    # Center line (x=0 수직선)
    msp.add_line((0.0, 0.0), (0.0, deck_z_mm),
                 dxfattribs={'layer': 'Center'})
    # 베이스라인 (keel)
    msp.add_line((0.0, 0.0), (x_end, 0.0),
                 dxfattribs={'layer': 'Hull'})
    # Deck line (CL)
    msp.add_line((0.0, deck_z_mm), (x_end, deck_z_mm),
                 dxfattribs={'layer': 'Hull'})
    # Inner bottom line
    msp.add_line((0.0, db_z_mm), (x_end, db_z_mm),
                 dxfattribs={'layer': 'Hull'})

    # Bulkhead (세로선)
    for bx in layout['bulkheads_mm']:
        msp.add_line((bx, 0.0), (bx, deck_z_mm),
                     dxfattribs={'layer': 'Bulkhead'})

    # 구간 레이블 (AFT, ER, HOLD n..1, FWD)
    for seg in segs:
        x0 = seg['x0_mm']
        x1 = seg['x1_mm']
        cx = 0.5 * (x0 + x1)
        name = seg['name']

        display_name = _seg_display_name(name)
        t = msp.add_mtext(display_name, dxfattribs={'char_height': text_height * 3,
                                            'layer': 'Label'})
        t.dxf.insert = (cx, deck_z_mm + 5 * text_height * 3)
        t.dxf.attachment_point = 5
        t.dxf.rotation = 0.0

        # 길이 숫자도 아래쪽에 추가
        seg_len_m = (x1 - x0) / 1000.0
        len_txt = f"{seg_len_m:.1f} m"
        t2 = msp.add_mtext(len_txt, dxfattribs={'char_height': text_height * 2.4,
                                                'layer': 'Label'})
        t2.dxf.insert = (cx, -4 * text_height * 3)
        t2.dxf.attachment_point = 5
        t2.dxf.rotation = 0.0

    # 전체 길이 레이블
    total_len_txt = f"L = {layout['L_m']:.1f} m"
    t3 = msp.add_mtext(total_len_txt, dxfattribs={'char_height': text_height * 3,
                                                  'layer': 'Label'})
    t3.dxf.insert = (x_end * 0.5, deck_z_mm + 10 * text_height * 3)
    t3.dxf.attachment_point = 5
    t3.dxf.rotation = 0.0

    # 저장
    doc.saveas(dxf_path)

    # --- Matplotlib PNG (colored zones, larger labels) ---
    png_path = None
    if png_dir is not None and _MATPLOT_OK:
        os.makedirs(png_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(dxf_path))[0]
        png_path = os.path.join(png_dir, base + ".png")
        try:
            from _compart_hullform import render_compartment_png
            render_compartment_png(
                png_path, layout, D_m, camber_m, DB_m,
                _SHIP_TYPE, png_dpi)
        except Exception:
            png_path = None

    return dxf_path, png_path


# ===============================
# 3D Model Generator (wireframe, full breadth)
#  - x: length (mm)
#  - y: breadth (mm), 좌우대칭 (STBD: +, PORT: -)
#  - z: vertical (mm)
# ===============================
def create_compartment3d_dxf(
    dxf_path: str,
    ship,          # Tanker instance
    layout: dict,
    text_height: float = 250,
    png_dir: str = None,
    png_dpi: int = 220,
):
    """
    Improved 3D wireframe model with compartment-based layers.
    - Compartment layers: 3D_OUTER_HULL, 3D_INNER_HULL, 3D_DB, 3D_DS, 3D_CARGO_HOLD, 3D_BH_FACE
    - HOLD transverse bulkheads: 3DFACE closed panels
    - Returns (dxf_path, png_path)
    """
    os.makedirs(os.path.dirname(dxf_path), exist_ok=True)

    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    def ensure_layer(name, color):
        if name in doc.layers:
            layer = doc.layers.get(name)
        else:
            layer = doc.layers.add(name)
        layer.dxf.color = color

    # Compartment-based layers with ACI colors
    ensure_layer("3D_OUTER_HULL",   3)   # green
    ensure_layer("3D_INNER_HULL",   4)   # cyan
    ensure_layer("3D_DB",           5)   # blue
    ensure_layer("3D_DS",           6)   # magenta
    ensure_layer("3D_CARGO_HOLD",   2)   # yellow
    ensure_layer("3D_BH_FACE",      1)   # red
    ensure_layer("3D_Label",        7)   # white

    segs = ship.seg_dict()   # {name: ((y1,z1),(y2,z2))} in mm (STBD half)

    # Map member names to compartment layers
    MEMBER_LAYER = {
        "Bottom_Shell": "3D_OUTER_HULL",
        "Side_Shell":   "3D_OUTER_HULL",
        "Upper_Deck":   "3D_OUTER_HULL",
        "LBHD":         "3D_INNER_HULL",
        "IBTM":         "3D_DB",
        "IHull":        "3D_DS",
        "Hopper":       "3D_DS",
        "CL_Girder":    "3D_DB",
        "Out_Girder":   "3D_DB",
        "HGirder1":     "3D_CARGO_HOLD",
        "HGirder2":     "3D_CARGO_HOLD",
        "HGirder3":     "3D_CARGO_HOLD",
        "Str1":         "3D_DS",
        "Str2":         "3D_DS",
        "Str3":         "3D_DS",
    }
    # Outer shell members (drawn in AFT/ER/FWD too)
    OUTER_MEMBERS = {"Bottom_Shell", "Side_Shell", "Upper_Deck"}

    EPSY = 1e-6
    B = ship.B; R = ship.r_bilge
    cy_mm = (B / 2.0 - R) * 1000.0
    cz_mm = R * 1000.0
    R_mm = R * 1000.0

    # Bilge curve points (y,z) in mm
    N_BILGE = 8
    bilge_pts = []
    for i in range(N_BILGE + 1):
        t = -0.5 * pi + (i / N_BILGE) * (0.5 * pi)
        bilge_pts.append((cy_mm + R_mm * cos(t), cz_mm + R_mm * sin(t)))

    def add_panel_wireframe(y1, z1, y2, z2, x0, x1, layer_name):
        """Draw wireframe rectangle (extruded 2D line) as 2 polylines (STBD + PORT)"""
        if abs(y1) < EPSY and abs(y2) < EPSY:
            pts = [(x0,y1,z1),(x0,y2,z2),(x1,y2,z2),(x1,y1,z1),(x0,y1,z1)]
            msp.add_polyline3d(pts, dxfattribs={'layer': layer_name})
            return
        # STBD
        pts_s = [(x0,y1,z1),(x0,y2,z2),(x1,y2,z2),(x1,y1,z1),(x0,y1,z1)]
        msp.add_polyline3d(pts_s, dxfattribs={'layer': layer_name})
        # PORT
        pts_p = [(x0,-y1,z1),(x0,-y2,z2),(x1,-y2,z2),(x1,-y1,z1),(x0,-y1,z1)]
        msp.add_polyline3d(pts_p, dxfattribs={'layer': layer_name})

    # --- Extruded panels per segment ---
    for seg_info in layout['segments']:
        x0 = seg_info['x0_mm']
        x1 = seg_info['x1_mm']
        seg_name = seg_info['name']
        is_hold = seg_name.startswith("HOLD")

        for nm, (p1, p2) in segs.items():
            y1, z1 = p1; y2, z2 = p2
            if not is_hold and nm not in OUTER_MEMBERS:
                continue
            layer = MEMBER_LAYER.get(nm, "3D_CARGO_HOLD")
            add_panel_wireframe(y1, z1, y2, z2, x0, x1, layer)

        # Bilge panels (outer hull)
        for (py1, pz1), (py2, pz2) in zip(bilge_pts[:-1], bilge_pts[1:]):
            add_panel_wireframe(py1, pz1, py2, pz2, x0, x1, "3D_OUTER_HULL")

    # --- Bulkhead at each boundary ---
    for bx in layout['bulkheads_mm']:
        # Determine if this bx is a HOLD boundary
        is_hold_bx = False
        for seg_info in layout['segments']:
            if seg_info['name'].startswith("HOLD"):
                if abs(bx - seg_info['x0_mm']) < 1 or abs(bx - seg_info['x1_mm']) < 1:
                    is_hold_bx = True
                    break

        for nm, (p1, p2) in segs.items():
            y1, z1 = p1; y2, z2 = p2
            # Non-HOLD boundaries: only draw outer shell contour
            if not is_hold_bx and nm not in OUTER_MEMBERS:
                continue
            layer = "3D_BH_FACE" if is_hold_bx else "3D_OUTER_HULL"
            msp.add_line((bx, y1, z1), (bx, y2, z2), dxfattribs={'layer': layer})
            if abs(y1) > EPSY or abs(y2) > EPSY:
                msp.add_line((bx, -y1, z1), (bx, -y2, z2), dxfattribs={'layer': layer})
        # Bilge on BH (always drawn — part of outer contour)
        for (py1,pz1),(py2,pz2) in zip(bilge_pts[:-1], bilge_pts[1:]):
            lyr = "3D_BH_FACE" if is_hold_bx else "3D_OUTER_HULL"
            msp.add_line((bx, py1, pz1), (bx, py2, pz2), dxfattribs={'layer': lyr})
            msp.add_line((bx, -py1, pz1), (bx, -py2, pz2), dxfattribs={'layer': lyr})

    # --- Labels (DXF: 1/3 text size) ---
    deck_z_mm = ship.z_deck(0.0) * 1000.0
    dxf_label_h = max(1, text_height // 3)
    for seg_info in layout['segments']:
        cx = 0.5 * (seg_info['x0_mm'] + seg_info['x1_mm'])
        display = _seg_display_name(seg_info['name'])
        t = msp.add_mtext(display, dxfattribs={'char_height': dxf_label_h, 'layer': '3D_Label'})
        t.dxf.insert = (cx, 0.0, deck_z_mm + 5 * dxf_label_h)
        t.dxf.attachment_point = 5

    doc.saveas(dxf_path)

    # --- Isometric PNG (equal-scale axes, layer-colored, labeled) ---
    # Layer → (hex color, alpha, linewidth)
    _3D_LAYER_STYLE = {
        "3D_OUTER_HULL":  ('#666666', 0.55, 0.5),
        "3D_INNER_HULL":  ('#00bbcc', 0.70, 0.5),
        "3D_DB":          ('#2255ff', 0.70, 0.5),   # ballast (double bottom) — blue
        "3D_DS":          ('#8833ff', 0.65, 0.5),   # ballast (double side) — purple
        "3D_CARGO_HOLD":  ('#dd2222', 0.65, 0.5),   # cargo — red
        "3D_BH_FACE":     ('#ff8800', 0.75, 0.6),   # bulkhead face — orange
    }

    png_path = None
    if png_dir is not None and _MATPLOT_OK:
        os.makedirs(png_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(dxf_path))[0]
        png_path = os.path.join(png_dir, base + ".png")
        try:
            # 1) 좌표 수집 + 범위 계산 (layer 포함)
            all_xs, all_ys, all_zs = [], [], []
            plot_lines = []  # (xs, ys, zs, layer_name)
            for entity in msp:
                try:
                    lyr = entity.dxf.layer
                    if lyr == '3D_Label':
                        continue  # skip text entities
                    if entity.dxftype() == 'LINE':
                        s = entity.dxf.start; e = entity.dxf.end
                        xs = [s.x, e.x]; ys = [s.y, e.y]; zs = [s.z, e.z]
                    elif entity.dxftype() == 'POLYLINE':
                        pts3 = [v.dxf.location for v in entity.vertices]
                        xs = [p.x for p in pts3]
                        ys = [p.y for p in pts3]
                        zs = [p.z for p in pts3]
                    else:
                        continue
                    all_xs.extend(xs); all_ys.extend(ys); all_zs.extend(zs)
                    plot_lines.append((xs, ys, zs, lyr))
                except Exception:
                    pass

            # 2) 데이터 범위로 box aspect 결정
            if all_xs:
                rx = max(all_xs) - min(all_xs)
                ry = max(all_ys) - min(all_ys)
                rz = max(all_zs) - min(all_zs)
            else:
                rx = ry = rz = 1.0
            ry = max(ry, 1.0); rz = max(rz, 1.0); rx = max(rx, 1.0)
            aspect_x = rx / max(ry, rz)
            fig_w = min(max(aspect_x * 5, 10), 28)
            fig_h = max(fig_w / aspect_x * 0.6, 4)

            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X (mm)', fontsize=4)
            ax.set_ylabel('Y (mm)', fontsize=4)
            ax.set_zlabel('Z (mm)', fontsize=4)
            ax.tick_params(labelsize=3)
            ax.view_init(elev=30, azim=-45)

            # 3) 실제 비율로 box aspect 설정
            ax.set_box_aspect((rx, ry, rz))

            # 4) Layer별 색상으로 선 그리기
            for xs, ys, zs, lyr in plot_lines:
                color, alpha, lw = _3D_LAYER_STYLE.get(lyr, ('#444444', 0.5, 0.4))
                ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)

            # 5) 구간 레이블 (matplotlib text)
            label_z = deck_z_mm * 1.04
            label_fs = max(3, min(4, 100 / max(len(layout['segments']), 1)))
            for seg_info in layout['segments']:
                cx = 0.5 * (seg_info['x0_mm'] + seg_info['x1_mm'])
                display = _seg_display_name(seg_info['name'])
                ax.text(cx, 0.0, label_z, display,
                        ha='center', va='bottom', fontsize=label_fs,
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.75, linewidth=0))

            fig.subplots_adjust(left=0.12, right=0.95, bottom=0.05, top=0.95)
            plt.savefig(png_path, dpi=png_dpi, bbox_inches='tight', pad_inches=0.4)
            plt.close(fig)
        except Exception:
            png_path = None

    return dxf_path, png_path



# ===============================
# 데이터셋 생성 루프 (LNGC-style 메타/HL/K 접두 적용)
# ===============================
def generate_tanker_dataset(
    save_dir,
    method='lhs',
    # --- LENGTH MODEL ---
    fwd_hold_ratio=0.10,
    er_hold_ratio=0.25,
    aft_hold_ratio=0.07,

    hold_len_factor=1.0,
    hold_vol_factor=0.7,
    number_of_hold_range=(5, 7, 1),
    use_L_fixed=False,
    L_fixed=250.0,

    compart_out_dir=None,
    compart_png_out_dir=None,
    compart3d_out_dir=None,
    compart3d_png_out_dir=None,
    json_out_dir=None,
    ship_data_defaults=None,

    # --- SAMPLING RANGES ---
    hold_length_range=(25.0, 35.0, 0.1),
    B_range=(38, 53, 1),
    D_range=(15, 30, 1),
    camber_range=(0.5, 2.0, 0.1),
    ds_range=(1.5, 3.5, 0.1),
    db_range=(1.5, 3.5, 0.1),
    bilge_range=(1.0, 3.0, 0.1),
    lbhd_ratio=(0.0, 0.0, 0.05),
    g1_ratio=(0.3, 0.5, 0.05),
    g2_ratio=(0.6, 0.8, 0.05),
    s1_ratio=(0.65, 0.75, 0.05),
    s2_ratio=(0.4, 0.55, 0.05),
    s3_ratio=(0.25, 0.35, 0.05),

    text_height=250,
    offset=300,
    MAX_FILES=100,
    PROGRESS_EVERY=20,
    SEED=None,
    png_out_dir=None,
    png_dpi=220
):
    os.makedirs(save_dir, exist_ok=True)
    rng = random.Random(SEED)

    if compart_out_dir is not None:
        os.makedirs(compart_out_dir, exist_ok=True)
    if compart_png_out_dir is not None:
        os.makedirs(compart_png_out_dir, exist_ok=True)
    if compart3d_out_dir is not None:
        os.makedirs(compart3d_out_dir, exist_ok=True)
    if compart3d_png_out_dir is not None:
        os.makedirs(compart3d_png_out_dir, exist_ok=True)
    if json_out_dir is not None:
        os.makedirs(json_out_dir, exist_ok=True)

    # HL 샘플
    HL_vals = sample_HL(MAX_FILES, hold_length_range)

    # 샘플 값 목록
    B_vals = list(range(B_range[0], B_range[1] + 1, B_range[2]))
    D_vals = list(range(D_range[0], D_range[1] + 1, D_range[2]))
    camber_vals = list(float_range(*camber_range))
    ds_vals = list(float_range(*ds_range))
    db_vals = list(float_range(*db_range))
    bilge_vals = list(float_range(*bilge_range))
    lb_vals = list(float_range(*lbhd_ratio))
    g1_vals = list(float_range(*g1_ratio))
    g2_vals = list(float_range(*g2_ratio))
    s1_vals = list(float_range(*s1_ratio))
    s2_vals = list(float_range(*s2_ratio))
    s3_vals = list(float_range(*s3_ratio))

    axes = [B_vals, D_vals, camber_vals, ds_vals, db_vals, bilge_vals, lb_vals, g1_vals, g2_vals, s1_vals, s2_vals, s3_vals]
    dims = [len(a) for a in axes]
    total = 1
    for n in dims: total *= n

    # 후보 파라미터 생성
    candidate_params = []
    if method == 'grid':
        all_idx = list(range(total))
        if total > MAX_FILES: all_idx = rng.sample(all_idx, MAX_FILES)
        for idx in all_idx:
            ii = unravel_index(idx, dims)
            B, D, C, DS, DB, R, LB, G1, G2, S1, S2, S3 = [axes[d][ii[d]] for d in range(len(dims))]
            candidate_params.append({'B': B,'D': D,'C': C,'DS': DS,'DB': DB,'R': R,
                                     'LB': LB,'G1': G1,'G2': G2,'S1': S1,'S2': S2,'S3': S3})
    elif method == 'random':
        k = min(MAX_FILES, total)
        sampled_linear_idx = rng.sample(range(total), k)
        for idx in sampled_linear_idx:
            ii = unravel_index(idx, dims)
            B, D, C, DS, DB, R, LB, G1, G2, S1, S2, S3 = [axes[d][ii[d]] for d in range(len(dims))]
            candidate_params.append({'B': B,'D': D,'C': C,'DS': DS,'DB': DB,'R': R,
                                     'LB': LB,'G1': G1,'G2': G2,'S1': S1,'S2': S2,'S3': S3})
    elif method == 'lhs':
        N = MAX_FILES
        specs = [
            {'name': 'B',  'min': B_range[0], 'max': B_range[1], 'type': 'int',   'step': B_range[2]},
            {'name': 'D',  'min': D_range[0], 'max': D_range[1], 'type': 'int',   'step': D_range[2]},
            {'name': 'C',  'min': camber_range[0], 'max': camber_range[1], 'type': 'float', 'step': camber_range[2]},
            {'name': 'DS', 'min': ds_range[0], 'max': ds_range[1], 'type': 'float', 'step': ds_range[2]},
            {'name': 'DB', 'min': db_range[0], 'max': db_range[1], 'type': 'float', 'step': db_range[2]},
            {'name': 'R',  'min': bilge_range[0], 'max': bilge_range[1], 'type': 'float', 'step': bilge_range[2]},
            {'name': 'LB', 'min': lbhd_ratio[0], 'max': lbhd_ratio[1], 'type': 'float', 'step': lbhd_ratio[2]},
            {'name': 'G1', 'min': g1_ratio[0], 'max': g1_ratio[1], 'type': 'float', 'step': g1_ratio[2]},
            {'name': 'G2', 'min': g2_ratio[0], 'max': g2_ratio[1], 'type': 'float', 'step': g2_ratio[2]},
            {'name': 'S1', 'min': s1_ratio[0], 'max': s1_ratio[1], 'type': 'float', 'step': s1_ratio[2]},
            {'name': 'S2', 'min': s2_ratio[0], 'max': s2_ratio[1], 'type': 'float', 'step': s2_ratio[2]},
            {'name': 'S3', 'min': s3_ratio[0], 'max': s3_ratio[1], 'type': 'float', 'step': s3_ratio[2]},
        ]
        lhs = lhs_samples(N, specs, seed=SEED)
        for s in lhs:
            candidate_params.append(s)
    else:
        raise ValueError("method must be one of ['lhs','random','grid']")

    # 인덱스 CSV 헤더
    index_csv = os.path.join(save_dir, "Tanker_dataset_index.csv")
    header = [
        'file', 'json', 'method', 'seed',
        'Cargo Capacity (K)', 'CargoCapacity_m3', 'HL_m', 'L_m',
        'NoHold', 'HoldLenFactor', 'HoldVolFactor',
        'B','D','C','DS','DB','R','LB','G1','G2','S1','S2','S3',
        'domain_ok','domain_issues',
        'generator_constraints_ok','generator_constraint_count','inactive_parameters',
        'csr_scope_status','csr_pass','csr_fail','csr_undetermined','csr_not_modeled',
        'DWT_t','LLL_m','framing_system',
        'qc_ok','label_overlaps','stiffeners_total','png'
    ]

    saved = 0
    for i, p in enumerate(candidate_params, start=1):
        # HL
        HL = HL_vals[(i - 1) % len(HL_vals)]

        # NEW: 홀드 수 샘플 (5~7, step=1 기본)
        nh_start, nh_end, nh_step = number_of_hold_range
        number_of_hold = rng.randrange(nh_start, nh_end + 1, nh_step)

        hold_total = hold_len_factor * HL * number_of_hold
        fwd_len = fwd_hold_ratio * hold_total
        er_len  = er_hold_ratio  * hold_total
        aft_len = aft_hold_ratio * hold_total

        # CHANGED: L 계산
        if use_L_fixed:
            L_use = L_fixed
        else:
            L_use = _estimate_length(
                HL,
                fwd_len=fwd_len, er_len=er_len, aft_len=aft_len,
                hold_len_factor=hold_len_factor,
                number_of_hold=number_of_hold
            )


        # 도메인 규칙 체크용 결합 파라미터
        p_all = dict(p); p_all['L'] = L_use
        ok, issues = domain_rules_ok_tanker(p_all)
        if not ok:
            continue

        B = p['B']; D = p['D']
        y_lbhd = 0.0
        y_1gir = p['G1'] * (B/2.0)
        y_2gir = p['G2'] * (B/2.0)
        z_1str = p['S1'] * D
        z_2str = p['S2'] * D
        z_3str = p['S3'] * D
        d_hgir = 1.0  # HGirder 폭(표현용)

        ship = Tanker(
            L=L_use, B=B, D=D,
            d_ds=p['DS'], d_db=p['DB'], d_hgir=d_hgir, h_camber=p['C'],
            y_lbhd=y_lbhd, y_1gir=y_1gir, y_2gir=y_2gir,
            z_3str=z_3str, z_2str=z_2str, z_1str=z_1str, r_bilge=p['R']
        )

        # CSR 평가용 입력 딕셔너리
        _gen_inputs_for_csr = {
            'L_m': L_use, 'B_m': B, 'D_m': D,
            'doubleSide_m': p['DS'], 'doubleBottom_m': p['DB'],
            'bilgeRadius_m': p['R'],
            'lbhd_ratio': p.get('LB', 0.0),
            'girder1_ratio': p['G1'], 'girder2_ratio': p['G2'],
            'requested_out_girder_y_m': y_2gir,
            'effective_out_girder_y_m': ship.y_2gir,
        }
        ship_data = build_ship_data_context(
            L_use, ship_data_defaults=ship_data_defaults,
            B_m=B, D_m=D, cb_estimate=0.82,  # Aframax/Suezmax oil tanker form factor
        )
        generator_constraints = build_generator_constraints_summary(_gen_inputs_for_csr, ship, issues)
        csr_eval = evaluate_csr_rules_tanker(_gen_inputs_for_csr, ship_data, ship)

        # 파일명(리네임 전)
        dxf_path = build_filename(
            save_dir, L_use, B, D, p['C'], p['DS'], p['DB'], p['R'],
            p.get('LB', 0.0), p['G1'], p['G2'], p['S1'], p['S2'], p['S3']
        )

        # 1) Exporter 실행
        exporter = DXFExporterMM(
            ship,
            text_height=text_height, offset=offset,
            hold_length_m=HL,
            hold_len_factor=hold_len_factor,   # NEW
            hold_vol_factor=hold_vol_factor,   # NEW
            number_of_hold=number_of_hold,     # NEW
            tank_length_m=HL
        )
        qc, png_path, stats = exporter.export(save_as=dxf_path, png_out_dir=png_out_dir, png_dpi=png_dpi)

        # 2) K-토큰 추출
        capacity_token = stats.get('compartments', {}).get('volumes', {}).get('cargo_capacity_token')
        final_dxf_path = dxf_path
        if capacity_token:
            base = os.path.basename(dxf_path)
            hold_tag = f"{number_of_hold}Hold"  # NEW
            if not base.startswith(capacity_token + "_"):
                # CHANGED: capa 다음에 Hold 표기 추가
                new_base = f"{capacity_token}_{hold_tag}_{base}"
                new_dxf = os.path.join(os.path.dirname(dxf_path), new_base)
                try:
                    os.replace(dxf_path, new_dxf)
                    final_dxf_path = new_dxf
                    # PNG도 같이
                    if png_path:
                        old_png = png_path
                        new_png = os.path.join(os.path.dirname(old_png), os.path.splitext(new_base)[0] + ".png")
                        try:
                            os.replace(old_png, new_png)
                            png_path = new_png
                        except Exception:
                            pass
                except Exception:
                    pass

        # 3) stats 내부 files 경로 동기화
        try:
            stats['drawing']['files']['dxf'] = final_dxf_path
            stats['drawing']['files']['png'] = png_path
        except Exception:
            pass

        # 3-a) Center line elevation & 3D model 생성
        base_noext = os.path.splitext(os.path.basename(final_dxf_path))[0]

        compart_dxf_path = None
        compart_png_path = None
        compart3d_dxf_path = None
        compart3d_png_path = None

        # 공통 longitudinal layout (AFT/ER/HOLD/FWD)
        layout = None
        if compart_out_dir is not None or compart3d_out_dir is not None:
            layout = build_longitudinal_layout(
                L_m=L_use,
                HL_m=HL,
                number_of_hold=number_of_hold,
                fwd_len_m=fwd_len,
                er_len_m=er_len,
                aft_len_m=aft_len,
                hold_len_factor=hold_len_factor,
            )

        # Elevation 도면 생성
        if compart_out_dir is not None and layout is not None:
            compart_dxf_path = os.path.join(compart_out_dir, base_noext + "_Compart.dxf")
            compart_dxf_path, compart_png_path = create_compartment_arrangement_drawing(
                compart_dxf_path,
                layout=layout,
                D_m=D,
                camber_m=p['C'],
                DB_m=p['DB'],
                text_height=text_height,
                png_dir=compart_png_out_dir,
                png_dpi=png_dpi,
            )

        # 3D 모델 생성
        compart3d_png_path = None
        if compart3d_out_dir is not None and layout is not None:
            model3d_dxf_path_candidate = os.path.join(compart3d_out_dir, base_noext + "_Compart3D.dxf")
            compart3d_dxf_path, compart3d_png_path = create_compartment3d_dxf(
                model3d_dxf_path_candidate,
                ship=ship,
                layout=layout,
                text_height=text_height,
                png_dir=compart3d_png_out_dir,
                png_dpi=png_dpi,
            )

        # 3) stats 내부 files 경로 동기화
        try:
            stats['drawing']['files']['dxf'] = final_dxf_path
            stats['drawing']['files']['png'] = png_path
        except Exception:
            pass

        # 4) 최종 JSON(meta)
        if json_out_dir:
            json_path = os.path.join(json_out_dir, os.path.splitext(os.path.basename(final_dxf_path))[0] + ".json")
        else:
            json_path = final_dxf_path.replace(".dxf", ".json")

        # Build new-style JSON
        sample_idx = saved + 1
        sample_id = f"TANKER-{sample_idx:04d}"

        fwd_len_used = layout['fwd_len_m'] if layout is not None else fwd_len
        er_len_used  = layout['er_len_m']  if layout is not None else er_len
        aft_len_used = layout['aft_len_m'] if layout is not None else aft_len

        meta = {
            'sample_id': sample_id,
            'ship_type': 'Tanker',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': method,
            'seed': SEED,
            'generator_inputs': {
                'L_m': L_use,
                'B_m': B,
                'D_m': D,
                'HL_m': HL,
                'number_of_hold': number_of_hold,
                'camberUpper_m': p['C'],
                'doubleBottom_m': p['DB'],
                'doubleSide_m': p['DS'],
                'bilgeRadius_m': p['R'],
                'girder1_ratio': p['G1'],
                'girder2_ratio': p['G2'],
                'stringer1_ratio': p['S1'],
                'stringer2_ratio': p['S2'],
                'stringer3_ratio': p['S3'],
                'lbhd_ratio': p.get('LB', 0.0),
                '_inactive_note': 'lbhd_ratio: always 0.0 for standard Tanker (no longitudinal BHD in center)',
            },
            'geometry': {
                'derived': {
                    'girderOut_y_m': round(ship.y_2gir, 3),
                    'innerHull_y_m': round(ship.y_ihull, 3),
                    'girder1_y_m': round(ship.y_1gir, 3),
                },
                'longitudinal_layout': layout,
                'length_model': {
                    'fwd_len_m': fwd_len_used,
                    'er_len_m': er_len_used,
                    'aft_len_m': aft_len_used,
                    'hold_len_factor': hold_len_factor,
                    'hold_vol_factor': hold_vol_factor,
                    'mode': 'fixed' if use_L_fixed else 'estimated',
                },
            },
            'member_semantics': {
                'Bottom_Shell': {'description': 'Outer bottom shell plating', 'compartment_boundary': ['SEA', 'DB'], 'structural_class': 'OUTER_HULL'},
                'Side_Shell':   {'description': 'Outer side shell plating', 'compartment_boundary': ['SEA', 'WBT_DS'], 'structural_class': 'OUTER_HULL'},
                'Upper_Deck':   {'description': 'Upper deck plating (cambered)', 'compartment_boundary': ['CARGO_HOLD', 'TOPSIDE'], 'structural_class': 'OUTER_HULL'},
                'LBHD':         {'description': 'Longitudinal BHD (inactive=CL only)', 'compartment_boundary': ['CARGO_CL'], 'structural_class': 'INNER_HULL', 'inactive': True},
                'IBTM':         {'description': 'Inner bottom plating', 'compartment_boundary': ['DB', 'CARGO_HOLD'], 'structural_class': 'INNER_HULL'},
                'IHull':        {'description': 'Inner hull (longitudinal BHD)', 'compartment_boundary': ['WBT_DS', 'CARGO_HOLD'], 'structural_class': 'INNER_HULL'},
                'Hopper':       {'description': 'Hopper plate (inner bottom slant)', 'compartment_boundary': ['DB', 'WBT_DS'], 'structural_class': 'INNER_HULL'},
                'CL_Girder':    {'description': 'Center girder in double bottom', 'compartment_boundary': ['DB'], 'structural_class': 'GIRDER'},
                'Out_Girder':   {'description': 'Outboard girder in double bottom', 'compartment_boundary': ['DB'], 'structural_class': 'GIRDER'},
                'HGirder1':     {'description': 'Horizontal girder 1 on LBHD', 'compartment_boundary': ['CARGO_HOLD'], 'structural_class': 'STRINGER'},
                'HGirder2':     {'description': 'Horizontal girder 2 on LBHD', 'compartment_boundary': ['CARGO_HOLD'], 'structural_class': 'STRINGER'},
                'HGirder3':     {'description': 'Horizontal girder 3 on LBHD', 'compartment_boundary': ['CARGO_HOLD'], 'structural_class': 'STRINGER'},
                'Str1':         {'description': 'Side stringer 1 (upper)', 'compartment_boundary': ['WBT_DS'], 'structural_class': 'STRINGER'},
                'Str2':         {'description': 'Side stringer 2 (mid)', 'compartment_boundary': ['WBT_DS'], 'structural_class': 'STRINGER'},
                'Str3':         {'description': 'Side stringer 3 (lower)', 'compartment_boundary': ['WBT_DS'], 'structural_class': 'STRINGER'},
            },
            'standard_refs': {'csr_standard': CSR_STANDARD_INFO},
            'ship_data': ship_data,
            'generator_constraints': generator_constraints,
            'rules': {**csr_eval, 'society': 'IACS_CSR_H'},  # unified schema (Phase 0.2.B1)
            'csr': csr_eval,  # legacy alias — kept for backward compat
            'cargo_summary': {
                'per_hold_full_m3': stats.get('compartments', {}).get('volumes', {}).get('cargo_per_hold_full_m3'),
                'total_full_m3': stats.get('compartments', {}).get('volumes', {}).get('cargo_total_full_m3'),
                'capacity_token': stats.get('compartments', {}).get('volumes', {}).get('cargo_capacity_token'),
            },
            'artifacts': {
                'section_dxf': final_dxf_path,
                'section_png': png_path,
                'compart_dxf': compart_dxf_path,
                'compart_png': compart_png_path,
                'compart3d_dxf': compart3d_dxf_path,
                'compart3d_png': compart3d_png_path,
                'json': json_path,
            },
            # Keep drawing stats for compatibility
            'drawing': stats,
        }
        write_json(json_path, meta)

        # stiffeners_total
        stf_total = stats.get('drawing', {}).get('stiffeners', {}).get('total', 0)

        row = {
            'file': os.path.basename(final_dxf_path),
            'json': os.path.basename(json_path),
            'method': method,
            'seed': SEED,

            'Cargo Capacity (K)': capacity_token or "",
            'CargoCapacity_m3': round(meta['cargo_summary']['total_full_m3'] or 0.0, 3),

            'HL_m': HL,
            'L_m': L_use,

            'NoHold': number_of_hold,
            'HoldLenFactor': hold_len_factor,
            'HoldVolFactor': hold_vol_factor,

            'B': B, 'D': D, 'C': p['C'], 'DS': p['DS'], 'DB': p['DB'], 'R': p['R'],
            'LB': p.get('LB', 0.0), 'G1': p['G1'], 'G2': p['G2'],
            'S1': p['S1'], 'S2': p['S2'], 'S3': p['S3'],

            'domain_ok': ok,
            'domain_issues': "|".join(issues),
            'generator_constraints_ok': generator_constraints['status'] == 'pass',
            'generator_constraint_count': len(generator_constraints['issues']),
            'inactive_parameters': "|".join(item['parameter'] for item in generator_constraints.get('inactive_parameters', [])),
            'csr_scope_status': next((c.get('status') for c in csr_eval.get('auto_checks', []) if c.get('check_id') == 'oil_tanker_scope'), ""),
            'csr_pass': csr_eval.get('summary', {}).get('check_counts', {}).get('pass', 0),
            'csr_fail': csr_eval.get('summary', {}).get('check_counts', {}).get('fail', 0),
            'csr_undetermined': csr_eval.get('summary', {}).get('check_counts', {}).get('undetermined', 0),
            'csr_not_modeled': csr_eval.get('summary', {}).get('check_counts', {}).get('not_modeled', 0),
            'DWT_t': ship_data.get('DWT_t') if ship_data.get('DWT_t') is not None else "",
            'LLL_m': ship_data.get('LLL_m'),
            'framing_system': ship_data.get('framing_system'),
            'qc_ok': qc.get('ok', False),
            'label_overlaps': qc.get('label_overlaps', -1),
            'stiffeners_total': stf_total,
            'png': os.path.basename(png_path) if png_path else ""
        }
        append_csv(index_csv, header, row)

        saved += 1
        if saved % PROGRESS_EVERY == 0:
            print(f"[{method}] Saved {saved} ... last: {os.path.basename(row['file'])}")
        if saved >= MAX_FILES:
            break

    print(f"Done. Saved files: {saved} (method={method})")



# ===============================
# 예시 실행
# ===============================
if __name__ == "__main__":
    _BASE = "<SHIPBENCH_ROOT>/data/processed/Tanker"

    SAVE_DIR        = os.path.join(_BASE, "section_dxf")
    PNG_DIR         = os.path.join(_BASE, "section_png")
    COMPART_DIR        = os.path.join(_BASE, "compart_dxf")
    COMPART_PNG_DIR    = os.path.join(_BASE, "compart_png")
    COMPART3D_DIR     = os.path.join(_BASE, "compart3d_dxf")
    COMPART3D_PNG_DIR = os.path.join(_BASE, "compart3d_png")
    JSON_DIR        = os.path.join(_BASE, "json")

    generate_tanker_dataset(
        save_dir=SAVE_DIR,
        json_out_dir=JSON_DIR,
        method='lhs',
        fwd_hold_ratio=0.10,
        er_hold_ratio=0.25,
        aft_hold_ratio=0.07,

        hold_len_factor=1.0,
        hold_vol_factor=0.7,
        number_of_hold_range=(5, 7, 1),
        use_L_fixed=False,
        L_fixed=250.0,

        compart_out_dir=COMPART_DIR,
        compart_png_out_dir=COMPART_PNG_DIR,
        compart3d_out_dir=COMPART3D_DIR,
        compart3d_png_out_dir=COMPART3D_PNG_DIR,

        hold_length_range=(25.0, 35.0, 0.1),
        B_range=(38, 53, 1),
        D_range=(15, 30, 1),
        camber_range=(0.5, 2.0, 0.1),
        ds_range=(1.5, 3.5, 0.1),
        db_range=(1.5, 3.5, 0.1),
        bilge_range=(1.0, 3.0, 0.1),
        lbhd_ratio=(0.0, 0.0, 0.05),
        g1_ratio=(0.3, 0.5, 0.05),
        g2_ratio=(0.6, 0.8, 0.05),
        s1_ratio=(0.65, 0.75, 0.05),
        s2_ratio=(0.4, 0.55, 0.05),
        s3_ratio=(0.25, 0.35, 0.05),

        text_height=250,
        offset=300,
        MAX_FILES=100,
        PROGRESS_EVERY=20,
        SEED=42,
        png_out_dir=PNG_DIR,
        png_dpi=220
    )
