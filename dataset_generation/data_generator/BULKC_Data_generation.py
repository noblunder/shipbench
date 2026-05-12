# =========================================
#   BULKC Midship Generator (LNGC-style outputs, No Cargo Tank)
#   - Drawing visuals kept as-is (from your BULKC base)
#   - JSON/CSV/filename aligned to LNGC style
#   - Hold Length (HL) sampled; capacity uses HL (no Tank)
#   - Estimated ship length uses Hold Length (HL) model (configurable)
# =========================================

import os, csv, json, time, random, re
from math import radians, pi, hypot, atan2, degrees

import ezdxf

# PNG (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MAT_OK = True
except Exception:
    _MAT_OK = False
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
# CSR Rule Registry — Bulk Carrier
# IACS CSR-H 2024, Part 1 (Bulk Carrier 전용 규정)
# Oil Tanker와 다른 챕터 적용 — double side 없음, hopper angle 규정, TSWT 존재
# ================================
CSR_RULE_REGISTRY_BC = {
    "bulk_carrier_scope":        {"rule_ref": "Pt1.Ch1.Sec2[1.3.1]",               "title": "CSR-H scope — bulk carrier applicability",   "level": "scope"},
    "typical_midship_arrangement":{"rule_ref": "Pt1.Ch2.Sec2[1.1]",                "title": "Bulk carrier midship arrangement",            "level": "arrangement"},
    "double_bottom_height":      {"rule_ref": "Pt1.Ch2.Sec4[2.3.1]",               "title": "Minimum double bottom height",                "level": "arrangement"},
    "bilge_radius":              {"rule_ref": "Pt1.Ch2.Sec4[2.4.1]",               "title": "Minimum bilge radius",                        "level": "arrangement"},
    "hopper_knuckle_angle":      {"rule_ref": "Pt1.Ch2.Sec4[3.1.1]",               "title": "Minimum hopper knuckle angle",                "level": "arrangement"},
    "topside_wing_tank_framing": {"rule_ref": "Pt1.Ch3.Sec4[2.1]",                 "title": "Topside wing tank framing system",            "level": "arrangement"},
    "longitudinal_framing_bc":   {"rule_ref": "Pt1.Ch3.Sec2[2.1]",                 "title": "Bottom / inner bottom longitudinal framing",  "level": "arrangement"},
    "weld_joint_detail":         {"rule_ref": "Pt1.Ch12.Sec3",                      "title": "Weld joint detail requirements",              "level": "detail_design"},
    "upper_hopper_knuckle":      {"rule_ref": "Pt1.Ch3.Sec6[3.1]+Pt1.Ch9.Sec6[3]","title": "Upper hopper knuckle fatigue",                "level": "detail_design"},
    "lower_hopper_knuckle":      {"rule_ref": "Pt1.Ch9.Sec6[3]",                   "title": "Lower hopper knuckle fatigue",                "level": "detail_design"},
    "inner_bottom_floor_toe":    {"rule_ref": "Pt1.Ch9.Sec6[2]",                   "title": "Inner bottom / floor toe connection",         "level": "detail_design"},
    "bulkhead_lower_stool":      {"rule_ref": "Pt1.Ch9.Sec7[2]",                   "title": "Transverse BHD lower stool connection",       "level": "detail_design"},
}


# ── Ship type identifier for hull-form renderer ──
_SHIP_TYPE = 'BULKC'

def _bc_rule_meta(check_id):
    r = CSR_RULE_REGISTRY_BC.get(check_id, {})
    return r.get("rule_ref", ""), r.get("title", check_id), r.get("level", "")

def make_csr_check_bc(check_id, status, *, inputs=None, actual=None, required=None,
                      unit=None, notes=None):
    rule_ref, title, level = _bc_rule_meta(check_id)
    out = {"check_id": check_id, "rule_ref": rule_ref, "title": title,
           "level": level, "status": status}
    if inputs is not None:   out["inputs"] = inputs
    if actual is not None:   out["actual"] = actual
    if required is not None: out["required"] = required
    if unit is not None:     out["unit"] = unit
    if notes is not None:    out["notes"] = notes
    return out

def build_ship_data_context(L_m, ship_data_defaults=None):
    """Build ship-level metadata used in CSR checks."""
    d = dict(ship_data_defaults or {})
    LLL_m = d.get("LLL_m") or L_m
    LLL_basis = "provided" if d.get("LLL_m") else "proxy_from_L_m"
    return {
        "ship_type": d.get("ship_type", "bulk_carrier"),
        "DWT_t": d.get("DWT_t"),
        "LLL_m": float(LLL_m),
        "LLL_basis": LLL_basis,
        "framing_system": d.get("framing_system", "longitudinal"),
    }

def build_generator_constraints_summary(generator_inputs, ship, issues):
    """Build enriched generator_constraints block for BULKC."""
    feature_flags = {
        "has_TSWT": True,
        "has_double_bottom": True,
        "no_double_side": True,
    }
    return {
        "status": "pass" if not issues else "issues",
        "issues": issues,
        "inactive_parameters": [],
        "parameter_overrides": [],
        "suppressed_members": [],
        "feature_flags": feature_flags,
    }

def _seg_intersection_bc(ship, member_a, member_b):
    """Compute (y,z) mm intersection of two BULKC members using seg_dict()."""
    try:
        segs = ship.seg_dict()
        sa = segs.get(member_a)
        sb = segs.get(member_b)
        if sa is None or sb is None:
            return None
        (ay1, az1), (ay2, az2) = sa
        (by1, bz1), (by2, bz2) = sb
        day, daz = ay2 - ay1, az2 - az1
        dby, dbz = by2 - by1, bz2 - bz1
        denom = day * dbz - daz * dby
        if abs(denom) < 1e-9:
            return None
        dy = by1 - ay1; dz = bz1 - az1
        t = (dy * dbz - dz * dby) / denom
        return (round(ay1 + t * day, 3), round(az1 + t * daz, 3))
    except Exception:
        return None

def evaluate_csr_rules_bulkc(generator_inputs, ship_data, ship):
    """
    Comprehensive CSR-H 2024 evaluation for Bulk Carrier.
    Oil Tanker와 다른 규정 챕터 적용:
      DB >= max(B/20, 0.76)  [Pt1.Ch2.Sec4[2.3.1]]  (tanker는 max(1.0, min(B/15, 2.0)))
      No double side rule    (TSWT는 double side 아님)
      Hopper angle >= 40 deg [Pt1.Ch2.Sec4[3.1.1]]
    4 states: pass / fail / undetermined / not_modeled
    """
    checks = []
    assumptions = []

    if ship_data.get("LLL_basis") == "proxy_from_L_m":
        assumptions.append("LLL_m not provided; L_m used as proxy for scope screening.")
    if ship_data.get("framing_system") == "longitudinal":
        assumptions.append("Framing system assumed longitudinal.")

    # 1. Scope check (BC도 동일 150m threshold)
    LLL_m = float(ship_data.get("LLL_m", generator_inputs.get("L_m", 0)))
    checks.append(make_csr_check_bc(
        "bulk_carrier_scope", "pass" if LLL_m >= 150.0 else "fail",
        inputs={"LLL_m": round(LLL_m, 3), "LLL_basis": ship_data.get("LLL_basis")},
        actual=round(LLL_m, 3), required={"min_m": 150.0}, unit="m",
        notes="Scope applicability screening check.",
    ))

    # 2. Arrangement check (BC: double bottom + hopper + upper deck + TSWT)
    segs = ship.seg_dict()
    arr = {
        "double_bottom":  all(n in segs for n in ("Bottom_Shell", "IBTM")),
        "hopper_plate":   "Hopper" in segs,
        "upper_deck":     any(k.startswith("Upper_Deck") for k in segs),
        "topside_wing":   any(k.startswith("TSWT") for k in segs),
    }
    arr_status = "pass" if arr["double_bottom"] and arr["hopper_plate"] and arr["upper_deck"] else "fail"
    checks.append(make_csr_check_bc(
        "typical_midship_arrangement", arr_status,
        inputs=arr, actual=arr,
        required={"double_bottom": True, "hopper_plate": True, "upper_deck": True},
        notes="BC에는 double side 없음; topside wing tank(TSWT)가 상부 강도 부재 역할.",
    ))

    # 3. Double bottom height — max(B/20, 0.76) (BC 전용 공식)
    B_m  = float(generator_inputs.get("B_m", ship.B))
    DB_m = float(generator_inputs.get("doubleBottom_m", generator_inputs.get("DB_m", ship.DB)))
    required_db = max(B_m / 20.0, 0.76)
    checks.append(make_csr_check_bc(
        "double_bottom_height", "pass" if DB_m >= required_db - 1e-9 else "fail",
        inputs={"B_m": round(B_m, 3), "DB_m": round(DB_m, 3)},
        actual=round(DB_m, 3), required={"min_m": round(required_db, 4)}, unit="m",
    ))

    # 4. Bilge radius >= 0.5 m
    R_m = float(generator_inputs.get("bilgeRadius_m", ship.R))
    checks.append(make_csr_check_bc(
        "bilge_radius", "pass" if R_m >= 0.5 - 1e-9 else "fail",
        inputs={"R_m": round(R_m, 3)},
        actual=round(R_m, 3), required={"min_m": 0.5}, unit="m",
    ))

    # 5. Hopper knuckle angle >= 40 deg
    hopper_angle = float(generator_inputs.get("tswt_ext_deg", getattr(ship, 'tswt_ext', 42.0)))
    checks.append(make_csr_check_bc(
        "hopper_knuckle_angle", "pass" if hopper_angle >= 40.0 - 1e-9 else "fail",
        inputs={"hopper_angle_deg": round(hopper_angle, 2)},
        actual=round(hopper_angle, 2), required={"min_deg": 40.0}, unit="deg",
    ))

    # 6. Longitudinal framing (L > 120 m)
    framing = str(ship_data.get("framing_system") or "").strip().lower()
    L_m = float(generator_inputs.get("L_m", ship.L))
    if L_m > 120.0:
        if framing == "longitudinal":
            frm_st, frm_nt = "pass", "Longitudinal framing declared."
        elif framing:
            frm_st, frm_nt = "fail", "For L > 120 m, longitudinal framing required."
        else:
            frm_st, frm_nt = "undetermined", "framing_system not declared."
    else:
        frm_st, frm_nt = "not_modeled", "Rule applies only for L > 120 m."
    checks.append(make_csr_check_bc("longitudinal_framing_bc", frm_st,
        inputs={"L_m": round(L_m, 3), "framing_system": framing}, notes=frm_nt))

    # 7. Topside wing tank framing (undetermined — parametric model 미상세)
    checks.append(make_csr_check_bc(
        "topside_wing_tank_framing", "undetermined",
        inputs={"framing_system": framing},
        notes="Topside wing tank framing arrangement not detailed in parametric model.",
    ))

    # 8. Weld joint detail
    checks.append(make_csr_check_bc("weld_joint_detail", "undetermined",
        inputs={"weld_type": None},
        notes="Generator does not create fabrication/weld metadata.",
    ))

    # --- CSR Hotspots (BC 전용) ---
    hotspots = []

    # Upper hopper knuckle: TSWT slope / Side_Shell 교점
    upper_pt = _seg_intersection_bc(ship, "TSWT", "Side_Shell")
    hotspots.append({
        "hotspot_id": "upper_hopper_knuckle",
        "rule_ref": CSR_RULE_REGISTRY_BC["upper_hopper_knuckle"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_BC["upper_hopper_knuckle"]["title"],
        "availability": "modeled" if upper_pt is not None else "not_modeled",
        "point_mm": upper_pt, "related_members": ["TSWT", "Side_Shell"],
        "csr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "bracket_geometry", "weld_penetration_type"],
        "description": "Fatigue-sensitive upper hopper (topside wing tank lower) knuckle.",
    })

    # Lower hopper knuckle: Hopper / IBTM 교점
    lower_pt = _seg_intersection_bc(ship, "Hopper", "IBTM")
    hotspots.append({
        "hotspot_id": "lower_hopper_knuckle",
        "rule_ref": CSR_RULE_REGISTRY_BC["lower_hopper_knuckle"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_BC["lower_hopper_knuckle"]["title"],
        "availability": "modeled" if lower_pt is not None else "not_modeled",
        "point_mm": lower_pt, "related_members": ["Hopper", "IBTM", "Out_Girder"],
        "csr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "bracket_geometry", "weld_penetration_type"],
        "description": "Fatigue-sensitive lower hopper knuckle intersection.",
    })

    # Inner bottom floor toe: Out_Girder / IBTM 교점
    floor_pt = _seg_intersection_bc(ship, "Out_Girder", "IBTM")
    hotspots.append({
        "hotspot_id": "inner_bottom_floor_toe",
        "rule_ref": CSR_RULE_REGISTRY_BC["inner_bottom_floor_toe"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_BC["inner_bottom_floor_toe"]["title"],
        "availability": "modeled" if floor_pt is not None else "not_modeled",
        "point_mm": floor_pt, "related_members": ["Out_Girder", "IBTM"],
        "csr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "floor_web_geometry", "scallop_type"],
        "description": "Fatigue-sensitive inner bottom / floor plate toe.",
    })

    # BHD lower stool (not modeled)
    hotspots.append({
        "hotspot_id": "bulkhead_lower_stool",
        "rule_ref": CSR_RULE_REGISTRY_BC["bulkhead_lower_stool"]["rule_ref"],
        "title": CSR_RULE_REGISTRY_BC["bulkhead_lower_stool"]["title"],
        "availability": "not_modeled", "point_mm": None, "related_members": [],
        "csr_evaluation_status": "not_modeled",
        "required_additional_inputs": ["bulkhead_type", "stool_geometry", "local_FE_geometry"],
        "description": "Transverse BHD lower stool not modeled in parametric section.",
    })

    counts = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
    for c in checks:
        s = c.get("status")
        if s in counts: counts[s] += 1
    overall = "fail" if counts["fail"] > 0 else ("partial" if counts["undetermined"] + counts["not_modeled"] > 0 else "pass")

    return {
        "standard": CSR_STANDARD_INFO,
        "ship_type": ship_data.get("ship_type", "bulk_carrier"),
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


# ================================
# Longitudinal Layout Helper
# ================================
def build_longitudinal_layout(L_m, HL_m, number_of_hold,
                               fwd_len_m, er_len_m, aft_len_m, hold_len_factor):
    if number_of_hold <= 0:
        raise ValueError("number_of_hold must be >= 1")
    hold_seg_m = hold_len_factor * HL_m
    model_L = aft_len_m + er_len_m + number_of_hold * hold_seg_m + fwd_len_m
    fwd_adj_m = fwd_len_m + (L_m - model_L)
    if fwd_adj_m < 0:
        fwd_adj_m = max(0.0, fwd_len_m)
    scale = 1000.0
    segs = []; bulkheads = []; x = 0.0
    x0 = x; x1 = x0 + aft_len_m * scale
    segs.append({'name': 'AFT', 'x0_mm': x0, 'x1_mm': x1}); bulkheads.append(x0); x = x1
    x0 = x; x1 = x0 + er_len_m * scale
    segs.append({'name': 'ER', 'x0_mm': x0, 'x1_mm': x1}); bulkheads.append(x0); x = x1
    for k in range(number_of_hold):
        idx_from_aft = number_of_hold - k
        x0 = x; x1 = x0 + hold_seg_m * scale
        segs.append({'name': f"HOLD {idx_from_aft}", 'x0_mm': x0, 'x1_mm': x1})
        bulkheads.append(x0); x = x1
    x0 = x; x1 = x0 + fwd_adj_m * scale
    segs.append({'name': 'FWD', 'x0_mm': x0, 'x1_mm': x1})
    bulkheads.append(x0); bulkheads.append(x1)
    return {
        'L_m': L_m, 'HL_m': HL_m, 'hold_seg_m': hold_seg_m,
        'fwd_len_m': fwd_adj_m, 'er_len_m': er_len_m, 'aft_len_m': aft_len_m,
        'segments': segs, 'bulkheads_mm': bulkheads,
    }


# ================================
# Segment display name helper
# ================================
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

# ================================
# Center Line Elevation Drawing
# ================================
def create_compartment_arrangement_drawing(dxf_path, layout, D_m, camber_m, DB_m,
                                         text_height=250, png_dir=None, png_dpi=220):
    os.makedirs(os.path.dirname(dxf_path), exist_ok=True)
    doc = ezdxf.new(setup=True); msp = doc.modelspace()
    for name, color in [("Hull", 3), ("Center", 8), ("Bulkhead", 2), ("Label", 1)]:
        layer = doc.layers.get(name) if name in doc.layers else doc.layers.add(name)
        layer.dxf.color = color
    segs = layout['segments']; x_end = segs[-1]['x1_mm']
    deck_z_mm = (D_m + camber_m) * 1000.0; db_z_mm = DB_m * 1000.0
    msp.add_line((0.0, 0.0), (0.0, deck_z_mm), dxfattribs={'layer': 'Center'})
    msp.add_line((0.0, 0.0), (x_end, 0.0), dxfattribs={'layer': 'Hull'})
    msp.add_line((0.0, deck_z_mm), (x_end, deck_z_mm), dxfattribs={'layer': 'Hull'})
    msp.add_line((0.0, db_z_mm), (x_end, db_z_mm), dxfattribs={'layer': 'Hull'})
    for bx in layout['bulkheads_mm']:
        msp.add_line((bx, 0.0), (bx, deck_z_mm), dxfattribs={'layer': 'Bulkhead'})
    for seg in segs:
        x0 = seg['x0_mm']; x1 = seg['x1_mm']; cx = 0.5 * (x0 + x1)
        display_name = _seg_display_name(seg['name'])
        t = msp.add_mtext(display_name, dxfattribs={'char_height': text_height * 3, 'layer': 'Label'})
        t.dxf.insert = (cx, deck_z_mm + 5 * text_height * 3); t.dxf.attachment_point = 5
        t2 = msp.add_mtext(f"{(x1-x0)/1000.0:.1f} m", dxfattribs={'char_height': text_height * 2.4, 'layer': 'Label'})
        t2.dxf.insert = (cx, -4 * text_height * 3); t2.dxf.attachment_point = 5
    t3 = msp.add_mtext(f"L = {layout['L_m']:.1f} m", dxfattribs={'char_height': text_height * 3, 'layer': 'Label'})
    t3.dxf.insert = (x_end * 0.5, deck_z_mm + 10 * text_height * 3); t3.dxf.attachment_point = 5
    doc.saveas(dxf_path)

    # --- Matplotlib PNG (colored zones, larger labels) ---
    png_path = None
    if png_dir is not None and _MAT_OK:
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


# ================================
# 3D Model Generator (BULKC)
# ================================
def create_compartment3d_dxf(dxf_path, ship, layout, text_height=250,
                         png_dir=None, png_dpi=220):
    """Compartment-based 3D wireframe model for BULKC. Returns (dxf_path, png_path)."""
    from math import sin, cos, pi
    os.makedirs(os.path.dirname(dxf_path), exist_ok=True)
    doc = ezdxf.new(setup=True); msp = doc.modelspace()

    def ensure_layer(name, color):
        layer = doc.layers.get(name) if name in doc.layers else doc.layers.add(name)
        layer.dxf.color = color

    ensure_layer("3D_OUTER_HULL",  3)
    ensure_layer("3D_INNER_HULL",  4)
    ensure_layer("3D_DB",          5)
    ensure_layer("3D_CARGO_HOLD",  2)
    ensure_layer("3D_BH_FACE",     1)
    ensure_layer("3D_Label",       7)

    segs = ship.seg_dict()  # {name: ((y1,z1),(y2,z2))} in mm

    MEMBER_LAYER = {
        "Bottom_Shell":  "3D_OUTER_HULL",
        "Side_Shell":    "3D_OUTER_HULL",
        "Upper_Deck_F":  "3D_OUTER_HULL",
        "Upper_Deck_S":  "3D_OUTER_HULL",
        "IBTM":          "3D_DB",
        "Hopper":        "3D_INNER_HULL",
        "Out_Girder":    "3D_DB",
        "Girder1":       "3D_DB",
        "Girder2":       "3D_DB",
        "Girder3":       "3D_DB",
        "TSWT_V":        "3D_INNER_HULL",
        "TSWT":          "3D_INNER_HULL",
    }
    OUTER_MEMBERS = {"Bottom_Shell", "Side_Shell", "Upper_Deck_F", "Upper_Deck_S"}
    EPSY = 1e-6

    B = ship.B; R = ship.R
    cy_mm = (B / 2.0 - R) * 1000.0; cz_mm = R * 1000.0; R_mm = R * 1000.0
    N_BILGE = 8
    bilge_pts = [(cy_mm + R_mm * cos(-0.5*pi + (i/N_BILGE)*0.5*pi),
                  cz_mm + R_mm * sin(-0.5*pi + (i/N_BILGE)*0.5*pi))
                 for i in range(N_BILGE + 1)]

    def add_panel(y1, z1, y2, z2, x0, x1, lyr):
        if abs(y1) < EPSY and abs(y2) < EPSY:
            msp.add_polyline3d([(x0,y1,z1),(x0,y2,z2),(x1,y2,z2),(x1,y1,z1),(x0,y1,z1)], dxfattribs={'layer': lyr})
            return
        msp.add_polyline3d([(x0,y1,z1),(x0,y2,z2),(x1,y2,z2),(x1,y1,z1),(x0,y1,z1)], dxfattribs={'layer': lyr})
        msp.add_polyline3d([(x0,-y1,z1),(x0,-y2,z2),(x1,-y2,z2),(x1,-y1,z1),(x0,-y1,z1)], dxfattribs={'layer': lyr})

    for seg_info in layout['segments']:
        x0 = seg_info['x0_mm']; x1 = seg_info['x1_mm']
        is_hold = seg_info['name'].startswith("HOLD")
        for nm, (p1, p2) in segs.items():
            y1, z1 = p1; y2, z2 = p2
            if not is_hold and nm not in OUTER_MEMBERS:
                continue
            add_panel(y1, z1, y2, z2, x0, x1, MEMBER_LAYER.get(nm, "3D_CARGO_HOLD"))
        for (py1,pz1),(py2,pz2) in zip(bilge_pts[:-1], bilge_pts[1:]):
            add_panel(py1, pz1, py2, pz2, x0, x1, "3D_OUTER_HULL")

    for bx in layout['bulkheads_mm']:
        is_hold_bx = any(
            seg['name'].startswith("HOLD") and (abs(bx - seg['x0_mm']) < 1 or abs(bx - seg['x1_mm']) < 1)
            for seg in layout['segments']
        )
        lyr = "3D_BH_FACE" if is_hold_bx else "3D_OUTER_HULL"
        for nm, (p1, p2) in segs.items():
            y1, z1 = p1; y2, z2 = p2
            # Non-HOLD boundaries: only outer shell contour
            if not is_hold_bx and nm not in OUTER_MEMBERS:
                continue
            msp.add_line((bx, y1, z1), (bx, y2, z2), dxfattribs={'layer': lyr})
            if abs(y1) > EPSY or abs(y2) > EPSY:
                msp.add_line((bx, -y1, z1), (bx, -y2, z2), dxfattribs={'layer': lyr})
        for (py1,pz1),(py2,pz2) in zip(bilge_pts[:-1], bilge_pts[1:]):
            msp.add_line((bx, py1, pz1), (bx, py2, pz2), dxfattribs={'layer': lyr})
            msp.add_line((bx, -py1, pz1), (bx, -py2, pz2), dxfattribs={'layer': lyr})

    deck_z_mm = ship.z_deck(0.0) * 1000.0
    dxf_label_h = max(1, text_height // 3)
    for seg_info in layout['segments']:
        cx = 0.5 * (seg_info['x0_mm'] + seg_info['x1_mm'])
        display = _seg_display_name(seg_info['name'])
        t = msp.add_mtext(display, dxfattribs={'char_height': dxf_label_h, 'layer': '3D_Label'})
        t.dxf.insert = (cx, 0.0, deck_z_mm + 5 * dxf_label_h); t.dxf.attachment_point = 5

    doc.saveas(dxf_path)

    # Layer → (hex color, alpha, linewidth)
    _3D_LAYER_STYLE = {
        "3D_OUTER_HULL":  ('#666666', 0.55, 0.5),
        "3D_INNER_HULL":  ('#00bbcc', 0.70, 0.5),   # hopper/TSWT — inner structure
        "3D_DB":          ('#2255ff', 0.70, 0.5),   # ballast (double bottom) — blue
        "3D_CARGO_HOLD":  ('#dd2222', 0.65, 0.5),   # cargo hold — red
        "3D_BH_FACE":     ('#ff8800', 0.75, 0.6),   # bulkhead face — orange
    }

    # --- Isometric PNG (equal-scale axes, layer-colored, labeled) ---
    png_path = None
    if png_dir is not None and _MAT_OK:
        os.makedirs(png_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(dxf_path))[0]
        png_path = os.path.join(png_dir, base + ".png")
        try:
            all_xs, all_ys, all_zs = [], [], []
            plot_lines = []  # (xs, ys, zs, layer_name)
            for entity in msp:
                try:
                    lyr_name = entity.dxf.layer
                    if lyr_name == '3D_Label':
                        continue
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
                    plot_lines.append((xs, ys, zs, lyr_name))
                except Exception:
                    pass

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
            ax.set_box_aspect((rx, ry, rz))

            for xs, ys, zs, lyr_name in plot_lines:
                color, alpha, lw = _3D_LAYER_STYLE.get(lyr_name, ('#444444', 0.5, 0.4))
                ax.plot(xs, ys, zs, color=color, linewidth=lw, alpha=alpha)

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


# ---------- helpers ----------
EPS = 1e-9
EXPORT_INCLUDE_MEMBER_BBOX = False

def fmt_token(val, nd=1):
    """파일명 안전용 토큰 (소수점 -> 'p')"""
    s = f"{val:.{nd}f}"
    # 소수점이 있을 때(nd>0)만 뒷자리 0과 점을 제거
    if nd > 0 and '.' in s:
         s = s.rstrip('0').rstrip('.')
    return s.replace('.', 'p')

def build_filename(base_dir, L,B,D,C, DB,R,
                   GY, OG, TSWT_EXT,
                   DS, STRCLR,
                   S1,S2):
    name=(f"BULKC_L{fmt_token(L, 0)}_"
          f"B{fmt_token(B)}_D{fmt_token(D)}_"
          f"C{fmt_token(C)}_DB{fmt_token(DB)}_R{fmt_token(R)}_"
          f"GY{fmt_token(GY)}_OG{fmt_token(OG)}_TSWTE{fmt_token(TSWT_EXT)}_"
          f"DS{fmt_token(DS)}_SC{fmt_token(STRCLR)}_"
          f"S1{fmt_token(S1)}_S2{fmt_token(S2)}.dxf")
    return os.path.join(base_dir, name)

def quantize_to_step(x, start, step):
    return round(round((x-start)/step)*step + start, 10)

def lhs_samples(N, specs, seed=None):
    rng = random.Random(seed)
    per = []
    for sp in specs:
        lo, hi = sp['min'], sp['max']
        w = (hi - lo) / max(N, 1)
        vals = [lo + i * w + rng.random() * w for i in range(N)]
        rng.shuffle(vals)
        if sp.get('step') is not None:
            vals = [quantize_to_step(v, sp['min'], sp['step']) for v in vals]
        if sp['type'] == 'int':
            vals = [int(round(v)) for v in vals]
        per.append(vals)

    out = []
    for i in range(N):
        d = {}
        for j, sp in enumerate(specs):
            d[sp['name']] = per[j][i]
        out.append(d)
    return out

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
    with open(path,'w',encoding='utf-8') as f: json.dump(_round_floats(obj,3), f, ensure_ascii=False, indent=2)

def append_csv(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    first=not os.path.exists(path)
    with open(path,'a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=header)
        if first: w.writeheader()
        w.writerow(row)

def clean_multiline_label(label: str) -> str:
    return label.replace("\\P", " ").replace("  ", " ").strip()

def expand_abbrev(token: str) -> str:
    return clean_multiline_label(token or "")

# =========================================
#                 Geometry
# =========================================
class BULKC:
    """
    우현 반쪽(y>=0).
    - Cargo Tank 없음
    - Cargo Hold 구획 생성
    - Upper_Deck: CL~3.5 m 수평 + 이후 camber
    """
    def __init__(self, L,B,D, DB,R, camber,
                 y_girder, y_og_ratio, tswt_ext_deg,
                 ds_from_side, str_clear,
                 s1_ratio, s2_ratio,
                 deck_flat_y=3.5):
        self.L=L; self.B=B; self.D=D
        self.DB=DB; self.R=R; self.camber=camber
        self.y_girder=y_girder
        self.y_og = y_og_ratio*(B/2)
        self.tswt_ext = tswt_ext_deg
        self.DS=ds_from_side
        self.str_clear=str_clear
        self.s1_ratio=s1_ratio; self.s2_ratio=s2_ratio
        self.deck_flat_y=deck_flat_y
        self._build_hull()

    # piecewise deck camber
    def z_deck(self, y):
        y_f = self.deck_flat_y
        z_center = self.D + self.camber
        if y <= y_f:
            return z_center
        y2 = self.B/2; z2 = self.D
        if abs(y2 - y_f) < 1e-9:
            return z2
        m = (z2 - z_center)/(y2 - y_f)
        return z_center + m*(y - y_f)

    def _build_hull(self):
        L, B, D, DB, R = self.L, self.B, self.D, self.DB, self.R

        # shells
        self.m_btm = [[L * 500, L * 500], [0, (B / 2 - R) * 1000], [0, 0]]
        self.m_side = [[L * 500, L * 500], [B * 500, B * 500], [R * 1000, D * 1000]]

        # deck flat + slope
        y_f = self.deck_flat_y
        z_f = self.z_deck(0)
        z_at_side = self.z_deck(B / 2)
        self.m_deck_flat = [[L * 500, L * 500], [0, y_f * 1000], [z_f * 1000, z_f * 1000]]
        self.m_deck_slope = [[L * 500, L * 500], [y_f * 1000, B * 500], [z_f * 1000, z_at_side * 1000]]

        # inner bottom & hopper
        import math
        self.m_ibtm = [[L * 500, L * 500], [0, self.y_og * 1000], [DB * 1000, DB * 1000]]
        hop_ang = math.radians(42.0)
        z_hopp_side = DB + math.tan(hop_ang) * (B / 2 - self.y_og)
        z_hopp_side = min(D - 0.8, max(R + 0.6, z_hopp_side))
        self.m_hopp = [[L * 500, L * 500], [self.y_og * 1000, (B / 2) * 1000], [DB * 1000, z_hopp_side * 1000]]

        # girders
        self.m_outg = [[L * 500, L * 500], [self.y_og * 1000, self.y_og * 1000], [0, DB * 1000]]

        # --- fixed: Girder1 at y = 2200 mm from CL ---
        y_g1 = 2200.0  # mm
        self.m_gird1 = [[L * 500, L * 500], [y_g1, y_g1], [0, DB * 1000]]

        # --- Girder2, Girder3 equally between Girder1 and Out_Girder ---
        y_outg = self.y_og * 1000.0
        if y_outg <= y_g1 + 10.0:
            y_outg = y_g1 + 1200.0
        span = y_outg - y_g1
        y_g2 = y_g1 + span / 3.0
        y_g3 = y_g1 + 2.0 * span / 3.0
        self.m_gird2 = [[L * 500, L * 500], [y_g2, y_g2], [0, DB * 1000]]
        self.m_gird3 = [[L * 500, L * 500], [y_g3, y_g3], [0, DB * 1000]]

        # TSWT vertical + slope
        y_tsy = (B / 2) / 2.0
        z_top = self.z_deck(y_tsy)
        z_kink = z_top - 0.7
        import math as _m
        phi_req = radians(90.0 - self.tswt_ext)
        phi = max(min(phi_req, _m.radians(-30.0)), _m.radians(-50.0))
        z_ts_side = z_kink + _m.tan(phi) * (B / 2 - y_tsy)
        self.tswt_vert = [[L * 500, L * 500], [y_tsy * 1000, y_tsy * 1000], [z_top * 1000, z_kink * 1000]]
        self.tswt_slope = [[L * 500, L * 500], [y_tsy * 1000, (B / 2) * 1000], [z_kink * 1000, z_ts_side * 1000]]

        # no stringers
        self.str1 = None
        self.str2 = None

    def seg_dict(self):

        def seg(m):
            if m is None: return None
            return ((float(m[1][0]), float(m[2][0])),
                    (float(m[1][1]), float(m[2][1])))
        M = {}
        M["Upper_Deck_F"] = seg(self.m_deck_flat)
        M["Upper_Deck_S"] = seg(self.m_deck_slope)
        M["Bottom_Shell"] = seg(self.m_btm)
        M["Side_Shell"] = seg(self.m_side)
        M["IBTM"] = seg(self.m_ibtm)
        M["Hopper"] = seg(self.m_hopp)
        M["Out_Girder"] = seg(self.m_outg)
        M["Girder1"] = seg(self.m_gird1)
        M["Girder2"] = seg(self.m_gird2)
        M["Girder3"] = seg(self.m_gird3)
        M["TSWT_V"] = seg(self.tswt_vert)
        M["TSWT"] = seg(self.tswt_slope)
        return {k: v for k, v in M.items() if v is not None}


# ============================================================
# Stiffener configuration:  member → (type, flange_half_mm)
#   "FB"  = Flat Bar (F.B)       — web line only, no flange
#   "IA"  = Inverted Angle (I.A) — web + one-side flange (L-shape)
#   "T"   = Built-up T-bar F.B(T)— web + both-side flanges (T-shape)
# ============================================================
_STF_CFG = {
    # (stf_type, flange_half_mm, web_h_mm)  ← web_h from _SCANTLING_TABLE
    "Upper_Deck_F": ("T",  75, 350),  # 350 x 12 + 150 x 20 F.B(T)
    "Upper_Deck_S": ("IA", 90, 250),  # 250 x 90 x 10/15 I.A
    "Bottom_Shell": ("T",  65, 350),  # 350 x 12 + 130 x 18 F.B(T)
    "Side_Shell":   ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)
    "IBTM":         ("T",  75, 380),  # 380 x 14 + 150 x 20 F.B(T)
    "Hopper":       ("IA", 90, 200),  # 200 x 90 x 10/14 I.A
    "TSWT":         ("IA", 90, 200),  # 200 x 90 x 10/14 I.A
    "TSWT_V":       ("IA", 90, 200),  # same as TSWT
    "Out_Girder":   ("FB",  0, 200),  # 200 x 12 F.B
    "Girder1":      ("FB",  0, 150),  # 150 x 10 F.B
    "Girder2":      ("FB",  0, 150),  # 150 x 10 F.B
    "Girder3":      ("FB",  0, 150),  # 150 x 10 F.B
}

_STF_TYPE_LEGEND = {
    "F.B":    "Flat Bar — web only, no flange",
    "I.A":    "Inverted Angle — web + one-side flange (L-shape)",
    "F.B(T)": "Built-up T-bar — web + both-side flanges (T-shape)",
}

_SCANTLING_TABLE = [
    ("MEMBER",          "PLATE (mm)", "STIFFENER"),
    ("Upper Deck (F)",  "14.0",       "350 x 12 + 150 x 20 F.B(T)"),
    ("Upper Deck (S)",  "12.0",       "250 x 90 x 10/15 I.A"),
    ("Bottom Shell",    "17.0",       "350 x 12 + 130 x 18 F.B(T)"),
    ("Side Shell",      "15.5",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Inner Bottom",    "16.0",       "380 x 14 + 150 x 20 F.B(T)"),
    ("Hopper",          "13.0",       "200 x 90 x 10/14 I.A"),
    ("TSWT",            "12.0",       "200 x 90 x 10/14 I.A"),
    ("Out Girder",      "12.0",       "200 x 12 F.B"),
    ("Girder",          "11.0",       "150 x 10 F.B"),
]

# =========================================
#               Exporter (drawing preserved; +metadata collection)
# =========================================
class DXFExporter:
    def __init__(self, ship, text_height=250, offset=300,
                 stf_min=700, stf_max=1000, stf_target=850, stf_len=400, edge_clear=10,
                 label_offset=300, label_dir=None, label_flip=None,
                 # Side frame params (mm)
                 sf_offset_min=600, sf_offset_max=700,
                 top_drop_mm=1100, bot_rise_mm=1100,
                 top_run_mm=1400, bot_run_mm=1700,
                 # Local deck girder (mm)
                 deck_local_girder_len=1000,
                 # NEW: LNGC-style meta (BULKC용)
                 hold_length_m=None,
                 hold_len_factor=1.0,
                 hold_vol_factor=0.7,
                 number_of_hold=9):
        self.s=ship; self.text_height=text_height; self.offset=offset
        self.stf_min=stf_min; self.stf_max=stf_max; self.stf_target=stf_target
        self.stf_len=stf_len; self.edge_clear=edge_clear
        self.doc=ezdxf.new(setup=True); self.msp=self.doc.modelspace()
        self.label_offset = label_offset

        # side frame
        self.sf_offset_min = sf_offset_min
        self.sf_offset_max = sf_offset_max
        self.top_drop_mm = top_drop_mm
        self.bot_rise_mm = bot_rise_mm
        self.top_run_mm = top_run_mm
        self.bot_run_mm = bot_run_mm

        # deck local girder len
        self.deck_local_girder_len = deck_local_girder_len

        # NEW: LNGC-style meta fields
        self.hold_length_m = hold_length_m
        self.hold_len_factor = float(hold_len_factor)
        self.hold_vol_factor = float(hold_vol_factor)
        self.number_of_hold = int(number_of_hold)

        # for display
        self.bilge_bottom_end=None; self.bilge_side_start=None

        # Side Frame kink 좌표 저장 (라벨링)
        self.sf_K_top = None
        self.sf_K_btm = None

        # metadata collectors
        self._labels = []
        self._stf_stats = {}
        self._compartment_data = []
        self._intersections = []

        default_dir = {
            "Upper_Deck_F": (0.0, +1.0), "Upper_Deck_S": (0.0, +1.0),
            "Bottom_Shell": (0.0, -1.0), "IBTM": (0.0, +1.0),
            "Side_Shell": (+1.0, 0.0),   # outboard
            "Out_Girder": (+1.0, 0.0),
            "Girder1": (+1.0, 0.0),
            "Girder2": (+1.0, 0.0),
            "Girder3": (+1.0, 0.0),
            "TSWT": (0.0, -1.0),
            "Side_Frame": (-1.0, 0.0),
        }
        self.LABEL_DIR = dict(default_dir if label_dir is None else label_dir)
        if label_flip:
            for k in label_flip:
                if k in self.LABEL_DIR:
                    ny, nz = self.LABEL_DIR[k]; self.LABEL_DIR[k] = (-ny, -nz)

    # ---------- basic drawing helpers (visuals unchanged) ----------
    def _label_on_member(self, key, y0, z0, y1, z1, layer="Label", offset=None, display=None):
        from math import atan2, degrees, hypot
        if offset is None: offset = self.label_offset
        dy, dz = (y1 - y0), (z1 - z0)
        L = hypot(dy, dz)
        if L < 1e-9: return
        ang_deg = degrees(atan2(dz, dy))
        nd = self.LABEL_DIR.get(key)
        if nd is None:
            nA = (-dz / L, dy / L); nB = (dz / L, -dy / L); nx, nz = (nA if nA[1]>=nB[1] else nB)
        else:
            ny, nz = nd; nlen = hypot(ny, nz); nx, nz = (ny/nlen, nz/nlen) if nlen>1e-9 else (0.0,0.0)
        cx, cz = (y0+y1)/2.0, (z0+z1)/2.0
        px, pz = cx + nx*offset, cz + nz*offset
        label_text = display if display is not None else key.replace("_"," ")
        txt = self.msp.add_mtext(label_text, dxfattribs={"char_height": self.text_height, "layer": layer})
        txt.dxf.insert=(px,pz); txt.dxf.attachment_point=5; txt.dxf.rotation=ang_deg
        self._labels.append({'name': label_text, 'pos': (float(px), float(pz)),
                             'rotation_deg': float(ang_deg), 'layer': layer})

    # 라벨 메타 기록 헬퍼
    def _add_label_record(self, text, pos, rot_deg, layer):
        self._labels.append({
            'name': text,
            'pos': (float(pos[0]), float(pos[1])),
            'rotation_deg': float(rot_deg),
            'layer': layer
        })

    def _text(self, s, pos, rot=0, layer="Label"):
        t=self.msp.add_mtext(s, dxfattribs={'char_height':self.text_height,'layer':layer})
        t.set_location(pos, rotation=rot)
        self._labels.append({'name': s, 'pos': (float(pos[0]), float(pos[1])),
                             'rotation_deg': float(rot), 'layer': layer})

    def draw_layers(self):
        def L(n, c):
            if n not in self.doc.layers: self.doc.layers.add(n).dxf.color = c
        L("Members", 3); L("Label", 1); L("Compartment", 6); L("Bilge", 3)
        L("Stiffeners (Longi)", 4)   # cyan — longitudinal stiffeners
        L("Stiffeners (Trans)", 30)  # orange — transverse members / deck indicators
        L("Center", 8); L("SideFrame", 30)  # orange — side frames
        L("Scantling", 252)          # dark gray — scantling table
        if "DeckGirder" in self.doc.layers:
            del self.doc.layers["DeckGirder"]

    def draw_centerline(self):
        top=max(self.s.m_deck_flat[2]+self.s.m_deck_slope[2])+500
        ln=self.msp.add_line((0,0),(0,top), dxfattribs={'layer':'Center'})
        try: ln.dxf.linetype="CENTER"; ln.dxf.ltscale=200
        except Exception: pass
        self._text("C.L.", (-500, top+300), rot=90)

    def draw_hull(self):
        parts = [
            ("Upper_Deck_F", "m_deck_flat"), ("Upper_Deck_S", "m_deck_slope"),
            ("Bottom_Shell", "m_btm"), ("Side_Shell", "m_side"),
            ("IBTM", "m_ibtm"), ("Hopper", "m_hopp"),
            ("Out_Girder", "m_outg"),
            ("Girder1", "m_gird1"), ("Girder2", "m_gird2"), ("Girder3", "m_gird3"),
            ("TSWT_V", "tswt_vert"), ("TSWT", "tswt_slope"),
        ]
        display_map = {
            "Upper_Deck_F": "Upper Deck(Flat)",
            "Upper_Deck_S": "Upper Deck(Camber)",
            "Girder1": "Girder1", "Girder2": "Girder2", "Girder3": "Girder3",
        }

        for label, attr in parts:
            m = getattr(self.s, attr, None)
            if m is None: continue
            y, z = m[1], m[2]
            self.msp.add_line((y[0], z[0]), (y[1], z[1]), dxfattribs={'layer': 'Members'})
            if label != "TSWT_V":
                self._label_on_member(
                    label, y[0], z[0], y[1], z[1],
                    layer="Label",
                    display=display_map.get(label)
                )

        # Bilge arc (for display)
        R = self.s.R*1000; B2 = self.s.B*1000/2
        bottom_end = (self.s.m_btm[1][1], self.s.m_btm[2][1])
        side_start = (self.s.m_side[1][0], self.s.m_side[2][0])
        self.bilge_bottom_end = bottom_end
        self.bilge_side_start = side_start

        from math import atan2, cos, sin
        cy = B2 - R; cz = R
        a1 = atan2(bottom_end[1] - cz, bottom_end[0] - cy)
        a2 = atan2(side_start[1] - cz, side_start[0] - cy)
        self.msp.add_arc((cy, cz), R, a1 * 180 / pi, a2 * 180 / pi, dxfattribs={'layer': 'Bilge'})

        theta_mid = (a1 + a2) / 2
        lx = cy + (R + 300) * cos(theta_mid)
        lz = cz + (R + 300) * sin(theta_mid)
        lab = self.msp.add_mtext("Bilge", dxfattribs={'char_height': self.text_height, 'layer': 'Label'})
        lab.dxf.insert=(lx,lz); lab.dxf.attachment_point=4; lab.dxf.rotation=0
        self._labels.append({'name':'Bilge','pos':(float(lx),float(lz)),'rotation_deg':0.0,'layer':'Label'})

        # Side Frame 그리기
        self._draw_side_frame()

        # Side Frame 라벨
        if self.sf_K_top and self.sf_K_btm:
            self._label_on_member(
                "Side_Frame",
                self.sf_K_top[0], self.sf_K_top[1],
                self.sf_K_btm[0], self.sf_K_btm[1],
                layer="Label",
                display="Side Frame"
            )

    # --- utils ---
    @staticmethod
    def _seg_intersection(p1,p2,p3,p4, tol=1e-7):
        x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
        den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        if abs(den) < 1e-12: return None
        px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
        py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
        P=(px,py)
        def on_seg(a,b,p):
            ax,ay=a; bx,by=b
            L=(bx-ax)**2+(by-ay)**2
            if L < 1e-12: return False
            t=((p[0]-ax)*(bx-ax)+(p[1]-ay)*(by-ay))/L
            return -tol<=t<=1+tol and \
                   min(ax,bx)-tol<=p[0]<=max(ax,bx)+tol and \
                   min(ay,by)-tol<=p[1]<=max(ay,by)+tol
        return P if on_seg(p1,p2,P) and on_seg(p3,p4,P) else None

    @staticmethod
    def _poly_centroid(pts):
        A=0.0; Cx=0.0; Cy=0.0; n=len(pts)
        for i in range(n):
            x1,y1=pts[i]; x2,y2=pts[(i+1)%n]
            cross = x1*y2 - x2*y1
            A += cross; Cx += (x1+x2)*cross; Cy += (y1+y2)*cross
        A *= 0.5
        if abs(A) < 1e-9:
            mx=sum(p[0] for p in pts)/n; my=sum(p[1] for p in pts)/n
            return (mx,my)
        return (Cx/(6*A), Cy/(6*A))

    @staticmethod
    def _poly_area_perimeter(verts):
        n=len(verts)
        if n<3: return 0.0,0.0
        area2=0.0; per=0.0
        for i in range(n):
            x1,y1=verts[i]; x2,y2=verts[(i+1)%n]
            area2 += x1*y2 - x2*y1
            per   += hypot(x2-x1, y2-y1)
        return abs(area2)*0.5, per

    def _as_seg(self, mat):
        return ((float(mat[1][0]), float(mat[2][0])),
                (float(mat[1][1]), float(mat[2][1])))

    # ---- Side Frame (최종 사양) ----
    def _draw_side_frame(self):
        """
        - Side Shell에서 0.6~0.7 m 인보드 위치의 수직부
        - 상부: K_top(z = (TSWT∩SS).z - 1100) → (Q_top에서 TSWT 인보드방향 1400)
        - 하부: K_btm(z = (Hopper∩SS).z + 1100) → (Q_btm에서 Hopper 인보드방향 1700)
        """
        p_ss  = self._as_seg(self.s.m_side)       # Side Shell (vertical)
        p_ts  = self._as_seg(self.s.tswt_slope)   # TSWT slope
        p_hp  = self._as_seg(self.s.m_hopp)       # Hopper

        # clamp DS to 600~700 mm
        ds_user = self.s.DS * 1000.0
        DS = min(max(ds_user, self.sf_offset_min), self.sf_offset_max)
        y_side = p_ss[0][0]
        y_off  = y_side - DS  # inboard

        # junctions with SideShell
        Q_top = self._seg_intersection(*p_ts, *p_ss)
        Q_btm = self._seg_intersection(*p_hp, *p_ss)
        if not Q_top or not Q_btm:
            return

        z_top_target = Q_top[1] - self.top_drop_mm
        z_btm_target = Q_btm[1] + self.bot_rise_mm
        K_top = (y_off, z_top_target)
        K_btm = (y_off, z_btm_target)

        # save for labeling
        self.sf_K_top = K_top
        self.sf_K_btm = K_btm

        # unit vector in "inboard" direction (y decreasing)
        def inboard_dir(seg):
            (y1,z1),(y2,z2) = seg
            dy, dz = (y2 - y1, z2 - z1)
            L = hypot(dy, dz)
            if L < 1e-9:
                return (-1.0, 0.0)
            uy, uz = dy / L, dz / L
            return (uy, uz) if uy < 0 else (-uy, -uz)

        u_ts = inboard_dir(p_ts)
        u_hp = inboard_dir(p_hp)

        # points along TSWT/Hopper from the SideShell junctions
        P_top = (Q_top[0] + u_ts[0]*self.top_run_mm, Q_top[1] + u_ts[1]*self.top_run_mm)
        P_btm = (Q_btm[0] + u_hp[0]*self.bot_run_mm, Q_btm[1] + u_hp[1]*self.bot_run_mm)

        # draw side frame: top leg, vertical, bottom leg
        self.msp.add_polyline2d([K_top, P_top],  dxfattribs={'layer':'SideFrame'})
        self.msp.add_polyline2d([K_top, K_btm],  dxfattribs={'layer':'SideFrame'})
        self.msp.add_polyline2d([K_btm, P_btm],  dxfattribs={'layer':'SideFrame'})

    # ---- Title & Specs  ----
    def draw_title_and_specs(self, title: str = "ORDINARY SECTION (STBD)"):
        # CL에서의 상갑판 z(mm) 추출 (BULKC: m_deck_flat 사용)
        try:
            z_deck_cl = float(self.s.m_deck_flat[2][0])  # at CL, mm
        except Exception:
            z_deck_cl = 0.0

        # 제목을 갑판 위로 띄움 (Tanker 예시와 유사)
        base_z = z_deck_cl + 5000.0
        center_y = 0.0  # 중심선(C.L.) 기준 정렬

        def put_line(text, dy_mult, size_mult=1.0):
            char_h = self.text_height * size_mult
            ty = base_z - self.text_height * dy_mult
            t = self.msp.add_mtext(text, dxfattribs={'char_height': char_h, 'layer': 'Label'})
            t.dxf.insert = (center_y, ty)
            t.dxf.attachment_point = 5  # middle center
            t.dxf.rotation = 0
            self._add_label_record(text, (center_y, ty), 0.0, "Label")

        # Title
        put_line(title, dy_mult=-0.0, size_mult=1.5)

        # BREADTH, DEPTH only — section drawing excludes longitudinal info
        # (NUMBER OF HOLD / HOLD LENGTH / SHIP LENGTH belong to compartment view).
        put_line(f"BREADTH = {float(self.s.B):.1f} m", dy_mult=2.6)
        put_line(f"DEPTH = {float(self.s.D):.1f} m",   dy_mult=4.4)


    # ---- Compartments ----
    def draw_compartments(self):
        if "Compartment" not in self.doc.layers:
            self.doc.layers.add("Compartment").dxf.color = 6

        S = self.s.seg_dict()
        deck_top_cl = self.s.z_deck(0) * 1000.0
        S["CL"] = ((0.0, 0.0), (0.0, deck_top_cl))
        if self.bilge_bottom_end and self.bilge_side_start:
            S["Bilge"] = (self.bilge_bottom_end, self.bilge_side_start)

        def label_poly(name, edges):
            verts=[]
            for A,B in edges:
                if A not in S or B not in S: return
                p1,p2=S[A]; q1,q2=S[B]
                ip=self._seg_intersection(p1,p2,q1,q2)
                if ip is None: return
                verts.append((float(ip[0]), float(ip[1])))
            if len(verts)<3: return
            cx,cy = self._poly_centroid(verts)
            # Cargo Hold label: placed later in draw_scantling_table (above table)
            if "Cargo Hold" not in name:
                t = self.msp.add_mtext(name, dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
                t.dxf.insert=(cx,cy); t.dxf.attachment_point=5; t.dxf.rotation=0
            area_mm2, per_all = self._poly_area_perimeter(verts)
            per_excl_cl = 0.0
            n=len(verts)
            for i in range(n):
                j=(i+1)%n
                v1=verts[i]; v2=verts[j]
                # exclude edges that are on CL by name adjacency (approx via edges list)
                ea, eb = edges[i]
                if "CL" in (ea, eb) and "CL" in edges[j]:
                    continue
                per_excl_cl += hypot(v2[0]-v1[0], v2[1]-v1[1])

            meta = {
                "raw_label": name,
                "clean_label": clean_multiline_label(name),
                "centroid_mm": (round(cx,3), round(cy,3)),
                "vertices_mm": [(round(v[0],3), round(v[1],3)) for v in verts],
                "area_mm2": round(area_mm2,3),
                "area_m2": round(area_mm2/1e6,6),
                "perimeter_mm_excl_CL": round(per_excl_cl,3),
                "edges_used": edges,
            }
            self._compartment_data.append(meta)

        # ---- compartments
        label_poly("Cargo Hold", [
            ("Upper_Deck_F","Upper_Deck_S"),
            ("Upper_Deck_S","TSWT_V"),
            ("TSWT_V","TSWT"),
            ("TSWT","Side_Shell"),
            ("Side_Shell","Hopper"),
            ("Hopper","Out_Girder"),
            ("Out_Girder","IBTM"),
            ("IBTM","CL"),
            ("CL","Upper_Deck_F"),
        ])

        label_poly("Ballast tank 1\\P(T.S.W.B.T.)", [
            ("Upper_Deck_S","TSWT_V"),
            ("TSWT_V","TSWT"),
            ("TSWT","Side_Shell"),
            ("Side_Shell","Upper_Deck_S"),
        ])

        label_poly("Ballast tank 2\\P(D.B.W.B.T.)", [
            ("Hopper","Side_Shell"),
            ("Side_Shell","Bilge"),
            ("Bilge","Bottom_Shell"),
            ("Bottom_Shell","Out_Girder"),
            ("Out_Girder","Hopper"),
        ])

        label_poly("Ballast\\Ptank 3\\P(D.B.W.B.T.)", [
            ("IBTM","Out_Girder"),
            ("Out_Girder","Bottom_Shell"),
            ("Bottom_Shell","Girder3"),
            ("Girder3","IBTM"),
        ])

        label_poly("Ballast\\Ptank 4\\P(D.B.W.B.T.)", [
            ("IBTM","Girder3"),
            ("Girder3","Bottom_Shell"),
            ("Bottom_Shell","Girder2"),
            ("Girder2","IBTM"),
        ])

        label_poly("Ballast\\Ptank 5\\P(D.B.W.B.T.)", [
            ("IBTM","Girder2"),
            ("Girder2","Bottom_Shell"),
            ("Bottom_Shell","Girder1"),
            ("Girder1","IBTM"),
        ])

        label_poly("Pipe\\Pduct", [
            ("IBTM","Girder1"),
            ("Girder1","Bottom_Shell"),
            ("Bottom_Shell","CL"),
            ("CL","IBTM"),
        ])

        # collect intersections
        keys = sorted(S.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1,k2 = keys[i], keys[j]
                p1,p2 = S[k1]; q1,q2 = S[k2]
                ip = self._seg_intersection(p1,p2,q1,q2)
                if ip is not None:
                    self._intersections.append({"a": k1, "b": k2, "point_mm": (round(ip[0],3), round(ip[1],3))})

    # ---- Typed stiffeners (FB / I.A / F.B(T)) + Deck local girders ----
    def draw_stiffeners(self):
        if "Stiffeners (Longi)" not in self.doc.layers:
            self.doc.layers.add("Stiffeners (Longi)").dxf.color = 4

        def seg(m):
            return ((float(m[1][0]), float(m[2][0])),
                    (float(m[1][1]), float(m[2][1])))

        segs_all = {}
        for name, attr in [
            ("Upper_Deck_F", "m_deck_flat"),
            ("Upper_Deck_S", "m_deck_slope"),
            ("Side_Shell", "m_side"),
            ("Bottom_Shell", "m_btm"), ("IBTM", "m_ibtm"),
            ("Hopper", "m_hopp"),
            ("Out_Girder", "m_outg"),
            ("Girder1", "m_gird1"), ("Girder2", "m_gird2"), ("Girder3", "m_gird3"),
            ("TSWT", "tswt_slope"), ("TSWT_V", "tswt_vert"),
        ]:
            m = getattr(self.s, attr, None)
            if m is not None: segs_all[name] = seg(m)

        # Bilge chord(분할용 교차선)
        if self.bilge_bottom_end and self.bilge_side_start:
            segs_all["Bilge"] = (self.bilge_bottom_end, self.bilge_side_start)

        from math import hypot
        LEN_DEFAULT = self.stf_len
        LEN_SHORT = 150.0

        def dir_len(p1, p2):
            dy=p2[0]-p1[0]; dz=p2[1]-p1[1]; L=hypot(dy,dz)
            return (dy/L if L>1e-9 else 0.0, dz/L if L>1e-9 else 0.0, L)
        def proj_t(p1, p2, p):
            uy, uz, L = dir_len(p1,p2)
            if L<1e-9: return 0.0
            return (p[0]-p1[0])*uy + (p[1]-p1[1])*uz
        def intersect(p1, p2, q1, q2):
            return self._seg_intersection(p1,p2,q1,q2)
        def split(name):
            p1,p2 = segs_all[name]
            uy,uz,L = dir_len(p1,p2)
            if L<1e-9: return []
            ts=[0.0, L]
            for other,(q1,q2) in segs_all.items():
                if other==name: continue
                ip=intersect(p1,p2,q1,q2)
                if ip is None: continue
                t = proj_t(p1,p2,ip); tq=proj_t(q1,q2,ip); Lq=dir_len(q1,q2)[2]
                if -1e-6<=t<=L+1e-6 and -1e-6<=tq<=Lq+1e-6:
                    ts.append(max(0.0, min(L, t)))
            ts=sorted(set(round(t,6) for t in ts))
            out=[]
            for a,b in zip(ts[:-1], ts[1:]):
                if b-a>1e-3:
                    s=(p1[0]+uy*a, p1[1]+uz*a)
                    e=(p1[0]+uy*b, p1[1]+uz*b)
                    out.append((s,e))
            return out
        def choose_spacing(L):
            eff=L-2*self.edge_clear
            if eff<self.stf_min: return 0,None
            nmax=int(eff//self.stf_min); best=(0,None)
            for n in range(1, nmax+1):
                s = eff/(n+1)
                if self.stf_min<=s<=self.stf_max and (best[0]==0 or abs(s-self.stf_target)<abs((best[1] or s)-self.stf_target)):
                    best=(n,s)
            return best

        def draw_stiffener_shape(base, nvec, along_vec, stf_type, web_len, flange_half, layer):
            """
            Draw stiffener cross-section as DXF lines:
              FB      : single web line
              IA (I.A): web + one-side flange (Inverted Angle / L-shape)
              T  (F.B(T)): web + both-side flange (T-shape)
            """
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

        # 법선 방향
        def normal_of(name, p1, p2):
            uy, uz, _ = dir_len(p1, p2)
            nA, nB = (-uz, uy), (uz, -uy)
            if name in ("Upper_Deck_F", "Upper_Deck_S"):
                return nA if nA[1] < nB[1] else nB  # -z
            if name == "Bottom_Shell": return (0.0, +1.0)
            if name in ("IBTM",): return (0.0, -1.0)
            if name == "Side_Shell": return (-1.0, 0.0)  # inboard
            if name == "Out_Girder": return (-1.0, 0.0)
            if name in ("Girder1", "Girder2", "Girder3"): return (-1.0, 0.0)
            if name == "TSWT": return nA if nA[1] > nB[1] else nB
            if name == "Hopper": return nA if nA[0] > nB[0] else nB
            return nA if nA[1] < nB[1] else nB

        # 사이드쉘 분절 경계 z 값(상/하 BT 구간)
        def z_at_side(seg):
            p_ss = segs_all["Side_Shell"]
            ip = intersect(*seg, *p_ss)
            return None if ip is None else ip[1]
        z_ts_side = z_at_side(segs_all["TSWT"])        if "TSWT" in segs_all else None
        z_hp_side = z_at_side(segs_all["Hopper"])      if "Hopper" in segs_all else None
        z_deck_side = segs_all["Upper_Deck_S"][1][1]   if "Upper_Deck_S" in segs_all else None
        z_bilge_side = segs_all["Side_Shell"][0][1]

        # TSWT_V의 y (갑판 Camber 구간 중 Cargo Hold 경계)
        y_tsy_mm = (self.s.B/2)/2.0 * 1000.0

        # ---- Junction interference exclusion --------------------------------
        CLEAR_ZONE = 1500.0   # mm — ensures ≥400mm flange-tip clearance at TSWT∩SS and Hopper∩SS

        _excl = {}   # {member_name: [(t_lo, t_hi), ...]}  global param t

        def _junction_excl(name_a, name_b):
            """Compute end-stiffener exclusion zones at the acute junction."""
            if name_a not in segs_all or name_b not in segs_all:
                return
            q = intersect(*segs_all[name_a], *segs_all[name_b])
            if q is None:
                return
            for name in (name_a, name_b):
                p1, p2 = segs_all[name]
                uy_f, uz_f, L_f = dir_len(p1, p2)
                t_q = (q[0]-p1[0])*uy_f + (q[1]-p1[1])*uz_f
                t_q = max(0.0, min(L_f, t_q))
                t_lo = max(0.0, t_q - CLEAR_ZONE)
                t_hi = min(L_f, t_q + CLEAR_ZONE)
                _excl.setdefault(name, []).append((t_lo, t_hi))

        _junction_excl("TSWT",   "Side_Shell")
        _junction_excl("Hopper", "Side_Shell")

        def _in_excl(name, base_pt):
            """Return True if base_pt falls within any exclusion zone on member."""
            zones = _excl.get(name)
            if not zones:
                return False
            p1, p2 = segs_all[name]
            uy_f, uz_f, _ = dir_len(p1, p2)
            t_g = (base_pt[0]-p1[0])*uy_f + (base_pt[1]-p1[1])*uz_f
            return any(t_lo - 1.0 <= t_g <= t_hi + 1.0 for t_lo, t_hi in zones)

        # ---- stiffener 생성 루프 ----
        for name in list(segs_all.keys()):
            if name == "Bilge":
                continue
            if name == "Upper_Deck_F":
                continue

            pieces=split(name)
            if not pieces: continue

            stf_type, flange_half, web_h = _STF_CFG.get(name, ("FB", 0, 400))
            tick_len = web_h
            layer = "Stiffeners (Longi)"

            for s,e in pieces:
                # Upper_Deck_S는 Cargo Hold 구간(y <= y_tsy) 스킵
                if name == "Upper_Deck_S":
                    midy = 0.5*(s[0]+e[0])
                    if midy <= y_tsy_mm:
                        continue

                # Side_Shell은 BT1(위) & BT2(아래) 구간만 허용
                if name == "Side_Shell":
                    midz = 0.5*(s[1]+e[1])
                    allow = False
                    if z_ts_side is not None and z_deck_side is not None:
                        if z_ts_side <= midz <= z_deck_side:  # 상부 BT1
                            allow = True
                    if z_bilge_side is not None and z_hp_side is not None:
                        if z_bilge_side <= midz <= z_hp_side:  # 하부 BT2
                            allow = True
                    if not allow:
                        continue

                uy,uz,L = dir_len(s,e)
                n,sp = choose_spacing(L)
                if n<=0: continue
                nvec = normal_of(name,s,e)
                t0=self.edge_clear
                added = 0
                for i in range(1, n+1):
                    t=t0+sp*i
                    if t>=L-self.edge_clear+1e-6: break
                    base=(s[0]+uy*t, s[1]+uz*t)
                    # Skip if within junction clear zone (interference risk)
                    if _in_excl(name, base):
                        continue
                    draw_stiffener_shape(base, nvec, (uy, uz),
                                        stf_type, tick_len, flange_half, layer)
                    added += 1
                self._stf_stats[name] = self._stf_stats.get(name, 0) + max(0, added)

        # ---- CH 상부 데크 평행선 250mm 이격(표시용) ----
        y_f_m = self.s.deck_flat_y
        y_f = y_f_m * 1000.0
        y_tsy = (self.s.B / 2) / 2.0
        y_tsy_mm = y_tsy * 1000.0
        z_center = self.s.z_deck(0.0) * 1000.0
        z_flat_off = z_center - 250.0
        self.msp.add_line((0.0, z_flat_off), (y_f, z_flat_off), dxfattribs={'layer': 'Stiffeners (Trans)'})
        z_f = self.s.z_deck(y_f_m) * 1000.0
        z_tsy = self.s.z_deck(y_tsy) * 1000.0
        self.msp.add_line((y_f, z_f - 250.0), (y_tsy_mm, z_tsy - 250.0), dxfattribs={'layer': 'Stiffeners (Trans)'})

        # ---- Bilge-end stiffeners (100mm from bilge toe) ----
        BILGE_END_OFFSET = 100.0
        if self.bilge_bottom_end and self.bilge_side_start:
            # Bottom Shell: 100mm inboard of bilge_bottom_end
            if "Bottom_Shell" in segs_all:
                p1, p2 = segs_all["Bottom_Shell"]
                uy_f, uz_f, _ = dir_len(p1, p2)
                be = self.bilge_bottom_end
                # move 100mm away from bilge end (toward CL, i.e., -uy direction)
                base_be = (be[0] - uy_f * BILGE_END_OFFSET, be[1] - uz_f * BILGE_END_OFFSET)
                nv_be = normal_of("Bottom_Shell", p1, p2)
                st, fh, wh = _STF_CFG.get("Bottom_Shell", ("T", 65, 350))
                draw_stiffener_shape(base_be, nv_be, (uy_f, uz_f),
                                     st, wh, fh, "Stiffeners (Longi)")
            # Side Shell: 100mm above bilge_side_start (BT2 zone)
            if "Side_Shell" in segs_all:
                p1, p2 = segs_all["Side_Shell"]
                uy_f, uz_f, _ = dir_len(p1, p2)
                bs = self.bilge_side_start
                # move 100mm along Side_Shell direction away from bilge end
                base_bs = (bs[0] + uy_f * BILGE_END_OFFSET, bs[1] + uz_f * BILGE_END_OFFSET)
                nv_bs = normal_of("Side_Shell", p1, p2)
                st, fh, wh = _STF_CFG.get("Side_Shell", ("T", 65, 300))
                draw_stiffener_shape(base_bs, nv_bs, (uy_f, uz_f),
                                     st, wh, fh, "Stiffeners (Longi)")

        # ---- Deck Local Girders (3개, -z 방향 1000mm) ----
        self._draw_deck_local_girders()

    # ---- local deck girder (3 positions) ----
    def _draw_deck_local_girders(self):
        L = float(self.deck_local_girder_len)
        y_f = self.s.deck_flat_y
        y_tsy = (self.s.B / 2) / 2.0

        base1 = (0.0, self.s.z_deck(0.0) * 1000.0)
        tip1 = (base1[0], base1[1] - L)
        self.msp.add_line(base1, tip1, dxfattribs={'layer': 'Members'})

        zf = self.s.z_deck(y_f) * 1000.0
        base2 = (y_f * 1000.0, zf)
        tip2 = (base2[0], base2[1] - L)
        self.msp.add_line(base2, tip2, dxfattribs={'layer': 'Members'})

        y_mid = 0.5 * (y_f + y_tsy)
        z_mid = self.s.z_deck(y_mid) * 1000.0
        base3 = (y_mid * 1000.0, z_mid)
        tip3 = (base3[0], base3[1] - L)
        self.msp.add_line(base3, tip3, dxfattribs={'layer': 'Members'})

    # ---------------------- EXPORT META (LNGC-style) ----------------------
    def _build_export_stats(self, qc, png_path):
        # Member geometry props
        member_props = {}
        for name, m in self.s.seg_dict().items():
            (y1,z1),(y2,z2) = m
            Lmm = hypot(y2 - y1, z2 - z1)
            angle_deg = degrees(atan2(z2 - z1, y2 - y1)) if Lmm > EPS else 0.0
            bbox = {"min_y_mm": round(min(y1,y2),3), "max_y_mm": round(max(y1,y2),3),
                    "min_z_mm": round(min(z1,z2),3), "max_z_mm": round(max(z1,z2),3)}
            member_props[name] = {
                "full_name": expand_abbrev(name),
                "endpoints_mm": [(round(y1,3), round(z1,3)), (round(y2,3), round(z2,3))],
                "length_mm": round(Lmm,3),
                "length_m": round(Lmm/1000.0,6),
                "slope_deg": round(angle_deg,6),
                "bbox_mm": bbox
            }

        hold_len_m = float(self.hold_length_m) if self.hold_length_m is not None else None

        # Areas per member (plate projection) — HL 기준
        member_areas = {}
        if hold_len_m is not None:
            for nm, prop in member_props.items():
                length_m = prop["length_m"]
                area_half = length_m * hold_len_m
                area_full = area_half * 2.0
                member_areas[nm] = {
                    "area_m2_half": round(area_half, 6),
                    "area_m2_full": round(area_full, 6),
                }

        # ---- Compartments + volumes (ALL use HL) ----
        comp_items = list(self._compartment_data)
        comp_vols = []
        group_sums = {
            "Cargo Hold (STBD)": 0.0,
            "W.B.T (STBD)": 0.0,
            "Pipe duct (STBD)": 0.0,
            "Void (STBD)": 0.0,
        }

        def cname(meta):
            return clean_multiline_label(meta["raw_label"])

        for c in comp_items:
            A = float(c["area_m2"])
            nm = cname(c)
            low = nm.lower()
            if hold_len_m is None:
                continue
            vol_half = A * hold_len_m
            vol_full = vol_half * 2.0
            comp_vols.append({
                "name": nm,
                "volume_m3_half": round(vol_half, 6),
                "volume_m3_full": round(vol_full, 6),
            })
            if low.startswith("cargo hold"):
                group_sums["Cargo Hold (STBD)"] += vol_half
            elif low.startswith("ballast"):
                group_sums["W.B.T (STBD)"] += vol_half
            elif low.startswith("pipe"):
                group_sums["Pipe duct (STBD)"] += vol_half
            elif low.startswith("void"):
                group_sums["Void (STBD)"] += vol_half

        group_sums_full = {k.replace("(STBD)", "(FULL)"): v * 2.0 for k, v in group_sums.items()}

        # ---- Capacity token (FULL, cargo hold only) ----
        hold_list_full = [v["volume_m3_full"] for v in comp_vols if v["name"].lower().startswith("cargo hold")]
        cargo_per_hold_full = float(sum(hold_list_full)) if hold_list_full else None

        total_cargo_full = None
        capacity_token_k = None
        if cargo_per_hold_full is not None and cargo_per_hold_full > 0.0:
            total_cargo_full = ((self.number_of_hold - 1) * cargo_per_hold_full) + (cargo_per_hold_full * self.hold_vol_factor)
            capacity_token_k = f"{int(round(total_cargo_full / 1000.0))}K"


        # Bilge info
        bilge_info = None
        if (self.bilge_bottom_end is not None) and (self.bilge_side_start is not None):
            B2 = self.s.B*1000/2.0; R = self.s.R*1000.0
            cy, cz = (B2 - R), R
            a0 = atan2(self.bilge_bottom_end[1]-cz, self.bilge_bottom_end[0]-cy)
            a1 = atan2(self.bilge_side_start[1]-cz, self.bilge_side_start[0]-cy)
            arc_len = abs(R*(a1-a0))
            bilge_info = {
                'center_mm': (round(cy,3), round(cz,3)),
                'radius_mm': round(R,3),
                'start_deg': round(a0*180.0/pi,6),
                'end_deg': round(a1*180.0/pi,6),
                'arc_length_mm': round(arc_len,3),
                'toe_points_mm': {'bottom_end': self.bilge_bottom_end, 'side_start': self.bilge_side_start}
            }

        # Labels summary
        label_summary = {
            'count': len(self._labels),
            'items': [
                {'name': r['name'],
                 'full_name': expand_abbrev(clean_multiline_label(r['name'])),
                 'pos_mm': (round(r['pos'][0],3), round(r['pos'][1],3)),
                 'rotation_deg': r['rotation_deg'],
                 'layer': r['layer']}
                for r in self._labels
            ]
        }

        # Stiffeners
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

        # Layer counts
        layer_counts = {}
        for e in self.msp:
            ly = e.dxf.layer if hasattr(e.dxf, 'layer') else 'UNKNOWN'
            layer_counts[ly] = layer_counts.get(ly, 0) + 1

        # Drawing bbox
        ys, zs = [], []
        for m in self.s.seg_dict().values():
            ys += [float(m[0][0]), float(m[1][0])]
            zs += [float(m[0][1]), float(m[1][1])]
        for p in [self.bilge_bottom_end, self.bilge_side_start]:
            if p:
                ys.append(float(p[0])); zs.append(float(p[1]))
        bbox = None
        if ys and zs:
            bbox = {'min_y_mm': round(min(ys),3), 'max_y_mm': round(max(ys),3),
                    'min_z_mm': round(min(zs),3), 'max_z_mm': round(max(zs),3)}

        # Compartments summary
        compartments = {
            "items": self._compartment_data,
            "count": len(self._compartment_data),
            "total_area_m2": round(sum(c["area_m2"] for c in self._compartment_data), 6),
        }

        # conventions
        doc_conventions = {
            "units": {"lengths": {"drawing": "mm", "model": "m"},
                      "area": "m^2", "volume": "m^3"},
            "coordinate_system": {
                "axes": "y(horizontal, +outboard), z(vertical, +up)",
                "origin": "Centerline keel point at (0,0) in this section drawing",
                "section": "Midship transverse section (2D, y–z plane)"},
            "drawing_conventions": {
                "deck_camber": "Upper_Deck_F is flat near CL; Upper_Deck_S is cambered linear to side",
                "labels_multiline": "\\P are line breaks in CAD text",
                "members": "Member lines are given as two points in mm in (y,z)",
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
                "units": {'L_m':'m','B_m':'m','D_m':'m','HL_m':'m',
                          'camberUpper_m':'m','doubleBottom_m':'m','bilgeRadius_m':'m'},
                "symbols": {'B_m':'B','D_m':'D','L_m':'L','HL_m':'HL','camberUpper_m':'C','doubleBottom_m':'DB','bilgeRadius_m':'R'}
            }
        }

        drawing_meta = {
            'layers': layer_counts,
            'bbox_mm': bbox,
            'labels': {'count': label_summary['count'], 'items': label_summary['items']},
            'stiffeners': stiffeners,
            'intersections': self._intersections,
            'qc': {'label_overlaps': qc.get('label_overlaps', 0), 'labels_ok': qc.get('ok', True)},
            'files': {'dxf': None, 'png': png_path}
        }

        if not EXPORT_INCLUDE_MEMBER_BBOX:
            member_props = {name: {k: v for k, v in props.items() if k != 'bbox_mm'}
                            for name, props in member_props.items()}

        export_stats = {
            'hold': {
                'length_m': hold_len_m,
                'hold_len_factor': self.hold_len_factor,
                'hold_vol_factor': self.hold_vol_factor,
                'number_of_hold': self.number_of_hold,
                'length_basis_note': 'Member areas & compartment volumes use HL; L uses hold_len_factor*HL*number_of_hold; total cargo uses number_of_hold * per-hold FULL * hold_vol_factor',
            },
            'members': {'geometry': member_props, 'areas': member_areas},
            'compartments': {
                'items': self._compartment_data,
                'volumes': {
                    'items': comp_vols,
                    'groups_half': {k: round(v,6) for k,v in group_sums.items()},
                    'groups_full': {k: round(v,6) for k,v in group_sums_full.items()},
                    'cargo_per_hold_full_m3': round(cargo_per_hold_full,6) if cargo_per_hold_full is not None else None,
                    'cargo_total_full_m3': round(total_cargo_full,6) if total_cargo_full is not None else None,
                    'cargo_capacity_token': capacity_token_k,
                },
                'count': len(self._compartment_data),
                'total_area_m2_half': round(sum(c["area_m2"] for c in self._compartment_data), 6),
                'total_area_m2_full': round(2.0 * sum(c["area_m2"] for c in self._compartment_data), 6),
            },
            'drawing': drawing_meta,
            'domain': {
                'legend': {k: expand_abbrev(k) for k in self.s.seg_dict().keys()},
                'registry_version': "1.0",
                'conventions': doc_conventions,
                'rule_refs': {
                    'stiffener_rules': f"{self.stf_min} ≤ spacing ≤ {self.stf_max} (target {self.stf_target}), edge clear {self.edge_clear} mm",
                },
                'stiffener_types': _STF_TYPE_LEGEND,
                'scantling_table': {'header': _SCANTLING_TABLE[0], 'rows': list(_SCANTLING_TABLE[1:])},
            }
        }
        return export_stats

    # ---- Scantling table inside Cargo Hold ----
    def draw_scantling_table(self):
        """Draw scantling table as DXF entities centered in the Cargo Hold space."""
        from math import hypot

        layer = "Scantling"
        txt_h = 180.0; txt_h_hdr = 200.0
        col_w = [3600.0, 1900.0, 6800.0]; row_h = 700.0
        rows = _SCANTLING_TABLE
        n_rows = len(rows)
        total_w = sum(col_w); total_h = n_rows * row_h

        ch_cy, ch_cz = None, None
        for c in self._compartment_data:
            if "cargo hold" in c.get("clean_label", "").lower():
                ch_cy, ch_cz = c["centroid_mm"]; break
        if ch_cy is None:
            ch_cy = (self.s.B / 2) / 2.0 * 1000.0
            ch_cz = (self.s.DB + self.s.D) / 2.0 * 1000.0

        ay = max(ch_cy - total_w / 2.0, 200.0)
        az = ch_cz + total_h / 2.0

        corners = [(ay,az),(ay+total_w,az),(ay+total_w,az-total_h),(ay,az-total_h),(ay,az)]
        self.msp.add_polyline2d(corners, dxfattribs={'layer': layer})
        x = ay
        for cw in col_w[:-1]:
            x += cw
            self.msp.add_line((x,az),(x,az-total_h), dxfattribs={'layer': layer})
        for r in range(1, n_rows):
            z_line = az - r * row_h
            self.msp.add_line((ay,z_line),(ay+total_w,z_line), dxfattribs={'layer': layer})
        for r, row_data in enumerate(rows):
            z_top = az - r * row_h
            ch = txt_h_hdr if r == 0 else txt_h
            x = ay
            for cell_text, cw in zip(row_data, col_w):
                mt = self.msp.add_mtext(cell_text, dxfattribs={'layer': layer, 'char_height': ch})
                mt.dxf.insert = (x + cw/2.0, z_top - row_h/2.0)
                mt.dxf.attachment_point = 5; mt.dxf.rotation = 0
                x += cw
        title = self.msp.add_mtext("SCANTLING TABLE (LONGITUDINALS)",
            dxfattribs={'layer': layer, 'char_height': txt_h_hdr + 30})
        title.dxf.insert = (ay, az + 380.0)
        title.dxf.attachment_point = 4; title.dxf.rotation = 0

        # Cargo Hold label above table
        z_deck_top = self.s.z_deck(0.0) * 1000.0
        ch_label = self.msp.add_mtext("Cargo Hold",
            dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
        ch_label.dxf.insert = (ch_cy, (az + z_deck_top) / 2.0)
        ch_label.dxf.attachment_point = 5; ch_label.dxf.rotation = 0

    def export(self, save_as=None, png_out_dir=None, png_dpi=220):
        # 1) 도면 생성 (시각 요소는 그대로)
        self.draw_layers()
        self.draw_centerline()
        self.draw_title_and_specs(title="ORDINARY SECTION (STBD)")
        self.draw_hull()
        # BULKC는 Cargo Tank가 없으므로 draw_cargo() 없음
        self.draw_compartments()
        self.draw_stiffeners()
        self.draw_scantling_table()

        qc = {'ok': True, 'label_overlaps': 0}

        # 2) DXF 저장
        if save_as:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            self.doc.saveas(save_as)

        # 3) PNG 렌더(옵션)
        png_path = None
        if png_out_dir and _MAT_OK:
            os.makedirs(png_out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(save_as or "bulkc"))[0]
            out = os.path.join(png_out_dir, base + ".png")
            face = mcolors.to_rgba("white")
            fig = plt.figure(figsize=(12, 12), dpi=png_dpi, facecolor=face)
            ax = fig.add_axes([0, 0, 1, 1], facecolor=face)
            Frontend(RenderContext(self.doc), MatplotlibBackend(ax)).draw_layout(self.doc.modelspace())
            ax.set_aspect("equal")
            ax.set_axis_off()
            fig.savefig(out, dpi=png_dpi, facecolor=face, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            png_path = out

        # 4) 메타 빌드 (LNGC 스타일)
        export_stats = self._build_export_stats(qc, png_path)

        # 5) 파일 경로를 메타에 채우기
        try:
            export_stats['drawing']['files']['dxf'] = save_as
            export_stats['drawing']['files']['png'] = png_path
        except Exception:
            pass

        return qc, png_path, export_stats


# =========================================
#            Domain rules (간단화)
# =========================================
def domain_rules_ok(p):
    import math
    B=p['B']; D=p['D']; DB=p['DB']; R=p['R']; C=p['C']
    GY=p['GY']; OG=p['OG']; TSE=p['TSWT_EXT']
    S2=p['S2']; DS=p['DS']

    issues=[]
    if not (B>0 and D>0 and DB>0 and R>0): issues.append("PositiveDims")
    if C<0 or C>0.05*B or C>0.10*D: issues.append("Camber_limit")
    if R >= (D-DB) - 0.2: issues.append("BilgeR_vs_Depth")

    y_bilge_toe = (B/2) - R
    y_og = OG*(B/2)
    if not (0.6 <= GY <= (B/2) - R - 1.0): issues.append("Girder_y_out_of_range")
    if y_og > (y_bilge_toe - 0.8) or y_og < (GY + 0.6): issues.append("OutGirder_range_vs_bilge/girder")

    if not (110.0 <= TSE <= 130.0): issues.append("TSWT_ext_angle_110_130")

    # Side Frame offset 0.6~0.7 m
    if not (0.6 <= DS <= 0.7): issues.append("SideFrame_offset_0p6_0p7m")

    # legacy checks (무해)
    def z_deck_at(y):
        y_f=3.5
        z_center=D+C
        if y<=y_f: return z_center
        y2=B/2; z2=D
        m=(z2-z_center)/(y2-y_f)
        return z_center + m*(y-y_f)

    y_tsy = (B/2)/2.0
    z_top = z_deck_at(y_tsy)
    z_kink = z_top - 0.7
    phi = math.radians(90.0 - TSE)
    phi = max(min(phi, math.radians(-30.0)), math.radians(-50.0))
    z_ts_side = z_kink + math.tan(phi)*(B/2 - y_tsy)
    z1 = z_ts_side - 1.7
    if z1 < DB + 0.5: issues.append("Str1_below_DB_margin")
    if z_ts_side <= z1: issues.append("TSWT_not_above_Str1_by_1p7m")
    if (S2*D) <= DB + 0.5: issues.append("Str2_low_vs_DB")

    return len(issues)==0, issues


# =========================================
#        Length / Hold-length utilities
# =========================================
def estimate_L_from_HL(hl_m: float, *, k0: float = 0.35, bias_m: float = 10.0) -> float:
    """
    간단 L 추정 모델:
      L ≈ HL * (1 + k0) + bias
    - k0, bias는 임시값(주석 교체 용이)
    """
    return float(max(hl_m * (1.0 + k0) + bias_m, hl_m + 1.0))

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
    # 길면 잘라내고, 부족하면 순환
    if len(vals) >= N:
        return vals[:N]
    out = []
    import random
    rng = random.Random(1234)
    i = 0
    while len(out) < N:
        base = vals[i % len(vals)]
        jitter = rng.uniform(-0.25*step, 0.25*step)
        out.append(round(max(lo, min(hi, base + jitter)), 3))
        i += 1
    return out


# =========================================
#            Dataset generator
# =========================================
def generate_bulkc_dataset(
    save_dir,
    method='lhs',
    # 길이/홀드 길이 설정
    use_estimated_L=True,
    L_fixed=300.0,
    HL_mode='sample',
    HL_fixed=25.0,
    HL_range=(20.0, 30.0, 0.1),

    # 길이/용적 계수 및 홀드 수 범위
    hold_len_factor=1.0,
    hold_vol_factor=0.7,
    number_of_hold_range=(8, 10, 1),

    # 형상 범위
    B_range=(40,50,1),
    D_range=(20,30,1),
    camber_range=(0.5,2.5,0.1),
    ds_from_side=(0.6, 0.7, 0.01),  # 600~700 mm
    db_range=(1.5,3.5,0.1),
    bilge_range=(1.0,3.0,0.1),
    girder_y=(2.0,2.4,0.05),
    outgir_ratio=(0.7,0.8,0.05),
    tswt_ext=(110.0,130.0,2.0),
    str_clear=(0.2,0.4,0.1),
    s1_ratio=(0.7,0.8,0.05),
    s2_ratio=(0.4,0.6,0.05),


    # 도면
    text_height=250, offset=300,
    MAX_FILES=100, PROGRESS_EVERY=20, SEED=42,
    png_out_dir=None, png_dpi=220,
    json_out_dir=None,
    compart_out_dir=None,
    compart_png_out_dir=None,
    compart3d_out_dir=None,
    compart3d_png_out_dir=None,
    fwd_hold_ratio=0.10, er_hold_ratio=0.14, aft_hold_ratio=0.075,
    ship_data_defaults=None,
):
    os.makedirs(save_dir, exist_ok=True)
    for d in [compart_out_dir, compart_png_out_dir, compart3d_out_dir, compart3d_png_out_dir]:
        if d is not None:
            os.makedirs(d, exist_ok=True)
    rng = random.Random(SEED)

    # 샘플 파라미터(L 제외)
    specs=[
        {'name':'B','min':B_range[0],'max':B_range[1],'type':'int','step':B_range[2]},
        {'name':'D','min':D_range[0],'max':D_range[1],'type':'int','step':D_range[2]},
        {'name':'C','min':camber_range[0],'max':camber_range[1],'type':'float','step':camber_range[2]},
        {'name':'DB','min':db_range[0],'max':db_range[1],'type':'float','step':db_range[2]},
        {'name':'R','min':bilge_range[0],'max':bilge_range[1],'type':'float','step':bilge_range[2]},
        {'name':'GY','min':girder_y[0],'max':girder_y[1],'type':'float','step':girder_y[2]},
        {'name':'OG','min':outgir_ratio[0],'max':outgir_ratio[1],'type':'float','step':outgir_ratio[2]},
        {'name':'TSWT_EXT','min':tswt_ext[0],'max':tswt_ext[1],'type':'float','step':tswt_ext[2]},
        {'name':'DS','min':ds_from_side[0],'max':ds_from_side[1],'type':'float','step':ds_from_side[2]},
        {'name':'STRCLR','min':str_clear[0],'max':str_clear[1],'type':'float','step':str_clear[2]},
        {'name':'S1','min':s1_ratio[0],'max':s1_ratio[1],'type':'float','step':s1_ratio[2]},
        {'name':'S2','min':s2_ratio[0],'max':s2_ratio[1],'type':'float','step':s2_ratio[2]},
    ]
    samples = lhs_samples(MAX_FILES, specs, seed=SEED)

    # HL 목록 준비
    if HL_mode == 'fixed':
        hl_list = [float(HL_fixed)] * MAX_FILES
    else:
        hl_list = sample_HL(MAX_FILES, HL_range)

    # CSV 헤더: LNGC 스타일 정렬
    header=[
        'file','json','method','seed',
        'Cargo Capacity (K)','Number of Hold','Hold Len. Factor','Hold Vol. Factor',
        'HL','TL','L',
        'B','D','C','DB','R','GY','OG','TSWT_EXT','DS','STRCLR','S1','S2',
        'domain_ok','domain_issues',
        'generator_constraints_ok','generator_constraint_count','inactive_parameters',
        'csr_scope_status','csr_pass','csr_fail','csr_undetermined','csr_not_modeled',
        'LLL_m','framing_system',
        'qc_ok','label_overlaps','filesize','png'
    ]
    _index_dir = os.path.dirname(os.path.abspath(save_dir))
    index_csv = os.path.join(_index_dir, "BULKC_dataset_index.csv")

    saved=0
    for i, p in enumerate(samples):
        # 길이 설정
        HL_m = float(hl_list[i])
        # 홀드 수 샘플링
        nh_min, nh_max, nh_step = number_of_hold_range
        number_of_hold = rng.randrange(nh_min, nh_max + 1, nh_step)
        hold_total = hold_len_factor * HL_m * number_of_hold
        fwd_len = fwd_hold_ratio * hold_total
        er_len  = er_hold_ratio  * hold_total
        aft_len = aft_hold_ratio * hold_total
        if use_estimated_L:
            # NEW: L = hold_len_factor * HL * number_of_hold
            p['L'] = float(hold_len_factor * HL_m * number_of_hold)
        else:
            p['L'] = float(L_fixed)

        # Phase 0.2.B1 Fix B1: IACS CSR-H bulk carrier rules apply only for L >= 150 m.
        # Reject sub-scope samples outright to keep every emitted bulker inside CSR applicability.
        if p['L'] < 150.0:
            continue

        ok,issues=domain_rules_ok(p)
        if not ok:
            continue

        # Ship build
        ship = BULKC(
            L=p['L'], B=p['B'], D=p['D'], DB=p['DB'], R=p['R'], camber=p['C'],
            y_girder=p['GY'], y_og_ratio=p['OG'], tswt_ext_deg=p['TSWT_EXT'],
            ds_from_side=p['DS'], str_clear=p['STRCLR'],
            s1_ratio=p['S1'], s2_ratio=p['S2']
        )

        # CSR 평가용 입력 딕셔너리
        _gen_inputs_for_csr = {
            'L_m': p['L'], 'B_m': p['B'], 'D_m': p['D'],
            'doubleBottom_m': p['DB'],
            'bilgeRadius_m': p['R'],
            'tswt_ext_deg': p['TSWT_EXT'],
        }
        ship_data = build_ship_data_context(p['L'], ship_data_defaults=ship_data_defaults)
        generator_constraints = build_generator_constraints_summary(_gen_inputs_for_csr, ship, issues)
        csr_eval = evaluate_csr_rules_bulkc(_gen_inputs_for_csr, ship_data, ship)

        # 파일명(초기: capacity prefix 없이)
        dxf_path = build_filename(save_dir, p['L'], p['B'], p['D'], p['C'], p['DB'], p['R'],
                                  p['GY'], p['OG'], p['TSWT_EXT'],
                                  p['DS'], p['STRCLR'],
                                  p['S1'], p['S2'])

        # 1) Exporter 실행
        exp = DXFExporter(
            ship,
            text_height=text_height, offset=offset,
            hold_length_m=HL_m,
            hold_len_factor=hold_len_factor,
            hold_vol_factor=hold_vol_factor,
            number_of_hold=number_of_hold
        )
        qc, png_path, stats = exp.export(save_as=dxf_path, png_out_dir=png_out_dir, png_dpi=png_dpi)

        # 2) K-토큰 추출
        capacity_token = stats.get('compartments', {}) \
            .get('volumes', {}) \
            .get('cargo_capacity_token')
        final_dxf_path = dxf_path

        if capacity_token:
            base = os.path.basename(dxf_path)
            hold_tag = f"{number_of_hold}Hold"  # 예: "8Hold"
            if not base.startswith(capacity_token + "_"):
                new_base = f"{capacity_token}_{hold_tag}_{base}"
                new_dxf = os.path.join(os.path.dirname(dxf_path), new_base)
                try:
                    os.replace(dxf_path, new_dxf)
                    final_dxf_path = new_dxf

                    # PNG도 동일 접두사 적용
                    if png_path:
                        old_png = png_path
                        new_png = os.path.join(
                            os.path.dirname(old_png),
                            os.path.splitext(new_base)[0] + ".png"
                        )
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

        # 4) Longitudinal layout + Elev + 3D
        dxf_path = final_dxf_path
        fsize = os.path.isfile(final_dxf_path) and os.path.getsize(final_dxf_path) or -1

        layout = None
        if compart_out_dir is not None or compart3d_out_dir is not None:
            layout = build_longitudinal_layout(
                L_m=p['L'], HL_m=HL_m, number_of_hold=number_of_hold,
                fwd_len_m=fwd_len, er_len_m=er_len, aft_len_m=aft_len,
                hold_len_factor=hold_len_factor,
            )

        compart_dxf_path = None; compart_png_path = None
        base_noext = os.path.splitext(os.path.basename(final_dxf_path))[0]
        if compart_out_dir is not None and layout is not None:
            compart_dxf_path = os.path.join(compart_out_dir, base_noext + "_Compart.dxf")
            compart_dxf_path, compart_png_path = create_compartment_arrangement_drawing(
                compart_dxf_path, layout=layout, D_m=p['D'], camber_m=p['C'], DB_m=p['DB'],
                text_height=text_height, png_dir=compart_png_out_dir, png_dpi=png_dpi,
            )

        compart3d_dxf_path = None; compart3d_png_path = None
        if compart3d_out_dir is not None and layout is not None:
            _m3d = os.path.join(compart3d_out_dir, base_noext + "_Compart3D.dxf")
            compart3d_dxf_path, compart3d_png_path = create_compartment3d_dxf(
                _m3d, ship=ship, layout=layout,
                text_height=text_height, png_dir=compart3d_png_out_dir, png_dpi=png_dpi,
            )

        _json_base = os.path.basename(final_dxf_path).replace(".dxf", ".json")
        if json_out_dir:
            os.makedirs(json_out_dir, exist_ok=True)
            json_path = os.path.join(json_out_dir, _json_base)
        else:
            json_path = final_dxf_path.replace(".dxf", ".json")

        # New-style JSON
        sample_idx = saved + 1
        sample_id = f"BULKC-{sample_idx:04d}"

        meta = {
            'sample_id': sample_id,
            'ship_type': 'BULKC',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': method,
            'seed': SEED,
            'generator_inputs': {
                'L_m': p['L'], 'B_m': p['B'], 'D_m': p['D'], 'HL_m': HL_m,
                'number_of_hold': number_of_hold,
                'camber_m': p['C'],
                'doubleBottom_m': p['DB'],
                'bilgeRadius_m': p['R'],
                'girder_y_m': p['GY'],
                'girderOut_ratio': p['OG'],
                'tswt_ext_deg': p['TSWT_EXT'],
                'doubleSide_m': p['DS'],
                'strClearance_m': p['STRCLR'],
                'str1_ratio': p['S1'],
                'str2_ratio': p['S2'],
            },
            'geometry': {
                'derived': {
                    'girderOut_y_m': round(p['OG'] * (p['B'] / 2.0), 3),
                    'girder1_y_m': round(p['GY'], 3),
                },
                'longitudinal_layout': layout,
                'length_model': {
                    'fwd_len_m': fwd_len, 'er_len_m': er_len, 'aft_len_m': aft_len,
                    'hold_len_factor': hold_len_factor, 'hold_vol_factor': hold_vol_factor,
                    'mode': 'estimated_from_HL' if use_estimated_L else 'fixed',
                },
            },
            'member_semantics': {
                'Bottom_Shell':  {'description': 'Outer bottom shell', 'structural_class': 'OUTER_HULL'},
                'Side_Shell':    {'description': 'Outer side shell', 'structural_class': 'OUTER_HULL'},
                'Upper_Deck_F':  {'description': 'Upper deck (flat near CL)', 'structural_class': 'OUTER_HULL'},
                'Upper_Deck_S':  {'description': 'Upper deck (sloped to side)', 'structural_class': 'OUTER_HULL'},
                'IBTM':          {'description': 'Inner bottom plating', 'structural_class': 'INNER_HULL'},
                'Hopper':        {'description': 'Hopper plate (lower wing slant)', 'structural_class': 'INNER_HULL'},
                'Out_Girder':    {'description': 'Outboard girder in double bottom', 'structural_class': 'GIRDER'},
                'Girder1':       {'description': 'Girder 1 in double bottom', 'structural_class': 'GIRDER'},
                'Girder2':       {'description': 'Girder 2 in double bottom', 'structural_class': 'GIRDER'},
                'Girder3':       {'description': 'Girder 3 in double bottom', 'structural_class': 'GIRDER'},
                'TSWT_V':        {'description': 'Topside wing tank vertical plate', 'structural_class': 'INNER_HULL'},
                'TSWT':          {'description': 'Topside wing tank sloped plate', 'structural_class': 'INNER_HULL'},
            },
            'standard_refs': {'csr_standard': CSR_STANDARD_INFO},
            'ship_data': ship_data,
            'generator_constraints': generator_constraints,
            'rules': {**csr_eval, 'society': 'IACS_CSR_H'},  # unified schema (Phase 0.2.B1)
            'csr': csr_eval,  # legacy alias — kept for backward compat
            'cargo_summary': {
                'per_hold_full_m3': stats.get('compartments', {}).get('volumes', {}).get('cargo_per_hold_full_m3'),
                'total_full_m3':    stats.get('compartments', {}).get('volumes', {}).get('cargo_total_full_m3'),
                'capacity_token':   capacity_token,
            },
            'artifacts': {
                'section_dxf': final_dxf_path, 'section_png': png_path,
                'compart_dxf': compart_dxf_path, 'compart_png': compart_png_path,
                'compart3d_dxf': compart3d_dxf_path, 'compart3d_png': compart3d_png_path,
                'json': json_path,
            },
            'drawing': stats,
        }
        write_json(json_path, meta)

        # ---- CSV 행 (기존 헤더 유지) ----
        row = {
            'file': os.path.basename(final_dxf_path),
            'json': os.path.basename(json_path),
            'method': method,
            'seed': SEED,
            'Cargo Capacity (K)': capacity_token,
            'Number of Hold': number_of_hold,
            'Hold Len. Factor': hold_len_factor,
            'Hold Vol. Factor': hold_vol_factor,
            'HL': HL_m,
            'TL': "",               # BULKC는 탱크 없음
            'L': p['L'],
            'B': p['B'],'D': p['D'],'C': p['C'],'DB': p['DB'],'R': p['R'],
            'GY': p['GY'],'OG': p['OG'],'TSWT_EXT': p['TSWT_EXT'],
            'DS': p['DS'],'STRCLR': p['STRCLR'],'S1': p['S1'],'S2': p['S2'],
            'domain_ok': ok,
            'domain_issues': "|".join(issues),
            'generator_constraints_ok': generator_constraints['status'] == 'pass',
            'generator_constraint_count': len(generator_constraints['issues']),
            'inactive_parameters': "|".join(item['parameter'] for item in generator_constraints.get('inactive_parameters', [])),
            'csr_scope_status': next((c.get('status') for c in csr_eval.get('auto_checks', []) if c.get('check_id') == 'bulk_carrier_scope'), ""),
            'csr_pass': csr_eval.get('summary', {}).get('check_counts', {}).get('pass', 0),
            'csr_fail': csr_eval.get('summary', {}).get('check_counts', {}).get('fail', 0),
            'csr_undetermined': csr_eval.get('summary', {}).get('check_counts', {}).get('undetermined', 0),
            'csr_not_modeled': csr_eval.get('summary', {}).get('check_counts', {}).get('not_modeled', 0),
            'LLL_m': ship_data.get('LLL_m'),
            'framing_system': ship_data.get('framing_system'),
            'qc_ok': qc.get('ok', True),
            'label_overlaps': qc.get('label_overlaps', 0),
            'filesize': fsize,
            'png': os.path.basename(png_path) if png_path else ""
        }
        append_csv(index_csv, header, row)

        saved+=1
        if saved%PROGRESS_EVERY==0:
            print(f"[{method}] Saved {saved} ... last: {os.path.basename(row['file'])}")
        if saved>=MAX_FILES:
            break

    print(f"Done. Saved files: {saved} (method={method})")


# ---------- example ----------
if __name__ == "__main__":
    _BASE = "<SHIPBENCH_ROOT>/data/processed/BULKC"

    SAVE_DIR        = os.path.join(_BASE, "section_dxf")
    PNG_DIR         = os.path.join(_BASE, "section_png")
    COMPART_DIR        = os.path.join(_BASE, "compart_dxf")
    COMPART_PNG_DIR    = os.path.join(_BASE, "compart_png")
    COMPART3D_DIR     = os.path.join(_BASE, "compart3d_dxf")
    COMPART3D_PNG_DIR = os.path.join(_BASE, "compart3d_png")
    JSON_DIR        = os.path.join(_BASE, "json")

    generate_bulkc_dataset(
        save_dir=SAVE_DIR,
        json_out_dir=JSON_DIR,
        method='lhs',

        use_estimated_L=True,
        L_fixed=300.0,
        HL_mode='sample',
        HL_fixed=25.0,
        HL_range=(20.0, 30.0, 0.1),

        hold_len_factor=1.0,
        hold_vol_factor=0.7,
        number_of_hold_range=(8, 10, 1),

        fwd_hold_ratio=0.10,
        er_hold_ratio=0.14,
        aft_hold_ratio=0.075,

        compart_out_dir=COMPART_DIR,
        compart_png_out_dir=COMPART_PNG_DIR,
        compart3d_out_dir=COMPART3D_DIR,
        compart3d_png_out_dir=COMPART3D_PNG_DIR,

        B_range=(40,50,1),
        D_range=(20,30,1),
        camber_range=(0.5,2.5,0.1),
        ds_from_side=(0.6, 0.7, 0.01),
        db_range=(1.5,3.5,0.1),
        bilge_range=(1.0,3.0,0.1),
        girder_y=(2.0,2.4,0.05),
        outgir_ratio=(0.7,0.8,0.05),
        tswt_ext=(110.0,130.0,2.0),
        str_clear=(0.2,0.4,0.1),
        s1_ratio=(0.7,0.8,0.05),
        s2_ratio=(0.4,0.6,0.05),

        text_height=250,
        offset=300,
        MAX_FILES=100,
        PROGRESS_EVERY=20,
        SEED=42,
        png_out_dir=PNG_DIR,
        png_dpi=220,
    )
