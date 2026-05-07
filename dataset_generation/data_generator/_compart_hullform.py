"""
_compart_hullform.py
====================
Shared Compartment Arrangement hull-form renderer for all ship-type generators.

Provides:
  HULL_PROFILES   — per-ship-type bow/stern geometry parameters
  render_compartment_png(png_path, layout, D_m, camber_m, DB_m, ship_type, png_dpi)
      → renders a Compartment Arrangement PNG with proper hull-form outline
        (raked stem, smooth forefoot, skeg, transom/normal stern)

Design notes
------------
* stem_xs uses a MONOTONICALLY DECREASING power curve (bow_tip_x → fwd_x1).
  The old formula `rake*(1-t^0.7)*t^0.15` created a forward loop at the
  keel-end of the stem, producing a detached circle artifact when matplotlib
  filled the polygon.  This version has no such loop.
* AFT / FWD segments are filled as hull-form polygons (bow/stern), not boxes.
* Midbody segments (ER, HOLD*) are filled as rectangles.
* Waterline marker is drawn at ~70 % of D.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Ship-type hull-form profiles ──────────────────────────────────────────────
HULL_PROFILES: dict[str, dict] = {
    'BULKC': {
        'bow_deck_sheer':  0.015,   # bow sheer as fraction of D
        'stem_rake':       0.20,    # max forward rake as fraction of L_bow
        'forefoot_z_ratio':0.06,    # forefoot height as fraction of D
        'stern_deck_sheer':0.010,
        'stern_style':     'normal',
        'skeg_rise':       0.10,
        'skeg_start':      0.35,
    },
    'CNTR': {
        'bow_deck_sheer':  0.020,
        'stem_rake':       0.25,
        'forefoot_z_ratio':0.05,
        'stern_deck_sheer':0.008,
        'stern_style':     'transom',
        'transom_height':  0.50,    # transom base as fraction of D
        'skeg_rise':       0.08,
        'skeg_start':      0.30,
    },
    'LNGC': {
        'bow_deck_sheer':  0.012,
        'stem_rake':       0.18,
        'forefoot_z_ratio':0.05,
        'stern_deck_sheer':0.010,
        'stern_style':     'normal',
        'skeg_rise':       0.08,
        'skeg_start':      0.30,
    },
    'LPGC': {
        'bow_deck_sheer':  0.012,
        'stem_rake':       0.18,
        'forefoot_z_ratio':0.05,
        'stern_deck_sheer':0.010,
        'stern_style':     'normal',
        'skeg_rise':       0.08,
        'skeg_start':      0.30,
    },
    'Tanker': {
        'bow_deck_sheer':  0.012,
        'stem_rake':       0.18,
        'forefoot_z_ratio':0.05,
        'stern_deck_sheer':0.008,
        'stern_style':     'normal',
        'skeg_rise':       0.08,
        'skeg_start':      0.30,
    },
    'VLCC': {
        'bow_deck_sheer':  0.010,
        'stem_rake':       0.15,
        'forefoot_z_ratio':0.04,
        'stern_deck_sheer':0.006,
        'stern_style':     'normal',
        'skeg_rise':       0.06,
        'skeg_start':      0.25,
    },
}


# ── Segment helpers ───────────────────────────────────────────────────────────

def _seg_display_name(name: str) -> str:
    n = name.strip()
    if n == 'AFT':    return 'AFT End'
    if n == 'ER':     return 'Engine Room'
    if n == 'FWD':    return 'FWD End'
    if n.upper().startswith('CD'): return 'C.D.'
    if n.upper().startswith('HOLD'):
        num = n[4:].replace('_', '').strip()
        return f'Hold {num}' if num else 'Hold'
    return n


def _compart_zone_color(seg_name: str) -> str:
    n = seg_name.strip().upper()
    if n.startswith('HOLD'): return '#ffcccc'
    if n == 'ER':             return '#cccccc'
    if n.startswith('CD'):    return '#e0e0e0'
    return '#cce0ff'


# ── Curve generators ──────────────────────────────────────────────────────────

def _bow_curves(fwd_x0: float, fwd_x1: float, deck_z: float,
                D_mm: float, profile: dict):
    """
    Bow profile.

    bow_tip = (fwd_x1 + rake, deck_z + sheer)  — furthest-forward point
    stem    = bow_tip → (fwd_x1, forefoot_z)    — monotonically aft & down, NO loop
    keel    = (fwd_x1, forefoot_z) → (fwd_x0, 0)

    Returns (deck_pts, stem_pts, keel_pts) as lists of (x, z) tuples.
    Polygon order: deck → stem → keel → close.
    """
    L_bow      = fwd_x1 - fwd_x0
    sheer      = profile['bow_deck_sheer']   * D_mm
    rake       = profile['stem_rake']        * L_bow
    forefoot_z = profile['forefoot_z_ratio'] * D_mm

    bow_tip_x = fwd_x1 + rake
    bow_tip_z = deck_z  + sheer

    # 1. Deck: slight sheer, extends to bow tip
    td       = np.linspace(0, 1, 40)
    deck_xs  = fwd_x0 + td * (bow_tip_x - fwd_x0)
    deck_zs  = deck_z  + sheer * td**3
    deck_pts = list(zip(deck_xs, deck_zs))

    # 2. Stem: x monotonically decreases bow_tip_x → fwd_x1 (no forward protrusion)
    #    Power exponent > 1 gives a slightly concave (modern clipper) bow.
    st       = np.linspace(0, 1, 50)
    stem_xs  = bow_tip_x + (fwd_x1 - bow_tip_x) * st**1.15
    stem_zs  = bow_tip_z  + (forefoot_z - bow_tip_z) * st
    stem_pts = list(zip(stem_xs, stem_zs))

    # 3. Keel: flat with gentle rise at forefoot
    tk       = np.linspace(0, 1, 40)
    keel_xs  = fwd_x1 - tk * L_bow
    rise_t   = np.clip((0.15 - tk) / 0.15, 0, 1)
    keel_zs  = forefoot_z * rise_t**2.0
    keel_pts = list(zip(keel_xs, keel_zs))

    return deck_pts, stem_pts, keel_pts


def _stern_curves(aft_x0: float, aft_x1: float, deck_z: float,
                  D_mm: float, profile: dict):
    """
    Stern profile.

    Returns (deck_pts, post_pts, keel_pts) as lists of (x, z) tuples.
    """
    L_stern    = aft_x1 - aft_x0
    sheer      = profile['stern_deck_sheer'] * D_mm
    skeg_rise  = profile['skeg_rise']        * D_mm
    skeg_start = profile.get('skeg_start', 0.30)
    is_transom = profile.get('stern_style') == 'transom'

    # 1. Deck
    td       = np.linspace(0, 1, 40)
    deck_xs  = aft_x0 + td * L_stern
    deck_zs  = deck_z  + sheer * (1 - td)**4
    deck_pts = list(zip(deck_xs, deck_zs))

    # 2. Keel
    tk       = np.linspace(0, 1, 40)
    keel_xs  = aft_x0 + tk * L_stern
    rise_t   = np.clip((skeg_start - tk) / skeg_start, 0, 1)
    keel_zs  = skeg_rise * rise_t**1.8
    keel_pts = list(zip(keel_xs, keel_zs))

    # 3. Stern post / transom
    if is_transom:
        transom_z = profile.get('transom_height', 0.50) * D_mm
        post_pts = [
            (aft_x0, deck_zs[0]),
            (aft_x0, transom_z),
            (aft_x0, keel_zs[0]),
        ]
    else:
        sp       = np.linspace(0, 1, 20)
        post_zs  = deck_zs[0] * (1 - sp) + keel_zs[0] * sp
        overhang = L_stern * 0.03 * (1 - sp)**2 * sp
        post_xs  = aft_x0 - overhang
        post_pts = list(zip(post_xs, post_zs))

    return deck_pts, post_pts, keel_pts


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_compartment_png(
    png_path: str,
    layout:   dict,
    D_m:      float,
    camber_m: float,
    DB_m:     float,
    ship_type: str,
    png_dpi:  int = 220,
) -> None:
    """
    Render a Compartment Arrangement PNG with hull-form outline.

    Parameters
    ----------
    png_path  : output file path (directory must already exist or will be created)
    layout    : dict with keys 'L_m', 'segments', 'bulkheads_mm'
                segments: list of {'name', 'x0_mm', 'x1_mm'}
                  first  segment = AFT end (stern)
                  last   segment = FWD end (bow)
    D_m       : moulded depth [m]
    camber_m  : camber at centreline [m]
    DB_m      : double-bottom height [m]
    ship_type : one of HULL_PROFILES keys; falls back to 'Tanker'
    png_dpi   : output DPI
    """
    import os
    os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)

    profile    = HULL_PROFILES.get(ship_type, HULL_PROFILES['Tanker'])
    segs       = layout['segments']
    x_end      = segs[-1]['x1_mm']
    deck_z_mm  = (D_m + camber_m) * 1000.0
    db_z_mm    = DB_m  * 1000.0
    D_mm       = D_m   * 1000.0

    aft_seg    = segs[0]
    fwd_seg    = segs[-1]
    midbody_x0 = aft_seg['x1_mm']
    midbody_x1 = fwd_seg['x0_mm']

    # ── Generate hull curves ──
    bow_deck, bow_stem, bow_keel = _bow_curves(
        midbody_x1, fwd_seg['x1_mm'], deck_z_mm, D_mm, profile)
    stern_deck, stern_post, stern_keel = _stern_curves(
        aft_seg['x0_mm'], midbody_x0, deck_z_mm, D_mm, profile)

    # ── Figure ──
    fig_w = max(18, x_end / 1000.0 * 0.06)
    fig_h = max(5,  deck_z_mm / 1000.0 * 1.2)
    fig, ax = plt.subplots(figsize=(min(fig_w, 32), min(fig_h, 10)))
    lw = 1.8

    # ── Fill midbody segments (AFT / FWD handled as polygons below) ──
    for seg in segs:
        sname = seg['name'].strip().upper()
        if sname in ('AFT', 'FWD'):
            continue
        x0s, x1s = seg['x0_mm'], seg['x1_mm']
        color = _compart_zone_color(seg['name'])
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0s, 0), x1s - x0s, deck_z_mm,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=color, alpha=0.55))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0s, 0), x1s - x0s, db_z_mm,
            boxstyle="square,pad=0", linewidth=0,
            facecolor='#aaaaee', alpha=0.45))

    # ── Fill bow polygon (FWD region with hull-form shape) ──
    bow_poly = bow_deck + bow_stem + bow_keel
    ax.add_patch(plt.Polygon(
        bow_poly, closed=True,
        facecolor=_compart_zone_color('FWD'), alpha=0.55, linewidth=0))

    # ── Fill stern polygon (AFT region with hull-form shape) ──
    stern_poly = stern_deck[::-1] + stern_post + stern_keel
    ax.add_patch(plt.Polygon(
        stern_poly, closed=True,
        facecolor=_compart_zone_color('AFT'), alpha=0.55, linewidth=0))

    # ── Hull outline ──
    def _pl(pts, **kw):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **kw)

    _pl(stern_keel, color='k', linewidth=lw)
    _pl(stern_deck, color='k', linewidth=lw)
    _pl(stern_post, color='k', linewidth=lw)
    ax.plot([midbody_x0, midbody_x1], [0,         0        ], 'k-', linewidth=lw)
    ax.plot([midbody_x0, midbody_x1], [deck_z_mm, deck_z_mm], 'k-', linewidth=lw)
    _pl(bow_deck, color='k', linewidth=lw)
    _pl(bow_stem, color='k', linewidth=lw)
    _pl(bow_keel, color='k', linewidth=lw)

    # DB line (midbody)
    ax.plot([midbody_x0, midbody_x1], [db_z_mm, db_z_mm],
            color='#3344bb', linewidth=1.0, linestyle='--')

    # Bulkheads (midbody only)
    for bkx in layout.get('bulkheads_mm', []):
        if midbody_x0 <= bkx <= midbody_x1:
            ax.plot([bkx, bkx], [0, deck_z_mm], 'k-', linewidth=0.8)

    # ── Labels ──
    label_fs = max(7, min(11, 180 / max(len(segs), 1)))
    for seg in segs:
        x0s, x1s = seg['x0_mm'], seg['x1_mm']
        cx       = 0.5 * (x0s + x1s)
        display  = _seg_display_name(seg['name'])
        sname    = seg['name'].strip().upper()
        if sname.startswith('CD'):
            continue  # cofferdam too narrow for labels
        label_y  = deck_z_mm * (0.50 if sname in ('AFT', 'FWD') else 0.72)
        ax.text(cx, label_y, display,
                ha='center', va='center', fontsize=label_fs,
                fontweight='bold', color='#111111')
        if sname not in ('AFT', 'FWD'):
            ax.text(cx, deck_z_mm * 0.18,
                    f'{(x1s - x0s) / 1000.0:.1f} m',
                    ha='center', va='center',
                    fontsize=label_fs * 0.85, color='#333333')

    # DB label
    ax.text(midbody_x1 + x_end * 0.01, db_z_mm,
            f'DB {db_z_mm / 1000.0:.2f} m',
            va='center', fontsize=label_fs * 0.8, color='#3344bb')

    # L label
    ax.text(x_end * 0.5, deck_z_mm + deck_z_mm * 0.12,
            f"L = {layout.get('L_m', x_end / 1000.0):.1f} m",
            ha='center', va='bottom',
            fontsize=label_fs + 1, fontweight='bold')

    # Waterline
    wl_z = D_mm * 0.70
    ax.plot([aft_seg['x0_mm'] - x_end * 0.01,
             fwd_seg['x1_mm'] + x_end * 0.02],
            [wl_z, wl_z],
            color='#0077cc', linewidth=0.8, linestyle='-.', alpha=0.6)
    ax.text(midbody_x1 + x_end * 0.01, wl_z + D_mm * 0.02,
            'W.L.', va='bottom',
            fontsize=label_fs * 0.7, color='#0077cc')

    # Title
    ax.set_title(f'{ship_type} — Compartment Arrangement',
                 fontsize=label_fs + 2)

    # Axis limits: x_max extends to bow tip
    x_max_all = max(p[0] for p in bow_deck)
    margin_x  = x_end * 0.04
    ax.set_xlim(-margin_x, x_max_all + margin_x * 2)
    ax.set_ylim(-D_mm * 0.04, deck_z_mm * 1.25)
    ax.set_aspect('equal')
    ax.set_xlabel('Length (mm)', fontsize=label_fs)
    ax.set_ylabel('Height (mm)', fontsize=label_fs)

    plt.tight_layout()
    plt.savefig(png_path, dpi=png_dpi, bbox_inches='tight')
    plt.close(fig)
