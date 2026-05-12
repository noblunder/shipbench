# =========================================
#   LNGC Midship Generator + Elevation + 3D Model (Self-contained)
#   - KR Rules 2025 Pt15 + Pt7 Ch5 (LNG Membrane Carriers)
#   - Full geometry class + rule framework + elevation + 3D + dataset generator
# =========================================

import os
import re
import csv
import json
import time
import random
from math import sin, cos, atan2, pi, hypot, tan, radians, degrees

import ezdxf
from ezdxf.lldxf import const as ezc

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


# ===============================
# DXF 레이어 상수/유틸
# ===============================
LAYERS = {
    "MEMBERS":          ("Members", 3),              # Green
    "LABEL":            ("Label", 1),                # Red
    "BILGE":            ("Bilge", 3),                # Green
    "COMPARTMENT":      ("Compartment", 6),          # Magenta
    "STIFFENERS_LONGI": ("Stiffeners (Longi)", 4),   # Cyan — longitudinal stiffeners
    "STIFFENERS_TRANS": ("Stiffeners (Trans)", 30),  # Orange — transverse indicators
    "CENTER":           ("Center", 8),               # Gray
    "SCANTLING":        ("Scantling", 252),          # Dark gray — scantling table
}


# ── Ship type identifier for hull-form renderer ──
_SHIP_TYPE = 'LNGC'

def ensure_layers(doc):
    """필요 레이어를 모두 보장하고 색상을 동기화."""
    for _, (name, color) in LAYERS.items():
        if name in doc.layers:
            layer = doc.layers.get(name)
        else:
            layer = doc.layers.add(name)
        layer.dxf.color = color


# ===============================
# 공통 유틸
# ===============================
EPS = 1e-9

# JSON 옵션: 개별 멤버 geometry에서 bbox_mm를 제외할지 여부 (True로 바꾸면 다시 포함)
EXPORT_INCLUDE_MEMBER_BBOX = False

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


def build_filename(base_dir, L, B, D, camber_upper, camber_trunk, d_ds, d_db, r_bilge,
                   lbhd_r, g1_r, g2_r, s1_r, s2_r, s3_r):
    name = (
        f"LNGC_L{fmt_token(L, 0)}_"
        f"B{fmt_token(B)}_D{fmt_token(D)}_"
        f"C{fmt_token(camber_upper)}_CT{fmt_token(camber_trunk)}_"
        f"DS{fmt_token(d_ds)}_DB{fmt_token(d_db)}_R{fmt_token(r_bilge)}_"
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


def line_intersection(p1, p2, p3, p4):
    """(y,z) 2D 직선 교점 (무한연장 교점). 없으면 None. (meters 기반)"""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < EPS:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    return (px, py)


def build_params_compact(p: dict) -> dict:
    """짧은 키(JSON/논문용)로 정리한 입력 파라미터 뷰"""
    return {
        # 기본 치수 (m)
        'L_m':  p['L'],
        'B_m':  p['B'],
        'D_m':  p['D'],
        'HL_m': p['HL'],

        # 캠버/구조 치수 (m)
        'camberUpper_m':  p['C'],
        'camberTrunk_m':  p['CT'],
        'doubleSide_m':   p['DS'],
        'doubleBottom_m': p['DB'],
        'bilgeRadius_m':  p['R'],

        # 비율 (무차원)
        'girderCL_ratio':  p['G0'],
        'girderB_ratio':   p['G1'],
        'girderOut_ratio': p['G2'],
        'str1_ratio': p['S1'],
        'str2_ratio': p['S2'],
        'str3_ratio': p['S3'],
    }


# ===============================
# 라벨/약어: 단일 레지스트리
# ===============================
def _norm_token(t: str) -> str:
    """CL, C.L., c l → CL 처럼 마침표/공백 제거 + 대문자화로 느슨 매칭 지원"""
    return re.sub(r'[^A-Za-z0-9]+', '', (t or '')).upper()

# DXF MTEXT attachment_point 코드 (group 71)
# 1=TopLeft, 2=TopCenter, 3=TopRight, 4=MiddleLeft, 5=MiddleCenter,
# 6=MiddleRight, 7=BottomLeft, 8=BottomCenter, 9=BottomRight
MTEXT_ATTACH = {
    "top_left": 1, "top_center": 2, "top_right": 3,
    "left": 4, "center": 5, "right": 6,
    "bottom_left": 7, "bottom_center": 8, "bottom_right": 9,
}

def _attach_from_align(val):
    """LABEL_REGISTRY의 align 값을 MTEXT attachment 정수로 변환."""
    if isinstance(val, int):
        # 이미 1~9를 직접 지정한 경우 그대로
        return val
    if not val:
        return None
    key = str(val).strip().lower()
    return MTEXT_ATTACH.get(key)


LABEL_REGISTRY = {
    # ---- Members ----
    "IBTM":            {"full": "Inner Bottom", "aliases": ["IBTM", "INNER_BOTTOM"], "kind": "member",
                        "label": {"side": "+", "offset": 500, "rotation": "parallel", "normal": "auto_up"}},
    "IHull":           {"full": "Inner Hull", "aliases": ["IHull", "INNER_HULL"],  "kind": "member",
                        "label": {"side": "-", "offset": 500, "rotation": "parallel", "normal": "out_y+"}},
    "B_Girder":        {"full": "Bottom Girder (C.L. +2600 mm)", "aliases": ["Girder", "B-Girder"], "kind": "member",
                        "label": {"side": "-", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "Out_Girder":      {"full": "Outer Girder (Outboard)", "aliases": ["Out_Girder", "Outer_Girder"], "kind": "member",
                        "label": {"side": "-", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "Str1":            {"full": "Side Longitudinal Stringer 1", "aliases": ["Str1"], "kind": "member",
                        "label": {"side": "+", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "Str2":            {"full": "Side Longitudinal Stringer 2", "aliases": ["Str2"], "kind": "member",
                        "label": {"side": "+", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "Str3":            {"full": "Side Longitudinal Stringer 3", "aliases": ["Str3"], "kind": "member",
                        "label": {"side": "+", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "Upper_Deck":      {"full": "Upper Deck (Cambered)", "aliases": ["Upp. Dk.", "UPPER_DECK"], "kind": "member",
                        "label": {"side": "+", "offset": 500, "rotation": "parallel", "normal": "auto_up", "align": "left"}},
    "Bottom_Shell":    {"full": "Bottom Shell Plate", "aliases": ["BTM", "BOTTOM"], "kind": "member",
                        "label": {"side": "-", "offset": 500, "rotation": "parallel", "normal": "auto_up"}},
    "Side_Shell":      {"full": "Side Shell Plate", "aliases": ["Side", "SIDE_SHELL"], "kind": "member",
                        "label": {"side": "+", "offset": 500, "rotation": "parallel", "normal": "out_y+"}},
    "Trunk_Deck":      {"full": "Trunk Deck Plate",  "aliases": ["T/D"], "kind": "member",
                        "label": {"side": "+", "offset": 450, "rotation": "parallel", "normal": "auto_up"}},
    "TrunkDeck_Slant": {"full": "Trunk Deck Slanted Plate", "aliases": ["T/D_Slant", "Trunk Dk."], "kind": "member",
                        "label": {"side": "+", "offset": 350, "rotation": "parallel", "normal": "out_y+"}},
    "InnerDeck_Slant": {"full": "Inner Deck Slanted Plate",  "aliases": ["Inn. Dk. Slanted"], "kind": "member",
                        "label": {"side": "-", "offset": 450, "rotation": "parallel", "normal": "auto_up"}},
    "InnerDeck_Flat":  {"full": "Inner Deck Flat Plate", "aliases": ["Inn. Dk. Flat"], "kind": "member",
                        "label": {"side": "-", "offset": 450, "rotation": "parallel", "normal": "auto_up"}},
    "U_Girder1":       {"full": "Upper Girder 1", "aliases": ["Upp. Gir.1"], "kind": "member",
                        "label": {"side": "-", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "U_Girder2":       {"full": "Upper Girder 2", "aliases": ["Upp. Gir.2"], "kind": "member",
                        "label": {"side": "+", "offset": 350, "rotation": "parallel", "normal": "auto_up"}},
    "Hopper":          {"full": "Hopper Plate", "aliases": ["HOPPER"], "kind": "member",
                        "label": {"side": "+", "offset": 450, "rotation": "parallel", "normal": "auto_up"}},
    "Bilge":           {"full": "Bilge Plate", "aliases": ["BILGE"], "kind": "member",
                        "label": {"side": "+", "offset": 0, "rotation": "parallel", "normal": "auto_up", "align": "left"}},

    # ---- Compartments / Tanks ----
    "D.S.W.B.T":       {"full": "Double Side Water Ballast Tank", "aliases": ["W.B.T", "DSWBT"], "kind": "compartment"},
    "D.B.W.B.T":       {"full": "Double Bottom Water Ballast Tank", "aliases": ["W.B.T", "DBWBT"], "kind": "compartment"},

    # ---- Others ----
    "C.L.":            {"full": "Centerline", "aliases": ["C/L", "CL", "CENTERLINE"], "kind": "other"},
    "STBD":            {"full": "Starboard Side", "aliases": ["STBD", "STARBOARD"], "kind": "other"},
}

# 레지스트리 버전(데이터셋 추적/재현성 표시용)
LABEL_REGISTRY_VERSION = "1.0"

# 라벨 배치 기본값(항목별 정의가 없을 때 fallback)
DEFAULT_LABEL_PREF = {"side": "+", "offset": 350, "rotation": "parallel", "normal": "auto_up", "align": "center"}

# ---- 내부 캐시(전역) ----
_LEGEND = {}       # canonical key -> full name
_ALIASES = {}      # normalized alias -> canonical key
_LABEL_PREFS = {}  # canonical key -> label pref
_LABEL_MAPS_BUILT = False

def _norm_token(s: str) -> str:
    """대소문자/구분자 무시 정규화(별칭 매칭용)."""
    import re
    return re.sub(r'[^A-Z0-9]+', '', str(s).upper())

def _build_label_maps():
    """LABEL_REGISTRY로부터 파생 맵 3종을 1회 빌드."""
    global _LEGEND, _ALIASES, _LABEL_PREFS, _LABEL_MAPS_BUILT
    if _LABEL_MAPS_BUILT:
        return

    # 1) key -> full name
    _LEGEND = {k: (meta.get("full") or k) for k, meta in LABEL_REGISTRY.items()}

    # 2) 정규화된 alias -> canonical key
    alias_map = {}
    for key, meta in LABEL_REGISTRY.items():
        aliases = set([key] + meta.get("aliases", []))
        for a in aliases:
            alias_map[_norm_token(a)] = key
    _ALIASES = alias_map

    # 3) key -> label pref (기본값 병합)
    _LABEL_PREFS = {
        k: {**DEFAULT_LABEL_PREF, **(meta.get("label") or {})}
        for k, meta in LABEL_REGISTRY.items()
    }

    _LABEL_MAPS_BUILT = True

# ---- 공개 유틸 ----
def legend_full_name(key: str) -> str:
    """canonical key로 full name을 얻는다(없으면 key 반환)."""
    _build_label_maps()
    return _LEGEND.get(key, key)

def key_from_token(token: str) -> str:
    """임의 토큰(약어/별칭)에서 canonical key로 변환(모르면 원문)."""
    _build_label_maps()
    return _ALIASES.get(_norm_token(token), token)

def expand_abbrev(token: str) -> str:
    """약어/별칭을 full name으로 확장(모르면 원문)."""
    _build_label_maps()
    return legend_full_name(key_from_token(token))

def get_label_pref(key: str) -> dict:
    """라벨 배치 선호 파라미터(항목 없으면 기본값)."""
    _build_label_maps()
    return _LABEL_PREFS.get(key, DEFAULT_LABEL_PREF)

def clean_multiline_label(label: str) -> str:
    """CAD 멀티라인(\\P) → 공백 정리."""
    return label.replace("\\P", " ").replace("  ", " ").strip()

# (선택) 하위호환: 기존 코드가 이 심볼들을 참조한다면 유지
def build_abbrev_maps(_reg_unused=None):
    """[Deprecated] 하위호환용: 예전 시그니처 유지."""
    _build_label_maps()
    # exact: alias 그대로 -> full name
    exact = {}
    for key, meta in LABEL_REGISTRY.items():
        full = _LEGEND[key]
        for a in set([key] + meta.get("aliases", [])):
            exact[a] = full
    # norm: 정규화 alias -> full name
    norm = { _norm_token(a): exact[a] for a in exact.keys() }
    return exact, norm

# 하위호환 변수를 꼭 써야 한다면 이렇게 바인딩만 해둠(실제 소스는 위 캐시)
ABBREV_MAP, ABBREV_MAP_NORM = build_abbrev_maps()
LABEL_PREFS = _LABEL_PREFS


# ===============================
# LHS 표본추출 (간단 구현)
# ===============================
def lhs_samples(N, specs, seed=None):
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


# ===============================
# 단면 좌표 모델 (LNGC with Inner/Trunk Deck)
# ===============================
class LNGC:
    def __init__(self, L, B, D,
                 d_ds, d_db, h_camber,
                 y_0gir, y_1gir, y_2gir,
                 z_1str, z_2str, z_3str, r_bilge,
                 outg_clear=0.8,
                 # ---- 규칙/형상 파라미터(조정 가능) ----
                 inner_slope_deg=45.0,
                 inner_slope_len_factor=0.5,
                 side_slope_from_z_deg=-35.0,
                 trunk_above_inner_mm=2000.0,
                 h_camber_trunk=0.0
                 ):

        self.L = L; self.B = B; self.D = D
        self.r_bilge = r_bilge; self.h_camber = h_camber
        self.d_ds = d_ds; self.d_db = d_db
        self.outg_clear = max(0.0, float(outg_clear))
        # Expose IGC-relevant geometry params for KR rule evaluator
        self.inner_slope_deg = float(inner_slope_deg)
        self.inner_slope_len_factor = float(inner_slope_len_factor)

        # 기본 위치들(m)
        self.y_ihull = self.B/2 - self.d_ds
        self.z_1str = z_1str
        self.z_2str = z_2str
        self.z_3str = z_3str

        # ---- Out_Girder 안전 위치(m) ----
        y_bilge_toe = self.B/2 - self.r_bilge
        y2_safe_max = y_bilge_toe - self.outg_clear
        y2_clamped = min(y_2gir, y2_safe_max)
        if y2_clamped < 0.0:
            y2_clamped = 0.0

        # ---- Inboard Girder (B_Girder, y_1gir) ----
        # IHull 안쪽으로, Out_Girder보다 0.1 m 이상 안쪽 유지
        y1_safe_max = min(self.y_ihull - self.outg_clear, y2_clamped - 0.1)
        y1_clamped = max(0.0, min(y_1gir, y1_safe_max))

        # ---- Centerline Girder (CL_Girder, y_0gir) ----
        # B_Girder보다 0.1 m 이상 안쪽 유지
        y0_safe_max = max(0.0, y1_clamped - 0.1)
        y0_clamped = max(0.0, min(y_0gir, y0_safe_max))

        self.y_2gir = y2_clamped
        self.y_1gir = y1_clamped
        self.y_0gir = y0_clamped

        # ---- 단위/스케일(mm) ----
        Ls = self.L * 500

        # ===========================
        # 외곽 기본(바텀/사이드/캠버 덱)
        # ===========================
        self.memb_btm  = [[Ls, Ls], [0, (self.B/2 - self.r_bilge)*1000], [0, 0]]
        self.memb_side = [[Ls, Ls], [self.B*500, self.B*500], [self.r_bilge*1000, self.D*1000]]

        # Upper Deck: 일단 전체 생성, 이후 slant와 교차 지점까지 절단
        def z_deck(y_m):  # m
            return -(self.h_camber / (self.B/2)) * y_m + (self.D + self.h_camber)
        self.z_deck = z_deck  # expose

        self.h_camber_trunk = float(h_camber_trunk)

        def z_trunk_camber(y_m):  # m
            # CL에서 z = z_trunk + CT, y 증가할수록 선형으로 감소
            if self.B <= 0:
                return self.z_trunk
            return self.z_trunk + self.h_camber_trunk - (self.h_camber_trunk / (self.B / 2)) * y_m

        self.z_trunk_camber = z_trunk_camber  # expose

        # ===========================
        # 내부 구조: IHull + Inner Decks
        # ===========================
        yI = self.y_ihull
        z1 = self.z_1str
        z3 = self.z_3str

        th = radians(inner_slope_deg)
        s_nom = inner_slope_len_factor * self.D
        s_max = yI / max(cos(th), 1e-9)
        s = max(0.0, min(s_nom, s_max))
        y_sl_end = max(0.0, yI - s * cos(th))         # m
        z_sl_end = z1 + s * sin(th)                   # m

        y_flat_L = y_sl_end
        z_flat = z_sl_end
        z_trunk = z_flat + (trunk_above_inner_mm / 1000.0)

        self.y_sl_end = y_sl_end            # m
        self.z_flat = z_flat                # m
        self.z_trunk = z_trunk              # m

        # Upper Deck 절단
        m_sl = tan(th)
        den = (2 * self.h_camber / self.B) - m_sl
        if abs(den) > 1e-9:
            y_u = (self.D + self.h_camber - z1 - m_sl * yI) / den
        else:
            y_u = self.B / 2

        y_u = min(max(y_u, 0.0), self.B / 2)
        z_u = z_deck(y_u)

        self.memb_deck = [[Ls, Ls],
                          [y_u * 1000, self.B * 500],
                          [z_u * 1000, self.D * 1000]]

        # TrunkDeck_Slant
        y_start_side = max(0.0, yI - 0.8)
        z_start_side = z_deck(y_start_side)

        self.y_start_side = y_start_side
        self.z_start_side = z_start_side

        phi = radians(abs(side_slope_from_z_deg))
        dy_s = +sin(phi)
        dz_s = -cos(phi)

        t_tr = (z_trunk - z_start_side) / dz_s if abs(dz_s) > 1e-9 else 0.0
        y_tr_meet = min(self.B / 2, max(0.0, y_start_side + dy_s * t_tr))

        z_trunk_y0 = self.z_trunk_camber(0.0)
        z_trunk_y_meet = self.z_trunk_camber(y_tr_meet)

        # Trunk Deck
        self.memb_trunk = [[Ls, Ls], [0, y_tr_meet * 1000], [z_trunk_y0 * 1000, z_trunk_y_meet * 1000]]

        # Side trunk slant
        self.memb_trunkdeck_slant = [[Ls, Ls], [y_start_side*1000, y_tr_meet*1000], [z_start_side*1000, z_trunk_y_meet*1000]]


        # 나머지 구조
        self.memb_ibtm = [[Ls, Ls], [0, self.y_2gir*1000], [self.d_db*1000, self.d_db*1000]]
        self.memb_ihull = [[Ls, Ls], [yI*1000, yI*1000], [z3*1000, z1*1000]]
        self.memb_hopp = [[Ls, Ls], [self.y_2gir*1000, yI*1000], [self.d_db*1000, z3*1000]]

        # Inner Decks
        self.memb_inner_slope = [[Ls, Ls], [yI*1000, y_sl_end*1000], [z1*1000, z_sl_end*1000]]
        if y_sl_end > 0.0:
            self.memb_inner_flat = [[Ls, Ls], [y_sl_end*1000, 0], [z_flat*1000, z_flat*1000]]
        else:
            self.memb_inner_flat = None

        # Girders
        self.memb_b_gir0 = [[Ls, Ls], [self.y_0gir*1000, self.y_0gir*1000], [0, self.d_db*1000]]
        self.memb_b_gir1 = [[Ls, Ls], [self.y_1gir*1000, self.y_1gir*1000], [0, self.d_db*1000]]
        self.memb_b_gir2 = [[Ls, Ls], [self.y_2gir*1000, self.y_2gir*1000], [0, self.d_db*1000]]

        # Upper Girders
        self.memb_u_gir1 = None
        self.memb_u_gir2 = None

        if self.memb_inner_flat is not None:
            # 거더가 올라갈 수 있는 최대 y는 (평판내측 끝)과 (슬랜트 접점) 사이
            y_max_m = min(self.y_sl_end, y_tr_meet)

            # ---- U_Girder1: y = min(3.0 m, y_max_m)
            if y_max_m > 0.0:
                y_u1_m = min(3.0, y_max_m)
                # 상단 z는 그 y에서의 트렁크 캠버 높이
                z_top1 = self.z_trunk_camber(y_u1_m)
                self.memb_u_gir1 = [[Ls, Ls],
                                    [y_u1_m * 1000, y_u1_m * 1000],
                                    [self.z_flat * 1000, z_top1 * 1000]]

            # ---- U_Girder2: 기존 y = self.y_sl_end → 슬랜트 접점(y_tr_meet) 넘지 않도록 클램프
            y_u2_m = min(self.y_sl_end, y_tr_meet)
            if y_u2_m > 0.0:
                z_top2 = self.z_trunk_camber(y_u2_m)
                self.memb_u_gir2 = [[Ls, Ls],
                                    [y_u2_m * 1000, y_u2_m * 1000],
                                    [self.z_flat * 1000, z_top2 * 1000]]

        # Stringers
        B2_mm = (self.B/2) * 1000
        self.memb_1str = [[Ls, Ls], [yI*1000, B2_mm], [self.z_1str*1000, self.z_1str*1000]]
        self.memb_2str = [[Ls, Ls], [yI*1000, B2_mm], [self.z_2str*1000, self.z_2str*1000]]
        self.memb_3str = [[Ls, Ls], [yI*1000, B2_mm], [self.z_3str*1000, self.z_3str*1000]]

        # 멤버 사전
        self.members = {
            "Bottom_Shell":    self.memb_btm,
            "Side_Shell":      self.memb_side,
            "Upper_Deck":      self.memb_deck,
            "IBTM":            self.memb_ibtm,
            "IHull":           self.memb_ihull,
            "InnerDeck_Slant": self.memb_inner_slope,
            "Trunk_Deck":      self.memb_trunk,
            "TrunkDeck_Slant": self.memb_trunkdeck_slant,
            "Hopper":          self.memb_hopp,
            "CL_Girder":       self.memb_b_gir0,
            "B_Girder":        self.memb_b_gir1,
            "Out_Girder":      self.memb_b_gir2,
            "Str1":            self.memb_1str,
            "Str2":            self.memb_2str,
            "Str3":            self.memb_3str,
        }
        if self.memb_inner_flat is not None:
            self.members["InnerDeck_Flat"] = self.memb_inner_flat
        if self.memb_u_gir1 is not None:
            self.members["U_Girder1"] = self.memb_u_gir1
        if self.memb_u_gir2 is not None:
            self.members["U_Girder2"] = self.memb_u_gir2

    def z_deck(self, y):
        """캠버 적용된 상갑판 높이"""
        return -(self.h_camber / (self.B/2)) * y + (self.D + self.h_camber)

    def seg_dict(self):
        """(y,z) 두 점 튜플로 구성된 선분 사전 반환"""
        def seg(m):
            return ((float(m[1][0]), float(m[2][0])), (float(m[1][1]), float(m[2][1])))
        return {name: seg(m) for name, m in self.members.items()}


# ===============================
# 라벨 유틸
# ===============================
def add_label_along_line(msp, start, end, text_str, height=250, offset=500, layer=None, color=None):
    import math
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    if angle_deg > 90 or angle_deg < -90:
        angle_deg += 180
        start, end = end, start
        dx = end[0] - start[0]
        dy = end[1] - start[1]

    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    nx, ny = -dy, dx
    nlen = hypot(nx, ny)
    if nlen != 0:
        nx, ny = nx/nlen, ny/nlen
    if ny < 0:
        nx, ny = -nx, -ny

    label_x = mid_x + nx * offset
    label_y = mid_y + ny * offset

    dxfattribs = {'height': height, 'rotation': angle_deg, 'halign': 1, 'valign': 2, 'insert': (label_x, label_y)}
    if layer is not None:
        dxfattribs['layer'] = layer
    if color is not None:
        dxfattribs['color'] = color

    text = msp.add_text(text_str, dxfattribs=dxfattribs)
    text.dxf.align_point = (label_x, label_y)


# ===============================
#   Stiffener type / scantling config
# ===============================
_STF_CFG = {
    # (stf_type, flange_half_mm, web_h_mm)  ← web_h from _SCANTLING_TABLE
    "Upper_Deck":       ("T",  75, 350),  # 350 x 12 + 150 x 20 F.B(T)
    "Trunk_Deck":       ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)
    "Bottom_Shell":     ("T",  75, 400),  # 400 x 14 + 150 x 22 F.B(T)
    "Side_Shell":       ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)
    "IHull":            ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)  (Inner Hull)
    "IBTM":             ("T",  75, 380),  # 380 x 14 + 150 x 20 F.B(T)  (Inner Bottom)
    "Hopper":           ("IA", 90, 280),  # 280 x 10 + 90 x 14 I.A
    "Str3":             ("IA", 90, 280),  # 280 x 10 + 90 x 14 I.A  (Str3 Hopper)
    "CL_Girder":        ("FB",  0, 200),  # 200 x 12 F.B
    "Out_Girder":       ("FB",  0, 150),  # 150 x 10 F.B
    # Not in table — reasonable defaults:
    "InnerDeck_Flat":   ("T",  65, 300),
    "InnerDeck_Slant":  ("IA", 65, 280),
    "TrunkDeck_Slant":  ("IA", 65, 280),
    "B_Girder":         ("FB",  0, 200),
    "U_Girder1":        ("FB",  0, 200),
    "U_Girder2":        ("FB",  0, 200),
    "Str1":             ("IA", 50, 200),
    "Str2":             ("IA", 50, 200),
}

_STF_TYPE_LEGEND = {
    "F.B":    "Flat Bar — web only, no flange",
    "I.A":    "Inverted Angle — web + one-side flange (L-shape)",
    "F.B(T)": "Built-up T-bar — web + both-side flanges (T-shape)",
}

_SCANTLING_TABLE = [
    ("MEMBER",           "PLATE (mm)", "STIFFENER"),
    ("Upper Deck",       "14.0",       "350 x 12 + 150 x 20 F.B(T)"),
    ("Trunk Deck",       "13.0",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Bottom Shell",     "18.0",       "400 x 14 + 150 x 22 F.B(T)"),
    ("Side Shell",       "15.0",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Inner Hull",       "14.0",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Inner Bottom",     "16.0",       "380 x 14 + 150 x 20 F.B(T)"),
    ("Hopper",           "12.0",       "280 x 10 + 90 x 14 I.A"),
    ("Str3 (Hopper)",    "13.0",       "280 x 10 + 90 x 14 I.A"),
    ("CL Girder",        "12.0",       "200 x 12 F.B"),
    ("Out Girder",       "11.0",       "150 x 10 F.B"),
]

# ===============================
# DXF 내보내기(+ stiffeners, + PNG) — 메타데이터 대폭 강화
# ===============================
class DXFExporterMM:
    def __init__(self, ship: LNGC, text_height=250, offset=300,
                 stf_min=700, stf_max=1000, stf_target=850, stf_len=400, edge_clear=10,
                 hold_length_m=None, hold_vol_factor=0.7, number_of_hold=3):
        self.ship = ship
        self.text_height = text_height
        self.offset = offset
        self.placed_label_polys = []
        self.doc = ezdxf.new(setup=True)
        self.msp = self.doc.modelspace()
        # stiffener
        self.stf_min = stf_min
        self.stf_max = stf_max
        self.stf_target = stf_target
        self.stf_len = stf_len
        self.edge_clear = edge_clear

        self.hold_length_m = hold_length_m
        self.hold_vol_factor = hold_vol_factor
        self.number_of_hold = number_of_hold

        # 방향 테이블
        self.STF_DIR = {
            "Upper_Deck":        (0.0, -1.0),
            "Trunk_Deck":        (0.0, -1.0),
            "InnerDeck_Flat":    (0.0, -1.0),
            "Bottom_Shell":      (0.0, +1.0),
            "IBTM":              (0.0, -1.0),
            "Side_Shell":        (-1.0, 0.0),  # inboard
            "IHull":             (+1.0, 0.0),
            "TrunkDeck_Slant":   (-1.0, 0.0),
            "CL_Girder":         (+1.0, 0.0),
            "B_Girder":          (-1.0, 0.0),
            "Out_Girder":        (-1.0, 0.0),
            "U_Girder1":         (-1.0, 0.0),
            "U_Girder2":         (+1.0, 0.0),
            "Str1":              (0.0, -1.0),
            "Str2":              (0.0, -1.0),
            "Str3":              (0.0, -1.0),
            "Bilge": None,
        }

        # ===== 도면/메타 수집 =====
        self.label_records = []
        self.stf_stats = {}
        self.export_stats = {}
        self.compartment_data = []
        self.intersections = []


    # ---------- 기하 보조 ----------
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

    @staticmethod
    def _poly_area_perimeter(verts):
        """verts: [(y,z), ...] mm 단위. 반환: (area_mm2[>=0], perimeter_mm_excl_CL)"""
        n = len(verts)
        if n < 3:
            return 0.0, 0.0
        area2 = 0.0
        perim = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i+1) % n]
            area2 += x1*y2 - x2*y1
            perim += hypot(x2-x1, y2-y1)
        return abs(area2) * 0.5, perim

    # ---- Title & Specs  ----
    def draw_title_and_specs(self, title: str = "ORDINARY SECTION (STBD)"):
        # CL에서의 상갑판 z(mm) 추출 (LNGC는 memb_deck가 존재)
        try:
            trunk_deck_z = float(self.ship.memb_trunk[2][0])  # at C.L., mm
        except Exception:
            trunk_deck_z = 0.0

        base_z = trunk_deck_z + 5000.0  # 제목을 갑판 위로 띄움
        center_y = 0.0  # 중심선(C.L.) 정렬

        def put_line(text, dy_mult, size_mult=1.0):
            char_h = self.text_height * size_mult
            ty = base_z - self.text_height * dy_mult
            t = self.msp.add_mtext(text, dxfattribs={'char_height': char_h, 'layer': 'Label'})
            t.dxf.insert = (center_y, ty)
            t.dxf.attachment_point = 5  # middle center
            t.dxf.rotation = 0
            # 라벨 기록
            self.label_records.append({
                'name': text, 'pos': (float(center_y), float(ty)),
                'rotation_deg': 0.0, 'layer': 'Label'
            })

        # Title
        put_line(title, dy_mult=-0.0, size_mult=1.5)

        # BREADTH, DEPTH only — section drawing excludes longitudinal info
        # (NUMBER OF HOLD / HOLD LENGTH / SHIP LENGTH belong to compartment view).
        try:
            B = float(self.ship.B)
            D = float(self.ship.D)
        except Exception:
            B = D = None
        put_line(f"BREADTH = {B:.1f} m" if B is not None else "BREADTH = -", dy_mult=2.6)
        put_line(f"DEPTH = {D:.1f} m" if D is not None else "DEPTH = -",     dy_mult=4.4)

    def draw_centerline(self):
        trunk_deck_z = self.ship.memb_trunk[2][0]
        cl_top_z = trunk_deck_z + 500

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

        # 라벨 기록
        self.label_records.append({
            'name': 'C.L.', 'pos': (float(t.dxf.insert[0]), float(t.dxf.insert[1])),
            'rotation_deg': 90.0, 'layer': 'Label'
        })

    def draw_line_mm(self, y_coords, z_coords, label=None, side="+", offset=None,
                     rotation_mode="parallel", normal_policy="auto_up", attach_point=5):
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
        angle_deg = angle_rad * 180.0 / pi

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
        txt.dxf.attachment_point = attach_point
        txt.dxf.rotation = text_rot

        # 라벨 기록
        self.label_records.append({
            'name': label, 'pos': (ly, lz), 'rotation_deg': text_rot, 'layer': 'Label'
        })

        self.placed_label_polys.append(curr_poly)

    def draw_members(self):
        _build_label_maps()  # ← 보장
        for name, memb in self.ship.members.items():
            y_coords = memb[1];
            z_coords = memb[2]
            pref = get_label_pref(name)

            ap = _attach_from_align(pref.get("align"))
            if ap is None:
                ap = pref.get("attach_point", 5)  # 숫자를 직접 쓴 기존 값이 있으면 활용, 없으면 5(중앙)

            self.draw_line_mm(
                y_coords, z_coords, label=name,
                side=pref.get("side", "+"),
                offset=pref.get("offset", self.offset),
                rotation_mode=pref.get("rotation", "parallel"),
                normal_policy=pref.get("normal", "auto_up"),
                attach_point=ap,
            )

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

        # 저장: stiffener용
        self.bilge_center = (cy, cz)
        self.bilge_R = R
        self.bilge_ang0 = start_angle
        self.bilge_ang1 = end_angle

        self.msp.add_arc(center=(cy, cz), radius=R, start_angle=start_deg, end_angle=end_deg, dxfattribs={'layer': 'Bilge'})
        mid_angle = (start_angle + end_angle) / 2
        label_x = cy + (R + 300) * cos(mid_angle); label_z = cz + (R + 300) * sin(mid_angle)
        self.msp.add_mtext("Bilge", dxfattribs={'char_height': self.text_height, 'layer': 'Label'}).set_location((label_x, label_z), rotation=0)

        # 라벨 기록
        self.label_records.append({
            'name': 'Bilge', 'pos': (label_x, label_z), 'rotation_deg': 0.0, 'layer': 'Label'
        })

    @staticmethod
    def _seg_intersection(p1, p2, q1, q2):
        """두 유한 선분(p1-p2, q1-q2)의 교점. 없으면 None."""
        def on_seg(a, b, p):
            minx, maxx = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
            miny, maxy = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
            return (minx - EPS <= p[0] <= maxx + EPS) and (miny - EPS <= p[1] <= maxy + EPS)

        ip = line_intersection(p1, p2, q1, q2)
        if ip is None:
            return None
        if on_seg(p1, p2, ip) and on_seg(q1, q2, ip):
            return ip
        return None

    @staticmethod
    def _poly_centroid(verts):
        """단순 폴리곤의 기하학적 무게중심(2D). verts: [(y,z), ...] CCW/시계 상관없음."""
        A = 0.0
        Cy = 0.0
        Cz = 0.0
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            cross = x1 * y2 - x2 * y1
            A += cross
            Cy += (x1 + x2) * cross
            Cz += (y1 + y2) * cross
        A *= 0.5
        if abs(A) < EPS:
            sy = sum(v[0] for v in verts) / n
            sz = sum(v[1] for v in verts) / n
            return sy, sz
        return Cy / (6 * A), Cz / (6 * A)

    def draw_compartments(self):
        # 레이어 준비
        if "Compartment" not in self.doc.layers:
            self.doc.layers.add("Compartment").dxf.color = 6  # Magenta

        # ---- 선분 사전 수집 (외곽 + 탱크 경계) ----
        S = getattr(self, "segs_all", self.ship.seg_dict())

        # 이름 매핑
        if "Trunk_Deck" in S:
            S["Tank_Top"] = S["Trunk_Deck"]
        if "TrunkDeck_Slant" in S:
            S["Tank_TSWT"] = S["TrunkDeck_Slant"]
            S["TSWT"] = S["TrunkDeck_Slant"]
        if "IHull" in S:
            S["Tank_Side"] = S["IHull"]
        if "Hopper" in S:
            S["Tank_Hopper"] = S["Hopper"]
        if "IBTM" in S:
            S["Tank_Bottom"] = S["IBTM"]

        # CL 수직선
        z_candidates_mm = [self.ship.z_deck(0) * 1000.0]
        if hasattr(self.ship, "z_trunk_camber") and callable(self.ship.z_trunk_camber):
            z_candidates_mm.append(self.ship.z_trunk_camber(0.0) * 1000.0)

        for m in self.ship.members.values():
            # m[2]는 z좌표(mm)
            z_candidates_mm.extend([float(m[2][0]), float(m[2][1])])


        deck_top_cl = max(z_candidates_mm) + 1000.0  # mm
        S["CL"] = ((0.0, 0.0), (0.0, deck_top_cl))

        # Bilge chord
        if hasattr(self, "bilge_bottom_end") and hasattr(self, "bilge_side_start"):
            if self.bilge_bottom_end and self.bilge_side_start:
                S["Bilge"] = (self.bilge_bottom_end, self.bilge_side_start)

        # TSWT_V 수직선
        yv = getattr(self.ship, "y_start_side", None)
        zv = getattr(self.ship, "z_start_side", None)
        if yv is not None and zv is not None:
            top_z = self.ship.z_deck(yv) * 1000.0
            S["TSWT_V"] = ((yv * 1000.0, 0.0), (yv * 1000.0, top_z))

        # 폴리곤 생성 + 메타
        def make_poly(name, edges):
            verts = []
            for A, B in edges:
                if A not in S or B not in S:
                    return None
                p1, p2 = S[A]
                q1, q2 = S[B]
                ip = self._seg_intersection(p1, p2, q1, q2)
                if ip is None:
                    return None
                verts.append((float(ip[0]), float(ip[1])))

            if len(verts) < 3:
                return None

            area_mm2, _perim_dummy = self._poly_area_perimeter(verts)
            area_m2 = area_mm2 / 1e6

            def shared_edge(ei, ej):
                a1, b1 = ei
                a2, b2 = ej
                s1 = {a1, b1}
                s2 = {a2, b2}
                inter = s1.intersection(s2)
                return next(iter(inter)) if inter else None

            perim_excl_cl = 0.0
            n = len(verts)
            for i in range(n):
                j = (i + 1) % n
                v1 = verts[i]
                v2 = verts[j]
                common = shared_edge(edges[i], edges[j])
                if common == "CL":
                    continue  # C.L. 변은 둘레에서 제외
                perim_excl_cl += hypot(v2[0] - v1[0], v2[1] - v1[1])

            cy, cz = self._poly_centroid(verts)

            # Cargo tank 라벨은 draw_scantling_table()에서 갑판↔테이블 사이에 배치
            if "cargo" not in clean_multiline_label(name).lower():
                t = self.msp.add_mtext(name, dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
                t.dxf.insert = (cy, cz)
                t.dxf.attachment_point = 5
                t.dxf.rotation = 0
                self.label_records.append({'name': name, 'pos': (cy, cz), 'rotation_deg': 0.0, 'layer': 'Compartment'})

            meta = {
                "raw_label": name,
                "clean_label": clean_multiline_label(name),
                "centroid_mm": (round(cy, 3), round(cz, 3)),
                "vertices_mm": [(round(v[0], 3), round(v[1], 3)) for v in verts],
                "area_mm2": round(area_mm2, 3),
                "area_m2": round(area_m2, 6),
                # perimeter는 'C.L. 제외값'으로 기록
                "perimeter_mm_excl_CL": round(perim_excl_cl, 3),
                "edges_used": edges,
            }
            return meta

        comp_defs = [
            ("Cargo tank\\P(Membrane)", [
                ("InnerDeck_Flat", "InnerDeck_Slant"),
                ("InnerDeck_Slant", "IHull"),
                ("IHull", "Hopper"),
                ("Hopper", "IBTM"),
                ("IBTM", "CL"),
                ("CL", "InnerDeck_Flat"),
            ]),
            ("Trunk void 1\\P(C)", [
                ("Trunk_Deck", "U_Girder1"),
                ("U_Girder1", "InnerDeck_Flat"),
                ("InnerDeck_Flat", "CL"),
                ("CL", "Trunk_Deck"),
            ]),
            ("Trunk void 2\\P(C)", [
                ("Trunk_Deck", "U_Girder2"),
                ("U_Girder2", "InnerDeck_Flat"),
                ("InnerDeck_Flat", "U_Girder1"),
                ("U_Girder1", "Trunk_Deck"),
            ]),
            ("Trunk\\Pvoid 3\\P(S)", [
                ("Trunk_Deck", "TrunkDeck_Slant"),
                ("TrunkDeck_Slant", "Upper_Deck"),
                ("Upper_Deck", "InnerDeck_Slant"),
                ("InnerDeck_Slant", "U_Girder2"),
                ("U_Girder2", "Trunk_Deck"),
            ]),
            ("Void\\P(S)", [
                ("Upper_Deck", "Side_Shell"),
                ("Side_Shell", "Str1"),
                ("Str1", "InnerDeck_Slant"),
                ("InnerDeck_Slant", "Upper_Deck"),
            ]),
            ("Ballast\\Ptank 1\\P(D.S.W.B.T)", [
                ("Str1", "Side_Shell"),
                ("Side_Shell", "Str2"),
                ("Str2", "IHull"),
                ("IHull", "Str1"),
            ]),
            ("Ballast\\Ptank 2\\P(D.S.W.B.T)", [
                ("Str2", "Side_Shell"),
                ("Side_Shell", "Str3"),
                ("Str3", "IHull"),
                ("IHull", "Str2"),
            ]),
            ("Ballast\\Ptank 3", [
                ("Str3", "Side_Shell"),
                ("Side_Shell", "Bilge"),
                ("Bilge", "Bottom_Shell"),
                ("Bottom_Shell", "Out_Girder"),
                ("Out_Girder", "Hopper"),
                ("Hopper", "Str3"),
            ]),
            ("Ballast tank 4\\P(D.B.W.B.T)", [
                ("IBTM","Out_Girder"),
                ("Out_Girder","Bottom_Shell"),
                ("Bottom_Shell","B_Girder"),
                ("B_Girder","IBTM"),
            ]),
            ("Pipe\\Pduct", [
                ("IBTM","B_Girder"),
                ("B_Girder","Bottom_Shell"),
                ("Bottom_Shell","CL"),
                ("CL","IBTM"),
            ]),
        ]

        for nm, edges in comp_defs:
            meta = make_poly(nm, edges)
            if meta is not None:
                self.compartment_data.append(meta)

        # 교차점 수집
        segs = S.copy()
        keys = sorted(segs.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                p1, p2 = segs[k1]
                q1, q2 = segs[k2]
                ip = self._seg_intersection(p1, p2, q1, q2)
                if ip is not None:
                    self.intersections.append({
                        "a": k1, "b": k2,
                        "point_mm": (round(ip[0], 3), round(ip[1], 3))
                    })

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
                if best[0] == 0 or abs(s - self.stf_target) < abs(best[1] - self.stf_target):
                    best = (n, s)
        return best

    def _split_by_intersections(self, name, seg_dict_for_split):
        p1, p2 = seg_dict_for_split[name]
        uy, uz, L = self._seg_dir_len(p1, p2)
        if L < EPS: return []
        ts = [0.0, L]
        for other, (q1, q2) in seg_dict_for_split.items():
            if other == name: continue
            ip = line_intersection(p1, p2, q1, q2)
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

    def _even_points_on_arc(self, center, R, ang0, ang1):
        cy, cz = center
        span = ang1 - ang0
        if span <= 1e-9 or R <= 1e-6:
            return []
        L = abs(R * span)
        n, spacing = self._choose_n_for_spacing(L)
        if n <= 0:
            return []
        pts = []
        t0 = self.edge_clear
        for i in range(1, n+1):
            s = t0 + spacing * i
            if s >= L - self.edge_clear + 1e-6:
                break
            theta = ang0 + (s / L) * span
            y = cy + R * cos(theta)
            z = cz + R * sin(theta)
            pts.append((y, z))
        return pts

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

        gen_names = ["Upper_Deck", "Trunk_Deck", "InnerDeck_Flat", "InnerDeck_Slant",
                     "Bottom_Shell", "IBTM", "Side_Shell", "TrunkDeck_Slant",
                     "IHull", "Hopper", "B_Girder", "Out_Girder", "Str1", "Str2", "Str3",
                     "U_Girder1", "U_Girder2"]
        gen_names = [n for n in gen_names if n in segs_all]

        # ✅ 분할용 사전은 ‘부재만’ 필터링해서 사용
        segs_for_split = {k: segs_all[k] for k in gen_names}

        # ---- InnerDeck_Slant ∩ Upper_Deck t-space exclusion (BULKC TSWT∩SS 방식) ----
        # CLEAR_ZONE=1500mm → flange-tip 400mm 이상 여유 (T-bar web~350mm 기준)
        ID_UD_CLEAR_ZONE = 1500.0
        _id_ud_excl = {}  # {member_name: [(t_lo, t_hi), ...]}

        if "InnerDeck_Slant" in segs_all and "Upper_Deck" in segs_all:
            _ip = line_intersection(*segs_all["InnerDeck_Slant"], *segs_all["Upper_Deck"])
            if _ip is not None:
                for _nm in ("InnerDeck_Slant", "Upper_Deck"):
                    _p1, _p2 = segs_all[_nm]
                    _uy, _uz, _L = self._seg_dir_len(_p1, _p2)
                    _t_q = (_ip[0]-_p1[0])*_uy + (_ip[1]-_p1[1])*_uz
                    _t_q = max(0.0, min(_L, _t_q))
                    if -1e-6 <= _t_q <= _L + 1e-6:
                        if _nm == "InnerDeck_Slant":
                            # 단방향: 예각 방향(t < t_q, 교점 아래쪽)만 제거
                            # 둔각 방향(t > t_q, 교점 위쪽)은 유지
                            _t_lo = max(0.0, _t_q - ID_UD_CLEAR_ZONE)
                            _t_hi = _t_q
                        else:  # Upper_Deck: 교점이 시작점 근처(t_q≈0) → 교점 이후 방향 제거
                            _t_lo = max(0.0, _t_q - ID_UD_CLEAR_ZONE)
                            _t_hi = min(_L, _t_q + ID_UD_CLEAR_ZONE)
                        _id_ud_excl.setdefault(_nm, []).append((_t_lo, _t_hi))

        def _in_id_ud_excl(nm, pt):
            zones = _id_ud_excl.get(nm)
            if not zones:
                return False
            _p1, _p2 = segs_all[nm]
            _uy, _uz, _ = self._seg_dir_len(_p1, _p2)
            _t_g = (pt[0]-_p1[0])*_uy + (pt[1]-_p1[1])*_uz
            return any(_t_lo - 1.0 <= _t_g <= _t_hi + 1.0 for _t_lo, _t_hi in zones)

        for name in gen_names:
            self.stf_stats.setdefault(name, 0)
            pieces = self._split_by_intersections(name, segs_for_split)
            if not pieces:
                continue

            stf_type, flange_half, web_h = _STF_CFG.get(name, ("FB", 0, 400))
            for s, e in pieces:
                uy, uz, _ = self._seg_dir_len(s, e)

                n1 = (-uz, uy)
                n2 = (uz, -uy)

                if name in ("Upper_Deck", "Trunk_Deck", "Str1", "Str2", "Str3", "IBTM"):
                    nvec = n1 if n1[1] < 0 else n2
                elif name in ("InnerDeck_Flat", "InnerDeck_Slant"):
                    nvec = n1 if n1[1] > 0 else n2
                elif name == "TrunkDeck_Slant":
                    nvec = n1 if n1[0] < 0 else n2
                elif name == "Hopper":
                    nvec = n1 if n1[0] >= n2[0] else n2
                else:
                    nvec = self.STF_DIR.get(name, (-1.0, 0.0))

                pts = self._even_points_on_piece(s, e)
                drawn = 0
                for p in pts:
                    # t-space exclusion: InnerDeck_Slant∩Upper_Deck 교점 주변 1500mm
                    if name in ("InnerDeck_Slant", "Upper_Deck") and _in_id_ud_excl(name, p):
                        continue
                    self._draw_stiffener_shape(p, nvec, (uy, uz), stf_type, web_h, flange_half, _longi_layer)
                    drawn += 1
                self.stf_stats[name] += drawn

        # Bilge arc stiffeners
        if hasattr(self, "bilge_center") and hasattr(self, "bilge_R") and \
                hasattr(self, "bilge_ang0") and hasattr(self, "bilge_ang1"):
            cy, cz = self.bilge_center
            R = self.bilge_R
            ang0 = self.bilge_ang0
            ang1 = self.bilge_ang1

            arc_pts = self._even_points_on_arc((cy, cz), R, ang0, ang1)
            for p in arc_pts:
                nvec = (cy - p[0], cz - p[1])
                theta = atan2(p[1] - cz, p[0] - cy)
                along = (-sin(theta), cos(theta))
                self._draw_stiffener_shape(p, nvec, along, "FB", self.stf_len, 0, _longi_layer)
            self.stf_stats['Bilge_Arc'] = self.stf_stats.get('Bilge_Arc', 0) + len(arc_pts)

            # Bilge-end anchor stiffeners at toe points
            added_at_toes = 0
            for p in [getattr(self, "bilge_side_start", None),
                      getattr(self, "bilge_bottom_end", None)]:
                if p:
                    nvec = (cy - p[0], cz - p[1])
                    th = atan2(float(p[1]) - cz, float(p[0]) - cy)
                    along = (-sin(th), cos(th))
                    self._draw_stiffener_shape(p, nvec, along, "FB", self.stf_len, 0, _longi_layer)
                    added_at_toes += 1
            self.stf_stats['Bilge_Toes'] = self.stf_stats.get('Bilge_Toes', 0) + added_at_toes

        # Bottom_Shell–Bilge 접점에도 Bottom_Shell 방향(+z)으로 1개 추가
        segs_all = getattr(self, "segs_all", self.ship.seg_dict())
        if "Bottom_Shell" in segs_all and hasattr(self, "bilge_bottom_end"):
            p = getattr(self, "bilge_bottom_end", None)
            if p:
                nvec_bs = self.STF_DIR.get("Bottom_Shell", (0.0, 1.0))
                p1_bs, p2_bs = segs_all["Bottom_Shell"]
                uy_bs, uz_bs, _ = self._seg_dir_len(p1_bs, p2_bs)
                stf_t, flg_h, wh_bs = _STF_CFG.get("Bottom_Shell", ("T", 75, 400))
                self._draw_stiffener_shape(p, nvec_bs, (uy_bs, uz_bs), stf_t, wh_bs, flg_h, _longi_layer)
                self.stf_stats["Bottom_Shell"] = self.stf_stats.get("Bottom_Shell", 0) + 1

        # Upper_Deck ∩ TrunkDeck_Slant 교점 추가
        segs_all = getattr(self, "segs_all", self.ship.seg_dict())
        if "Upper_Deck" in segs_all and "TrunkDeck_Slant" in segs_all:
            p1, p2 = segs_all["Upper_Deck"]
            q1, q2 = segs_all["TrunkDeck_Slant"]
            ip = line_intersection(p1, p2, q1, q2)
            if ip is not None:
                uy, uz, Lp = self._seg_dir_len(p1, p2)
                tq = self._project_t(p1, p2, ip)
                uy2, uz2, Lq = self._seg_dir_len(q1, q2)
                t2 = self._project_t(q1, q2, ip)
                if (-1e-6 <= tq <= Lp + 1e-6) and (-1e-6 <= t2 <= Lq + 1e-6):
                    n1 = (-uz, uy)
                    n2 = (uz, -uy)
                    nvec = n1 if n1[1] < 0 else n2
                    stf_t, flg_h, wh_ud = _STF_CFG.get("Upper_Deck", ("T", 75, 350))
                    self._draw_stiffener_shape(ip, nvec, (uy, uz), stf_t, wh_ud, flg_h, _longi_layer)
                    self.stf_stats['Upper_Deck'] = self.stf_stats.get('Upper_Deck', 0) + 1

    # ---------- 스캔틀링 표 ----------
    def draw_scantling_table(self):
        layer = "Scantling"
        txt_h = 180.0; txt_h_hdr = 200.0
        col_w = [3600.0, 1900.0, 6800.0]; row_h = 700.0
        rows = _SCANTLING_TABLE
        n_rows = len(rows); total_w = sum(col_w); total_h = n_rows * row_h

        ch_cy, ch_cz = None, None
        for c in self.compartment_data:
            if "cargo" in c.get("clean_label", "").lower():
                ch_cy, ch_cz = c["centroid_mm"]; break
        if ch_cy is None:
            B = self.ship.B * 1000.0
            D = self.ship.D * 1000.0
            DB = getattr(self.ship, 'DB', 0) * 1000.0
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

        # Cargo Tank 라벨 — Compartment 레이어, 갑판(Trunk Deck 포함)과 테이블 사이 중앙
        z_deck_top = self.ship.z_deck(0) * 1000.0
        ch_label = self.msp.add_mtext("Cargo Tank",
            dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
        ch_label.dxf.insert = (ch_cy, (az + z_deck_top) / 2.0)
        ch_label.dxf.attachment_point = 5; ch_label.dxf.rotation = 0

    # ---------- PNG 저장 ----------
    def save_png(self, path_png, dpi=220, bgcolor="white", debug=True):
        if not _MATPLOT_OK:
            if debug: print("[PNG] matplotlib/ezdxf drawing 불가(_MATPLOT_OK=False)")
            return False

        fig = None
        try:
            face = mcolors.to_rgba(bgcolor)
            fig = plt.figure(figsize=(12, 12), dpi=dpi, facecolor=face)
            ax = fig.add_axes([0, 0, 1, 1], facecolor=face)

            ctx = RenderContext(self.doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(self.doc.modelspace())

            ax.set_aspect("equal")
            ax.set_axis_off()

            fig.savefig(path_png, dpi=dpi, facecolor=face, bbox_inches="tight", pad_inches=0)

            ok = os.path.isfile(path_png) and os.path.getsize(path_png) > 0
            if not ok and debug:
                print("[PNG] 파일 저장 실패(파일 없음/0바이트)")
            return ok
        except Exception as e:
            if debug: print(f"[PNG] 렌더 실패: {e}")
            return False
        finally:
            try:
                if fig is not None:
                    plt.close(fig)
                if plt is not None:
                    plt.close("all")
            except Exception:
                pass

    # ---------- 전체 Export ----------
    def export(self, save_as=None, dxf_version='R2018', png_out_dir=None, png_dpi=220):
        qc = {'label_overlaps': -1, 'ok': False}
        png_path = None

        try:
            self.doc = ezdxf.new(setup=True, dxfversion=dxf_version)
            self.msp = self.doc.modelspace()
            self.placed_label_polys = []
            self.label_records = []
            self.stf_stats = {}
            self.compartment_data = []
            self.intersections = []

            ensure_layers(self.doc)
            self.segs_all = self.ship.seg_dict()

            # 드로잉
            self.draw_centerline()
            self.draw_title_and_specs(title="ORDINARY SECTION (STBD)")
            self.draw_members()
            self.draw_bilge_curve()
            self.draw_compartments()
            self.draw_stiffeners()
            self.draw_scantling_table()

            # QC: 라벨 중첩 확인
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

            # PNG 저장
            if png_out_dir:
                os.makedirs(png_out_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(save_as or "lngc_section"))[0]
                candidate = os.path.join(png_out_dir, base + ".png")
                if self.save_png(candidate, dpi=png_dpi, bgcolor="white", debug=True):
                    png_path = candidate
                else:
                    print(f"[PNG] 생성 실패: {candidate}")

            # ===== 도면 요약 정보 만들기 =====
            member_props = {}
            for name, m in self.ship.members.items():
                y1, z1 = float(m[1][0]), float(m[2][0])
                y2, z2 = float(m[1][1]), float(m[2][1])
                Lmm = hypot(y2 - y1, z2 - z1)
                angle_deg = degrees(atan2(z2 - z1, y2 - y1)) if Lmm > EPS else 0.0
                bbox = {
                    "min_y_mm": round(min(y1, y2), 3), "max_y_mm": round(max(y1, y2), 3),
                    "min_z_mm": round(min(z1, z2), 3), "max_z_mm": round(max(z1, z2), 3)
                }
                member_props[name] = {
                    "full_name": expand_abbrev(name),
                    "endpoints_mm": [(round(y1, 3), round(z1, 3)), (round(y2, 3), round(z2, 3))],
                    "length_mm": round(Lmm, 3),
                    "length_m": round(Lmm/1000.0, 6),
                    "slope_deg": round(angle_deg, 6),
                    "bbox_mm": bbox
                }

            # --- NEW: Hold length 기반 면적/부피 계산 ---
            hold_len_m = float(self.hold_length_m) if self.hold_length_m is not None else None

            # (1) Member areas (plate-like area = projected length × hold length)
            member_areas = {}
            if hold_len_m is not None:
                for nm, prop in member_props.items():
                    length_m = prop["length_m"]
                    area_half = length_m * hold_len_m           # STBD half
                    area_full = area_half * 2.0                 # PORT+STBD
                    member_areas[nm] = {
                        "area_m2_half": round(area_half, 6),
                        "area_m2_full": round(area_full, 6),
                    }

            # (2) Compartment volumes (= area × hold length)
            comp_items = self.compartment_data[:]  # list of metas
            comp_vols = []
            group_sums = {
                "Void (STBD)": 0.0,
                "W.B.T (STBD)": 0.0,
                "Cargo tank (STBD)": 0.0,
                "Pipe duct (STBD)": 0.0,
            }

            def clean_name(meta):
                return clean_multiline_label(meta["raw_label"])

            if hold_len_m is not None:
                for c in comp_items:
                    A = float(c["area_m2"])
                    vol_half = A * hold_len_m
                    vol_full = vol_half * 2.0
                    cname = clean_name(c)
                    comp_vols.append({
                        "name": cname,
                        "volume_m3_half": round(vol_half, 6),
                        "volume_m3_full": round(vol_full, 6),
                    })
                    # 그룹 매핑
                    low = cname.lower()
                    if low.startswith("trunk void 1") or low.startswith("trunk void 2") or \
                       low.startswith("trunk void 3") or low.startswith("void"):
                        group_sums["Void (STBD)"] += vol_half
                    if low.startswith("ballast tank 1") or low.startswith("ballast tank 2") or \
                       low.startswith("ballast tank 3") or low.startswith("ballast tank 4"):
                        group_sums["W.B.T (STBD)"] += vol_half
                    if low.startswith("cargo tank"):
                        group_sums["Cargo tank (STBD)"] += vol_half
                    if low.startswith("pipe duct"):
                        group_sums["Pipe duct (STBD)"] += vol_half

            # full(좌우 합)로 변환한 그룹값도 제공
            group_sums_full = {k.replace("(STBD)", "(FULL)"): v*2.0 for k, v in group_sums.items()}

            # --- Cargo 용량(1홀, 전체) 및 토큰 계산 ---
            cargo_per_hold_full = 0.0
            if hold_len_m is not None:
                for v in comp_vols:
                    if v["name"].lower().startswith("cargo tank"):
                        cargo_per_hold_full += float(v["volume_m3_full"])

            total_cargo_full = None
            cargo_token_k = None
            if cargo_per_hold_full > 0.0:
                # FWD-most hold volume reduced by hold_vol_factor; remaining holds at full volume
                total_cargo_full = ((self.number_of_hold - 1) * cargo_per_hold_full) + (cargo_per_hold_full * self.hold_vol_factor)
                cargo_token_k = f"{int(round(total_cargo_full / 1000.0))}K"


            bilge_info = None
            if hasattr(self, "bilge_center"):
                cy, cz = self.bilge_center
                R = self.bilge_R
                a0, a1 = self.bilge_ang0, self.bilge_ang1
                arc_len = abs(R * (a1 - a0))
                bilge_info = {
                    'center_mm': (round(cy, 3), round(cz, 3)),
                    'radius_mm': round(R, 3),
                    'start_deg': round(a0 * 180.0 / pi, 6),
                    'end_deg': round(a1 * 180.0 / pi, 6),
                    'arc_length_mm': round(arc_len, 3),
                    'toe_points_mm': {
                        'bottom_end': getattr(self, 'bilge_bottom_end', None),
                        'side_start': getattr(self, 'bilge_side_start', None),
                    }
                }

            label_summary = {
                'count': len(self.label_records),
                'items': [
                    {'name': r['name'],
                     'full_name': expand_abbrev(clean_multiline_label(r['name'])),
                     'pos_mm': (round(r['pos'][0], 3), round(r['pos'][1], 3)),
                     'rotation_deg': r['rotation_deg'],
                     'layer': r['layer']}
                    for r in self.label_records
                ]
            }

            stiffeners_total = sum(self.stf_stats.values()) if self.stf_stats else 0
            stiffeners = {
                'per_member': dict(sorted(self.stf_stats.items())),
                'total': stiffeners_total,
                'rules': {
                    'min_spacing_mm': self.stf_min,
                    'max_spacing_mm': self.stf_max,
                    'target_spacing_mm': self.stf_target,
                    'tick_length_mm': self.stf_len,
                    'edge_clear_mm': self.edge_clear,
                }
            }

            layer_counts = {}
            for e in self.msp:
                ly = e.dxf.layer if hasattr(e.dxf, 'layer') else 'UNKNOWN'
                layer_counts[ly] = layer_counts.get(ly, 0) + 1

            ys, zs = [], []
            for m in self.ship.members.values():
                ys += [float(m[1][0]), float(m[1][1])]
                zs += [float(m[2][0]), float(m[2][1])]
            for p in [getattr(self, 'bilge_bottom_end', None),
                      getattr(self, 'bilge_side_start', None),
                      getattr(self, 'bilge_center', None)]:
                if p:
                    ys.append(float(p[0])); zs.append(float(p[1]))
            bbox = None
            if ys and zs:
                bbox = {
                    'min_y_mm': round(min(ys), 3), 'max_y_mm': round(max(ys), 3),
                    'min_z_mm': round(min(zs), 3), 'max_z_mm': round(max(zs), 3)
                }

            intersections = self.intersections
            compartments = {
                "items": self.compartment_data,
                "count": len(self.compartment_data),
                "total_area_m2": round(sum(c["area_m2"] for c in self.compartment_data), 6),
            }

            # ---- 단위/좌표/작도 컨벤션 ----
            doc_conventions = {
                "units": {
                    "lengths": {"drawing": "mm", "model": "m"},  # ← 명시적으로 key 정리
                    "area": "m^2",
                    "volume": "m^3",
                },
                "coordinate_system": {
                    "axes": "y(horizontal, +outboard), z(vertical, +up)",
                    "origin": "Centerline keel point at (0,0) in this section drawing",
                    "section": "Midship transverse section (2D, y–z plane)",
                },
                "drawing_conventions": {
                    "deck_camber": "Upper_Deck is cambered; z = D + camber - (camber/(B/2))*y",
                    "labels_multiline": "\\P represents line breaks in CAD text",
                    "members": "Member lines given as two points in mm in (y,z)",
                },
            }

            # ---- 축약 파라미터 스키마: 한 군데에서 정의(불일치 방지) ----
            PARAM_SPEC = {
                'L_m': {'desc': 'Ship length between perpendiculars (L), meters', 'unit': 'm', 'symbol': 'L'},
                'B_m': {'desc': 'Moulded breadth (B), meters', 'unit': 'm', 'symbol': 'B'},
                'D_m': {'desc': 'Moulded depth (D), meters', 'unit': 'm', 'symbol': 'D'},
                'HL_m': {'desc': 'Hold length per hold (HL), meters', 'unit': 'm', 'symbol': 'HL'},
                'camberUpper_m': {'desc': 'Upper deck camber height at CL (C), meters', 'unit': 'm', 'symbol': 'C'},
                'camberTrunk_m': {'desc': 'Trunk deck camber height at CL (CT), meters', 'unit': 'm', 'symbol': 'CT'},
                'doubleSide_m': {'desc': 'Double side width (DS), meters', 'unit': 'm', 'symbol': 'DS'},
                'doubleBottom_m': {'desc': 'Double bottom height (DB), meters', 'unit': 'm', 'symbol': 'DB'},
                'bilgeRadius_m': {'desc': 'Bilge radius (R), meters', 'unit': 'm', 'symbol': 'R'},
                'girderCL_ratio_B2': {'desc': 'Centerline girder location ratio to B/2 (G0), -', 'unit': 'ratio',
                                      'symbol': 'G0'},
                'girderB_ratio_B2': {'desc': 'Bottom girder location ratio to B/2 (G1), -', 'unit': 'ratio',
                                     'symbol': 'G1'},
                'girderOut_ratio_B2': {'desc': 'Outboard girder location ratio to B/2 (G2), -', 'unit': 'ratio',
                                       'symbol': 'G2'},
                'str1_ratio_D': {'desc': 'Stringer-1 vertical ratio to depth D (S1), -', 'unit': 'ratio',
                                 'symbol': 'S1'},
                'str2_ratio_D': {'desc': 'Stringer-2 vertical ratio to depth D (S2), -', 'unit': 'ratio',
                                 'symbol': 'S2'},
                'str3_ratio_D': {'desc': 'Stringer-3 vertical ratio to depth D (S3), -', 'unit': 'ratio',
                                 'symbol': 'S3'},
            }

            # conventions에 넣을 때 파생 뷰 생성(원하면 descriptions/units/symbols를 유지)
            doc_conventions['params'] = {
                'spec': PARAM_SPEC,
                'descriptions': {k: v['desc'] for k, v in PARAM_SPEC.items()},
                'units': {k: v['unit'] for k, v in PARAM_SPEC.items()},
                'symbols': {k: v['symbol'] for k, v in PARAM_SPEC.items()},
                # 보기 순서 고정이 필요하면 order 추가 가능
                # 'order': ['L_m','B_m','D_m','HL_m', ...]
            }

            # ---- 규칙 참고 ----
            rule_refs = {
                "camber_limits": "C >= 0; practical band approx 0.2%–2% of B; also C <= 0.05*B and C <= 0.10*D (sanity)",
                "inner_hull_clear": "DS < B/2; IHull y = B/2 - DS >= 0.8 m for trunk slant start",
                "inner_bottom_clear": "DB < D/2",
                "bilge_radius_clear": "R < (B/2 - DS) - 0.1 and R < (D - DB) - 0.1",
                "out_girder_clear": "G2 in (0,1); Out_Girder y < IHull y - 0.5 m and < (B/2 - R - 0.8)",
                "stringer_spacing": "z1>z2>z3; each vertical gap >= 0.6 m; S1 deck clearance >= 0.5 m",
                "hopper_geom": "S3*D > DB + 0.5 m",
                "stiffener_rules": f"{self.stf_min} ≤ spacing ≤ {self.stf_max} (target {self.stf_target}), edge clear {self.edge_clear} mm",
            }

            drawing_meta = {
                'layers': layer_counts,
                'bbox_mm': bbox,
                'labels': {
                    'count': len(self.label_records),
                    'items': [
                        {
                            'key': r['name'],  # ← 표준화: 라벨 키
                            'full': legend_full_name(r['name']),  # ← 정식명
                            'pos_mm': (round(r['pos'][0], 3), round(r['pos'][1], 3)),
                            'rotation_deg': r['rotation_deg'],
                            'layer': r['layer'],
                        } for r in self.label_records
                    ]
                },
                'stiffeners': stiffeners,
                'intersections': intersections,
                'qc': {
                    'label_overlaps': qc.get('label_overlaps', -1),
                    'labels_ok': qc.get('ok', False),
                },
            }

            if not EXPORT_INCLUDE_MEMBER_BBOX:
                member_props = {
                    name: {k: v for k, v in props.items() if k != 'bbox_mm'}
                    for name, props in member_props.items()
                }

            self.export_stats = {
                'hold': {
                    'length_m': hold_len_m,
                    'number_of_hold': self.number_of_hold,
                    'hold_vol_factor': self.hold_vol_factor,
                },
                'members': {'geometry': member_props, 'areas': member_areas},
                'compartments': {
                    'items': self.compartment_data,  # (각 item에서 abbrev_map은 제거됨)
                    'volumes': {
                        'items': comp_vols,
                        'groups_half': {k: round(v, 6) for k, v in group_sums.items()},
                        'groups_full': {k: round(v, 6) for k, v in group_sums_full.items()},
                        'cargo_per_hold_full_m3': round(cargo_per_hold_full,
                                                        6) if cargo_per_hold_full is not None else None,
                        'cargo_total_full_m3': round(total_cargo_full, 6) if total_cargo_full is not None else None,
                        'cargo_capacity_token': cargo_token_k,
                    },
                    'count': len(self.compartment_data),
                    'total_area_m2_half': round(sum(c["area_m2"] for c in self.compartment_data), 6),
                    'total_area_m2_full': round(2.0 * sum(c["area_m2"] for c in self.compartment_data), 6),
                },
                'drawing': drawing_meta,  # ← canvas 제거
                'domain': {
                    'legend': {k: legend_full_name(k) for k in self.ship.members.keys()},
                    'registry_version': LABEL_REGISTRY_VERSION,
                    'conventions': doc_conventions,
                    'rule_refs': rule_refs,
                    'stiffener_types': _STF_TYPE_LEGEND,
                    'scantling_table': [
                        {'member': r[0], 'plate_mm': r[1], 'stiffener': r[2]}
                        for r in _SCANTLING_TABLE[1:]
                    ],
                }
            }

        except Exception as e:
            print(f"[EXPORT] 오류: {e}")

        return qc, png_path


# ===============================
# 선급 Rule 체크 (Geometry-focused heuristic packs)
# ===============================
def _build_rulepacks():
    """
    선급별로 '단면 파라미터로 확인 가능한' 범위/여유 기반 규칙 세트.
    실 Rule book 대체 아님. 형상 sanity + 일반 관행 범위.
    값 단위: 입력 파라미터는 m/비율, 내부 계산에서 mm 병행 표기.
    """
    # 공통 상수
    CAMBER_MIN_RATIO = 0.002   # ≈ B/500
    CAMBER_MAX_RATIO = 0.02    # ≈ B/50 (실무 상한 권장)
    STRINGER_MIN_GAP = 0.6     # m (수직 간격)
    STR1_DECK_CLEAR = 0.5      # m (Str1와 deck 사이 최소 여유)
    IHULL_MIN_Y = 0.8          # m (트렁크 슬랜트 시작을 위한 최소 outboard 여유)
    OUTG_INBOARD_CLEAR = 0.5   # m (Out_Girder vs IHull)
    OUTG_BILGE_TOE_CLEAR = 0.8 # m (Out_Girder vs bilge toe)
    HOPPER_MIN_CLEAR = 0.5     # m (S3*D vs DB)

    # Bilge radius 실무 범위(비율)
    BILGE_R_B_MIN = 0.03
    BILGE_R_B_MAX = 0.08

    # 선급별 미세 조정(여기서는 같은 값 사용, 향후 필요시 분화 가능)
    base = dict(
        camber_min_ratio=CAMBER_MIN_RATIO,
        camber_max_ratio=CAMBER_MAX_RATIO,
        stringer_min_gap=STRINGER_MIN_GAP,
        str1_deck_clear=STR1_DECK_CLEAR,
        ihull_min_y=IHULL_MIN_Y,
        outg_inboard_clear=OUTG_INBOARD_CLEAR,
        outg_bilge_toe_clear=OUTG_BILGE_TOE_CLEAR,
        hopper_min_clear=HOPPER_MIN_CLEAR,
        bilge_r_b_min=BILGE_R_B_MIN,
        bilge_r_b_max=BILGE_R_B_MAX,
    )

    return {
        "DNV": base.copy(),
        "ABS": base.copy(),
        "LR":  base.copy(),
        "BV":  base.copy(),
    }


def _eval_rule(rule_id, desc, value, limits, expr, note=""):
    try:
        passed = bool(expr(value, limits))
    except Exception:
        passed = False
    return {
        "id": rule_id,
        "desc": desc,
        "value": value,
        "limits": limits,
        "passed": passed,
        "note": note
    }


def _evaluate_society_pack(pack, params):
    B = params['B']; D = params['D']
    C = params['C']; CT = params.get('CT', 0.0)    # <-- trunk camber
    DS = params['DS']; DB = params['DB']; R = params['R']
    G2 = params['G2']; S1 = params['S1']; S2 = params['S2']; S3 = params['S3']

    # 파생 값
    y_ihull = B/2 - DS
    y_outg  = G2 * (B/2)
    y_bilge_toe = B/2 - R
    z1 = S1*D; z2 = S2*D; z3 = S3*D

    rules = []

    # 기본 sanity (단위 혼용 주의: 비율 vs m)
    rules.append(_eval_rule(
        "DIM-001", "All key dims positive (B,D,DS,DB,R)",
        {"B": B, "D": D, "DS": DS, "DB": DB, "R": R},
        None, lambda v, l: (v["B"]>0 and v["D"]>0 and v["DS"]>0 and v["DB"]>0 and v["R"]>0)
    ))
    rules.append(_eval_rule(
        "IHULL-001", "Inner hull outboard y >= minimum",
        {"y_ihull": y_ihull, "min": pack["ihull_min_y"]},
        {"min": pack["ihull_min_y"]},
        lambda v, lim: v["y_ihull"] >= lim["min"],
        note="Ensures space for trunk slant start"
    ))
    rules.append(_eval_rule(
        "BOTTOM-001", "Inner bottom below mid-depth (DB < D/2)",
        {"DB": DB, "D": D},
        None, lambda v, l: v["DB"] < v["D"]/2.0
    ))

    # Trunk camber (CT)
    rules.append(_eval_rule(
        "CAM-TR-001", "Trunk camber positive",
        {"CT": CT}, None, lambda v, l: v["CT"] > 0.0
    ))
    rules.append(_eval_rule(
        "CAM-TR-002", "Trunk camber practical band vs B",
        {"CT": CT, "B": B, "ratio": CT / (B if B > 0 else 1)},
        {"min_ratio": pack["camber_min_ratio"], "max_ratio": pack["camber_max_ratio"]},
        lambda v, lim: (v["ratio"] >= lim["min_ratio"]) and (v["ratio"] <= lim["max_ratio"])
    ))
    rules.append(_eval_rule(
        "CAM-TR-003", "Trunk camber upper sanity caps",
        {"CT": CT, "B": B, "D": D},
        {"maxB": 0.05 * B, "maxD": 0.10 * D},
        lambda v, lim: (v["CT"] <= lim["maxB"]) and (v["CT"] <= lim["maxD"])
    ))

    # Upper Deck Camber (C)
    rules.append(_eval_rule(
        "CAM-UP-001", "Upper deck camber positive",
        {"C": C}, None, lambda v, l: v["C"] >= 0.0
    ))
    rules.append(_eval_rule(
        "CAM-UP-002", "Upper deck camber practical band vs B",
        {"C": C, "B": B, "ratio": C/(B if B>0 else 1)},
        {"min_ratio": pack["camber_min_ratio"], "max_ratio": pack["camber_max_ratio"]},
        lambda v, lim: (v["ratio"] >= lim["min_ratio"]) and (v["ratio"] <= lim["max_ratio"]),
        note="Typical camber ~ B/100 ~ B/50 range"
    ))
    rules.append(_eval_rule(
        "CAM-UP-003", "Upper deck camber upper sanity caps",
        {"C": C, "B": B, "D": D},
        {"maxB": 0.05*B, "maxD": 0.10*D},
        lambda v, lim: (v["C"] <= lim["maxB"]) and (v["C"] <= lim["maxD"])
    ))


    # Bilge radius
    rules.append(_eval_rule(
        "BILGE-001", "Bilge R ratio vs B within practical band",
        {"R": R, "B": B, "ratio": R/(B if B>0 else 1)},
        {"min": pack["bilge_r_b_min"], "max": pack["bilge_r_b_max"]},
        lambda v, lim: (v["ratio"] >= lim["min"]) and (v["ratio"] <= lim["max"])
    ))
    rules.append(_eval_rule(
        "BILGE-002", "Bilge R clearance vs inner hull & depth",
        {"R": R, "B": B, "DS": DS, "D": D, "DB": DB},
        None,
        lambda v, l: (v["R"] < (v["B"]/2 - v["DS"]) - 0.1) and (v["R"] < (v["D"] - v["DB"]) - 0.1),
        note="Keep bilge inside inner hull and above inner bottom"
    ))

    # Out girder
    rules.append(_eval_rule(
        "OGIR-001", "Out Girder ratio range (0<G2<1)",
        {"G2": G2}, None, lambda v, l: 0.0 < v["G2"] < 1.0
    ))
    rules.append(_eval_rule(
        "OGIR-002", "Out Girder inboard clearance from inner hull",
        {"y_outg": y_outg, "y_ihull": y_ihull, "clear": y_ihull - y_outg},
        {"min_clear": pack["outg_inboard_clear"]},
        lambda v, lim: v["clear"] >= lim["min_clear"]
    ))
    rules.append(_eval_rule(
        "OGIR-003", "Out Girder not too close to bilge toe",
        {"y_outg": y_outg, "y_bilge_toe": y_bilge_toe, "clear": y_bilge_toe - y_outg},
        {"min_clear": pack["outg_bilge_toe_clear"]},
        lambda v, lim: v["clear"] >= lim["min_clear"]
    ))

    # Stringers
    rules.append(_eval_rule(
        "STR-001", "Stringer order z1>z2>z3",
        {"z1": z1, "z2": z2, "z3": z3}, None, lambda v, l: (v["z1"]>v["z2"]>v["z3"])
    ))
    rules.append(_eval_rule(
        "STR-002", "Vertical gaps between stringers >= min",
        {"gap12": z1 - z2, "gap23": z2 - z3},
        {"min_gap": pack["stringer_min_gap"]},
        lambda v, lim: (v["gap12"] >= lim["min_gap"]) and (v["gap23"] >= lim["min_gap"])
    ))
    rules.append(_eval_rule(
        "STR-003", "Str1 deck clearance >= minimum",
        {"deck_clear": (D - z1)},
        {"min": pack["str1_deck_clear"]},
        lambda v, lim: v["deck_clear"] >= lim["min"]
    ))

    # Hopper
    rules.append(_eval_rule(
        "HOP-001", "Hopper top (z3*D) above inner bottom by margin",
        {"z3D": z3, "DB": DB, "margin": z3 - DB},
        {"min_margin": pack["hopper_min_clear"]},
        lambda v, lim: v["margin"] >= lim["min_margin"]
    ))

    passed_ids = [r["id"] for r in rules if r["passed"]]
    failed = [r for r in rules if not r["passed"]]

    summary = {
        "overall_pass": len(failed) == 0,
        "passed_count": len(passed_ids),
        "failed_count": len(failed),
        "failed_rule_ids": [r["id"] for r in failed],
        "rules": rules
    }
    return summary


def domain_rules_ok_lngc(params):
    """
    기존 공통 체크 + 선급별(rule packs) 형상 점검.
    반환:
      ok_all: bool (모든 공통 + 모든 선급 pack 통과 여부)
      issues: [str] (간단 코드 요약)
      detail: {
        "common": {...},
        "society_checks": {"DNV": {...}, "ABS": {...}, "LR": {...}, "BV": {...}}
      }
    """
    L = params['L']; B = params['B']; D = params['D']
    C = params['C']; DS = params['DS']; DB = params['DB']; R = params['R']
    G0 = params['G0']; G1 = params['G1']; G2 = params['G2']
    S1 = params['S1']; S2 = params['S2']; S3 = params['S3']

    # ===== 기존 공통 체크 =====
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

    if not (0.0 <= G0 < G1 < G2 < 1.0):
        issues.append("Girder_ratio_order_G0<G1<G2")
    y_ihull = B/2 - DS; y_2gir = G2 * (B/2)
    if y_2gir >= y_ihull - 0.5:
        issues.append("G2_too_close_to_inner_hull")

    if (D - S1 * D) < 0.5:
        issues.append("Str1_too_close_to_deck")
    if y_ihull < 0.8:
        issues.append("IHull_y_less_than_0p8m_for_side_trunk_knuckle")

    if (S3 * D) <= DB + 0.5: issues.append("Hopper_low_height")
    if (G2 * (B/2)) >= (B/2 - DS) - 0.5: issues.append("Hopper_low_width")

    y_bilge_toe = B/2 - R
    if (G2 * (B/2)) > (y_bilge_toe - 0.8):
        issues.append("Out_Girder_too_outboard_vs_bilge_toe")

    # ===== 선급별 Rule packs 평가 =====
    rulepacks = _build_rulepacks()
    society_checks = {}
    for name, pack in rulepacks.items():
        sc = _evaluate_society_pack(pack, {
            'B': B, 'D': D, 'C': C, 'CT': params.get('CT', 0.0),
            'DS': DS, 'DB': DB, 'R': R, 'G2': G2, 'S1': S1, 'S2': S2, 'S3': S3,
            # 필요시 pack 평가에 G1/G0을 더 녹이고 싶다면 pack/평가 함수에 항목 추가
        })
        society_checks[name] = sc

    # 공통 + 선급 요약
    common_ok = len(issues) == 0
    societies_ok = all(v["overall_pass"] for v in society_checks.values())
    ok_all = common_ok and societies_ok

    detail = {
        "common": {
            "ok": common_ok,
            "issues": issues
        },
        "society_checks": society_checks
    }
    return ok_all, issues, detail


# ===============================
# 메타데이터 + QC 기록
# ===============================
def _round_floats(obj, ndigits=3):
    """dict/list 내부의 float만 ndigits 자리로 반올림해서 '최대' 소수 3자리로 제한.
    - 예: 0.12 -> 0.12 유지, 0.123456 -> 0.123 로 줄임, 2.0 -> 2.0
    """
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def write_metadata_json(json_path, meta):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    # 전체 구조 안의 float을 '최대' 소수 3자리로 제한
    meta_3dp = _round_floats(meta, ndigits=3)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_3dp, f, ensure_ascii=False, indent=2)



def append_index_csv(csv_path, header, row_dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        # --- 헤더에 있는 것만 쓰고, 빠진 건 빈칸으로 ---
        safe_row = {h: row_dict.get(h, "") for h in header}
        writer.writerow(safe_row)



# ================================
# KR Standard Info — KR Rules 2025 (Pt15 + Pt7 Ch5: LNG Carriers)
# ================================
KR_STANDARD_INFO = {
    "title": "Korean Register Rules for the Classification of Ships",
    "edition": "2025",
    "short_name": "KR Rules 2025",
    "effective_from": "2025-01-01",
    "source_file": "KR-Rules-2025.pdf",
}

KR_RULE_REGISTRY_LNGC = {
    "lngc_scope":           {"rule_ref": "Pt15.Ch1.Sec1[1.1]",  "title": "KR scope — LNG carrier applicability",        "level": "scope"},
    "membrane_tank":        {"rule_ref": "Pt15.Ch2.Sec1[1.1]",  "title": "Membrane containment system type",             "level": "arrangement"},
    "double_bottom_height": {"rule_ref": "Pt15.Ch2.Sec3[2.1]",  "title": "Minimum double bottom height for LNGC",        "level": "arrangement"},
    "double_side_width":    {"rule_ref": "Pt15.Ch2.Sec3[3.1]",  "title": "Minimum double side width for LNGC",           "level": "arrangement"},
    "cofferdam_req":        {"rule_ref": "Pt15.Ch2.Sec4[1.1]",  "title": "Cofferdam between cargo holds",                "level": "arrangement"},
    "trunk_clearance":      {"rule_ref": "Pt15.Ch2.Sec3[5.1]",  "title": "Trunk deck clearance above inner hull",        "level": "arrangement"},
    "inner_hull_slope":     {"rule_ref": "Pt15.Ch2.Sec3[4.1]",  "title": "Inner hull slope angle requirement",           "level": "arrangement"},
    "gas_freeing":          {"rule_ref": "Pt7.Ch5.Sec4[3.1]",   "title": "Gas freeing arrangement (Pt7 Ch5)",            "level": "arrangement"},
    "longitudinal_framing": {"rule_ref": "Pt15.Ch3.Sec3[2.1]",  "title": "Longitudinal framing requirement",             "level": "arrangement"},
    "weld_joint_detail":    {"rule_ref": "Pt15.Ch5.Sec3",        "title": "Weld joint detail requirements",              "level": "detail_design"},
}

def _lngc_rule_meta(check_id):
    r = KR_RULE_REGISTRY_LNGC.get(check_id, {})
    return r.get("rule_ref", ""), r.get("title", check_id), r.get("level", "")

def make_kr_check_lngc(check_id, status, *, inputs=None, actual=None, required=None, unit=None, notes=None):
    rule_ref, title, level = _lngc_rule_meta(check_id)
    out = {"check_id": check_id, "rule_ref": rule_ref, "title": title, "level": level, "status": status}
    if inputs is not None:   out["inputs"] = inputs
    if actual is not None:   out["actual"] = actual
    if required is not None: out["required"] = required
    if unit is not None:     out["unit"] = unit
    if notes is not None:    out["notes"] = notes
    return out

def evaluate_kr_rules_lngc(generator_inputs, ship):
    """KR Rules 2025 Pt15 + Pt7 Ch5 evaluation for LNG carrier. 4 states: pass/fail/undetermined/not_modeled"""
    checks = []
    assumptions = ["Framing system assumed longitudinal.", "KR Pt15 + Pt7 Ch5 2025 applied.", "Membrane type containment assumed."]

    L_m  = float(generator_inputs.get("L_m", 0))
    B_m  = float(generator_inputs.get("B_m", ship.B))
    DB_m = float(generator_inputs.get("doubleBottom_m", ship.d_db))
    DS_m = float(generator_inputs.get("doubleSide_m", ship.d_ds))

    checks.append(make_kr_check_lngc("lngc_scope", "pass" if L_m >= 150.0 else "fail",
        inputs={"L_m": round(L_m, 3)}, actual=round(L_m, 3), required={"min_m": 150.0}, unit="m",
        notes="KR Pt15 applies to LNG carriers >= 150 m."))

    checks.append(make_kr_check_lngc("membrane_tank", "pass",
        notes="Membrane containment system (Mark III / NO96 equivalent) assumed."))

    # IMO IGC Code Ch.19 Products List: LNG (methane) is Type 2G (not 1G).
    # IGC 2.4.1 / KR Pt15 Ch2 Sec3[2.1] for Type 2G:
    #   double bottom: d >= max(0.76, B/15)
    # (The 2.0 m floor and the min(B/5, 11.5) formula are Type 1G — do NOT apply to LNG.)
    required_db = max(0.76, B_m / 15.0)
    checks.append(make_kr_check_lngc("double_bottom_height", "pass" if DB_m >= required_db - 1e-9 else "fail",
        inputs={"B_m": round(B_m, 3), "DB_m": round(DB_m, 3)},
        actual=round(DB_m, 3), required={"min_m": round(required_db, 4)}, unit="m",
        notes="IGC 2.4.1 Type 2G (LNG is Type 2G per IGC Ch.19): d >= max(0.76 m, B/15)."))

    # IGC 2.4.1 for Type 2G: double side inboard distance from side shell
    #   ds >= max(0.76, B/15)   (measured at right angles to the side shell)
    required_ds = max(0.76, B_m / 15.0)
    checks.append(make_kr_check_lngc("double_side_width", "pass" if DS_m >= required_ds - 1e-9 else "fail",
        inputs={"B_m": round(B_m, 3), "DS_m": round(DS_m, 3)},
        actual=round(DS_m, 3), required={"min_m": round(required_ds, 4)}, unit="m",
        notes="IGC 2.4.1 Type 2G: ds >= max(0.76 m, B/15) from side shell."))

    # Cofferdam: check if layout has CD segments (pass if cofferdam_len > 0)
    n_cofferdam = int(generator_inputs.get("number_of_cofferdam", 0))
    checks.append(make_kr_check_lngc("cofferdam_req",
        "pass" if n_cofferdam > 0 else "fail",
        inputs={"number_of_cofferdam": n_cofferdam},
        notes="Cofferdam required between cargo holds for LNG membrane carriers."))

    # Trunk clearance: z_trunk - z_flat >= 0.5 m
    z_trunk = float(getattr(ship, 'z_trunk', None) or generator_inputs.get('z_trunk_m', 0))
    z_flat  = float(getattr(ship, 'z_flat', None) or generator_inputs.get('z_flat_m', 0))
    trunk_clr = z_trunk - z_flat if z_trunk and z_flat else None
    if trunk_clr is not None:
        trunk_st = "pass" if trunk_clr >= 0.5 - 1e-9 else "fail"
    else:
        trunk_st = "undetermined"
    checks.append(make_kr_check_lngc("trunk_clearance", trunk_st,
        inputs={"z_trunk_m": round(z_trunk, 3) if z_trunk else None,
                "z_flat_m": round(z_flat, 3) if z_flat else None},
        actual=round(trunk_clr, 3) if trunk_clr is not None else None,
        required={"min_m": 0.5}, unit="m",
        notes="Trunk deck clearance above inner hull deck." if trunk_clr is not None else "z_trunk/z_flat not available."))

    # Inner hull slope angle (Type B membrane LNGC trunk-side knuckle support).
    # Source: ship geometry (set in LNGC.__init__ as self.inner_slope_deg).
    inner_slope = getattr(ship, 'inner_slope_deg', None)
    if inner_slope is None:
        inner_slope = generator_inputs.get('inner_slope_deg', None)
    if inner_slope is not None:
        inner_slope = float(inner_slope)
        checks.append(make_kr_check_lngc("inner_hull_slope", "pass" if inner_slope >= 5.0 else "fail",
            inputs={"inner_slope_deg": round(inner_slope, 2)}, actual=round(inner_slope, 2),
            required={"min_deg": 5.0}, unit="deg",
            notes="Inner hull slope >= 5 deg supports trunk-side knuckle load path."))
    else:
        checks.append(make_kr_check_lngc("inner_hull_slope", "undetermined",
            notes="inner_slope_deg not available."))

    # IGC 9.5: gas-freeing requires trunk + vent routing. Parametric model captures
    # trunk presence but not duct details, so we mark "pass" if a trunk deck is
    # modeled (necessary condition) and "undetermined" otherwise.
    _seg_for_trunk = ship.seg_dict()
    _has_trunk = ("Trunk_Deck" in _seg_for_trunk) or ("TrunkDeck_Slant" in _seg_for_trunk)
    if _has_trunk:
        checks.append(make_kr_check_lngc("gas_freeing", "pass",
            inputs={"trunk_deck_modeled": True},
            notes="Trunk deck present (necessary condition); duct routing not modeled."))
    else:
        checks.append(make_kr_check_lngc("gas_freeing", "undetermined",
            inputs={"trunk_deck_modeled": False},
            notes="Trunk deck not modeled; gas-freeing arrangement cannot be assessed."))
    checks.append(make_kr_check_lngc("longitudinal_framing", "pass",
        inputs={"framing_system": "longitudinal"}, notes="Longitudinal framing assumed."))
    checks.append(make_kr_check_lngc("weld_joint_detail", "undetermined",
        notes="Weld joint geometry not included in parametric model."))

    def _isect(seg_a, seg_b):
        if seg_a is None or seg_b is None: return None
        (ay1,az1),(ay2,az2)=seg_a; (by1,bz1),(by2,bz2)=seg_b
        day,daz=ay2-ay1, az2-az1; dby,dbz=by2-by1, bz2-bz1
        denom=day*dbz-daz*dby
        if abs(denom)<1e-9: return None
        t=((by1-ay1)*dbz-(bz1-az1)*dby)/denom
        return (round(ay1+t*day,3), round(az1+t*daz,3))

    segs = ship.seg_dict()
    hotspots = []
    ih_pt = _isect(segs.get("IHull"), segs.get("IBTM"))
    hotspots.append({"hotspot_id": "inner_hull_knuckle", "rule_ref": "Pt15.Ch2.Sec3[3.1]",
        "title": "Inner hull-inner bottom connection knuckle",
        "availability": "modeled" if ih_pt else "not_modeled", "point_mm": ih_pt,
        "related_members": ["IHull", "IBTM"], "kr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "weld_detail"]})
    trunk_pt = _isect(segs.get("Trunk_Deck"), segs.get("TrunkDeck_Slant"))
    hotspots.append({"hotspot_id": "trunk_slant_connection", "rule_ref": "Pt15.Ch2.Sec3[5.1]",
        "title": "Trunk deck slant connection",
        "availability": "modeled" if trunk_pt else "not_modeled", "point_mm": trunk_pt,
        "related_members": ["Trunk_Deck", "TrunkDeck_Slant"], "kr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm"]})

    counts = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
    for c in checks:
        s = c.get("status")
        if s in counts: counts[s] += 1
    overall = "fail" if counts["fail"] > 0 else (
        "partial" if counts["undetermined"] + counts["not_modeled"] > 0 else "pass")

    return {
        "standard": KR_STANDARD_INFO, "ship_type": "lng_carrier",
        "assumptions": assumptions, "auto_checks": checks, "detail_hotspots": hotspots,
        "needs_additional_input": [{"check_id": c["check_id"], "rule_ref": c.get("rule_ref"), "notes": c.get("notes")}
            for c in checks if c.get("status") in ("undetermined", "not_modeled")],
        "summary": {"check_counts": counts,
            "hotspot_counts": {"modeled": sum(1 for h in hotspots if h.get("availability") == "modeled"),
                "not_modeled": sum(1 for h in hotspots if h.get("availability") == "not_modeled")},
            "overall_arrangement_status": overall},
    }


multi_society_checks = {}  # placeholder for LR/DNV/ABS/BV extension



def build_longitudinal_layout(L_m: float,
                              HL_m: float,
                              number_of_hold: int,
                              fwd_len_m: float,
                              er_len_m: float,
                              aft_len_m: float,
                              hold_len_factor: float,
                              cofferdam_len_m: float = 0.0):
    """
    ship 길이 L_m와 길이 모델을 이용해
    AFT / ER / [CD / HOLD N].. / CD / FWD 구간의 x 위치(mm)를 계산.
    cofferdam_len_m > 0 이면 각 Hold 앞뒤에 Cofferdam(CD) 구간을 삽입.
    n_cofferdam = n_hold + 1 (ER-Hold N 사이, 각 Hold 사이, Hold 1-FWD 사이)

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
          {'name': 'CD', ...},   # cofferdam (if cofferdam_len_m > 0)
          {'name': 'HOLD N', ...},
          ...
          {'name': 'HOLD 1', ...},
          {'name': 'CD', ...},
          {'name': 'FWD', ...},
        ],
        'bulkheads_mm': [x0, x1, x2, ...]
      }
    """
    if number_of_hold <= 0:
        raise ValueError("number_of_hold must be >= 1")

    hold_seg_m = hold_len_factor * HL_m
    n_cd = (number_of_hold + 1) if cofferdam_len_m > 0 else 0
    cd_total_m = n_cd * cofferdam_len_m

    model_L = aft_len_m + er_len_m + cd_total_m + number_of_hold * hold_seg_m + fwd_len_m

    # L과 길이 모델 오차를 FWD에 흡수 (부동소수점 수mm 수준 정리)
    fwd_adj_m = fwd_len_m + (L_m - model_L)
    if fwd_adj_m < 0:
        fwd_adj_m = max(0.0, fwd_len_m)

    scale = 1000.0  # m -> mm
    segs = []
    bulkheads = []

    x = 0.0

    # AFT
    x0 = x; x1 = x0 + aft_len_m * scale
    segs.append({'name': 'AFT', 'x0_mm': x0, 'x1_mm': x1})
    bulkheads.append(x0); x = x1

    # ER
    x0 = x; x1 = x0 + er_len_m * scale
    segs.append({'name': 'ER', 'x0_mm': x0, 'x1_mm': x1})
    bulkheads.append(x0); x = x1

    # HOLD N .. HOLD 1 각각 앞에 CD 삽입
    for k in range(number_of_hold):
        idx_from_aft = number_of_hold - k  # aftmost: N, fwdmost: 1
        # Cofferdam before each hold
        if cofferdam_len_m > 0:
            x0 = x; x1 = x0 + cofferdam_len_m * scale
            segs.append({'name': 'CD', 'x0_mm': x0, 'x1_mm': x1})
            bulkheads.append(x0); x = x1
        # Hold
        x0 = x; x1 = x0 + hold_seg_m * scale
        segs.append({'name': f'HOLD {idx_from_aft}', 'x0_mm': x0, 'x1_mm': x1})
        bulkheads.append(x0); x = x1

    # Cofferdam before FWD
    if cofferdam_len_m > 0:
        x0 = x; x1 = x0 + cofferdam_len_m * scale
        segs.append({'name': 'CD', 'x0_mm': x0, 'x1_mm': x1})
        bulkheads.append(x0); x = x1

    # FWD
    x0 = x; x1 = x0 + fwd_adj_m * scale
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


# ===============================
# 3D Model Generator
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
        "Bottom_Shell":    "3D_OUTER_HULL",
        "Side_Shell":      "3D_OUTER_HULL",
        "Upper_Deck":      "3D_OUTER_HULL",
        "IBTM":            "3D_DB",
        "IHull":           "3D_DS",
        "InnerDeck_Slant": "3D_DS",
        "Trunk_Deck":      "3D_CARGO_HOLD",
        "TrunkDeck_Slant": "3D_CARGO_HOLD",
        "Hopper":          "3D_DS",
        "CL_Girder":       "3D_DB",
        "B_Girder":        "3D_DB",
        "Out_Girder":      "3D_DB",
        "InnerDeck_Flat":  "3D_DS",
        "U_Girder1":       "3D_DB",
        "U_Girder2":       "3D_DB",
        "Str1":            "3D_DS",
        "Str2":            "3D_DS",
        "Str3":            "3D_DS",
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
        "3D_CARGO_HOLD":  ('#2266cc', 0.65, 0.5),   # cargo — royalblue (LNGC)
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

# ===============================
# 데이터셋 생성 루프 (KR Pt15+Pt7Ch5 / Elev / 3D 포함)
# ===============================
def generate_lngc_dataset(
    save_dir,
    method='lhs',
    fwd_hold_ratio=0.16,
    er_hold_ratio=0.28,
    aft_hold_ratio=0.14,
    cofferdam_len=2.5,
    hold_len_factor=1.0,
    hold_vol_factor=0.7,
    number_of_hold_range=(3, 5, 1),
    L_fixed=None,
    compart_out_dir=None,
    compart_png_out_dir=None,
    compart3d_out_dir=None,
    compart3d_png_out_dir=None,
    json_out_dir=None,
    B_range=(40, 50, 1),
    D_range=(20, 30, 1),
    hold_length_range=(40.0, 50.0, 0.1),
    camber_range_upper=(0.3, 1.5, 0.1),
    camber_range_trunk=(0.1, 1.0, 0.1),
    ds_range=(1.5, 3.5, 0.1),
    db_range=(2.2, 4.2, 0.1),
    bilge_range=(3.0, 5.0, 0.5),
    lbhd_ratio=(0.0, 0.0, 0.1),
    g0_ratio=(0.0, 0.0, 0.05),
    g1_ratio=(0.2, 0.3, 0.05),
    g2_ratio=(0.5, 0.7, 0.05),
    s1_ratio=(0.8, 0.9, 0.05),
    s2_ratio=(0.55, 0.7, 0.05),
    s3_ratio=(0.3, 0.4, 0.05),
    text_height=250,
    offset=300,
    MAX_FILES=100,
    PROGRESS_EVERY=20,
    SEED=None,
    png_out_dir=None,
    png_dpi=220,
):
    os.makedirs(save_dir, exist_ok=True)
    rng = random.Random(SEED)

    for d in [compart_out_dir, compart_png_out_dir, compart3d_out_dir, compart3d_png_out_dir, json_out_dir]:
        if d is not None:
            os.makedirs(d, exist_ok=True)

    def pick_n_hold():
        lo, hi, step = number_of_hold_range
        return rng.randrange(int(lo), int(hi) + 1, int(step))

    def _est_L(HL, n_hold):
        hold_total = hold_len_factor * HL * n_hold
        n_cd = n_hold + 1
        hold_part = n_cd * cofferdam_len + hold_total
        return (fwd_hold_ratio + er_hold_ratio + aft_hold_ratio) * hold_total + hold_part

    B_vals = list(range(B_range[0], B_range[1] + 1, B_range[2]))
    D_vals = list(range(D_range[0], D_range[1] + 1, D_range[2]))
    hold_length_vals = list(float_range(*hold_length_range))
    camber_upper_vals = list(float_range(*camber_range_upper))
    camber_trunk_vals = list(float_range(*camber_range_trunk))
    ds_vals = list(float_range(*ds_range))
    db_vals = list(float_range(*db_range))
    bilge_vals = list(float_range(*bilge_range))
    lb_vals = list(float_range(*lbhd_ratio))
    g0_vals = list(float_range(*g0_ratio))
    g1_vals = list(float_range(*g1_ratio))
    g2_vals = list(float_range(*g2_ratio))
    s1_vals = list(float_range(*s1_ratio))
    s2_vals = list(float_range(*s2_ratio))
    s3_vals = list(float_range(*s3_ratio))

    axes = [B_vals, D_vals, hold_length_vals, camber_upper_vals, camber_trunk_vals,
            ds_vals, db_vals, bilge_vals, lb_vals, g0_vals, g1_vals, g2_vals, s1_vals, s2_vals, s3_vals]
    dims = [len(a) for a in axes]
    total = 1
    for n in dims: total *= n

    candidate_params = []

    if method == 'grid':
        all_idx = list(range(total))
        if total > MAX_FILES:
            all_idx = rng.sample(all_idx, MAX_FILES)
        for idx in all_idx:
            ii = unravel_index(idx, dims)
            B, D, HL, C, CT, DS, DB, R, LB, G0, G1, G2, S1, S2, S3 = [axes[d][ii[d]] for d in range(len(dims))]
            n_hold = pick_n_hold()
            L_use = _est_L(HL, n_hold) if L_fixed is None else L_fixed
            candidate_params.append({'B': B,'D': D,'HL': HL,'C': C,'CT': CT,'DS': DS,'DB': DB,'R': R,
                'LB': LB,'G0': G0,'G1': G1,'G2': G2,'S1': S1,'S2': S2,'S3': S3,'L': L_use,'N_HOLD': n_hold})

    elif method == 'random':
        k = min(MAX_FILES, total)
        sampled = rng.sample(range(total), k)
        for idx in sampled:
            ii = unravel_index(idx, dims)
            B, D, HL, C, CT, DS, DB, R, LB, G0, G1, G2, S1, S2, S3 = [axes[d][ii[d]] for d in range(len(dims))]
            n_hold = pick_n_hold()
            L_use = _est_L(HL, n_hold) if L_fixed is None else L_fixed
            candidate_params.append({'B': B,'D': D,'HL': HL,'C': C,'CT': CT,'DS': DS,'DB': DB,'R': R,
                'LB': LB,'G0': G0,'G1': G1,'G2': G2,'S1': S1,'S2': S2,'S3': S3,'L': L_use,'N_HOLD': n_hold})

    elif method == 'lhs':
        N = MAX_FILES
        specs = [
            {'name': 'B',  'min': B_range[0],            'max': B_range[1],            'type': 'int',   'step': B_range[2]},
            {'name': 'D',  'min': D_range[0],            'max': D_range[1],            'type': 'int',   'step': D_range[2]},
            {'name': 'HL', 'min': hold_length_range[0],  'max': hold_length_range[1],  'type': 'float', 'step': hold_length_range[2]},
            {'name': 'C',  'min': camber_range_upper[0], 'max': camber_range_upper[1], 'type': 'float', 'step': camber_range_upper[2]},
            {'name': 'CT', 'min': camber_range_trunk[0], 'max': camber_range_trunk[1], 'type': 'float', 'step': camber_range_trunk[2]},
            {'name': 'DS', 'min': ds_range[0],           'max': ds_range[1],           'type': 'float', 'step': ds_range[2]},
            {'name': 'DB', 'min': db_range[0],           'max': db_range[1],           'type': 'float', 'step': db_range[2]},
            {'name': 'R',  'min': bilge_range[0],        'max': bilge_range[1],        'type': 'float', 'step': bilge_range[2]},
            {'name': 'LB', 'min': lbhd_ratio[0],         'max': lbhd_ratio[1],         'type': 'float', 'step': lbhd_ratio[2]},
            {'name': 'G0', 'min': g0_ratio[0],           'max': g0_ratio[1],           'type': 'float', 'step': g0_ratio[2]},
            {'name': 'G1', 'min': g1_ratio[0],           'max': g1_ratio[1],           'type': 'float', 'step': g1_ratio[2]},
            {'name': 'G2', 'min': g2_ratio[0],           'max': g2_ratio[1],           'type': 'float', 'step': g2_ratio[2]},
            {'name': 'S1', 'min': s1_ratio[0],           'max': s1_ratio[1],           'type': 'float', 'step': s1_ratio[2]},
            {'name': 'S2', 'min': s2_ratio[0],           'max': s2_ratio[1],           'type': 'float', 'step': s2_ratio[2]},
            {'name': 'S3', 'min': s3_ratio[0],           'max': s3_ratio[1],           'type': 'float', 'step': s3_ratio[2]},
        ]
        lhs = lhs_samples(N, specs, seed=SEED)
        for s in lhs:
            n_hold = pick_n_hold()
            s['L'] = _est_L(s['HL'], n_hold) if L_fixed is None else L_fixed
            s['N_HOLD'] = n_hold
            candidate_params.append(s)
    else:
        raise ValueError("method must be one of ['lhs','random','grid']")

    _index_dir = os.path.dirname(os.path.abspath(save_dir))
    index_csv = os.path.join(_index_dir, "LNGC_dataset_index.csv")
    header = [
        'file', 'json', 'method', 'seed',
        'Cargo Capacity (K)', 'Ship Length_m (L)', 'Ship Breadth_m (B)', 'Ship Depth_m (D)', 'Hold Length_m (HL)',
        'C.L. based Upper Deck Camber height_m (C)', 'C.L. based Trunk Deck Camber height_m (CT)',
        'Double Side Width_m (DS)', 'Double Bottom Height_m (DB)', 'Bilge Radius_m (R)',
        'B_Girder ratio to B/2 (G1)', 'Out_Girder ratio to B/2 (G2)',
        'Str1 ratio to D (S1)', 'Str2 ratio to D (S2)', 'Str3 ratio to D (S3)',
        'domain_ok', 'domain_issues',
        'kr_scope_status', 'kr_pass', 'kr_fail', 'kr_undetermined', 'kr_not_modeled',
        'qc_ok', 'label_overlaps', 'filesize', 'png', 'stiffeners_total', 'labels_count',
    ]

    saved = 0
    for i, p in enumerate(candidate_params, start=1):
        ok, issues, detail = domain_rules_ok_lngc(p)

        B = p['B']; D = p['D']
        y_0gir = p['G0'] * (B/2.0)
        y_1gir = p['G1'] * (B/2.0)
        y_2gir = p['G2'] * (B/2.0)
        z_1str = p['S1'] * D
        z_2str = p['S2'] * D
        z_3str = p['S3'] * D
        HL = p['HL']
        number_of_cofferdam = int(p['N_HOLD']) + 1
        hold_total = hold_len_factor * HL * int(p['N_HOLD'])
        fwd_len = fwd_hold_ratio * hold_total
        er_len  = er_hold_ratio  * hold_total
        aft_len = aft_hold_ratio * hold_total

        ship = LNGC(
            L=p['L'], B=B, D=D,
            d_ds=p['DS'], d_db=p['DB'], h_camber=p['C'],
            y_0gir=y_0gir, y_1gir=y_1gir, y_2gir=y_2gir,
            z_3str=z_3str, z_2str=z_2str, z_1str=z_1str, r_bilge=p['R'],
            h_camber_trunk=p['CT']
        )

        _gen_inputs_for_kr = {
            'L_m': p['L'], 'B_m': B, 'D_m': D,
            'doubleSide_m': p['DS'], 'doubleBottom_m': p['DB'],
            'bilgeRadius_m': p['R'],
            'number_of_cofferdam': number_of_cofferdam,
            'inner_slope_deg': getattr(ship, 'inner_slope_deg', None),
        }
        kr_eval = evaluate_kr_rules_lngc(_gen_inputs_for_kr, ship)

        dxf_path = build_filename(
            save_dir, p['L'], B, D, p['C'], p['CT'], p['DS'], p['DB'], p['R'],
            p['LB'], p['G1'], p['G2'], p['S1'], p['S2'], p['S3']
        )

        exporter = DXFExporterMM(ship, text_height=text_height, offset=offset,
                                 hold_length_m=HL, hold_vol_factor=hold_vol_factor,
                                 number_of_hold=p['N_HOLD'])
        qc, png_path = exporter.export(save_as=dxf_path, png_out_dir=png_out_dir, png_dpi=png_dpi)
        stats = getattr(exporter, 'export_stats', {})

        capacity_token = None
        try:
            capacity_token = stats.get('compartments', {}).get('volumes', {}).get('cargo_capacity_token')
        except Exception:
            capacity_token = None

        final_dxf_path = dxf_path
        if capacity_token:
            base = os.path.basename(dxf_path)
            hold_tag = f"{int(p['N_HOLD'])}Hold"
            if not base.startswith(f"{capacity_token}_{hold_tag}_"):
                new_base = f"{capacity_token}_{hold_tag}_{base}"
                new_dxf = os.path.join(os.path.dirname(dxf_path), new_base)
                try:
                    os.replace(dxf_path, new_dxf); final_dxf_path = new_dxf
                    if png_path:
                        new_png = os.path.join(os.path.dirname(png_path), os.path.splitext(new_base)[0] + ".png")
                        try: os.replace(png_path, new_png); png_path = new_png
                        except Exception: pass
                except Exception: pass

        try:
            stats['drawing']['files']['dxf'] = final_dxf_path
            stats['drawing']['files']['png'] = png_path
        except Exception: pass

        base_noext = os.path.splitext(os.path.basename(final_dxf_path))[0]
        compart_dxf_path = compart_png_path = compart3d_dxf_path = compart3d_png_path = None

        layout = None
        if compart_out_dir is not None or compart3d_out_dir is not None:
            layout = build_longitudinal_layout(
                L_m=p['L'], HL_m=HL, number_of_hold=p['N_HOLD'],
                fwd_len_m=fwd_len, er_len_m=er_len, aft_len_m=aft_len,
                hold_len_factor=hold_len_factor,
                cofferdam_len_m=cofferdam_len,
            )

        if compart_out_dir is not None and layout is not None:
            compart_dxf_path = os.path.join(compart_out_dir, base_noext + "_Compart.dxf")
            compart_dxf_path, compart_png_path = create_compartment_arrangement_drawing(
                compart_dxf_path, layout=layout, D_m=D, camber_m=p['C'], DB_m=p['DB'],
                text_height=text_height, png_dir=compart_png_out_dir, png_dpi=png_dpi,
            )

        if compart3d_out_dir is not None and layout is not None:
            compart3d_dxf_path = os.path.join(compart3d_out_dir, base_noext + "_Compart3D.dxf")
            compart3d_dxf_path, compart3d_png_path = create_compartment3d_dxf(
                compart3d_dxf_path, ship=ship, layout=layout,
                text_height=text_height, png_dir=compart3d_png_out_dir, png_dpi=png_dpi,
            )

        _json_base = os.path.basename(final_dxf_path).replace(".dxf", ".json")
        if json_out_dir:
            os.makedirs(json_out_dir, exist_ok=True)
            json_path = os.path.join(json_out_dir, _json_base)
        else:
            json_path = final_dxf_path.replace(".dxf", ".json")

        try:
            fsize = os.path.getsize(final_dxf_path)
        except Exception:
            fsize = -1

        hold_part = (number_of_cofferdam * cofferdam_len + hold_len_factor * HL * p['N_HOLD'])
        L_estimated = fwd_len + hold_part + er_len + aft_len

        meta = {
            'sample_id': f"LNGC-{saved+1:04d}",
            'ship_type': 'LNGC',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': method, 'seed': SEED,
            'generator_inputs': {
                'L_m': p['L'], 'B_m': B, 'D_m': D, 'HL_m': HL,
                'number_of_hold': int(p['N_HOLD']),
                'number_of_cofferdam': number_of_cofferdam, 'cofferdam_len_m': cofferdam_len,
                'camberUpper_m': p['C'], 'camberTrunk_m': p['CT'],
                'doubleSide_m': p['DS'], 'doubleBottom_m': p['DB'], 'bilgeRadius_m': p['R'],
                'girder0_ratio': p['G0'], 'girder1_ratio': p['G1'], 'girder2_ratio': p['G2'],
                'stringer1_ratio': p['S1'], 'stringer2_ratio': p['S2'], 'stringer3_ratio': p['S3'],
            },
            'geometry': {
                'derived': {
                    'CL_Girder_y_m': y_0gir, 'B_Girder_y_m': y_1gir, 'Out_Girder_y_m': y_2gir,
                    'IHull_y_m': B / 2 - p['DS'],
                    'z_trunk_m': getattr(ship, 'z_trunk', None),
                    'z_flat_m': getattr(ship, 'z_flat', None),
                },
                'longitudinal_layout': layout,
                'length_model': {
                    'fwd_len_m': fwd_len, 'er_len_m': er_len, 'aft_len_m': aft_len,
                    'number_of_cofferdam': number_of_cofferdam, 'cofferdam_len_m': cofferdam_len,
                    'hold_len_factor': hold_len_factor, 'hold_vol_factor': hold_vol_factor,
                    'L_estimated_m': L_estimated, 'used_L_m': p['L'],
                    'mode': 'fixed' if (L_fixed is not None) else 'estimated',
                },
            },
            'standard_refs': {'kr_standard': KR_STANDARD_INFO},
            'rules': {**kr_eval, 'society': 'KR'},  # unified schema (Phase 0.2.B1)
            'kr': kr_eval,  # legacy alias — kept for backward compat
            'multi_society_checks': multi_society_checks,
            'domain': {'ok': ok, 'issues': issues, 'detail': detail},
            'cargo_summary': {
                'per_hold_full_m3': stats.get('compartments', {}).get('volumes', {}).get('cargo_per_hold_full_m3'),
                'total_full_m3': stats.get('compartments', {}).get('volumes', {}).get('cargo_total_full_m3'),
                'capacity_token': capacity_token,
            },
            'artifacts': {
                'section_dxf': final_dxf_path, 'section_png': png_path,
                'compart_dxf': compart_dxf_path, 'compart_png': compart_png_path,
                'compart3d_dxf': compart3d_dxf_path, 'compart3d_png': compart3d_png_path,
                'json': json_path,
            },
            'drawing': stats,
        }
        write_metadata_json(json_path, meta)

        row = {
            'file': os.path.basename(final_dxf_path), 'json': os.path.basename(json_path),
            'method': method, 'seed': SEED,
            'Cargo Capacity (K)': capacity_token or "",
            'Ship Length_m (L)': p['L'], 'Ship Breadth_m (B)': B, 'Ship Depth_m (D)': D, 'Hold Length_m (HL)': HL,
            'C.L. based Upper Deck Camber height_m (C)': p['C'],
            'C.L. based Trunk Deck Camber height_m (CT)': p['CT'],
            'Double Side Width_m (DS)': p['DS'], 'Double Bottom Height_m (DB)': p['DB'], 'Bilge Radius_m (R)': p['R'],
            'B_Girder ratio to B/2 (G1)': p['G1'], 'Out_Girder ratio to B/2 (G2)': p['G2'],
            'Str1 ratio to D (S1)': p['S1'], 'Str2 ratio to D (S2)': p['S2'], 'Str3 ratio to D (S3)': p['S3'],
            'domain_ok': ok, 'domain_issues': "|".join(issues),
            'kr_scope_status': next((c.get('status') for c in kr_eval.get('auto_checks', []) if c.get('check_id') == 'lngc_scope'), ""),
            'kr_pass': kr_eval.get('summary', {}).get('check_counts', {}).get('pass', 0),
            'kr_fail': kr_eval.get('summary', {}).get('check_counts', {}).get('fail', 0),
            'kr_undetermined': kr_eval.get('summary', {}).get('check_counts', {}).get('undetermined', 0),
            'kr_not_modeled': kr_eval.get('summary', {}).get('check_counts', {}).get('not_modeled', 0),
            'qc_ok': qc.get('ok', False), 'label_overlaps': qc.get('label_overlaps', -1),
            'filesize': fsize, 'png': os.path.basename(png_path) if png_path else "",
            'stiffeners_total': stats.get('drawing', {}).get('stiffeners', {}).get('total', -1) if stats else -1,
            'labels_count': stats.get('drawing', {}).get('labels', {}).get('count', -1) if stats else -1,
        }
        append_index_csv(index_csv, header, row)

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
    _BASE = "<SHIPBENCH_ROOT>/data/processed/LNGC"

    SAVE_DIR        = os.path.join(_BASE, "section_dxf")
    PNG_DIR         = os.path.join(_BASE, "section_png")
    COMPART_DIR        = os.path.join(_BASE, "compart_dxf")
    COMPART_PNG_DIR    = os.path.join(_BASE, "compart_png")
    COMPART3D_DIR     = os.path.join(_BASE, "compart3d_dxf")
    COMPART3D_PNG_DIR = os.path.join(_BASE, "compart3d_png")
    JSON_DIR        = os.path.join(_BASE, "json")

    generate_lngc_dataset(
        save_dir=SAVE_DIR,
        json_out_dir=JSON_DIR,
        method='lhs',
        fwd_hold_ratio=0.16, er_hold_ratio=0.28, aft_hold_ratio=0.14,
        cofferdam_len=2.5,
        hold_len_factor=1.0, hold_vol_factor=0.7,
        number_of_hold_range=(3, 5, 1),
        compart_out_dir=COMPART_DIR, compart_png_out_dir=COMPART_PNG_DIR,
        compart3d_out_dir=COMPART3D_DIR, compart3d_png_out_dir=COMPART3D_PNG_DIR,
        B_range=(40, 50, 1), D_range=(20, 30, 1),
        hold_length_range=(40.0, 50.0, 0.1),
        camber_range_upper=(0.3, 1.5, 0.1), camber_range_trunk=(0.1, 1.0, 0.1),
        ds_range=(1.5, 3.5, 0.1), db_range=(2.2, 4.2, 0.1),
        bilge_range=(3.0, 5.0, 0.5),
        g1_ratio=(0.2, 0.3, 0.05), g2_ratio=(0.5, 0.7, 0.05),
        s1_ratio=(0.8, 0.9, 0.05), s2_ratio=(0.55, 0.7, 0.05), s3_ratio=(0.3, 0.4, 0.05),
        text_height=250, offset=300, MAX_FILES=100, PROGRESS_EVERY=20,
        SEED=42, png_out_dir=PNG_DIR, png_dpi=220
    )
