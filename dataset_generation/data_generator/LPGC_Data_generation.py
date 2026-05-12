# =========================================
#   LPGC Midship Generator + Elevation + 3D Model (Self-contained)
#   - KR Rules 2025 Pt7 Ch5 (LPG Carriers)
#   - Full geometry class + rule framework + elevation + 3D + dataset generator
# =========================================

# =========================================
#   LPGC Midship Generator (LNGC-style outputs, with Independent Cargo Tank)
#   - Drawing visuals preserved
#   - JSON/CSV/filename aligned to LNGC
#   - Hold Length sampled; Tank Length = HL - 2.5 m (fore/aft clearances)
#   - Cargo capacity uses Tank Length and Cargo-layer area
#   - Estimated ship length uses Hold Length (HL)
# =========================================

import os, csv, json, time, random, re
from math import sin, cos, tan, atan2, radians, pi, hypot, degrees

import ezdxf
from ezdxf.lldxf import const as ezc

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
_MATPLOT_OK = _MAT_OK

# =========================================
#            Common helpers (LNGC-like)
# =========================================
EPS = 1e-9
EXPORT_INCLUDE_MEMBER_BBOX = False


# ── Ship type identifier for hull-form renderer ──
_SHIP_TYPE = 'LPGC'

def float_range(start, stop, step):
    n0=int(round(start/step)); n1=int(round(stop/step))
    for k in range(n0, n1+1):
        yield round(k*step, 10)

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

def line_intersection(p1, p2, p3, p4):
    x1,y1=p1; x2,y2=p2; x3,y3=p3; x4,y4=p4
    den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if abs(den)<EPS: return None
    px=((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/den
    py=((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/den
    return (px,py)

def _round_floats(obj, ndigits=3):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj

def clean_multiline_label(label: str) -> str:
    return label.replace("\\P", " ").replace("  ", " ").strip()

# Abbrev expansion (identity fallback to keep LNGC compatibility fields present)
def expand_abbrev(token: str) -> str:
    return clean_multiline_label(token or "")

# ------- LNGC-compact params view (from LPGC inputs) -------
def build_params_compact_lngc_from_lpgc(p: dict) -> dict:
    """
    Produce LNGC-style compact params keys from LPGC variables.
    Unknowns are filled with 0.0 or ratios if derivable.
    """
    B = p['B']; D = p['D']
    G1_ratio = (p['GY'] / (B/2.0)) if B>0 else 0.0          # inboard girder y to B/2
    G2_ratio = p['OG']                                      # out girder ratio to B/2
    return {
        'L_m':  p.get('L', 0.0),
        'B_m':  B,
        'D_m':  D,
        'HL_m': p.get('HL', None),                     # Hold Length for this hold
        'camberUpper_m':  p['C'],
        'camberTrunk_m':  0.0,                         # N/A for LPGC midship; keep 0.0 for schema parity
        'doubleBottom_m': p['DB'],
        'bilgeRadius_m':  p['R'],
        'girderCL_ratio': 0.0,                         # not used in this model
        'girderB_ratio':  G1_ratio,
        'girderOut_ratio': G2_ratio,
        'str1_ratio': p.get('S1', 0.0),
        'str2_ratio': p.get('S2', 0.0),
        'str3_ratio': 0.0,
    }

# =========================================
#            File naming (base; token 앞에 붙임)
# =========================================
def fmt_token(val, nd=1):
    """파일명 안전용 토큰 (소수점 -> 'p')"""
    s = f"{val:.{nd}f}"
    # 소수점이 있을 때(nd>0)만 뒷자리 0과 점을 제거
    if nd > 0 and '.' in s:
         s = s.rstrip('0').rstrip('.')
    return s.replace('.', 'p')

def build_filename(base_dir, L,B,D,C, DB,R,
                   GY, OG, TSWT_EXT,
                   GAP_TSWT, GAP_HOP, STRCLR,
                   S1,S2):
    name=(f"LPGC_L{fmt_token(L, 0)}_B{fmt_token(B)}_D{fmt_token(D)}_"
          f"C{fmt_token(C)}_DB{fmt_token(DB)}_R{fmt_token(R)}_"
          f"GY{fmt_token(GY)}_OG{fmt_token(OG)}_TSWTE{fmt_token(TSWT_EXT)}_"
          f"GTS{fmt_token(GAP_TSWT)}_GH{fmt_token(GAP_HOP)}_SC{fmt_token(STRCLR)}_"
          f"S1{fmt_token(S1)}_S2{fmt_token(S2)}.dxf")
    return os.path.join(base_dir, name)

# =========================================
#                 Geometry
# =========================================
class LPGC:
    """Right half (y>=0). Green=hull, Blue=cargo."""
    def __init__(self, L,B,D, DB,R, camber,
                 y_girder, y_og_ratio, tswt_ext_deg,
                 gap_tswt, gap_hopper, str_clear,
                 s1_ratio, s2_ratio):
        self.L=L; self.B=B; self.D=D
        self.DB=DB; self.R=R; self.camber=camber
        self.y_girder=y_girder
        self.y_og = y_og_ratio*(B/2)
        self.tswt_ext = tswt_ext_deg
        self.gap_tswt=gap_tswt; self.gap_hopper=gap_hopper; self.str_clear=str_clear
        self.s1_ratio=s1_ratio; self.s2_ratio=s2_ratio

        # Tank-to-side-shell clearance (IGC 2.4.1 Type 2G, single-side hull).
        # LPG is classified Type 2G per IMO IGC Ch.19 Products List, requiring
        #   cargo tank inboard distance from side shell >= max(0.76 m, B/15).
        # We add a +0.3 m engineering margin over the IGC minimum so that small-beam
        # ships still keep the traditional ~1.8 m clearance while large-beam ships
        # (B~42) scale up to ~3.1 m. This is exposed as an attribute so the rule
        # evaluator can read the actual value instead of proxying via gap_hopper.
        import math as _math
        self.tank_side_clearance = max(1.8, B / 15.0 + 0.3)
        self.y_ts = (B / 2) - self.tank_side_clearance

        # ---- hull (green) ----
        self._build_hull()
        # ---- cargo (blue) ----
        self._build_cargo()

    def z_deck(self, y):
        return -(self.camber/(self.B/2))*y + (self.D + self.camber)

    def _build_hull(self):
        L, B, D, DB, R = self.L, self.B, self.D, self.DB, self.R
        self.str1 = None; self.str2 = None

        # Shell lines
        self.m_btm  = [[L*500, L*500], [0, (B/2 - R)*1000], [0, 0]]
        self.m_side = [[L*500, L*500], [B*500, B*500], [R*1000, D*1000]]
        self.m_deck = [[L*500, L*500], [0, B*500], [self.z_deck(0)*1000, D*1000]]

        # Inner bottom, hopper
        import math
        self.m_ibtm = [[L*500, L*500], [0, self.y_og*1000], [DB*1000, DB*1000]]
        hop_ang = math.radians(42.0)
        z_hopp_side = DB + math.tan(hop_ang) * (B/2 - self.y_og)
        z_hopp_side = min(D - 0.8, max(R + 0.6, z_hopp_side))
        self.m_hopp = [[L*500, L*500], [self.y_og*1000, (B/2)*1000], [DB*1000, z_hopp_side*1000]]

        # Girders
        self.m_outg = [[L*500, L*500], [self.y_og*1000, self.y_og*1000], [0, DB*1000]]
        self.m_gird = [[L*500, L*500], [self.y_girder*1000, self.y_girder*1000], [0, DB*1000]]

        # TSWT (kink)
        y_tsy = (B/2)/2.0
        z_top = self.z_deck(y_tsy)
        z_kink = z_top - 0.7
        phi_req = math.radians(90.0 - self.tswt_ext)           # -50° ~ -30°
        phi = max(min(phi_req, math.radians(-30.0)), math.radians(-50.0))
        z_ts_side = z_kink + math.tan(phi) * (B/2 - y_tsy)
        self.tswt_vert  = [[L*500, L*500], [y_tsy*1000, y_tsy*1000], [z_top*1000,  z_kink*1000]]
        self.tswt_slope = [[L*500, L*500], [y_tsy*1000, (B/2)*1000], [z_kink*1000, z_ts_side*1000]]

        # Stringers — inboard end must sit 0.2 m outboard of the tank side wall
        # (y_ts), so stringer span scales with tank_side_clearance. Previously
        # hardcoded to (B/2 - 1.6), which collided with enlarged clearances.
        y_str_inboard = self.y_ts + 0.2
        z1 = z_ts_side - 1.7
        self.str1 = [[L*500, L*500], [y_str_inboard*1000, (B/2)*1000], [z1*1000, z1*1000]]
        dz_gap = abs(z_hopp_side - z1)
        if dz_gap >= 5.0:
            z2 = 0.5*(z1 + z_hopp_side)
            self.str2 = [[L*500, L*500], [y_str_inboard*1000, (B/2)*1000], [z2*1000, z2*1000]]
        else:
            self.str2 = None

    def _build_cargo(self):
        import math
        L, B, D, DB = self.L, self.B, self.D, self.DB
        y_tsy = (B/2)/2.0
        # y_ts is now derived from tank_side_clearance in __init__ (IGC 2.4.1
        # Type 2G: >= max(0.76, B/15) with +0.3 m engineering margin).
        y_ts  = self.y_ts

        # TSWT parallel offset
        y1 = self.tswt_slope[1][0]/1000; z1 = self.tswt_slope[2][0]/1000
        y2 = self.tswt_slope[1][1]/1000; z2 = self.tswt_slope[2][1]/1000
        dy = y2-y1; dz = z2-z1
        phi = math.atan2(dz, dy) if abs(dy)>1e-9 else -math.radians(40.0)
        m_ts = math.tan(phi); n_ts = (math.sin(phi), -math.cos(phi))
        z_kink = self.tswt_vert[2][1]/1000
        p_off0 = (y_tsy + n_ts[0]*self.gap_tswt, z_kink + n_ts[1]*self.gap_tswt)

        z_top_at_yts = p_off0[1] + m_ts*(y_tsy - p_off0[0])
        p_top0=(0.0, z_top_at_yts); p_top1=(y_tsy, z_top_at_yts)
        z_top_at_ts  = p_off0[1] + m_ts*(y_ts  - p_off0[0])
        p_slope0=(y_tsy, z_top_at_yts)

        # Hopper parallel
        h_s=(self.m_hopp[1][0]/1000, self.m_hopp[2][0]/1000)
        h_e=(self.m_hopp[1][1]/1000, self.m_hopp[2][1]/1000)
        alpha = math.atan2(h_e[1]-h_s[1], h_e[0]-h_s[0])
        n_hp = (-math.sin(alpha), math.cos(alpha))
        th_s=(h_s[0]+n_hp[0]*self.gap_hopper, h_s[1]+n_hp[1]*self.gap_hopper)
        th_e=(h_e[0]+n_hp[0]*self.gap_hopper, h_e[1]+n_hp[1]*self.gap_hopper)

        # Tank_Side bottom z = Tank_Hopper at y=y_ts
        if abs(th_e[0]-th_s[0])<1e-9: z_hop_at_ts = th_s[1]
        else:
            t2=(y_ts - th_s[0])/(th_e[0]-th_s[0])
            z_hop_at_ts = th_s[1] + t2*(th_e[1]-th_s[1])

        # Tank bottom
        z_bottom = DB + 0.5
        if abs(th_e[1]-th_s[1])<1e-9: y_bot_end = th_e[0]
        else:
            y_bot_end = th_s[0] + (z_bottom - th_s[1]) * ((th_e[0]-th_s[0])/(th_e[1]-th_s[1]))
        y_bot_end = max(0.0, min(y_bot_end, B/2))

        # Entities (mm)
        self.c_Tank_Top_h = [[L*500, L*500], [p_top0[0]*1000, p_top1[0]*1000], [p_top0[1]*1000, p_top1[1]*1000]]
        self.c_Tank_Top_s = [[L*500, L*500], [p_slope0[0]*1000, y_ts*1000],    [p_slope0[1]*1000, z_top_at_ts*1000]]
        self.c_Tank_Side  = [[L*500, L*500], [y_ts*1000, y_ts*1000],          [z_top_at_ts*1000, z_hop_at_ts*1000]]
        self.c_Tank_Hopp  = [[L*500, L*500], [y_bot_end*1000, y_ts*1000],     [z_bottom*1000,    z_hop_at_ts*1000]]
        self.c_Tank_Btm   = [[L*500, L*500], [0, y_bot_end*1000],             [z_bottom*1000,    z_bottom*1000]]

    # == standard segment dict ==
    def seg_dict(self):
        def seg(m):
            if m is None: return None
            return ((float(m[1][0]), float(m[2][0])),
                    (float(m[1][1]), float(m[2][1])))
        M={}
        M["Upper_Deck"]   = seg(self.m_deck)
        M["Bottom_Shell"] = seg(self.m_btm)
        M["Side_Shell"]   = seg(self.m_side)
        M["IBTM"]         = seg(self.m_ibtm)
        M["Hopper"]       = seg(self.m_hopp)
        M["Out_Girder"]   = seg(self.m_outg)
        M["Girder"]       = seg(self.m_gird)
        M["TSWT_V"]       = seg(self.tswt_vert)
        M["TSWT"]         = seg(self.tswt_slope)
        M["Str1"]         = seg(self.str1)
        M["Str2"]         = seg(self.str2)
        # cargo
        M["Tank_Top"]     = seg(self.c_Tank_Top_h)
        M["Tank_TSWT"]    = seg(self.c_Tank_Top_s)
        M["Tank_Side"]    = seg(self.c_Tank_Side)
        M["Tank_Hopper"]  = seg(self.c_Tank_Hopp)
        M["Tank_Bottom"]  = seg(self.c_Tank_Btm)
        return {k:v for k,v in M.items() if v is not None}

# =========================================
#          Stiffener type / scantling config
# =========================================
_STF_CFG = {
    # (stf_type, flange_half_mm, web_h_mm)  ← web_h from _SCANTLING_TABLE
    "Upper_Deck":   ("T",  75, 350),  # 350 x 12 + 150 x 20 F.B(T)
    "Bottom_Shell": ("T",  75, 380),  # 380 x 14 + 150 x 20 F.B(T)
    "Side_Shell":   ("T",  65, 300),  # 300 x 12 + 130 x 18 F.B(T)
    "IBTM":         ("T",  75, 350),  # 350 x 12 + 150 x 18 F.B(T)  (Inner Bottom)
    "Hopper":       ("IA", 90, 280),  # 280 x 10 + 90 x 14 I.A
    "TSWT":         ("IA", 90, 300),  # 300 x 10 + 90 x 14 I.A
    "TSWT_V":       ("FB",  0, 300),  # vertical TSWT (not in table)
    "Girder":       ("FB",  0, 150),  # 150 x 10 F.B
    "Out_Girder":   ("FB",  0, 120),  # 120 x 10 F.B
    "Str1":         ("FB",  0, 120),  # 120 x 10 F.B
    "Str2":         ("IA", 60, 150),  # 150 x 8 + 60 x 12 I.A
}

_STF_TYPE_LEGEND = {
    "F.B":    "Flat Bar — web only, no flange",
    "I.A":    "Inverted Angle — web + one-side flange (L-shape)",
    "F.B(T)": "Built-up T-bar — web + both-side flanges (T-shape)",
}

_SCANTLING_TABLE = [
    ("MEMBER",         "PLATE (mm)", "STIFFENER"),
    ("Upper Deck",     "14.0",       "350 x 12 + 150 x 20 F.B(T)"),
    ("Bottom Shell",   "16.0",       "380 x 14 + 150 x 20 F.B(T)"),
    ("Side Shell",     "14.0",       "300 x 12 + 130 x 18 F.B(T)"),
    ("Inner Bottom",   "14.0",       "350 x 12 + 150 x 18 F.B(T)"),
    ("Hopper",         "12.0",       "280 x 10 + 90 x 14 I.A"),
    ("TSWT (slope)",   "13.0",       "300 x 10 + 90 x 14 I.A"),
    ("Girder",         "11.0",       "150 x 10 F.B"),
    ("Out Girder",     "10.0",       "120 x 10 F.B"),
    ("Str1",           "10.0",       "120 x 10 F.B"),
    ("Str2",           "11.0",       "150 x 8 + 60 x 12 I.A"),
]

# =========================================
#               Exporter (drawing preserved; +metadata collection)
# =========================================
class DXFExporter:
    def __init__(self, ship, text_height=250, offset=300,
                 stf_min=700, stf_max=1000, stf_target=850, stf_len=400, edge_clear=10,
                 label_offset=300, label_dir=None, label_flip=None,
                 hold_length_m=None, tank_length_m=None,
                 hold_len_factor=1.0, hold_vol_factor=0.7, number_of_hold=3):
        self.s=ship; self.text_height=text_height; self.offset=offset
        self.stf_min=stf_min; self.stf_max=stf_max; self.stf_target=stf_target
        self.stf_len=stf_len; self.edge_clear=edge_clear
        self.doc=ezdxf.new(setup=True); self.msp=self.doc.modelspace()
        self.label_offset = label_offset

        self.bilge_bottom_end=None; self.bilge_side_start=None
        self.hold_length_m = hold_length_m
        self.tank_length_m = tank_length_m

        # NEW
        self.hold_len_factor = float(hold_len_factor)
        self.hold_vol_factor = float(hold_vol_factor)
        self.number_of_hold = int(number_of_hold)

        default_dir = {
            "Upper_Deck":   (0.0, +1.0), "Bottom_Shell": (0.0, -1.0), "IBTM": (0.0, +1.0),
            "Side_Shell":   (+1.0, 0.0), "Out_Girder": (+1.0, 0.0), "Girder": (-1.0, 0.0),  # Side_Shell outboard
            "TSWT":         (0.0, -1.0),
            "Tank_Top":     (0.0, -1.0), "Tank_TSWT": (0.0, -1.0),
            "Tank_Side":    (-1.0, 0.0), "Tank_Hopper": (-1.0, 0.0), "Tank_Bottom": (0.0, +1.0),
        }
        self.LABEL_DIR = dict(default_dir if label_dir is None else label_dir)
        if label_flip:
            for k in label_flip:
                if k in self.LABEL_DIR:
                    ny, nz = self.LABEL_DIR[k]; self.LABEL_DIR[k] = (-ny, -nz)

        # metadata collectors
        self._labels = []
        self._stf_stats = {}
        self._compartment_data = []
        self._intersections = []
        self._placed_label_polys = []
        self._segs_all = None

    # -------- drawing utilities (unchanged visuals) --------
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
        label_text = display if display is not None else key
        txt = self.msp.add_mtext(label_text, dxfattribs={"char_height": self.text_height, "layer": layer})
        txt.dxf.insert=(px,pz); txt.dxf.attachment_point=5; txt.dxf.rotation=ang_deg
        self._labels.append({'name': label_text, 'pos': (float(px), float(pz)),
                             'rotation_deg': float(ang_deg), 'layer': layer})

    def _text(self, s, pos, rot=0, layer="Label"):
        t=self.msp.add_mtext(s, dxfattribs={'char_height':self.text_height,'layer':layer})
        t.set_location(pos, rotation=rot)
        self._labels.append({'name': s, 'pos': (float(pos[0]), float(pos[1])),
                             'rotation_deg': float(rot), 'layer': layer})

    def draw_layers(self):
        def L(n,c):
            if n not in self.doc.layers: self.doc.layers.add(n).dxf.color=c
        L("Members",3); L("Label",1); L("Compartment",6); L("Bilge",3)
        L("Stiffeners (Longi)", 4)   # cyan — longitudinal stiffeners
        L("Stiffeners (Trans)", 30)  # orange — transverse indicators
        L("Center",8); L("Cargo",5)
        L("Scantling", 252)          # dark gray — scantling table

    def draw_centerline(self):
        top=self.s.m_deck[2][0] + 500
        ln=self.msp.add_line((0,0),(0,top), dxfattribs={'layer':'Center'})
        try: ln.dxf.linetype="CENTER"; ln.dxf.ltscale=200
        except Exception: pass
        self._text("C.L.", (-500, top+300), rot=90)

    def draw_hull(self):
        parts = [
            ("Upper_Deck","m_deck"), ("Bottom_Shell","m_btm"), ("Side_Shell","m_side"),
            ("IBTM","m_ibtm"), ("Hopper","m_hopp"),
            ("Out_Girder","m_outg"), ("Girder","m_gird"),
            ("TSWT_V","tswt_vert"), ("TSWT","tswt_slope"),
            ("Str1","str1"), ("Str2","str2"),
        ]
        for label, attr in parts:
            m = getattr(self.s, attr, None)
            if m is None: continue
            y, z = m[1], m[2]
            self.msp.add_line((y[0], z[0]), (y[1], z[1]), dxfattribs={'layer': 'Members'})
            if label != "TSWT_V":
                self._label_on_member(label, y[0], z[0], y[1], z[1], layer="Label")

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

        label_radial_offset = 300
        theta_mid = (a1 + a2) / 2
        lx = cy + (R + label_radial_offset) * cos(theta_mid)
        lz = cz + (R + label_radial_offset) * sin(theta_mid)
        lab = self.msp.add_mtext("Bilge", dxfattribs={'char_height': self.text_height, 'layer': 'Label'})
        lab.dxf.insert=(lx,lz); lab.dxf.attachment_point=4; lab.dxf.rotation=0
        self._labels.append({'name': 'Bilge', 'pos': (float(lx), float(lz)), 'rotation_deg': 0.0, 'layer': 'Label'})

    def draw_cargo(self):
        parts = {
            "Tank_Top": self.s.c_Tank_Top_h,
            "Tank_TSWT": self.s.c_Tank_Top_s,
            "Tank_Side": self.s.c_Tank_Side,
            "Tank_Hopper": self.s.c_Tank_Hopp,
            "Tank_Bottom": self.s.c_Tank_Btm,
        }
        for nm, m in parts.items():
            y, z = m[1], m[2]
            self.msp.add_line((y[0], z[0]), (y[1], z[1]), dxfattribs={"layer": "Cargo"})
            self._label_on_member(nm, y[0], z[0], y[1], z[1], layer="Label")

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
    def _poly_area_perimeter(verts):
        n=len(verts)
        if n<3: return 0.0,0.0
        area2=0.0; per=0.0
        for i in range(n):
            x1,y1=verts[i]; x2,y2=verts[(i+1)%n]
            area2 += x1*y2 - x2*y1
            per   += hypot(x2-x1, y2-y1)
        return abs(area2)*0.5, per


    # ---- Title & Specs ----
    def draw_title_and_specs(self, title: str = "ORDINARY SECTION (STBD)"):
        # CL에서의 상갑판 z(mm)
        try:
            upper_deck_z = float(self.s.m_deck[2][0])  # z_deck(0)*1000
        except Exception:
            upper_deck_z = 0.0

        # Tanker와 동일 스타일: 제목을 갑판 위로 충분히 띄움
        base_z = upper_deck_z + 5000.0
        center_y = 0.0

        line_gap = self.text_height * 1.8

        def put_line(text: str, row: int, size_mult: float = 1.0):
            ty = base_z - line_gap * row
            t = self.msp.add_mtext(text, dxfattribs={'char_height': self.text_height * size_mult,
                                                     'layer': 'Label'})
            t.dxf.insert = (center_y, ty)
            t.dxf.attachment_point = 5
            t.dxf.rotation = 0
            # 메타 수집 (기존 컨벤션 유지)
            self._labels.append({
                'name': text,
                'pos': (float(center_y), float(ty)),
                'rotation_deg': 0.0,
                'layer': 'Label'
            })

        # Title
        put_line(title, row=0, size_mult=1.5)

        # BREADTH, DEPTH only — section drawing excludes longitudinal info
        # (NUMBER OF HOLD / HOLD LENGTH / TANK LENGTH / SHIP LENGTH belong to
        # the compartment-arrangement view).
        put_line(f"BREADTH = {float(self.s.B):.1f} m", row=1)
        put_line(f"DEPTH = {float(self.s.D):.1f} m", row=2)


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

    def draw_compartments(self):
        if "Compartment" not in self.doc.layers:
            self.doc.layers.add("Compartment").dxf.color = 6

        # ---- collect segments ----
        S = self.s.seg_dict()
        self._segs_all = dict(S)

        # ---- CL
        deck_top_cl = self.s.z_deck(0) * 1000.0
        S["CL"] = ((0.0, 0.0), (0.0, deck_top_cl))

        # ---- Bilge chord
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
            # Cargo tank 라벨은 draw_scantling_table()에서 갑판↔테이블 사이에 배치
            if "cargo" not in clean_multiline_label(name).lower():
                t = self.msp.add_mtext(name, dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
                t.dxf.insert=(cx,cy); t.dxf.attachment_point=5; t.dxf.rotation=0
                self._labels.append({'name': name, 'pos': (float(cx), float(cy)), 'rotation_deg': 0.0, 'layer': 'Compartment'})

            area_mm2, per_all = self._poly_area_perimeter(verts)
            per_excl_cl = 0.0
            def share_edge(ei, ej):
                a1,b1=ei; a2,b2=ej
                s1={a1,b1}; s2={a2,b2}
                inter=s1.intersection(s2)
                return next(iter(inter)) if inter else None
            n=len(verts)
            for i in range(n):
                j=(i+1)%n
                v1=verts[i]; v2=verts[j]
                common=share_edge(edges[i], edges[j])
                if common=="CL": continue
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
        label_poly("Cargo tank\\P(Type A)", [
            ("Tank_Top","Tank_TSWT"),
            ("Tank_TSWT","Tank_Side"),
            ("Tank_Side","Tank_Hopper"),
            ("Tank_Hopper","Tank_Bottom"),
            ("Tank_Bottom","CL"),
            ("CL","Tank_Top"),
        ])

        label_poly("Ballast tank 1\\P(T.S.W.B.T.)", [
            ("Upper_Deck","TSWT_V"),
            ("TSWT_V","TSWT"),
            ("TSWT","Side_Shell"),
            ("Side_Shell","Upper_Deck"),
        ])

        label_poly("Ballast tank 2\\P(D.B.W.B.T.)", [
            ("Hopper","Side_Shell"),
            ("Side_Shell","Bilge"),
            ("Bilge","Bottom_Shell"),
            ("Bottom_Shell","Out_Girder"),
            ("Out_Girder","Hopper"),
        ])

        label_poly("Ballast tank 3\\P(D.B.W.B.T.)", [
            ("IBTM","Out_Girder"),
            ("Out_Girder","Bottom_Shell"),
            ("Bottom_Shell","Girder"),
            ("Girder","IBTM"),
        ])

        label_poly("Pipe\\Pduct", [
            ("IBTM","Girder"),
            ("Girder","Bottom_Shell"),
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

        def seg(m):
            return ((float(m[1][0]), float(m[2][0])),
                    (float(m[1][1]), float(m[2][1])))

        segs_all = {}
        for name, attr in [
            ("Upper_Deck","m_deck"), ("Bottom_Shell","m_btm"), ("IBTM","m_ibtm"),
            ("Side_Shell","m_side"), ("Hopper","m_hopp"),
            ("Out_Girder","m_outg"), ("Girder","m_gird"),
            ("TSWT","tswt_slope"), ("TSWT_V","tswt_vert"),
            ("Str1","str1"), ("Str2","str2")
        ]:
            m = getattr(self.s, attr, None)
            if m is not None: segs_all[name] = seg(m)

        gen_names = [n for n in segs_all.keys() if n != "TSWT_V"]

        LEN_DEFAULT = self.stf_len; LEN_SHORT = 150.0

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
        def normal_of(name,p1,p2):
            uy,uz,_=dir_len(p1,p2)
            nA,nB=(-uz,uy),(uz,-uy)
            if name=="Bottom_Shell": return (0.0,+1.0)
            if name in ("IBTM","Str1","Str2"): return (0.0,-1.0)
            if name=="Side_Shell": return (-1.0,0.0)   # inboard
            if name=="Out_Girder": return (-1.0,0.0)
            if name=="Girder": return (+1.0,0.0)
            if name=="TSWT": return nA if nA[1]>nB[1] else nB
            if name=="Hopper": return nA if nA[0]>nB[0] else nB
            return nA if nA[1]<nB[1] else nB

        # ---- TSWT∩Side_Shell / Hopper∩Side_Shell 예각 방향 t-space exclusion ----
        JUNC_CLEAR = 1500.0
        _junc_excl = {}  # {member_name: [(t_lo, t_hi), ...]}

        if "TSWT" in segs_all and "Side_Shell" in segs_all:
            _ip_ts = line_intersection(*segs_all["TSWT"], *segs_all["Side_Shell"])
            if _ip_ts is not None:
                # TSWT: 교점이 끝단 → 예각 접근 방향(t < t_q)만 제거
                _p1, _p2 = segs_all["TSWT"]
                _uy, _uz, _L = dir_len(_p1, _p2)
                _t_q = (_ip_ts[0]-_p1[0])*_uy + (_ip_ts[1]-_p1[1])*_uz
                _t_q = max(0.0, min(_L, _t_q))
                _junc_excl.setdefault("TSWT", []).append((max(0.0, _t_q - JUNC_CLEAR), _t_q))
                # Side_Shell: TSWT 접합 위쪽이 예각 → [t_q, t_q+1500]
                _p1, _p2 = segs_all["Side_Shell"]
                _uy, _uz, _L = dir_len(_p1, _p2)
                _t_q = (_ip_ts[0]-_p1[0])*_uy + (_ip_ts[1]-_p1[1])*_uz
                _t_q = max(0.0, min(_L, _t_q))
                _junc_excl.setdefault("Side_Shell", []).append((_t_q, min(_L, _t_q + JUNC_CLEAR)))

        if "Hopper" in segs_all and "Side_Shell" in segs_all:
            _ip_hp = line_intersection(*segs_all["Hopper"], *segs_all["Side_Shell"])
            if _ip_hp is not None:
                # Hopper: 교점이 끝단 → 예각 접근 방향(t < t_q)만 제거
                _p1, _p2 = segs_all["Hopper"]
                _uy, _uz, _L = dir_len(_p1, _p2)
                _t_q = (_ip_hp[0]-_p1[0])*_uy + (_ip_hp[1]-_p1[1])*_uz
                _t_q = max(0.0, min(_L, _t_q))
                _junc_excl.setdefault("Hopper", []).append((max(0.0, _t_q - JUNC_CLEAR), _t_q))
                # Side_Shell: Hopper 접합 아래쪽이 예각 → [t_q-1500, t_q]
                _p1, _p2 = segs_all["Side_Shell"]
                _uy, _uz, _L = dir_len(_p1, _p2)
                _t_q = (_ip_hp[0]-_p1[0])*_uy + (_ip_hp[1]-_p1[1])*_uz
                _t_q = max(0.0, min(_L, _t_q))
                _junc_excl.setdefault("Side_Shell", []).append((max(0.0, _t_q - JUNC_CLEAR), _t_q))

        def _in_junc_excl(nm, pt):
            zones = _junc_excl.get(nm)
            if not zones: return False
            _p1, _p2 = segs_all[nm]
            _uy, _uz, _ = dir_len(_p1, _p2)
            _t = (pt[0]-_p1[0])*_uy + (pt[1]-_p1[1])*_uz
            return any(_t_lo - 1.0 <= _t <= _t_hi + 1.0 for _t_lo, _t_hi in zones)

        for name in gen_names:
            self._stf_stats.setdefault(name, 0)
            pieces=split(name)
            if not pieces: continue
            stf_type, flange_half, web_len = _STF_CFG.get(name, ("FB", 0, 400))
            for s,e in pieces:
                uy,uz,L = dir_len(s,e)
                n,sp = choose_spacing(L)
                if n<=0: continue
                nvec = normal_of(name,s,e)
                t0=self.edge_clear
                for i in range(1, n+1):
                    t=t0+sp*i
                    if t>=L-self.edge_clear+1e-6: break
                    base=(s[0]+uy*t, s[1]+uz*t)
                    if name in ("TSWT", "Hopper", "Side_Shell") and _in_junc_excl(name, base):
                        continue
                    self._draw_stiffener_shape(base, nvec, (uy,uz), stf_type, web_len, flange_half, _longi_layer)
                    self._stf_stats[name] += 1

        # ---- Bilge-end anchor stiffeners (BULKC-style: 100mm inboard on members) ----
        BILGE_END_OFFSET = 100.0
        if self.bilge_bottom_end is not None and self.bilge_side_start is not None:
            if "Bottom_Shell" in segs_all:
                p1, p2 = segs_all["Bottom_Shell"]
                uy_f, uz_f, _ = dir_len(p1, p2)
                be = self.bilge_bottom_end
                base_be = (be[0] - uy_f * BILGE_END_OFFSET, be[1] - uz_f * BILGE_END_OFFSET)
                nv_be = normal_of("Bottom_Shell", p1, p2)
                st, fh, wh = _STF_CFG.get("Bottom_Shell", ("T", 75, 380))
                self._draw_stiffener_shape(base_be, nv_be, (uy_f, uz_f), st, wh, fh, _longi_layer)
                self._stf_stats["Bottom_Shell"] = self._stf_stats.get("Bottom_Shell", 0) + 1
            if "Side_Shell" in segs_all:
                p1, p2 = segs_all["Side_Shell"]
                uy_f, uz_f, _ = dir_len(p1, p2)
                bs = self.bilge_side_start
                base_bs = (bs[0] + uy_f * BILGE_END_OFFSET, bs[1] + uz_f * BILGE_END_OFFSET)
                nv_bs = normal_of("Side_Shell", p1, p2)
                st, fh, wh = _STF_CFG.get("Side_Shell", ("T", 65, 300))
                self._draw_stiffener_shape(base_bs, nv_bs, (uy_f, uz_f), st, wh, fh, _longi_layer)
                self._stf_stats["Side_Shell"] = self._stf_stats.get("Side_Shell", 0) + 1

    # ---------------------- 스캔틀링 표 ----------------------
    def draw_scantling_table(self):
        layer = "Scantling"
        txt_h = 180.0; txt_h_hdr = 200.0
        col_w = [3600.0, 1900.0, 6800.0]; row_h = 700.0
        rows = _SCANTLING_TABLE
        n_rows = len(rows); total_w = sum(col_w); total_h = n_rows * row_h

        # 중심: Cargo Tank 구획 centroid 사용
        ch_cy, ch_cz = None, None
        for c in self._compartment_data:
            if "cargo" in c.get("clean_label", "").lower():
                ch_cy, ch_cz = c["centroid_mm"]; break
        if ch_cy is None:
            ch_cy = (self.s.B / 2.0) / 2.0 * 1000.0
            ch_cz = (self.s.DB + self.s.D) / 2.0 * 1000.0

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
        z_deck_top = self.s.z_deck(0) * 1000.0
        ch_label = self.msp.add_mtext("Cargo Tank",
            dxfattribs={"char_height": self.text_height, "layer": "Compartment"})
        ch_label.dxf.insert = (ch_cy, (az + z_deck_top) / 2.0)
        ch_label.dxf.attachment_point = 5; ch_label.dxf.rotation = 0

    # ---------------------- EXPORT ----------------------
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
        tank_len_m = float(self.tank_length_m) if self.tank_length_m is not None else None

        # Areas per member (plate projection) — HL 기준은 LNGC와 동일
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

        # ---- Compartments + volumes ----
        # Ballast/pipe/void = HL, Cargo = TL (없으면 HL fallback)
        comp_items = list(self._compartment_data)
        comp_vols = []
        group_sums = {
            "Void (STBD)": 0.0,
            "W.B.T (STBD)": 0.0,
            "Cargo tank (STBD)": 0.0,
            "Pipe duct (STBD)": 0.0,
        }

        def cname(meta):
            return clean_multiline_label(meta["raw_label"])

        for c in comp_items:
            A = float(c["area_m2"])
            nm = cname(c)
            low = nm.lower()

            # 길이 선택: Cargo는 TL 사용(없으면 HL 대체), 그 외는 HL
            if low.startswith("cargo"):
                L_used = tank_len_m if tank_len_m is not None else hold_len_m
            else:
                L_used = hold_len_m

            # 길이를 정할 수 없으면 스킵
            if L_used is None:
                continue

            vol_half = A * L_used
            vol_full = vol_half * 2.0

            comp_vols.append({
                "name": nm,
                "volume_m3_half": round(vol_half, 6),
                "volume_m3_full": round(vol_full, 6),
            })

            if low.startswith("ballast"):
                group_sums["W.B.T (STBD)"] += vol_half
            elif low.startswith("cargo"):
                group_sums["Cargo tank (STBD)"] += vol_half
            elif low.startswith("pipe"):
                group_sums["Pipe duct (STBD)"] += vol_half
            elif low.startswith("void"):
                group_sums["Void (STBD)"] += vol_half

        group_sums_full = {k.replace("(STBD)", "(FULL)"): v * 2.0 for k, v in group_sums.items()}

        # ---- Cargo capacity per hold (FULL) = Cargo 항목의 FULL 볼륨 합계 ----
        cargo_list_full = [v["volume_m3_full"] for v in comp_vols if v["name"].lower().startswith("cargo")]
        cargo_per_hold_full = float(sum(cargo_list_full)) if cargo_list_full else None

        total_cargo_full = None
        cargo_token_k = None
        if cargo_per_hold_full is not None and cargo_per_hold_full > 0.0:
            # FWD-most hold volume reduced by hold_vol_factor; remaining holds at full volume
            total_cargo_full = ((self.number_of_hold - 1) * cargo_per_hold_full) + (cargo_per_hold_full * self.hold_vol_factor)
            cargo_token_k = f"{int(round(total_cargo_full / 1000.0))}K"

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
                "section": "Midship transverse section (2D, y–z plane)",
            },
            "drawing_conventions": {
                "deck_camber": "Upper_Deck is cambered; z = D + camber - (camber/(B/2))*y",
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
                "units": {
                    'L_m': 'm', 'B_m': 'm', 'D_m': 'm', 'HL_m': 'm',
                    'camberUpper_m': 'm', 'doubleBottom_m': 'm', 'bilgeRadius_m': 'm',
                },
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
                'tank_length_m': tank_len_m,
                'hold_len_factor': self.hold_len_factor,
                'hold_vol_factor': self.hold_vol_factor,
                'number_of_hold': self.number_of_hold,
                'length_basis_note': 'Member areas use HL; cargo volumes: TL per hold; total cargo uses number_of_hold * per-hold FULL * hold_vol_factor'
            },

        'members': {'geometry': member_props, 'areas': member_areas},
            'compartments': {
                'items': self._compartment_data,
                'volumes': {
                    'items': comp_vols,                  # by hold length
                    'groups_half': {k: round(v,6) for k,v in group_sums.items()},
                    'groups_full': {k: round(v,6) for k,v in group_sums_full.items()},
                    # >>> cargo capacity uses tank length <<<
                    'cargo_per_hold_full_m3': round(cargo_per_hold_full,6) if cargo_per_hold_full is not None else None,
                    'cargo_total_full_m3': round(total_cargo_full,6) if total_cargo_full is not None else None,
                    'cargo_capacity_token': cargo_token_k,
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
                'scantling_table': [
                    {'member': r[0], 'plate_mm': r[1], 'stiffener': r[2]}
                    for r in _SCANTLING_TABLE[1:]
                ],
            }
        }
        return export_stats

    def export(self, save_as=None, png_out_dir=None, png_dpi=220):
        self.draw_layers()
        self.draw_centerline()
        self.draw_title_and_specs(title="ORDINARY SECTION (STBD)")
        self.draw_hull()
        self.draw_cargo()
        self.draw_compartments()
        self.draw_stiffeners()
        self.draw_scantling_table()

        qc = {'ok': True, 'label_overlaps': 0}

        if save_as:
            os.makedirs(os.path.dirname(save_as), exist_ok=True)
            self.doc.saveas(save_as)

        png_path=None
        if png_out_dir and _MAT_OK:
            os.makedirs(png_out_dir, exist_ok=True)
            base=os.path.splitext(os.path.basename(save_as or "lpgc"))[0]
            out=os.path.join(png_out_dir, base+".png")
            face=mcolors.to_rgba("white")
            fig=plt.figure(figsize=(12,12), dpi=png_dpi, facecolor=face)
            ax=fig.add_axes([0,0,1,1], facecolor=face)
            Frontend(RenderContext(self.doc), MatplotlibBackend(ax)).draw_layout(self.doc.modelspace())
            ax.set_aspect("equal"); ax.set_axis_off()
            fig.savefig(out, dpi=png_dpi, facecolor=face, bbox_inches="tight", pad_inches=0)
            plt.close(fig); png_path=out

        export_stats = self._build_export_stats(qc, png_path)

        try:
            export_stats['drawing']['files']['dxf'] = save_as
            export_stats['drawing']['files']['png'] = png_path
        except Exception:
            pass

        return qc, png_path, export_stats

# =========================================
#               Domain rules (original)
# =========================================
def domain_rules_ok(p):
    import math
    B=p['B']; D=p['D']; DB=p['DB']; R=p['R']; C=p['C']
    GY=p['GY']; OG=p['OG']; TSE=p['TSWT_EXT']
    GH=p['GAP_HOP']; GTS=p['GAP_TSWT']; SC=p['STRCLR']
    S2=p['S2']

    issues=[]
    if not (B>0 and D>0 and DB>0 and R>0): issues.append("PositiveDims")
    if C<0 or C>0.05*B or C>0.10*D: issues.append("Camber_limit")
    if R >= (D-DB) - 0.2: issues.append("BilgeR_vs_Depth")

    y_bilge_toe = (B/2) - R
    y_og = OG*(B/2)
    if not (0.6 <= GY <= (B/2) - R - 1.0): issues.append("Girder_y_out_of_range")
    if y_og > (y_bilge_toe - 0.8) or y_og < (GY + 0.6): issues.append("OutGirder_range_vs_bilge/girder")

    if not (110.0 <= TSE <= 130.0): issues.append("TSWT_ext_angle_120_140")
    if not (0.6 <= GTS <= 0.8): issues.append("gap_tswt_0p3_0p6")
    if not (0.6 <= GH  <= 0.8): issues.append("gap_hopper_0p3_0p6")
    if not (0.2 <= SC  <= 0.4): issues.append("str_clear_0p2_0p4")

    if (S2*D) <= DB + 0.5: issues.append("Str2_low_vs_DB")

    def z_deck_at(y): return -(C/(B/2.0))*y + (D + C)
    y_tsy = (B/2)/2.0
    z_top = z_deck_at(y_tsy)
    z_kink = z_top - 0.7
    phi = math.radians(90.0 - TSE)
    phi = max(min(phi, math.radians(-30.0)), math.radians(-50.0))
    z_ts_side = z_kink + math.tan(phi)*(B/2 - y_tsy)
    z1 = z_ts_side - 1.7
    if z1 < DB + 0.5: issues.append("Str1_below_DB_margin")
    if z_ts_side <= z1: issues.append("TSWT_not_above_Str1_by_1p7m")

    return len(issues)==0, issues

# =========================================
#            I/O helpers
# =========================================
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


# ================================
# KR Standard Info — KR Rules 2025 (Pt7 Ch5: LPG Carriers)
# ================================
KR_STANDARD_INFO = {
    "title": "Korean Register Rules for the Classification of Ships",
    "edition": "2025",
    "short_name": "KR Rules 2025",
    "effective_from": "2025-01-01",
    "source_file": "KR-Rules-2025.pdf",
}

KR_RULE_REGISTRY_LPGC = {
    "lpgc_scope":           {"rule_ref": "Pt7.Ch5.Sec1[1.1]",   "title": "KR scope — LPG carrier applicability",       "level": "scope"},
    "independent_tank":     {"rule_ref": "Pt7.Ch5.Sec2[1.1]",   "title": "Independent tank type C arrangement",         "level": "arrangement"},
    "double_bottom_height": {"rule_ref": "Pt7.Ch5.Sec3[2.1]",   "title": "Minimum double bottom height for LPG",        "level": "arrangement"},
    "tank_inboard_clearance": {"rule_ref": "IGC 2.4.1 (Type 2G)", "title": "Cargo tank inboard clearance from side shell (LPG single-side hull)", "level": "arrangement"},
    "cargo_tank_clearance": {"rule_ref": "Pt7.Ch5.Sec4[2.1]",   "title": "Clearance between cargo tank and hull",       "level": "arrangement"},
    "hopper_slope_angle":   {"rule_ref": "Pt7.Ch5.Sec3[4.1]",   "title": "Hopper plate slope angle requirement",        "level": "arrangement"},
    "tswt_arrangement":     {"rule_ref": "Pt7.Ch5.Sec3[5.1]",   "title": "Transverse swash bulkhead arrangement",       "level": "arrangement"},
    "longitudinal_framing": {"rule_ref": "Pt7.Ch5.Sec6[2.1]",   "title": "Longitudinal framing requirement",            "level": "arrangement"},
    "weld_joint_detail":    {"rule_ref": "Pt7.Ch5.Sec8.Sec3",   "title": "Weld joint detail requirements",              "level": "detail_design"},
}

def _lpgc_rule_meta(check_id):
    r = KR_RULE_REGISTRY_LPGC.get(check_id, {})
    return r.get("rule_ref", ""), r.get("title", check_id), r.get("level", "")

def make_kr_check_lpgc(check_id, status, *, inputs=None, actual=None, required=None, unit=None, notes=None):
    rule_ref, title, level = _lpgc_rule_meta(check_id)
    out = {"check_id": check_id, "rule_ref": rule_ref, "title": title, "level": level, "status": status}
    if inputs is not None:   out["inputs"] = inputs
    if actual is not None:   out["actual"] = actual
    if required is not None: out["required"] = required
    if unit is not None:     out["unit"] = unit
    if notes is not None:    out["notes"] = notes
    return out

def evaluate_kr_rules_lpgc(generator_inputs, ship):
    """KR Rules 2025 Pt7 Ch5 evaluation for LPG carrier. 4 states: pass/fail/undetermined/not_modeled"""
    checks = []
    assumptions = ["Framing system assumed longitudinal.", "KR Pt7 Ch5 2025 applied."]

    L_m  = float(generator_inputs.get("L_m", 0))
    B_m  = float(generator_inputs.get("B_m", ship.B))
    DB_m = float(generator_inputs.get("doubleBottom_m", ship.DB))

    checks.append(make_kr_check_lpgc("lpgc_scope", "pass" if L_m >= 80.0 else "fail",
        inputs={"L_m": round(L_m, 3)}, actual=round(L_m, 3), required={"min_m": 80.0}, unit="m",
        notes="KR Pt7 Ch5 applies to LPG carriers >= 80 m."))

    checks.append(make_kr_check_lpgc("independent_tank", "pass",
        notes="Independent type C cargo tanks assumed per model geometry."))

    # IGC 2.4.1 Type 2G: db >= max(0.76 m, B/15) — collision protection for Type C independent tanks
    required_db = max(0.76, B_m / 15.0)
    checks.append(make_kr_check_lpgc("double_bottom_height", "pass" if DB_m >= required_db - 1e-9 else "fail",
        inputs={"B_m": round(B_m, 3), "DB_m": round(DB_m, 3)},
        actual=round(DB_m, 3), required={"min_m": round(required_db, 4)}, unit="m",
        notes="IGC 2.4.1 Type 2G: cargo tank inboard distance from molded line of bottom shall be >= max(0.76 m, B/15)."))

    # IGC 2.4.1 Type 2G: cargo tank inboard distance from side shell >= max(0.76 m, B/15).
    # LPGC has a single-side hull, so this is enforced as tank-to-shell clearance, not double-side width.
    # Read the actual clearance from ship.tank_side_clearance (computed in LPGC.__init__).
    # If the attribute is missing (legacy stub/test object), fall back to B/2 - y_ts if
    # available, else mark undetermined.
    required_clear = max(0.76, B_m / 15.0)
    tank_clear = getattr(ship, "tank_side_clearance", None)
    if tank_clear is None and hasattr(ship, "y_ts"):
        try:
            tank_clear = float(B_m) / 2.0 - float(ship.y_ts)
        except Exception:
            tank_clear = None
    if tank_clear is None:
        checks.append(make_kr_check_lpgc("tank_inboard_clearance", "undetermined",
            inputs={"B_m": round(B_m, 3)},
            actual=None,
            required={"min_m": round(required_clear, 4)}, unit="m",
            notes="ship.tank_side_clearance not exposed; cannot evaluate IGC 2.4.1 Type 2G."))
    else:
        tc = float(tank_clear)
        checks.append(make_kr_check_lpgc("tank_inboard_clearance",
            "pass" if tc >= required_clear - 1e-9 else "fail",
            inputs={"B_m": round(B_m, 3), "tank_side_clearance_m": round(tc, 3)},
            actual=round(tc, 3),
            required={"min_m": round(required_clear, 4)}, unit="m",
            notes="LPGC is single-side hull (LPG is IGC Type 2G). IGC 2.4.1 requires cargo tank "
                  "inboard distance from side shell >= max(0.76 m, B/15). Actual is the generator's "
                  "tank_side_clearance = max(1.8, B/15 + 0.3)."))

    # Cargo tank clearance: gap_tswt
    gap_ts = float(generator_inputs.get("gap_tswt_m", 0.7))
    checks.append(make_kr_check_lpgc("cargo_tank_clearance", "pass" if gap_ts >= 0.6 else "fail",
        inputs={"gap_tswt_m": round(gap_ts, 3)}, actual=round(gap_ts, 3),
        required={"min_m": 0.6}, unit="m",
        notes="Gap between cargo tank TSWT and ship structure."))

    # Hopper slope: tswt_ext_deg (interior angle convention from inner bottom).
    # KR Pt7 Ch5 Sec3[4.1] requires the hopper plate angle to be adequate for cargo flow / sloshing
    # mitigation but does NOT mandate a specific numeric range. Typical design ~110-130 deg.
    # Record the value as informational; status undetermined because no explicit rule threshold exists.
    tswt_deg = float(generator_inputs.get("tswt_ext_deg", 120.0))
    _typical_ok = 110.0 <= tswt_deg <= 130.0
    checks.append(make_kr_check_lpgc("hopper_slope_angle", "undetermined",
        inputs={"tswt_ext_deg": round(tswt_deg, 2)},
        actual=round(tswt_deg, 2),
        required={"typical_design_min_deg": 110.0, "typical_design_max_deg": 130.0},
        unit="deg",
        notes=("KR Pt7 Ch5 Sec3[4.1] does not mandate a specific hopper slope angle range. "
               f"Recorded value is {'within' if _typical_ok else 'outside'} the typical design "
               "range (110-130 deg, interior angle from inner bottom). Engineering judgement required.")))

    # TSWT arrangement check
    segs = ship.seg_dict()
    tswt_present = "TSWT_V" in segs or "TSWT" in segs
    checks.append(make_kr_check_lpgc("tswt_arrangement", "pass" if tswt_present else "not_modeled",
        notes="TSWT members checked in segment dict." if tswt_present else "TSWT not found in model."))

    checks.append(make_kr_check_lpgc("longitudinal_framing", "pass",
        inputs={"framing_system": "longitudinal"}, notes="Longitudinal framing assumed."))
    checks.append(make_kr_check_lpgc("weld_joint_detail", "undetermined",
        notes="Weld joint geometry not included in parametric model."))

    def _isect(seg_a, seg_b):
        if seg_a is None or seg_b is None: return None
        (ay1,az1),(ay2,az2)=seg_a; (by1,bz1),(by2,bz2)=seg_b
        day,daz=ay2-ay1, az2-az1; dby,dbz=by2-by1, bz2-bz1
        denom=day*dbz-daz*dby
        if abs(denom)<1e-9: return None
        t=((by1-ay1)*dbz-(bz1-az1)*dby)/denom
        return (round(ay1+t*day,3), round(az1+t*daz,3))

    hotspots = []
    tank_hop = _isect(segs.get("Tank_Hopper"), segs.get("Hopper"))
    hotspots.append({"hotspot_id": "tank_hopper_knuckle", "rule_ref": "Pt7.Ch5.Sec3[4.1]",
        "title": "Tank-hopper knuckle connection", "availability": "modeled" if tank_hop else "not_modeled",
        "point_mm": tank_hop, "related_members": ["Tank_Hopper", "Hopper"],
        "kr_evaluation_status": "undetermined",
        "required_additional_inputs": ["plate_thickness_mm", "weld_detail"],
        "description": "Fatigue-sensitive tank hopper knuckle."})
    tank_tswt = _isect(segs.get("Tank_TSWT"), segs.get("TSWT_V"))
    hotspots.append({"hotspot_id": "tank_tswt_connection", "rule_ref": "Pt7.Ch5.Sec3[5.1]",
        "title": "Tank TSWT connection", "availability": "modeled" if tank_tswt else "not_modeled",
        "point_mm": tank_tswt, "related_members": ["Tank_TSWT", "TSWT_V"],
        "kr_evaluation_status": "undetermined",
        "required_additional_inputs": ["bracket_geometry"],
        "description": "Connection between cargo tank and transverse swash bulkhead."})

    counts = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
    for c in checks:
        s = c.get("status")
        if s in counts: counts[s] += 1
    overall = "fail" if counts["fail"] > 0 else (
        "partial" if counts["undetermined"] + counts["not_modeled"] > 0 else "pass")

    return {
        "standard": KR_STANDARD_INFO, "ship_type": "lpg_carrier",
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
        "Bottom_Shell":  "3D_OUTER_HULL",
        "Side_Shell":    "3D_OUTER_HULL",
        "Upper_Deck":    "3D_OUTER_HULL",
        "IBTM":          "3D_DB",
        "Hopper":        "3D_DS",
        "Out_Girder":    "3D_DB",
        "Girder":        "3D_DB",
        "TSWT_V":        "3D_DS",
        "TSWT":          "3D_DS",
        "Str1":          "3D_DS",
        "Str2":          "3D_DS",
        "Tank_Top":      "3D_CARGO_HOLD",
        "Tank_TSWT":     "3D_CARGO_HOLD",
        "Tank_Side":     "3D_CARGO_HOLD",
        "Tank_Hopper":   "3D_CARGO_HOLD",
        "Tank_Bottom":   "3D_CARGO_HOLD",
    }
    # Outer shell members (drawn in AFT/ER/FWD too)
    OUTER_MEMBERS = {"Bottom_Shell", "Side_Shell", "Upper_Deck"}

    EPSY = 1e-6
    B = ship.B; R = ship.R
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
        "3D_CARGO_HOLD":  ('#33aa77', 0.65, 0.5),   # cargo — mediumseagreen (LPGC)
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
# 데이터셋 생성 루프 (KR Pt7 Ch5 / Elev / 3D 포함)
# ===============================
def generate_lpgc_dataset(
    save_dir,
    method='lhs',
    fwd_hold_ratio=0.12,
    er_hold_ratio=0.23,
    aft_hold_ratio=0.16,
    hold_len_factor=1.0,
    hold_vol_factor=0.7,
    number_of_hold_range=(3, 5, 1),
    use_L_fixed=False,
    L_fixed=220.0,
    compart_out_dir=None,
    compart_png_out_dir=None,
    compart3d_out_dir=None,
    compart3d_png_out_dir=None,
    json_out_dir=None,
    hold_length_range=(35.0, 45.0, 0.1),
    B_range=(32, 42, 1),
    D_range=(18, 28, 1),
    camber_range=(0.5, 2.5, 0.5),
    db_range=(1.5, 3.5, 0.1),
    bilge_range=(2.0, 5.0, 0.5),
    girder_y=(1.6, 1.8, 0.05),
    outgir_ratio=(0.7, 0.8, 0.05),
    tswt_ext=(110.0, 130.0, 2.0),
    gap_tswt=(0.6, 0.8, 0.1),
    gap_hopper=(0.6, 0.8, 0.1),
    str_clear=(0.2, 0.4, 0.1),
    s1_ratio=(0.7, 0.8, 0.05),
    s2_ratio=(0.4, 0.6, 0.05),
    text_height=250, offset=300,
    MAX_FILES=100, PROGRESS_EVERY=20, SEED=42,
    png_out_dir=None, png_dpi=220,
):
    os.makedirs(save_dir, exist_ok=True)
    rng = random.Random(SEED)

    for d in [compart_out_dir, compart_png_out_dir, compart3d_out_dir, compart3d_png_out_dir, json_out_dir]:
        if d is not None:
            os.makedirs(d, exist_ok=True)

    specs = [
        {'name':'HL','min':hold_length_range[0],'max':hold_length_range[1],'type':'float','step':hold_length_range[2]},
        {'name':'B','min':B_range[0],'max':B_range[1],'type':'int','step':B_range[2]},
        {'name':'D','min':D_range[0],'max':D_range[1],'type':'int','step':D_range[2]},
        {'name':'C','min':camber_range[0],'max':camber_range[1],'type':'float','step':camber_range[2]},
        {'name':'DB','min':db_range[0],'max':db_range[1],'type':'float','step':db_range[2]},
        {'name':'R','min':bilge_range[0],'max':bilge_range[1],'type':'float','step':bilge_range[2]},
        {'name':'GY','min':girder_y[0],'max':girder_y[1],'type':'float','step':girder_y[2]},
        {'name':'OG','min':outgir_ratio[0],'max':outgir_ratio[1],'type':'float','step':outgir_ratio[2]},
        {'name':'TSWT_EXT','min':tswt_ext[0],'max':tswt_ext[1],'type':'float','step':tswt_ext[2]},
        {'name':'GAP_TSWT','min':gap_tswt[0],'max':gap_tswt[1],'type':'float','step':gap_tswt[2]},
        {'name':'GAP_HOP','min':gap_hopper[0],'max':gap_hopper[1],'type':'float','step':gap_hopper[2]},
        {'name':'STRCLR','min':str_clear[0],'max':str_clear[1],'type':'float','step':str_clear[2]},
        {'name':'S1','min':s1_ratio[0],'max':s1_ratio[1],'type':'float','step':s1_ratio[2]},
        {'name':'S2','min':s2_ratio[0],'max':s2_ratio[1],'type':'float','step':s2_ratio[2]},
    ]
    if method not in ('lhs', 'random', 'grid'):
        raise ValueError("method must be one of ['lhs','random','grid']")
    samples = lhs_samples(MAX_FILES, specs, seed=SEED)

    _index_dir = os.path.dirname(os.path.abspath(save_dir))
    index_csv = os.path.join(_index_dir, "LPGC_dataset_index.csv")
    header = [
        'file', 'json', 'method', 'seed',
        'Cargo Capacity (K)', 'Number of Hold',
        'Ship Length_m (L)', 'Ship Breadth_m (B)', 'Ship Depth_m (D)',
        'Hold Len. Factor', 'Hold Length_m (HL)', 'Tank Length_m (TL)', 'Hold Vol. Factor', 'Tank Volume_m3',
        'C.L. based Upper Deck Camber height_m (C)',
        'Double Bottom Height_m (DB)', 'Bilge Radius_m (R)',
        'Girder y_m', 'Out_Girder ratio to B/2 (OG)',
        'TSWT exterior angle_deg', 'Gap TSWT_m', 'Gap Hopper_m', 'Stringer clear_m',
        'Str1 ratio to D (S1)', 'Str2 ratio to D (S2)',
        'domain_ok', 'domain_issues',
        'kr_scope_status', 'kr_pass', 'kr_fail', 'kr_undetermined', 'kr_not_modeled',
        'qc_ok', 'label_overlaps', 'filesize', 'png', 'stiffeners_total', 'labels_count'
    ]

    def _est_L(HL, n_hold):
        hold_total = hold_len_factor * HL * n_hold
        return (fwd_hold_ratio + er_hold_ratio + aft_hold_ratio) * hold_total + hold_total

    saved = 0
    for p in samples:
        HL = float(p['HL'])
        TL = max(0.0, HL - 2.5)

        nh_min, nh_max, nh_step = number_of_hold_range
        number_of_hold = rng.randrange(nh_min, nh_max + 1, nh_step)

        hold_total = hold_len_factor * HL * number_of_hold
        fwd_len = fwd_hold_ratio * hold_total
        er_len  = er_hold_ratio  * hold_total
        aft_len = aft_hold_ratio * hold_total

        L_est = _est_L(HL, number_of_hold)
        p['L'] = L_fixed if use_L_fixed else L_est

        ok, issues = domain_rules_ok(p)
        if not ok:
            continue

        ship = LPGC(
            L=p['L'], B=p['B'], D=p['D'], DB=p['DB'], R=p['R'], camber=p['C'],
            y_girder=p['GY'], y_og_ratio=p['OG'], tswt_ext_deg=p['TSWT_EXT'],
            gap_tswt=p['GAP_TSWT'], gap_hopper=p['GAP_HOP'], str_clear=p['STRCLR'],
            s1_ratio=p['S1'], s2_ratio=p['S2']
        )

        _gen_inputs_for_kr = {
            'L_m': p['L'], 'B_m': p['B'], 'D_m': p['D'],
            'doubleBottom_m': p['DB'], 'bilgeRadius_m': p['R'],
            'tswt_ext_deg': p['TSWT_EXT'],
            'gap_tswt_m': p['GAP_TSWT'], 'gap_hopper_m': p['GAP_HOP'],
        }
        kr_eval = evaluate_kr_rules_lpgc(_gen_inputs_for_kr, ship)

        dxf_path = build_filename(
            save_dir, p['L'], p['B'], p['D'], p['C'], p['DB'], p['R'],
            p['GY'], p['OG'], p['TSWT_EXT'], p['GAP_TSWT'], p['GAP_HOP'], p['STRCLR'], p['S1'], p['S2']
        )

        exp = DXFExporter(
            ship, text_height=text_height, offset=offset,
            hold_length_m=HL, tank_length_m=TL,
            hold_len_factor=hold_len_factor, hold_vol_factor=hold_vol_factor,
            number_of_hold=number_of_hold
        )
        qc, png_path, stats = exp.export(save_as=dxf_path, png_out_dir=png_out_dir, png_dpi=png_dpi)

        capacity_token = stats.get('compartments', {}).get('volumes', {}).get('cargo_capacity_token')
        final_dxf_path = dxf_path
        if capacity_token:
            base = os.path.basename(dxf_path)
            hold_tag = f"{number_of_hold}Hold"
            if not base.startswith(capacity_token + "_"):
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
                L_m=p['L'], HL_m=HL, number_of_hold=number_of_hold,
                fwd_len_m=fwd_len, er_len_m=er_len, aft_len_m=aft_len,
                hold_len_factor=hold_len_factor,
            )

        if compart_out_dir is not None and layout is not None:
            compart_dxf_path = os.path.join(compart_out_dir, base_noext + "_Compart.dxf")
            compart_dxf_path, compart_png_path = create_compartment_arrangement_drawing(
                compart_dxf_path, layout=layout, D_m=p['D'], camber_m=p['C'], DB_m=p['DB'],
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

        fsize = os.path.getsize(final_dxf_path) if os.path.exists(final_dxf_path) else -1
        tank_vol_per_hold_full = stats.get('compartments', {}).get('volumes', {}).get('cargo_per_hold_full_m3')

        fwd_len_used = layout['fwd_len_m'] if layout else fwd_len
        er_len_used  = layout['er_len_m']  if layout else er_len
        aft_len_used = layout['aft_len_m'] if layout else aft_len

        meta = {
            'sample_id': f"LPGC-{saved+1:04d}",
            'ship_type': 'LPGC',
            'generated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': method, 'seed': SEED,
            'generator_inputs': {
                'L_m': p['L'], 'B_m': p['B'], 'D_m': p['D'],
                'HL_m': HL, 'TL_m': TL,
                'camberUpper_m': p['C'],
                'doubleBottom_m': p['DB'], 'bilgeRadius_m': p['R'],
                'girder_y_m': p['GY'], 'girderOut_ratio': p['OG'],
                'tswt_ext_deg': p['TSWT_EXT'],
                'gap_tswt_m': p['GAP_TSWT'], 'gap_hopper_m': p['GAP_HOP'],
                'strClearance_m': p['STRCLR'],
                'str1_ratio': p['S1'], 'str2_ratio': p['S2'],
                'number_of_hold': number_of_hold,
            },
            'geometry': {
                'derived': {
                    'girderOut_y_m': round(p['OG'] * (p['B'] / 2.0), 3),
                    'girder_y_m': round(p['GY'], 3),
                },
                'longitudinal_layout': layout,
                'length_model': {
                    'fwd_len_m': fwd_len_used, 'er_len_m': er_len_used, 'aft_len_m': aft_len_used,
                    'hold_len_factor': hold_len_factor, 'hold_vol_factor': hold_vol_factor,
                    'mode': 'fixed' if use_L_fixed else 'estimated',
                },
            },
            'standard_refs': {'kr_standard': KR_STANDARD_INFO},
            'rules': {**kr_eval, 'society': 'KR'},  # unified schema (Phase 0.2.B1)
            'kr': kr_eval,  # legacy alias — kept for backward compat
            'multi_society_checks': multi_society_checks,
            'domain': {'ok': ok, 'issues': issues},
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
        write_json(json_path, meta)

        row = {
            'file': os.path.basename(final_dxf_path), 'json': os.path.basename(json_path),
            'method': method, 'seed': SEED,
            'Cargo Capacity (K)': capacity_token or "", 'Number of Hold': number_of_hold,
            'Ship Length_m (L)': p['L'], 'Ship Breadth_m (B)': p['B'], 'Ship Depth_m (D)': p['D'],
            'Hold Len. Factor': hold_len_factor, 'Hold Length_m (HL)': HL,
            'Tank Length_m (TL)': TL, 'Hold Vol. Factor': hold_vol_factor,
            'Tank Volume_m3': tank_vol_per_hold_full,
            'C.L. based Upper Deck Camber height_m (C)': p['C'],
            'Double Bottom Height_m (DB)': p['DB'], 'Bilge Radius_m (R)': p['R'],
            'Girder y_m': p['GY'], 'Out_Girder ratio to B/2 (OG)': p['OG'],
            'TSWT exterior angle_deg': p['TSWT_EXT'],
            'Gap TSWT_m': p['GAP_TSWT'], 'Gap Hopper_m': p['GAP_HOP'], 'Stringer clear_m': p['STRCLR'],
            'Str1 ratio to D (S1)': p['S1'], 'Str2 ratio to D (S2)': p['S2'],
            'domain_ok': ok, 'domain_issues': "|".join(issues),
            'kr_scope_status': next((c.get('status') for c in kr_eval.get('auto_checks', []) if c.get('check_id') == 'lpgc_scope'), ""),
            'kr_pass': kr_eval.get('summary', {}).get('check_counts', {}).get('pass', 0),
            'kr_fail': kr_eval.get('summary', {}).get('check_counts', {}).get('fail', 0),
            'kr_undetermined': kr_eval.get('summary', {}).get('check_counts', {}).get('undetermined', 0),
            'kr_not_modeled': kr_eval.get('summary', {}).get('check_counts', {}).get('not_modeled', 0),
            'qc_ok': qc.get('ok', True), 'label_overlaps': qc.get('label_overlaps', 0),
            'filesize': fsize, 'png': os.path.basename(png_path) if png_path else "",
            'stiffeners_total': stats.get('drawing', {}).get('stiffeners', {}).get('total', -1) if isinstance(stats, dict) else -1,
            'labels_count': stats.get('drawing', {}).get('labels', {}).get('count', -1) if isinstance(stats, dict) else -1,
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
    _BASE = "<SHIPBENCH_ROOT>/data/processed/LPGC"

    SAVE_DIR        = os.path.join(_BASE, "section_dxf")
    PNG_DIR         = os.path.join(_BASE, "section_png")
    COMPART_DIR        = os.path.join(_BASE, "compart_dxf")
    COMPART_PNG_DIR    = os.path.join(_BASE, "compart_png")
    COMPART3D_DIR     = os.path.join(_BASE, "compart3d_dxf")
    COMPART3D_PNG_DIR = os.path.join(_BASE, "compart3d_png")
    JSON_DIR        = os.path.join(_BASE, "json")

    generate_lpgc_dataset(
        save_dir=SAVE_DIR,
        json_out_dir=JSON_DIR,
        method='lhs',
        fwd_hold_ratio=0.12, er_hold_ratio=0.23, aft_hold_ratio=0.16,
        hold_len_factor=1.0, hold_vol_factor=0.7,
        number_of_hold_range=(3, 5, 1),
        use_L_fixed=False,
        compart_out_dir=COMPART_DIR, compart_png_out_dir=COMPART_PNG_DIR,
        compart3d_out_dir=COMPART3D_DIR, compart3d_png_out_dir=COMPART3D_PNG_DIR,
        hold_length_range=(35.0, 45.0, 0.1),
        B_range=(32, 42, 1), D_range=(18, 28, 1),
        camber_range=(0.5, 2.5, 0.5), db_range=(1.5, 3.5, 0.1),
        bilge_range=(2.0, 5.0, 0.5), girder_y=(1.6, 1.8, 0.05),
        outgir_ratio=(0.7, 0.8, 0.05), tswt_ext=(110.0, 130.0, 2.0),
        gap_tswt=(0.6, 0.8, 0.1), gap_hopper=(0.6, 0.8, 0.1),
        str_clear=(0.2, 0.4, 0.1), s1_ratio=(0.7, 0.8, 0.05), s2_ratio=(0.4, 0.6, 0.05),
        text_height=250, offset=300, MAX_FILES=100, PROGRESS_EVERY=20,
        SEED=42, png_out_dir=PNG_DIR, png_dpi=220
    )
