#!/usr/bin/env python3
"""Phase 0.2.B1 sanity check — dry-run the rule evaluator of every ship type.

Rather than spin up the full generator pipeline (which drags in matplotlib,
ezdxf, and the 3D renderer), this script imports each ``*_Data_generation.py``
module via ``importlib`` and calls the ``evaluate_*_rules_*()`` function with a
handful of representative synthetic inputs and a lightweight FakeShip stub.

For each ship type we run three cases:
  (a) a nominal "pass-biased" sample (principal particulars inside realistic
      ranges, rule-compliant arrangement),
  (b) a deliberate "fail-biased" sample (one parameter below rule minimum),
  (c) an off-scope sample (length below the CSR/KR scope threshold).

After the B1 fixes, we expect:
  * Tanker / VLCC: ``double_side_width`` moves from 100% undetermined to
    pass/fail as soon as DWT_t is estimated from L·B·D.
  * LNGC: ``double_bottom_height`` and ``double_side_width`` use IGC 2.4.1 Type
    1G formulas (``max(2.0, B/15)`` and ``min(B/5, 11.5)``), and the nominal
    sample at DS=9.5 m passes; ``inner_hull_slope`` is populated, not
    undetermined; ``gas_freeing`` is pass when the trunk deck is modeled.
  * LPGC: ``tank_inboard_clearance`` replaces ``double_side_width``;
    ``double_bottom_height`` uses B/15 (IGC 2.4.1 Type 2G); ``hopper_slope_angle``
    is undetermined (no explicit rule threshold exists).
  * CNTR: ``hatch_opening_ratio`` uses b_hatch/B with a realistic cap of 0.92;
    ``double_side_width`` is removed; ``double_bottom_height`` uses an absolute
    1.5 m floor; ``torsional_stiffness`` is always ``undetermined`` (KR Pt14
    Ch4 Sec1 requires detailed FE warping analysis for container ships).
  * LPGC ``tank_inboard_clearance`` is read from ``ship.tank_side_clearance``
    (derived from ``max(1.8, B/15 + 0.3)`` in the LPGC generator).
  * BULKC: the generator now rejects L<150 m pre-evaluation, but the evaluator
    still handles a below-scope sample cleanly.

Run from the repo root:

    python scripts/01_rule_sanity.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = REPO_ROOT / "data" / "data_generator"


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _load_generator_module(stem: str):
    path = GEN_DIR / f"{stem}_Data_generation.py"
    spec = importlib.util.spec_from_file_location(stem.lower(), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem.lower()] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# FakeShip stubs — provide just enough surface area for each evaluator.
# ---------------------------------------------------------------------------

class _FakeShipBase:
    """Shared attributes used by both CSR and KR evaluators."""

    def __init__(self, *, L, B, D, DB, DS):
        self.L = L
        self.B = B
        self.D = D
        self.DB = DB  # KR evaluators read .DB
        self.d_db = DB  # tanker/vlcc evaluators read .d_db
        self.d_ds = DS  # tanker/vlcc/cntr read .d_ds
        self.R = 2.0  # bilge radius (BULKC)
        self.tswt_ext = 45.0  # BULKC hopper knuckle angle default
        self.members = {
            "Upper_Deck": True,
            "Side_Shell": True,
            "IHull": True,
            "Bottom_Shell": True,
            "IBTM": True,
            "Str1": True,
            "Str2": True,
            "Str3": True,
        }

    def seg_dict(self):
        return {
            "Upper_Deck": ((0, 0), (self.B * 500, 0)),
            "Side_Shell": ((self.B * 500, 0), (self.B * 500, self.D * 1000)),
            "Bottom_Shell": ((0, 0), (self.B * 500, 0)),
            "IBTM": ((0, self.DB * 1000), (self.B * 500, self.DB * 1000)),
            "IHull": ((self.B * 500 - self.d_ds * 1000, self.DB * 1000),
                      (self.B * 500 - self.d_ds * 1000, self.D * 1000)),
            "Hopper": ((0, self.DB * 1000), (self.B * 500, self.DB * 1000)),
            "Str1": ((self.B * 500 - self.d_ds * 1000, self.D * 500),
                    (self.B * 500, self.D * 500)),
            "Str2": ((self.B * 500 - self.d_ds * 1000, self.D * 400),
                    (self.B * 500, self.D * 400)),
            "Str3": ((self.B * 500 - self.d_ds * 1000, self.D * 300),
                    (self.B * 500, self.D * 300)),
        }


class FakeCNTR(_FakeShipBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.y_ihull = self.B / 2 - self.d_ds

    def seg_dict(self):
        s = super().seg_dict()
        s["Hatch_Coaming"] = ((self.y_ihull * 1000, 0), (self.y_ihull * 1000, 2000))
        s["Bench_Girder"] = ((0, self.DB * 1000), (self.y_ihull * 1000, self.DB * 1000))
        return s


class FakeLNGC(_FakeShipBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.inner_slope_deg = 45.0
        self.z_trunk = self.D + 1.0
        self.z_flat = self.D

    def seg_dict(self):
        s = super().seg_dict()
        s["Trunk_Deck"] = ((0, self.D * 1000 + 1000), (self.B * 300, self.D * 1000 + 1000))
        s["TrunkDeck_Slant"] = ((self.B * 300, self.D * 1000 + 1000),
                                 (self.B * 500, self.D * 1000))
        return s


class FakeLPGC(_FakeShipBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        # Mirror LPGC.__init__: tank_side_clearance = max(1.8, B/15 + 0.3)
        self.tank_side_clearance = max(1.8, self.B / 15.0 + 0.3)
        self.y_ts = (self.B / 2) - self.tank_side_clearance

    def seg_dict(self):
        s = super().seg_dict()
        s["TSWT_V"] = ((self.B * 400, self.D * 800), (self.B * 400, self.D * 1000))
        s["Tank_Hopper"] = ((self.B * 200, self.DB * 1000), (self.B * 300, self.D * 600))
        s["Tank_TSWT"] = ((self.B * 350, self.D * 800), (self.B * 400, self.D * 900))
        return s


class FakeTanker(_FakeShipBase):
    pass


class FakeVLCC(_FakeShipBase):
    """VLCC evaluator reads ``ship.segments()`` instead of ``ship.seg_dict()``."""

    def segments(self):
        return self.seg_dict()


class FakeBULKC(_FakeShipBase):
    def seg_dict(self):
        s = super().seg_dict()
        s["TSWT_V"] = ((self.B * 400, self.D * 800), (self.B * 400, self.D * 1000))
        s["TSWT"] = ((self.B * 300, self.D * 800), (self.B * 400, self.D * 800))
        return s


# ---------------------------------------------------------------------------
# Ship-type harnesses — each returns a list of (case_name, summary, status_by_id)
# tuples so the script can print a compact table.
# ---------------------------------------------------------------------------

def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    counts = result.get("summary", {}).get("check_counts", {})
    overall = result.get("summary", {}).get("overall_arrangement_status", "?")
    by_id = {c["check_id"]: c.get("status") for c in result.get("auto_checks", [])}
    return {"overall": overall, "counts": counts, "by_id": by_id}


def run_cntr() -> list[tuple[str, dict[str, Any]]]:
    mod = _load_generator_module("CNTR")
    cases: list[tuple[str, dict[str, Any]]] = []

    # (a) nominal ULCS: B=48 DS=2.2 DB=2.0 L=300 → b_hatch/B≈0.908
    ship = FakeCNTR(L=300, B=48, D=28, DB=2.0, DS=2.2)
    gi = {"L_m": 300.0, "B_m": 48.0, "doubleBottom_m": 2.0, "doubleSide_m": 2.2}
    cases.append(("nominal-ulcs", _summarise(mod.evaluate_kr_rules_cntr(gi, ship))))

    # (b) too-small DB → should fail double_bottom_height
    ship = FakeCNTR(L=300, B=48, D=28, DB=1.0, DS=2.2)
    gi = {"L_m": 300.0, "B_m": 48.0, "doubleBottom_m": 1.0, "doubleSide_m": 2.2}
    cases.append(("fail-db", _summarise(mod.evaluate_kr_rules_cntr(gi, ship))))

    # (c) narrow-opening (DS=8) → pass hatch_opening_ratio AND pass torsional
    ship = FakeCNTR(L=300, B=48, D=28, DB=2.0, DS=8.0)
    gi = {"L_m": 300.0, "B_m": 48.0, "doubleBottom_m": 2.0, "doubleSide_m": 8.0}
    cases.append(("narrow-hatch", _summarise(mod.evaluate_kr_rules_cntr(gi, ship))))
    return cases


def run_lngc() -> list[tuple[str, dict[str, Any]]]:
    mod = _load_generator_module("LNGC")
    cases: list[tuple[str, dict[str, Any]]] = []

    # LNG is Type 2G per IGC Ch.19 (not Type 1G). IGC 2.4.1 for Type 2G:
    #   db, ds >= max(0.76, B/15).  For B=46 → required = 3.067 m.
    # (a) nominal 174K membrane LNGC: B=46 DS=3.2 DB=3.2 L=300 → all pass
    ship = FakeLNGC(L=300, B=46, D=26, DB=3.2, DS=3.2)
    gi = {"L_m": 300.0, "B_m": 46.0, "doubleBottom_m": 3.2, "doubleSide_m": 3.2,
          "number_of_cofferdam": 3}
    cases.append(("nominal-type2g", _summarise(mod.evaluate_kr_rules_lngc(gi, ship))))

    # (b) narrow DS and shallow DB → should fail both
    ship = FakeLNGC(L=300, B=46, D=26, DB=1.5, DS=1.5)
    gi = {"L_m": 300.0, "B_m": 46.0, "doubleBottom_m": 1.5, "doubleSide_m": 1.5,
          "number_of_cofferdam": 3}
    cases.append(("fail-thin-hull", _summarise(mod.evaluate_kr_rules_lngc(gi, ship))))

    # (c) below scope (L<150)
    ship = FakeLNGC(L=130, B=46, D=26, DB=3.2, DS=3.2)
    gi = {"L_m": 130.0, "B_m": 46.0, "doubleBottom_m": 3.2, "doubleSide_m": 3.2,
          "number_of_cofferdam": 3}
    cases.append(("below-scope", _summarise(mod.evaluate_kr_rules_lngc(gi, ship))))
    return cases


def run_lpgc() -> list[tuple[str, dict[str, Any]]]:
    mod = _load_generator_module("LPGC")
    cases: list[tuple[str, dict[str, Any]]] = []

    # (a) nominal single-side: B=32 DB=2.5 gap_hopper=2.5 (>= B/15 = 2.133)
    ship = FakeLPGC(L=200, B=32, D=20, DB=2.5, DS=2.5)
    gi = {"L_m": 200.0, "B_m": 32.0, "doubleBottom_m": 2.5,
          "gap_hopper_m": 2.5, "gap_tswt_m": 0.7, "tswt_ext_deg": 120.0}
    cases.append(("nominal-type2g", _summarise(mod.evaluate_kr_rules_lpgc(gi, ship))))

    # (b) under-cleared tank → fail tank_inboard_clearance + double_bottom.
    # Simulate a legacy / non-compliant geometry by forcing tank_side_clearance
    # below the IGC 2.4.1 Type 2G floor (max(0.76, B/15)).
    ship = FakeLPGC(L=200, B=32, D=20, DB=1.5, DS=1.0)
    ship.tank_side_clearance = 1.5  # below B/15 = 2.133
    ship.y_ts = (ship.B / 2) - ship.tank_side_clearance
    gi = {"L_m": 200.0, "B_m": 32.0, "doubleBottom_m": 1.5,
          "gap_hopper_m": 1.0, "gap_tswt_m": 0.7, "tswt_ext_deg": 120.0}
    cases.append(("fail-clearance", _summarise(mod.evaluate_kr_rules_lpgc(gi, ship))))

    # (c) below scope (L<80)
    ship = FakeLPGC(L=70, B=32, D=20, DB=2.5, DS=2.5)
    gi = {"L_m": 70.0, "B_m": 32.0, "doubleBottom_m": 2.5,
          "gap_hopper_m": 2.5, "gap_tswt_m": 0.7, "tswt_ext_deg": 120.0}
    cases.append(("below-scope", _summarise(mod.evaluate_kr_rules_lpgc(gi, ship))))
    return cases


def _run_csr_ship(stem: str, ShipCls, cb_estimate: float,
                  nominal: tuple[float, float, float, float, float],
                  fail: tuple[float, float, float, float, float]):
    mod = _load_generator_module(stem)
    cases: list[tuple[str, dict[str, Any]]] = []

    def _one(name, L, B, D, DB, DS):
        ship = ShipCls(L=L, B=B, D=D, DB=DB, DS=DS)
        gi = {"L_m": L, "B_m": B, "D_m": D,
              "doubleBottom_m": DB, "doubleSide_m": DS}
        if stem in ("Tanker", "VLCC"):
            ship_data = mod.build_ship_data_context(
                L, ship_data_defaults=None, B_m=B, D_m=D, cb_estimate=cb_estimate)
        else:
            ship_data = mod.build_ship_data_context(L, ship_data_defaults=None)
        eval_fn_name = f"evaluate_csr_rules_{stem.lower()}"
        if stem == "Tanker":
            eval_fn_name = "evaluate_csr_rules_tanker"
        elif stem == "VLCC":
            eval_fn_name = "evaluate_csr_rules_vlcc"
        elif stem == "BULKC":
            eval_fn_name = "evaluate_csr_rules_bulkc"
        result = getattr(mod, eval_fn_name)(gi, ship_data, ship)
        summary = _summarise(result)
        summary["DWT_t"] = ship_data.get("DWT_t")
        summary["DWT_basis"] = ship_data.get("DWT_basis")
        return (name, summary)

    cases.append(_one("nominal", *nominal))
    cases.append(_one("fail-thin", *fail))
    L0, B0, D0, DB0, DS0 = nominal
    cases.append(_one("below-scope", max(L0 - 100, 80.0), B0, D0, DB0, DS0))
    return cases


def run_tanker() -> list[tuple[str, dict[str, Any]]]:
    # L=210 B=44 D=22 → Aframax-ish, expect DWT~104k, required_ds saturates at 2.0
    return _run_csr_ship("Tanker", FakeTanker, cb_estimate=0.82,
                         nominal=(210, 44, 22, 2.0, 2.5),
                         fail=(210, 44, 22, 0.5, 1.0))


def run_vlcc() -> list[tuple[str, dict[str, Any]]]:
    # L=330 B=60 D=30 → VLCC-class
    return _run_csr_ship("VLCC", FakeVLCC, cb_estimate=0.83,
                         nominal=(330, 60, 30, 2.2, 2.5),
                         fail=(330, 60, 30, 0.5, 1.0))


def run_bulkc() -> list[tuple[str, dict[str, Any]]]:
    return _run_csr_ship("BULKC", FakeBULKC, cb_estimate=0.80,
                         nominal=(230, 44, 22, 2.5, 2.5),
                         fail=(230, 44, 22, 1.0, 1.0))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SHIPS: list[tuple[str, Callable[[], list[tuple[str, dict[str, Any]]]]]] = [
    ("CNTR", run_cntr),
    ("LNGC", run_lngc),
    ("LPGC", run_lpgc),
    ("Tanker", run_tanker),
    ("VLCC", run_vlcc),
    ("BULKC", run_bulkc),
]


def _print_row(label: str, summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    overall = summary["overall"]
    extras = []
    if summary.get("DWT_t") is not None:
        extras.append(f"DWT={summary['DWT_t']:.0f}t ({summary.get('DWT_basis')})")
    print(f"  {label:14s} overall={overall:8s} "
          f"pass={counts.get('pass', 0):2d} "
          f"fail={counts.get('fail', 0):2d} "
          f"undet={counts.get('undetermined', 0):2d} "
          f"notmod={counts.get('not_modeled', 0):2d}"
          + (f"  [{', '.join(extras)}]" if extras else ""))


def _check_expectations(ship: str, cases: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Return a list of assertion failures for the expected post-fix behaviour."""
    failures: list[str] = []

    def _get(case, status_key):
        for name, summary in cases:
            if name == case:
                return summary["by_id"].get(status_key)
        return None

    if ship == "CNTR":
        if _get("nominal-ulcs", "double_side_width") is not None:
            failures.append("CNTR double_side_width should be removed from registry")
        if _get("nominal-ulcs", "double_bottom_height") != "pass":
            failures.append("CNTR nominal double_bottom_height expected pass")
        if _get("fail-db", "double_bottom_height") != "fail":
            failures.append("CNTR fail-db double_bottom_height expected fail")
        # torsional_stiffness is always undetermined (KR Pt14 Ch4 Sec1 requires FE analysis)
        if _get("narrow-hatch", "torsional_stiffness") != "undetermined":
            failures.append("CNTR narrow-hatch torsional_stiffness expected undetermined")
        if _get("nominal-ulcs", "torsional_stiffness") != "undetermined":
            failures.append("CNTR nominal-ulcs torsional_stiffness expected undetermined")

    if ship == "LNGC":
        # LNG is Type 2G per IGC Ch.19. For B=46, required db/ds = max(0.76, B/15) = 3.067 m.
        if _get("nominal-type2g", "double_side_width") != "pass":
            failures.append("LNGC nominal-type2g double_side_width expected pass (DS=3.2 vs B/15=3.067)")
        if _get("nominal-type2g", "double_bottom_height") != "pass":
            failures.append("LNGC nominal-type2g double_bottom_height expected pass (DB=3.2 vs B/15=3.067)")
        if _get("nominal-type2g", "inner_hull_slope") != "pass":
            failures.append("LNGC nominal-type2g inner_hull_slope expected pass")
        if _get("nominal-type2g", "gas_freeing") != "pass":
            failures.append("LNGC nominal-type2g gas_freeing expected pass when trunk present")
        if _get("fail-thin-hull", "double_side_width") != "fail":
            failures.append("LNGC fail-thin-hull double_side_width expected fail (DS=1.5 < 3.067)")
        if _get("fail-thin-hull", "double_bottom_height") != "fail":
            failures.append("LNGC fail-thin-hull double_bottom_height expected fail (DB=1.5 < 3.067)")
        if _get("below-scope", "lngc_scope") != "fail":
            failures.append("LNGC below-scope lngc_scope expected fail")

    if ship == "LPGC":
        # tank_inboard_clearance replaces double_side_width
        if _get("nominal-type2g", "double_side_width") is not None:
            failures.append("LPGC double_side_width should be renamed to tank_inboard_clearance")
        if _get("nominal-type2g", "tank_inboard_clearance") != "pass":
            failures.append("LPGC nominal tank_inboard_clearance expected pass (gap=2.5 >= B/15)")
        if _get("nominal-type2g", "hopper_slope_angle") != "undetermined":
            failures.append("LPGC nominal hopper_slope_angle expected undetermined (no hard rule)")
        if _get("fail-clearance", "tank_inboard_clearance") != "fail":
            failures.append("LPGC fail-clearance tank_inboard_clearance expected fail")

    if ship in ("Tanker", "VLCC"):
        if _get("nominal", "double_side_width") not in ("pass", "fail"):
            failures.append(f"{ship} nominal double_side_width should be pass/fail after DWT fix, "
                            f"not {_get('nominal','double_side_width')}")
        nominal_summary = next((s for n, s in cases if n == "nominal"), None)
        if nominal_summary and nominal_summary.get("DWT_basis") != "estimated_from_LBD":
            failures.append(f"{ship} DWT should be estimated_from_LBD (got {nominal_summary.get('DWT_basis')})")

    if ship == "BULKC":
        if _get("nominal", "bulk_carrier_scope") != "pass":
            failures.append("BULKC nominal bulk_carrier_scope expected pass")
        if _get("below-scope", "bulk_carrier_scope") != "fail":
            failures.append("BULKC below-scope bulk_carrier_scope expected fail")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Optional path to dump full results as JSON.")
    args = parser.parse_args()

    all_results: dict[str, Any] = {}
    all_failures: list[str] = []

    for name, runner in SHIPS:
        print(f"=== {name} ===")
        try:
            cases = runner()
        except Exception as exc:  # pragma: no cover — report and continue
            print(f"  ERROR during {name}: {type(exc).__name__}: {exc}")
            all_failures.append(f"{name}: runner raised {type(exc).__name__}: {exc}")
            continue

        for case_name, summary in cases:
            _print_row(case_name, summary)

        failures = _check_expectations(name, cases)
        for f in failures:
            print(f"  ! {f}")
        all_failures.extend(failures)
        all_results[name] = [{"case": n, **s} for n, s in cases]
        print()

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"Wrote full results to {out}")

    if all_failures:
        print(f"FAIL: {len(all_failures)} expectation(s) did not hold")
        return 1
    print("PASS: all rule-evaluator expectations satisfied after Phase 0.2.B1 fixes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
