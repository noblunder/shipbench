#!/usr/bin/env python3
"""Re-run the fixed rule evaluators against existing saved samples.

Phase 0.2.B1 fixes only change the *evaluator* functions, not the geometry
generator, so we can take the existing sample JSONs (each of which stores
``generator_inputs`` with the full parameter dictionary that was originally
fed to the evaluator) and call the *current* evaluator offline. This produces
a new rule-state distribution that reflects the fixes without having to
regenerate DXF/PNG artifacts.

Ship geometry is approximated by lightweight ``FakeShip*`` stubs that provide
just enough surface area (``B``, ``d_db``, ``d_ds``, ``seg_dict``/``segments``,
``members``, etc.) for each evaluator to run. This means the ``detail_hotspots``
intersection checks will be ``undetermined`` for most stubs — but we are only
interested in the ``auto_checks`` pass/fail/undetermined/not_modeled counts,
which depend on ``generator_inputs`` arithmetic rather than on segment
intersections.

Outputs:
  outputs/audit/rule_state_rev2.json  — per-check counts, per ship type
  stdout  — a compact comparison table vs the previous ``rule_state.json``

Usage::

    python scripts/02_rule_reeval.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
GEN_DIR = REPO_ROOT / "data" / "data_generator"
SAMPLES_ROOT = REPO_ROOT / "data" / "processed"
OUT_PATH = REPO_ROOT / "outputs" / "audit" / "rule_state_rev2.json"
PREV_PATH = REPO_ROOT / "outputs" / "audit" / "rule_state.json"


def _load_generator_module(stem: str):
    path = GEN_DIR / f"{stem}_Data_generation.py"
    spec = importlib.util.spec_from_file_location(stem.lower(), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem.lower()] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Lightweight ship stubs
# ---------------------------------------------------------------------------

class _FakeShip:
    """Shared stub with attributes read by both CSR and KR evaluators."""

    def __init__(self, gi: dict):
        self.L = float(gi.get("L_m", 0))
        self.B = float(gi.get("B_m", 0))
        self.D = float(gi.get("D_m", 0))
        self.DB = float(gi.get("doubleBottom_m", 0))
        self.d_db = self.DB
        self.d_ds = float(gi.get("doubleSide_m", 0))
        self.R = float(gi.get("bilgeRadius_m", 2.0))
        self.tswt_ext = float(gi.get("tswt_ext_deg", 45.0))
        self.members = {
            "Upper_Deck": True,
            "Side_Shell": True,
            "IHull": True,
            "Bottom_Shell": True,
            "IBTM": True,
            "Str1": True, "Str2": True, "Str3": True,
        }
        self.y_ihull = max(0.0, self.B / 2 - self.d_ds)

    def _base_seg(self) -> dict:
        B_mm_half = self.B * 500.0
        D_mm = self.D * 1000.0
        DB_mm = self.DB * 1000.0
        y_inner_mm = self.y_ihull * 1000.0
        return {
            "Upper_Deck": ((0.0, D_mm), (B_mm_half, D_mm)),
            "Side_Shell": ((B_mm_half, 0.0), (B_mm_half, D_mm)),
            "Bottom_Shell": ((0.0, 0.0), (B_mm_half, 0.0)),
            "IBTM": ((0.0, DB_mm), (B_mm_half, DB_mm)),
            "IHull": ((y_inner_mm, DB_mm), (y_inner_mm, D_mm)),
            "Hopper": ((y_inner_mm, DB_mm), (B_mm_half, 0.0)),
            "Str1": ((y_inner_mm, D_mm * 0.7), (B_mm_half, D_mm * 0.7)),
            "Str2": ((y_inner_mm, D_mm * 0.5), (B_mm_half, D_mm * 0.5)),
            "Str3": ((y_inner_mm, D_mm * 0.3), (B_mm_half, D_mm * 0.3)),
        }

    def seg_dict(self) -> dict:
        return self._base_seg()


class FakeCNTR(_FakeShip):
    def seg_dict(self) -> dict:
        s = self._base_seg()
        s["Hatch_Coaming"] = ((self.y_ihull * 1000.0, self.D * 1000.0),
                               (self.y_ihull * 1000.0, self.D * 1000.0 + 2000.0))
        s["Bench_Girder"] = ((0.0, self.DB * 1000.0),
                              (self.y_ihull * 1000.0, self.DB * 1000.0))
        return s


class FakeLNGC(_FakeShip):
    def __init__(self, gi: dict):
        super().__init__(gi)
        # Reasonable defaults — the parametric LNGC geometry always has a trunk
        # deck and a 45° inner hull slope, so the re-audit mirrors that.
        self.inner_slope_deg = 45.0
        self.z_trunk = self.D + 1.0
        self.z_flat = self.D

    def seg_dict(self) -> dict:
        s = self._base_seg()
        s["Trunk_Deck"] = ((0.0, (self.D + 1.0) * 1000.0),
                            (self.B * 300.0, (self.D + 1.0) * 1000.0))
        s["TrunkDeck_Slant"] = ((self.B * 300.0, (self.D + 1.0) * 1000.0),
                                 (self.B * 500.0, self.D * 1000.0))
        return s


class FakeLPGC(_FakeShip):
    def __init__(self, gi: dict):
        super().__init__(gi)
        # Mirror LPGC.__init__: IGC 2.4.1 Type 2G tank inboard clearance.
        self.tank_side_clearance = max(1.8, self.B / 15.0 + 0.3)
        self.y_ts = (self.B / 2.0) - self.tank_side_clearance

    def seg_dict(self) -> dict:
        s = self._base_seg()
        s["TSWT_V"] = ((self.B * 400.0, self.D * 800.0),
                        (self.B * 400.0, self.D * 1000.0))
        s["Tank_Hopper"] = ((self.B * 200.0, self.DB * 1000.0),
                             (self.B * 300.0, self.D * 600.0))
        s["Tank_TSWT"] = ((self.B * 350.0, self.D * 800.0),
                           (self.B * 400.0, self.D * 900.0))
        return s


class FakeTanker(_FakeShip):
    pass


class FakeVLCC(_FakeShip):
    def segments(self) -> dict:  # VLCC evaluator calls ship.segments()
        return self._base_seg()


class FakeBULKC(_FakeShip):
    def seg_dict(self) -> dict:
        s = self._base_seg()
        s["TSWT_V"] = ((self.B * 400.0, self.D * 800.0),
                        (self.B * 400.0, self.D * 1000.0))
        s["TSWT"] = ((self.B * 300.0, self.D * 800.0),
                      (self.B * 400.0, self.D * 800.0))
        return s


# ---------------------------------------------------------------------------
# Per-ship evaluation driver
# ---------------------------------------------------------------------------

def _reeval_csr(stem: str, ShipCls, cb_estimate: float,
                gi: dict) -> dict:
    mod = _load_generator_module(stem)
    ship = ShipCls(gi)
    if stem in ("Tanker", "VLCC"):
        ship_data = mod.build_ship_data_context(
            ship.L, ship_data_defaults=None,
            B_m=ship.B, D_m=ship.D, cb_estimate=cb_estimate)
    else:
        ship_data = mod.build_ship_data_context(ship.L, ship_data_defaults=None)
    fn = {
        "Tanker": "evaluate_csr_rules_tanker",
        "VLCC": "evaluate_csr_rules_vlcc",
        "BULKC": "evaluate_csr_rules_bulkc",
    }[stem]
    return getattr(mod, fn)(gi, ship_data, ship)


def _reeval_kr(stem: str, ShipCls, gi: dict) -> dict:
    mod = _load_generator_module(stem)
    ship = ShipCls(gi)
    fn = {
        "CNTR": "evaluate_kr_rules_cntr",
        "LNGC": "evaluate_kr_rules_lngc",
        "LPGC": "evaluate_kr_rules_lpgc",
    }[stem]
    return getattr(mod, fn)(gi, ship)


SHIP_CONFIG: dict[str, dict[str, Any]] = {
    "BULKC":  {"cls": FakeBULKC,  "eval": "csr", "cb": 0.80, "dir": "BULKC"},
    "CNTR":   {"cls": FakeCNTR,   "eval": "kr",             "dir": "CNTR"},
    "LNGC":   {"cls": FakeLNGC,   "eval": "kr",             "dir": "LNGC"},
    "LPGC":   {"cls": FakeLPGC,   "eval": "kr",             "dir": "LPGC"},
    "Tanker": {"cls": FakeTanker, "eval": "csr", "cb": 0.82, "dir": "Tanker"},
    "VLCC":   {"cls": FakeVLCC,   "eval": "csr", "cb": 0.83, "dir": "VLCC"},
}


def _load_samples(ship: str) -> list[tuple[str, dict]]:
    root = SAMPLES_ROOT / SHIP_CONFIG[ship]["dir"] / "json"
    out: list[tuple[str, dict]] = []
    if not root.exists():
        return out
    for p in sorted(root.glob("*.json")):
        try:
            with p.open() as f:
                d = json.load(f)
        except Exception:
            continue
        gi = d.get("generator_inputs")
        if not isinstance(gi, dict) or not gi:
            continue
        out.append((p.name, gi))
    return out


def _aggregate(checks: list[dict]) -> dict:
    per_check: dict[str, dict[str, int]] = {}
    for c in checks:
        cid = c.get("check_id")
        st = c.get("status")
        if not cid or not st:
            continue
        per_check.setdefault(cid, {}).setdefault(st, 0)
        per_check[cid][st] += 1
    return per_check


def _merge(total: dict, per_check: dict) -> None:
    for cid, counts in per_check.items():
        tgt = total.setdefault(cid, {})
        for st, n in counts.items():
            tgt[st] = tgt.get(st, 0) + n


def _fail_bucket(total_per_check: dict, n_samples: int) -> dict:
    """Reconstruct the 'all_pass / 1-2 fails / 3+ fails' bucket — not precise
    per-sample since we don't track per-sample here, but we still report the
    gross fail counts for a sanity signal."""
    del n_samples
    return total_per_check


def main() -> int:
    if not SAMPLES_ROOT.exists():
        print(f"ERROR: samples dir not found: {SAMPLES_ROOT}")
        return 1

    prev = {}
    if PREV_PATH.exists():
        with PREV_PATH.open() as f:
            prev = json.load(f)

    summary: dict[str, Any] = {}
    print(f"{'Ship':7s} {'N':>4s}  status shift (pass/fail/undet/notmod)")
    print("-" * 72)
    for ship, cfg in SHIP_CONFIG.items():
        samples = _load_samples(ship)
        if not samples:
            print(f"{ship:7s} {0:4d}  (no samples with generator_inputs)")
            continue
        total: dict[str, dict[str, int]] = {}
        n_ok = 0
        n_err = 0
        for name, gi in samples:
            try:
                if cfg["eval"] == "csr":
                    result = _reeval_csr(ship, cfg["cls"], cfg.get("cb", 0.82), gi)
                else:
                    result = _reeval_kr(ship, cfg["cls"], gi)
                checks = result.get("auto_checks", [])
                _merge(total, _aggregate(checks))
                n_ok += 1
            except Exception as exc:
                n_err += 1
                if n_err <= 3:
                    print(f"  ! {ship}/{name}: {type(exc).__name__}: {exc}")
        summary[ship] = {
            "n_samples_with_checks": n_ok,
            "n_samples_errors": n_err,
            "per_check": total,
        }
        prev_ship = prev.get(ship, {}).get("per_check", {})
        # Compute aggregate totals
        new_totals = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
        for cid, counts in total.items():
            for st, n in counts.items():
                if st in new_totals:
                    new_totals[st] += n
        old_totals = {"pass": 0, "fail": 0, "undetermined": 0, "not_modeled": 0}
        for cid, counts in prev_ship.items():
            for st, n in counts.items():
                if st in old_totals:
                    old_totals[st] += n
        print(f"{ship:7s} {n_ok:4d}  "
              f"old={old_totals['pass']:4d}/{old_totals['fail']:4d}/"
              f"{old_totals['undetermined']:4d}/{old_totals['not_modeled']:4d}  →  "
              f"new={new_totals['pass']:4d}/{new_totals['fail']:4d}/"
              f"{new_totals['undetermined']:4d}/{new_totals['not_modeled']:4d}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_PATH.relative_to(REPO_ROOT)}")

    # Emit a per-check diff for any check where fail/undet counts changed.
    print("\nPer-check diffs (check_id : old → new for each status):")
    for ship, data in summary.items():
        new = data["per_check"]
        old = prev.get(ship, {}).get("per_check", {})
        all_cids = sorted(set(list(new.keys()) + list(old.keys())))
        changes = []
        for cid in all_cids:
            o = old.get(cid, {})
            n = new.get(cid, {})
            if o == n:
                continue
            statuses = sorted(set(list(o.keys()) + list(n.keys())))
            parts = []
            for st in statuses:
                if o.get(st, 0) != n.get(st, 0):
                    parts.append(f"{st}:{o.get(st, 0)}→{n.get(st, 0)}")
            if parts:
                changes.append(f"  {cid:26s} {' '.join(parts)}")
        if changes:
            print(f"\n[{ship}]")
            for c in changes:
                print(c)

    return 0


if __name__ == "__main__":
    sys.exit(main())
