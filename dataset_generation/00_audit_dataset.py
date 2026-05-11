"""
Phase 0.2.A — Diversity Audit of current processed dataset.

Measures the 6 diversity axes:
  P1 Parameter coverage  — marginal + joint distribution of numeric params
  P2 Visual diversity    — DINOv2 embedding pairwise distance
  P3 Topology            — member counts, hold count
  P4 Rule state          — KR rule check status distribution
  P5 Realism             — (basic) parameter range vs textbook heuristics
  P6 Difficulty          — placeholder (requires frontier baseline, run later)

Outputs:
  outputs/audit/
    ├── params_stats.json          # per-ship numeric summary
    ├── params_<ship>.csv          # all samples wide table
    ├── rule_state.json            # per-ship rule pass/fail/undetermined counts
    ├── visual_embeddings.npz      # DINOv2 embeddings + ids
    ├── visual_diversity.json      # mean/median/5p pairwise distance
    ├── figures/
    │     ├── params_<ship>.png    # marginal histograms
    │     ├── rule_state_<ship>.png
    │     └── visual_tsne.png
    └── audit_report.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# matplotlib (Agg)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHIPS = ["BULKC", "CNTR", "LNGC", "LPGC", "Tanker", "VLCC"]
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data" / "processed"
OUT_DIR = REPO / "outputs" / "audit"
FIG_DIR = OUT_DIR / "figures"

# Parameter name aliases across ship types (canonical name -> possible source names)
PARAM_ALIASES: dict[str, list[str]] = {
    "L_m": ["L_m"],
    "B_m": ["B_m"],
    "D_m": ["D_m"],
    "HL_m": ["HL_m"],
    "n_hold": ["number_of_hold"],
    "camber_m": ["camberUpper_m", "camber_m"],
    "DB_m": ["doubleBottom_m"],
    "DS_m": ["doubleSide_m"],
    "bilge_R_m": ["bilgeRadius_m"],
}


def _get_param(gi: dict, key: str) -> float | None:
    for src in PARAM_ALIASES[key]:
        if src in gi and gi[src] is not None:
            try:
                return float(gi[src])
            except (TypeError, ValueError):
                return None
    return None


def load_sample(json_path: Path) -> dict[str, Any] | None:
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return None
    gi = d.get("generator_inputs", {}) or {}
    row: dict[str, Any] = {
        "sample_id": d.get("sample_id") or json_path.stem,
        "ship_type": d.get("ship_type", "?"),
        "json_path": str(json_path),
    }
    for canon in PARAM_ALIASES:
        row[canon] = _get_param(gi, canon)

    # rule state — CNTR/LNGC/LPGC emit under "kr"; BULKC/Tanker/VLCC under "csr"
    kr = d.get("kr", {}) or {}
    csr = d.get("csr", {}) or {}
    if kr.get("auto_checks"):
        checks = kr.get("auto_checks", []) or []
        row["rule_schema"] = "kr"
    elif csr.get("auto_checks"):
        checks = csr.get("auto_checks", []) or []
        row["rule_schema"] = "csr"
    else:
        checks = []
        row["rule_schema"] = "none"
    row["n_checks"] = len(checks)
    counts = Counter()
    for c in checks:
        counts[c.get("status", "unknown")] += 1
    for k in ("pass", "fail", "undetermined", "not_modeled", "unknown"):
        row[f"rule_{k}"] = counts.get(k, 0)
    row["rule_check_ids"] = [c.get("check_id") for c in checks]
    row["rule_statuses"] = {c.get("check_id"): c.get("status") for c in checks}

    # topology proxies from geometry
    geom = d.get("geometry", {}) or {}
    derived = geom.get("derived", {}) or {}
    row["derived_keys"] = list(derived.keys())

    # image path
    stem = json_path.stem
    row["section_png"] = str(DATA_DIR / row["ship_type"] / "section_png" / f"{stem}.png")
    return row


def load_all() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for ship in SHIPS:
        json_dir = DATA_DIR / ship / "json"
        if not json_dir.exists():
            out[ship] = []
            continue
        rows = []
        for jp in sorted(json_dir.glob("*.json")):
            r = load_sample(jp)
            if r is not None:
                rows.append(r)
        out[ship] = rows
    return out


# ---------- P1 Parameter stats ----------
def param_stats(rows: list[dict]) -> dict:
    stats = {}
    for key in PARAM_ALIASES:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            stats[key] = None
            continue
        arr = np.array(vals, dtype=float)
        stats[key] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "unique": int(np.unique(np.round(arr, 3)).size),
        }
    return stats


def plot_param_hist(rows: list[dict], ship: str) -> None:
    keys = [k for k in PARAM_ALIASES if any(r.get(k) is not None for r in rows)]
    if not keys:
        return
    n = len(keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.4))
    axes = np.array(axes).reshape(-1)
    for i, k in enumerate(keys):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        ax = axes[i]
        ax.hist(vals, bins=max(5, min(20, len(vals) // 4)), color="#3366cc", edgecolor="white")
        ax.set_title(f"{k}  n={len(vals)}", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(len(keys), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{ship} — parameter marginal distributions", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"params_{ship}.png", dpi=130)
    plt.close(fig)


# ---------- P4 Rule state ----------
def rule_state_summary(rows: list[dict]) -> dict:
    # per check_id: {status: count}
    per_check: dict[str, Counter] = defaultdict(Counter)
    per_sample_counts = Counter()
    n_with_checks = 0
    for r in rows:
        if r["n_checks"] == 0:
            continue
        n_with_checks += 1
        for cid, status in (r["rule_statuses"] or {}).items():
            per_check[cid][status] += 1
        # per-sample fail count bucket
        nf = r["rule_fail"]
        if nf == 0:
            per_sample_counts["all_pass"] += 1
        elif nf <= 2:
            per_sample_counts["1_2_fails"] += 1
        else:
            per_sample_counts["3_plus_fails"] += 1
    return {
        "n_samples_with_checks": n_with_checks,
        "per_check": {k: dict(v) for k, v in per_check.items()},
        "per_sample_bucket": dict(per_sample_counts),
    }


def plot_rule_state(rule_summary: dict, ship: str) -> None:
    per_check = rule_summary.get("per_check", {})
    if not per_check:
        return
    checks = list(per_check.keys())
    statuses = ["pass", "fail", "undetermined", "not_modeled"]
    colors = {"pass": "#2ca02c", "fail": "#d62728", "undetermined": "#7f7f7f", "not_modeled": "#bcbd22"}
    data = {s: [per_check[c].get(s, 0) for c in checks] for s in statuses}
    fig, ax = plt.subplots(figsize=(max(5, len(checks) * 0.6), 3.2))
    bottom = np.zeros(len(checks))
    for s in statuses:
        vals = np.array(data[s])
        ax.bar(checks, vals, bottom=bottom, color=colors[s], label=s)
        bottom += vals
    ax.set_title(f"{ship} — KR rule check states")
    ax.set_ylabel("count")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"rule_state_{ship}.png", dpi=130)
    plt.close(fig)


# ---------- P2 Visual diversity via DINOv2 ----------
def embed_images(all_rows: dict[str, list[dict]], device: str = "cuda", batch: int = 16) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    model_name = "facebook/dinov2-base"
    print(f"[embed] loading {model_name} on {device}")
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embs = {}  # ship -> list of (sample_id, vec)
    with torch.no_grad():
        for ship, rows in all_rows.items():
            if not rows:
                all_embs[ship] = {"ids": [], "vecs": np.zeros((0, 1), dtype=np.float32)}
                continue
            ids = []
            vecs = []
            imgs = []
            cache_ids = []
            for r in rows:
                p = Path(r["section_png"])
                if not p.exists():
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                imgs.append(img)
                cache_ids.append(r["sample_id"])
                if len(imgs) == batch:
                    inp = proc(images=imgs, return_tensors="pt").to(device)
                    out = model(**inp)
                    feats = out.last_hidden_state[:, 0, :]  # CLS token
                    feats = torch.nn.functional.normalize(feats, dim=-1)
                    vecs.append(feats.cpu().numpy())
                    ids.extend(cache_ids)
                    imgs, cache_ids = [], []
            if imgs:
                inp = proc(images=imgs, return_tensors="pt").to(device)
                out = model(**inp)
                feats = out.last_hidden_state[:, 0, :]
                feats = torch.nn.functional.normalize(feats, dim=-1)
                vecs.append(feats.cpu().numpy())
                ids.extend(cache_ids)
            if vecs:
                arr = np.concatenate(vecs, axis=0)
            else:
                arr = np.zeros((0, 768), dtype=np.float32)
            print(f"[embed] {ship}: {arr.shape}")
            all_embs[ship] = {"ids": ids, "vecs": arr}
    return all_embs


def visual_diversity(embs: dict) -> dict:
    out = {}
    for ship, d in embs.items():
        V = d["vecs"]
        ids = d["ids"]
        n = V.shape[0]
        if n < 2:
            out[ship] = {"n": n}
            continue
        # cosine distance since embeddings are L2-normalized → 1 - V @ V.T
        sims = V @ V.T
        dists = 1.0 - sims
        iu = np.triu_indices(n, k=1)
        pair = dists[iu]
        # nearest-neighbor distance per point
        np.fill_diagonal(dists, np.inf)
        nn = dists.min(axis=1)
        out[ship] = {
            "n": int(n),
            "pairwise_mean": float(pair.mean()),
            "pairwise_median": float(np.median(pair)),
            "pairwise_p05": float(np.percentile(pair, 5)),
            "nn_mean": float(nn.mean()),
            "nn_median": float(np.median(nn)),
            "nn_p05": float(np.percentile(nn, 5)),
        }
    return out


def plot_tsne(embs: dict) -> None:
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("[tsne] sklearn not available, skipping")
        return
    all_vecs = []
    all_labels = []
    for ship, d in embs.items():
        V = d["vecs"]
        if V.shape[0] == 0:
            continue
        all_vecs.append(V)
        all_labels.extend([ship] * V.shape[0])
    if not all_vecs:
        return
    X = np.concatenate(all_vecs, axis=0)
    print(f"[tsne] fitting on {X.shape}")
    ts = TSNE(n_components=2, perplexity=min(30, max(5, X.shape[0] // 10)), init="pca", random_state=42)
    Y = ts.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"BULKC": "#1f77b4", "CNTR": "#ff7f0e", "LNGC": "#2ca02c",
              "LPGC": "#d62728", "Tanker": "#9467bd", "VLCC": "#8c564b"}
    for ship in SHIPS:
        idx = [i for i, l in enumerate(all_labels) if l == ship]
        if not idx:
            continue
        ax.scatter(Y[idx, 0], Y[idx, 1], s=18, alpha=0.7, c=colors.get(ship, "#333"), label=ship, edgecolor="none")
    ax.set_title("DINOv2 embedding t-SNE (section PNG, R1 current 431)")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "visual_tsne.png", dpi=140)
    plt.close(fig)


# ---------- Reporting ----------
def write_params_csv(all_rows: dict[str, list[dict]]) -> None:
    for ship, rows in all_rows.items():
        if not rows:
            continue
        keys = ["sample_id"] + list(PARAM_ALIASES.keys()) + ["rule_pass", "rule_fail", "rule_undetermined", "rule_not_modeled", "n_checks"]
        lines = [",".join(keys)]
        for r in rows:
            lines.append(",".join(str(r.get(k, "")) for k in keys))
        (OUT_DIR / f"params_{ship}.csv").write_text("\n".join(lines))


def write_report(all_rows, pstats, rule_summ, vdiv) -> None:
    total = sum(len(v) for v in all_rows.values())
    lines = [
        "# Diversity Audit Report — processed (current)",
        "",
        f"Total samples: **{total}** across {len(SHIPS)} ship types.",
        "",
        "## 1. Sample counts",
        "",
        "| Ship | N | Has KR rule checks? |",
        "|---|---:|---|",
    ]
    for ship in SHIPS:
        rows = all_rows[ship]
        has_rule = any(r["n_checks"] > 0 for r in rows)
        lines.append(f"| {ship} | {len(rows)} | {'yes' if has_rule else 'NO (legacy CSR schema)'} |")
    lines += ["", "⚠️ BULKC / Tanker / VLCC: legacy CSR schema (`kr: {}`). Re-generation in Phase 0.2.C will unify all to KR schema."]

    lines += ["", "## 2. Parameter statistics (P1 axis)", ""]
    for ship in SHIPS:
        s = pstats.get(ship) or {}
        if not s:
            continue
        lines += [f"### {ship}", "", "| Param | n | mean | std | min | p50 | max | unique |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in s.items():
            if v is None:
                continue
            lines.append(f"| {k} | {v['n']} | {v['mean']:.2f} | {v['std']:.2f} | {v['min']:.2f} | {v['p50']:.2f} | {v['max']:.2f} | {v['unique']} |")
        lines.append("")

    lines += ["## 3. KR rule state (P4 axis)", ""]
    for ship in SHIPS:
        rs = rule_summ.get(ship, {})
        n = rs.get("n_samples_with_checks", 0)
        if n == 0:
            lines.append(f"- **{ship}**: no rule checks (legacy schema)")
            continue
        bucket = rs.get("per_sample_bucket", {})
        total_checks = sum(sum(v.values()) for v in rs["per_check"].values())
        pass_total = sum(v.get("pass", 0) for v in rs["per_check"].values())
        fail_total = sum(v.get("fail", 0) for v in rs["per_check"].values())
        und_total = sum(v.get("undetermined", 0) for v in rs["per_check"].values())
        nm_total = sum(v.get("not_modeled", 0) for v in rs["per_check"].values())
        lines.append(f"### {ship} (n={n})")
        lines.append("")
        lines.append(f"- Check total: {total_checks} → pass {pass_total}, fail {fail_total}, undetermined {und_total}, not_modeled {nm_total}")
        lines.append(f"- Per-sample buckets: {bucket}")
        lines.append("")
    lines.append("**Target stratification (Phase 0.2.D goal)**: all_pass 40% / 1-2 fails 30% / 3+ fails 20% / undetermined 10%.")

    lines += ["", "## 4. Visual diversity (P2 axis, DINOv2-base)", "", "| Ship | n | pair mean | pair median | NN mean | NN 5% |", "|---|---:|---:|---:|---:|---:|"]
    for ship in SHIPS:
        d = vdiv.get(ship, {})
        if d.get("n", 0) < 2 or "pairwise_mean" not in d:
            lines.append(f"| {ship} | {d.get('n', 0)} | — | — | — | — |")
            continue
        lines.append(f"| {ship} | {d['n']} | {d['pairwise_mean']:.3f} | {d['pairwise_median']:.3f} | {d['nn_mean']:.3f} | {d['nn_p05']:.3f} |")
    lines += [
        "",
        "*(pair = pairwise cosine distance, NN = nearest-neighbor cosine distance; ↑ better = more diverse)*",
        "",
        "## 5. Key gaps identified",
        "",
        "1. **Schema inconsistency** — BULKC/Tanker/VLCC still use legacy CSR schema without KR rule checks. Phase 0.2.C re-generation must unify.",
        "2. **Small sample sizes** for LPGC (38) and CNTR (51) — DASG target (1000 each) addresses this.",
        "3. **Rule state imbalance** — (fill in after audit run).",
        "4. **Visual NN distance p05** is the single most important metric; if this is very low (<0.05) there are near-duplicate pairs that FPS filtering must remove.",
        "",
        "## 6. Figures",
        "",
        "- `figures/params_<ship>.png` — parameter histograms per ship",
        "- `figures/rule_state_<ship>.png` — KR rule check state bars (CNTR/LNGC/LPGC only)",
        "- `figures/visual_tsne.png` — t-SNE of DINOv2 embeddings colored by ship type",
        "",
        "---",
        "_Generated by `scripts/00_audit_dataset.py`._",
    ]
    (OUT_DIR / "audit_report.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true", help="skip DINOv2 embedding (P2)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[audit] loading samples")
    all_rows = load_all()
    for ship, rows in all_rows.items():
        print(f"  {ship}: {len(rows)}")

    # P1 params
    print("[audit] computing parameter stats")
    pstats = {ship: param_stats(rows) for ship, rows in all_rows.items()}
    (OUT_DIR / "params_stats.json").write_text(json.dumps(pstats, indent=2))
    for ship, rows in all_rows.items():
        if rows:
            plot_param_hist(rows, ship)
    write_params_csv(all_rows)

    # P4 rule state
    print("[audit] rule state summary")
    rule_summ = {ship: rule_state_summary(rows) for ship, rows in all_rows.items()}
    (OUT_DIR / "rule_state.json").write_text(json.dumps(rule_summ, indent=2))
    for ship in SHIPS:
        plot_rule_state(rule_summ[ship], ship)

    # P2 visual
    if args.skip_embed:
        vdiv = {ship: {"n": len(rows)} for ship, rows in all_rows.items()}
    else:
        print("[audit] embedding images with DINOv2")
        embs = embed_images(all_rows, device=args.device)
        np.savez(OUT_DIR / "visual_embeddings.npz",
                 **{f"{ship}_ids": np.array(d["ids"]) for ship, d in embs.items()},
                 **{f"{ship}_vecs": d["vecs"] for ship, d in embs.items()})
        vdiv = visual_diversity(embs)
        (OUT_DIR / "visual_diversity.json").write_text(json.dumps(vdiv, indent=2))
        plot_tsne(embs)

    # report
    print("[audit] writing report")
    write_report(all_rows, pstats, rule_summ, vdiv)
    print(f"[audit] done → {OUT_DIR}")


if __name__ == "__main__":
    main()
