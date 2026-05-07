#!/usr/bin/env python3
"""
Phase 0.2.E (step 2) — Farthest Point Sampling for visual diversity.

Uses simple pixel-based embeddings (resize→grayscale→PCA) since PyTorch/DINOv2
are not available. For ship section drawings (line art), this captures the key
geometric differences (proportions, member positions, structural layout).

Per ship × stratum:
  1. Load section PNGs → resize to 128×128 grayscale → flatten
  2. PCA to 64 dims
  3. Farthest Point Sampling to select target count
  4. Output: selected candidate IDs + diversity metrics

Usage:
    python scripts/05_fps_select.py                        # default 6000 final
    python scripts/05_fps_select.py --final-target 6000
    python scripts/05_fps_select.py --ships LPGC --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
STRATIFIED_DIR = ROOT / "data" / "candidates_R1" / "stratified"
SHIPS = ["Tanker", "VLCC", "BULKC", "CNTR", "LNGC", "LPGC"]

# Per-ship final target counts
FINAL_TARGETS = {
    "Tanker": 900,
    "VLCC":   900,
    "BULKC":  1050,
    "CNTR":   1050,
    "LNGC":   1200,
    "LPGC":   900,
}  # Total = 6000

IMG_SIZE = 128
PCA_DIMS = 64


def load_embeddings(ship: str, png_dir: Path, candidate_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load section PNGs, resize, grayscale, flatten → feature matrix."""
    features = []
    valid_ids = []
    for cid in candidate_ids:
        png_path = png_dir / f"{cid}.png"
        if not png_path.exists():
            continue
        try:
            img = Image.open(png_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
            features.append(np.array(img, dtype=np.float32).flatten())
            valid_ids.append(cid)
        except Exception:
            continue
    if not features:
        return np.zeros((0, IMG_SIZE * IMG_SIZE)), []
    X = np.stack(features)
    # Normalize
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    return X, valid_ids


def pca_reduce(X: np.ndarray, n_components: int = PCA_DIMS) -> np.ndarray:
    """PCA dimensionality reduction."""
    if X.shape[0] <= n_components:
        return X
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


def farthest_point_sampling(X: np.ndarray, k: int, seed: int = 42) -> list[int]:
    """Greedy FPS: select k points from X maximizing min distance to selected set."""
    n = X.shape[0]
    if k >= n:
        return list(range(n))

    rng = np.random.RandomState(seed)
    selected = [rng.randint(n)]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last = selected[-1]
        dists = np.sum((X - X[last]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1  # exclude already selected
        next_idx = np.argmax(min_dists)
        selected.append(int(next_idx))

    return selected


def diversity_metrics(X: np.ndarray, selected_idx: list[int]) -> dict:
    """Compute diversity metrics for selected subset."""
    X_sel = X[selected_idx]
    n = len(selected_idx)
    if n < 2:
        return {"mean_nn_dist": 0, "mean_pairwise_dist": 0}

    # Mean nearest-neighbor distance
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2).fit(X_sel)
    dists, _ = nn.kneighbors(X_sel)
    mean_nn = float(dists[:, 1].mean())

    # Mean pairwise distance (sample if too many)
    if n <= 2000:
        from sklearn.metrics import pairwise_distances
        D = pairwise_distances(X_sel)
        mean_pw = float(D.sum() / (n * (n - 1)))
    else:
        # Sample 2000 pairs
        pairs = np.random.RandomState(42).choice(n, (2000, 2))
        diffs = X_sel[pairs[:, 0]] - X_sel[pairs[:, 1]]
        mean_pw = float(np.sqrt((diffs ** 2).sum(axis=1)).mean())

    return {"mean_nn_dist": round(mean_nn, 4), "mean_pairwise_dist": round(mean_pw, 4)}


def fps_select_ship(ship: str, target: int) -> dict:
    """Run FPS for one ship type, selecting across all strata."""
    ship_dir = STRATIFIED_DIR / ship
    png_dir = ship_dir / "section_png"

    if not png_dir.exists():
        print(f"  SKIP {ship}: no section_png dir")
        return {"ship": ship, "selected": 0, "error": "no section_png"}

    # Collect all candidate IDs with their stratum
    cid_stratum = {}
    for stratum_dir in sorted(ship_dir.iterdir()):
        if not stratum_dir.is_dir() or stratum_dir.name == "section_png":
            continue
        for jf in sorted(stratum_dir.glob(f"{ship}-*.json")):
            cid = jf.stem
            cid_stratum[cid] = stratum_dir.name

    all_cids = sorted(cid_stratum.keys())
    print(f"  Loading {len(all_cids)} embeddings...")
    X_raw, valid_cids = load_embeddings(ship, png_dir, all_cids)
    if len(valid_cids) == 0:
        return {"ship": ship, "selected": 0, "error": "no valid PNGs"}

    print(f"  PCA {X_raw.shape[1]} → {PCA_DIMS} dims ({len(valid_cids)} samples)...")
    X_pca = pca_reduce(X_raw)

    actual_target = min(target, len(valid_cids))
    print(f"  FPS selecting {actual_target} from {len(valid_cids)}...")
    selected_idx = farthest_point_sampling(X_pca, actual_target)
    selected_cids = [valid_cids[i] for i in selected_idx]

    metrics_before = diversity_metrics(X_pca, list(range(len(valid_cids))))
    metrics_after = diversity_metrics(X_pca, selected_idx)

    # Stratum distribution in selection
    stratum_counts = {}
    for cid in selected_cids:
        st = cid_stratum.get(cid, "unknown")
        stratum_counts[st] = stratum_counts.get(st, 0) + 1

    result = {
        "ship": ship,
        "pool_size": len(valid_cids),
        "selected": len(selected_cids),
        "target": target,
        "stratum_distribution": stratum_counts,
        "diversity_before": metrics_before,
        "diversity_after": metrics_after,
        "selected_ids": selected_cids,
    }

    print(f"  {ship}: {len(selected_cids)} selected from {len(valid_cids)}")
    print(f"  Diversity NN dist: {metrics_before['mean_nn_dist']:.4f} → {metrics_after['mean_nn_dist']:.4f}")
    print(f"  Strata: {stratum_counts}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 0.2.E: FPS visual diversity selection")
    parser.add_argument("--final-target", type=int, default=6000)
    parser.add_argument("--ships", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ships = args.ships or SHIPS

    # Scale per-ship targets if custom final target
    if args.final_target != 6000:
        scale = args.final_target / 6000
        targets = {s: int(FINAL_TARGETS[s] * scale) for s in ships}
    else:
        targets = {s: FINAL_TARGETS[s] for s in ships}

    print(f"Phase 0.2.E FPS selection")
    print(f"  Final target: {args.final_target}")
    print(f"  Per-ship: {targets}")
    print()

    t0 = time.time()
    results = {}
    grand_total = 0

    for ship in ships:
        print(f"=== {ship} (target={targets[ship]}) ===")
        r = fps_select_ship(ship, targets[ship])
        results[ship] = r
        grand_total += r.get("selected", 0)
        print()

    # Save FPS report
    report = {
        "grand_total": grand_total,
        "per_ship": {s: {k: v for k, v in r.items() if k != "selected_ids"}
                     for s, r in results.items()},
    }
    report_path = STRATIFIED_DIR / "fps_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save selected manifest
    manifest_path = STRATIFIED_DIR / "fps_selected.jsonl"
    with open(manifest_path, "w") as f:
        for ship in ships:
            for cid in results[ship].get("selected_ids", []):
                f.write(json.dumps({"candidate_id": cid, "ship_type": ship}) + "\n")

    elapsed = time.time() - t0
    print(f"Done. {grand_total} selected in {elapsed:.0f}s.")
    print(f"  Report: {report_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
