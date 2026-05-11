#!/usr/bin/env python3
"""
Phase 0.4 — Ship-type stratified train/val/test splits.

Reads manifest.jsonl and creates splits_v2/ with 80/10/10 split,
stratified by ship_type to ensure each split has proportional representation.

Output:
  data/splits_v2/train.jsonl
  data/splits_v2/val.jsonl
  data/splits_v2/test.jsonl
  data/splits_v2/split_stats.json

Usage:
    python scripts/09_make_splits.py
    python scripts/09_make_splits.py --train-frac 0.8 --val-frac 0.1
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "processed" / "manifest.jsonl"
SPLITS_DIR = ROOT / "data" / "splits_v2"


def main():
    parser = argparse.ArgumentParser(description="Phase 0.4: Generate splits")
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    test_frac = 1.0 - args.train_frac - args.val_frac
    assert test_frac > 0, "train + val must be < 1.0"

    rng = random.Random(args.seed)

    # Load manifest grouped by ship_type
    by_ship = defaultdict(list)
    with open(MANIFEST) as f:
        for line in f:
            d = json.loads(line)
            by_ship[d["ship_type"]].append(d)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    stats = {}

    for ship, entries in sorted(by_ship.items()):
        rng.shuffle(entries)
        n = len(entries)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        n_test = n - n_train - n_val

        splits["train"].extend(entries[:n_train])
        splits["val"].extend(entries[n_train:n_train + n_val])
        splits["test"].extend(entries[n_train + n_val:])

        stats[ship] = {
            "total": n,
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }

    # Shuffle within each split
    for split_name in splits:
        rng.shuffle(splits[split_name])

    # Write splits
    for split_name, entries in splits.items():
        path = SPLITS_DIR / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"  {split_name:5s}: {len(entries):5d} entries → {path}")

    # Write stats
    total_stats = {
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": round(test_frac, 2),
        "seed": args.seed,
        "total": sum(s["total"] for s in stats.values()),
        "train": sum(s["train"] for s in stats.values()),
        "val": sum(s["val"] for s in stats.values()),
        "test": sum(s["test"] for s in stats.values()),
        "per_ship": stats,
    }
    with open(SPLITS_DIR / "split_stats.json", "w") as f:
        json.dump(total_stats, f, indent=2, ensure_ascii=False)

    print(f"\n  Total: train={total_stats['train']}, val={total_stats['val']}, test={total_stats['test']}")
    print(f"\n  Per-ship breakdown:")
    for ship, s in sorted(stats.items()):
        print(f"    {ship:6s}: train={s['train']:4d}  val={s['val']:4d}  test={s['test']:4d}")


if __name__ == "__main__":
    main()
