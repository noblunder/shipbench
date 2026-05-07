#!/usr/bin/env python3
"""Frontier model (Claude Opus 4.7) zero-shot eval on the new 9-task v2 benchmark.

Subsample: n=50 per task × 9 tasks = 450 items. Cost estimate: ~$25-50.
Adds a frontier reference baseline to the four open-weight VLMs +  two LoRA SFT.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scripts/run_frontier_v2.py --output results/main/claude_opus_main.jsonl

Output format mirrors results/main/{model}_main.jsonl for compatibility with
eval_main.py.
"""
import argparse
import base64
import json
import os
import random
import time
from pathlib import Path

ROOT = Path("<SHIPBENCH_ROOT>")
GT_PATH = ROOT / "data" / "shipbench3d_v2" / "task_main_eval.jsonl"
PROCESSED = ROOT / "data" / "processed_R1"

MODEL_ID = "claude-opus-4-7"  # latest Opus
SYSTEM_PROMPT = (
    "You are a ship-structural-design assistant. Answer the user's question about the provided "
    "ship-structural drawing(s) in the exact format requested. Output only the answer, no explanation."
)

TASKS = [
    "A1_shiptype", "A2_stiffener_type",
    "B1_plate_thickness", "B2_stiffener_size",
    "B3_cargo_capacity_v1", "B4_section_area_v1",
    "C1_compartment_locate", "C2_compartment_boundary", "C3_bulkhead_position",
]


def section_png(candidate_id: str, ship_type: str) -> Path | None:
    p = PROCESSED / ship_type / "section_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def compart_png(candidate_id: str, ship_type: str) -> Path | None:
    p = PROCESSED / ship_type / "compart_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def get_image_paths(item: dict) -> list[Path]:
    paths = []
    for k in item.get("images", []):
        if k == "section_png":
            p = section_png(item["candidate_id"], item["ship_type"])
        elif k == "compart_png":
            p = compart_png(item["candidate_id"], item["ship_type"])
        else:
            p = None
        if p:
            paths.append(p)
    return paths


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-per-task", type=int, default=50)
    ap.add_argument("--task-file", type=str, default=None,
        help="Path to task jsonl. Defaults to data/shipbench3d_v2/task_main_eval.jsonl")
    ap.add_argument("--tasks", type=str, default=None,
        help="Comma-separated task names to run. Defaults to all 9 main tasks.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=128,
        help="max_tokens for generation (use 8192 for B4 v3 CoT, 1024 for B3 v3, 64 for C3 v3, 128 for short answers)")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: ANTHROPIC_API_KEY env var not set")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    # Group items by task and subsample
    rng = random.Random(args.seed)
    gt_path = args.task_file or str(GT_PATH)
    all_items = [json.loads(line) for line in open(gt_path)]
    tasks_to_run = TASKS if args.tasks is None else [s.strip() for s in args.tasks.split(",") if s.strip()]
    by_task = {t: [it for it in all_items if it["task"] == t] for t in tasks_to_run}
    sampled = []
    for t in tasks_to_run:
        items = by_task[t]
        rng.shuffle(items)
        sampled.extend(items[: args.n_per_task])
    print(f"Sampled {len(sampled)} items ({args.n_per_task}/task × {len(tasks_to_run)} tasks)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["qa_id"])
        print(f"  Resuming: {len(done_ids)} already done")

    pending = [it for it in sampled if it["qa_id"] not in done_ids]
    print(f"  Pending: {len(pending)}")

    t0 = time.time()
    n_done = 0
    n_err = 0
    with open(out_path, "a") as f:
        for i, qa in enumerate(pending):
            paths = get_image_paths(qa)
            content = []
            for p in paths:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": encode_image(p)}})
            content.append({"type": "text", "text": qa["question"]})
            try:
                resp = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=args.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": content}],
                )
                pred = resp.content[0].text.strip() if resp.content else ""
                n_done += 1
            except Exception as e:
                pred = f"[ERROR: {type(e).__name__}: {e}]"
                n_err += 1
            record = {
                "qa_id": qa["qa_id"], "task": qa["task"], "subtask": qa.get("subtask"),
                "ship_type": qa["ship_type"], "prediction": pred, "model": MODEL_ID,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] {qa['task']}  pred={pred[:50]!r}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min  err={n_err}", flush=True)
            time.sleep(0.4)  # gentle rate limiting

    print(f"\n[DONE] {n_done} ok, {n_err} errors in {(time.time()-t0)/60:.1f}min")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
