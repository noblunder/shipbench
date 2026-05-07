#!/usr/bin/env python3
"""GPT-5 frontier inference on ShipBench.

Mirrors run_frontier_v3.py but uses OpenAI's API. Reads OPENAI_API_KEY from env.
Default model: gpt-5 (override via --model).

Usage:
  export OPENAI_API_KEY=sk-...
  python scripts/run_frontier_openai.py \
      --output outputs/frontier_eval/gpt5_main.jsonl \
      --n-per-task 594

Resume logic: existing qa_ids in --output are skipped.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("<SHIPBENCH_ROOT>")
GT_PATH = ROOT / "data" / "shipbench3d_v2" / "task_main_eval.jsonl"
PROCESSED = ROOT / "data" / "processed_R1"

DEFAULT_MODEL = "gpt-5"
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


def section_png(candidate_id: str, ship_type: str):
    p = PROCESSED / ship_type / "section_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def compart_png(candidate_id: str, ship_type: str):
    p = PROCESSED / ship_type / "compart_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def get_image_paths(item: dict):
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
    ap.add_argument("--n-per-task", type=int, default=594)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task-file", type=str, default=None)
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--reasoning-effort", type=str, default=None,
                    choices=[None, "minimal", "low", "medium", "high"],
                    help="GPT-5/o-series reasoning_effort. 'minimal' fastest+cheapest, 'medium' default, 'high' most thinking. None = API default.")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Parallel API workers (default 8). Reduces wall-clock; OpenAI tier-5 tolerates up to ~32.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY env var not set")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

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
    print(f"Sampled {len(sampled)} items ({args.n_per_task}/task × {len(tasks_to_run)} tasks)", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["qa_id"])
        print(f"  Resuming: {len(done_ids)} already done", flush=True)

    pending = [it for it in sampled if it["qa_id"] not in done_ids]
    print(f"  Pending: {len(pending)}", flush=True)

    def run_one(item):
        """Single API call → (item, rec_dict, error_or_None). Thread-safe."""
        image_paths = get_image_paths(item)
        content = [{"type": "text", "text": item["question"]}]
        for p in image_paths:
            b64 = encode_image(p)
            # detail='high': preserve drawing-level detail. ShipBench compart_png
            # images are 3937x702; default 'auto' downsamples to ~512px and the
            # model literally cannot read mm dimensions / hold labels (verified
            # via 5/7 smoke: GPT-5 refused with 'please provide the image'.)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })
        kwargs = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        if args.model.startswith(("gpt-5", "o1", "o3", "o4")):
            kwargs["max_completion_tokens"] = args.max_tokens
            if args.reasoning_effort:
                kwargs["reasoning_effort"] = args.reasoning_effort
        else:
            kwargs["max_tokens"] = args.max_tokens
            kwargs["temperature"] = 0

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            return item, None, e

        pred = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason if resp.choices else None
        usage = resp.usage
        reasoning_tokens = None
        if usage and hasattr(usage, "completion_tokens_details"):
            details = usage.completion_tokens_details
            if details and hasattr(details, "reasoning_tokens"):
                reasoning_tokens = details.reasoning_tokens
        rec = {
            "qa_id": item["qa_id"],
            "task": item["task"],
            "subtask": item.get("subtask", item["task"]),
            "ship_type": item["ship_type"],
            "prediction": pred,
            "model": args.model,
            "finish_reason": finish_reason,
            "reasoning_tokens": reasoning_tokens,
            "completion_tokens": usage.completion_tokens if usage else None,
        }
        return item, rec, None

    write_lock = threading.Lock()
    n_ok = n_err = 0
    t0 = time.time()

    print(f"  Running with concurrency={args.concurrency}", flush=True)
    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(run_one, it) for it in pending]
        for i, fut in enumerate(as_completed(futures)):
            item, rec, err = fut.result()
            if err is not None:
                n_err += 1
                print(f"  ERR on {item['qa_id']}: {err}", flush=True)
                continue
            with write_lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            n_ok += 1
            if (i + 1) % 25 == 0 or i < 5:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(pending) - i - 1) / max(rate, 1e-6) / 60
                pred = rec.get("prediction", "")
                print(f"[{i+1}/{len(pending)}] {item['task']} pred={pred[:50]!r} | rate={rate:.2f}/s ETA={eta:.0f}min", flush=True)

    print(f"[DONE] {n_ok} ok, {n_err} errors in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
