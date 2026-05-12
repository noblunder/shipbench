#!/usr/bin/env python3
"""Gemini frontier inference on ShipBench.

Mirrors run_frontier_openai.py / run_frontier_v3.py. Uses google-genai (new SDK).
Reads GOOGLE_API_KEY or GEMINI_API_KEY from env. Default model: gemini-2.5-pro.

Gemini 2.5 supports a 'thinking_budget' param controlling hidden reasoning tokens
(analogous to GPT-5's reasoning_effort). -1 = auto, 0 = disabled, >0 = explicit cap.
Native-resolution image support; no detail=high parameter needed.

Usage:
  export GOOGLE_API_KEY=AIza...
  python scripts/run_frontier_gemini.py \
      --task-file data/shipbench/task_main_eval_opus_paired.jsonl \
      --output outputs/frontier_eval/gemini25pro_main.jsonl \
      --n-per-task 200 --max-tokens 4096 --thinking-budget 1024

Resume logic: existing qa_ids in --output are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("<SHIPBENCH_ROOT>")
GT_PATH = ROOT / "data" / "shipbench" / "task_main_eval.jsonl"
PROCESSED = ROOT / "data" / "processed"

DEFAULT_MODEL = "gemini-3.1-pro-preview"  # Google flagship at submission time; gemini-2.5-pro is the fallback
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


def section_png(candidate_id, ship_type):
    p = PROCESSED / ship_type / "section_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def compart_png(candidate_id, ship_type):
    p = PROCESSED / ship_type / "compart_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def get_image_paths(item):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-per-task", type=int, default=594)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task-file", type=str, default=None)
    ap.add_argument("--tasks", type=str, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--thinking-budget", type=int, default=-1,
                    help="Gemini 2.5+ thinking_budget. -1=auto (default), 0=disabled, "
                         ">0=explicit token cap. Try 1024 for moderate, 4096 for heavy thinking.")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Parallel API workers (default 8). Reduces wall-clock; tune to vendor rate limits.")
    args = ap.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY env var not set")

    from google import genai
    from google.genai import types as gtypes

    client = genai.Client(api_key=api_key)

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

    # Build reusable generation config
    gen_config_kwargs = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0,
        "max_output_tokens": args.max_tokens,
    }
    # Gemini 2.5+ supports thinking; older Gemini (1.5, 2.0) does not.
    # Pattern covers: gemini-2.5-*, gemini-3-*, gemini-3.1-*, *thinking*.
    model_lower = args.model.lower()
    supports_thinking = (
        model_lower.startswith("gemini-2.5")
        or model_lower.startswith("gemini-3")
        or "thinking" in model_lower
    )
    if supports_thinking:
        gen_config_kwargs["thinking_config"] = gtypes.ThinkingConfig(
            thinking_budget=args.thinking_budget,
        )

    def run_one(item):
        image_paths = get_image_paths(item)
        parts = [item["question"]]
        for p in image_paths:
            parts.append(gtypes.Part.from_bytes(data=p.read_bytes(), mime_type="image/png"))
        try:
            resp = client.models.generate_content(
                model=args.model,
                contents=parts,
                config=gtypes.GenerateContentConfig(**gen_config_kwargs),
            )
        except Exception as e:
            return item, None, e

        pred = resp.text or ""
        finish_reason = None
        if resp.candidates:
            fr = resp.candidates[0].finish_reason
            finish_reason = fr.name if hasattr(fr, "name") else (str(fr) if fr else None)
        usage = resp.usage_metadata
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        reasoning_tokens = getattr(usage, "thoughts_token_count", None) if usage else None

        rec = {
            "qa_id": item["qa_id"],
            "task": item["task"],
            "subtask": item.get("subtask", item["task"]),
            "ship_type": item["ship_type"],
            "prediction": pred,
            "model": args.model,
            "finish_reason": finish_reason,
            "reasoning_tokens": reasoning_tokens,
            "completion_tokens": completion_tokens,
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
