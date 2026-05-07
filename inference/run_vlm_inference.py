#!/usr/bin/env python3
"""
Local VLM inference runner for ShipBench.

Loads an open-source VLM (default: Qwen3-VL-8B-Instruct), runs it against QA
items in data/shipbench3d/, and writes predictions.jsonl for use with
eval_shipbench3d.py.

Usage:
    # Pilot (100 samples, mix of tasks)
    python scripts/run_vlm_inference.py \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --limit 100 \\
        --output results/qwen3vl8b_pilot.jsonl

    # Full test split
    python scripts/run_vlm_inference.py \\
        --model Qwen/Qwen3-VL-8B-Instruct \\
        --split test \\
        --output results/qwen3vl8b_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_DIR = ROOT / "data" / "shipbench3d"
PROCESSED = ROOT / "data" / "processed_R1"


# ═══════════════════════════════════════════════════════════════
# PROMPT TEMPLATES per task
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert naval architect assistant analyzing ship midship "
    "section drawings. Answer questions concisely. "
    "For yes/no questions, reply with 'yes' or 'no'. "
    "For numeric questions, reply with just the number. "
    "For compliance questions (B1/B2), reply with exactly 'pass' or 'fail' "
    "(not 'yes'/'no'). "
    "For JSON questions (C1/C2), use these exact key names: "
    "L_m (length), B_m (breadth), D_m (depth), HL_m (hold length), "
    "doubleSide_m, doubleBottom_m, bilgeRadius_m, camberUpper_m, "
    "camberTrunk_m, number_of_hold, number_of_cofferdam, "
    "girder0_ratio, girder1_ratio, girder2_ratio, "
    "str1_ratio, str2_ratio, str3_ratio, "
    "lbhd_ratio, tswt_ext_deg, inner_slope_deg, "
    "ds_from_side_m, girder_y_m, outgir_ratio, "
    "gap_tswt_m, gap_hopper_m, strClearance_m. "
    "Reply with a valid JSON object only."
)

MAX_NEW_TOKENS = {
    "A1": 8,    # yes/no
    "A3": 16,   # numeric
    "A4": 32,   # (y, z) coords
    "B1": 8,    # pass/fail
    "B2": 8,    # pass/fail
    "B3": 128,  # free-text rule citation
    "C1": 512,  # JSON output
    "C2": 512,  # JSON output
}


def get_section_png(candidate_id: str, ship_type: str) -> Path | None:
    """Resolve section PNG path from candidate_id."""
    p = PROCESSED / ship_type / "section_png" / f"{candidate_id}.png"
    return p if p.exists() else None


def load_qa_items(split: str | None, limit: int, tasks: list[str] | None) -> list[dict]:
    """Load QA items, optionally filtered by split and subsampled."""
    items = []
    with open(QA_DIR / "all_qa.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if tasks and d["task"] not in tasks:
                continue
            items.append(d)

    if split:
        # Load candidate_ids from split
        split_file = ROOT / "data" / "splits_v2" / f"{split}.jsonl"
        split_cids = set()
        with open(split_file) as f:
            for line in f:
                split_cids.add(json.loads(line)["candidate_id"])
        items = [q for q in items if q["candidate_id"] in split_cids]

    if limit and limit < len(items):
        # Balanced subsample: equal per task
        by_task = {}
        for q in items:
            by_task.setdefault(q["task"], []).append(q)
        per_task = max(1, limit // len(by_task))
        sampled = []
        for task, pool in sorted(by_task.items()):
            sampled.extend(pool[:per_task])
        items = sampled[:limit]

    return items


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING — multi-architecture support
# ═══════════════════════════════════════════════════════════════

def detect_family(model_name: str) -> str:
    """Detect model family from HF model name."""
    lower = model_name.lower()
    if "internvl" in lower:
        return "internvl"
    if "llava" in lower:
        return "llava"
    if "qwen" in lower:
        return "qwen"
    return "auto"


def load_model(model_name: str, dtype: str = "bfloat16"):
    """Load model + processor, auto-detecting family."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoTokenizer

    family = detect_family(model_name)
    print(f"Loading {model_name} (family={family})...")
    t0 = time.time()
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)

    if family == "internvl":
        # OpenGVLab/InternVL3-* uses custom code via trust_remote_code + its own .chat() API.
        # transformers 5.x defaults to meta-tensor init, which breaks InternVL3's custom
        # __init__ (torch.linspace(...).item() fails on meta). Force materialization on CPU.
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        with torch.device("cpu"):
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=False,
                trust_remote_code=True,
            )
        model = model.eval().cuda()
        print(f"  Loaded in {time.time()-t0:.1f}s; dtype={torch_dtype}, family={family}")
        return model, tokenizer, family

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s; dtype={torch_dtype}, family={family}")
    return model, processor, family


def run_inference(model, processor, family: str, qa: dict, image_path: Path | None) -> str:
    """Run single inference, dispatching by model family."""
    from PIL import Image
    import torch

    max_new = MAX_NEW_TOKENS.get(qa["task"], 64)

    if family == "internvl":
        return _run_internvl(model, processor, qa, image_path, max_new)
    elif family == "llava":
        return _run_llava(model, processor, qa, image_path, max_new)
    else:
        return _run_qwen(model, processor, qa, image_path, max_new)


def _run_qwen(model, processor, qa, image_path, max_new):
    from PIL import Image
    import torch

    content = []
    if image_path is not None:
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": qa["question"]})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = [Image.open(image_path).convert("RGB")] if image_path else None
    inputs = processor(
        text=[text], images=images, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, temperature=None, top_p=None,
        )
    generated = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


_INTERNVL_MEAN = (0.485, 0.456, 0.406)
_INTERNVL_STD = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size=448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_INTERNVL_MEAN, std=_INTERNVL_STD),
    ])


def _internvl_find_closest_aspect_ratio(ar, target_ratios, w, h, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = w * h
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(ar - target_ar)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best = ratio
    return best


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_w, orig_h = image.size
    ar = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tar = _internvl_find_closest_aspect_ratio(ar, target_ratios, orig_w, orig_h, image_size)
    target_w = image_size * tar[0]
    target_h = image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    for i in range(blocks):
        cols = target_w // image_size
        x0 = (i % cols) * image_size
        y0 = (i // cols) * image_size
        tiles.append(resized.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_load_image(image_path, input_size=448, max_num=12):
    from PIL import Image
    import torch
    image = Image.open(image_path).convert("RGB")
    transform = _internvl_build_transform(input_size)
    tiles = _internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values


def _run_internvl(model, tokenizer, qa, image_path, max_new):
    import torch

    question = f"{SYSTEM_PROMPT}\n\n{qa['question']}"
    gen_cfg = dict(max_new_tokens=max_new, do_sample=False)

    if image_path is not None:
        pixel_values = _internvl_load_image(image_path).to(torch.bfloat16).cuda()
        question = f"<image>\n{question}"
        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, question, gen_cfg)
    else:
        with torch.no_grad():
            response = model.chat(tokenizer, None, question, gen_cfg)
    return response.strip()


def _run_llava(model, processor, qa, image_path, max_new):
    from PIL import Image
    import torch

    if image_path is not None:
        question_text = f"{SYSTEM_PROMPT}\n\n{qa['question']}"
        content = [
            {"type": "image"},
            {"type": "text", "text": question_text},
        ]
        images = [Image.open(image_path).convert("RGB")]
    else:
        content = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{qa['question']}"}]
        images = None

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text], images=images, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, temperature=None, top_p=None,
        )
    generated = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Local VLM inference for ShipBench")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", required=True, help="predictions.jsonl path")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Filter by task IDs (e.g. A1 B1)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Subsample to N items (balanced per task)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already in output file")
    args = parser.parse_args()

    qa_items = load_qa_items(args.split, args.limit, args.tasks)
    print(f"Loaded {len(qa_items)} QA items (split={args.split}, limit={args.limit})")

    # Resume support
    done_ids = set()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["qa_id"])
        print(f"  Resuming: {len(done_ids)} already done")

    pending = [q for q in qa_items if q["qa_id"] not in done_ids]
    print(f"  Pending: {len(pending)}")

    # Load model
    model, processor, family = load_model(args.model, args.dtype)

    # Run
    t0 = time.time()
    with open(out_path, "a") as f:
        for i, qa in enumerate(pending):
            img = get_section_png(qa["candidate_id"], qa["ship_type"])
            # Task C may not need an image
            if qa["task"].startswith("C") and not qa.get("images"):
                img = None

            try:
                pred = run_inference(model, processor, family, qa, img)
            except Exception as e:
                pred = f"[ERROR: {e}]"

            record = {
                "qa_id": qa["qa_id"],
                "task": qa["task"],
                "ship_type": qa["ship_type"],
                "prediction": pred,
                "model": args.model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] {qa['task']} "
                      f"pred={pred[:50]!r}  "
                      f"rate={rate:.2f}/s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone: {len(pending)} items in {elapsed:.1f}s "
          f"({len(pending)/elapsed:.2f}/s)")
    print(f"Predictions → {out_path}")


if __name__ == "__main__":
    main()
