#!/usr/bin/env python3
"""
A2 multi-view inference runner for ShipBench.

Loads section_png + (compart_png OR compart3d_png) per A2 item and queries
the model with both images. Output format compatible with eval_main.py.

Usage:
    python scripts/run_a2_inference.py \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --output results/qwen3vl8b_a2.jsonl \
        [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA_DIR = ROOT / "data" / "shipbench"
PROCESSED = ROOT / "data" / "processed"
HF_CACHE = "<SHIPBENCH_ROOT>/hf_cache"

SYSTEM_PROMPT = (
    "You are an expert naval architect assistant analyzing ship structural "
    "drawings. The user will show you TWO views of the same ship: a midship "
    "section (cross-section) and a compartment layout (longitudinal). Use "
    "BOTH views to answer the question. Answer concisely with just the "
    "requested value or letter."
)


def get_image_path(candidate_id: str, ship_type: str, key: str) -> Path | None:
    """Resolve image path for one of section_png / compart_png / compart3d_png."""
    folder = key  # folder name matches the key
    suffix = ""
    if key == "section_png":
        suffix = ""
    elif key == "compart_png":
        suffix = "_Compart"
    elif key == "compart3d_png":
        suffix = "_Compart3D"
    p = PROCESSED / ship_type / folder / f"{candidate_id}{suffix}.png"
    if p.exists():
        return p
    # fallback: try without suffix
    p2 = PROCESSED / ship_type / folder / f"{candidate_id}.png"
    return p2 if p2.exists() else None


def load_a2_items(split: str | None, limit: int | None, task_file: str | None = None):
    items = []
    a2_path = Path(task_file) if task_file else QA_DIR / "task_A2_sampled.jsonl"
    with open(a2_path) as f:
        for line in f:
            items.append(json.loads(line))
    if split:
        split_file = ROOT / "data" / "splits_v2" / f"{split}.jsonl"
        cids = set()
        with open(split_file) as f:
            for line in f:
                cids.add(json.loads(line)["candidate_id"])
        items = [q for q in items if q["candidate_id"] in cids]
    if limit:
        items = items[:limit]
    return items


def load_qwen(model_name: str, dtype: str, device_id: int = 0):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, cache_dir=HF_CACHE)
    if "Qwen3" in model_name:
        ModelCls = Qwen3VLForConditionalGeneration
    else:
        ModelCls = Qwen2_5_VLForConditionalGeneration
    model = ModelCls.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map={"": device_id},
        trust_remote_code=True, cache_dir=HF_CACHE,
    )
    model.eval()
    return model, processor


def load_internvl3(model_name: str, dtype: str, device_id: int = 0):
    import torch
    from transformers import AutoTokenizer, AutoModel
    import transformers.modeling_utils as _mu

    # transformers 5.x renamed _tied_weights_keys → all_tied_weights_keys;
    # InternVL3 trust_remote_code models still use the old private attribute.
    # Patch get_total_byte_count to handle the AttributeError gracefully.
    _orig_gtbc = _mu.get_total_byte_count
    def _patched_gtbc(model, *args, **kwargs):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return _orig_gtbc(model, *args, **kwargs)
    _mu.get_total_byte_count = _patched_gtbc

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=HF_CACHE,
    )
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map={"": device_id},
        trust_remote_code=True, cache_dir=HF_CACHE,
        low_cpu_mem_usage=True,
    )
    _mu.get_total_byte_count = _orig_gtbc  # restore
    model.eval()
    return model, tokenizer


def load_llavaov(model_name: str, dtype: str, device_id: int = 0):
    import torch
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=HF_CACHE,
    )
    # Use the llava-hf HuggingFace version (LlavaOnevisionForConditionalGeneration).
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map={"": device_id},
        trust_remote_code=True, cache_dir=HF_CACHE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def run_qwen_a2(model, processor, qa: dict, image_paths: list[Path], max_new: int) -> str:
    from PIL import Image
    import torch
    content = []
    for ip in image_paths:
        content.append({"type": "image", "image": str(ip)})
    content.append({"type": "text", "text": qa["question"]})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_paths:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None)
    generated = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _internvl3_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """InternVL3 official dynamic_preprocess — splits image into 1..max_num
    448x448 tiles based on aspect ratio, optionally adds a thumbnail.
    Reference: https://huggingface.co/OpenGVLab/InternVL3-8B
    """
    from PIL import Image
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    # Find best (n_w, n_h) tiling within [min_num, max_num] tiles
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # Pick aspect-ratio closest match
    best = min(target_ratios, key=lambda r: abs(r[0]/r[1] - aspect))
    n_w, n_h = best
    # Resize and split
    target_w, target_h = image_size * n_w, image_size * n_h
    resized = image.resize((target_w, target_h))
    tiles = []
    for j in range(n_h):
        for i in range(n_w):
            box = (i*image_size, j*image_size, (i+1)*image_size, (j+1)*image_size)
            tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def run_internvl3_a2(model, tokenizer, qa: dict, image_paths: list[Path], max_new: int) -> str:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    if image_paths:
        # Use dynamic patching for high-resolution inputs (scantling tables, dense text)
        all_tiles = []
        num_patches_list = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tiles = _internvl3_dynamic_preprocess(img, max_num=12)
            num_patches_list.append(len(tiles))
            for t in tiles:
                all_tiles.append(transform(t).unsqueeze(0))
        pixel_values = torch.cat(all_tiles, dim=0).to(torch.bfloat16).to(model.device)
        n_img = len(image_paths)
        img_tokens = "".join([f"Image-{i+1}: <image>\n" for i in range(n_img)])
        prompt = SYSTEM_PROMPT + "\n" + img_tokens + qa["question"]
        gen_cfg = dict(max_new_tokens=max_new, do_sample=False)
        response = model.chat(tokenizer, pixel_values, prompt, gen_cfg,
                              num_patches_list=num_patches_list)
    else:
        prompt = SYSTEM_PROMPT + "\n" + qa["question"]
        gen_cfg = dict(max_new_tokens=max_new, do_sample=False)
        response = model.chat(tokenizer, None, prompt, gen_cfg)
    return response.strip()


def run_llavaov_a2(model, processor, qa: dict, image_paths: list[Path], max_new: int) -> str:
    from PIL import Image
    import torch

    if image_paths:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": qa["question"]})
    else:
        images = None
        content = [{"type": "text", "text": qa["question"]}]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images is not None:
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None)
    generated = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _model_family(model_name: str) -> str:
    n = model_name.lower()
    if "internvl" in n:
        return "internvl3"
    if "llava" in n:
        return "llavaov"
    return "qwen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default=None, help="restrict to test/val split candidates")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--device", type=int, default=0, help="GPU device index")
    ap.add_argument("--task-file", default=None,
                    help="Override task JSONL path (e.g., for ablations)")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="max_new_tokens for generation (CoT probes need >256)")
    args = ap.parse_args()

    items = load_a2_items(args.split, args.limit, args.task_file)
    print(f"[DATA] A2 items: {len(items)}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["qa_id"])
        print(f"  Resuming: {len(done_ids)} already done")
    pending = [q for q in items if q["qa_id"] not in done_ids]
    print(f"  Pending: {len(pending)}")

    family = _model_family(args.model)
    print(f"[MODEL] Loading {args.model} (family={family}, device={args.device})...")
    if family == "internvl3":
        model, processor = load_internvl3(args.model, args.dtype, args.device)
    elif family == "llavaov":
        model, processor = load_llavaov(args.model, args.dtype, args.device)
    else:
        model, processor = load_qwen(args.model, args.dtype, args.device)
    print(f"[MODEL] Loaded.")

    t0 = time.time()
    with open(out_path, "a") as f:
        for i, qa in enumerate(pending):
            image_keys = qa.get("images", ["section_png"])
            paths = []
            for k in image_keys:
                p = get_image_path(qa["candidate_id"], qa["ship_type"], k)
                if p is None:
                    print(f"  [WARN] Missing {k} for {qa['qa_id']}")
                    continue
                paths.append(p)
            text_only = (len(image_keys) == 0)  # intentional text-only task
            if not paths and not text_only:
                pred = "[ERROR: no images]"
            else:
                try:
                    if family == "internvl3":
                        pred = run_internvl3_a2(model, processor, qa, paths, max_new=args.max_new_tokens)
                    elif family == "llavaov":
                        pred = run_llavaov_a2(model, processor, qa, paths, max_new=args.max_new_tokens)
                    else:
                        pred = run_qwen_a2(model, processor, qa, paths, max_new=args.max_new_tokens)
                except Exception as e:
                    pred = f"[ERROR: {e}]"
            record = {
                "qa_id": qa["qa_id"], "task": qa["task"], "subtask": qa.get("subtask"),
                "ship_type": qa["ship_type"], "prediction": pred, "model": args.model,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(pending) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(pending)}] {qa.get('subtask','A2')} "
                      f"pred={pred[:60]!r}  rate={rate:.2f}/s  eta={eta/60:.1f}min")

    print(f"[DONE] {len(pending)} predictions in {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
