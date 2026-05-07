# ShipBench Code (NeurIPS 2026 D&B Track)

This is the anonymous code release for the ShipBench paper, hosted at https://anonymous.4open.science/status/shipbench2026.

## Structure

```
code/
├── README.md                       # this file
├── requirements.txt                # Python dependencies
├── inference/                      # 5 model-inference scripts
│   ├── run_frontier_v3.py          # Anthropic Claude Opus 4.7
│   ├── run_frontier_openai.py      # OpenAI gpt-5.5 (concurrent)
│   ├── run_frontier_gemini.py      # Google Gemini (preserved for reproducibility)
│   ├── run_vlm_inference.py        # primary open-weight VLM inference (HF)
│   └── run_a2_inference.py         # multi-view A2 inference (HF)
├── eval/                           # 6 evaluation + analysis scripts
│   ├── eval_main.py                # main 9-task evaluator
│   ├── eval_v3_unit_aware.py       # v3 prompt-clarification unit-aware grader
│   ├── paired_frontier_analysis.py # paired bootstrap + McNemar + Clopper-Pearson
│   ├── gpt5_diagnostic_analysis.py # MAIN/DIAGNOSTIC classification (20% threshold, pre-specified)
│   ├── generate_paper_table_rows.py # generate paper Table 3 LaTeX
│   └── validate_predictions.py     # quality validation (refusal/parse_fail/OOR)
├── dataset_generation/             # dataset construction pipeline (00–12 + 7 generators)
│   ├── 00_generate_candidates.py
│   ├── 01_rule_sanity.py
│   ├── 02_rule_reeval.py
│   ├── 03_stratify_candidates.py
│   ├── 04_render_sections.py
│   ├── 05_fps_select.py
│   ├── 06_borderline_inject.py
│   ├── 07_topology_label.py
│   ├── 08_final_assembly.py
│   ├── 09_make_splits.py
│   ├── 10_build_qa.py
│   ├── 11_build_task_c.py
│   ├── 12_build_task_a2.py
│   └── data_generator/             # 6 ship-type parametric generators + helper
│       ├── BULKC_Data_generation.py
│       ├── CNTR_Data_generation.py
│       ├── LNGC_Data_generation.py
│       ├── LPGC_Data_generation.py
│       ├── Tanker_Data_generation.py
│       ├── VLCC_Data_generation.py
│       ├── _compart_hullform.py
│       └── README.md
└── examples/
    ├── reproduce_table3.sh         # one-command reproduction of paper Table 3
    └── reproduce_paired.sh         # one-command paired Opus↔gpt-5.5 analysis
```

## Setup

```bash
pip install -r requirements.txt

# Set API keys (Anthropic + OpenAI)
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...

# Set repo root
export SHIPBENCH_ROOT=/path/to/this/repo
```

## Quick start: reproduce paper Table 3

```bash
cd examples
bash reproduce_table3.sh
```

This will:
1. Run frontier inference (Opus 4.7 + gpt-5.5, ~2 hours, ~$200)
2. Run paired analysis (paired bootstrap + McNemar + Clopper-Pearson)
3. Generate Table 3 LaTeX rows

## Re-running individual stages

### 1. Run frontier inference

```bash
# Anthropic Opus 4.7 (existing main + v3 ablation)
python inference/run_frontier_v3.py \
  --task-file <SHIPBENCH_ROOT>/data/task_main_eval_opus_paired.jsonl \
  --output predictions_repro/claude_opus_main.jsonl \
  --n-per-task 200 --max-tokens 1024 --seed 42

# OpenAI gpt-5.5 (concurrency=16)
python inference/run_frontier_openai.py \
  --task-file <SHIPBENCH_ROOT>/data/task_main_eval_opus_paired.jsonl \
  --output predictions_repro/gpt-5.5_main_paired.jsonl \
  --model gpt-5.5 --n-per-task 200 --max-tokens 8192 \
  --reasoning-effort medium --concurrency 16 --seed 42

# Open-weight VLMs (HuggingFace, e.g., Qwen3-VL-8B)
python inference/run_vlm_inference.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --output predictions_repro/qwen3vl_main.jsonl \
  --split test
```

### 2. Run paired analysis

```bash
python eval/paired_frontier_analysis.py \
  --opus predictions_repro/claude_opus_main.jsonl \
  --gpt5 predictions_repro/gpt-5.5_main_paired.jsonl \
  --output analysis_repro/paired_opus_gpt55.json \
  --name-a Opus --name-b GPT-5.5

# GPT-5.5 per-task MAIN/DIAGNOSTIC classification
python eval/gpt5_diagnostic_analysis.py \
  --pred predictions_repro/gpt-5.5_main_paired.jsonl \
  --opus predictions_repro/claude_opus_main.jsonl \
  --output analysis_repro/gpt55_classification.json
```

### 3. Generate paper Table 3 LaTeX

```bash
python eval/generate_paper_table_rows.py \
  --opus predictions_repro/claude_opus_main.jsonl \
  --gpt5 predictions_repro/gpt-5.5_main_paired.jsonl \
  --classifier analysis_repro/gpt55_classification.json \
  --out repro/table3_rows.tex
```

## Reproducing the dataset

```bash
cd dataset_generation
python 00_generate_candidates.py --seed 42 --output <SHIPBENCH_ROOT>/data/candidates.jsonl
# ... (continue with 01_*.py through 12_*.py)
```

The 7 ship-type generators in `data_generator/` are deterministic given `seed=42`; the released ShipBench dataset can be exactly reproduced from these scripts.

## Models / API endpoints

| Vendor | Model | Released | API |
|---|---|---|---|
| Anthropic | `claude-opus-4-7` | 2026-04-14 | `messages.create` |
| OpenAI | `gpt-5.5` | 2026-04-23 | `chat.completions.create` (reasoning_effort=medium, image detail=high) |
| HuggingFace | `Qwen/Qwen3-VL-8B-Instruct@0c351dd` | (pinned commit) | local, bf16 |
| HuggingFace | `Qwen/Qwen2.5-VL-7B-Instruct@cc59489` | (pinned) | local |
| HuggingFace | `OpenGVLab/InternVL3-8B@853e3a7` | (pinned) | local |
| HuggingFace | `lmms-lab/llava-onevision-qwen2-7b-ov@0d50680` | (pinned) | local |

## Statistical methods

- **Paired bootstrap CI**: 1,000 resamples, `seed=42`
- **McNemar test**: auto-selects exact binomial when `b+c<25`, chi² with continuity correction otherwise
- **Matched-pair OR**: `b/c` paired-aware effect size
- **Clopper-Pearson exact CI**: per-task accuracy (especially for sparse-success cells)
- **Wilson score CI**: refusal-rate CIs
- **Holm-Bonferroni**: multiple-test correction for the 9-task secondary analyses

## Pre-specified analyses

The 20% bad-output threshold for GPT-5.5 MAIN/DIAGNOSTIC classification is hardcoded as `REFUSAL_THRESHOLD = 0.20` in `eval/gpt5_diagnostic_analysis.py`, **committed before main inference launch**. Justification (paper §F8): "above this level, accuracy is dominated by interface compliance rather than task performance."

## License

MIT. See LICENSE file.
