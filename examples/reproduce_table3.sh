#!/bin/bash
# Reproduce paper Table 3 (zero-shot accuracy on 9 ShipBench sub-tasks).
# Estimated cost: ~$200 (Opus + gpt-5.5 frontier API). Time: ~2 hours.
# Prerequisites: SHIPBENCH_ROOT set, ANTHROPIC_API_KEY + OPENAI_API_KEY exported.

set -e

if [ -z "$SHIPBENCH_ROOT" ]; then
    echo "ERROR: SHIPBENCH_ROOT not set. Set to the dataset directory containing data/ and task_files/."
    exit 1
fi
[ -z "$ANTHROPIC_API_KEY" ] && { echo "ERROR: ANTHROPIC_API_KEY not set"; exit 1; }
[ -z "$OPENAI_API_KEY" ] && { echo "ERROR: OPENAI_API_KEY not set"; exit 1; }

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
mkdir -p $REPO_ROOT/predictions_repro $REPO_ROOT/analysis_repro $REPO_ROOT/repro

echo "[1/4] Anthropic Opus 4.7 inference (~30 min, ~$80)"
python $REPO_ROOT/inference/run_frontier_v3.py \
  --task-file $SHIPBENCH_ROOT/task_files/task_main_eval_opus_paired.jsonl \
  --output $REPO_ROOT/predictions_repro/claude_opus_main.jsonl \
  --n-per-task 200 --max-tokens 1024 --seed 42

echo "[2/4] OpenAI gpt-5.5 inference (~1.5 hours, ~$170)"
python $REPO_ROOT/inference/run_frontier_openai.py \
  --task-file $SHIPBENCH_ROOT/task_files/task_main_eval_opus_paired.jsonl \
  --output $REPO_ROOT/predictions_repro/gpt-5.5_main_paired.jsonl \
  --model gpt-5.5 --n-per-task 200 --max-tokens 8192 \
  --reasoning-effort medium --concurrency 16 --seed 42

echo "[3/4] Per-task MAIN/DIAGNOSTIC classification + paired analysis"
python $REPO_ROOT/eval/gpt5_diagnostic_analysis.py \
  --pred $REPO_ROOT/predictions_repro/gpt-5.5_main_paired.jsonl \
  --opus $REPO_ROOT/predictions_repro/claude_opus_main.jsonl \
  --output $REPO_ROOT/analysis_repro/gpt55_classification.json

python $REPO_ROOT/eval/paired_frontier_analysis.py \
  --opus $REPO_ROOT/predictions_repro/claude_opus_main.jsonl \
  --gpt5 $REPO_ROOT/predictions_repro/gpt-5.5_main_paired.jsonl \
  --output $REPO_ROOT/analysis_repro/paired_opus_gpt55.json \
  --name-a Opus --name-b GPT-5.5

echo "[4/4] Generate paper Table 3 LaTeX"
python $REPO_ROOT/eval/generate_paper_table_rows.py \
  --opus $REPO_ROOT/predictions_repro/claude_opus_main.jsonl \
  --gpt5 $REPO_ROOT/predictions_repro/gpt-5.5_main_paired.jsonl \
  --classifier $REPO_ROOT/analysis_repro/gpt55_classification.json \
  --out $REPO_ROOT/repro/table3_rows.tex

echo ""
echo "✅ Done. Paper Table 3 rows written to $REPO_ROOT/repro/table3_rows.tex"
echo "   Compare against paper Table 3 (Tab. 3 in main.tex)"
