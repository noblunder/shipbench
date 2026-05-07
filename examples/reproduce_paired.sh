#!/bin/bash
# Reproduce paired Opus 4.7 ↔ gpt-5.5 paired-bootstrap analysis from the released predictions.
# Does NOT re-run inference; instead, uses prediction outputs released in the supplementary.
# Cost: $0. Time: <1 min.

set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Expects predictions/ folder (from supplementary or HF dataset)
PRED_DIR="${PRED_DIR:-$REPO_ROOT/predictions}"
if [ ! -d "$PRED_DIR" ]; then
    echo "ERROR: PRED_DIR=$PRED_DIR not found."
    echo "  Either: (a) download predictions from HF dataset, OR (b) re-run reproduce_table3.sh first."
    exit 1
fi

mkdir -p $REPO_ROOT/analysis_repro

echo "[1/3] Paired Opus ↔ GPT-5.5 (9-task, n=1607)"
python $REPO_ROOT/eval/paired_frontier_analysis.py \
  --opus $PRED_DIR/frontier/claude_opus_main.jsonl \
  --gpt5 $PRED_DIR/frontier/gpt-5.5_main_paired.jsonl \
  --output $REPO_ROOT/analysis_repro/paired_opus_gpt55.json \
  --name-a Opus --name-b GPT-5.5

echo "[2/3] Paired A1 v2 (section-only)"
python $REPO_ROOT/eval/paired_frontier_analysis.py \
  --opus $PRED_DIR/frontier/claude_opus_a1_section_only.jsonl \
  --gpt5 $PRED_DIR/frontier/gpt-5.5_a1_v2_paired.jsonl \
  --output $REPO_ROOT/analysis_repro/paired_opus_gpt55_a1_v2.json \
  --name-a Opus --name-b GPT-5.5

echo "[3/3] GPT-5.5 per-task MAIN/DIAGNOSTIC classification"
python $REPO_ROOT/eval/gpt5_diagnostic_analysis.py \
  --pred $PRED_DIR/frontier/gpt-5.5_main_paired.jsonl \
  --opus $PRED_DIR/frontier/claude_opus_main.jsonl \
  --output $REPO_ROOT/analysis_repro/gpt55_classification.json

echo ""
echo "✅ Done. Three JSON outputs in $REPO_ROOT/analysis_repro/"
echo "   Compare against the released analysis/ JSONs to verify reproducibility."
