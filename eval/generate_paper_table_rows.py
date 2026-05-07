#!/usr/bin/env python3
"""Generate LaTeX rows for Table 3 (zero-shot) integrating 3-vendor frontier results.

Reads:
  - Opus existing predictions (claude_opus_main.jsonl)
  - GPT-5.5 paired predictions (gpt-5.5_main_paired.jsonl)
  - Gemini 3.1 Pro Preview predictions (gemini-3.1-pro-preview_main.jsonl)
  - GPT-5 per-task classification JSON (gpt5_per_task_classification.json)

Emits:
  - LaTeX rows for Table 3 (Opus, Gemini, GPT-5.5 with em-dash for DIAGNOSTIC tasks)
  - Updated caption with 3-vendor parity disclosure + 20% threshold pre-specification
  - Appendix table for GPT-5.5 Pitfall 10 diagnostic
  - Per-task n disclosure footnote

Usage:
  python scripts/generate_paper_table_rows.py \
      --opus outputs/frontier_eval/claude_opus_main.jsonl \
      --gpt5 outputs/frontier_eval/gpt-5.5_main_paired.jsonl \
      --gemini outputs/frontier_eval/gemini-3.1-pro-preview_main.jsonl \
      --classifier outputs/main_eval/gpt5_per_task_classification.json \
      --out outputs/main_eval/table3_rows.tex
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

NUM_UNIT_RE = re.compile(
    r"(-?\d+(?:[\.,]\d+)?(?:\s*[eE][+-]?\d+)?)\s*"
    r"(m\^[23]|m[²³]|m[23]|mm\^[23]|mm[²³]|mm[23]|mm|m|cm\^[23]|cm[²³]|cm[23])?"
)
NUMERIC_TASKS = {
    "B1_plate_thickness", "B2_stiffener_size",
    "B3_cargo_capacity_v1", "B3_cargo_capacity_v3",
    "B4_section_area_v1", "B4_section_area_v3_cot",
    "C3_bulkhead_position", "C3_bulkhead_position_v3",
}
MCQ_TASKS = {"A1_shiptype", "A1_shiptype_section_only", "A2_stiffener_type",
             "C1_compartment_locate", "C2_compartment_boundary"}
TOLERANCE_PCT = {
    "B1_plate_thickness": 5.0, "B2_stiffener_size": 5.0,
    "B3_cargo_capacity_v1": 10.0, "B4_section_area_v1": 10.0,
    "C3_bulkhead_position": 10.0,
}

# Column order matching tab:zeroshot
TASK_COLUMN_ORDER = [
    "A1_shiptype", "A2_stiffener_type",
    "B1_plate_thickness", "B2_stiffener_size",
    "B3_cargo_capacity_v1", "B4_section_area_v1",
    "C1_compartment_locate", "C2_compartment_boundary", "C3_bulkhead_position",
]


def extract_value(s: str):
    s = (s or "").strip()
    matches = list(NUM_UNIT_RE.finditer(s))
    last = next((m for m in reversed(matches) if m.group(2)), None) or (matches[-1] if matches else None)
    if not last:
        return None
    try:
        return float(last.group(1).replace(",", ""))
    except ValueError:
        return None


def grade_pred(pred: str, gt_item: dict):
    task = gt_item.get("task")
    pred = (pred or "").strip()
    if task in MCQ_TASKS:
        m = re.search(r"\b([A-F])\b", pred.upper())
        if not m:
            return None
        return int(m.group(1) == gt_item["answer"].strip().upper())
    if task in NUMERIC_TASKS:
        v = extract_value(pred)
        if v is None:
            return None
        gt_val = float(gt_item["metadata"]["value"])
        tol = gt_item["metadata"].get("tolerance_pct", TOLERANCE_PCT.get(task, 10.0))
        rel_err = abs(v - gt_val) / max(abs(gt_val), 1e-9)
        return int(rel_err <= tol / 100.0)
    return None


def clopper_pearson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0.0, 1.0)
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    except ImportError:
        # Wilson fallback
        z = 1.96
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denom
        margin = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
        lo, hi = max(0, center - margin), min(1, center + margin)
    return lo, hi


def compute_accuracy(pred_file: str, gt_by_id: dict) -> dict:
    """Per-task accuracy on a prediction JSONL."""
    if not Path(pred_file).exists():
        return {}
    by_task = defaultdict(list)  # task → list of 1/0
    n_pred_per_task = defaultdict(int)
    for line in open(pred_file):
        p = json.loads(line)
        task = p["task"]
        n_pred_per_task[task] += 1
        gt = gt_by_id.get(p["qa_id"])
        if gt is None:
            continue
        g = grade_pred(p["prediction"], gt)
        if g is not None:
            by_task[task].append(g)
    out = {}
    for task, hits in by_task.items():
        n = len(hits)
        k = sum(hits)
        acc = k / n * 100 if n else 0.0
        lo, hi = clopper_pearson_ci(k, n)
        out[task] = {
            "n": n,
            "n_predicted": n_pred_per_task[task],
            "n_correct": k,
            "accuracy_pct": round(acc, 2),
            "cp_95ci": [round(lo * 100, 2), round(hi * 100, 2)],
        }
    return out


def fmt_cell(value, is_dash=False, bold=False, italic=False):
    """Format a single LaTeX cell. Em-dash for diagnostic, optional bold/italic."""
    if is_dash:
        return "---"
    if value is None:
        return "n/a"
    s = f"{value:.1f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    if italic:
        s = f"\\emph{{{s}}}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opus", required=True)
    ap.add_argument("--gpt5", required=True)
    ap.add_argument("--gemini", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--gt-files", nargs="*", default=[
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_main_eval.jsonl",
        "<SHIPBENCH_ROOT>/data/shipbench3d_v2/task_main_eval_opus_paired.jsonl",
    ])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Load GT
    gt_by_id = {}
    for f in args.gt_files:
        if Path(f).exists():
            for line in open(f):
                d = json.loads(line)
                gt_by_id.setdefault(d["qa_id"], d)

    # Compute accuracies
    opus_acc = compute_accuracy(args.opus, gt_by_id)
    gpt_acc = compute_accuracy(args.gpt5, gt_by_id)
    gem_acc = compute_accuracy(args.gemini, gt_by_id)

    # Load classifier output for GPT-5 per-task MAIN/DIAGNOSTIC
    classifier = {}
    if Path(args.classifier).exists():
        classifier = json.loads(Path(args.classifier).read_text()).get("per_task", {})

    # Build LaTeX rows
    lines = []
    lines.append("% =================================================================")
    lines.append("% Auto-generated 3-vendor frontier rows for tab:zeroshot")
    lines.append("% Generated by scripts/generate_paper_table_rows.py")
    lines.append("% =================================================================")
    lines.append("")
    lines.append("% Replace the existing single Opus row with these 3 frontier rows:")
    lines.append("")

    def build_row(name_label, acc_dict, role_dict=None, footnote_marker=""):
        cells = []
        for task in TASK_COLUMN_ORDER:
            if role_dict is not None:
                # GPT-5.5: use main_table_cell (em-dash for DIAGNOSTIC)
                role_info = role_dict.get(task)
                if role_info and role_info.get("role") == "DIAGNOSTIC":
                    cells.append("---")
                    continue
            if task in acc_dict:
                cells.append(fmt_cell(acc_dict[task]["accuracy_pct"]))
            else:
                cells.append("n/a")
        return f"\\emph{{{name_label}}}{footnote_marker} & " + " & ".join(cells) + " \\\\"

    # Opus row
    lines.append(build_row("Claude Opus 4.7", opus_acc, footnote_marker="$^c$"))
    # Gemini row
    lines.append(build_row("Gemini 3.1 Pro Preview", gem_acc, footnote_marker="$^f$"))
    # GPT-5.5 row (with DIAGNOSTIC dashes)
    lines.append(build_row("GPT-5.5", gpt_acc, role_dict=classifier, footnote_marker="$^g$"))

    lines.append("")
    lines.append("% =================================================================")
    lines.append("% Updated caption (replace existing tab:zeroshot caption):")
    lines.append("% =================================================================")
    lines.append("""
% \\caption{Zero-shot accuracy (\\,\\%) on all nine ShipBench sub-tasks
% ($n{=}594$ per cell for open-weight; 95\\,\\% bootstrap CI in App.~\\ref{app:results_bar}).
% Tolerances: B1/B2 $\\pm$5\\,\\%; B3/B4/C3 $\\pm$10\\,\\%; MCQ raw accuracy.
% \\,$^a$~Ship-type prior. \\,$^b$~Metadata oracle.
% \\,$^c$~Claude Opus 4.7 (released 2026-04-14) at $n{=}200$ per task.
% \\,$^f$~Gemini 3.1 Pro Preview (released 2026-05) at $n{=}200$ per task; preview-tier
%        offers full inference capability and is the closest contemporary match to Opus 4.7.
% \\,$^g$~GPT-5.5 (released 2026-04-23) at $n{=}200$ per task. We pre-specify a 20\\,\\% bad-output
%        threshold: above this level, accuracy is dominated by interface compliance rather than
%        task performance, so the task is reported diagnostically (Appendix~\\ref{app:gpt55_diagnostic})
%        rather than as a main accuracy estimate. Tasks marked '---' for GPT-5.5 are diagnostic-only.
% \\,$^d$~A1-stype is the section-only (v2) reformulation. \\,$^e$~Opus on A1-v2 (section-only).
% Three frontier vendors selected at submission time (2026-05-07), all latest flagships within
% a 3-week release window. ...}""")

    # Appendix diagnostic table
    lines.append("")
    lines.append("% =================================================================")
    lines.append("% APPENDIX — GPT-5.5 Pitfall 10 diagnostic table")
    lines.append("% =================================================================")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{GPT-5.5 per-task outcome diagnostic ($n{=}200$ per task, "
                 "reasoning\\_effort=medium, detail=high, identical prompts to Opus 4.7). "
                 "Tasks with bad-output rate $\\leq 20\\,\\%$ (refusal + empty + parse-fail + "
                 "out-of-range) are included in Table~\\ref{tab:zeroshot}; remaining tasks "
                 "(marked '---' in main table) are characterized here as Pitfall 10 evidence.}")
    lines.append("\\label{tab:gpt55_diagnostic}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Task & $n$ & Refusal\\,\\% (95\\,\\% CI) & Empty\\,\\% & Parse-fail\\,\\% & "
                 "OOR\\,\\% & avg reasoning tok \\\\")
    lines.append("\\midrule")
    for task in TASK_COLUMN_ORDER:
        info = classifier.get(task)
        if info is None:
            continue
        rates = info["rates_pct"]
        wci = info["wilson_95ci_pct"]["refusal"]
        n = info["n"]
        avg_rt = info.get("tokens", {}).get("avg_reasoning") or 0
        ref_str = f"{rates['refusal']:.1f} [{wci[0]:.1f}, {wci[1]:.1f}]"
        empty = info["outcomes"]["empty"] / n * 100 if n else 0
        pf = info["outcomes"]["parse_fail"] / n * 100 if n else 0
        oor = info["outcomes"]["oor"] / n * 100 if n else 0
        marker = "$^*$" if info["role"] == "DIAGNOSTIC" else ""
        task_short = task.replace("_", "-")
        lines.append(f"{task_short}{marker} & {n} & {ref_str} & {empty:.1f} & {pf:.1f} & {oor:.1f} & {avg_rt:.0f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\\\$^*$ tasks excluded from Table~\\ref{tab:zeroshot} (DIAGNOSTIC, marked '---').")
    lines.append("\\end{table}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Wrote LaTeX integration to: {out_path}")
    print()
    print("=== Quick summary ===")
    for vendor, accs in [("Opus 4.7", opus_acc), ("Gemini 3.1", gem_acc), ("GPT-5.5", gpt_acc)]:
        print(f"\n{vendor}:")
        for task in TASK_COLUMN_ORDER:
            d = accs.get(task)
            if d is None:
                print(f"  {task:30s}: n/a")
            else:
                role = ""
                if vendor == "GPT-5.5":
                    info = classifier.get(task)
                    if info:
                        role = f"  [{info['role']}]"
                print(f"  {task:30s}: {d['accuracy_pct']:5.1f}%  n={d['n']}{role}")

    # Also list summary of MAIN vs DIAGNOSTIC for GPT-5.5
    if classifier:
        main_t = [t for t, v in classifier.items() if v.get("role") == "MAIN"]
        diag_t = [t for t, v in classifier.items() if v.get("role") == "DIAGNOSTIC"]
        print(f"\n\nGPT-5.5 classification summary:")
        print(f"  MAIN ({len(main_t)}): {main_t}")
        print(f"  DIAGNOSTIC ({len(diag_t)}): {diag_t}")


if __name__ == "__main__":
    main()
