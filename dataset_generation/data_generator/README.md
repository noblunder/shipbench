# Parametric Ship Generators (data_generator/)

The 6 ship-type generators that produce all ground-truth ShipBench candidates. These are the **source of truth** for every metadata field in the dataset.

## Files

| File | Purpose |
|---|---|
| `Tanker_Data_generation.py` | Tanker (MR/Aframax/Suezmax pooled) |
| `VLCC_Data_generation.py` | VLCC (Very Large Crude Carrier) |
| `BULKC_Data_generation.py` | Bulk Carrier |
| `CNTR_Data_generation.py` | Container ship |
| `LNGC_Data_generation.py` | LNG Carrier (membrane) |
| `LPGC_Data_generation.py` | LPG Carrier (Type A prismatic) |
| `_compart_hullform.py` | Shared compartment + hull-form helper |

## Usage

These generators are imported by `00_generate_candidates.py` (one directory up):

```python
GEN_DIR = ROOT / "data" / "data_generator"   # path expected by 00_generate_candidates.py
sys.path.insert(0, str(GEN_DIR))
from Tanker_Data_generation import generate_candidate as tanker_gen
# ...etc for each ship type
```

For reproduction, place this folder at `<repo_root>/data/data_generator/` so the dataset-pipeline scripts (`00_generate_candidates.py`, `02_rule_reeval.py`, `04_render_sections.py`, `06_borderline_inject.py`, `08_final_assembly.py`) can locate them via the canonical `ROOT/data/data_generator` path.

Alternatively, set the `SHIPBENCH_ROOT` environment variable and patch the `ROOT = Path("...")` line in each pipeline script.

## What each generator produces

A generator call returns:
- `candidate_id`: unique ID
- input dictionary (15–20 design parameters: length, beam, depth, segment layout, scantlings)
- recovered geometry block (per-member endpoints, polygon vertices, boundary edges)
- segment layout (compartment x-positions)

The dataset-pipeline scripts call generators in batch, save outputs as JSONL, render section + compartment PNGs (matplotlib), and build QA items.

## Reproducibility

- Generators are deterministic given a seed. ShipBench used `seed=42` for the published dataset.
- Generated candidates pass three rule-sanity layers (`01_rule_sanity.py`, `02_rule_reeval.py`) before being included.
- Ground truth for the nine released sub-tasks is computed entirely from the input dictionary and recovered geometry — no human annotation, no rule-citation labels.

## Note on rule-related code

Each generator contains an IACS CSR-H 2024 / KR / IMO IGC rule-registry block (`CSR_RULE_REGISTRY_*`, `KR_RULE_REGISTRY_*`, `make_csr_check_*`, `make_kr_check_*`) and writes auxiliary `kr_eval` / `kr_summary` fields into each per-candidate JSON. **None of these fields are used by any of the nine ShipBench sub-tasks released here** (A1–A2, B1–B4, C1–C3): all task ground truth is derived from the geometric / parametric metadata layers only. The rule-evaluation scaffolding is retained as forward-looking infrastructure for the rule-grounded extension listed in the paper's Future Work (IACS CSR / IMO IGC clause-citation tasks).

## License

MIT (see top-level `submission/full_paper/README.md`).
