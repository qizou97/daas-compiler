# daas-compiler Stage-Based Redesign

**Date:** 2026-05-18  
**Status:** Approved  
**Approach:** A — Hard cut: extract_sample.py becomes pure

---

## 1. Motivation

The current `extract_sample.py` embeds all biological filtering logic inline
(biological policy resolution, nucleus-boundary masking, `filtered_table`
auto-detection). This makes it impossible to:

- Add upstream tissue or nucleus-overlap filtering without touching the
  extraction script.
- Compose filters independently (e.g. tissue-only, or overlap-only).
- Test filtering in isolation.
- Plan a multi-stage pipeline from a natural-language request.

The redesign separates **filtering** (writes new table keys back into zarr)
from **extraction** (reads the final table key and writes shards).

---

## 2. Core Principle

> **Filtering stages own table keys. Extraction consumes the final table key.**

`extract_sample.py` accepts `--table-key <FINAL_KEY>` and trusts it is correct.
It never auto-detects `filtered_table`, never applies a biological policy.
All cell selection happens upstream in stage scripts.

`filtered_table` is treated as an optional pre-existing shortcut (the user may
pass `--input-table-key filtered_table` to a stage script), not as a required
canonical name.

---

## 3. File Structure

### New files

```
scripts/
  inspect_spatialdata.py          # Stage 0: print zarr keys / shapes / tables
  filter_tissue.py                # Stage: tissue-inside filter (SOPA if needed)
  filter_nucleus_presence.py      # Stage: keep cells with nucleus_boundaries entry
  filter_nucleus_overlap.py       # Stage: Xenium-vs-HE nucleus overlap score

daas/
  planning.py                     # NL → StagePlan dataclass → CLI command strings
  reports.py                      # StageReport dataclass + write_stage_report()
  filters/
    __init__.py
    tissue.py                     # filter_by_tissue(sdata, ...) → (keep_mask, StageReport)
    nucleus_presence.py           # filter_by_nucleus_presence(sdata, ...) → (keep_mask, StageReport)
    nucleus_overlap.py            # filter_by_nucleus_overlap(sdata, ...) → (keep_mask, StageReport)

references/
  workflow-planning.md            # How to read an NL request and build a stage plan
  filtering-recipes.md            # Per-recipe trigger phrases, inputs, outputs
  table-key-contract.md           # Table-key naming convention and propagation rules
  sopa-integration.md             # SOPA API calls for tissue + HE nucleus segmentation

tests/
  test_planning.py                # Unit tests for daas.planning
  test_filter_nucleus_presence.py # Unit tests for nucleus_presence filter
```

### Modified files

| File | Change |
|------|--------|
| `scripts/extract_sample.py` | Remove `--biological-filter-policy`, `--filtered-table-key`, `--nucleus-boundaries-key`; remove all `resolve_biological_policy` calls. Only `--table-key` + `--shapes-key` remain for cell selection. |
| `daas/filtering.py` | Remove `BiologicalPolicy`, `resolve_biological_policy`, `mask_by_nucleus_boundaries`, `BiologicalResolution`. Keep patch policy, alignment, centroid, report helpers. |
| `daas/cli_args.py` | Remove biological policy args from `build_extract_sample_parser`. |
| `scripts/compile_dataset.py` | Add `--samples A_001,A_002,A_004` optional flag; when given, compile only named sample subdirs. |
| `scripts/extract_all.py` | Remove `--biological-filter-policy`, `--filtered-table-key`, `--nucleus-boundaries-key` from both `parse_args()` and the `extra` list forwarded to `extract_sample.py`. No other structural changes needed. |
| `SKILL.md` | Full rewrite: stage-based workflow, recipe selection rules, table-key propagation, worked example. |
| `references/filtering.md` | Update to reflect that biological policies are now stage scripts, not extract flags. |

---

## 4. Stage Scripts — Interface

Every filter stage script follows this CLI:

```bash
python3 scripts/filter_<recipe>.py \
    --zarr            <path.zarr>              # required
    --input-table-key <key>                    # default: "table"
    --input-shape-key <key>                    # default: "cell_circles"
    --output-table-key <key>                   # default: auto-named (see §4.1)
    --report-dir      <dir>                    # default: <zarr>/../filter_reports/
```

For nucleus-overlap only:
```bash
    --overlap-threshold 0.5                    # default: 0.5
    --he-nucleus-key    he_nucleus_boundaries  # default; created by SOPA if absent
```

### 4.1 Auto-naming for output table keys

| Script | Input key | Auto-output key |
|--------|-----------|-----------------|
| `filter_tissue.py` | `table` | `table_tissue` |
| `filter_nucleus_presence.py` | `table_tissue` | `table_tissue_nucleus` |
| `filter_nucleus_overlap.py` | `table_tissue_nucleus` | `table_tissue_nucleus_he` |
| (any) | `<key>` | `<key>_<stage_suffix>` |

Stage suffixes: `tissue`, `nucleus`, `he`.

### 4.2 What each script does

**`filter_tissue.py` (tissue_inside recipe)**
1. Check `sdata.shapes` for an existing tissue polygon (key `"tissue_boundaries"` or similar).
2. If absent: call `sopa.segmentation.tissue(sdata)` to create it.
3. Spatially filter: keep cells whose centroid lies inside any tissue polygon.
4. Write filtered table: `sdata.write_element(output_table_key, filtered_adata)`.
5. Write `StageReport` JSON to `--report-dir`.

**`filter_nucleus_presence.py` (nucleus_presence recipe)**
1. Load `sdata.tables[input_table_key]` and `sdata.shapes[nucleus_boundaries_key]`.
2. Keep rows where `obs["cell_id"]` is in `sdata.shapes[nucleus_boundaries_key].index`.
3. Write filtered table + `StageReport` JSON.

**`filter_nucleus_overlap.py` (xenium_he_nucleus_overlap recipe)**
1. Check `sdata.shapes` for `he_nucleus_boundaries`.
2. If absent: call `sopa.segmentation.cellpose(...)` on HE image to create it.
3. Compute per-cell overlap score: intersection-over-union between Xenium
   `nucleus_boundaries` and nearest HE nucleus.
4. Keep cells where `overlap_score >= --overlap-threshold`.
5. Write filtered table + `StageReport` JSON (includes per-cell overlap scores summary).

### 4.3 SOPA integration

Both SOPA calls are wrapped in try/except with a clear error if `sopa` is not installed:
```
ImportError: sopa is required for tissue/HE-nucleus segmentation.
Run: pip install sopa
```
Reference: `references/sopa-integration.md`.

---

## 5. StageReport (daas/reports.py)

```python
@dataclass
class StageReport:
    stage: str                        # recipe name
    zarr_path: str
    input_table_key: str
    output_table_key: str
    input_shape_key: str
    output_shape_key: str             # same as input unless stage creates a new shape
    n_cells_in: int
    n_cells_out: int
    drop_counts_by_reason: dict       # {"missing_nucleus_boundary": N, ...}
    report_path: str
    warnings: list[str]

def write_stage_report(report: StageReport, report_dir: Path) -> Path:
    """Write report as JSON; return path."""
```

Report filename convention: `{stage}_{input_table_key}.json`
(e.g. `nucleus_presence_table_tissue.json`).

---

## 6. Planner (daas/planning.py)

### 6.1 Public API

```python
@dataclass
class StageSpec:
    name: str              # "tissue_inside" | "nucleus_presence" | "xenium_he_nucleus_overlap"
    script: str            # relative path to stage script
    input_table_key: str
    output_table_key: str

@dataclass
class StagePlan:
    stages: list[StageSpec]
    extract_args: dict     # --table-key, --shapes-key, --extract-mode, --mpp, --patch-size, --n-sample, ...
    compile_args: dict     # --samples, --bundle-wds, --shard-size
    final_table_key: str   # = stages[-1].output_table_key if stages else extract_args["table_key"]

def parse_stage_plan(text: str, base_table_key: str = "table",
                     base_shapes_key: str = "cell_circles", **kwargs) -> StagePlan:
    """Map natural-language request to StagePlan. Pure Python, no I/O."""

def render_cli(plan: StagePlan, zarr_paths: list[str],
               output_dir: str, skill_dir: str = "${SKILL_DIR}") -> str:
    """Return a shell script string of ordered CLI commands to run."""
```

### 6.2 NL normalization

**Filter recipe triggers (keyword matching, case-insensitive):**

| Trigger phrases | Stage |
|---|---|
| "inside tissue", "out of tissue", "outside tissue", "tissue filter" | `tissue_inside` |
| "with nucleus boundaries", "has nucleus", "only cells with nucleus", "nucleus boundary" | `nucleus_presence` |
| "xenium nucleus overlaps he nucleus", "he nucleus", "nucleus overlap", "overlap >" | `xenium_he_nucleus_overlap` |

**Extract-arg normalization:**

| Trigger | Resolved arg |
|---|---|
| "optim_ops_level", "ops_level", "optimized ops level", "full_ops_level" | `extract_mode = "full_ops_level"` |
| "sample N cells", "N cells per sample", "sample N" | `n_sample = N` |
| "mpp=N", "target mpp N", "mpp N" | `mpp = N` |
| "patch size N", "N×N patches" | `patch_size = N` |

**Stage ordering:** always `tissue_inside` → `nucleus_presence` → `xenium_he_nucleus_overlap`.
The planner never reorders; if multiple triggers fire, all matching stages are added
in that canonical order.

### 6.3 Table-key propagation

The planner chains `output_table_key` of stage N to `input_table_key` of stage N+1.
The final `extract_args["table_key"]` = last stage's `output_table_key`.
If no stages are added, `table_key` = `base_table_key`.

---

## 7. extract_sample.py after refactor

**Removed args:** `--biological-filter-policy`, `--filtered-table-key`, `--nucleus-boundaries-key`.

**Retained args:** `--table-key`, `--shapes-key`, `--extract-mode`, `--mpp`, `--patch-size`,
`--n-sample`, `--shard-size`, `--seed`, `--image-key`, `--patch-filter-policy`,
`--cell-id-column`, `--filter-report-name`.

**Execution flow (unchanged except for removal of Layer 1):**
1. Load zarr.
2. Load `sdata.tables[table_key]` directly — no auto-detection.
3. `resolve_table_shape_alignment(adata, gdf)`.
4. MPP derivation.
5. Patch-validity filter (Layer 2, unchanged).
6. Sampling.
7. Filter report (no biological_policy fields; `source_table_key` is just `table_key`).
8. Spatial sort → TileSpec → pre-flight viz → extract → shards.
9. h5ad + manifest.parquet + validation.

**`filter_report.json` fields removed:** `biological_policy_requested`,
`biological_policy_applied`. Field `source_table_key` still present.

---

## 8. compile_dataset.py changes

Add `--samples` optional flag:

```bash
python3 scripts/compile_dataset.py \
    --per-sample-dir /data/out \
    --output         /data/compiled \
    --samples        A_001,A_002,A_004 \
    [--bundle-wds] [--shard-size 500]
```

When `--samples` is provided, only subdirs matching the listed names are included.
Other subdirs are silently skipped (not an error). If a named sample is missing,
exit 1 with a clear message naming the missing sample.

---

## 9. Testing

### Integration test data (skip if absent)
```
/home/zouqi/datasets/mash/spatialdata/A_001.zarr
/home/zouqi/datasets/mash/spatialdata/A_002.zarr
```

Tests using these paths are decorated:
```python
A001 = Path("/home/zouqi/datasets/mash/spatialdata/A_001.zarr")
pytestmark = pytest.mark.skipif(not A001.exists(), reason="test data not present")
```

### tests/test_planning.py

| Test | Assertion |
|------|-----------|
| `test_tissue_inside_phrase` | "filter out cells outside tissue" → `stages[0].name == "tissue_inside"` |
| `test_nucleus_boundary_phrase` | "only keep cells with nucleus boundaries" → `stages[0].name == "nucleus_presence"` |
| `test_combined_tissue_and_nucleus` | "outside tissue + nucleus boundaries" → `[tissue_inside, nucleus_presence]` |
| `test_xenium_he_overlap_phrase` | "Xenium nucleus and HE nucleus overlap > 0.5" → `[xenium_he_nucleus_overlap]` |
| `test_optim_ops_level_alias` | "optim_ops_level" → `plan.extract_args["extract_mode"] == "full_ops_level"` |
| `test_n_sample_parsed` | "sample 3000 cells per sample" → `plan.extract_args["n_sample"] == 3000` |
| `test_table_key_propagation_two_stages` | tissue → nucleus → extract sees `table_tissue_nucleus` |
| `test_table_key_propagation_three_stages` | all three stages chained correctly |
| `test_render_cli_contains_final_table_key` | rendered string contains `--table-key table_tissue_nucleus` |
| `test_compile_samples_flag` | `plan.compile_args["samples"] == ["A_001","A_002","A_004"]` |
| `test_no_stages_uses_base_table_key` | empty request with base_table_key="filtered_table" → `final_table_key == "filtered_table"` |

### tests/test_filter_nucleus_presence.py

| Test | Assertion |
|------|-----------|
| `test_filter_keeps_cells_with_nucleus` | 5 cells, 3 have nucleus boundaries → `n_cells_out == 3` |
| `test_stage_report_fields` | `drop_counts_by_reason["missing_nucleus_boundary"] == 2`, report_path set |
| `test_empty_nucleus_boundaries_raises` | raises `ValueError` with message |
| `test_output_table_key_written_to_sdata` | in-memory sdata has the output key after filter |

### tests/test_compile.py additions

| Test | Assertion |
|------|-----------|
| `test_compile_samples_flag_filters_dirs` | 3 sample dirs present, `--samples A_001,A_003` → compiled h5ad has only those 2 |
| `test_compile_missing_sample_exits_1` | `--samples A_001,A_999` when A_999 absent → SystemExit |

### Verification commands
```bash
python -m compileall skills/daas-compiler
cd skills/daas-compiler && pytest tests -q
```

---

## 10. Worked Example

**Natural-language request:**
> "Process A_001,A_002,A_004 under /home/zouqi/datasets/mash/spatialdata into
> cell-centered HE patches. Filter out cells outside tissue and only keep cells
> with nucleus boundaries. Target mpp=0.5, patch size=224, use optim_ops_level,
> output to /home/zouqi/datasets/mash/stvisuome, sample 3000 cells per sample,
> compile, and write bundled WebDataset shards."

**`parse_stage_plan` resolves to:**
- stages: `[tissue_inside, nucleus_presence]`
- `extract_args`: `{table_key: "table_tissue_nucleus", extract_mode: "full_ops_level", mpp: 0.5, patch_size: 224, n_sample: 3000}`
- `compile_args`: `{samples: ["A_001","A_002","A_004"], bundle_wds: True}`

**`render_cli` output:**
```bash
# ── Stage 0: inspect ──────────────────────────────────────────────────────
python3 ${SKILL_DIR}/scripts/inspect_spatialdata.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_001.zarr
python3 ${SKILL_DIR}/scripts/inspect_spatialdata.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_002.zarr
python3 ${SKILL_DIR}/scripts/inspect_spatialdata.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_004.zarr

# ── Stage 1: tissue_inside ────────────────────────────────────────────────
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_001.zarr \
    --input-table-key table --output-table-key table_tissue
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_002.zarr \
    --input-table-key table --output-table-key table_tissue
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_004.zarr \
    --input-table-key table --output-table-key table_tissue

# ── Stage 2: nucleus_presence ─────────────────────────────────────────────
python3 ${SKILL_DIR}/scripts/filter_nucleus_presence.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_001.zarr \
    --input-table-key table_tissue --output-table-key table_tissue_nucleus
...

# ── Stage 3: extract ──────────────────────────────────────────────────────
python3 ${SKILL_DIR}/scripts/extract_sample.py \
    --zarr   /home/zouqi/datasets/mash/spatialdata/A_001.zarr \
    --output /home/zouqi/datasets/mash/stvisuome/A_001 \
    --table-key table_tissue_nucleus \
    --extract-mode full_ops_level --mpp 0.5 --patch-size 224 --n-sample 3000
...

# ── Stage 4: compile ──────────────────────────────────────────────────────
python3 ${SKILL_DIR}/scripts/compile_dataset.py \
    --per-sample-dir /home/zouqi/datasets/mash/stvisuome \
    --output         /home/zouqi/datasets/mash/stvisuome/compiled \
    --samples        A_001,A_002,A_004 \
    --bundle-wds
```

---

## 11. SKILL.md Updates

The SKILL.md rewrite must cover:

1. **Stage-based workflow diagram** replacing the current three-phase diagram.
2. **"filtered_table is optional"** — prominent callout: the skill does not require
   `filtered_table`. It is one possible `--input-table-key` value, nothing more.
3. **Recipe selection rules** — map trigger phrases to stage scripts (mirrors §6.2).
4. **Table-key propagation** — diagram showing stage N output → stage N+1 input → extract.
5. **Worked example** — the full NL request from §10 with the rendered CLI output.
6. **Stage contract** — what every filter script writes (zarr table + JSON report).
7. **extract_sample.py reference** updated — no biological policy args.
8. **compile_dataset.py reference** updated — `--samples` flag documented.
