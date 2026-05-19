---
name: daas-compiler
version: 0.7.8
description: Extract cell-centered HE image patches from SpatialData into an indexed WebDataset for ML model training. Covers single-sample extraction, multi-sample parallel batch extraction, compile-step gene-intersection merge, and CellPatchDataset with LRU mmap loader. Use when building HE patch datasets for predicting gene expression from tissue morphology, or when scaling a single-sample pipeline to 10s–100s of zarr samples.
---

## Agent Contract

These rules govern the agent's behavior when executing daas-compiler workflows.
They cannot be overridden by conversational context.

**Stage plan required.** The agent must produce and present an explicit stage plan
before executing any multi-step workflow (filtering → extraction → compile → task-ready).
See `references/agent-contract.md` for required stage plan fields, including `task_type`,
`filter_stages`, `final_table_key`, `split_config`, and `loader_config`.

**Preserve the training-ready contract.** The agent must not describe outputs as
"training-ready" unless a task-ready packaging stage has produced split metadata,
loader-ready artifacts, and validation reports. See `references/training-ready-contract.md`.

**Distinguish artifact levels.** The agent must use correct level terminology in all
responses:
- L2 = patch-compiled (`extract_sample.py` output)
- L3 = dataset-compiled (`compile_dataset.py` output)
- L4 = training-ready (task adapter output, task-specific, split-aware via metadata)

**Do not physically partition shards by split by default.** The default L4 output stores
all cells in `data/shard-*.tar`. Split selection is done by the loader at runtime via
`splits/split_membership.parquet`. Physical `train/`, `val/`, `test/` shard directories
require an explicit `--materialize-split-shards` flag and must not be described as the
default layout.

**Follow versioning and commit rules.** When modifying the skill (SKILL.md, scripts,
daas/ package), the agent must follow the commit scopes in `CONTRIBUTING.md` and note
any schema version bumps required by `VERSIONING.md`.

**No silent behavior changes.** The agent must not silently change default filtering,
extraction, or task-ready packaging behavior. Any change to defaults must be announced
to the user and reflected in the stage plan before execution.

**Use documented CLI invocations exactly.** When calling any script (`filter_tissue.py`,
`filter_nucleus_presence.py`, `extract_sample.py`, `compile_dataset.py`, etc.) the agent
must use the argument names from this skill document verbatim. Never guess or invent
argument names. If the invocation is unclear, run `python3 <script> --help` first and
read the output before constructing the call. Suppress stderr (`2>/dev/null`) only after
a successful dry-run; otherwise always capture stderr so failures are visible.

**Multi-sample parallel extraction must use `extract_all.py`.** Never use background
bash loops, bare subprocess calls, or background Bash tool invocations to parallelize
`extract_sample.py` across samples — these cannot capture failures and will silently drop
samples. For multi-sample extraction, always use:
```bash
python3 "${SKILL_DIR}/scripts/extract_all.py" \
    --zarr-dir <dir> --output <out> --workers <N> \
    --samples A_001,A_002,A_004   # restrict to specific samples
    # ... other extract_sample.py flags forwarded
```
`extract_all.py` exits with code 1 if any sample fails, prints the last 3k chars of
stderr for each failure, and skips already-completed samples safely.

**Verify all extractions before compiling.** After extraction (whether via `extract_all.py`
or individual `extract_sample.py` calls), verify that every expected sample directory
contains both `manifest.parquet` **and** `expression.h5ad` before calling
`compile_dataset.py`. A missing file means that sample's extraction failed silently.
Do not rely on Monitor tool output or background-task completion signals — only file
system presence is authoritative.

**Do not use Monitor output to assess extraction completion or cell counts.** The Monitor
tool may display output from stale background processes started earlier in the session.
The only reliable sources of truth are: the file system (`manifest.parquet` +
`expression.h5ad` exist), and `filter_report.json` for definitive cell counts.

**Read `filter_report.json` after every extraction and report drops to the user.**
Each `{output}/{sample_id}/meta/filter_report.json` contains the definitive
`n_cells_source → n_out` count and `drop_counts_by_reason`. After extraction, read this
file for every sample and check `drop_counts_by_reason`. If `full_oob` or `need_pad` is
non-zero, explicitly tell the user how many cells were dropped and why (boundary cells
that cannot yield a complete patch). Never report cell counts from progress-bar output
or monitor streams — always read `meta/filter_report.json`.

**Ask before using existing filtered tables.** When `inspect_spatialdata.py` reports
that `table_tissue`, `table_tissue_nucleus`, or any other pre-filtered table already
exists in a zarr, the agent must ask the user whether to use the existing table or
re-run the filter from scratch. Never auto-decide silently.

**Consult reference docs for task-ready requests.** Before producing any L4 output plan:
- `references/training-ready-contract.md` — what L4 requires
- `references/artifact-levels.md` — level definitions and distinctions
- `references/task-adapters.md` — task-specific artifact layouts
- `references/agent-contract.md` — stage plan fields and worked example

**For complex spatial transcriptomics training tasks, first consult
`references/agent-contract.md` and create a stage plan.** The agent must
distinguish patch-compiled (L2), dataset-compiled (L3), split-pending task
skeleton, and fully training-ready (L4) artifacts. Splits are metadata consumed
by loaders at runtime — do not physically split shards by default. If split
allocation is missing, prompt the user or produce split-pending artifacts, not
a falsely training-ready dataset. Generated splits must be sample-level or
group-level (`sample_holdout`, `ratio_by_group`, `group_kfold`); DAAS never
generates random cell-level train/val/test splits.

---

## Version Compatibility

Tested with: spatialdata 0.7+, wsidata 0.9+, lazyslide 0.10+, anndata 0.10+, numpy 1.26+, pandas 2.2+, scipy 1.12+

Key API surface to verify on version mismatch:
- `wsidata.open_wsi(sdata, image_key, store)` — SpatialDataImage2DReader creation
- `wsi.set_mpp(value)` — must be called before TileSpec; zarr reader auto-detects None
- `TileSpec.from_wsidata(wsi, tile_px, mpp, slide_mpp)` — base_width/ops_level computation
- `wsidata.io.add_tiles(wsi, key, xys, tile_spec, tissue_ids)` — tile GeoDataFrame creation
- `wsi.iter.tile_images(key)` — tile generator
- `lazyslide.pl.tiles(wsi, tile_key)` — returns None, use `plt.gcf()` after
- `wsi.reader.get_thumbnail(size)` — requires `size` parameter

## Installation

This skill ships its own scripts under `${SKILL_DIR}/scripts/` and a small
`daas` Python package under `${SKILL_DIR}/daas/`. Run once per environment:

```bash
pip install -e "${SKILL_DIR}"
pip install -r "${SKILL_DIR}/requirements.txt"
```

After this, `from daas.dataset import CellPatchDataset` works in user code
and all CLI scripts can be invoked via their full skill-relative path.

`${SKILL_DIR}` is the absolute path of the directory containing this SKILL.md
file. When this skill loads, the system prints:

```
Base directory for this skill: <path>
```

**Set `SKILL_DIR` to exactly that `<path>` value.** Do not guess or construct
the path from the plugin cache layout — read it from the system message.
If imports fail, ask the user which Python interpreter to use; never hardcode a path.

---

## Stage-Based Workflow

Extract cell-centered patches by composing stages. Each stage writes a new
table key back into the zarr; the final key flows into `extract_sample.py`.

```
sdata.zarr
    │
    │  [Stage 0: inspect]  always first — verify keys
    │
    │  [Stage 1–N: filter]  optional, any combination, in order:
    │    filter_tissue.py          → writes table_tissue
    │    filter_nucleus_presence.py → writes table_tissue_nucleus
    │    filter_nucleus_overlap.py  → writes table_tissue_nucleus_he
    │
    │  [Stage N+1: extract]  per-sample, parallelizable
    │    extract_sample.py --table-key <FINAL_KEY>
    │
    │  [Stage N+2: compile]  once all samples done
    │    compile_dataset.py [--samples A,B,C] [--bundle-wds]
    ▼
compiled/  or  per-sample bundled WebDataset shards
```

### filtered_table is optional — confirm before use

`filtered_table` is not required. It is one possible `--input-table-key` value.

**If inspect reveals a pre-existing `filtered_table` (or any other filtered table
key) in a zarr, do NOT use it silently.** Stop and ask the user:

> "Sample `<id>` already contains `filtered_table` in its zarr. Do you want to
> use it directly as the starting table, or run the filter stages from scratch
> starting from `table`?"

Only pass `--input-table-key filtered_table` after the user explicitly confirms.
If the user says to run from scratch, start from `--input-table-key table`.

### Building a stage plan from natural language

Use `daas.planning.parse_stage_plan()` and `render_cli()`:

```python
from daas.planning import parse_stage_plan, render_cli

plan = parse_stage_plan(
    "filter outside tissue, keep nucleus boundaries, "
    "mpp=0.5, patch size 224, use optim_ops_level, sample 3000 cells per sample"
)
# StagePlan fields available for presentation:
#   plan.task_type        → str, e.g. "he2st"
#   plan.tissue_key       → str, default "tissue" — passed to filter_tissue.py as --tissue-key
#   plan.filtered_table_key → str, default "filtered_table" — agent checks inspect output against this key
#   plan.filter_stages    → list[str], e.g. ["tissue_inside", "nucleus_presence"]
#   plan.final_table_key  → str, e.g. "table_tissue_nucleus"
#   plan.extract_args     → dict, e.g. {"mpp": 0.5, "patch_size": 224, ...}
#   plan.compile_args     → dict, e.g. {"bundle_wds": True}
#   plan.stages           → list[StageSpec] with .name/.script/.input_table_key/.output_table_key
print(render_cli(plan, ["/data/A_001.zarr"], "/data/out"))
```

Trigger phrase → stage mapping:

| Phrase | Stage |
|---|---|
| "inside tissue", "out of tissue", "outside tissue" | tissue_inside |
| "with nucleus boundaries", "only cells with nucleus" | nucleus_presence |
| "Xenium nucleus overlaps HE nucleus", "overlap >" | xenium_he_nucleus_overlap |
| "optim_ops_level", "ops_level" | extract_mode=full_ops_level |
| "sample N cells per sample" | --n-sample N |

### `--n-sample` is strictly opt-in

**Never add `--n-sample` unless the user explicitly states a cell count.**
Omit the flag entirely when the user does not specify one — the default
behaviour (`None`) processes all valid cells. Do not infer a count from
previous conversations, do not suggest a "safe" default, and do not add it
as a convenience. It is only valid when the user explicitly asks to subsample
(e.g. "sample 3000 cells per sample", "use 5000 cells for a quick test").

### Tissue segmentation — required confirmation

When including `tissue_inside` in a stage plan, confirm `image_key` with the
user **before** generating CLI commands if it cannot be unambiguously inferred:

- If `sdata.images` has exactly one key, use it as the default without asking.
- If there are multiple image keys (e.g. `he_image`, `dapi_image`), ask the
  user which one to use — do not assume.

`--allow-holes` defaults to `False`; pass it only when the user explicitly asks.
`--tissue-key` defaults to `"tissue"`; always pass it explicitly so the shape key is predictable.

**If the `--tissue-key` shape already exists in `sdata.shapes`**, do NOT run the
script without asking. The script will warn but still overwrite. Ask the user first:

> "Sample `<id>` already has a tissue shape `<key>` in its zarr. Re-run SOPA
> tissue segmentation (will overwrite `<key>`), or skip segmentation and reuse
> the existing shape for filtering?"

- If user says **re-run**: run `filter_tissue.py` with `--tissue-key <key> --force`. SOPA re-runs and overwrites the existing shape.
- If user says **reuse**: run `filter_tissue.py` with `--tissue-key <key>` but WITHOUT `--force`. The script detects the key already exists and skips SOPA, using the existing shape directly for filtering.

### Filter scripts write into the original zarr — confirm first

Every filter script (`filter_tissue.py`, `filter_nucleus_presence.py`,
`filter_nucleus_overlap.py`) writes its output table (and optionally tissue
shapes) **directly into the original zarr** via `sdata.write_element()`.

**Before presenting the stage plan for approval, explicitly state** which keys
will be written into each zarr and at what path. The stage plan approval acts as
the user's confirmation. Do not run any filter script before the user approves
the plan.

Example disclosure in the stage plan:

> **Zarr writes** (each filter stage writes back into the source zarr):
> - `A_001.zarr` ← `table_tissue`, `region_of_interest` (tissue shape), `table_tissue_nucleus`
> - `A_002.zarr` ← `table_tissue`, `region_of_interest`, `table_tissue_nucleus`
> - `A_004.zarr` ← `table_tissue`, `region_of_interest`, `table_tissue_nucleus`

If the user has not approved zarr writes, do not run the filter scripts.

### Stage report contract

Every stage script writes a JSON to `--report-dir`:

```json
{
  "stage": "nucleus_presence",
  "input_table_key": "table_tissue",
  "output_table_key": "table_tissue_nucleus",
  "n_cells_in": 184523,
  "n_cells_out": 173210,
  "drop_counts_by_reason": {"missing_nucleus_boundary": 11313},
  "warnings": []
}
```

### Worked example

Request:
> "Process A_001,A_002,A_004 under /home/zouqi/datasets/mash/spatialdata into
> cell-centered HE patches. Filter out cells outside tissue and only keep cells
> with nucleus boundaries. Target mpp=0.5, patch size=224, use optim_ops_level,
> output to /home/zouqi/datasets/mash/stvisuome, sample 3000 cells per sample,
> compile, and write bundled WebDataset shards."

Stage plan resolves to:
- stages: `tissue_inside` → `nucleus_presence`
- final table key: `table_tissue_nucleus`
- extract: `--extract-mode full_ops_level --mpp 0.5 --patch-size 224 --n-sample 3000`
- compile: `--samples A_001,A_002,A_004 --bundle-wds`

Generated CLI (run in order):
```bash
# Stage 0: inspect
python3 ${SKILL_DIR}/scripts/inspect_spatialdata.py \
    --zarr .../A_001.zarr \
    --report-dir .../stvisuome/A_001/meta
# (repeat for A_002, A_004)
```

After inspect: if `--report-dir` is given, read `meta/inspect_report.json`. Check whether `plan.tissue_key` appears in `shapes[*].key` and `plan.filtered_table_key` appears in `tables[*].key`. If either exists, ask the user before proceeding.

```bash
# Stage 1: tissue_inside
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr .../A_001.zarr \
    --input-table-key table --output-table-key table_tissue
# (repeat for A_002, A_004)

# Stage 2: nucleus_presence
python3 ${SKILL_DIR}/scripts/filter_nucleus_presence.py \
    --zarr .../A_001.zarr \
    --input-table-key table_tissue --output-table-key table_tissue_nucleus \
    --nucleus-boundaries-key nucleus_boundaries
# (repeat for A_002, A_004)

# Stage 3: extract (parallel via extract_all.py — never use background bash loops)
python3 ${SKILL_DIR}/scripts/extract_all.py \
    --zarr-dir .../spatialdata \
    --output   .../stvisuome \
    --workers  3 \
    --samples  A_001,A_002,A_004 \
    --table-key table_tissue_nucleus \
    --extract-mode full_ops_level --mpp 0.5 --patch-size 224 --n-sample 3000

# After extract_all.py completes: read filter_report.json for every sample and
# report any non-zero drop_counts_by_reason (full_oob, need_pad) to the user.
# Verify manifest.parquet + expression.h5ad exist in all sample dirs before compile.

# Stage 4: compile
python3 ${SKILL_DIR}/scripts/compile_dataset.py \
    --per-sample-dir .../stvisuome \
    --output .../stvisuome/compiled \
    --samples A_001,A_002,A_004 \
    --bundle-wds
```

---

# Cell Patch Dataset: Single-Sample to Multi-Sample Pipeline

## Three-Phase Architecture

```
N × SpatialData.zarr  [L0 raw / L1 canonical after filter stages]
        │
        │  [Phase 1: extract]  per-sample, parallelizable  → L2 patch-compiled
        ▼
per-sample/
  {sample_id}/
    shard-{N:06d}.tar     JPEG patches (WebDataset)
    shard-{N:06d}.idx     binary offset index (CIDX0001)
    expression.h5ad       raw counts sparse matrix
    manifest.parquet      per-cell metadata
    filter_report.json
    viz/
      viz_global_tiles.png         lazyslide.pl.tiles overview (dpi=300, pre-shard)
      viz_patch_grid.png           5×5 random cells with cell+nucleus boundary overlays (dpi=300, pre-shard)
        │
        │  [Phase 2: compile]  run once all samples done, <2 min  → L3 dataset-compiled
        ▼
compiled/
  manifest.parquet        global_idx → image location + expr location
  expression.h5ad         gene intersection across all samples
  {sample_id}/            [optional: --bundle-wds] bundled shards, one sample only
    shard-NNNNNN.tar      jpg + expr.npz + json per cell
  gene_panel.json         [--bundle-wds] gene names matching .expr.npz indices
  bundled_manifest.parquet  [--bundle-wds] cell_id, sample_id, shard_path, global_idx
        │                 ← L3 artifacts are NOT training-ready
        │  [Phase 3: task-ready packaging]  task adapter  → L4 training-ready
        ▼
{task_output}/
  data/
    shard-000000.tar  ...   ← all cells, NOT split-partitioned
  splits/
    train.json  val.json  test.json   ← split membership metadata
    split_membership.parquet          ← per-cell: global_idx, sample_id, split
    split_report.json
  gene_panel.json  gene_panel.sha256
  task_config.yaml  loader_config.yaml
  dataset_card.json  validation_report.json
```

**Key invariant:** `compiled/manifest.parquet` row `i` == `compiled/expression.h5ad` row `i` == `global_idx=i`. Both are produced by the same `sorted()` traversal in compile.

**Split policy:** Splits are metadata. The loader filters by `splits/split_membership.parquet`
at runtime. Physical `train/`, `val/`, `test/` shard directories are NOT produced by default.
Use `--materialize-split-shards` only when an explicit export is needed.

**L3 → L4:** `CellPatchDataset` and `BundledCellPatchDataset` consume L3 artifacts directly. They are useful for research and iteration but do NOT constitute L4 training-ready output — the loader must still handle splitting (via `sample_ids`), and no `splits/`, `task_config.yaml`, `loader_config.yaml`, or validation reports exist. To produce L4, run a task adapter (e.g., `make_task_dataset.py` for HE2ST).

---

## Phase 1: Single-Sample Extraction

### CLI

```bash
python3 "${SKILL_DIR}/scripts/extract_sample.py" \
    --zarr   /data/A_002.zarr \
    --output /data/out/A_002 \
    [--sample-id A_002]        # default: inferred from zarr dirname
    [--n-sample 10000]         # default: all valid cells
    [--patch-size 224] [--mpp 0.5] [--shard-size 500] [--seed 42]
    [--image-key he_image] [--shapes-key cell_circles] [--table-key table] \
    [--patch-filter-policy auto|strict_no_padding|stvisuome_minimal|strict_with_padding] \
    [--cell-id-column cell_id] [--filter-report-name filter_report.json]
```

### Output

```
{output}/
  shard-000000.tar     500 JPEG patches per shard
  shard-000000.idx     binary offset index (CIDX0001, 32B/record)
  expression.h5ad      AnnData (n_cells, n_genes), obs has sample_id
  manifest.parquet     per-cell metadata (see columns below)
  filter_report.json   biological + patch policy applied, drop counts, warnings
```

### manifest.parquet columns

| column | type | notes |
|--------|------|-------|
| `sample_id` | str | e.g. "A_002" |
| `sample_index` | int | 0..N-1 within this sample |
| `sample_key` | str | zero-padded `f"{i:07d}"` |
| `cell_id` | str | original cell ID |
| `shard_path` | str | absolute path to .tar file |
| `tar_offset` | int64 | byte offset of JPEG in tar |
| `jpg_size` | int32 | JPEG byte count |
| `expr_row` | int32 | = sample_index (local row in per-sample h5ad) |
| `center_x_pixel` | float64 | centroid x in level-0 px |
| `center_y_pixel` | float64 | centroid y in level-0 px |
| `bbox_x0` | float64 | tile top-left x |
| `bbox_y0` | float64 | tile top-left y |
| `source_table_key` | str | table key that produced this cell (e.g. `filtered_table`) |
| `source_shape_key` | str | shape key that produced this cell (e.g. `cell_circles`) |

### Filtering (patch validity only)

`extract_sample.py` loads `--table-key` directly from the zarr. Run filter
stage scripts first to produce the appropriate table key (see Stage-Based
Workflow above).

#### Patch-validity filtering (Layer 2)

| Policy | Drops | Allowed extract modes |
|---|---|---|
| `auto` (default) | Resolves to `strict_no_padding`. | All. |
| `strict_no_padding` | `full_oob` ∪ `need_pad`. **Required for `full_scale0` / `full_ops_level`** — those modes silently clip boundary-crossing tiles. | All. |
| `stvisuome_minimal` | `full_oob` only; keeps `need_pad` (boundary-crossing tiles). | **`tile_images` only** |
| `strict_with_padding` | Reserved. Raises. | — |

Positive-centroid filtering (`cx_px > 0 & cy_px > 0`) runs before the patch mask under every policy.

#### filter_report.json

Written to `{output}/filter_report.json` **before any shard**. Key fields:

- `source_table_key`, `source_shape_key` — the keys actually consumed
- `patch_policy_requested` / `_applied`
- `image_width_px`, `image_height_px` — level-0 H&E dimensions
- Sequential counters: `n_cells_source → n_after_shape_alignment → n_after_positive_centroid → n_after_patch_policy → n_out`
- `drop_counts_by_reason` keyed on `unaligned_with_shapes`, `non_positive_centroid`, `full_oob`, `need_pad`, `requested_subsample`
- `warnings` — emit them verbatim in reply

#### Row-alignment invariants (must hold every run)

- Within a sample: `manifest.parquet` row `i` ≡ `expression.h5ad` row `i`, `expr_row == sample_index`, `manifest.cell_id == adata_out.obs.cell_id`, `gene_row_index` resolves to the cell of the aligned table.
- After Phase 2 compile: `compiled/manifest.parquet[i] == compiled/expression.h5ad[i] == global_idx=i`.

These checks are asserted in `_validate(...)`. If you change the filter pipeline, re-run the tests under `tests/test_filtering*.py` before declaring success.

### Pipeline Internals

**MPP derivation** (from affine transforms, no hardcoding):

```python
img_aff   = get_transformation(sdata.images[image_key],
                to_coordinate_system="global").to_affine_matrix(
                input_axes=("y","x"), output_axes=("y","x"))
shape_aff = get_transformation(gdf,
                to_coordinate_system="global").to_affine_matrix(
                input_axes=("y","x"), output_axes=("y","x"))

sx = np.sqrt(img_aff[0,0]**2 + img_aff[1,0]**2)   # rotation-invariant
sy = np.sqrt(img_aff[0,1]**2 + img_aff[1,1]**2)
SCALE_SHAPE = (shape_aff[0,0] + shape_aff[1,1]) / 2.0   # µm → global
SLIDE_MPP   = ((sx + sy) / 2.0) / SCALE_SHAPE
```

| Image tf | Shape tf | SLIDE_MPP | Example |
|----------|----------|-----------|---------|
| Identity | Scale(s) | 1/s | Xenium: s=4.7059 → 0.2125 µm/px |
| Scale(p) | Scale(s) | p/s | Visium: p=0.5, s=1.0 → 0.5 |
| Affine   | Affine   | sqrt(a²+b²) / (sx+sy)/2 | rotated scans |

**OOB filtering:**

```python
BASE_SIZE = round(PATCH_SIZE * MPP_TGT / SLIDE_MPP)
BASE_HALF = BASE_SIZE / 2.0   # MUST be float, not integer division

full_oob = ((sx0+BASE_SIZE<=0)|(sx0>=IMG_W)|(sy0+BASE_SIZE<=0)|(sy0>=IMG_H))
need_pad = (((sx0<0)|(sx0+BASE_SIZE>IMG_W)|(sy0<0)|(sy0+BASE_SIZE>IMG_H)) & ~full_oob)
valid_mask = ~full_oob & ~need_pad
# full_oob: skip entirely; need_pad: skip (zarr clips silently, no zero-fill)
```

**Spatial sort** (zarr cache locality):

```python
sort_key = ((np.maximum(0,sy0)//4096)*10000 + (np.maximum(0,sx0)//4096))
proc_order = np.argsort(sort_key, kind="stable")
```

**TileSpec setup:**

```python
wsi = open_wsi(sdata, image_key=image_key, store=None)
wsi.set_mpp(SLIDE_MPP)
spec = TileSpec.from_wsidata(wsi, tile_px=PATCH_SIZE, mpp=MPP_TGT, slide_mpp=SLIDE_MPP)
assert spec.base_width == BASE_SIZE   # must match
```

**Patch extraction** — only `wsi.iter.tile_images()` is allowed:

```python
add_tiles(wsi, key="cell_tiles",
          xys=np.column_stack([sx0_ord, sy0_ord]),
          tile_spec=spec, tissue_ids=np.zeros(n_out, dtype=int))
for tile in wsi.iter.tile_images("cell_tiles"):
    jpg = io.BytesIO()
    Image.fromarray(tile.image).save(jpg, format="JPEG", quality=95)
```

Never use numpy/dask/zarr/PIL/cv2 to read HE pixel data directly.

### Extraction Strategies

> **Required: ask the user which `--extract-mode` to use before running any extraction.**
> The three modes have very different memory and speed profiles; the right
> choice depends on the user's hardware. Do NOT default silently. Present the
> table below, ask the user to pick one of `tile_images`, `full_scale0`, or
> `full_ops_level`, and pass that choice as `--extract-mode <choice>` on the
> CLI. This applies to both `extract_sample.py` (single sample) and
> `extract_all.py` (batch — forwards the flag to every worker).

| Strategy | `--extract-mode` | Memory | Speed (3000 tiles) | Quality |
|----------|------------------|--------|---------------------|---------|
| Original `tile_images` | `tile_images` | Low (~50 MB) | 224s (1×) | Baseline |
| Full scale0 load | `full_scale0` | scale0 size (~1.6 GB) | 24s (9×) | Good (插值差异) |
| Full ops_level load | `full_ops_level` | ops_level size (~0.4 GB) | 6s (36×) | Good (插值差异) |

**How they work:**

```
tile_images:     add_tiles → iter.tile_images → each tile triggers 3 chunk reads
full_scale0:     sdata["scale0"].values (1.6GB) → memory crop 527×527 → resize 224
full_ops_level:  sdata["scaleN"].values (0.4GB) → memory crop ops_w×ops_w → resize 224
```

**`full_ops_level` — recommended for most cases:**

Reads the entire pyramid level that TileSpec selected (`spec.ops_level`), crops tiles from memory, and resizes. This is the fastest strategy because:
- The ops_level is already the optimal pyramid level (closest to target MPP)
- Only one chunk-read pass (loading `.values` triggers dask compute once)
- Memory scales with the pyramid level: ~0.4 GB for typical Xenium scale1, less for higher levels

```python
lvl_key = f"scale{spec.ops_level}"
full_img = sdata.images[image_key][lvl_key]["image"].values  # (C, H, W)
ds_y = scale0_img.shape[1] / full_img.shape[1]   # level-0 → ops_level ratio
ds_x = scale0_img.shape[2] / full_img.shape[2]

for i in range(n_out):
    x0 = int(sx0_ord[i] / ds_x)
    y0 = int(sy0_ord[i] / ds_y)
    tile = full_img[:, y0:y0+spec.ops_width, x0:x0+spec.ops_width]
    tile_224 = Image.fromarray(tile.transpose(1,2,0)).resize((224,224), LANCZOS)
```

**`full_scale0` — when ops_level is not available:**

Loads the full-resolution scale0 image. Higher memory but same principle.

**`tile_images` — when memory is scarce:**

The original wsidata pipeline. Each tile triggers zarr chunk reads. Slowest but uses minimal memory.

**Choosing the ops_level:**

TileSpec selects `ops_level` to minimize resize magnitude:

| MPP_TGT | ops_level | ops_width | Rationale |
|---------|-----------|-----------|-----------|
| 0.2 | 0 | 211→224 | scale0 MPP closest to target |
| 0.3-0.6 | 1 | 158-316→224 | scale1 best match |
| 0.8-1.2 | 2 | 210-316→224 | scale2 best match |
| 1.5-2.5 | 3 | 197-329→224 | scale3 best match |

The rule: prefer `ops_width >= PATCH_SIZE` (downsample) over upsampling. The code always follows `spec.ops_level` dynamically — no hardcoding.

**Shard writing — two-phase (write then re-read for accurate offsets):**

```python
IDX_MAGIC      = b"CIDX0001"
IDX_RECORD_FMT = "<iQIQII"   # sample_index(i4) jpg_offset(Q) jpg_size(I)
                              # json_offset(Q) json_size(I) reserved(I)
assert struct.calcsize(IDX_RECORD_FMT) == 32   # must be 32, never use hardcoded size

def flush_shard(shard_buf, shard_no, output_dir):
    tar_path = output_dir / f"shard-{shard_no:06d}.tar"
    # Phase 1: write
    with tarfile.open(tar_path, "w") as tf:
        for si, sk, jpg, jsn in shard_buf:
            for ext, data in [(".jpg", jpg), (".json", jsn)]:
                ti = tarfile.TarInfo(name=f"{sk}{ext}"); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    # Phase 2: re-read for accurate offset_data (tell() during write is wrong)
    records = []
    with tarfile.open(tar_path, "r") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for si, sk, jpg, jsn in shard_buf:
            jm, nm = members[f"{sk}.jpg"], members[f"{sk}.json"]
            records.append((si, jm.offset_data, jm.size, nm.offset_data, nm.size))
    with open(output_dir / f"shard-{shard_no:06d}.idx", "wb") as f:
        f.write(IDX_MAGIC)
        f.write(struct.pack("<I", len(records)))
        for rec in records:
            f.write(struct.pack(IDX_RECORD_FMT, *rec, 0))
    return str(tar_path), records
```

After `flush_shard`, back-patch `shard_path`/`tar_offset`/`jpg_size` into `cells_rows`:

```python
tar_path, recs = flush_shard(shard_buf, shard_no, output_dir)
for row, rec in zip(cells_rows[-len(shard_buf):], recs):
    row["shard_path"] = tar_path
    row["tar_offset"] = rec[1]   # jm.offset_data
    row["jpg_size"]   = rec[2]   # jm.size
```

---

## Phase 1b: Parallel Multi-Sample Extraction

### CLI

```bash
python3 "${SKILL_DIR}/scripts/extract_all.py" \
    --zarr-dir /data/spatialdata \
    --output   /data/out \
    --workers  4 \
    [--samples A_001,A_002,A_004]   # restrict to these sample IDs; default: all
    [--n-sample 10000] [--pattern "*.zarr"]
    # all other extract_sample.py args forwarded
```

### Behavior

- Discovers all `*.zarr` dirs under `--zarr-dir` (sorted)
- Runs `extract_sample.py` as a subprocess per sample using `ProcessPoolExecutor`
- **Skips** samples whose `{output}/{sample_id}/manifest.parquet` + `expression.h5ad` already exist — safe to re-run after partial failures
- Exits with code 1 if any sample fails; prints last 3k chars of stderr for each failure

### Worker count guidance

| Scenario | `--workers` | Notes |
|----------|-------------|-------|
| 72 samples, fast NFS | 4–8 | Zarr I/O is the bottleneck |
| Local NVMe | 8–16 | CPU/memory bound instead |
| Debug single sample | 1 | Easier to read logs |

Each worker loads one full zarr into memory (~2–4 GB RAM). Set workers ≤ `floor(RAM_GB / 4)`.

### Implementation pattern

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_one(zarr_path, output, extra_args):
    sample_id = Path(zarr_path).stem
    out_dir   = Path(output) / sample_id
    if (out_dir/"manifest.parquet").exists() and (out_dir/"expression.h5ad").exists():
        return sample_id, 0, "ALREADY_DONE"
    cmd = [sys.executable, "scripts/extract_sample.py",
           "--zarr", zarr_path, "--output", str(out_dir)] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    return sample_id, r.returncode, (r.stdout + r.stderr)[-3000:]

with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(run_one, str(zp), output, extra): zp.stem
               for zp in zarr_paths}
    for fut in as_completed(futures):
        sid, rc, log = fut.result()
        print("OK" if rc == 0 else f"FAIL: {log}")
```

---

## Phase 2: Compile

### CLI

```bash
python3 "${SKILL_DIR}/scripts/compile_dataset.py" \
    --per-sample-dir /data/out \
    --output         /data/compiled \
    [--samples A_001,A_002,A_004]   # default: all subdirs with manifest + h5ad
    [--bundle-wds] [--shard-size 500]
```

Scans all subdirs of `--per-sample-dir` that have **both** `manifest.parquet` and `expression.h5ad`. Skips others (e.g. smoke test dirs).

**`--bundle-wds`** writes bundled shards directly under `{output}/{sample_id}/shard-NNNNNN.tar` — no extra `wds/` subdirectory. Each shard contains only cells from that sample (jpg + sparse expr.npz + json per cell). Shards never mix samples. Training does not require mmap or the compiled h5ad. The gene panel is in `{output}/gene_panel.json`; the bundled manifest (with shard_path, no tar_offset) is in `{output}/bundled_manifest.parquet`.

### Implementation

```python
sample_dirs = sorted(d for d in per_sample.iterdir()
                     if d.is_dir()
                     and (d/"manifest.parquet").exists()
                     and (d/"expression.h5ad").exists())

# CRITICAL: same sorted() for both — guarantees manifest row i == h5ad row i
parts  = [pd.read_parquet(d/"manifest.parquet") for d in sample_dirs]
adatas = [anndata.read_h5ad(d/"expression.h5ad") for d in sample_dirs]

global_manifest = pd.concat(parts, ignore_index=True)
global_manifest["global_idx"] = np.arange(len(global_manifest), dtype=np.int64)

common_genes = adatas[0].var_names
for a in adatas[1:]:
    common_genes = common_genes.intersection(a.var_names)
assert len(common_genes) > 0, "Gene intersection is empty"

combined = anndata.concat([a[:, common_genes] for a in adatas],
                          axis=0, merge="same")
combined.obs_names_make_unique()
assert combined.n_obs == len(global_manifest)   # invariant check

global_manifest.to_parquet(compiled/"manifest.parquet", index=False)
combined.write_h5ad(compiled/"expression.h5ad")
```

### Compile speed (72 samples, ~5M cells)

| Step | Time |
|------|------|
| concat 72 parquet | <5s |
| read 72 h5ad | ~30s |
| gene intersection + concat | ~20s |
| write compiled h5ad | ~30s |
| **total** | **<2 min** |

---

## Phase 3: CellPatchDataset

### Usage

```python
from daas.dataset import CellPatchDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Split is injected externally via sample_ids — not hardcoded in dataset
train_samples = json.load(open("splits/train.json"))

ds = CellPatchDataset(
    manifest_path  = "compiled/manifest.parquet",
    h5ad_path      = "compiled/expression.h5ad",
    sample_ids     = train_samples,          # None = all samples
    transform      = T.Compose([T.ToTensor()]),
    mmap_cache_size = 256,   # see sizing note below
)
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8)
batch  = next(iter(loader))
# batch["image"].shape      → (256, 3, 224, 224)
# batch["expression"].shape → (256, n_genes)
# batch["cell_id"]          → list of str
# batch["sample_id"]        → list of str
```

### Three training modes

```python
ds = CellPatchDataset(manifest, h5ad, sample_ids=train, transform=dino_augment)
batch["image"]              # self-supervised patch pretraining

ds = CellPatchDataset(manifest, h5ad, sample_ids=train)
batch["expression"]         # expression foundation model

ds = CellPatchDataset(manifest, h5ad, sample_ids=train, transform=val_tf)
batch["image"], batch["expression"]   # multimodal HE → cell states
```

### BundledCellPatchDataset (no mmap, no h5ad)

When `--bundle-wds` is set on compile, every cell's image + sparse expression is co-located in one tar entry. The training-time API mirrors `CellPatchDataset` but reads via `tarfile` (no mmap, no h5ad):

```python
from daas.dataset import BundledCellPatchDataset

ds = BundledCellPatchDataset(
    compiled_dir     = "compiled",       # same dir as CellPatchDataset
    sample_ids       = train_samples,    # None = all samples
    transform        = T.Compose([T.ToTensor()]),
    dense_expression = True,             # False → (indices, values, n_genes)
)
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8)
batch  = next(iter(loader))
# batch["image"].shape      → (256, 3, 224, 224)
# batch["expression"].shape → (256, n_genes)  (dense_expression=True)
```

| | `CellPatchDataset` (mmap) | `BundledCellPatchDataset` |
|---|---|---|
| Image source | mmap on shard tar + offset | `tarfile.extractfile` on bundled tar |
| Expression source | compiled `expression.h5ad` indexed by `global_idx` | `{key}.expr.npz` inside the same tar entry |
| Speed (per-cell read) | very fast (zero-copy mmap) | slightly slower (~2×) but no mmap RAM growth |
| Total memory at scale | mmap pages may grow per worker | bounded; OS page cache only |
| Files needed at training time | manifest.parquet + expression.h5ad + per-sample shards | compiled/ only — fully self-contained |

Pick `BundledCellPatchDataset` when training infra disallows large mmap working sets, or when shipping a single tarballed dataset to a different machine.

### Pure-`webdataset` loader (no daas classes)

For users who prefer the canonical `webdataset` library — streaming pipeline, shard-aware multi-worker sharding, declarative decoders — `compiled/wds/` is directly compatible. A worked example lives at:

```
${SKILL_DIR}/examples/wds_only_example.py
```

Key points it demonstrates:
- Brace-expansion URL syntax: `shard-{000000..000099}.tar` reads many shards at once.
- Custom decoder for `.expr.npz` reconstructs the dense vector from `(indices, values)`.
- `wds.WebLoader` for PyTorch-compatible multi-worker iteration.

Install the optional dep:

```bash
pip install -e "${SKILL_DIR}[wds]"   # pulls webdataset>=0.2
```

Sketch of the pipeline (full file in `examples/wds_only_example.py`):

```python
import json, io, numpy as np, webdataset as wds
N_GENES = len(json.load(open("compiled/gene_panel.json")))

def decode_expr_npz(data):
    npz = np.load(io.BytesIO(data))
    expr = np.zeros(N_GENES, dtype=np.float32)
    expr[npz["indices"]] = npz["values"]
    return expr

# Shards are in per-sample subdirs: compiled/{sample_id}/shard-NNNNNN.tar
# Pass the full sorted list — WebDataset accepts a list of paths.
shards = sorted(Path("compiled").rglob("shard-*.tar"))
ds = (
    wds.WebDataset([str(s) for s in shards])
    .decode("pil",
            wds.handle_extension("expr.npz", decode_expr_npz),
            wds.handle_extension("json", lambda d: json.loads(d)))
    .to_tuple("jpg", "expr.npz", "json")
)
```

### LRU mmap cache

Each DataLoader worker inherits an empty cache (Linux fork semantics) and builds its own. No cross-worker sharing.

**mmap_cache_size sizing:**

```python
# Per worker: need to cache ≈ ceil(n_shards / num_workers) handles
# 72 samples × 138 shards/sample = ~10k shards
# 8 workers → ceil(10k/8) = 1250 per worker
# Default 128 causes cache thrash at this scale

ds = CellPatchDataset(..., mmap_cache_size=512)   # safe for most runs
```

| Scale | Shards total | Workers | Recommended cache size |
|-------|-------------|---------|------------------------|
| 2 samples, 40 shards | 40 | 8 | 16 (default 128 fine) |
| 72 samples, ~10k shards | 10000 | 8 | 512–1024 |

### Implementation sketch

```python
class LRUMmapCache:
    def __init__(self, maxsize=128):
        self._cache = OrderedDict()   # path → mmap.mmap
        self._files = {}              # path → file handle
        self.maxsize = maxsize

    def get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path); return self._cache[path]
        if len(self._cache) >= self.maxsize:
            ep, mm = self._cache.popitem(last=False)
            mm.close(); self._files.pop(ep).close()
        f  = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._files[path] = f; self._cache[path] = mm
        return mm

def __getitem__(self, idx):
    row      = self.manifest.iloc[idx]
    mm       = self._mmap.get(str(row["shard_path"]))
    jpg      = bytes(mm[int(row["tar_offset"]) : int(row["tar_offset"]) + int(row["jpg_size"])])
    img      = Image.open(io.BytesIO(jpg)).convert("RGB")
    if self.transform: img = self.transform(img)

    row_x = self.X[int(row["global_idx"])]
    expr  = (row_x.toarray() if issparse(row_x) else np.array(row_x)).reshape(-1).astype(np.float32)
    return {"image": img, "expression": expr,
            "cell_id": str(row["cell_id"]), "sample_id": str(row["sample_id"])}
```

### Memory footprint (5M cells × 5001 genes, ~1% sparsity)

| Component | Size |
|-----------|------|
| CSR expression matrix | ~4 GB |
| manifest DataFrame | ~1 GB |
| LRU mmap (512 shards) | OS page cache, on-demand |
| **Training process total** | **~5 GB** |

---

## Visualization Validation

`extract_sample.py` produces two viz files **before** writing any tile shards, so the user can sanity-check tile content + slide coverage and abort if something looks wrong:

- `{output}/viz/viz_global_tiles.png` — `lazyslide.pl.tiles` overview of every registered cell tile on the slide, dpi=300
- `{output}/viz/viz_patch_grid.png` — 5×5 random cells, each with cell-boundary (cyan), nucleus-boundary (yellow), and center crosshair (red) overlays, dpi=300

These are produced unconditionally — there is no `--skip-viz` flag. Generating them is cheap (~1–2 s for the patch grid; lpl.tiles renders a thumbnail) and the safety they provide is high.

### Optional standalone re-render

`viz_sample.py` lets you re-render the same two outputs after extraction, e.g. if you tweak rendering or want a fresh copy.

```bash
python3 "${SKILL_DIR}/scripts/viz_sample.py" \
    --zarr   /data/A_002.zarr \
    --output /data/out/A_002    # reads manifest.parquet from here
```

It writes the same `viz_global_tiles.png` and `viz_patch_grid.png` (reading JPEG content from existing shards for the patch grid).

### Boundary overlay coordinate transform

```python
SCALE = PATCH_SIZE / BASE_SIZE   # level-0 px → output px (e.g. 224/527)

def shape_um_to_patch_px(coords_um, x0, y0):
    arr  = np.array(coords_um)
    col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
    row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
    return np.column_stack([col, row_])

# cell boundary (cyan)
pts = shape_um_to_patch_px(list(cell_bounds.loc[cell_id].exterior.coords), x0, y0)
ax.add_patch(MplPolygon(pts, closed=True, edgecolor="cyan", facecolor="none", lw=0.8))

# nucleus boundary (yellow, if present)
if cell_id in nucl_ids:
    pts = shape_um_to_patch_px(list(nucl_bounds.loc[cell_id].exterior.coords), x0, y0)
    ax.add_patch(MplPolygon(pts, closed=True, edgecolor="yellow", facecolor="none", lw=0.8))

# crosshair at patch center (red)
cx = cy = PATCH_SIZE / 2; arm = PATCH_SIZE * 0.08
ax.plot([cx-arm, cx+arm], [cy, cy], color="red", lw=0.8)
ax.plot([cx, cx], [cy-arm, cy+arm], color="red", lw=0.8)
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `${SKILL_DIR}/scripts/inspect_spatialdata.py` | Print zarr tables/shapes/images |
| `${SKILL_DIR}/scripts/filter_tissue.py` | Tissue-inside filter (SOPA if needed) |
| `${SKILL_DIR}/scripts/filter_nucleus_presence.py` | Keep cells with nucleus boundary |
| `${SKILL_DIR}/scripts/filter_nucleus_overlap.py` | Xenium-vs-HE nucleus IoU filter |
| `${SKILL_DIR}/scripts/extract_sample.py` | Single-sample HE patch extraction |
| `${SKILL_DIR}/scripts/extract_all.py` | Parallel multi-sample extraction |
| `${SKILL_DIR}/scripts/compile_dataset.py` | Compile per-sample dirs; --samples flag |
| `${SKILL_DIR}/scripts/viz_sample.py` | Re-render viz outputs |
| `${SKILL_DIR}/daas/dataset.py` | LRUMmapCache + CellPatchDataset |
| `${SKILL_DIR}/daas/planning.py` | NL → StagePlan → render_cli |
| `${SKILL_DIR}/daas/reports.py` | StageReport + write_stage_report |
| `${SKILL_DIR}/daas/filters/nucleus_presence.py` | nucleus_presence filter logic |
| `${SKILL_DIR}/daas/filters/tissue.py` | tissue_inside filter logic |
| `${SKILL_DIR}/daas/filters/nucleus_overlap.py` | xenium_he_nucleus_overlap logic |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `struct.calcsize(fmt) != 32` | Wrong IDX_RECORD_FMT | Use `"<iQIQII"` (32B); assert at script top |
| `lpl.tiles(...)` returns None | lazyslide API | Call `lpl.tiles(...)` then `fig = plt.gcf()` |
| tar random read returns wrong JPEG | `tell()` during write is inaccurate | Two-phase: write-then-reread `offset_data` |
| `'>' not supported between 'tuple' and 'int'` in `get_thumbnail` | SpatialDataImage2DReader API | `except Exception: thumb = sdata.images[k]["scale4"]["image"].values.transpose(1,2,0)` |
| `AttributeError: 'DataTree' has no attribute 'shape'` | Multiscale image is DataTree | Use `sdata.images[k]["scale0"]["image"].shape` |
| `BASE_HALF` causes 0.5px offset | Integer division `BASE_SIZE // 2` | Use `BASE_SIZE / 2.0` (float) |
| subprocess cannot find lazyslide | Python interpreter mismatch | Use `sys.executable` (or ask user for interpreter path); never use `conda run` |
| `sample_key` read as int after parquet round-trip | Old CSV workflow; parquet preserves str | Use `manifest["sample_key"].astype(str)` after `read_parquet` for safety |
| `todense()` AttributeError on dense X | h5ad stored without sparse compression | `row_x.toarray() if issparse(row_x) else np.array(row_x)` |
| `h5ad rows != manifest rows` during compile | Different sort order for manifest vs h5ad | Both must use the same `sorted(sample_dirs)` list |
| mmap cache thrash at large scale | `mmap_cache_size=128` too small for 72 samples | Set `mmap_cache_size ≈ ceil(total_shards / num_workers)` |
| extract_all worker OOM | Too many workers × zarr in memory | Reduce `--workers`; each worker needs ~4 GB RAM |

## Pre-Flight Viz (before any tile shards are written)

The pipeline produces `viz_global_tiles.png` + `viz_patch_grid.png` **before** writing any shards. This lets the user verify that:

1. Tile positions cover the slide as expected (via `lazyslide.pl.tiles`).
2. Patch crops actually contain cell content with correct cell+nucleus boundary alignment.

If either looks wrong, abort and adjust config (MPP, patch size, shape key) before paying the cost of full extraction.

```python
# Register ALL cell tile positions on the WSI (needed for lpl.tiles + tile_images mode).
add_tiles(wsi, key="cell_tiles",
          xys=np.column_stack([sx0_ord, sy0_ord]),
          tile_spec=spec, tissue_ids=np.zeros(n_out, dtype=int))

# 1. lazyslide.pl.tiles overview — dpi=300, always produced
viz_dir = output_dir / "viz"
viz_dir.mkdir(exist_ok=True)
lpl.tiles(wsi, tile_key="cell_tiles")
fig = plt.gcf()
fig.savefig(viz_dir / "viz_global_tiles.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# 2. Patch grid: 25 random in-memory test patches + boundary overlays
n_grid = min(25, n_out)
rng_grid = np.random.default_rng(seed)
grid_idx = rng_grid.choice(n_out, n_grid, replace=False)
add_tiles(wsi, key="patch_grid",
          xys=np.column_stack([sx0_ord[grid_idx], sy0_ord[grid_idx]]),
          tile_spec=spec, tissue_ids=np.zeros(n_grid, dtype=int))
grid_images = [tile.image for tile in wsi.iter.tile_images("patch_grid")]
# Render overlays + save to viz_patch_grid.png at dpi=300 (see coord transform below)
```

### Boundary overlay coordinate transform

```python
SCALE = PATCH_SIZE / BASE_SIZE   # level-0 px → output px (e.g. 224/527)

def shape_um_to_patch_px(coords_um, x0, y0):
    arr  = np.array(coords_um)
    col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
    row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
    return np.column_stack([col, row_])

# cell boundary (cyan)
pts = shape_um_to_patch_px(list(cell_bounds.loc[cell_id].exterior.coords), x0, y0)
ax.add_patch(MplPolygon(pts, closed=True, edgecolor="cyan", facecolor="none", lw=0.8))

# nucleus boundary (yellow)
if cell_id in nucl_ids:
    pts = shape_um_to_patch_px(list(nucl_bounds.loc[cell_id].exterior.coords), x0, y0)
    ax.add_patch(MplPolygon(pts, closed=True, edgecolor="yellow", facecolor="none", lw=0.8))

# crosshair at patch center (red)
cx = cy = PATCH_SIZE / 2; arm = PATCH_SIZE * 0.08
ax.plot([cx-arm, cx+arm], [cy, cy], color="red", lw=0.8)
ax.plot([cx, cx], [cy-arm, cy+arm], color="red", lw=0.8)
```
