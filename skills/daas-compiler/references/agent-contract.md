# Agent Contract: Natural Language → Auditable, Reproducible, Loader-Ready Dataset

## Overview

When a user makes a natural-language spatial transcriptomics training request,
the agent must convert it into an **explicit stage plan** before executing any
scripts. The agent must not begin extraction, compilation, or packaging without
first presenting the stage plan for user review.

The agent's goal is to produce a dataset that is:
- **task-specific** — structured for the intended ML task
- **split-aware** — split membership is metadata, consumed by the loader at runtime
- **loader-ready** — directly consumable without extra preprocessing, joining,
  gene reordering, image conversion, target conversion, or artifact conversion
- **I/O optimized** — canonical storage is efficient for the training loader
- **validated** — accompanied by filter reports, split reports, validation reports
- **auditable and reproducible** — every parameter and decision is recorded

---

## 1. Training-Ready Definition

**Training-ready does NOT mean patches were extracted.**
**Training-ready does NOT mean physically separating shards into train/val/test directories.**

Training-ready (L4) means the dataset can be consumed **directly** by the intended
training loader without any additional steps:

| Forbidden additional step | Example |
|---|---|
| Preprocessing | normalization, gene reordering, filtering |
| Joining | merging manifest with h5ad after the fact |
| Splitting | materializing train/val/test at load time from scratch |
| Gene reordering | aligning expression vectors to a fixed gene panel |
| Image conversion | JPEG → tensor (loader handles this) |
| Target conversion | raw counts → normalized expression, sparse → dense |
| Artifact conversion | h5ad → tensor, parquet → index |

### Artifact levels

| Level | Name | Description |
|---|---|---|
| L2 | patch-compiled | `extract_sample.py` output: per-sample shards + h5ad + manifest |
| L3 | dataset-compiled | `compile_dataset.py` output: global manifest + expression.h5ad |
| L4 | training-ready | task adapter output: canonical storage + split metadata + task/loader config + validation |

L2 and L3 are **not** training-ready.

### Split-pending vs. fully training-ready

| Status | Meaning |
|---|---|
| `training_ready` | All L4 artifacts present: split metadata, loader config, validation reports |
| `split_pending` | L2/L3/task skeleton complete but split metadata is missing |
| `compile_only` | L3 dataset-compiled, no task metadata |
| `patch_compiled_only` | L2 only |
| `validation_failed` | Artifacts present but validation checks failed |

### HE2ST training-ready artifacts

For a HE2ST task to be fully training-ready, **all** of the following must be present:

```
{task_output}/
  data/
    shard-000000.tar   ...   ← canonical WDS storage, NOT split-partitioned
  splits/
    split_membership.parquet
    train.json  val.json  test.json
    split_report.json
  gene_panel.json
  gene_panel.sha256
  task_config.yaml
  loader_config.yaml
  dataset_card.json
  validation_report.json
```

**Physical `train/`, `val/`, `test/` shard directories are NOT the default.**
They are an optional export mode only, triggered by `--materialize-split-shards`.

---

## 2. Critical Split Policy

> **Splits are metadata views over a canonical dataset.**

The default DAAS design keeps canonical storage stable and reusable:
- `global_idx` is stable
- `shard_path + sample_key` are stable
- `gene_panel` order is stable
- split membership is metadata
- loaders select by split membership at runtime

**Do NOT physically duplicate or rewrite WebDataset shards into train/, val/, test/ by default.**
Physical split-specific shards are allowed only as an explicit optional export mode:
`--materialize-split-shards`.

---

## 3. Missing-Split Behavior

If the user requests a task-ready or training-ready dataset but does **NOT** specify
train/val/test allocation, split policy, split file, or split ratios, the agent must
**NOT** invent a split silently.

### Split intent detection

The agent must treat these as **explicit split information**:
- train/val/test sample lists
- train/validation/test ratios (e.g. "80/10/10")
- holdout sample names
- cross-validation folds
- an existing split file path
- a `split_membership.parquet` file
- a grouping column plus split policy
- a benchmark split name

The agent must treat split information as **missing** when the user only says:
- "training-ready"
- "make dataset for training"
- "compile for HE2ST"
- "create loader-ready data"

...without specifying how train/val/test or folds should be assigned.

### When split information is missing, the agent must:

1. Tell the user that split metadata is required before a dataset can be
   considered fully training-ready.
2. Present reasonable split options (see Section 6).
3. Allow the user to choose one.
4. Allow the user to postpone split generation and produce a split-pending
   task package that can be completed later.

### The agent may continue with:
- L2 patch extraction
- L3 dataset compilation
- task metadata skeleton
- loader config skeleton
- split-pending validation report

### But must clearly label the result as:
- `"task-ready skeleton"` or `"split-pending"` — **not** fully training-ready.

### A split-pending package must include:
- `task_config.yaml`
- `loader_config.yaml` with `split_required: true`
- `dataset_card.json` with `training_ready: false`
- `validation_report.json` explaining missing split metadata
- Instructions or command for generating split metadata later

### A later split-generation stage must be able to add:
- `split_membership.parquet`
- `train.json`, `val.json`, `test.json`
- `split_report.json`

...without rewriting image/expression shards.

---

## 4. Parse User Intent

When the agent receives a natural-language training request, it must identify:

| Field | Examples |
|---|---|
| `task_type` | `he2st`, `celltype`, `contrastive`, `multimodal_pretraining`, `spatial_neighbor`, `custom` |
| `input_modality` | `he_image`, `if_image`, `multiplex_image` |
| `target_modality` | `gene_expression`, `cell_type`, `spatial_coordinates` |
| `sample_unit` | `cell`, `spot`, `tile`, `region`, `patient` |
| `requested_filters` | tissue, nucleus_presence, xenium_he_overlap |
| `requested_split_policy` | sample_holdout, ratio_by_group, group_kfold, existing_file |
| `requested_split_allocation` | explicit lists, ratios, folds |
| `requested_output_format` | webdataset, h5ad, zarr |
| `requested_loader_format` | CellPatchDataset, HE2STDataset, custom |
| `training_ready_requested` | whether user asks for task-ready / training-ready output |

### Trigger phrase → task_type mapping

| Phrase | task_type |
|---|---|
| "HE2ST", "HE → gene expression", "spatial transcriptomics prediction" | `he2st` |
| "cell type", "celltype classification" | `celltype` |
| "contrastive", "self-supervised" | `contrastive` |
| "multimodal pretraining", "foundation model" | `multimodal_pretraining` |
| "spatial neighbor", "neighborhood graph" | `spatial_neighbor` |

### Trigger phrase → filter stage mapping

| Phrase | Stage |
|---|---|
| "inside tissue", "out of tissue", "outside tissue" | `tissue_inside` |
| "with nucleus boundaries", "only cells with nucleus" | `nucleus_presence` |
| "Xenium nucleus overlaps HE nucleus", "overlap >" | `xenium_he_nucleus_overlap` |
| "optim_ops_level", "ops_level" | `extract_mode=full_ops_level` |
| "sample N cells per sample" | `--n-sample N` |

---

## 5. Inspect Inputs

Before constructing the stage plan, the agent must determine:

- sample list
- zarr paths
- image keys
- table keys (including `filtered_table` if present)
- shape keys (including `filtered_cell_boundaries`, `nucleus_boundaries` if present)
- nucleus/tissue artifacts
- existing filtered tables
- gene identifiers
- coordinate systems
- existing compiled artifacts (L2/L3 already done?)
- existing `gene_panel.json` / `gene_panel.sha256`
- existing split metadata if any
- existing `task_config.yaml` / `loader_config.yaml` if any

---

## 6. Create an Explicit Stage Plan

Every complex workflow must first produce a stage plan. The stage plan must be
presented to the user for approval before any script executes.

### Required stage plan fields

| Field | Description | Example |
|---|---|---|
| `task_type` | Training task | `"he2st"` |
| `samples` | List of sample identifiers | `["A_001", "A_002", "A_004"]` |
| `zarr_paths` | Absolute path to each sample's zarr | `["/data/A_001.zarr", ...]` |
| `input_modality` | Input data type | `"he_image"` |
| `target_modality` | Target data type | `"gene_expression"` |
| `initial_table_key` | Starting table key in zarr | `"table"` |
| `initial_image_key` | H&E image key in zarr | `"he_image"` |
| `initial_shape_key` | Cell shape key in zarr | `"cell_circles"` |
| `filter_stages` | Ordered list of filter stages | `["tissue_inside", "nucleus_presence"]` |
| `final_table_key` | Table key after all filter stages | `"table_tissue_nucleus"` |
| `final_shape_key` | Shape key to use at extraction | `"cell_circles"` |
| `extraction_config` | `extract_sample.py` parameters | `{mpp: 0.5, patch_size: 224, extract_mode: "full_ops_level", n_sample: 3000}` |
| `compile_config` | `compile_dataset.py` parameters | `{bundle_wds: false}` |
| `gene_panel_policy` | How to fix gene panel | `"intersect_all"` |
| `task_ready_config` | Task adapter parameters | `{task: "he2st"}` |
| `split_policy` | How splits are assigned | `"sample_holdout"` or `null` if missing |
| `split_metadata_paths` | Where split files will land | `"splits/split_membership.parquet"` |
| `loader_config_path` | Where loader_config.yaml will be written | `"he2st_task/loader_config.yaml"` |
| `reports` | Reports to produce | `["filter_report", "compile_report", "validation_report", "split_report"]` |
| `training_ready_status` | Final status of planned output | `training_ready` / `split_pending` / `compile_only` / `patch_compiled_only` |

### `training_ready_status` rules

| Condition | Status |
|---|---|
| All L4 artifacts planned, split info provided | `training_ready` |
| L2/L3/task skeleton planned, split info missing | `split_pending` |
| L3 compile only, no task metadata | `compile_only` |
| L2 extract only | `patch_compiled_only` |
| Artifacts present but validation checks fail | `validation_failed` |

When split information is missing, the stage plan **must** set:

```yaml
training_ready_status: split_pending
missing_requirements:
  - split_policy_or_split_file
```

---

## 7. Split Decision Protocol

When split information is missing, the agent must present the following options:

### Option A: sample_holdout

User provides explicit `train_samples`, `val_samples`, `test_samples`.

```
train_samples: [A_001, A_002]
val_samples:   [A_004]
test_samples:  []
```

Good when: samples/patients must not leak across splits.

### Option B: ratio_by_sample_group

User provides ratios (e.g. 80/10/10) and a grouping column (e.g. `sample_id`, `patient_id`).

```
ratios: {train: 0.8, val: 0.1, test: 0.1}
group_column: sample_id
seed: 42
```

Good when: preventing leakage by group while automating the assignment.

### Option C: group_kfold

User provides a grouping column and number of folds. Each group is assigned to
exactly one fold. Folds are rotated to produce k train/val splits.

```
group_column: patient_id
n_folds: 5
seed: 42
```

Good when: cross-validation is required across patient/sample/donor groups.

### Option D: existing_split_file

User provides path to an existing split file or `split_membership.parquet`.
The file may define splits by `sample_id`, `patient_id`, `donor_id`, `group_id`,
or `global_idx`.

```
split_file: /data/splits/benchmark_split.parquet
```

Good when: reproducing a benchmark experiment.

If the file is indexed by `global_idx` (cell-level), DAAS accepts it as an
externally provided benchmark split and emits a leakage warning: DAAS did not
generate this cell-level split and cannot verify that cells from the same
sample/patient do not appear in multiple splits.

### Option E: defer_split

Generate L2/L3/task skeleton now. Mark output as `split_pending`.
Run a split-generation command later to complete the dataset.

The agent must **not** silently pick one of these options unless the user
explicitly authorizes a default.

---

## 8. Execute Deterministic Stages

Each stage must have:
- inputs
- outputs
- output keys
- schema versions
- report path
- warnings
- validation checks

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

### Stages may include:
- `inspect`
- `tissue_filtering`
- `nucleus_presence_filtering`
- `xenium_he_nucleus_overlap_filtering`
- `patch_extraction` (L2)
- `dataset_compile` (L3)
- `gene_panel_validation`
- `task_metadata_skeleton_generation`
- `task_ready_packaging`
- `split_metadata_generation`
- `loader_config_generation`
- `visualization_validation`
- `post_save_tile_validation`

---

## 9. Preserve Invariants

### Table / shape invariants

- If a filter stage changes `table_key`, every downstream stage must use the new `table_key`.
- If a filter stage changes `shape_key`, every downstream stage must use the new `shape_key`.
- Never extract from a stale `table_key` after a filter stage.

```
filter_tissue.py    → writes table_tissue
  ↓ final_table_key = "table_tissue"
filter_nucleus_presence.py  → writes table_tissue_nucleus
  ↓ final_table_key = "table_tissue_nucleus"
extract_sample.py   --table-key table_tissue_nucleus
```

### Expression / gene invariants

- Gene order must be fixed by `gene_panel.json`.
- `gene_panel.sha256` must be written and referenced.
- Expression targets must follow `gene_panel.json` order.

### Manifest invariants

- `manifest` row `i` must match `expression` row `i` for compiled artifacts.
- `global_idx` must match compiled manifest row index.
- `global_idx` must remain stable across split definitions.

### Image / WDS invariants

- Every manifest row must resolve to an image sample.
- Every WDS sample metadata must include `global_idx`, `sample_id`, `cell_id`, and
  `gene_panel_sha256` when expression targets are present.
- WDS shards are canonical storage, not split-specific by default.

### Split invariants

- Split membership is metadata, not physical shard partitioning by default.
- `split_membership.parquet` must define split assignment when splits are available.
- `train/val/test.json` files reference `global_idx` or `sample_id` depending on split policy.
- Loaders must filter by split metadata at runtime.
- Changing split metadata must not require rewriting image/expression shards.
- If split metadata is missing, `loader_config` must mark `split_required: true`
  and `training_ready_status: split_pending`.
- **DAAS does not generate random cell-level train/val/test splits.**
  Generated splits must be sample-level or group-level (sample_holdout,
  ratio_by_group, or group_kfold).
- `split_membership.parquet` may be indexed by `global_idx` for loader efficiency,
  but every generated row must inherit its split assignment from `sample_id`,
  `patient_id`, `donor_id`, `slide_id`, `batch_id`, or another explicit group column.
  Individual cells must not be randomly scattered across splits.
- For `sample_holdout`: no `sample_id` may appear in more than one split.
- For `ratio_by_group` and `group_kfold`: no `group_id` may appear in more than
  one split.
- An externally provided `global_idx`-level split is accepted only via
  `existing_file`. DAAS emits a leakage warning because it cannot verify that
  cells from the same sample/patient are confined to one split.

### Loader invariants

- `loader_config.yaml` must specify how to find canonical storage.
- `loader_config.yaml` must specify split metadata location if split metadata exists.
- `loader_config.yaml` must require a split argument at runtime unless the task is explicitly unsplit.
- `HE2STDataset.from_config(loader_config, split="train")` must select train examples
  from split metadata, not from train-specific shard directories.
- If split metadata is missing, `HE2STDataset.from_config(..., split="train")` must fail
  with a clear message explaining how to generate split metadata.

---

## 10. Do Not Overclaim

The agent must not call L2 or L3 outputs training-ready.

| Output | Correct label |
|---|---|
| `extract_sample.py` output | L2 patch-compiled |
| `compile_dataset.py` output | L3 dataset-compiled |
| Task metadata without split metadata | split-pending task skeleton |
| Full L4 package with split metadata + loader config + validation | training-ready |

**Do not describe physical `train/`, `val/`, `test/` shard directories as the default layout.**
The default is canonical shards in `data/` with split membership as metadata.

---

## 11. Worked Examples

### Example A: HE2ST from SpatialData to fully training-ready with explicit splits

**User request:**
> "Process A_001,A_002,A_004 into a HE2ST training-ready dataset. Use A_001,A_002 as
> train and A_004 as validation. Filter tissue and nucleus cells. Use mpp=0.5,
> patch size=224, sample 3000 cells per sample."

**Split intent detected:** `sample_holdout` with `train=[A_001,A_002]`, `val=[A_004]`.

**Agent stage plan:**

```yaml
task_type: he2st
samples: [A_001, A_002, A_004]
zarr_paths: [/data/A_001.zarr, /data/A_002.zarr, /data/A_004.zarr]
initial_table_key: table
filter_stages: [tissue_inside, nucleus_presence]
final_table_key: table_tissue_nucleus
extraction_config: {mpp: 0.5, patch_size: 224, extract_mode: full_ops_level, n_sample: 3000}
compile_config: {bundle_wds: false}
task_ready_config: {task: he2st}
split_policy: sample_holdout
split_config:
  train_samples: [A_001, A_002]
  val_samples: [A_004]
  test_samples: []
loader_config: {split_metadata: splits/split_membership.parquet}
reports: [filter_report, compile_report, validation_report, split_report]
training_ready_status: training_ready
```

**Generated CLI (run in order):**

```bash
# Stage 0: inspect
python3 ${SKILL_DIR}/scripts/inspect_spatialdata.py --zarr .../A_001.zarr
# (repeat for A_002, A_004)

# Stage 1: tissue_inside
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr .../A_001.zarr \
    --input-table-key table --output-table-key table_tissue
# (repeat for A_002, A_004)

# Stage 2: nucleus_presence
python3 ${SKILL_DIR}/scripts/filter_nucleus_presence.py \
    --zarr .../A_001.zarr \
    --input-table-key table_tissue --output-table-key table_tissue_nucleus
# (repeat for A_002, A_004)

# Stage 3: extract  (L2)
python3 ${SKILL_DIR}/scripts/extract_sample.py \
    --zarr .../A_001.zarr \
    --output .../out/A_001 \
    --table-key table_tissue_nucleus \
    --extract-mode full_ops_level --mpp 0.5 --patch-size 224 --n-sample 3000
# (repeat for A_002, A_004)

# Stage 4: compile  (L3)
python3 ${SKILL_DIR}/scripts/compile_dataset.py \
    --per-sample-dir .../out \
    --output .../compiled \
    --samples A_001,A_002,A_004

# Stage 5: task-ready packaging  (L4)
python3 ${SKILL_DIR}/scripts/make_task_dataset.py \
    --compiled-dir .../compiled \
    --output .../he2st_task_ready \
    --task he2st \
    --split-policy sample_holdout \
    --train-samples A_001,A_002 \
    --val-samples A_004
```

**Expected L4 outputs (no physical train/val/test shard dirs):**

```
he2st_task_ready/
  data/shard-000000.tar  ...   ← all cells, NOT partitioned
  splits/
    train.json  val.json  test.json
    split_membership.parquet
    split_report.json
  gene_panel.json  gene_panel.sha256
  task_config.yaml  loader_config.yaml
  dataset_card.json  validation_report.json
```

---

### Example B: HE2ST request without split information

**User request:**
> "Process A_001,A_002,A_004 into a HE2ST training-ready dataset. Filter tissue
> and nucleus cells. Use mpp=0.5, patch size=224, sample 3000 cells per sample."

**Split intent detected:** None.

**Agent behavior:**

1. Detect missing split allocation.
2. Do not invent train/val/test split.
3. Tell the user:

> "Split metadata is required before this dataset can be considered fully
> training-ready. Please choose one of the following options:"
>
> **A. sample_holdout** — provide train_samples, val_samples, test_samples.
> All cells from the same sample_id are assigned to the same split.
>
> **B. ratio_by_group** — provide ratios (e.g. 80/10/10) and a grouping column
> (sample_id, patient_id, donor_id, slide_id, batch_id, or another group column).
> All cells from the same group are assigned to the same split.
>
> **C. group_kfold** — provide a grouping column and number of folds.
> Each group is assigned to exactly one fold.
>
> **D. existing_split_file** — provide path to an existing split file.
> Good for benchmark or reproducible experiments. If the file is global_idx-level,
> DAAS emits a leakage warning.
>
> **E. defer_split** — generate L2/L3/task skeleton now. Mark output as
> split-pending. Run a split-generation command later.

4. If the user chooses `defer_split`, the agent produces:

```
Stage plan with training_ready_status: split_pending
```

and proceeds to:
- L2 patch-compiled outputs
- L3 dataset-compiled outputs
- HE2ST `task_config.yaml`
- `loader_config.yaml` with `split_required: true`
- `dataset_card.json` with `training_ready: false`
- `validation_report.json` with `training_ready_status: split_pending`
- Instructions for completing split metadata later (see Example C)

---

### Example C: Later split generation over an existing compiled dataset

**User request:**
> "Use A_001,A_002 as train and A_004 as val for the HE2ST dataset we already compiled."

**Agent stage plan:**

```yaml
training_ready_status: training_ready  # will be training_ready after split generation
reuse_compiled_dir: .../compiled
split_policy: sample_holdout
split_config:
  train_samples: [A_001, A_002]
  val_samples: [A_004]
  test_samples: []
outputs:
  - splits/split_membership.parquet
  - splits/train.json
  - splits/val.json
  - splits/split_report.json
  - loader_config.yaml (updated split metadata path)
  - dataset_card.json (updated training_ready: true)
  - validation_report.json (updated)
```

**CLI:**

```bash
python3 ${SKILL_DIR}/scripts/make_split.py \
    --compiled-dir .../compiled \
    --task he2st \
    --policy sample_holdout \
    --train-samples A_001,A_002 \
    --val-samples A_004 \
    --output-split-dir .../compiled/splits/he2st
```

**Invariant:** This stage does **not** rewrite WDS shards. It only writes metadata files.

---

### Example D: Cell type classification reusing existing compiled patches

**User request:**
> "Create a cell-type classification dataset from the same compiled patches."

**Agent stage plan:**

```yaml
task_type: celltype
reuse_compiled_dir: .../compiled
resolve_label_column: celltype
validate_labels: true
split_policy: null   # → training_ready_status: split_pending until user provides split
training_ready_status: split_pending
missing_requirements:
  - split_policy_or_split_file
outputs:
  - tasks/celltype/task_config.yaml
  - tasks/celltype/loader_config.yaml (split_required: true)
  - tasks/celltype/dataset_card.json (training_ready: false)
  - tasks/celltype/validation_report.json
```

The agent must **not** rewrite image shards unless explicitly requested.
When the user provides split information, run `make_split.py` to add split metadata.

---

### Example E: Xenium nucleus / HE nucleus overlap filtering followed by HE2ST

**User request:**
> "Keep cells whose Xenium nucleus overlaps HE nucleus by more than 0.5, then
> create HE2ST training-ready data."

**Agent stage plan:**

```yaml
task_type: he2st
filter_stages: [xenium_he_nucleus_overlap]
filter_config:
  xenium_he_nucleus_overlap:
    min_overlap: 0.5
final_table_key: table_tissue_nucleus_he
split_policy: null   # → agent prompts for split info
training_ready_status: split_pending
missing_requirements:
  - split_policy_or_split_file
```

The agent prompts for split information before proceeding beyond L3.
If the user provides splits, `training_ready_status` becomes `training_ready`.

---

## 12. Split Generation Guidance

Split generation is a **separate stage** that can run after L3 compile.
It must not rewrite image/expression shards.

### `make_split.py` CLI

```bash
python3 ${SKILL_DIR}/scripts/make_split.py \
    --compiled-dir /data/compiled \
    --task he2st \
    --policy sample_holdout | ratio_by_group | group_kfold | existing_file \
    [--train-samples A_001,A_002] \
    [--val-samples A_004] \
    [--test-samples A_005] \
    [--ratios 0.8 0.1 0.1] \
    [--group-column sample_id] \
    [--n-folds 5] \
    [--seed 42] \
    [--split-file /data/splits/benchmark_split.parquet] \
    --output-split-dir /data/compiled/splits/he2st
```

**`random_cell` is not a supported policy.** If a user requests a random cell-level
split, the agent must explain that DAAS does not generate cell-level random splits
because they cause sample/patient leakage, and suggest `sample_holdout` or
`ratio_by_group` instead. The only way to use a cell-level split in DAAS is to
provide it as an external benchmark file via `--policy existing_file`.

### Outputs

| File | Description |
|---|---|
| `split_membership.parquet` | Per-cell: `global_idx`, `sample_id`, `cell_id`, `split` |
| `train.json` | Ordered list of `global_idx` values assigned to train |
| `val.json` | Ordered list of `global_idx` values assigned to val |
| `test.json` | Ordered list of `global_idx` values assigned to test |
| `split_report.json` | Method, seed, per-sample per-split counts |

### After running `make_split.py`

The agent must also:
- Update `loader_config.yaml` to reference the split metadata path
- Update `dataset_card.json` to set `training_ready: true`
- Validate that the loader can select split at runtime

### Loader validation

```python
ds = HE2STDataset.from_config(".../loader_config.yaml", split="train")
# Must succeed when split_membership.parquet is present
# Must fail with a clear message when split_membership.parquet is missing
```

---

## 13. Required Stage Plan Fields (reference table)

| Field | Description | Example |
|---|---|---|
| `task_type` | Training task | `"he2st"` |
| `samples` | List of sample identifiers | `["A_001", "A_002", "A_004"]` |
| `zarr_paths` | Absolute path to each sample's zarr | `["/data/A_001.zarr", ...]` |
| `input_modality` | Input data type | `"he_image"` |
| `target_modality` | Target data type | `"gene_expression"` |
| `initial_table_key` | Starting table key in zarr | `"table"` |
| `initial_image_key` | H&E image key in zarr | `"he_image"` |
| `initial_shape_key` | Cell shape key in zarr | `"cell_circles"` |
| `filter_stages` | Ordered list of filter stages to apply | `["tissue_inside", "nucleus_presence"]` |
| `final_table_key` | Table key after all filter stages | `"table_tissue_nucleus"` |
| `final_shape_key` | Shape key to use at extraction | `"cell_circles"` |
| `extraction_config` | `extract_sample.py` parameters | `{mpp: 0.5, patch_size: 224, extract_mode: "full_ops_level", n_sample: 3000}` |
| `compile_config` | `compile_dataset.py` parameters | `{bundle_wds: false}` |
| `gene_panel_policy` | Gene panel strategy | `"intersect_all"` |
| `task_ready_config` | Task adapter parameters | `{task: "he2st"}` |
| `split_policy` | Split assignment method (null if missing) | `"sample_holdout"` |
| `split_metadata_paths` | Where split files will be written | `"splits/split_membership.parquet"` |
| `loader_config_path` | Loader config output path | `"he2st_task/loader_config.yaml"` |
| `reports` | Reports to produce | `["filter_report", "compile_report", "validation_report", "split_report"]` |
| `training_ready_status` | Final output status | `training_ready` / `split_pending` / `compile_only` |

---

## 14. Stage Plan → CLI Mapping

After presenting and receiving user approval of the stage plan, the agent renders
the corresponding CLI commands using `daas.planning.render_cli()` or equivalent.

See `references/artifact-levels.md` for the L2/L3/L4 distinction.
See `references/training-ready-contract.md` for what L4 requires.
See `references/task-adapters.md` for task-specific L4 layouts.
