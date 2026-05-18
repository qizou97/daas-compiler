# daas-compiler Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add project governance docs, versioning policy, artifact level vocabulary, training-ready contracts, task adapter documentation, and dependency policy to daas-compiler — preparing the 0.6.x line for task-specific training-ready dataset generation.

**Architecture:** Vocabulary-first order — reference docs that define terminology are written before governance docs that reference that terminology. Root governance docs (VERSIONING, CONTRIBUTING, RELEASE, CHANGELOG) follow. File updates (README, SKILL.md) and code changes (pyproject.toml, requirements-preprocess.txt) come last. Final validation runs compileall and pytest.

**Tech Stack:** Markdown, TOML (pyproject.toml), git conventional commits. No new Python code. Version stays at 0.6.1.

---

## File Map

| # | Path | Action |
|---|---|---|
| 1 | `skills/daas-compiler/references/artifact-levels.md` | Create |
| 2 | `skills/daas-compiler/references/training-ready-contract.md` | Create |
| 3 | `skills/daas-compiler/references/agent-contract.md` | Create |
| 4 | `skills/daas-compiler/references/task-adapters.md` | Create |
| 5 | `skills/daas-compiler/references/dependency-policy.md` | Create |
| 6 | `VERSIONING.md` | Create |
| 7 | `CONTRIBUTING.md` | Create |
| 8 | `RELEASE.md` | Create |
| 9 | `CHANGELOG.md` | Create |
| 10 | `README.md` | Update (add vision, artifact levels, training-ready, links, extras) |
| 11 | `skills/daas-compiler/SKILL.md` | Update (add agent contract rules section) |
| 12 | `skills/daas-compiler/pyproject.toml` | Update (add preprocess + tasks extras) |
| 13 | `skills/daas-compiler/requirements-preprocess.txt` | Create |

---

### Task 1: Create artifact-levels.md

**Files:**
- Create: `skills/daas-compiler/references/artifact-levels.md`

- [ ] **Step 1: Write the file**

```markdown
# Artifact Levels

daas-compiler produces outputs at six levels. Each level builds on the previous.
Higher levels require task-specific configuration and cannot be produced automatically
from lower levels without explicit task adapter configuration.

## Level Definitions

| Level | Name | Description | Primary script/stage |
|---|---|---|---|
| L0 | Raw | Input SpatialData zarr, untouched | — |
| L1 | Canonical | Zarr after table/shape key normalization and optional filtering stages | filter_tissue.py, filter_nucleus_presence.py, filter_nucleus_overlap.py |
| L2 | Patch-compiled | Per-sample: JPEG shards, expression.h5ad, manifest.parquet, filter_report.json, viz/ | extract_sample.py |
| L3 | Dataset-compiled | Cross-sample: global manifest.parquet, expression.h5ad with gene intersection, optional bundled WDS shards | compile_dataset.py |
| L4 | Task-ready / Training-ready | Task-specific loader-ready artifacts: WebDataset shards by split, gene_panel.json, task_config.yaml, loader_config.yaml, dataset_card.json, validation reports | Task adapters (e.g. make_task_dataset.py) |
| L5 | Benchmark-ready | Frozen L4 artifacts with provenance hashes, fixed splits, reproducibility metadata | Benchmark tooling (future) |

## Key Clarifications

- `extract_sample.py` produces **L2** (patch-compiled) artifacts for one sample.
- `compile_dataset.py` produces **L3** (dataset-compiled) artifacts across all samples.
- Task adapters (e.g., `make_task_dataset.py` for HE2ST) produce **L4** (training-ready) artifacts.
- L4 is **task-specific**: the same L3 artifacts can feed multiple different L4 task adapters.
- L3 artifacts alone are NOT training-ready. They require a task adapter to become L4.

## Why These Levels Matter

The distinction prevents the common mistake of calling patch extraction "training-ready."
A training loop consuming L2 or L3 directly would need to handle splitting, gene reordering,
shard layout, and metadata assembly itself — work that belongs in the compiler, not the trainer.

See `references/training-ready-contract.md` for the full definition of what L4 requires.
See `references/task-adapters.md` for task-specific L4 artifact layouts.
```

- [ ] **Step 2: Verify**

```bash
ls skills/daas-compiler/references/
```
Expected: `artifact-levels.md` appears in the listing alongside the 5 existing files.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/references/artifact-levels.md
git commit -m "docs(skill): add artifact-levels reference (L0–L5 vocabulary)"
```

---

### Task 2: Create training-ready-contract.md

**Files:**
- Create: `skills/daas-compiler/references/training-ready-contract.md`

- [ ] **Step 1: Write the file**

```markdown
# Training-Ready Contract

## Definition

A dataset is **training-ready** (L4) only when it can be consumed directly by the
intended training loader **without** any of the following additional steps:

- Preprocessing (normalization, filtering, gene reordering)
- Joining (merging manifest with h5ad)
- Splitting (train/val/test materialization)
- Gene reordering (aligning expression vectors to a fixed gene panel)
- Image conversion (JPEG → tensor)
- Artifact conversion (h5ad → tensor, parquet → index)

If the training code must perform any of these steps, the dataset is not training-ready.

## Explicit Non-Examples

| Artifact | Level | Why NOT training-ready |
|---|---|---|
| Per-sample JPEG shards + expression.h5ad + manifest.parquet | L2 | Requires joining, splitting, global gene panel |
| Compiled global manifest.parquet + expression.h5ad | L3 | Requires splitting, task config, loader config |
| Bundled WDS shards without split files or gene_panel.json | L3b | Incomplete: missing split materialization and metadata |

## HE2ST Training-Ready Requirements

A dataset compiled for H&E → Spatial Transcriptomics prediction is training-ready when
ALL of the following artifacts are present and consistent:

### Required files

| File | Description |
|---|---|
| `train/shard-NNNNNN.tar`, `val/...`, `test/...` | WebDataset shards organized by split |
| `gene_panel.json` | Ordered list of gene names matching expression vector indices |
| `gene_panel.sha256` | SHA-256 hash of gene_panel.json for reproducibility |
| `task_config.yaml` | Task type, input/target modalities, gene panel ref, n_genes |
| `loader_config.yaml` | Shard paths, split sizes, batch size recommendations, num_workers |
| `dataset_card.json` | Dataset provenance: n_cells, n_genes, sample_ids, creation date, daas version |
| `validation_report.json` | Per-split cell counts, gene panel consistency check, shard integrity |
| `split_report.json` | Split assignment method, seed, per-sample split counts |

### Required per-cell shard content

Each tar entry must contain:
- `{key}.jpg` — JPEG-encoded patch image
- `{key}.expr.npz` — sparse expression target (`indices`, `values`, `n_genes`)
- `{key}.json` — metadata with required fields:

```json
{
  "global_idx": 12345,
  "sample_id": "A_002",
  "cell_id": "cell_000123",
  "split": "train",
  "task": "he2st",
  "n_genes": 313,
  "gene_panel_sha256": "abc123..."
}
```

### Required json fields

| Field | Type | Description |
|---|---|---|
| `global_idx` | int | Row index in the L3 compiled manifest/h5ad |
| `sample_id` | str | Sample identifier |
| `cell_id` | str | Original cell identifier from SpatialData |
| `split` | str | One of: `train`, `val`, `test` |
| `task` | str | Task type (e.g. `"he2st"`) |
| `n_genes` | int | Length of expression vector |
| `gene_panel_sha256` | str | SHA-256 of `gene_panel.json` for loader-side validation |

## Summary

> Patch extraction (L2) is patch-compiled, not training-ready.
> Compiled h5ad + manifest (L3) is dataset-compiled, not training-ready.
> Training-ready (L4) requires task-specific splits, gene panel, loader config, and validation artifacts.

See `references/artifact-levels.md` for the full level hierarchy.
See `references/task-adapters.md` for task-specific L4 layouts.
```

- [ ] **Step 2: Verify**

```bash
grep "gene_panel_sha256" skills/daas-compiler/references/training-ready-contract.md
```
Expected: line appears in the required json fields table and in the json example.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/references/training-ready-contract.md
git commit -m "docs(skill): add training-ready-contract reference (L4 definition + HE2ST requirements)"
```

---

### Task 3: Create agent-contract.md

**Files:**
- Create: `skills/daas-compiler/references/agent-contract.md`

- [ ] **Step 1: Write the file**

```markdown
# Agent Contract: Natural Language → Stage Plan

## Overview

When a user makes a natural-language spatial transcriptomics training request,
the agent must produce an explicit **stage plan** before executing any scripts.
The agent must not begin extraction, compilation, or packaging without first
presenting the stage plan for user review.

## Required Stage Plan Fields

A valid stage plan must specify all of the following:

| Field | Description | Example |
|---|---|---|
| `samples` | List of sample identifiers | `["A_001", "A_002", "A_004"]` |
| `zarr_paths` | Absolute path to each sample's zarr | `["/data/A_001.zarr", ...]` |
| `task_type` | Training task | `"he2st"` |
| `input_modality` | Input data type | `"he_image"` |
| `target_modality` | Target data type | `"gene_expression"` |
| `initial_table_key` | Starting table key in zarr | `"table"` |
| `initial_image_key` | H&E image key in zarr | `"he_image"` |
| `initial_shape_key` | Cell shape key in zarr | `"cell_circles"` |
| `filter_stages` | Ordered list of filter stages to apply | `["tissue_inside", "nucleus_presence"]` |
| `final_table_key` | Table key after all filter stages | `"table_tissue_nucleus"` |
| `final_shape_key` | Shape key to use at extraction | `"cell_circles"` |
| `extraction_config` | extract_sample.py parameters | `{mpp: 0.5, patch_size: 224, extract_mode: "full_ops_level", n_sample: 3000}` |
| `compile_config` | compile_dataset.py parameters | `{bundle_wds: true}` |
| `task_ready_config` | Task adapter parameters | `{task: "he2st", split_ratios: [0.8, 0.1, 0.1]}` |
| `split_config` | Split assignment config | `{method: "random", seed: 42, ratios: {train: 0.8, val: 0.1, test: 0.1}}` |
| `loader_config` | Loader-ready output config | `{format: "webdataset", shard_size: 500}` |
| `reports` | Reports and validation outputs to produce | `["filter_report", "compile_report", "validation_report", "split_report"]` |

## Training-Ready Gate

The agent must NOT describe outputs as "training-ready" unless the stage plan includes:

1. A `task_ready_config` specifying a task type and split configuration
2. A `split_config` with explicit train/val/test ratios
3. A `loader_config` specifying the output format
4. `"validation_report"` and `"split_report"` in the `reports` list

If the plan only covers filtering + extraction + compile (L2/L3), the agent must
describe the result as "patch-compiled" or "dataset-compiled" — not "training-ready."

## Stage Plan → CLI Mapping

After presenting and receiving user approval of the stage plan, the agent renders
the corresponding CLI commands using `daas.planning.render_cli()` or equivalent.

See `references/workflow-planning.md` for the stage → script mapping.
See `references/artifact-levels.md` for the L2/L3/L4 distinction.
See `references/training-ready-contract.md` for what L4 requires.
```

- [ ] **Step 2: Verify**

```bash
grep "Training-Ready Gate" skills/daas-compiler/references/agent-contract.md
```
Expected: heading found.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/references/agent-contract.md
git commit -m "docs(skill): add agent-contract reference (NL → stage plan requirements)"
```

---

### Task 4: Create task-adapters.md

**Files:**
- Create: `skills/daas-compiler/references/task-adapters.md`

- [ ] **Step 1: Write the file**

```markdown
# Task Adapters

Task adapters convert L3 (dataset-compiled) artifacts into L4 (training-ready) artifacts
for a specific training task. Each adapter is task-specific and produces a different
output layout.

---

## HE2ST (H&E → Spatial Transcriptomics)

Predict gene expression from H&E patch morphology.

| Property | Value |
|---|---|
| Input | L3 compiled manifest.parquet + expression.h5ad + per-sample JPEG shards |
| Target | Gene expression vector (normalized, gene-panel ordered) |
| Sample unit | Cell |
| Task type | `he2st` |

### Required L4 artifacts

- `train/shard-NNNNNN.tar`, `val/...`, `test/...` — WebDataset shards by split
- `gene_panel.json` — ordered gene names
- `gene_panel.sha256` — SHA-256 of gene_panel.json
- `task_config.yaml` — task_type, n_genes, gene_panel_path, input_modality, target_modality
- `loader_config.yaml` — train/val/test shard glob patterns, recommended batch_size, num_workers
- `dataset_card.json` — n_cells, n_genes, sample_ids, split counts, daas_version, created_at
- `validation_report.json` — per-split cell counts, gene panel hash check, shard count vs manifest count
- `split_report.json` — method, seed, per-sample per-split counts

### Per-cell shard content

Each tar entry contains:
- `{key}.jpg` — JPEG patch
- `{key}.expr.npz` — sparse expression (`indices`, `values`, `n_genes`)
- `{key}.json` — `{global_idx, sample_id, cell_id, split, task, n_genes, gene_panel_sha256}`

### Split requirements

- Train/val/test split materialized into separate shard directories before packaging
- Splits are assigned at the cell level; cell-level splits must not mix samples in val/test
- `split_report.json` must record split assignment method and per-sample counts

### Loader-ready layout

```
{output}/
  train/
    shard-000000.tar
    shard-000001.tar
    ...
  val/
    shard-000000.tar
    ...
  test/
    shard-000000.tar
    ...
  gene_panel.json
  gene_panel.sha256
  task_config.yaml
  loader_config.yaml
  dataset_card.json
  validation_report.json
  split_report.json
```

### Validation checks

- Gene panel consistency: `gene_panel.json` SHA-256 matches `gene_panel_sha256` in each per-cell JSON
- Shard cell count vs manifest cell count: must be equal per split
- No cell appears in more than one split
- All required files are present and non-empty

---

## Cell Type Classification

Predict cell type label from H&E patch morphology.

| Property | Value |
|---|---|
| Input | L3 compiled patches + cell type annotations from obs table |
| Target | Cell type label (integer class index) |
| Sample unit | Cell |
| Task type | `cell_type_classification` |

### Required L4 artifacts

- `train/`, `val/`, `test/` — WebDataset shards by split
- `label_map.json` — `{class_name: class_index}` mapping
- `task_config.yaml` — task_type, n_classes, label_map_path
- `loader_config.yaml` — shard patterns, batch_size, num_workers
- `dataset_card.json` — n_cells, n_classes, class_distribution, split counts
- `validation_report.json` — class distribution per split, label consistency check
- `split_report.json` — method, seed, per-class per-split counts

### Per-cell shard content

- `{key}.jpg` — JPEG patch
- `{key}.cls` — 4-byte little-endian int32 (class index)
- `{key}.json` — `{global_idx, sample_id, cell_id, split, task, class_name, class_idx}`

### Split requirements

Stratified split recommended: each split should preserve the overall class distribution.

---

## Contrastive / Multimodal Pretraining

Pair H&E patches with gene expression vectors for contrastive learning.

| Property | Value |
|---|---|
| Input | L3 compiled patches + expression.h5ad |
| Target | Paired (image, expression) for contrastive loss |
| Sample unit | Cell — positive pair is same cell's image + expression |
| Task type | `contrastive_he_expr` |

### Required L4 artifacts

- `train/`, `val/` — WebDataset shards (test split optional for pretraining)
- `gene_panel.json` + `gene_panel.sha256`
- `task_config.yaml` — task_type, contrastive_mode (`"image_expression"`), n_genes
- `loader_config.yaml` — shard patterns, recommended batch_size (large for contrastive)
- `dataset_card.json`
- `validation_report.json`

### Per-cell shard content

Same as HE2ST: `{key}.jpg` + `{key}.expr.npz` + `{key}.json`.
The loader pairs image and expression from the same tar entry as the positive pair.

### Split requirements

Random split at the cell level. No per-sample stratification required.

---

## Future: Spatial Neighborhood Tasks

Predict cell identity or behavior from the morphological context of its spatial neighborhood.

| Property | Value |
|---|---|
| Input | L3 patches + spatial graph (cell adjacency) |
| Target | Neighbor context embedding or neighborhood composition |
| Sample unit | Cell + k-hop neighborhood |
| Task type | `spatial_neighborhood` (placeholder) |

Contract TBD when implemented. Requires spatial graph export from SpatialData at L3.
```

- [ ] **Step 2: Verify**

```bash
grep -c "## " skills/daas-compiler/references/task-adapters.md
```
Expected: output is `7` (the 7 `##` headings in the file).

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/references/task-adapters.md
git commit -m "docs(skill): add task-adapters reference (he2st, cell-type, contrastive, future)"
```

---

### Task 5: Create dependency-policy.md

**Files:**
- Create: `skills/daas-compiler/references/dependency-policy.md`

- [ ] **Step 1: Write the file**

```markdown
# Dependency Policy

## Dependency Groups

| Group | pyproject.toml extra | Install command | Purpose |
|---|---|---|---|
| core | (base, always installed) | `pip install -e .` | anndata, numpy, pandas, scipy, Pillow, pyarrow — required for all loading and metadata |
| extract | `[extract]` | `pip install -e .[extract]` | spatialdata, wsidata, lazyslide, matplotlib — required for extraction scripts |
| preprocess | `[preprocess]` | `pip install -e .[preprocess]` | sopa, geopandas, shapely, scikit-image — required for SOPA-backed filtering stages |
| wds | `[wds]` | `pip install -e .[wds]` | webdataset — required for pure webdataset streaming pipeline |
| tasks | `[tasks]` | `pip install -e .[tasks]` | pyyaml — required for writing task_config.yaml and loader_config.yaml |
| test | `[test]` | `pip install -e .[test]` | pytest, geopandas, shapely — required for running the test suite |

## SOPA Rules

SOPA belongs to the `preprocess` group. The following rules are non-negotiable:

**1. SOPA must be lazy-imported.** Any code that uses SOPA must import it inside the
function or block that uses it, not at module level:

```python
# Correct — lazy import inside the function
def filter_with_sopa(sdata, ...):
    import sopa
    ...

# Wrong — breaks installs without the preprocess group
import sopa
def filter_with_sopa(sdata, ...):
    ...
```

**2. Plain extraction must work without SOPA.** `extract_sample.py`, `compile_dataset.py`,
and HE2ST task adapter packaging must complete successfully with only
`pip install -e .` and `pip install -e .[extract]`.

**3. The test suite must not require SOPA.** Tests that exercise SOPA-backed filtering
must guard the import:

```python
pytest.importorskip("sopa")
```

For SOPA integration details — API surface, filter stage implementation, tissue detection
— see `references/sopa-integration.md`.

## Adding New Dependencies

Before adding a new dependency, answer these questions in order:

1. Is it required for core loading or metadata functionality that runs unconditionally?
   → Add to `core`. Requires explicit review — affects all users.

2. Is it required only for extraction (spatialdata, wsidata, lazyslide)?
   → Add to `extract`.

3. Is it required only for SOPA-backed preprocessing?
   → Add to `preprocess`. Must be lazy-imported wherever used.

4. Is it required for task adapter output (task_config.yaml, loader_config.yaml)?
   → Add to `tasks`.

5. Is it a heavy optional dependency used only in one task adapter?
   → Add a new named extra for that task, or add to `tasks` if it's universal across task adapters.

6. Is it only needed for tests?
   → Add to `test`.

**New heavy dependencies must be optional** unless they are required for core
loading or metadata functionality that runs unconditionally at import time.
Do not add SOPA to `requirements.txt` or to `core`. `requirements.txt` covers
extraction/viz runtime only.
```

- [ ] **Step 2: Verify**

```bash
grep "lazy-imported" skills/daas-compiler/references/dependency-policy.md
```
Expected: line found in the SOPA rules section.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/references/dependency-policy.md
git commit -m "docs(deps): add dependency-policy reference (groups, SOPA rules, addition guidelines)"
```

---

### Task 6: Create VERSIONING.md

**Files:**
- Create: `VERSIONING.md` (repo root)

- [ ] **Step 1: Write the file**

```markdown
# Versioning

## Package and Skill Versions

daas-compiler uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

The package version (in `pyproject.toml`) and the skill version (referenced in `SKILL.md`
and marketplace metadata) are kept in sync. When you bump one, bump the other.

**Current baseline:** `0.6.1`

## Pre-1.0 Policy

While the version is below `1.0.0`:
- **Minor version bumps** (`0.x.0`) may include breaking changes to public APIs, output schemas, or artifact contracts. Document breaking changes prominently in `CHANGELOG.md`.
- **Patch version bumps** (`0.0.x`) are backwards-compatible bug fixes or documentation improvements.
- There is no deprecation period requirement before removing or changing APIs.

## Version Types

| Version type | Where it lives | What it governs |
|---|---|---|
| Code version | `skills/daas-compiler/pyproject.toml` → `version` | Python package |
| Skill version | `SKILL.md` frontmatter, marketplace metadata | Agent behavior contract |
| Artifact schema versions | JSON/YAML output field values | Shape of runtime output files |
| Task dataset contract versions | `task_config.yaml`, `dataset_card.json` | L4 training-ready artifact layout |

## Artifact Schema Versions

Each output file format has its own schema version field embedded in the file.
These are independent of the package version and must be incremented when the
format changes in a backwards-incompatible way.

| Schema version key | Output file | Bump when |
|---|---|---|
| `manifest_schema_version` | `manifest.parquet` (as metadata) | Column added, renamed, or removed |
| `filter_report_schema_version` | `filter_report.json` | Field added, renamed, or removed |
| `compile_report_schema_version` | `compile_report.json` | Field added, renamed, or removed |
| `wds_metadata_schema_version` | Per-cell `.json` inside shards | Field added, renamed, or removed |
| `gene_panel_schema_version` | `gene_panel.json` | Structure changes (currently a flat list) |
| `task_dataset_schema_version` | `dataset_card.json` | Field added, renamed, or removed |
| `split_schema_version` | `split_report.json` | Field added, renamed, or removed |
| `loader_config_schema_version` | `loader_config.yaml` | Field added, renamed, or removed |

Schema version fields use integer counters starting at `1`.

## When to Bump the Package Version

| Change | Version to bump |
|---|---|
| Bug fix, no output change | PATCH |
| New optional CLI flag, new optional output field | PATCH (also bump the relevant schema version) |
| New stage, new script, new task adapter | MINOR |
| Output schema change (field renamed or removed) | MINOR (also bump the relevant schema version) |
| API removal or breaking behavior change | MINOR (pre-1.0) or MAJOR (post-1.0) |
| Training-ready contract change (L4 layout) | MINOR — this is a contract change, not a docs change |

**Changing the definition or layout of training-ready task datasets is a contract
change, not just a docs change.** It must be accompanied by:
- A schema version bump for the affected output files
- A `CHANGELOG.md` entry under `[Unreleased]`
- Updated documentation in `references/training-ready-contract.md` and the relevant
  task adapter in `references/task-adapters.md`
- Updated tests

## How to Bump the Version

1. Update `version` in `skills/daas-compiler/pyproject.toml`
2. Update `CHANGELOG.md`: move `[Unreleased]` items to a new version section, add the date
3. Commit: `chore(release): bump to vX.Y.Z`
4. Tag: `git tag vX.Y.Z`

See `RELEASE.md` for the full release checklist.
```

- [ ] **Step 2: Verify**

```bash
grep "manifest_schema_version" VERSIONING.md
```
Expected: line found in the artifact schema versions table.

- [ ] **Step 3: Commit**

```bash
git add VERSIONING.md
git commit -m "docs: add VERSIONING.md (semver rules, schema versions, contract change policy)"
```

---

### Task 7: Create CONTRIBUTING.md

**Files:**
- Create: `CONTRIBUTING.md` (repo root)

- [ ] **Step 1: Write the file**

```markdown
# Contributing

## Commit Format

This repo uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <description>

[optional body]

[optional footer: Co-Authored-By, Fixes, etc.]
```

### Types

Standard conventional commit types apply: `feat`, `fix`, `chore`, `docs`,
`refactor`, `test`, `perf`, `ci`.

### Allowed Scopes

| Scope | Covers |
|---|---|
| `skill` | SKILL.md, agent behavior contract |
| `docs` | Documentation files (README, CONTRIBUTING, VERSIONING, RELEASE) |
| `planning` | daas/planning.py, stage plan parsing, CLI rendering |
| `filters` | daas/filters/, filter stage scripts |
| `extract` | scripts/extract_sample.py, scripts/extract_all.py |
| `compile` | scripts/compile_dataset.py |
| `viz` | daas/viz.py, scripts/viz_sample.py |
| `dataset` | daas/dataset.py, CellPatchDataset, BundledCellPatchDataset |
| `wds` | WebDataset integration, bundled shard writing |
| `genes` | Gene panel, gene intersection logic |
| `reports` | daas/reports.py, filter_report, compile_report, validation_report |
| `tasks` | Task adapters, make_task_dataset.py |
| `splits` | Split assignment, split_report |
| `loaders` | loader_config.yaml, loader utilities |
| `deps` | pyproject.toml dependency changes |
| `release` | Version bumps, changelog, tagging |
| `tests` | tests/ |

### Examples

```
feat(extract): add --tissue-shapes-key flag for flexible shape key selection
fix(compile): correct gene intersection when sample has zero common genes
chore(release): bump to v0.7.0
docs(skill): add training-ready contract section
refactor(dataset): extract LRUMmapCache into its own module
```

## PR Checklist

Before opening a pull request:

- [ ] `python -m compileall skills/daas-compiler` passes (no syntax errors)
- [ ] `cd skills/daas-compiler && pytest tests -q` passes (no failures or errors)
- [ ] If an output schema changed: schema version field bumped and documented in `VERSIONING.md`
- [ ] If a task-ready output changed: `references/training-ready-contract.md` and `references/task-adapters.md` updated
- [ ] If a new dependency was added: follows `references/dependency-policy.md` (correct group, lazy import for preprocess)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Commit messages use allowed scopes from the table above

## Test Expectations

- Tests live in `skills/daas-compiler/tests/`
- Tests must not require the `extract` group (spatialdata, wsidata, lazyslide) unless
  explicitly testing extraction. Guard with `pytest.importorskip` where needed.
- Tests that require SOPA must guard with `pytest.importorskip("sopa")`
- The lightweight `[test]` group provides geopandas and shapely for filter integration tests
- New behavior must have tests before the PR merges

## Dependency Rules

See `references/dependency-policy.md` for the full policy. Key points:

- SOPA must be lazy-imported inside functions that use it
- New heavy dependencies must be optional extras, not added to core
- `requirements.txt` covers extraction/viz runtime; do not add SOPA or task extras there

## Output Contract Changes

Any change that alters the shape, fields, or semantics of output files must:

1. Bump the affected schema version(s) — see `VERSIONING.md`
2. Update `references/training-ready-contract.md` if L4 artifacts are affected
3. Update `references/task-adapters.md` if the relevant task adapter's artifact list changes
4. Update or add tests that assert the new schema

Changing output contracts silently is not acceptable, even for "minor" field additions.
```

- [ ] **Step 2: Verify**

```bash
grep -c "| \`" CONTRIBUTING.md
```
Expected: output is `17` (the 17 scope rows in the table).

- [ ] **Step 3: Commit**

```bash
git add CONTRIBUTING.md
git commit -m "docs: add CONTRIBUTING.md (conventional commits, scopes, PR checklist, contract rules)"
```

---

### Task 8: Create RELEASE.md

**Files:**
- Create: `RELEASE.md` (repo root)

- [ ] **Step 1: Write the file**

```markdown
# Release Checklist

Use this checklist when cutting a new release from the `main` branch.

## 1. Pre-Release Validation

Run both commands and confirm exit code 0:

```bash
python -m compileall skills/daas-compiler
```
Expected: `Compiling ...` lines for each .py file, no `SyntaxError` output, exit code 0.

```bash
cd skills/daas-compiler && pytest tests -q
```
Expected: all tests pass. Exit code 0. No failures, errors, or xfail surprises.

## 2. Mini-Pipeline Smoke Test

Run a minimal end-to-end extraction + compile. Use an existing small test zarr or the
test fixtures in `skills/daas-compiler/tests/`:

```bash
# Extraction smoke test (requires pip install -e .[extract])
python3 skills/daas-compiler/scripts/extract_sample.py \
    --zarr <path_to_small_test_zarr> \
    --output /tmp/daas_smoke/sample_A \
    --n-sample 50 \
    --extract-mode tile_images

# Verify L2 outputs exist
ls /tmp/daas_smoke/sample_A/
# Expected: shard-000000.tar, shard-000000.idx, expression.h5ad, manifest.parquet,
#           filter_report.json, viz/viz_global_tiles.png, viz/viz_patch_grid.png

# Compile smoke test
python3 skills/daas-compiler/scripts/compile_dataset.py \
    --per-sample-dir /tmp/daas_smoke \
    --output /tmp/daas_smoke/compiled

# Verify L3 outputs exist
ls /tmp/daas_smoke/compiled/
# Expected: manifest.parquet, expression.h5ad
```

## 3. HE2ST Task-Ready Smoke Test

If the HE2ST task adapter (`make_task_dataset.py`) is available:

```bash
python3 skills/daas-compiler/scripts/make_task_dataset.py \
    --compiled-dir /tmp/daas_smoke/compiled \
    --output /tmp/daas_smoke/task_ready \
    --task he2st \
    --split-ratios 0.8 0.1 0.1 \
    --seed 42

# Verify L4 outputs exist
ls /tmp/daas_smoke/task_ready/
# Expected: train/, val/, test/, gene_panel.json, gene_panel.sha256,
#           task_config.yaml, loader_config.yaml, dataset_card.json,
#           validation_report.json, split_report.json
```

If the task adapter is not yet implemented, mark this step N/A and note it in the
release changelog.

## 4. Version Bump

Update `version` in `skills/daas-compiler/pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

## 5. Changelog Update

In `CHANGELOG.md`:

1. Add a new version section above `[Unreleased]`:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

2. Move all items from `[Unreleased]` into the new version section.
3. Leave `[Unreleased]` empty (with placeholder subsections) for future entries.

## 6. Commit and Tag

```bash
git add skills/daas-compiler/pyproject.toml CHANGELOG.md
git commit -m "chore(release): bump to vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 7. Post-Release

- Verify the tag is visible on GitHub: `git ls-remote --tags origin`
- If the skill is published to the marketplace, update marketplace metadata
- Announce any breaking changes (schema version bumps, contract changes) in the
  GitHub release notes
```

- [ ] **Step 2: Verify**

```bash
grep "compileall" RELEASE.md
```
Expected: appears in step 1 and the pre-release validation block.

- [ ] **Step 3: Commit**

```bash
git add RELEASE.md
git commit -m "docs: add RELEASE.md (checklist, smoke tests, version bump steps)"
```

---

### Task 9: Create CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md` (repo root)

- [ ] **Step 1: Write the file**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Deprecated

### Removed

### Security

---

## [0.6.1] - 2026-05-19

### Added
- Gene order contract: `compile_dataset.py` writes `gene_panel.json` with the gene intersection list
- Tissue overlay and post-save visualization in `extract_sample.py`: `--tissue-shapes-key`, `--cell-boundaries-key`, `--nucleus-boundaries-key` flags
- `daas/viz.py`: overlay key resolvers, tiles overview, patch grid, saved patch grid
- Project governance: `VERSIONING.md`, `CONTRIBUTING.md`, `RELEASE.md`, `CHANGELOG.md`
- Reference vocabulary: artifact levels (L0–L5), training-ready contract, agent contract, task adapters, dependency policy
- Dependency extras: `[preprocess]` (sopa, geopandas, shapely, scikit-image) and `[tasks]` (pyyaml)
- `requirements-preprocess.txt` for SOPA-backed filtering dependencies

### Changed

### Fixed
- `extract_sample.py` quality: use `validate_report`, remove redundant imports, fix phase label
- `viz.py` quality: narrow except, snake_case params, extract `_um_to_px`, decode_errors test
```

- [ ] **Step 2: Verify**

```bash
grep "\[Unreleased\]" CHANGELOG.md && grep "\[0.6.1\]" CHANGELOG.md
```
Expected: both lines found.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG.md (Keep-a-Changelog, 0.6.1 baseline)"
```

---

### Task 10: Update README.md

**Files:**
- Modify: `README.md` (repo root)

The current README has this structure:
1. `# daas-compiler` + intro paragraph
2. `## Install (via Claude Code)`
3. `## Manual install`
4. `## Quick start`
5. Training paths
6. `## Tests`
7. `## License`

Insert new content after the intro paragraph (after line 10, before `## Install`).

- [ ] **Step 1: Read current intro block to get exact insertion point**

Read `README.md` lines 1–12.

The intro paragraph ends at: `The same scripts can be used directly without the agent.`

- [ ] **Step 2: Insert new sections after the intro paragraph**

After the line `The same scripts can be used directly without the agent.` and before
`## Install (via Claude Code)`, insert:

```markdown

## Vision

daas-compiler is not merely an HE patch extractor. It is an AI/training-ready compiler
for arbitrary spatial transcriptomics training tasks. The agent converts a user's
natural-language training request into a stage plan, optional filtering/preprocessing
stages, image-expression aligned compiled artifacts, task-specific loader-ready outputs,
split-aware WebDataset I/O layout, and validation reports.

## Artifact Levels

| Level | Name | Script | Description |
|---|---|---|---|
| L2 | Patch-compiled | `extract_sample.py` | Per-sample JPEG shards + expression.h5ad + manifest |
| L3 | Dataset-compiled | `compile_dataset.py` | Cross-sample global manifest + expression with gene intersection |
| L4 | Training-ready | Task adapters | Split-aware shards + gene panel + loader config + validation |

## Training-Ready

A dataset is **training-ready** only when it can be consumed directly by the intended
training loader without additional preprocessing, splitting, or artifact conversion.

**Patch extraction ≠ training-ready:**
```
extract_sample.py → L2 (patch-compiled, per-sample)
compile_dataset.py → L3 (dataset-compiled, cross-sample)
```

**Training-ready (L4) for HE2ST requires:**
```
train/shard-NNNNNN.tar  val/...  test/...
gene_panel.json  gene_panel.sha256
task_config.yaml  loader_config.yaml
dataset_card.json  validation_report.json  split_report.json
```

See [`skills/daas-compiler/references/training-ready-contract.md`](skills/daas-compiler/references/training-ready-contract.md) for the full contract.

## Dependency Extras

| Extra | Command | Adds |
|---|---|---|
| `[extract]` | `pip install -e .[extract]` | spatialdata, wsidata, lazyslide, matplotlib |
| `[preprocess]` | `pip install -e .[preprocess]` | sopa, geopandas, shapely, scikit-image |
| `[wds]` | `pip install -e .[wds]` | webdataset |
| `[tasks]` | `pip install -e .[tasks]` | pyyaml |
| `[test]` | `pip install -e .[test]` | pytest, geopandas, shapely |

## Project Governance

- [`VERSIONING.md`](VERSIONING.md) — SemVer rules, schema versions, contract change policy
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Commit format, allowed scopes, PR checklist
- [`RELEASE.md`](RELEASE.md) — Release checklist and smoke tests

```

- [ ] **Step 3: Apply the edit**

In `README.md`, find the exact line:
```
The same scripts can be used directly without the agent.
```
And insert the new sections immediately after it, before `## Install (via Claude Code)`.

- [ ] **Step 4: Verify**

```bash
grep "Training-Ready" README.md && grep "Artifact Levels" README.md && grep "VERSIONING.md" README.md
```
Expected: all three lines found.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README with vision, artifact levels, training-ready def, extras table, governance links"
```

---

### Task 11: Update SKILL.md

**Files:**
- Modify: `skills/daas-compiler/SKILL.md`

Insert a new section **before** `## Version Compatibility` (the current first section).

- [ ] **Step 1: Insert agent contract rules section**

Find the line:
```
## Version Compatibility
```

Insert immediately before it:

```markdown
## Agent Contract

These rules govern the agent's behavior when executing daas-compiler workflows.
They cannot be overridden by conversational context.

**Stage plan required.** The agent must produce and present an explicit stage plan
before executing any multi-step workflow (filtering → extraction → compile → task-ready).
See `references/agent-contract.md` for required stage plan fields.

**Preserve the training-ready contract.** The agent must not describe outputs as
"training-ready" unless a task-ready packaging stage has produced splits, loader-ready
artifacts, and validation reports. See `references/training-ready-contract.md`.

**Distinguish artifact levels.** The agent must use correct level terminology in all
responses:
- L2 = patch-compiled (`extract_sample.py` output)
- L3 = dataset-compiled (`compile_dataset.py` output)
- L4 = training-ready (task adapter output, task-specific)

**Follow versioning and commit rules.** When modifying the skill (SKILL.md, scripts,
daas/ package), the agent must follow the commit scopes in `CONTRIBUTING.md` and note
any schema version bumps required by `VERSIONING.md`.

**No silent behavior changes.** The agent must not silently change default filtering,
extraction, or task-ready packaging behavior. Any change to defaults must be announced
to the user and reflected in the stage plan before execution.

---

```

- [ ] **Step 2: Verify**

```bash
grep "Agent Contract" skills/daas-compiler/SKILL.md && grep "L4 = training-ready" skills/daas-compiler/SKILL.md
```
Expected: both lines found.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/SKILL.md
git commit -m "docs(skill): add Agent Contract section (stage plan, training-ready gate, level vocab, rules)"
```

---

### Task 12: Update pyproject.toml and create requirements-preprocess.txt

**Files:**
- Modify: `skills/daas-compiler/pyproject.toml`
- Create: `skills/daas-compiler/requirements-preprocess.txt`

- [ ] **Step 1: Add preprocess and tasks extras to pyproject.toml**

Current `[project.optional-dependencies]` block ends with:
```toml
test = [
  "pytest>=7",
  # daas.filtering integration tests build a small in-memory SpatialData
  # stand-in using geopandas + shapely. They do NOT require the heavyweight
  # extract group (wsidata / lazyslide / matplotlib) — CLI-parser tests use
  # the lightweight daas.cli_args module that does not import those deps.
  "geopandas>=0.14",
  "shapely>=2.0",
]
```

After the `test` block (before the closing line or next section), insert:

```toml
preprocess = [
  "sopa",
  "geopandas>=0.14",
  "shapely>=2.0",
  "scikit-image",
]
tasks = [
  "pyyaml>=6",
]
```

- [ ] **Step 2: Create requirements-preprocess.txt**

```
sopa
geopandas>=0.14
shapely>=2.0
scikit-image
```

- [ ] **Step 3: Verify pyproject.toml is valid TOML and version is still 0.6.1**

```bash
python3 -c "
import tomllib
with open('skills/daas-compiler/pyproject.toml', 'rb') as f:
    d = tomllib.load(f)
print('version:', d['project']['version'])
print('preprocess:', d['project']['optional-dependencies']['preprocess'])
print('tasks:', d['project']['optional-dependencies']['tasks'])
"
```
Expected output:
```
version: 0.6.1
preprocess: ['sopa', 'geopandas>=0.14', 'shapely>=2.0', 'scikit-image']
tasks: ['pyyaml>=6']
```

- [ ] **Step 4: Verify requirements-preprocess.txt**

```bash
cat skills/daas-compiler/requirements-preprocess.txt
```
Expected:
```
sopa
geopandas>=0.14
shapely>=2.0
scikit-image
```

- [ ] **Step 5: Commit**

```bash
git add skills/daas-compiler/pyproject.toml skills/daas-compiler/requirements-preprocess.txt
git commit -m "deps: add [preprocess] and [tasks] optional extras; add requirements-preprocess.txt"
```

---

### Task 13: Final Validation

- [ ] **Step 1: Run compileall**

```bash
python -m compileall skills/daas-compiler
```
Expected: `Compiling ...` output for each `.py` file, no `SyntaxError`, exit code 0.

- [ ] **Step 2: Run pytest**

```bash
cd skills/daas-compiler && pytest tests -q
```
Expected: all tests pass, exit code 0. Note the pass/fail count and confirm no regressions.

- [ ] **Step 3: Confirm version is still 0.6.1**

```bash
python3 -c "import tomllib; f=open('skills/daas-compiler/pyproject.toml','rb'); d=tomllib.load(f); print(d['project']['version'])"
```
Expected: `0.6.1`

- [ ] **Step 4: Confirm all 13 deliverables exist**

```bash
ls skills/daas-compiler/references/agent-contract.md \
   skills/daas-compiler/references/artifact-levels.md \
   skills/daas-compiler/references/training-ready-contract.md \
   skills/daas-compiler/references/task-adapters.md \
   skills/daas-compiler/references/dependency-policy.md \
   VERSIONING.md CONTRIBUTING.md RELEASE.md CHANGELOG.md \
   skills/daas-compiler/requirements-preprocess.txt
```
Expected: all 10 paths listed without error.

```bash
grep "Vision" README.md && grep "Agent Contract" skills/daas-compiler/SKILL.md && grep "preprocess" skills/daas-compiler/pyproject.toml
```
Expected: all three lines found.

- [ ] **Step 5: Final summary commit (if any loose changes remain)**

If all validation passes and no files are unstaged:

```bash
git status
```
Expected: clean working tree. All changes committed in tasks 1–12.
