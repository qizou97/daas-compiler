# daas-compiler Governance, Versioning, and Training-Ready Contracts

**Date:** 2026-05-19
**Status:** Approved
**Baseline version:** 0.6.1

---

## Scope

Add project governance, versioning policy, artifact-level vocabulary, training-ready contracts, task adapter documentation, and dependency policy to the daas-compiler repo. This prepares the 0.6.x line for task-specific training-ready dataset generation while making the broader DAAS vision explicit.

No version bump in this change set. Version bump rules are defined in VERSIONING.md.

---

## Vision

daas-compiler is not merely an HE patch extractor. It is an AI/training-ready compiler for arbitrary spatial transcriptomics training tasks. The agent converts a user's natural-language training request into:

1. A stage plan
2. Optional filtering/preprocessing stages
3. Image-expression aligned compiled artifacts
4. Task-specific loader-ready outputs
5. Split-aware WebDataset or equivalent I/O layout
6. Validation reports and visualizations

**Terminology anchor:**
- Patch extraction alone is **not** training-ready.
- A compiled manifest + h5ad is **not** automatically training-ready.
- Training-ready means: task-specific, split-aware, loader-ready, I/O optimized, and validated.

---

## Execution approach

Vocabulary-first (approach B):

1. Reference vocabulary docs (`artifact-levels`, `training-ready-contract`, `agent-contract`, `task-adapters`, `dependency-policy`)
2. Root governance docs (`VERSIONING`, `CONTRIBUTING`, `RELEASE`, `CHANGELOG`)
3. File updates (`SKILL.md`, `README.md`)
4. Code changes (`pyproject.toml`, `requirements-preprocess.txt`)
5. Validation (`compileall`, `pytest`)

---

## Section 1: New reference vocabulary docs

All 5 files go in `skills/daas-compiler/references/`.

### agent-contract.md

How the agent converts natural-language ST training requests into explicit stage plans.

A stage plan must include:
- samples and input zarr paths
- task_type, input modality, target modality
- initial table/image/shape keys
- filtering/preprocessing stages
- final table_key and shape_key for extraction
- extraction config, compile config, task-ready packaging config
- split config, loader config
- reports and validation outputs

The agent must not call patch extraction "training-ready" unless a task-ready packaging stage has also produced splits and loader-ready artifacts.

### artifact-levels.md

| Level | Name | Script/stage |
|---|---|---|
| L0 | Raw | Input zarr |
| L1 | Canonical | After key normalization |
| L2 | Patch-compiled | `extract_sample.py` output |
| L3 | Dataset-compiled | `compile_dataset.py` output |
| L4 | Task-ready / Training-ready | Task adapter output (task-specific) |
| L5 | Benchmark-ready | Frozen splits + provenance |

L4 is task-specific. The same L3 artifacts can be the input to multiple L4 task adapters.

### training-ready-contract.md

A dataset is training-ready only when it can be consumed directly by the intended training loader without additional preprocessing, joining, splitting, gene reordering, image conversion, or artifact conversion.

For HE2ST, required artifacts:
- WebDataset shards organized by split
- `gene_panel.json` + `gene_panel.sha256`
- `task_config.yaml`
- `loader_config.yaml`
- `dataset_card.json`
- `validation_report.json`
- `split_report.json`
- Per-cell: jpg + expression target + json metadata
- Per-cell json fields: `global_idx`, `sample_id`, `cell_id`, `split`, `task`, `n_genes`, `gene_panel_sha256`

Explicit statements:
- Patch extraction is L2 (patch-compiled), not training-ready.
- Compiled h5ad + manifest is L3 (dataset-compiled), not necessarily training-ready.

### task-adapters.md

Documents task-specific adapters:

**he2st**
- Input: L3 compiled manifest + expression h5ad + extracted jpg patches
- Target: gene expression vector (normalized, gene-panel ordered)
- Sample unit: cell
- Required artifacts: WebDataset shards, split files, gene_panel.json, task_config.yaml, loader_config.yaml, dataset_card.json, validation_report.json
- Split requirements: train/val/test materialized into separate shards
- Loader layout: WebDataset tar shards, one cell per entry (jpg + .expr.npz + .json)
- Validation: gene panel consistency check, shard count vs manifest count, split coverage

**Cell type classification**
- Input: L3 patches + cell type annotations
- Target: cell type label
- Sample unit: cell
- Required artifacts: shards + label_map.json + split files

**Contrastive / multimodal pretraining**
- Input: L3 patches + expression
- Target: paired (image, expression) for contrastive loss
- Sample unit: cell pair or single cell
- Required artifacts: shards organized for contrastive sampling

**Future: spatial neighborhood tasks**
- Input: L3 patches + spatial graph
- Target: neighbor context embedding
- Placeholder — contract TBD when implemented

### dependency-policy.md

Dependency groups:

| Group | pyproject extra | Purpose |
|---|---|---|
| core | (base install) | anndata, numpy, pandas, scipy, Pillow, pyarrow |
| extract | `[extract]` | spatialdata, wsidata, lazyslide, matplotlib |
| preprocess | `[preprocess]` | sopa, geopandas, shapely, scikit-image |
| wds | `[wds]` | webdataset |
| tasks | `[tasks]` | pyyaml |
| test | `[test]` | pytest, geopandas, shapely |

Rules:
- SOPA belongs to `preprocess`. It must be lazy-imported inside SOPA-backed preprocessing/filtering stages only.
- Plain extraction, compile, and HE2ST packaging from already-compiled artifacts must work without SOPA installed.
- New heavy dependencies must be optional unless required for core loading/metadata functionality.
- See `sopa-integration.md` for SOPA integration details.

---

## Section 2: Root governance docs and file updates

### VERSIONING.md

- SemVer: `MAJOR.MINOR.PATCH`
- Pre-1.0 policy: minor bumps may include breaking changes; document in CHANGELOG
- Baseline: 0.6.1 (code version = skill version for now)
- Distinction: code version, skill version, artifact schema versions, task dataset contract versions
- 8 named artifact schema versions: `manifest_schema_version`, `filter_report_schema_version`, `compile_report_schema_version`, `wds_metadata_schema_version`, `gene_panel_schema_version`, `task_dataset_schema_version`, `split_schema_version`, `loader_config_schema_version`
- Changing the definition or layout of training-ready task datasets is a contract change, not just a docs change

### CONTRIBUTING.md

- Conventional Commit format
- 17 allowed scopes: `skill`, `docs`, `planning`, `filters`, `extract`, `compile`, `viz`, `dataset`, `wds`, `genes`, `reports`, `tasks`, `splits`, `loaders`, `deps`, `release`, `tests` (these are scopes, not types; types follow conventional commits: feat, fix, chore, etc.)
- PR checklist, test expectations, dependency rules
- Any output contract change must update docs, tests, and schema version notes
- Any task-ready output change must update relevant task adapter docs

### RELEASE.md

- Release checklist
- Mini-pipeline smoke test
- HE2ST task-ready smoke test
- Version bump steps, changelog update, tagging format `vX.Y.Z`
- Required validation:
  ```
  python -m compileall skills/daas-compiler
  cd skills/daas-compiler && pytest tests -q
  ```

### CHANGELOG.md

- Keep-a-Changelog style
- `[Unreleased]` section at top
- `[0.6.1]` as current baseline
- Placeholders for Added / Changed / Fixed / Deprecated / Removed / Security

### README.md (update)

Add to existing README:
- DAAS-wide vision statement
- Artifact level summary table
- Training-ready definition (patch extraction ≠ training-ready, WebDataset + splits + loader_config = training-ready)
- HE2ST example illustrating L2 → L3 → L4
- Links to VERSIONING.md, CONTRIBUTING.md, RELEASE.md
- Dependency extras summary table

Preserve existing structure (install, quick start, training paths).

### SKILL.md (update)

Add to existing SKILL.md:
- Agent must create a stage plan before executing complex workflows
- Agent must preserve the training-ready contract (see `references/training-ready-contract.md`)
- Agent must distinguish L2 patch-compiled, L3 dataset-compiled, and L4 training-ready
- Agent must follow versioning/schema/commit rules when modifying the skill
- Agent must not silently change default filtering/extraction/task-ready behavior

### pyproject.toml (update)

Add optional extras:
```toml
[project.optional-dependencies]
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

Version stays at 0.6.1. No changes to core dependencies.

### requirements-preprocess.txt (new)

```
sopa
geopandas>=0.14
shapely>=2.0
scikit-image
```

`requirements.txt` stays unchanged.

---

## Validation

```bash
python -m compileall skills/daas-compiler
cd skills/daas-compiler && pytest tests -q
```

---

## Deliverables summary

| # | File | Action |
|---|---|---|
| 1 | `skills/daas-compiler/references/agent-contract.md` | Create |
| 2 | `skills/daas-compiler/references/artifact-levels.md` | Create |
| 3 | `skills/daas-compiler/references/training-ready-contract.md` | Create |
| 4 | `skills/daas-compiler/references/task-adapters.md` | Create |
| 5 | `skills/daas-compiler/references/dependency-policy.md` | Create |
| 6 | `VERSIONING.md` | Create |
| 7 | `CONTRIBUTING.md` | Create |
| 8 | `RELEASE.md` | Create |
| 9 | `CHANGELOG.md` | Create |
| 10 | `README.md` | Update |
| 11 | `skills/daas-compiler/SKILL.md` | Update |
| 12 | `skills/daas-compiler/pyproject.toml` | Update |
| 13 | `skills/daas-compiler/requirements-preprocess.txt` | Create |
