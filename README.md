# daas-compiler

Compile SpatialData zarr samples into task-specific, loader-ready spatial transcriptomics
training datasets. Three phases: extract per-sample patches → compile global manifest →
produce L4 training-ready artifacts via task adapters.

The repo is shipped as a Claude Code skill plugin: the AI agent reads
`skills/daas-compiler/SKILL.md` and runs the bundled scripts on the user's
behalf. The same scripts can be used directly without the agent.

## Vision

daas-compiler is not merely an HE patch extractor. It is an **agent-guided compiler**
that turns arbitrary spatial transcriptomics training requests into auditable, reproducible,
task-specific, loader-ready datasets.

The agent converts a user's natural-language training request into:
- an explicit stage plan (filtering → extraction → compile → task-ready packaging)
- optional filtering/preprocessing stages
- image-expression aligned compiled artifacts
- task-specific loader-ready outputs
- split-aware metadata for loader-runtime split selection
- validation reports

## Artifact Levels

| Level | Name | Script | Description |
|---|---|---|---|
| L0 | Raw | — | Input SpatialData zarr, untouched |
| L1 | Canonical | Preprocessing stages | Zarr after table/shape normalization and optional filtering stages |
| L2 | Patch-compiled | `extract_sample.py` | Per-sample JPEG shards + expression.h5ad + manifest.parquet + filter_report.json + visualizations |
| L3 | Dataset-compiled | `compile_dataset.py` | Cross-sample global manifest.parquet (with global_idx) + expression.h5ad (gene intersection) + optional gene_panel.json + compile_report |
| L4 | Task-ready / Training-ready | Task adapters | Task-specific, split-aware, loader-ready, I/O-optimized, validated artifacts |
| L5 | Benchmark-ready | Benchmark tooling | Frozen L4 artifacts with provenance hashes, fixed splits, reproducibility metadata |

## Training-Ready

A dataset is **training-ready** only when it can be consumed directly by the intended
training loader without additional preprocessing, joining, splitting, gene reordering,
image conversion, target conversion, or artifact conversion.

**Patch extraction ≠ training-ready:**
```
extract_sample.py → L2 (patch-compiled, per-sample)
compile_dataset.py → L3 (dataset-compiled, cross-sample)
```

**Split policy:** Splits are metadata applied by the loader at runtime. The default
training-ready layout stores all cells in a single canonical `data/` directory.
Physical `train/`, `val/`, `test/` shard directories are an optional export mode
(`--materialize-split-shards`), not the default.

**Training-ready (L4) for HE2ST — default canonical layout:**
```
data/
  shard-000000.tar  shard-000001.tar  ...

splits/
  train.json  val.json  test.json
  split_membership.parquet  split_report.json

gene_panel.json  gene_panel.sha256
task_config.yaml  loader_config.yaml
dataset_card.json  validation_report.json
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

SOPA belongs to `[preprocess]` only. Core extraction and task-ready packaging do not
require SOPA. See [`skills/daas-compiler/references/dependency-policy.md`](skills/daas-compiler/references/dependency-policy.md).

## Project Governance

- [`VERSIONING.md`](VERSIONING.md) — SemVer rules, schema versions, contract change policy
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Commit format, allowed scopes, PR checklist
- [`RELEASE.md`](RELEASE.md) — Release checklist and smoke tests
- [`CHANGELOG.md`](CHANGELOG.md) — All notable changes

## Install (via Claude Code)

```bash
/plugin marketplace add qizou97/daas-compiler
/plugin install daas-compiler@daas-compiler
```

The first time the skill activates in a project, the agent will run:

```bash
pip install -e "${SKILL_DIR}"
pip install -r "${SKILL_DIR}/requirements.txt"
```

For SOPA-backed preprocessing:
```bash
pip install -r "${SKILL_DIR}/requirements-preprocess.txt"
# or: pip install -e "${SKILL_DIR}[preprocess]"
```

## Manual install

```bash
git clone https://github.com/qizou97/daas-compiler.git
cd daas-compiler/skills/daas-compiler
pip install -e .
pip install -r requirements.txt
```

## Quick start

```bash
# 1. Extract patches from one sample (you'll be prompted for --extract-mode)
python3 scripts/extract_sample.py --zarr /data/sample.zarr --output /data/out/sample

# 2. Compile multiple per-sample dirs into a global dataset
python3 scripts/compile_dataset.py --per-sample-dir /data/out --output /data/compiled

# 2b. (Optional) Also write a self-contained bundled WebDataset.
#     Each cell becomes one tar entry: jpg + sparse .expr.npz + json.
#     Training from this output needs no mmap and no h5ad.
python3 scripts/compile_dataset.py --per-sample-dir /data/out --output /data/compiled \
    --bundle-wds

# 3. (Task-ready) Package compiled/ into L4 HE2ST training-ready artifacts.
#    Produces data/, splits/, task_config.yaml, loader_config.yaml, validation_report.json.
#    Splits are metadata — no physical train/val/test shard duplication.
python3 scripts/make_task_dataset.py \
    --compiled-dir /data/compiled \
    --output /data/task_ready \
    --task he2st \
    --split-ratios 0.8 0.1 0.1
```

### Loading L3 artifacts for research and iteration

Three loading paths consume **L3 (dataset-compiled)** artifacts. These are useful for
research and prototyping but are **not** L4 training-ready: splitting is handled in
userspace via `sample_ids`, and no `splits/`, `task_config.yaml`, or validation reports exist.

```python
# A. mmap-indexed random access (fastest per-cell read)
from daas.dataset import CellPatchDataset
ds = CellPatchDataset(
    manifest_path = "/data/compiled/manifest.parquet",
    h5ad_path     = "/data/compiled/expression.h5ad",
)

# B. Self-contained no-mmap loader (single dir, ships easily)
from daas.dataset import BundledCellPatchDataset
ds = BundledCellPatchDataset(compiled_dir="/data/compiled")

# C. Pure `webdataset` library pipeline (streaming, library-canonical)
# See skills/daas-compiler/examples/wds_only_example.py
#   pip install -e .[wds]
```

Full reference: [`skills/daas-compiler/SKILL.md`](skills/daas-compiler/SKILL.md).
User-facing prompts: [`skills/daas-compiler/usage-guide.md`](skills/daas-compiler/usage-guide.md).

## Tests

```bash
cd skills/daas-compiler
pip install -e .[test]
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
