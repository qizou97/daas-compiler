# daas-compiler

Compile SpatialData zarr samples into a cell-centered HE patch WebDataset for
ML training. Three phases: extract per-sample → compile global manifest →
train via one of three loading paths (mmap, bundled tar, or raw
`webdataset` library).

The repo is shipped as a Claude Code skill plugin: the AI agent reads
`skills/daas-compiler/SKILL.md` and runs the bundled scripts on the user's
behalf. The same scripts can be used directly without the agent.

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

where `${SKILL_DIR}` is the installed plugin's
`skills/daas-compiler/` directory.

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
```

### Training paths

Three loading paths share the same `{image, expression, cell_id, sample_id}`
output shape; pick the one that fits your infra:

```python
# A. mmap-indexed random access (fastest per-cell read)
from daas.dataset import CellPatchDataset
ds = CellPatchDataset(
    manifest_path = "/data/compiled/manifest.parquet",
    h5ad_path     = "/data/compiled/expression.h5ad",
)

# B. Self-contained no-mmap loader (single dir, ships easily)
from daas.dataset import BundledCellPatchDataset
ds = BundledCellPatchDataset(wds_dir="/data/compiled/wds")

# C. Pure `webdataset` library pipeline (streaming, library-canonical)
# See skills/daas-compiler/examples/wds_only_example.py for the full
# pipeline. Install the optional dep first:
#   pip install -e .[wds]
```

Full reference: [`skills/daas-compiler/SKILL.md`](skills/daas-compiler/SKILL.md).
User-facing prompts: [`skills/daas-compiler/usage-guide.md`](skills/daas-compiler/usage-guide.md).
Pure-webdataset example: [`skills/daas-compiler/examples/wds_only_example.py`](skills/daas-compiler/examples/wds_only_example.py).

## Tests

```bash
cd skills/daas-compiler
pip install pytest
pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).
