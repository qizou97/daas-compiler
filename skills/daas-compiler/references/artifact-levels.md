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
