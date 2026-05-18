# Artifact Levels

daas-compiler produces outputs at six levels. Each level builds on the previous.
Higher levels require task-specific configuration and cannot be produced automatically
from lower levels without explicit task adapter configuration.

## Level Definitions

| Level | Name | Description | Primary script/stage |
|---|---|---|---|
| L0 | Raw | Input SpatialData zarr, untouched | — |
| L1 | Canonical | Zarr after table/shape key normalization and optional filtering stages | filter_tissue.py, filter_nucleus_presence.py, filter_nucleus_overlap.py |
| L2 | Patch-compiled | Per-sample: JPEG shards, expression.h5ad, manifest.parquet, filter_report.json, viz/ (visualizations) — **not training-ready** | extract_sample.py |
| L3 | Dataset-compiled | Cross-sample: global manifest.parquet (with global_idx), expression.h5ad (gene intersection), gene_panel.json (optional, with --bundle-wds), compile_report — **not necessarily training-ready** | compile_dataset.py |
| L4 | Task-ready / Training-ready | Task-specific, split-aware, loader-ready, I/O-optimized, validated: canonical `data/shard-*.tar`, split metadata in `splits/`, gene_panel.json, task_config.yaml, loader_config.yaml, dataset_card.json, validation reports — **splits are metadata, not physical shard partitions** | Task adapters (e.g. make_task_dataset.py) |
| L5 | Benchmark-ready | Frozen L4 artifacts with provenance hashes, fixed splits, reproducibility metadata | Benchmark tooling (future) |

## Key Clarifications

- `extract_sample.py` produces **L2** (patch-compiled) artifacts for one sample.
- `compile_dataset.py` produces **L3** (dataset-compiled) artifacts across all samples.
- Task adapters (e.g., `make_task_dataset.py` for HE2ST) produce **L4** (training-ready) artifacts.
- L4 is **task-specific**: the same L3 artifacts can feed multiple different L4 task adapters.
- L3 artifacts alone are NOT training-ready. They require a task adapter to become L4.

## Split Policy at L4

L4 artifacts use **split metadata**, not physical shard partitioning:

- All cells are stored in `data/shard-*.tar` (canonical storage, not split-partitioned).
- Split membership is determined by `splits/split_membership.parquet` at loader runtime.
- Physical `train/`, `val/`, `test/` shard directories are an optional export mode only
  (`--materialize-split-shards`), not the default L4 representation.

The same canonical `data/` storage can serve multiple split definitions (e.g., an 80/10/10
split and a 70/15/15 split) without duplicating or rewriting shards.

## Why These Levels Matter

The distinction prevents the common mistake of calling patch extraction "training-ready."
A training loop consuming L2 or L3 directly would need to handle splitting, gene reordering,
shard layout, and metadata assembly itself — work that belongs in the compiler, not the trainer.

See `references/training-ready-contract.md` for the full definition of what L4 requires.
See `references/task-adapters.md` for task-specific L4 artifact layouts.
