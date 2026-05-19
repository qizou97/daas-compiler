# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `run_tissue_segmentation` now accepts `allow_holes` and `key_added` parameters; skips SOPA if `key_added` already exists in `sdata.shapes`
- `filter_tissue.py`: `--allow-holes` and `--key-added` CLI flags
- `filter_tissue.py`: saves `viz/tissue_overlay.png` after segmentation for visual QC
- `SKILL.md`: tissue segmentation confirmation rule — confirm `image_key` when multiple images exist; pass `--key-added` if shape already exists
- `references/sopa-integration.md`: documented correct SOPA API signature (`allow_holes`, `key_added`) and interactive notebook viz snippet
- Split policy: `splits/split_membership.parquet` and per-split JSON files (`splits/train.json` etc.) as the default L4 split representation
- `split_schema_version` and `loader_config_schema_version` added to schema version table in VERSIONING.md
- `release` and `deps` as explicit commit types in CONTRIBUTING.md
- Worked HE2ST example in `references/agent-contract.md` showing full stage plan through L4
- Table key propagation rule in agent contract
- Split policy rule in agent contract and SKILL.md: default is metadata, not physical shard partitioning

### Changed
- L4 training-ready default layout: `data/shard-*.tar` + `splits/` metadata (no physical `train/`, `val/`, `test/` shard directories by default)
- Physical `train/`, `val/`, `test/` shard partitioning is now an optional export mode only (`--materialize-split-shards`), not the default
- Per-cell shard JSON no longer includes `split` field — split is determined at loader runtime from `splits/split_membership.parquet`
- RELEASE.md smoke test step 3 updated to verify canonical layout and loader-runtime split selection
- CONTRIBUTING.md PR checklist expanded with split/loader/gene order/visualization checks
- VERSIONING.md: added explicit MAJOR/MINOR/PATCH sections and split/loader contract rules
- README.md: updated HE2ST training-ready layout to reflect canonical `data/` + `splits/` structure

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
