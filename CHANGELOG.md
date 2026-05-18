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
