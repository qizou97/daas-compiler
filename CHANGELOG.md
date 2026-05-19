# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.7.8] - 2026-05-19

### Added
- `extract_all.py`: `--samples` flag (comma-separated) to restrict parallel extraction to a subset of samples in `--zarr-dir`

### Changed
- `SKILL.md` Agent Contract: four new rules
  - Multi-sample parallel extraction must use `extract_all.py`; background bash loops are prohibited
  - Pre-compile verification: agent must confirm `manifest.parquet` + `expression.h5ad` exist in all sample dirs before running `compile_dataset.py`
  - Monitor tool output must not be used to assess extraction completion or read cell counts
  - Agent must read `filter_report.json` after every extraction and report non-zero `drop_counts_by_reason` (`full_oob`, `need_pad`) to the user
- `SKILL.md` worked example: added `--nucleus-boundaries-key nucleus_boundaries` explicitly to `filter_nucleus_presence.py` call; replaced per-sample `extract_sample.py` loop with `extract_all.py --samples`

---

## [0.7.6] - 2026-05-19

### Changed (breaking vs 0.7.0)
- `run_tissue_segmentation`: restored `allow_holes: bool = False` and `key_added: str = "tissue"` parameters, passed through to `sopa.segmentation.tissue`
- `run_tissue_segmentation`: no-new-key result is now a `TissueKeyExistsWarning` (not `RuntimeError`); falls back to `key_added` or first known tissue key
- `filter_tissue.py`: restored `--allow-holes` and `--key-added` CLI flags (defaults `False` / `"tissue"`)
- `filter_tissue.py`: prints a clear warning if `--key-added` already exists before running SOPA, so the agent can surface this to the user
- `SKILL.md`: updated tissue segmentation guidance — agent must ask user before running `filter_tissue.py` when `key_added` shape already exists; documents re-run vs. reuse options

---

## [0.7.5] - 2026-05-19

### Fixed
- `daas/planning.py`: `StagePlan` was missing `task_type` (str, default `"he2st"`) and `filter_stages` (property returning `[s.name for s in self.stages]`); agent code that accessed these attributes raised `AttributeError`
- `SKILL.md`: updated `parse_stage_plan()` example to document all real `StagePlan` attributes so agents access existing fields, not imagined ones

---

## [0.7.4] - 2026-05-19

### Fixed
- `SKILL.md`: agent was guessing `SKILL_DIR` from the plugin cache layout instead of reading the `Base directory for this skill:` line emitted by the skill loader, causing the first `scripts/` path attempt to be wrong. Now explicitly instructs the agent to use that system-provided path verbatim.

---

## [0.7.3] - 2026-05-19

### Fixed
- `viz.save_tiles_overview`: tissue overlay was scaled using the cell_circles affine (`scale_shape`) instead of the tissue shapes' own SpatialData transformation, causing the tissue boundary to appear crammed into a corner. Now looks up `get_transformation(tissue_gdf, to_coordinate_system="global")` and applies the tissue shapes' own (x-scale, y-scale, x-translation, y-translation) to polygon vertices. Falls back to `scale_shape` with a warning if the lookup fails.

---

## [0.7.2] - 2026-05-19

### Fixed
- `SKILL.md`: agent must stop and confirm with user before using a pre-existing `filtered_table` found in a zarr — never pass it silently
- `SKILL.md`: agent must stop and confirm before reusing an existing tissue shape from `sdata.shapes` — never silently pass `--key-added`
- `SKILL.md`: stage plan must disclose all keys that will be written into the source zarrs; filter scripts must not run until user approves the plan

---

## [0.7.1] - 2026-05-19

### Added
- Implementation plan for fixing SOPA filter fallbacks (`docs/superpowers/plans/2026-05-19-fix-sopa-filter-fallbacks.md`)

---

## [0.7.0] - 2026-05-19

### Changed (breaking)
- `run_tissue_segmentation`: removed `allow_holes` and `key_added` parameters; always calls SOPA unconditionally and raises `RuntimeError` if no new shape key is created — eliminates silent reuse of pre-existing shapes
- `filter_tissue.py`: removed `--allow-holes` and `--key-added` CLI flags

### Fixed
- `filter_tissue.py`: `_save_tissue_viz` now imports `spatialdata_plot` explicitly so the `.pl` accessor is registered on `sdata`; added `spatialdata-plot` and `matplotlib` to the `preprocess` optional-dependency group
- `viz.save_tiles_overview`: expands axes to full slide extent after `lpl.tiles()` so `viz_global_tiles.png` shows global tissue context instead of zooming to the tiles bounding box; added `region_of_interest` to tissue key candidates

---

## [0.6.5] - 2026-05-19

### Fixed
- `SKILL.md`: `--n-sample` is now strictly opt-in — agent must never add it
  unless the user explicitly specifies a cell count; prevents silent subsampling
  when no count was requested

---

## [0.6.4] - 2026-05-19

### Fixed
- `run_tissue_segmentation`: fall back to existing known tissue key (`region_of_interest`, `tissue_boundaries`, `tissue`) when SOPA updates a shape in-place instead of creating a new one
- `filter_tissue.py`: `--allow-holes` now accepts `--allow-holes false` in addition to the bare flag, preventing argparse error from callers that pass an explicit boolean string

---

## [0.6.3] - 2026-05-19

### Added
- `run_tissue_segmentation`: `allow_holes` and `key_added` parameters; skips SOPA when `key_added` already exists in `sdata.shapes`
- `filter_tissue.py`: `--allow-holes` and `--key-added` CLI flags
- `filter_tissue.py`: saves `viz/tissue_overlay.png` after segmentation for visual QC
- `SKILL.md`: tissue segmentation confirmation rule — confirm `image_key` when multiple images exist; pass `--key-added` if shape already exists
- `references/sopa-integration.md`: correct SOPA API signature (`allow_holes`, `key_added`) and interactive notebook viz snippet
- Split policy: `splits/split_membership.parquet` and per-split JSON files as default L4 split representation
- `split_schema_version` and `loader_config_schema_version` in VERSIONING.md schema table
- `release` and `deps` commit types in CONTRIBUTING.md
- Worked HE2ST example in `references/agent-contract.md`
- Table key propagation rule and split policy rule in agent contract and SKILL.md

### Changed
- L4 training-ready default layout: `data/shard-*.tar` + `splits/` metadata (no physical `train/`/`val/`/`test/` shard dirs by default)
- Physical shard partitioning is now opt-in via `--materialize-split-shards`
- Per-cell shard JSON no longer includes `split` field — determined at loader runtime from `splits/split_membership.parquet`
- VERSIONING.md: added explicit MAJOR/MINOR/PATCH sections and split/loader contract rules
- README.md: updated HE2ST training-ready layout

### Fixed

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
