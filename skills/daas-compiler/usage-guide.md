# Cell-Centered HE Patch Extraction - Usage Guide

## Overview

This skill covers extracting cell-centered H&E image patches from SpatialData objects into an indexed WebDataset for machine learning model training. It handles MPP derivation from affine transforms, OOB cell filtering, spatial sorting for cache locality, wsidata/lazyslide-based tile extraction, and alignment with gene expression data in h5ad format.

It also supports an optional **self-contained bundled WebDataset** output (each cell packaged as JPEG + sparse expression in one tar entry), so training can skip mmap and h5ad entirely. Three loading paths are available downstream: `CellPatchDataset` (mmap, fastest), `BundledCellPatchDataset` (no mmap, self-contained), and the canonical `webdataset` library pipeline.

## Prerequisites

```bash
pip install spatialdata wsidata lazyslide anndata numpy pandas scipy matplotlib Pillow
```

Optional for segmentation mask validation:
```bash
pip install zarr
```

## Quick Start

Tell your AI agent what you want to do:

- "Process my SpatialData into a cell-centered HE patch dataset"
- "Build an indexed patch dataset from {sdata_path} at mpp=0.5, size=224"
- "Extract HE patches around each cell centroid and align with expression"

## Example Prompts

### Standard processing
> "Process this SpatialData zarr into cell-centered HE patches: /path/to/data.zarr. Output to /path/to/output/. Target mpp=0.5, patch size=224."

### With custom filtering
> "Extract HE patches from my Xenium data at 0.5 mpp. Skip cells with zero expression and cells at image boundaries."

### With specific config
> "Build a patch dataset from /data/sample.zarr. HE image is 'he_image', centroids from 'cell_circles', expression from table 'table'. Use wsidata for all tiling."

### Bundled WebDataset for training without mmap
> "Compile my per-sample dirs at /data/out into /data/compiled and ALSO write a bundled WebDataset (each cell as jpg + sparse expression + json in one tar). Use `--bundle-wds`."

### Filtering-policy prompts

Reference: [`references/filtering.md`](references/filtering.md). Two layers run before any shard: **biological** (which table/shape rows enter) and **patch validity** (which selected cells produce a writable tile). Defaults preserve historic strict behavior.

**Inspect-only filtering plan (no extraction)**
> "Open /data/sample.zarr and tell me what filtering would apply under default policies. List which tables and shape layers are present (table, filtered_table, cell_circles, nucleus_boundaries, filtered_*), say which `--biological-filter-policy` would resolve under `auto`, and report the predicted `n_cells_source` from each candidate table. Do not extract anything."

**stVisuome canonical extraction**
> "Extract /data/A_028.zarr at mpp=0.5, patch=224, into /data/out/A_028. The zarr was preprocessed by `stvisuome-daas preprocess`, so use `--biological-filter-policy stvisuome_canonical` to consume `filtered_table`. Patch policy stays at default (`strict_no_padding`). Ask me which `--extract-mode` to use, then write `filter_report.json` and summarize it."

**stVisuome nucleus-boundary filtering (raw zarr, has nucleus_boundaries)**
> "Extract /data/raw/B_041.zarr into /data/out/B_041. The zarr has `table` + `nucleus_boundaries` but no `filtered_table`. Use `--biological-filter-policy stvisuome_nucleus_boundary` to keep only table rows whose cell_id is in `nucleus_boundaries.index`. Patch policy: default. After extraction, print the filter report and the per-reason drop counts."

**Strict compiler-safe extraction (fast full-image mode)**
> "Extract /data/A_002.zarr into /data/out/A_002 with `--extract-mode full_ops_level` and `--patch-filter-policy strict_no_padding` (the safe default for full-image modes — boundary-crossing tiles would otherwise be silently clipped). For Layer 1 leave `--biological-filter-policy auto`. Confirm `filter_report.json` shows `n_after_patch_policy` and zero `need_pad` survivors before writing shards."

**tile_images with stVisuome-minimal coordinate policy**
> "Use `--extract-mode tile_images --patch-filter-policy stvisuome_minimal` on /data/sample.zarr so we keep boundary-crossing tiles (wsidata pads them). Drop only `full_oob` and non-positive centroids. This is **not allowed** with `full_scale0` / `full_ops_level` — refuse the run if I ask you to combine them."

**Batch extraction with explicit policies**
> "Run `extract_all.py` over /data/spatialdata with 4 workers into /data/out. Forward `--biological-filter-policy stvisuome_canonical --patch-filter-policy strict_no_padding --extract-mode full_ops_level` to every worker. When all samples are done, list any failures and aggregate the per-sample `filter_report.json` totals into one summary table."

## What the Agent Will Do

1. **Inspect** the SpatialData structure and report all elements (images, shapes, tables, labels, points)
2. **Derive MPP** from affine transforms without guessing
3. **Present config** for your confirmation before processing — including the two **filtering policies** (`--biological-filter-policy` and `--patch-filter-policy`, see `references/filtering.md`)
4. **Apply Layer 1 (biological)** — pick `filtered_table` for stVisuome-canonical zarrs, mask by `nucleus_boundaries` if requested, or pass through with a warning
5. **Apply Layer 2 (patch validity)** — drop `full_oob` (and `need_pad` under `strict_no_padding`) tiles; require `tile_images` mode for `stvisuome_minimal`
6. **Spatial sort** cells by zarr chunk for efficient I/O
7. **Ask which extract mode to use** — the agent must prompt before running extraction. Pick one of:
   - `tile_images` — wsidata iterator, low memory (~50 MB), slow (1×)
   - `full_ops_level` (recommended) — load ops_level into memory, 36× faster, ~0.4 GB
   - `full_scale0` — load scale0 into memory, 9× faster, ~1.6 GB
8. **Pre-flight viz (before any shards are written)** — save `viz_global_tiles.png` (lazyslide.pl.tiles overview) and `viz_patch_grid.png` (25 random cells with cell+nucleus boundary overlays) at dpi=300 to `{output}/viz/`. Sanity check; abort if alignment looks wrong.
9. **Write `filter_report.json`** to `{output}/` *before* any tar shard — biological + patch policies applied, drop counts by reason, warnings. Always summarize it back to the user after extraction completes.
10. **Extract patches** using the chosen mode
11. **Write WebDataset** shards with binary .idx for O(1) random access
12. **Save aligned h5ad** with expression data in patch sample_index order
13. **Validate** alignment invariants (`expr_row == sample_index`, manifest `cell_id` ≡ `adata_out.obs.cell_id`, JPEG sizes, random-access reads)
14. **(Compile step)** Merge per-sample dirs into a global manifest + h5ad. With `--bundle-wds`, also write `{compiled}/wds/` with each cell as a self-contained tar entry (jpg + sparse `.expr.npz` + json), plus `gene_panel.json` for column order.

## Training the compiled dataset

Three loading paths produce the same `{image, expression, cell_id, sample_id}` shape:

| Path | When | Class / API |
|---|---|---|
| mmap | fastest per-cell read, random access at scale | `daas.dataset.CellPatchDataset(manifest_path, h5ad_path, ...)` |
| bundled (no mmap) | shipping one self-contained dir, bounded memory | `daas.dataset.BundledCellPatchDataset(wds_dir, ...)` |
| pure `webdataset` library | streaming pipelines, library-canonical workers | `webdataset.WebDataset(urls).decode(...)` — see `examples/wds_only_example.py` |

The bundled and pure-wds paths require `--bundle-wds` on compile.

## Tips

- **All HE tiling must use wsidata/lazyslide** — never numpy/zarr/PIL/cv2 for patch extraction
- **Confirm config before bulk processing** — the agent will ask if anything is ambiguous
- **mini pipeline first** — verify 5 cells end-to-end before full-scale extraction
- **MPP is derived from transforms** — no manual MPP guessing; stored in zarr metadata if available
- **OOB cells are skipped entirely** under `strict_no_padding` (default) — no padding, no partial extraction, no resizing of clipped regions
- **Two filtering layers, always distinguished** — biological (`--biological-filter-policy`: which table/shape rows enter) vs patch validity (`--patch-filter-policy`: which selected cells produce writable tiles). See [`references/filtering.md`](references/filtering.md)
- **Prefer canonical stVisuome outputs** when the zarr has them (`filtered_table`, `filtered_nucleus_boundaries`, `filtered_cell_boundaries`). `auto` picks them automatically; reach for `stvisuome_nucleus_boundary` only when the zarr has `nucleus_boundaries` but no `filtered_table`
- **Never run upstream preprocessing inside this skill** — no tissue segmentation, no Cellpose, no nucleus matching. Point the user at `stvisuome-daas preprocess` instead
- **`stvisuome_minimal` is `tile_images`-only** — combining it with `full_scale0` / `full_ops_level` is rejected at CLI parse time because those modes silently clip boundary tiles
- **`filter_report.json` is the contract** — always written before any shard, always summarized to the user. Includes `source_table_key`, drop counts by reason, and any warnings
- **Spatial sort** groups tiles by zarr chunk, improving cache hit rate ~2x
- **Binary .idx** format enables random access without sequential tar scan
- **Extraction mode is a required choice, not a default** — the agent will ask which of `tile_images`, `full_ops_level`, or `full_scale0` to use before running. Picking `full_ops_level` gives 36× speedup at ~0.4 GB memory; pick `tile_images` only when memory < 1 GB. See SKILL.md for benchmarks.
- **h5ad output** contains only valid cells in sample_index order with full alignment metadata
- **Patches are JPEG quality=95** — balance between file size and image quality for ML training
