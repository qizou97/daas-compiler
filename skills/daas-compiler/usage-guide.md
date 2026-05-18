# Cell-Centered HE Patch Extraction - Usage Guide

## Overview

This skill covers extracting cell-centered H&E image patches from SpatialData objects into an indexed WebDataset for machine learning model training. It handles MPP derivation from affine transforms, OOB cell filtering, spatial sorting for cache locality, wsidata/lazyslide-based tile extraction, and alignment with gene expression data in h5ad format.

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

## What the Agent Will Do

1. **Inspect** the SpatialData structure and report all elements (images, shapes, tables, labels, points)
2. **Derive MPP** from affine transforms without guessing
3. **Present config** for your confirmation before processing
4. **Filter cells**: remove OOB (out-of-bounds) and zero-expression cells
5. **Spatial sort** cells by zarr chunk for efficient I/O
6. **Ask which extract mode to use** — the agent must prompt before running extraction. Pick one of:
   - `tile_images` — wsidata iterator, low memory (~50 MB), slow (1×)
   - `full_ops_level` (recommended) — load ops_level into memory, 36× faster, ~0.4 GB
   - `full_scale0` — load scale0 into memory, 9× faster, ~1.6 GB
7. **Pre-flight viz (before any shards are written)** — save `viz_global_tiles.png` (lazyslide.pl.tiles overview) and `viz_patch_grid.png` (25 random cells with cell+nucleus boundary overlays) at dpi=300 to `{output}/viz/`. Sanity check; abort if alignment looks wrong.
8. **Extract patches** using the chosen mode
9. **Write WebDataset** shards with binary .idx for O(1) random access
10. **Save aligned h5ad** with expression data in patch sample_index order
11. **Validate** with 6-point verification suite including random access checks

## Tips

- **All HE tiling must use wsidata/lazyslide** — never numpy/zarr/PIL/cv2 for patch extraction
- **Confirm config before bulk processing** — the agent will ask if anything is ambiguous
- **mini pipeline first** — verify 5 cells end-to-end before full-scale extraction
- **MPP is derived from transforms** — no manual MPP guessing; stored in zarr metadata if available
- **OOB cells are skipped entirely** — no padding, no partial extraction, no resizing of clipped regions
- **Spatial sort** groups tiles by zarr chunk, improving cache hit rate ~2x
- **Binary .idx** format enables random access without sequential tar scan
- **Extraction mode is a required choice, not a default** — the agent will ask which of `tile_images`, `full_ops_level`, or `full_scale0` to use before running. Picking `full_ops_level` gives 36× speedup at ~0.4 GB memory; pick `tile_images` only when memory < 1 GB. See SKILL.md for benchmarks.
- **h5ad output** contains only valid cells in sample_index order with full alignment metadata
- **Patches are JPEG quality=95** — balance between file size and image quality for ML training
