# Filtering Policy Reference

Authoritative reference for the two filtering layers in `daas-compiler`. Use this when designing CLI invocations, interpreting `filter_report.json`, or extending the policy module.

The implementation lives in [`daas/filtering.py`](../daas/filtering.py); orchestration in [`scripts/extract_sample.py`](../scripts/extract_sample.py); tests in [`tests/test_filtering.py`](../tests/test_filtering.py) and [`tests/test_filtering_integration.py`](../tests/test_filtering_integration.py).

---

## 1. Definitions

### `biological_filter_policy` (Layer 1)

Picks **which table and shape layer enter the pipeline**, and optionally drops table rows by a biological criterion (presence of a nucleus boundary, presence in a stVisuome-canonical `filtered_table`).

CLI flag: `--biological-filter-policy {auto, none, stvisuome_canonical, stvisuome_nucleus_boundary}`

| Value | Effect |
|---|---|
| `auto` (default) | If `--table-key` is at its default **and** `sdata.tables` contains `filtered_table`, behaves as `stvisuome_canonical`. Otherwise behaves as `none`. Warns if `filtered_table` exists but `--table-key` was explicitly set elsewhere. |
| `none` | Use the table at `--table-key` and the shapes at `--shapes-key` verbatim. Emits a warning that no biological filtering was applied. |
| `stvisuome_canonical` | Require `sdata.tables[--filtered-table-key]` (default `filtered_table`) to exist; consume it as the authoritative table. Shape layer is whatever `--shapes-key` names — alignment is by `cell_id`. |
| `stvisuome_nucleus_boundary` | Keep table rows whose `obs[cell_id_column]` appears in `sdata.shapes[--nucleus-boundaries-key].index`. Hard error if the overlap is empty. |

After Layer 1, `resolve_table_shape_alignment(adata, gdf, cell_id_column)` enforces 1:1 row alignment:

- Exact match preferred (`list(gdf.index.astype(str)) == list(adata.obs[cell_id].astype(str))`).
- Otherwise: intersection by `cell_id`, preserving the **table's** row order.
- Empty intersection raises `ValueError`.

### `patch_filter_policy` (Layer 2)

Decides which selected cells produce a writable tile. Always runs after a positive-centroid filter (`cx_px > 0 & cy_px > 0`), and emits separate `full_oob`/`need_pad` masks computed from the cell's level-0 bounding box vs the slide.

CLI flag: `--patch-filter-policy {auto, strict_no_padding, stvisuome_minimal, strict_with_padding}`

| Value | Drops | Allowed extract modes |
|---|---|---|
| `auto` (default) | Resolves to `strict_no_padding`. | All. |
| `strict_no_padding` | `full_oob ∪ need_pad`. The historic, safe default. | All. |
| `stvisuome_minimal` | `full_oob` only — keeps `need_pad` (boundary-crossing tiles). wsidata pads them on read. | **`tile_images` only**; raises `ValueError` for `full_scale0` / `full_ops_level`. |
| `strict_with_padding` | Reserved. Raises `NotImplementedError` until explicit padding is wired through `extract_sample.py`. | — |

---

## 2. Decision table

The table below maps the user's situation to the recommended policy pair. The agent should ask the user to confirm before running.

| Situation | `--biological-filter-policy` | `--patch-filter-policy` | `--extract-mode` |
|---|---|---|---|
| Preprocessed stVisuome zarr (has `filtered_table` + `filtered_nucleus_boundaries` + `filtered_cell_boundaries`) | `auto` (resolves to `stvisuome_canonical`) | `auto` (`strict_no_padding`) | `full_ops_level` (fastest) or `tile_images` |
| Raw zarr with `table` + `nucleus_boundaries` but no `filtered_table` | `stvisuome_nucleus_boundary` | `auto` (`strict_no_padding`) | `full_ops_level` or `tile_images` |
| Raw zarr with `table` + `cell_circles` only (no nucleus boundaries) | `none` | `auto` (`strict_no_padding`) | Any |
| stVisuome zarr but user wants stVisuome-like boundary behavior (keep need_pad tiles) | `auto` | `stvisuome_minimal` | **`tile_images` only** |
| Any `full_scale0` / `full_ops_level` extraction | as above | **must** be `strict_no_padding` (or `auto` which resolves to it) | `full_scale0` / `full_ops_level` |

**Defaults preserve historic strict behavior.** Old invocations that do not pass either filter flag run with `auto`+`auto` → `none` (or `stvisuome_canonical` if `filtered_table` is detected by default) + `strict_no_padding`. The `n_out` count is identical to the pre-filter-policy era unless `filtered_table` is auto-detected.

---

## 3. Drop reason taxonomy

`filter_report.json` reports `drop_counts_by_reason` keyed on the strings below. Counts may overlap across keys (e.g. a cell can be both `non_positive_centroid` and `full_oob`); they are **independent per-reason counts**, not a partition. The sequential `n_after_*` fields give the true post-filter survivor counts.

| Key | Meaning | Layer |
|---|---|---|
| `missing_nucleus_boundary` | Table cell_id absent from `nucleus_boundaries.index` (only emitted under `stvisuome_nucleus_boundary`). | 1 |
| `unaligned_with_shapes` | Bio-filtered table cell_id not present in the chosen shape layer's index (emitted when alignment falls back to intersection). | 1 |
| `non_positive_centroid` | `cx_px ≤ 0` or `cy_px ≤ 0`. | 2 (pre-patch) |
| `full_oob` | Tile's level-0 bounding box lies entirely off the slide. | 2 |
| `need_pad` | Tile partially extends past the slide — would require padding. Emitted only when `strict_no_padding` actually drops it. | 2 |
| `requested_subsample` | Survivors discarded by `--n-sample` after all filters. | 3 (sampling) |

**`zero_expression`** is *not* enforced inside daas-compiler. If the user wants to drop zero-expression cells, they should filter their `table` upstream (e.g. via `stvisuome-daas preprocess` or a scanpy step) before pointing daas-compiler at the zarr. We deliberately do not reimplement that policy here.

---

## 4. Example `filter_report.json`

### A. stVisuome-canonical zarr, default policies

```json
{
  "sample_id": "A_028",
  "zarr_path": "/data/spatialdata/A_028.zarr",
  "output_dir": "/data/out/A_028",
  "image_key": "he_image",
  "extract_mode": "full_ops_level",
  "source_table_key": "filtered_table",
  "source_shape_key": "cell_circles",
  "biological_policy_requested": "auto",
  "biological_policy_applied": "stvisuome_canonical",
  "patch_policy_requested": "auto",
  "patch_policy_applied": "strict_no_padding",
  "n_cells_source": 184523,
  "n_after_biological_filter": 184523,
  "n_after_positive_centroid": 184501,
  "n_after_patch_policy": 183872,
  "n_out": 10000,
  "drop_counts_by_reason": {
    "non_positive_centroid": 22,
    "full_oob": 41,
    "need_pad": 588,
    "requested_subsample": 173872
  },
  "patch_size": 224,
  "target_mpp": 0.5,
  "slide_mpp": 0.2125,
  "base_size": 527,
  "seed": 42,
  "warnings": []
}
```

### B. Raw zarr, nucleus-boundary filter, tile_images + stvisuome_minimal

```json
{
  "sample_id": "B_041",
  "zarr_path": "/data/spatialdata/B_041.zarr",
  "extract_mode": "tile_images",
  "source_table_key": "table",
  "source_shape_key": "cell_circles",
  "biological_policy_requested": "stvisuome_nucleus_boundary",
  "biological_policy_applied": "stvisuome_nucleus_boundary",
  "patch_policy_requested": "stvisuome_minimal",
  "patch_policy_applied": "stvisuome_minimal",
  "n_cells_source": 210004,
  "n_after_biological_filter": 197610,
  "n_after_positive_centroid": 197604,
  "n_after_patch_policy": 197441,
  "n_out": 197441,
  "drop_counts_by_reason": {
    "missing_nucleus_boundary": 12394,
    "non_positive_centroid": 6,
    "full_oob": 169,
    "requested_subsample": 0
  },
  "warnings": []
}
```

### C. Misconfiguration caught by the report — auto fell back to `none`

```json
{
  "sample_id": "Q_999",
  "biological_policy_requested": "auto",
  "biological_policy_applied": "none",
  "patch_policy_applied": "strict_no_padding",
  "source_table_key": "table",
  "n_cells_source": 100000,
  "n_after_biological_filter": 100000,
  "warnings": [
    "no biological filtering applied; all rows of the selected table are kept."
  ]
}
```

When `warnings` is non-empty, surface it verbatim in the user-facing reply.

---

## 5. Pitfalls

- **Centroid > 0 is not full containment.** `mask_positive_centroid` only checks the centroid sign; `strict_no_padding` is required to guarantee a fully-contained tile. Never substitute one for the other.
- **Do not silently switch extraction modes.** If the user picked `stvisuome_minimal` and `full_scale0`, raise — those modes silently clip boundary tiles, defeating the point. Always confirm `--extract-mode` with the user (see the "Extraction Strategies" section of SKILL.md).
- **Do not use stVisuome preprocessing internals.** `daas-compiler` must not import `stvisuome_daas`, run `sopa.segmentation.tissue`, call Cellpose, or perform nucleus matching. If those steps are missing from the input zarr, point the user at `stvisuome-daas preprocess` instead of reimplementing.
- **Do not desynchronize manifest and h5ad rows.** Every change to filter logic must preserve `manifest[i].cell_id == adata_out.obs.cell_id[i]` and `expr_row == sample_index`. The compile step concatenates per-sample manifests/h5ads in the same `sorted()` order — any per-sample drift propagates into `global_idx` and corrupts training-time lookup.
- **Auto's table-key sentinel.** `auto` only swaps to `filtered_table` when `--table-key` is at its default (`"table"`). If the user explicitly passes any other table key, `auto` falls back to `none` and emits a warning. To force the canonical path under a custom table name, pass `--biological-filter-policy stvisuome_canonical` explicitly.
- **Don't infer biology from absence.** A zero-row `n_after_biological_filter` under `stvisuome_nucleus_boundary` is a hard error (raises `ValueError`), not a permissive "drop all" — it almost always means cell-id formats don't match between table and nucleus shapes. Investigate first.
- **No `--skip-viz`.** The pre-flight viz runs unconditionally; abort and re-run with adjusted policy if `viz_patch_grid.png` shows misaligned boundaries.
- **Do not bypass `filter_report.json`.** It is the durable record of what each per-sample run actually did. Always summarize it back to the user, and never delete it as part of cleanup.
