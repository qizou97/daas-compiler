# SOPA Integration Reference

SOPA (Spatial Omics Pipeline Architecture) is an optional dependency used by
two stage scripts:

- `filter_tissue.py` — `sopa.segmentation.tissue()`
- `filter_nucleus_overlap.py` — `sopa.segmentation.cellpose()`

## Installation

```bash
pip install sopa
# or, if using extras:
pip install "sopa[cellpose]"
```

## Tissue segmentation (filter_tissue.py)

```python
import sopa.segmentation
sopa.segmentation.tissue(sdata, image_key="he_image", allow_holes=False, key_added="tissue")
```

`filter_tissue.py` skips calling SOPA if `--key-added` is given and that shape
key already exists in `sdata.shapes`. Otherwise SOPA is always called.

Expected result: a new polygon GeoDataFrame in `sdata.shapes` at the given
`key_added` name (or a name discovered by diff if `key_added` is omitted).

**Visual QC** — after segmentation the script saves `viz/tissue_overlay.png`
to the report directory. For interactive notebooks, run:
```python
(sdata
    .pl.render_images("he_image")
    .pl.render_shapes(tissue_key, fill_alpha=0., outline_width=0.5, outline_color="black")
    .pl.show())
```

**Verify the actual API against your installed sopa version:**
```python
import sopa.segmentation
help(sopa.segmentation.tissue)
```

## HE nucleus segmentation (filter_nucleus_overlap.py)

Called when `--he-nucleus-key` (default: `he_nucleus_boundaries`) is absent:

```python
import sopa.segmentation
sopa.segmentation.cellpose(sdata, image_key="he_image")
```

Expected result: a new polygon GeoDataFrame in `sdata.shapes` named
`he_nucleus_boundaries` (or similar — check `sopa` docs).

**If the created key differs from the default, pass `--he-nucleus-key`
explicitly.**

## Error if sopa missing

Both scripts raise `ImportError` with installation instructions when
`import sopa.segmentation` fails. The scripts do NOT require sopa if the
relevant shape keys already exist in the zarr.
