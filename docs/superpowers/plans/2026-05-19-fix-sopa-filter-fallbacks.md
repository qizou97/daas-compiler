# Fix SOPA Filter Fallbacks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the conditional "check-if-key-exists, else run SOPA" fallback pattern in tissue and nucleus-overlap filters with a direct, unconditional SOPA call that discovers the created shape key by diffing before/after.

**Architecture:** Each filter script owns one clear responsibility: always call the appropriate SOPA segmentation function, diff `sdata.shapes` before/after to find the new key (no hardcoded key names), then apply the spatial filter. The empty `daas/tasks/__init__.py` stub is deleted.

**Tech Stack:** Python 3.10+, spatialdata, sopa, geopandas, pytest

---

## File Map

| Action | Path |
|--------|------|
| Modify | `skills/daas-compiler/daas/filters/tissue.py` |
| Modify | `skills/daas-compiler/scripts/filter_tissue.py` |
| Modify | `skills/daas-compiler/daas/filters/nucleus_overlap.py` |
| Modify | `skills/daas-compiler/scripts/filter_nucleus_overlap.py` |
| Delete | `skills/daas-compiler/daas/tasks/__init__.py` |
| Add tests | `skills/daas-compiler/tests/test_filter_tissue.py` |
| Add tests | `skills/daas-compiler/tests/test_filter_nucleus_overlap.py` |

---

### Task 1: Fix `daas/filters/tissue.py` — always run SOPA, discover key by diff

**Files:**
- Modify: `skills/daas-compiler/daas/filters/tissue.py`
- Add: `skills/daas-compiler/tests/test_filter_tissue.py`

- [ ] **Step 1: Write failing tests**

Create `skills/daas-compiler/tests/test_filter_tissue.py`:

```python
"""Unit tests for daas.filters.tissue."""
import pytest
from unittest.mock import MagicMock, patch
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import anndata
from scipy.sparse import csr_matrix

from daas.filters.tissue import run_tissue_segmentation, filter_by_tissue


def _make_sdata(shape_keys):
    """Minimal sdata mock with shapes dict."""
    sdata = MagicMock()
    sdata.shapes = {k: MagicMock() for k in shape_keys}
    return sdata


def _make_adata(cell_ids):
    n = len(cell_ids)
    X = csr_matrix(np.ones((n, 2), dtype=np.float32))
    obs = pd.DataFrame({"cell_id": list(cell_ids)}, index=list(cell_ids))
    var = pd.DataFrame(index=["g0", "g1"])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _make_tissue_polygon():
    return gpd.GeoDataFrame(
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])]
    )


# ── run_tissue_segmentation ──────────────────────────────────────────────────

def test_run_tissue_segmentation_always_calls_sopa():
    """SOPA must always be called, never skipped even if a tissue key exists."""
    sdata = _make_sdata(["cell_circles"])

    def fake_sopa(sd, image_key):
        sd.shapes["region_of_interest"] = MagicMock()

    with patch("sopa.segmentation.tissue", side_effect=fake_sopa) as mock_sopa:
        key = run_tissue_segmentation(sdata, image_key="he_image")

    mock_sopa.assert_called_once_with(sdata, image_key="he_image")
    assert key == "region_of_interest"


def test_run_tissue_segmentation_raises_if_sopa_creates_nothing():
    """If SOPA runs but creates no new shape key, raise RuntimeError."""
    sdata = _make_sdata(["cell_circles"])

    with patch("sopa.segmentation.tissue"):  # does nothing to sdata.shapes
        with pytest.raises(RuntimeError, match="created no new shape"):
            run_tissue_segmentation(sdata, image_key="he_image")


def test_run_tissue_segmentation_raises_if_sopa_not_installed():
    sdata = _make_sdata(["cell_circles"])
    with patch.dict("sys.modules", {"sopa": None, "sopa.segmentation": None}):
        with pytest.raises((ImportError, TypeError)):
            run_tissue_segmentation(sdata, image_key="he_image")


# ── filter_by_tissue ─────────────────────────────────────────────────────────

def test_filter_by_tissue_keeps_inside_cells():
    cell_ids = ["c0", "c1", "c2"]
    adata = _make_adata(cell_ids)
    # c0 at (50,50) inside; c1 at (150,150) outside; c2 at (10,10) inside
    cell_shapes = gpd.GeoDataFrame(
        geometry=[Point(50, 50), Point(150, 150), Point(10, 10)],
        index=pd.Index(cell_ids),
    )
    tissue = _make_tissue_polygon()
    keep_mask, drop_counts = filter_by_tissue(adata, cell_shapes, tissue)
    np.testing.assert_array_equal(keep_mask, [True, False, True])
    assert drop_counts["outside_tissue"] == 1


def test_filter_by_tissue_all_outside():
    cell_ids = ["c0", "c1"]
    adata = _make_adata(cell_ids)
    cell_shapes = gpd.GeoDataFrame(
        geometry=[Point(200, 200), Point(300, 300)],
        index=pd.Index(cell_ids),
    )
    tissue = _make_tissue_polygon()
    keep_mask, drop_counts = filter_by_tissue(adata, cell_shapes, tissue)
    assert keep_mask.sum() == 0
    assert drop_counts["outside_tissue"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /yzyStorage/home/zouqi/codes/daas-compiler
python3 -m pytest skills/daas-compiler/tests/test_filter_tissue.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — `run_tissue_segmentation` doesn't exist yet.

- [ ] **Step 3: Rewrite `daas/filters/tissue.py`**

Replace the full file contents:

```python
from __future__ import annotations

import numpy as np


def run_tissue_segmentation(sdata, image_key: str) -> str:
    """Always run SOPA tissue segmentation and return the created shape key.

    Diffs sdata.shapes before/after to discover the new key — never hardcodes
    a key name. Raises RuntimeError if SOPA creates no new shape key.
    """
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for tissue segmentation. Install with: pip install sopa"
        )
    shapes_before = set(sdata.shapes.keys())
    sopa.segmentation.tissue(sdata, image_key=image_key)
    new_keys = set(sdata.shapes.keys()) - shapes_before
    if not new_keys:
        raise RuntimeError(
            f"sopa.segmentation.tissue ran but created no new shape key. "
            f"Shapes before: {sorted(shapes_before)}. "
            f"Shapes after: {sorted(sdata.shapes.keys())}."
        )
    if len(new_keys) == 1:
        return new_keys.pop()
    # Multiple new keys: prefer known tissue key names, else take sorted first.
    for candidate in ("region_of_interest", "tissue_boundaries", "tissue"):
        if candidate in new_keys:
            return candidate
    return sorted(new_keys)[0]


def filter_by_tissue(
    adata,
    cell_shapes,
    tissue_shapes,
    cell_id_column: str = "cell_id",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose centroid is inside
    any tissue polygon.

    cell_shapes: GeoDataFrame with Point or Polygon geometries
    tissue_shapes: GeoDataFrame with Polygon geometries (tissue regions)
    """
    import geopandas as gpd

    cell_ids = adata.obs[cell_id_column].astype(str)
    cell_shapes_aligned = cell_shapes.loc[
        cell_shapes.index.astype(str).isin(set(cell_ids.tolist()))
    ]
    centroids = cell_shapes_aligned.geometry.centroid
    tissue_union = tissue_shapes.geometry.union_all()
    inside = centroids.within(tissue_union)
    inside_series = inside.reindex(cell_ids.values, fill_value=False)
    keep_mask = inside_series.to_numpy(dtype=bool)
    n_dropped = int((~keep_mask).sum())
    return keep_mask, {"outside_tissue": n_dropped}


__all__ = ["run_tissue_segmentation", "filter_by_tissue"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest skills/daas-compiler/tests/test_filter_tissue.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
python3 -m pytest skills/daas-compiler/tests/ -q
```

Expected: 175 + 5 = 180 passed.

- [ ] **Step 6: Commit**

```bash
git add skills/daas-compiler/daas/filters/tissue.py \
        skills/daas-compiler/tests/test_filter_tissue.py
git commit -m "fix(filters): tissue always runs SOPA, discovers key by diff"
```

---

### Task 2: Update `scripts/filter_tissue.py` to call `run_tissue_segmentation`

**Files:**
- Modify: `skills/daas-compiler/scripts/filter_tissue.py`

- [ ] **Step 1: Replace the file**

```python
"""
Filter a SpatialData table to cells inside tissue regions.
Runs SOPA tissue segmentation to determine tissue boundaries.

Usage:
  python3 scripts/filter_tissue.py \
      --zarr /data/A_001.zarr \
      --input-table-key table \
      --output-table-key table_tissue \
      [--image-key he_image] \
      [--input-shape-key cell_circles] \
      [--report-dir /data/reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.tissue import run_tissue_segmentation, filter_by_tissue
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",             required=True)
    p.add_argument("--input-table-key",  default="table")
    p.add_argument("--input-shape-key",  default="cell_circles")
    p.add_argument("--output-table-key", default=None,
                   help="Default: {input_table_key}_tissue")
    p.add_argument("--image-key",        default="he_image",
                   help="Image key passed to SOPA tissue segmentation")
    p.add_argument("--cell-id-column",   default="cell_id")
    p.add_argument("--report-dir",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = Path(args.zarr)
    output_key = args.output_table_key or f"{args.input_table_key}_tissue"
    report_dir = Path(args.report_dir) if args.report_dir else (
        zarr_path.parent / "filter_reports"
    )

    print(f"[filter_tissue] {zarr_path.name}")
    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.input_table_key!r}. "
            f"Available: {list(sdata.tables.keys())}"
        )

    print(f"  running SOPA tissue segmentation (image_key={args.image_key!r}) …")
    tissue_key = run_tissue_segmentation(sdata, image_key=args.image_key)
    print(f"  tissue shape key: {tissue_key!r}")

    adata = sdata.tables[args.input_table_key]
    cell_shapes = sdata.shapes[args.input_shape_key]
    tissue_shapes = sdata.shapes[tissue_key]

    keep_mask, drop_counts = filter_by_tissue(
        adata, cell_shapes, tissue_shapes,
        cell_id_column=args.cell_id_column,
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  "
          f"(dropped {drop_counts.get('outside_tissue', 0)} outside tissue)")

    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="tissue_inside",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.input_shape_key,
        output_shape_key=tissue_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    path = write_stage_report(report, report_dir)
    print(f"  report → {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full test suite**

```bash
python3 -m pytest skills/daas-compiler/tests/ -q
```

Expected: 180 passed, 0 failed.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/scripts/filter_tissue.py
git commit -m "fix(scripts): filter_tissue uses run_tissue_segmentation"
```

---

### Task 3: Fix `daas/filters/nucleus_overlap.py` — always run Cellpose, discover key by diff

**Files:**
- Modify: `skills/daas-compiler/daas/filters/nucleus_overlap.py`
- Add: `skills/daas-compiler/tests/test_filter_nucleus_overlap.py`

- [ ] **Step 1: Write failing tests**

Create `skills/daas-compiler/tests/test_filter_nucleus_overlap.py`:

```python
"""Unit tests for daas.filters.nucleus_overlap."""
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import anndata
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon
from unittest.mock import MagicMock, patch

from daas.filters.nucleus_overlap import run_he_nucleus_segmentation, filter_by_nucleus_overlap


def _make_sdata(shape_keys):
    sdata = MagicMock()
    sdata.shapes = {k: MagicMock() for k in shape_keys}
    return sdata


def _make_adata(cell_ids):
    n = len(cell_ids)
    X = csr_matrix(np.ones((n, 2), dtype=np.float32))
    obs = pd.DataFrame({"cell_id": list(cell_ids)}, index=list(cell_ids))
    var = pd.DataFrame(index=["g0", "g1"])
    return anndata.AnnData(X=X, obs=obs, var=var)


# ── run_he_nucleus_segmentation ──────────────────────────────────────────────

def test_run_he_nucleus_segmentation_always_calls_cellpose():
    """Cellpose must always be called, never skipped."""
    sdata = _make_sdata(["cell_circles"])

    def fake_cellpose(sd, image_key):
        sd.shapes["he_nucleus_boundaries"] = MagicMock()

    with patch("sopa.segmentation.cellpose", side_effect=fake_cellpose) as mock_cp:
        key = run_he_nucleus_segmentation(sdata, image_key="he_image")

    mock_cp.assert_called_once_with(sdata, image_key="he_image")
    assert key == "he_nucleus_boundaries"


def test_run_he_nucleus_segmentation_raises_if_nothing_created():
    sdata = _make_sdata(["cell_circles"])
    with patch("sopa.segmentation.cellpose"):  # does nothing
        with pytest.raises(RuntimeError, match="created no new shape"):
            run_he_nucleus_segmentation(sdata, image_key="he_image")


# ── filter_by_nucleus_overlap ────────────────────────────────────────────────

def _rect(x0, y0, x1, y1):
    return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def test_filter_by_nucleus_overlap_keeps_high_iou():
    cell_ids = ["c0", "c1"]
    adata = _make_adata(cell_ids)

    xen_shapes = gpd.GeoDataFrame(
        geometry=[_rect(0, 0, 10, 10), _rect(50, 50, 60, 60)],
        index=pd.Index(cell_ids),
    )
    # he nucleus: c0 nearly identical (high IoU), c1 far away (no overlap)
    he_shapes = gpd.GeoDataFrame(
        geometry=[_rect(1, 1, 9, 9)],
    )
    keep_mask, drop_counts = filter_by_nucleus_overlap(
        adata, xen_shapes, he_shapes, overlap_threshold=0.5
    )
    assert keep_mask[0] == True   # c0: good overlap
    assert keep_mask[1] == False  # c1: no overlap


def test_filter_by_nucleus_overlap_missing_xenium_nucleus():
    """Cell not in xenium nucleus shapes → dropped."""
    adata = _make_adata(["c0", "c1"])
    xen_shapes = gpd.GeoDataFrame(
        geometry=[_rect(0, 0, 10, 10)],
        index=pd.Index(["c0"]),  # c1 missing
    )
    he_shapes = gpd.GeoDataFrame(geometry=[_rect(0, 0, 10, 10)])
    keep_mask, drop_counts = filter_by_nucleus_overlap(
        adata, xen_shapes, he_shapes, overlap_threshold=0.5
    )
    assert keep_mask[1] == False
    assert drop_counts.get("no_xenium_nucleus", 0) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest skills/daas-compiler/tests/test_filter_nucleus_overlap.py -v 2>&1 | head -20
```

Expected: `ImportError` — `run_he_nucleus_segmentation` doesn't exist yet.

- [ ] **Step 3: Rewrite `daas/filters/nucleus_overlap.py`**

```python
"""Xenium-vs-HE nucleus overlap filter.

Keeps cells whose Xenium nucleus_boundaries polygon overlaps the nearest
HE nucleus polygon with intersection-over-union >= overlap_threshold.

Requires sopa to be installed. Always runs sopa.segmentation.cellpose
to produce HE nucleus boundaries — no conditional skip.
"""
from __future__ import annotations

import numpy as np


def run_he_nucleus_segmentation(sdata, image_key: str) -> str:
    """Always run SOPA Cellpose HE nucleus segmentation and return the created key.

    Diffs sdata.shapes before/after to discover the new key — never hardcodes
    a key name. Raises RuntimeError if Cellpose creates no new shape key.
    """
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for HE nucleus segmentation. "
            "Install with: pip install sopa"
        )
    shapes_before = set(sdata.shapes.keys())
    sopa.segmentation.cellpose(sdata, image_key=image_key)
    new_keys = set(sdata.shapes.keys()) - shapes_before
    if not new_keys:
        raise RuntimeError(
            f"sopa.segmentation.cellpose ran but created no new shape key. "
            f"Shapes before: {sorted(shapes_before)}. "
            f"Shapes after: {sorted(sdata.shapes.keys())}."
        )
    if len(new_keys) == 1:
        return new_keys.pop()
    for candidate in ("he_nucleus_boundaries", "nucleus_boundaries"):
        if candidate in new_keys:
            return candidate
    return sorted(new_keys)[0]


def _iou(poly_a, poly_b) -> float:
    inter = poly_a.intersection(poly_b).area
    if inter == 0:
        return 0.0
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def filter_by_nucleus_overlap(
    adata,
    xenium_nucleus_shapes,
    he_nucleus_shapes,
    cell_id_column: str = "cell_id",
    overlap_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose Xenium nucleus has
    IoU >= overlap_threshold with the nearest HE nucleus polygon.
    """
    import pandas as pd
    from shapely.strtree import STRtree

    cell_ids = adata.obs[cell_id_column].astype(str).values
    xen_ids = pd.Index(xenium_nucleus_shapes.index).astype(str)
    xen_map = {str(cid): geom for cid, geom in
               zip(xen_ids, xenium_nucleus_shapes.geometry)}

    he_geoms = list(he_nucleus_shapes.geometry)
    tree = STRtree(he_geoms)

    keep_mask = np.zeros(len(cell_ids), dtype=bool)

    for i, cid in enumerate(cell_ids):
        if cid not in xen_map:
            continue
        xen_poly = xen_map[cid]
        candidates = tree.query(xen_poly)
        if len(candidates) == 0:
            continue
        best_iou = max(_iou(xen_poly, he_geoms[j]) for j in candidates)
        if best_iou >= overlap_threshold:
            keep_mask[i] = True

    n_dropped_no_nucleus = int(sum(1 for cid in cell_ids if cid not in xen_map))
    n_dropped_low_overlap = int((~keep_mask).sum()) - n_dropped_no_nucleus

    drop_counts = {}
    if n_dropped_no_nucleus:
        drop_counts["no_xenium_nucleus"] = n_dropped_no_nucleus
    if n_dropped_low_overlap > 0:
        drop_counts["low_he_overlap"] = n_dropped_low_overlap

    return keep_mask, drop_counts


__all__ = [
    "run_he_nucleus_segmentation",
    "filter_by_nucleus_overlap",
]
```

- [ ] **Step 4: Run new tests**

```bash
python3 -m pytest skills/daas-compiler/tests/test_filter_nucleus_overlap.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
python3 -m pytest skills/daas-compiler/tests/ -q
```

Expected: 184 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add skills/daas-compiler/daas/filters/nucleus_overlap.py \
        skills/daas-compiler/tests/test_filter_nucleus_overlap.py
git commit -m "fix(filters): nucleus_overlap always runs Cellpose, discovers key by diff"
```

---

### Task 4: Update `scripts/filter_nucleus_overlap.py` to call `run_he_nucleus_segmentation`

**Files:**
- Modify: `skills/daas-compiler/scripts/filter_nucleus_overlap.py`

- [ ] **Step 1: Replace the file**

```python
"""
Filter a SpatialData table to cells whose Xenium nucleus overlaps HE nucleus.
Runs SOPA Cellpose on the HE image to produce HE nucleus boundaries.

Usage:
  python3 scripts/filter_nucleus_overlap.py \
      --zarr /data/A_001.zarr \
      --input-table-key table_tissue \
      --output-table-key table_tissue_he \
      [--xenium-nucleus-key nucleus_boundaries] \
      [--image-key he_image] \
      [--overlap-threshold 0.5] \
      [--report-dir /data/reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.nucleus_overlap import (
    run_he_nucleus_segmentation,
    filter_by_nucleus_overlap,
)
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",                  required=True)
    p.add_argument("--input-table-key",       default="table")
    p.add_argument("--input-shape-key",       default="cell_circles")
    p.add_argument("--output-table-key",      default=None,
                   help="Default: {input_table_key}_he")
    p.add_argument("--xenium-nucleus-key",    default="nucleus_boundaries")
    p.add_argument("--image-key",             default="he_image",
                   help="Image key passed to SOPA Cellpose segmentation")
    p.add_argument("--overlap-threshold",     type=float, default=0.5)
    p.add_argument("--cell-id-column",        default="cell_id")
    p.add_argument("--report-dir",            default=None)
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = Path(args.zarr)
    output_key = args.output_table_key or f"{args.input_table_key}_he"
    report_dir = Path(args.report_dir) if args.report_dir else (
        zarr_path.parent / "filter_reports"
    )

    print(f"[filter_nucleus_overlap] {zarr_path.name}")
    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.input_table_key!r}. "
            f"Available: {list(sdata.tables.keys())}"
        )
    if args.xenium_nucleus_key not in sdata.shapes:
        raise KeyError(
            f"sdata has no shape {args.xenium_nucleus_key!r}. "
            f"Available: {list(sdata.shapes.keys())}"
        )

    print(f"  running SOPA Cellpose HE nucleus segmentation (image_key={args.image_key!r}) …")
    he_nucleus_key = run_he_nucleus_segmentation(sdata, image_key=args.image_key)
    print(f"  HE nucleus shape key: {he_nucleus_key!r}")

    adata = sdata.tables[args.input_table_key]
    xenium_nucleus = sdata.shapes[args.xenium_nucleus_key]
    he_nucleus = sdata.shapes[he_nucleus_key]

    keep_mask, drop_counts = filter_by_nucleus_overlap(
        adata, xenium_nucleus, he_nucleus,
        cell_id_column=args.cell_id_column,
        overlap_threshold=args.overlap_threshold,
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  (threshold={args.overlap_threshold})")

    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="xenium_he_nucleus_overlap",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.xenium_nucleus_key,
        output_shape_key=he_nucleus_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    path = write_stage_report(report, report_dir)
    print(f"  report → {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full suite**

```bash
python3 -m pytest skills/daas-compiler/tests/ -q
```

Expected: 184 passed, 0 failed.

- [ ] **Step 3: Commit**

```bash
git add skills/daas-compiler/scripts/filter_nucleus_overlap.py
git commit -m "fix(scripts): filter_nucleus_overlap uses run_he_nucleus_segmentation"
```

---

### Task 5: Delete `daas/tasks/__init__.py`

**Files:**
- Delete: `skills/daas-compiler/daas/tasks/__init__.py`

- [ ] **Step 1: Verify the file is empty and has no imports anywhere**

```bash
cat skills/daas-compiler/daas/tasks/__init__.py
grep -r "from daas.tasks\|import daas.tasks" skills/daas-compiler/ --include="*.py"
```

Expected: file is empty (0 lines of logic); no imports found.

- [ ] **Step 2: Delete file and directory**

```bash
rm skills/daas-compiler/daas/tasks/__init__.py
rmdir skills/daas-compiler/daas/tasks
```

- [ ] **Step 3: Run full suite**

```bash
python3 -m pytest skills/daas-compiler/tests/ -q
```

Expected: 184 passed, 0 failed.

- [ ] **Step 4: Commit**

```bash
git add -A skills/daas-compiler/daas/tasks/
git commit -m "chore: remove empty daas/tasks stub"
```

---

## Final Verification

After all tasks complete, confirm the pipeline commands work end-to-end:

```bash
SKILL_DIR=/yzyStorage/home/zouqi/codes/daas-compiler/skills/daas-compiler

# Verify filter_tissue.py imports cleanly
python3 -c "from daas.filters.tissue import run_tissue_segmentation, filter_by_tissue; print('OK')"

# Verify filter_nucleus_overlap.py imports cleanly
python3 -c "from daas.filters.nucleus_overlap import run_he_nucleus_segmentation, filter_by_nucleus_overlap; print('OK')"

# Full test suite
python3 -m pytest skills/daas-compiler/tests/ -q
# Expected: 184 passed
```
