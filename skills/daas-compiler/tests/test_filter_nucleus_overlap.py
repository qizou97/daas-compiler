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
