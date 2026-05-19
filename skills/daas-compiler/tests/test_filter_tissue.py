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

def test_run_tissue_segmentation_calls_sopa_when_no_key_added():
    """Without key_added, SOPA is always called and new key discovered by diff."""
    sdata = _make_sdata(["cell_circles"])

    def fake_sopa(sd, **kwargs):
        sd.shapes["region_of_interest"] = MagicMock()

    with patch("sopa.segmentation.tissue", side_effect=fake_sopa) as mock_sopa:
        key = run_tissue_segmentation(sdata, image_key="he_image")

    mock_sopa.assert_called_once_with(sdata, image_key="he_image", allow_holes=False)
    assert key == "region_of_interest"


def test_run_tissue_segmentation_skips_if_key_added_exists():
    """If key_added is given and already in sdata.shapes, skip SOPA."""
    sdata = _make_sdata(["cell_circles", "tissue"])

    with patch("sopa.segmentation.tissue") as mock_sopa:
        key = run_tissue_segmentation(sdata, image_key="he_image", key_added="tissue")

    mock_sopa.assert_not_called()
    assert key == "tissue"


def test_run_tissue_segmentation_calls_sopa_with_key_added_when_absent():
    """If key_added is given but not in shapes, call SOPA and pass key_added."""
    sdata = _make_sdata(["cell_circles"])

    def fake_sopa(sd, **kwargs):
        sd.shapes["tissue"] = MagicMock()

    with patch("sopa.segmentation.tissue", side_effect=fake_sopa) as mock_sopa:
        key = run_tissue_segmentation(
            sdata, image_key="he_image", allow_holes=True, key_added="tissue"
        )

    mock_sopa.assert_called_once_with(
        sdata, image_key="he_image", allow_holes=True, key_added="tissue"
    )
    assert key == "tissue"


def test_run_tissue_segmentation_raises_if_sopa_creates_nothing():
    """If SOPA runs but creates no new shape key, raise RuntimeError."""
    sdata = _make_sdata(["cell_circles"])

    with patch("sopa.segmentation.tissue"):  # does nothing to sdata.shapes
        with pytest.raises(RuntimeError, match="created no new shape"):
            run_tissue_segmentation(sdata, image_key="he_image")


def test_run_tissue_segmentation_raises_if_sopa_not_installed():
    sdata = _make_sdata(["cell_circles"])
    with patch.dict("sys.modules", {"sopa": None, "sopa.segmentation": None}):
        with pytest.raises(ImportError):
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
