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

    def fake_sopa(sd, image_key, allow_holes=False, key_added="tissue"):
        sd.shapes["region_of_interest"] = MagicMock()

    with patch("sopa.segmentation.tissue", side_effect=fake_sopa) as mock_sopa:
        key = run_tissue_segmentation(sdata, image_key="he_image")

    mock_sopa.assert_called_once_with(
        sdata, image_key="he_image", allow_holes=False, key_added="tissue"
    )
    assert key == "region_of_interest"


def test_run_tissue_segmentation_warns_if_sopa_creates_nothing():
    """If SOPA updates the key in-place (no new key), emit TissueKeyExistsWarning
    and return key_added."""
    from daas.filters.tissue import TissueKeyExistsWarning

    sdata = _make_sdata(["cell_circles", "tissue"])

    with patch("sopa.segmentation.tissue"):  # does nothing to sdata.shapes
        with pytest.warns(TissueKeyExistsWarning, match="created no new shape key"):
            key = run_tissue_segmentation(sdata, image_key="he_image")

    assert key == "tissue"


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


# ── filter_tissue.py script: --force / reuse logic ───────────────────────────

def _make_script_sdata(tissue_key="tissue"):
    """Minimal sdata mock suitable for script-level tests."""
    cell_ids = ["c0", "c1"]
    adata = _make_adata(cell_ids)
    tissue_shapes = _make_tissue_polygon()
    cell_shapes = gpd.GeoDataFrame(
        geometry=[Point(50, 50), Point(150, 150)],
        index=pd.Index(cell_ids),
    )

    sdata = MagicMock()
    sdata.shapes = {
        "cell_circles": cell_shapes,
        tissue_key: tissue_shapes,
    }
    sdata.tables = {"table": adata}
    sdata.__getitem__ = MagicMock(return_value=adata)
    return sdata, adata


def test_script_reuse_skips_segmentation_when_tissue_key_exists(tmp_path, capsys):
    """When tissue_key already exists and --force is NOT set, SOPA must NOT be called."""
    import sys
    from unittest.mock import patch, MagicMock
    import importlib.util
    import pathlib

    _script_path = pathlib.Path(__file__).parent.parent / "scripts" / "filter_tissue.py"
    spec = importlib.util.spec_from_file_location("filter_tissue", _script_path)
    ft_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft_script)

    sdata, adata = _make_script_sdata("tissue")

    report_dir = tmp_path / "reports"
    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir()

    argv = [
        "filter_tissue.py",
        "--zarr", str(zarr_path),
        "--input-table-key", "table",
        "--output-table-key", "table_tissue",
        "--tissue-key", "tissue",
        "--report-dir", str(report_dir),
    ]

    with patch("sys.argv", argv), \
         patch("spatialdata.read_zarr", return_value=sdata), \
         patch.object(ft_script, "run_tissue_segmentation") as mock_seg, \
         patch.object(ft_script, "_save_tissue_viz"), \
         patch("daas.reports.write_stage_report", return_value=report_dir / "r.json"):
        # Patch sdata write methods to be no-ops
        sdata.delete_element_from_disk = MagicMock()
        sdata.__setitem__ = MagicMock()
        sdata.write_element = MagicMock()

        ft_script.main()

    mock_seg.assert_not_called()
    captured = capsys.readouterr()
    assert "[reuse]" in captured.out
    assert "skipping SOPA" in captured.out


def test_script_force_runs_segmentation_when_tissue_key_exists(tmp_path, capsys):
    """When tissue_key already exists and --force IS set, SOPA must be called."""
    import sys
    from unittest.mock import patch, MagicMock
    import importlib.util
    import pathlib

    _script_path = pathlib.Path(__file__).parent.parent / "scripts" / "filter_tissue.py"
    spec = importlib.util.spec_from_file_location("filter_tissue", _script_path)
    ft_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft_script)

    sdata, adata = _make_script_sdata("tissue")

    report_dir = tmp_path / "reports"
    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir()

    argv = [
        "filter_tissue.py",
        "--zarr", str(zarr_path),
        "--input-table-key", "table",
        "--output-table-key", "table_tissue",
        "--tissue-key", "tissue",
        "--force",
        "--report-dir", str(report_dir),
    ]

    with patch("sys.argv", argv), \
         patch("spatialdata.read_zarr", return_value=sdata), \
         patch.object(ft_script, "run_tissue_segmentation",
               return_value="tissue") as mock_seg, \
         patch.object(ft_script, "_save_tissue_viz"), \
         patch("daas.reports.write_stage_report", return_value=report_dir / "r.json"):
        sdata.delete_element_from_disk = MagicMock()
        sdata.__setitem__ = MagicMock()
        sdata.write_element = MagicMock()

        ft_script.main()

    mock_seg.assert_called_once()
    captured = capsys.readouterr()
    assert "[force]" in captured.out
    assert "overwrite" in captured.out
