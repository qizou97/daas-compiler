"""Tests for inspect_spatialdata.py --report-dir / inspect_report.json output."""
import importlib.util
import json
import pathlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── load script module ────────────────────────────────────────────────────────

_SCRIPT_PATH = pathlib.Path(__file__).parent.parent / "scripts" / "inspect_spatialdata.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("inspect_spatialdata", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── minimal sdata mock ────────────────────────────────────────────────────────

def _make_sdata():
    """Minimal SpatialData mock with 1 table, 1 shape, 1 image."""
    sdata = MagicMock()

    # table
    table = MagicMock()
    table.n_obs = 184523
    table.n_vars = 5001
    sdata.tables = {"table": table}

    # shape
    shape = MagicMock()
    shape.__len__ = MagicMock(return_value=184523)
    sdata.shapes = {"cell_circles": shape}

    # image — DataTree style: img["scale0"]["image"].shape == (C, H, W)
    img_array = MagicMock()
    img_array.shape = (3, 45000, 38000)
    scale0_node = MagicMock()
    scale0_node.__getitem__ = MagicMock(return_value=img_array)
    img = MagicMock()
    img.__getitem__ = MagicMock(return_value=scale0_node)
    img.__len__ = MagicMock(return_value=5)
    sdata.images = {"he_image": img}

    return sdata


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_main(mod, zarr_path: str, report_dir=None):
    argv = ["inspect_spatialdata.py", "--zarr", zarr_path]
    if report_dir is not None:
        argv += ["--report-dir", str(report_dir)]
    sdata = _make_sdata()
    with patch("sys.argv", argv), \
         patch("spatialdata.read_zarr", return_value=sdata):
        mod.main()
    return sdata


# ── tests ─────────────────────────────────────────────────────────────────────

def test_report_written_when_report_dir_given(tmp_path):
    mod = _load_script()
    zarr_path = tmp_path / "A_001.zarr"
    zarr_path.mkdir()
    report_dir = tmp_path / "reports"

    _run_main(mod, str(zarr_path), report_dir=report_dir)

    report_file = report_dir / "inspect_report.json"
    assert report_file.exists(), "inspect_report.json should be created"

    data = json.loads(report_file.read_text())

    # top-level keys
    assert "zarr_path" in data
    assert "sample_id" in data
    assert "tables" in data
    assert "shapes" in data
    assert "images" in data

    # sample_id is the zarr stem
    assert data["sample_id"] == "A_001"

    # zarr_path is the absolute (resolved) path
    assert data["zarr_path"] == str(zarr_path.resolve())

    # table entry
    assert len(data["tables"]) == 1
    tbl = data["tables"][0]
    assert tbl["key"] == "table"
    assert tbl["n_obs"] == 184523
    assert tbl["n_vars"] == 5001

    # shape entry
    assert len(data["shapes"]) == 1
    shp = data["shapes"][0]
    assert shp["key"] == "cell_circles"
    assert shp["n_rows"] == 184523

    # image entry
    assert len(data["images"]) == 1
    img = data["images"][0]
    assert img["key"] == "he_image"
    assert img["height"] == 45000
    assert img["width"] == 38000
    assert img["n_levels"] == 5


def test_report_dir_is_created_if_missing(tmp_path):
    mod = _load_script()
    zarr_path = tmp_path / "B_002.zarr"
    zarr_path.mkdir()
    # deeply nested directory that does not yet exist
    report_dir = tmp_path / "nested" / "deep" / "reports"

    _run_main(mod, str(zarr_path), report_dir=report_dir)

    assert (report_dir / "inspect_report.json").exists()


def test_no_report_without_report_dir(tmp_path):
    mod = _load_script()
    zarr_path = tmp_path / "C_003.zarr"
    zarr_path.mkdir()

    _run_main(mod, str(zarr_path), report_dir=None)

    # no JSON files should exist anywhere under tmp_path
    json_files = list(tmp_path.rglob("*.json"))
    assert json_files == [], f"Unexpected JSON files: {json_files}"


def test_report_stdout_message_printed(tmp_path, capsys):
    mod = _load_script()
    zarr_path = tmp_path / "D_004.zarr"
    zarr_path.mkdir()
    report_dir = tmp_path / "out"

    _run_main(mod, str(zarr_path), report_dir=report_dir)

    captured = capsys.readouterr()
    assert "report →" in captured.out
    assert "inspect_report.json" in captured.out
