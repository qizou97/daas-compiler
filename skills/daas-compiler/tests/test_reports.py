import json
from pathlib import Path

import pytest

from daas.reports import StageReport, write_stage_report


def _make_report(**kwargs) -> StageReport:
    defaults = dict(
        stage="nucleus_presence",
        zarr_path="/data/A_001.zarr",
        input_table_key="table",
        output_table_key="table_nucleus",
        input_shape_key="cell_circles",
        output_shape_key="cell_circles",
        n_cells_in=100,
        n_cells_out=80,
        drop_counts_by_reason={"missing_nucleus_boundary": 20},
        warnings=[],
    )
    defaults.update(kwargs)
    return StageReport(**defaults)


def test_stage_report_fields():
    r = _make_report()
    assert r.n_cells_in == 100
    assert r.n_cells_out == 80
    assert r.drop_counts_by_reason["missing_nucleus_boundary"] == 20
    assert r.report_path == ""


def test_write_stage_report_creates_file(tmp_path):
    r = _make_report()
    path = write_stage_report(r, tmp_path)
    assert path.exists()
    assert path.name == "nucleus_presence_table.json"
    data = json.loads(path.read_text())
    assert data["stage"] == "nucleus_presence"
    assert data["n_cells_out"] == 80
    assert data["drop_counts_by_reason"]["missing_nucleus_boundary"] == 20


def test_write_stage_report_sets_report_path(tmp_path):
    r = _make_report()
    path = write_stage_report(r, tmp_path)
    assert r.report_path == str(path)


def test_write_stage_report_filename_uses_input_key(tmp_path):
    r = _make_report(stage="tissue_inside", input_table_key="filtered_table")
    path = write_stage_report(r, tmp_path)
    assert path.name == "tissue_inside_filtered_table.json"
