from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageReport:
    stage: str
    zarr_path: str
    input_table_key: str
    output_table_key: str
    input_shape_key: str
    output_shape_key: str
    n_cells_in: int
    n_cells_out: int
    drop_counts_by_reason: dict = field(default_factory=dict)
    report_path: str = ""
    warnings: list = field(default_factory=list)


def write_stage_report(report: StageReport, report_dir) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"{report.stage}_{report.input_table_key}.json"
    path = report_dir / name
    data = {
        "stage": report.stage,
        "zarr_path": report.zarr_path,
        "input_table_key": report.input_table_key,
        "output_table_key": report.output_table_key,
        "input_shape_key": report.input_shape_key,
        "output_shape_key": report.output_shape_key,
        "n_cells_in": report.n_cells_in,
        "n_cells_out": report.n_cells_out,
        "drop_counts_by_reason": report.drop_counts_by_reason,
        "warnings": report.warnings,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    report.report_path = str(path)
    return path


__all__ = ["StageReport", "write_stage_report"]
