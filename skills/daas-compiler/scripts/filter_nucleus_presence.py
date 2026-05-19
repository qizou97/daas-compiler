"""
Filter a SpatialData table to cells that have a nucleus boundary entry.
Writes the filtered table back into the zarr under a new key.

Usage:
  python3 scripts/filter_nucleus_presence.py \
      --zarr /data/A_001.zarr \
      --input-table-key table_tissue \
      --output-table-key table_tissue_nucleus \
      [--nucleus-boundaries-key nucleus_boundaries] \
      [--cell-id-column cell_id] \
      [--report-dir /data/A_001_reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.nucleus_presence import filter_by_nucleus_presence
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",                required=True)
    p.add_argument("--input-table-key",     default="table")
    p.add_argument("--input-shape-key",     default="cell_circles")
    p.add_argument("--output-table-key",    default=None,
                   help="Default: {input_table_key}_nucleus")
    p.add_argument("--nucleus-boundaries-key", default="nucleus_boundaries")
    p.add_argument("--cell-id-column",      default="cell_id")
    p.add_argument("--report-dir",          default=None,
                   help="Default: next to zarr in filter_reports/")
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = Path(args.zarr)
    output_key = args.output_table_key or f"{args.input_table_key}_nucleus"
    report_dir = Path(args.report_dir) if args.report_dir else None

    print(f"[filter_nucleus_presence] {zarr_path.name}")
    print(f"  input_table={args.input_table_key!r}  "
          f"nucleus_boundaries={args.nucleus_boundaries_key!r}  "
          f"output_table={output_key!r}")

    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.input_table_key!r}. "
            f"Available: {list(sdata.tables.keys())}"
        )
    if args.nucleus_boundaries_key not in sdata.shapes:
        raise KeyError(
            f"sdata has no shape {args.nucleus_boundaries_key!r}. "
            f"Available: {list(sdata.shapes.keys())}"
        )

    adata = sdata.tables[args.input_table_key]
    nucleus_gdf = sdata.shapes[args.nucleus_boundaries_key]

    keep_mask, drop_counts = filter_by_nucleus_presence(
        adata, nucleus_gdf, cell_id_column=args.cell_id_column
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  "
          f"(dropped {drop_counts.get('missing_nucleus_boundary', 0)} "
          f"missing nucleus boundary)")

    # Write filtered table back into zarr
    if output_key in sdata.tables:
        sdata.delete_element_from_disk(output_key)
    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="nucleus_presence",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.input_shape_key,
        output_shape_key=args.input_shape_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    if report_dir is not None:
        path = write_stage_report(report, report_dir)
        print(f"  report → {path}")


if __name__ == "__main__":
    main()
