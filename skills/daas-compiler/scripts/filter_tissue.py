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
