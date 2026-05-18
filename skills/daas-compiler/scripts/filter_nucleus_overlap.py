"""
Filter cells by Xenium-vs-HE nucleus overlap score.
Runs SOPA Cellpose HE nucleus segmentation if he_nucleus_boundaries absent.

Usage:
  python3 scripts/filter_nucleus_overlap.py \
      --zarr /data/A_001.zarr \
      --input-table-key table_tissue_nucleus \
      --output-table-key table_tissue_nucleus_he \
      [--xenium-nucleus-key nucleus_boundaries] \
      [--he-nucleus-key he_nucleus_boundaries] \
      [--overlap-threshold 0.5] \
      [--image-key he_image] \
      [--report-dir /data/reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.nucleus_overlap import (
    _ensure_he_nucleus_shapes,
    filter_by_nucleus_overlap,
)
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",                  required=True)
    p.add_argument("--input-table-key",       default="table")
    p.add_argument("--input-shape-key",       default="cell_circles")
    p.add_argument("--output-table-key",      default=None)
    p.add_argument("--xenium-nucleus-key",    default="nucleus_boundaries")
    p.add_argument("--he-nucleus-key",        default="he_nucleus_boundaries")
    p.add_argument("--overlap-threshold",     type=float, default=0.5)
    p.add_argument("--image-key",             default="he_image")
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

    print(f"[filter_nucleus_overlap] {zarr_path.name}  "
          f"threshold={args.overlap_threshold}")
    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(f"sdata has no table {args.input_table_key!r}.")
    if args.xenium_nucleus_key not in sdata.shapes:
        raise KeyError(
            f"sdata has no shape {args.xenium_nucleus_key!r} "
            "(Xenium nucleus boundaries). Run SOPA nucleus segmentation first."
        )

    he_nucleus_key = _ensure_he_nucleus_shapes(
        sdata, args.image_key, args.he_nucleus_key
    )

    adata = sdata.tables[args.input_table_key]
    xen_nucleus = sdata.shapes[args.xenium_nucleus_key]
    he_nucleus = sdata.shapes[he_nucleus_key]

    keep_mask, drop_counts = filter_by_nucleus_overlap(
        adata, xen_nucleus, he_nucleus,
        cell_id_column=args.cell_id_column,
        overlap_threshold=args.overlap_threshold,
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  drops={drop_counts}")

    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="xenium_he_nucleus_overlap",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.input_shape_key,
        output_shape_key=args.input_shape_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    path = write_stage_report(report, report_dir)
    print(f"  report → {path}")


if __name__ == "__main__":
    main()
