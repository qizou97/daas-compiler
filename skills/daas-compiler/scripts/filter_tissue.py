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
      [--key-added tissue] \
      [--allow-holes] \
      [--report-dir /data/reports]
"""
import argparse
import sys
import warnings
from pathlib import Path

import spatialdata as sd

from daas.filters.tissue import (
    TissueKeyExistsWarning,
    run_tissue_segmentation,
    filter_by_tissue,
)
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
    p.add_argument("--key-added",        default="tissue",
                   help="Shape key SOPA writes the tissue boundaries to (default: tissue)")
    p.add_argument("--allow-holes",      action="store_true", default=False,
                   help="Pass allow_holes=True to SOPA tissue segmentation")
    p.add_argument("--cell-id-column",   default="cell_id")
    p.add_argument("--report-dir",       default=None)
    return p.parse_args()


def _save_tissue_viz(sdata, image_key: str, tissue_key: str, report_dir: Path) -> None:
    """Save tissue-overlay PNG for visual QC."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import spatialdata_plot  # registers sdata.pl accessor

        viz_dir = report_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        out_path = viz_dir / "tissue_overlay.png"

        (
            sdata
            .pl.render_images(image_key)
            .pl.render_shapes(tissue_key, fill_alpha=0.0, outline_width=0.5,
                              outline_color="black")
            .pl.show(save=str(out_path))
        )
        print(f"  viz → {out_path}")
    except Exception as exc:
        print(f"  [warn] tissue viz skipped: {exc}")


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

    # Pre-run check: if the target tissue shape key already exists, the agent
    # must have already confirmed with the user (per SKILL.md contract).
    # Warn clearly so the agent/user can see what will be overwritten.
    if args.key_added in sdata.shapes:
        print(
            f"  [warn] Tissue shape key {args.key_added!r} already exists in "
            f"{zarr_path.name}. Re-running SOPA will overwrite it. "
            f"(Agent should have confirmed this with the user before running.)"
        )

    print(
        f"  running SOPA tissue segmentation "
        f"(image_key={args.image_key!r}  key_added={args.key_added!r}  "
        f"allow_holes={args.allow_holes}) …"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", TissueKeyExistsWarning)
        tissue_key = run_tissue_segmentation(
            sdata,
            image_key=args.image_key,
            allow_holes=args.allow_holes,
            key_added=args.key_added,
        )
    for w in caught:
        print(f"  [warn] {w.message}")
    print(f"  tissue shape key: {tissue_key!r}")

    _save_tissue_viz(sdata, args.image_key, tissue_key, report_dir)

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

    if output_key in sdata.tables:
        sdata.delete_element_from_disk(output_key)
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
