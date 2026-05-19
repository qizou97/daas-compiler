"""
Print a summary of all tables, shapes, and images in a SpatialData zarr.

Usage:
  python3 scripts/inspect_spatialdata.py --zarr /data/A_001.zarr
  python3 scripts/inspect_spatialdata.py --zarr /data/A_001.zarr --report-dir /tmp/reports
"""
import argparse
import json
from pathlib import Path

import spatialdata as sd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    p.add_argument("--report-dir", default=None,
                   help="Directory to write inspect_report.json. Default: no JSON output.")
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = args.zarr
    print(f"[inspect] Loading {zarr_path} …")
    sdata = sd.read_zarr(zarr_path)

    print(f"\n{'='*60}")
    print(f"  SpatialData: {Path(zarr_path).name}")
    print(f"{'='*60}")

    print(f"\n  Tables ({len(sdata.tables)}):")
    tables_info = []
    for key, tbl in sdata.tables.items():
        print(f"    {key!r:40s}  {tbl.n_obs} cells × {tbl.n_vars} genes")
        tables_info.append({"key": key, "n_obs": tbl.n_obs, "n_vars": tbl.n_vars})

    print(f"\n  Shapes ({len(sdata.shapes)}):")
    shapes_info = []
    for key, shp in sdata.shapes.items():
        print(f"    {key!r:40s}  {len(shp)} rows")
        shapes_info.append({"key": key, "n_rows": len(shp)})

    print(f"\n  Images ({len(sdata.images)}):")
    images_info = []
    for key, img in sdata.images.items():
        try:
            scale0 = img["scale0"]["image"]
            _, h, w = scale0.shape
            print(f"    {key!r:40s}  {h}×{w} px  "
                  f"({len(img)} pyramid levels)")
            images_info.append({"key": key, "height": h, "width": w, "n_levels": len(img)})
        except Exception:
            print(f"    {key!r}")
            images_info.append({"key": key})

    print()

    if args.report_dir is not None:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "zarr_path": str(Path(zarr_path).resolve()),
            "sample_id": Path(zarr_path).stem,
            "tables": tables_info,
            "shapes": shapes_info,
            "images": images_info,
        }
        report_path = report_dir / "inspect_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"  report → {report_path}")


if __name__ == "__main__":
    main()
