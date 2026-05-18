"""
Print a summary of all tables, shapes, and images in a SpatialData zarr.

Usage:
  python3 scripts/inspect_spatialdata.py --zarr /data/A_001.zarr
"""
import argparse
from pathlib import Path

import spatialdata as sd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
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
    for key, tbl in sdata.tables.items():
        print(f"    {key!r:40s}  {tbl.n_obs} cells × {tbl.n_vars} genes")

    print(f"\n  Shapes ({len(sdata.shapes)}):")
    for key, shp in sdata.shapes.items():
        print(f"    {key!r:40s}  {len(shp)} rows")

    print(f"\n  Images ({len(sdata.images)}):")
    for key, img in sdata.images.items():
        try:
            scale0 = img["scale0"]["image"]
            _, h, w = scale0.shape
            print(f"    {key!r:40s}  {h}×{w} px  "
                  f"({len(img)} pyramid levels)")
        except Exception:
            print(f"    {key!r}")

    print()


if __name__ == "__main__":
    main()
