"""
Visualization validation for a per-sample extraction output.
Usage:
  python3 scripts/viz_sample.py \
      --zarr   /data/A_002.zarr \
      --output /data/out/A_002
"""
import argparse, io, tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lazyslide.pl as lpl
import spatialdata as sd
from PIL import Image
from wsidata import open_wsi, TileSpec
from wsidata.io import add_tiles
from spatialdata.transformations import get_transformation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",       required=True)
    p.add_argument("--output",     required=True,
                   help="per-sample output dir (contains manifest.parquet)")
    p.add_argument("--image-key",  default="he_image")
    p.add_argument("--shapes-key", default="cell_circles")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--grid-n",     type=int, default=25,
                   help="number of patches in grid (must be a perfect square)")
    return p.parse_args()


def main():
    args       = parse_args()
    output_dir = Path(args.output)
    viz_dir    = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    print(f"[1] Loading manifest and SpatialData …")
    manifest = pd.read_parquet(output_dir / "manifest.parquet")
    # ensure sample_key is string
    manifest["sample_key"] = manifest["sample_key"].astype(str)
    print(f"    {len(manifest)} rows")

    sdata = sd.read_zarr(args.zarr)
    gdf   = sdata.shapes[args.shapes_key]

    # Derive slide MPP for TileSpec
    img_tf    = get_transformation(sdata.images[args.image_key],
                                   to_coordinate_system="global")
    shape_tf  = get_transformation(gdf, to_coordinate_system="global")
    img_aff   = img_tf.to_affine_matrix(input_axes=("y","x"),
                                         output_axes=("y","x"))
    shape_aff = shape_tf.to_affine_matrix(input_axes=("y","x"),
                                           output_axes=("y","x"))
    sx = np.sqrt(img_aff[0,0]**2 + img_aff[1,0]**2)
    sy = np.sqrt(img_aff[0,1]**2 + img_aff[1,1]**2)
    SCALE_SHAPE = (shape_aff[0,0] + shape_aff[1,1]) / 2.0
    SLIDE_MPP   = ((sx + sy) / 2.0) / SCALE_SHAPE
    PATCH_SIZE  = 224
    MPP_TGT     = 0.5
    BASE_SIZE   = round(PATCH_SIZE * MPP_TGT / SLIDE_MPP)
    SCALE       = PATCH_SIZE / BASE_SIZE   # level-0 px → output px

    scale0_img = sdata.images[args.image_key]["scale0"]["image"]
    _, IMG_H, IMG_W = scale0_img.shape

    print(f"    SLIDE_MPP={SLIDE_MPP:.6f}  BASE_SIZE={BASE_SIZE}  "
          f"image={IMG_W}x{IMG_H}")

    # ── 1. Global tiles overview ──────────────────────────────────────────────
    print("[2] Global tiles overview …")
    wsi = open_wsi(sdata, image_key=args.image_key, store=None)
    wsi.set_mpp(SLIDE_MPP)
    spec = TileSpec.from_wsidata(wsi, tile_px=PATCH_SIZE,
                                  mpp=MPP_TGT, slide_mpp=SLIDE_MPP)
    xys = manifest[["bbox_x0", "bbox_y0"]].values
    add_tiles(wsi, key="cell_tiles", xys=xys, tile_spec=spec,
              tissue_ids=np.zeros(len(manifest), dtype=int))
    lpl.tiles(wsi, tile_key="cell_tiles")
    fig = plt.gcf()
    out1 = viz_dir / "viz_global_tiles.png"
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out1}")

    sample_id = manifest["sample_id"].iloc[0]

    # ── 2. Patch grid with boundary overlays ──────────────────────────────────
    print(f"[3] Patch grid ({args.grid_n} patches) …")
    from matplotlib.patches import Polygon as MplPolygon

    cell_bounds = sdata.shapes["cell_boundaries"]
    nucl_bounds = sdata.shapes["nucleus_boundaries"]
    nucl_ids    = set(nucl_bounds.index)

    grid_side = int(np.sqrt(args.grid_n))
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(manifest), grid_side * grid_side, replace=False)

    fig, axes = plt.subplots(grid_side, grid_side,
                             figsize=(grid_side * 2.4, grid_side * 2.4))
    fig.suptitle(f"{sample_id} — {grid_side}×{grid_side} random patches "
                 f"(224px @ 0.5 MPP)\n"
                 "cyan=cell boundary  yellow=nucleus boundary  +=center",
                 fontsize=9)

    def shape_um_to_patch_px(coords_um, x0, y0):
        arr  = np.array(coords_um)
        col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
        row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
        return np.column_stack([col, row_])

    for ax, si in zip(axes.flat, sample_idx):
        row     = manifest.iloc[int(si)]
        cell_id = row["cell_id"]
        x0      = row["bbox_x0"]
        y0      = row["bbox_y0"]

        with tarfile.open(row["shard_path"], "r") as tf:
            jpg = tf.extractfile(f"{row['sample_key']}.jpg").read()
        ax.imshow(Image.open(io.BytesIO(jpg)))

        # cell boundary (cyan)
        try:
            cb_pts = shape_um_to_patch_px(
                list(cell_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
            ax.add_patch(MplPolygon(cb_pts, closed=True,
                                    edgecolor="cyan", facecolor="none",
                                    linewidth=0.8, alpha=0.9))
        except KeyError:
            pass

        # nucleus boundary (yellow)
        if cell_id in nucl_ids:
            try:
                nb_pts = shape_um_to_patch_px(
                    list(nucl_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(nb_pts, closed=True,
                                        edgecolor="yellow", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass

        # crosshair at patch center
        cx = cy = PATCH_SIZE / 2
        arm = PATCH_SIZE * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)

        ax.set_xlim(0, PATCH_SIZE); ax.set_ylim(PATCH_SIZE, 0)
        ax.set_title(cell_id[:12], fontsize=4.5)
        ax.axis("off")

    plt.tight_layout()
    out3 = viz_dir / "viz_patch_grid.png"
    fig.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {out3}")

    print(f"\nDone. Outputs in {viz_dir}/")


if __name__ == "__main__":
    main()
