"""
Per-sample HE patch extraction from SpatialData.
Usage:
  python3 scripts/extract_sample.py \
      --zarr   /data/A_002.zarr \
      --output /data/out/A_002 \
      [--n-sample 3000] [--patch-size 224] [--mpp 0.5] \
      [--shard-size 500] [--seed 42]
"""
import argparse, io, json, struct, tarfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import spatialdata as sd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from spatialdata.transformations import get_transformation
from wsidata import open_wsi, TileSpec
from wsidata.io import add_tiles
import lazyslide.pl as lpl

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",        required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--sample-id",   default=None,
                   help="样本 ID，默认从 zarr 目录名推断")
    p.add_argument("--n-sample",    type=int, default=None,
                   help="随机采样数量，默认处理全部有效细胞")
    p.add_argument("--patch-size",  type=int, default=224)
    p.add_argument("--mpp",         type=float, default=0.5)
    p.add_argument("--shard-size",  type=int, default=500)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--image-key",   default="he_image")
    p.add_argument("--shapes-key",  default="cell_circles")
    p.add_argument("--table-key",   default="table")
    p.add_argument("--skip-viz",    action="store_true",
                   help="Skip optional viz (centroid overlay + patch grid). "
                        "The lazyslide.pl.tiles overview is always produced.")
    p.add_argument("--extract-mode", default="tile_images",
                   choices=["tile_images", "full_scale0", "full_ops_level"],
                   help="Patch extraction strategy: tile_images (default, "
                        "low mem), full_scale0 (fast, ~1.6 GB), "
                        "full_ops_level (fastest, ~0.4 GB)")
    return p.parse_args()

# ── IDX format ────────────────────────────────────────────────────────────────
IDX_MAGIC      = b"CIDX0001"
IDX_RECORD_FMT = "<iQIQII"
IDX_RECORD_SIZE = struct.calcsize(IDX_RECORD_FMT)
assert IDX_RECORD_SIZE == 32, f"BUG: {IDX_RECORD_SIZE}"

def flush_shard(shard_buf, shard_no, output_dir):
    tar_path = output_dir / f"shard-{shard_no:06d}.tar"
    idx_path = output_dir / f"shard-{shard_no:06d}.idx"
    with tarfile.open(tar_path, "w") as tf:
        for si, sk, jpg_bytes, json_bytes in shard_buf:
            for ext, data in [(".jpg", jpg_bytes), (".json", json_bytes)]:
                ti = tarfile.TarInfo(name=f"{sk}{ext}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    records = []
    with tarfile.open(tar_path, "r") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for si, sk, jpg_bytes, json_bytes in shard_buf:
            jm = members[f"{sk}.jpg"]
            nm = members[f"{sk}.json"]
            records.append((si, jm.offset_data, jm.size,
                            nm.offset_data, nm.size))
    with open(idx_path, "wb") as f:
        f.write(IDX_MAGIC)
        f.write(struct.pack("<I", len(records)))
        for rec in records:
            f.write(struct.pack(IDX_RECORD_FMT, *rec, 0))
    return str(tar_path), records

def _render_boundary_grid(images, cell_ids, x0s, y0s, sdata, SCALE_SHAPE,
                          PATCH_SIZE, BASE_SIZE, sample_id, viz_dir):
    """Render pre-flight patch grid with cell+nucleus boundary overlays."""
    cell_bounds = sdata.shapes["cell_boundaries"]
    nucl_bounds = sdata.shapes["nucleus_boundaries"]
    nucl_ids    = set(nucl_bounds.index)
    SCALE       = PATCH_SIZE / BASE_SIZE

    def shape_um_to_patch_px(coords_um, x0, y0):
        arr  = np.array(coords_um)
        col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
        row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
        return np.column_stack([col, row_])

    n_test  = len(images)
    n_cols  = int(np.ceil(np.sqrt(n_test)))
    n_rows  = int(np.ceil(n_test / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_test == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    for i in range(n_test):
        ax = axes_flat[i]
        ax.imshow(images[i])

        cell_id = cell_ids[i]
        x0, y0  = x0s[i], y0s[i]

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

        # crosshair at patch center (red)
        cx = cy = PATCH_SIZE / 2
        arm = PATCH_SIZE * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)

        ax.set_xlim(0, PATCH_SIZE)
        ax.set_ylim(PATCH_SIZE, 0)
        ax.set_title(cell_id[:12], fontsize=5)
        ax.axis("off")

    for j in range(n_test, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"{sample_id} — pre-flight boundary check "
                 f"(cyan=cell  yellow=nucleus  +=center)",
                 fontsize=9, y=0.995)
    plt.tight_layout()
    out_path = viz_dir / "viz_preflight_boundary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Pre-flight boundary viz saved → {out_path}")


def _iter_full_load(full_img, x0s, y0s, crop_w, out_w, ds_x, ds_y):
    """Yield (out_w, out_w, 3) uint8 tiles from a pre-loaded full image.

    full_img: (C, H, W) numpy array  (CHW from spatialdata DataArray.values)
    x0s, y0s: level-0 tile top-left coordinates
    crop_w:   tile width in pixels at this pyramid level
    out_w:    target patch size (e.g. 224)
    ds_x, ds_y: level-0 → this pyramid level downsample ratios
    """
    for i in range(len(x0s)):
        x0 = int(x0s[i] / ds_x)
        y0 = int(y0s[i] / ds_y)
        tile_chw = full_img[:, y0:y0 + crop_w, x0:x0 + crop_w]
        tile_hwc = np.ascontiguousarray(tile_chw.transpose(1, 2, 0))
        if crop_w != out_w:
            tile_hwc = np.array(Image.fromarray(tile_hwc).resize(
                (out_w, out_w), Image.LANCZOS))
        yield tile_hwc


def _save_tiles_overview(output_dir, wsi, dpi=300):
    """Render lazyslide.pl.tiles overview. Always called, regardless of --skip-viz."""
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    print(f"  [viz] Global tiles overview (dpi={dpi}) …")
    lpl.tiles(wsi, tile_key="cell_tiles")
    fig = plt.gcf()
    out = viz_dir / "viz_global_tiles.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"        → {out}")
    return out


def _visualize(output_dir, sample_id, cells_df, sdata, wsi,
               image_key, SCALE_SHAPE, PATCH_SIZE, BASE_SIZE,
               IMG_W, IMG_H, seed, grid_n=25):
    """Generate post-extraction viz outputs in {output_dir}/viz/.

    The lazyslide.pl.tiles overview is produced separately by
    _save_tiles_overview() and is NOT gated by --skip-viz. This
    function handles the optional centroid overlay + patch grid.
    """
    import io as _io, tarfile as _tarfile
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    SCALE = PATCH_SIZE / BASE_SIZE

    # ── 1. Centroid overlay on thumbnail ─────────────────────────────────────
    print("  [viz] Centroid overlay …")
    try:
        thumb = wsi.reader.get_thumbnail(size=(2000, 2000))
    except Exception:
        thumb = sdata.images[image_key]["scale4"]["image"].values.transpose(1, 2, 0)

    th, tw = thumb.shape[:2]
    cx_t = cells_df["center_x_pixel"].values * (tw / IMG_W)
    cy_t = cells_df["center_y_pixel"].values * (th / IMG_H)

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(thumb)
    ax.scatter(cx_t, cy_t, s=1.0, c="cyan", alpha=0.6, linewidths=0)
    ax.set_title(f"{sample_id} — {len(cells_df)} sampled centroids", fontsize=11)
    ax.axis("off")
    out2 = viz_dir / "viz_centroid_overlay.png"
    fig.savefig(out2, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"        → {out2}")

    # ── 2. Patch grid with boundary overlays ─────────────────────────────────
    print(f"  [viz] Patch grid ({grid_n} patches) …")
    cell_bounds = sdata.shapes.get("cell_boundaries")
    nucl_bounds = sdata.shapes.get("nucleus_boundaries")
    nucl_ids    = set(nucl_bounds.index) if nucl_bounds is not None else set()

    def shape_um_to_patch_px(coords_um, x0, y0):
        arr  = np.array(coords_um)
        col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
        row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
        return np.column_stack([col, row_])

    grid_side  = int(np.sqrt(grid_n))
    rng_v      = np.random.default_rng(seed)
    sample_idx = rng_v.choice(len(cells_df), grid_side * grid_side, replace=False)

    fig, axes = plt.subplots(grid_side, grid_side,
                             figsize=(grid_side * 2.4, grid_side * 2.4))
    fig.suptitle(f"{sample_id} — {grid_side}×{grid_side} random patches "
                 f"({PATCH_SIZE}px @ 0.5 MPP)\n"
                 "cyan=cell boundary  yellow=nucleus boundary  +=center",
                 fontsize=9)

    for ax, si in zip(axes.flat, sample_idx):
        row     = cells_df.iloc[int(si)]
        cell_id = row["cell_id"]
        x0      = row["bbox_x0"]
        y0      = row["bbox_y0"]

        with _tarfile.open(row["shard_path"], "r") as tf:
            jpg = tf.extractfile(f"{row['sample_key']}.jpg").read()
        ax.imshow(Image.open(_io.BytesIO(jpg)))

        if cell_bounds is not None:
            try:
                cb_pts = shape_um_to_patch_px(
                    list(cell_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(cb_pts, closed=True,
                                        edgecolor="cyan", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass

        if cell_id in nucl_ids:
            try:
                nb_pts = shape_um_to_patch_px(
                    list(nucl_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(nb_pts, closed=True,
                                        edgecolor="yellow", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass

        cx = cy = PATCH_SIZE / 2
        arm = PATCH_SIZE * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)

        ax.set_xlim(0, PATCH_SIZE); ax.set_ylim(PATCH_SIZE, 0)
        ax.set_title(cell_id[:12], fontsize=4.5)
        ax.axis("off")

    plt.tight_layout()
    out3 = viz_dir / "viz_patch_grid.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"        → {out3}")


def main():
    args   = parse_args()
    zarr_path  = args.zarr
    output_dir = Path(args.output)
    sample_id  = args.sample_id or Path(zarr_path).stem  # e.g. "A_002"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Phase 1: Load ─────────────────────────────────────────────────────────
    print(f"[1/9] Loading {zarr_path} …")
    sdata  = sd.read_zarr(zarr_path)
    gdf    = sdata.shapes[args.shapes_key]
    adata  = sdata.tables[args.table_key]
    assert list(gdf.index) == list(adata.obs["cell_id"]), \
        "cell_id row-order mismatch"
    print(f"      {len(gdf)} cells, {adata.n_vars} genes — alignment OK")

    # ── Phase 2: MPP derivation ───────────────────────────────────────────────
    print("[2/9] Deriving SLIDE_MPP …")
    img_tf    = get_transformation(sdata.images[args.image_key],
                                   to_coordinate_system="global")
    shape_tf  = get_transformation(gdf, to_coordinate_system="global")
    img_aff   = img_tf.to_affine_matrix(input_axes=("y","x"),
                                         output_axes=("y","x"))
    shape_aff = shape_tf.to_affine_matrix(input_axes=("y","x"),
                                           output_axes=("y","x"))
    sx = np.sqrt(img_aff[0,0]**2 + img_aff[1,0]**2)
    sy = np.sqrt(img_aff[0,1]**2 + img_aff[1,1]**2)
    SCALE_SHAPE         = (shape_aff[0,0] + shape_aff[1,1]) / 2.0
    global_per_micron   = SCALE_SHAPE
    SLIDE_MPP           = ((sx + sy) / 2.0) / global_per_micron
    PATCH_SIZE          = args.patch_size
    MPP_TGT             = args.mpp
    BASE_SIZE           = round(PATCH_SIZE * MPP_TGT / SLIDE_MPP)
    BASE_HALF           = BASE_SIZE / 2.0
    scale0_img          = sdata.images[args.image_key]["scale0"]["image"]
    _, IMG_H, IMG_W     = scale0_img.shape
    print(f"      SLIDE_MPP={SLIDE_MPP:.6f}  BASE_SIZE={BASE_SIZE}")

    # ── Phase 3: OOB filtering ────────────────────────────────────────────────
    print("[3/9] Filtering OOB …")
    centroids = gdf.geometry
    cx_um = np.array([c.x for c in centroids], dtype=np.float64)
    cy_um = np.array([c.y for c in centroids], dtype=np.float64)
    cx_px = cx_um * SCALE_SHAPE
    cy_px = cy_um * SCALE_SHAPE
    sx0 = cx_px - BASE_HALF
    sy0 = cy_px - BASE_HALF
    full_oob = ((sx0+BASE_SIZE<=0)|(sx0>=IMG_W)|(sy0+BASE_SIZE<=0)|(sy0>=IMG_H))
    need_pad = (((sx0<0)|(sx0+BASE_SIZE>IMG_W)|(sy0<0)|(sy0+BASE_SIZE>IMG_H))
                & ~full_oob)
    valid_mask = ~full_oob & ~need_pad
    n_valid = valid_mask.sum()
    print(f"      valid={n_valid}  full_oob={full_oob.sum()}  "
          f"need_pad={need_pad.sum()}")

    # ── Phase 3b: Sampling ────────────────────────────────────────────────────
    rng           = np.random.default_rng(args.seed)
    valid_indices = np.where(valid_mask)[0]
    n_out         = args.n_sample if args.n_sample else n_valid
    assert n_out <= n_valid, f"n_sample={n_out} > n_valid={n_valid}"
    sampled_orig = (rng.choice(valid_indices, size=n_out, replace=False)
                    if args.n_sample else valid_indices)
    cx_px_s = cx_px[sampled_orig]; cy_px_s = cy_px[sampled_orig]
    sx0_s   = sx0[sampled_orig];   sy0_s   = sy0[sampled_orig]
    gene_row_s = sampled_orig.copy()

    # ── Phase 4: Spatial sort ─────────────────────────────────────────────────
    sort_key = ((np.maximum(0,sy0_s)//4096)*10000
                +(np.maximum(0,sx0_s)//4096))
    proc_order  = np.argsort(sort_key, kind="stable")
    cx_px_ord   = cx_px_s[proc_order]; cy_px_ord   = cy_px_s[proc_order]
    sx0_ord     = sx0_s[proc_order];   sy0_ord     = sy0_s[proc_order]
    cell_ids_ord = [gdf.index[sampled_orig[i]] for i in proc_order]
    gene_row_ord = gene_row_s[proc_order]
    orig_idx_ord = sampled_orig[proc_order]

    # ── Phase 5: TileSpec ─────────────────────────────────────────────────────
    print("[5/9] TileSpec …")
    wsi = open_wsi(sdata, image_key=args.image_key, store=None)
    wsi.set_mpp(SLIDE_MPP)
    spec = TileSpec.from_wsidata(wsi, tile_px=PATCH_SIZE,
                                  mpp=MPP_TGT, slide_mpp=SLIDE_MPP)
    assert spec.base_width == BASE_SIZE

    # ── Phase 6: Pre-flight + boundary viz ─────────────────────────────────────
    print("[6/9] Pre-flight: test tiles + boundary overlay …")
    n_test     = min(9, n_out)
    test_xys   = np.column_stack([sx0_ord[:n_test], sy0_ord[:n_test]])
    add_tiles(wsi, key="test_tiles", xys=test_xys, tile_spec=spec,
              tissue_ids=np.zeros(n_test, dtype=int))
    test_images = []
    for tile in wsi.iter.tile_images("test_tiles"):
        assert tile.image.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        assert tile.image.dtype == np.uint8
        test_images.append(tile.image)

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    _render_boundary_grid(test_images, cell_ids_ord[:n_test],
                          sx0_ord[:n_test], sy0_ord[:n_test],
                          sdata, SCALE_SHAPE, PATCH_SIZE, BASE_SIZE,
                          sample_id, viz_dir)

    # ── Phase 7: Extract + write shards ──────────────────────────────────────
    mode = args.extract_mode
    print(f"[7/9] Extracting {n_out} patches (mode={mode}) …")

    if mode == "tile_images":
        add_tiles(wsi, key="cell_tiles",
                  xys=np.column_stack([sx0_ord, sy0_ord]),
                  tile_spec=spec, tissue_ids=np.zeros(n_out, dtype=int))
        def _tile_gen():
            for tile in wsi.iter.tile_images("cell_tiles"):
                yield tile.image
        tile_iter = _tile_gen()
    elif mode == "full_scale0":
        full_img = sdata.images[args.image_key]["scale0"]["image"].values
        tile_iter = _iter_full_load(full_img, sx0_ord, sy0_ord,
                                     BASE_SIZE, PATCH_SIZE, 1.0, 1.0)
    elif mode == "full_ops_level":
        lvl_key = f"scale{spec.ops_level}"
        lvl_img = sdata.images[args.image_key][lvl_key]["image"]
        full_img = lvl_img.values
        ds_y = scale0_img.shape[1] / full_img.shape[1]
        ds_x = scale0_img.shape[2] / full_img.shape[2]
        print(f"      ops_level={spec.ops_level}  ds=({ds_x:.4f}, {ds_y:.4f})  "
              f"crop={spec.ops_width}px  mem={full_img.nbytes/1e6:.0f}MB")
        tile_iter = _iter_full_load(full_img, sx0_ord, sy0_ord,
                                     spec.ops_width, PATCH_SIZE, ds_x, ds_y)

    cells_rows, shard_buf, shard_no = [], [], 0
    t_ext = time.time()
    for local_i, tile_img in enumerate(tile_iter):
        sk       = f"{local_i:07d}"
        cell_id  = cell_ids_ord[local_i]
        orig_idx = int(orig_idx_ord[local_i])
        jpg_buf  = io.BytesIO()
        Image.fromarray(tile_img).save(jpg_buf, format="JPEG", quality=95)
        jpg_bytes  = jpg_buf.getvalue()
        meta = {
            "sample_index":       local_i,
            "sample_key":         sk,
            "sample_id":          sample_id,
            "cell_id":            cell_id,
            "center_x_shape_um":  float(cx_um[orig_idx]),
            "center_y_shape_um":  float(cy_um[orig_idx]),
            "center_x_pixel":     float(cx_px_ord[local_i]),
            "center_y_pixel":     float(cy_px_ord[local_i]),
            "bbox_x0":            float(sx0_ord[local_i]),
            "bbox_y0":            float(sy0_ord[local_i]),
            "gene_row_index":     int(gene_row_ord[local_i]),
            "orig_cell_index":    orig_idx,
            "mpp_target":         MPP_TGT,
            "slide_mpp":          SLIDE_MPP,
        }
        shard_buf.append((local_i, sk, jpg_bytes,
                          json.dumps(meta).encode()))
        cells_rows.append({**meta, "shard_no": shard_no,
                           "local_index": len(shard_buf)-1})
        if len(shard_buf) == args.shard_size:
            tar_path, recs = flush_shard(shard_buf, shard_no, output_dir)
            for row, rec in zip(cells_rows[-args.shard_size:], recs):
                row["shard_path"] = tar_path
                row["tar_offset"] = rec[1]
                row["jpg_size"]   = rec[2]
            shard_buf = []; shard_no += 1
        if (local_i+1) % 1000 == 0:
            rate = (local_i+1)/(time.time()-t_ext)
            print(f"      {local_i+1}/{n_out}  {rate:.0f} patches/s")
    if shard_buf:
        tar_path, recs = flush_shard(shard_buf, shard_no, output_dir)
        for row, rec in zip(cells_rows[-len(shard_buf):], recs):
            row["shard_path"] = tar_path
            row["tar_offset"] = rec[1]
            row["jpg_size"]   = rec[2]

    # ── Phase 8: h5ad ─────────────────────────────────────────────────────────
    print("[8/9] Saving h5ad …")
    cells_df = pd.DataFrame(cells_rows).reset_index(drop=True)
    cells_df["sample_id"] = sample_id
    X_out = adata.X[gene_row_ord, :]
    obs_out = cells_df[["sample_index","sample_key","sample_id","cell_id",
                         "shard_path","shard_no","local_index",
                         "gene_row_index","orig_cell_index",
                         "center_x_shape_um","center_y_shape_um",
                         "center_x_pixel","center_y_pixel",
                         "bbox_x0","bbox_y0"]].copy()
    obs_out.index = obs_out["sample_key"].values
    adata_out = anndata.AnnData(X=X_out, obs=obs_out, var=adata.var.copy())
    adata_out.obs["sample_id"] = sample_id
    adata_out.write_h5ad(output_dir / "expression.h5ad")

    # ── Phase 8b: manifest.parquet ────────────────────────────────────────────
    cells_df["expr_row"] = cells_df["sample_index"]   # local row in expression.h5ad
    cells_df.to_parquet(output_dir / "manifest.parquet", index=False)

    # ── Phase 9: Validation ───────────────────────────────────────────────────
    print("[9/9] Validation …")
    _validate(cells_df, adata_out, adata, BASE_HALF, n_out, rng, PATCH_SIZE)

    # Tiles overview is always produced (small, sample-essential).
    _save_tiles_overview(output_dir, wsi)

    if not args.skip_viz:
        _visualize(output_dir, sample_id, cells_df, sdata, wsi,
                   args.image_key, SCALE_SHAPE, PATCH_SIZE, BASE_SIZE,
                   IMG_W, IMG_H, args.seed)

    total_mb = sum(f.stat().st_size for f in output_dir.glob("*.tar")) / 1e6
    elapsed  = time.time()-t0
    n_shards_total = len(list(output_dir.glob("*.tar")))
    viz_note = f"\n  viz outputs → {output_dir}/viz/"
    print(f"""
{'='*60}
  EXTRACT COMPLETE — {sample_id}  ({n_out} cells, {n_shards_total} shards, {total_mb:.0f}MB){viz_note}
{'─'*60}
# 验证 patches:
import tarfile, io; from PIL import Image
with tarfile.open("{output_dir}/shard-000000.tar") as tf:
    img = Image.open(io.BytesIO(tf.extractfile("0000000.jpg").read()))

# 验证 expression:
import anndata
adata = anndata.read_h5ad("{output_dir}/expression.h5ad")
print(adata)  # AnnData ({n_out}, {adata.n_vars})
{'='*60}
  Total time: {elapsed:.1f}s
""")

def _validate(cells_df, adata_out, adata_in, BASE_HALF, n_out, rng, patch_size):
    import io, tarfile
    from PIL import Image
    assert len(cells_df) == n_out == adata_out.n_obs
    assert cells_df["sample_key"].nunique() == n_out
    assert cells_df["cell_id"].nunique() == n_out
    for i in range(min(200, n_out)):
        assert cells_df.iloc[i]["sample_key"] == adata_out.obs.iloc[i]["sample_key"]
        assert cells_df.iloc[i]["cell_id"] == adata_out.obs.iloc[i]["cell_id"]
    for i in range(min(50, n_out)):
        expected = adata_in.obs["cell_id"].iloc[int(cells_df.iloc[i]["gene_row_index"])]
        assert cells_df.iloc[i]["cell_id"] == expected
    bbox_err = np.abs(cells_df["bbox_x0"].values
                      - (cells_df["center_x_pixel"].values - BASE_HALF))
    assert bbox_err.max() < 1.0
    check_idx = rng.choice(n_out, 20, replace=False)
    for si in check_idx:
        row = cells_df.iloc[int(si)]
        with tarfile.open(row["shard_path"], "r") as tf:
            jpg = tf.extractfile(f"{row['sample_key']}.jpg").read()
        img = Image.open(io.BytesIO(jpg))
        assert img.size == (patch_size, patch_size)
    print("  [PASS] all 6 validation checks")

if __name__ == "__main__":
    main()
