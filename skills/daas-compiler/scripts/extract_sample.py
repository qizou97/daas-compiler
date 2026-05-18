"""
Per-sample HE patch extraction from SpatialData.
Usage:
  python3 scripts/extract_sample.py \
      --zarr   /data/A_002.zarr \
      --output /data/out/A_002 \
      [--n-sample 3000] [--patch-size 224] [--mpp 0.5] \
      [--shard-size 500] [--seed 42]
"""
import io, json, struct, tarfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import spatialdata as sd
from PIL import Image
from spatialdata.transformations import get_transformation
from wsidata import open_wsi, TileSpec
from wsidata.io import add_tiles
from daas.cli_args import parse_extract_sample_args
from daas.viz import (
    resolve_tissue_key,
    resolve_cell_boundaries_key,
    resolve_nucleus_key,
    save_tiles_overview,
    save_patch_grid,
    save_saved_patch_grid,
)
from daas.filtering import (
    PatchPolicy,
    build_filter_report,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    """Thin wrapper around the lightweight parser in daas.cli_args.

    Defined here so that ``python3 extract_sample.py --help`` still works
    via the module's own entrypoint. Policy combinations are validated at
    parse time (e.g. ``stvisuome_minimal`` + ``full_*`` exits 2).
    """
    return parse_extract_sample_args()

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

    # ── Phase 1b: Load table + shapes, align ─────────────────────────────────
    print(f"[1b] Loading table={args.table_key!r}  shapes={args.shapes_key!r} …")
    if args.table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.table_key!r}. "
            f"Available tables: {list(sdata.tables.keys())}. "
            "Run filter_tissue.py / filter_nucleus_presence.py first, or "
            "pass --table-key with the correct key."
        )
    adata_source = sdata.tables[args.table_key]
    gdf_full = sdata.shapes[args.shapes_key]
    n_cells_source = int(adata_source.n_obs)

    align = resolve_table_shape_alignment(
        adata_source, gdf_full, cell_id_column=args.cell_id_column,
    )
    if align.alignment_mode != "exact":
        print(f"      [align] non-exact: {align.n_aligned} aligned from "
              f"table={align.n_table_in}, shapes={align.n_shape_in}")
    adata = adata_source[align.adata_row_indices].copy()
    gdf   = gdf_full.iloc[align.shape_row_indices].copy()
    n_after_shape_alignment = int(adata.n_obs)
    print(f"      {n_after_shape_alignment} cells, {adata.n_vars} genes — alignment OK")

    # ── Resolve overlay keys (pure Python, no matplotlib) ────────────────────
    tissue_key  = resolve_tissue_key(sdata, hint=args.tissue_shapes_key)
    cell_key    = resolve_cell_boundaries_key(sdata, hint=args.cell_boundaries_key)
    nucleus_key = resolve_nucleus_key(sdata, hint=args.nucleus_boundaries_key)
    print(f"      overlay keys: tissue={tissue_key!r}  cell={cell_key!r}  "
          f"nucleus={nucleus_key!r}")

    # ── Phase 2: MPP derivation ───────────────────────────────────────────────
    print("[2/9] Deriving SLIDE_MPP …")
    img_tf    = get_transformation(sdata.images[args.image_key],
                                   to_coordinate_system="global")
    # Use the un-sliced shape layer so the SpatialData transformation
    # metadata is preserved (iloc-sliced GeoDataFrames may drop it).
    shape_tf  = get_transformation(gdf_full, to_coordinate_system="global")
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

    # ── Phase 3: Patch-validity filtering (Layer 2) ──────────────────────────
    print("[3/9] Filtering by patch policy …")
    centroids = gdf.geometry
    cx_um = np.array([c.x for c in centroids], dtype=np.float64)
    cy_um = np.array([c.y for c in centroids], dtype=np.float64)
    cx_px = cx_um * SCALE_SHAPE
    cy_px = cy_um * SCALE_SHAPE
    sx0 = cx_px - BASE_HALF
    sy0 = cy_px - BASE_HALF

    requested_patch_policy = PatchPolicy(args.patch_filter_policy)
    patch_policy_applied   = resolve_patch_policy(
        requested_patch_policy, args.extract_mode
    )

    pos_mask = mask_positive_centroid(cx_px, cy_px)
    n_after_positive_centroid = int(pos_mask.sum())

    patch_res = mask_patch_policy(
        sx0, sy0,
        base_size=BASE_SIZE, img_w=IMG_W, img_h=IMG_H,
        policy=patch_policy_applied,
        extract_mode=args.extract_mode,
    )
    final_mask = pos_mask & patch_res.valid_mask
    n_after_patch_policy = int(final_mask.sum())
    n_valid = n_after_patch_policy
    print(f"      valid={n_valid}  pos_centroid={n_after_positive_centroid}  "
          f"full_oob={int(patch_res.full_oob_mask.sum())}  "
          f"need_pad={int(patch_res.need_pad_mask.sum())}  "
          f"policy={patch_res.policy}")

    # ── Phase 3b: Sampling ────────────────────────────────────────────────────
    rng           = np.random.default_rng(args.seed)
    valid_indices = np.where(final_mask)[0]
    n_out         = args.n_sample if args.n_sample else n_valid
    assert n_out <= n_valid, f"n_sample={n_out} > n_valid={n_valid}"
    sampled_orig = (rng.choice(valid_indices, size=n_out, replace=False)
                    if args.n_sample else valid_indices)
    cx_px_s = cx_px[sampled_orig]; cy_px_s = cy_px[sampled_orig]
    sx0_s   = sx0[sampled_orig];   sy0_s   = sy0[sampled_orig]
    gene_row_s = sampled_orig.copy()

    # ── Phase 3c: Filter report (written BEFORE any shards) ──────────────────
    drop_counts_by_reason: dict = {}
    unaligned_dropped = n_cells_source - n_after_shape_alignment
    if unaligned_dropped:
        drop_counts_by_reason["unaligned_with_shapes"] = int(unaligned_dropped)
    drop_counts_by_reason["non_positive_centroid"] = int((~pos_mask).sum())
    drop_counts_by_reason.update(
        {k: int(v) for k, v in patch_res.drop_counts.items()}
    )
    drop_counts_by_reason["requested_subsample"] = int(n_valid - n_out)

    report_dict = build_filter_report(
        sample_id=sample_id,
        zarr_path=str(zarr_path),
        output_dir=str(output_dir),
        image_key=args.image_key,
        extract_mode=args.extract_mode,
        source_table_key=args.table_key,
        source_shape_key=args.shapes_key,
        patch_policy_requested=requested_patch_policy.value,
        patch_policy_applied=patch_policy_applied.value,
        n_cells_source=n_cells_source,
        n_after_shape_alignment=int(n_after_shape_alignment),
        n_after_positive_centroid=int(n_after_positive_centroid),
        n_after_patch_policy=int(n_after_patch_policy),
        n_out=int(n_out),
        drop_counts_by_reason=drop_counts_by_reason,
        patch_size=int(PATCH_SIZE),
        target_mpp=float(MPP_TGT),
        slide_mpp=float(SLIDE_MPP),
        base_size=int(BASE_SIZE),
        image_width_px=int(IMG_W),
        image_height_px=int(IMG_H),
        seed=int(args.seed),
        warnings=[],
    )
    report_path = write_filter_report(
        report_dict, output_dir, name=args.filter_report_name
    )
    print(f"      filter report → {report_path}")

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

    # ── Phase 6: Pre-flight viz (BEFORE writing any shards) ──────────────────
    print("[6/9] Pre-flight viz: global tiles overview + patch grid …")
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    add_tiles(wsi, key="cell_tiles",
              xys=np.column_stack([sx0_ord, sy0_ord]),
              tile_spec=spec, tissue_ids=np.zeros(n_out, dtype=int))

    # 6a. Global tiles overview (+ tissue overlay if key resolved)
    overview_result = save_tiles_overview(
        output_dir, wsi, sdata=sdata,
        tissue_key=tissue_key, scale_shape=SCALE_SHAPE,
    )
    print(f"        → {overview_result['viz_global_tiles']}")
    if overview_result["viz_global_tiles_tissue_overlay"]:
        print(f"        → {overview_result['viz_global_tiles_tissue_overlay']} (tissue overlay)")
    for w in overview_result["warnings"]:
        print(f"        [warn] {w}")

    # 6b. Patch grid: 25 random in-memory test patches with boundary overlays
    n_grid   = min(25, n_out)
    rng_grid = np.random.default_rng(args.seed)
    grid_idx = rng_grid.choice(n_out, n_grid, replace=False)
    grid_xys = np.column_stack([sx0_ord[grid_idx], sy0_ord[grid_idx]])
    add_tiles(wsi, key="patch_grid", xys=grid_xys, tile_spec=spec,
              tissue_ids=np.zeros(n_grid, dtype=int))
    grid_images = []
    for tile in wsi.iter.tile_images("patch_grid"):
        assert tile.image.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        assert tile.image.dtype == np.uint8
        grid_images.append(tile.image)
    grid_out = save_patch_grid(
        grid_images, [cell_ids_ord[i] for i in grid_idx],
        sx0_ord[grid_idx], sy0_ord[grid_idx],
        sdata, SCALE_SHAPE, PATCH_SIZE, BASE_SIZE,
        sample_id, viz_dir,
        cell_key=cell_key, nucleus_key=nucleus_key,
    )
    print(f"        → {grid_out}")

    # 6c. Write viz_report.json
    viz_report = {
        "tissue_key": tissue_key,
        "cell_key": cell_key,
        "nucleus_key": nucleus_key,
        "viz_global_tiles": overview_result["viz_global_tiles"],
        "viz_global_tiles_tissue_overlay": overview_result["viz_global_tiles_tissue_overlay"],
        "viz_patch_grid": str(grid_out),
        "warnings": overview_result["warnings"],
    }
    (viz_dir / "viz_report.json").write_text(json.dumps(viz_report, indent=2))

    # ── Phase 7: Extract + write shards ──────────────────────────────────────
    mode = args.extract_mode
    print(f"[7/9] Extracting {n_out} patches (mode={mode}) …")

    if mode == "tile_images":
        # cell_tiles was already registered in Phase 6
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
            "source_table_key":   args.table_key,
            "source_shape_key":   args.shapes_key,
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

    # ── Phase 8c: Post-save viz (reads from actual tar shards) ───────────────
    print("[8c] Post-save patch grid validation …")
    saved_viz_result = save_saved_patch_grid(
        manifest_df=cells_df,
        sdata=sdata,
        viz_dir=viz_dir,
        sample_id=sample_id,
        patch_size=PATCH_SIZE,
        scale_shape=SCALE_SHAPE,
        x0_col="bbox_x0",
        y0_col="bbox_y0",
        cell_key=cell_key,
        nucleus_key=nucleus_key,
        base_size=BASE_SIZE,
        n_grid=25,
        seed=args.seed,
    )
    print(f"      → {saved_viz_result.get('viz_saved_patch_grid')}")
    print(f"      n_checked={saved_viz_result['n_checked']}  "
          f"n_rendered={saved_viz_result['n_rendered']}  "
          f"missing={saved_viz_result['missing_members']}  "
          f"bad_size={saved_viz_result['bad_image_size']}")

    # ── Phase 9: Validation ───────────────────────────────────────────────────
    print("[9/9] Validation …")
    validate_report = _validate(
        cells_df, adata_out, adata, BASE_HALF, n_out, rng, PATCH_SIZE,
        cell_id_column=args.cell_id_column,
        overlay_keys_used=[k for k in [tissue_key, cell_key, nucleus_key] if k],
    )

    total_mb = sum(f.stat().st_size for f in output_dir.glob("*.tar")) / 1e6
    elapsed  = time.time()-t0
    n_shards_total = len(list(output_dir.glob("*.tar")))
    viz_note = (f"\n  viz outputs → {output_dir}/viz/"
                f"\n  filter report → {report_path}")
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

def _validate(cells_df, adata_out, adata_in, BASE_HALF, n_out, rng, patch_size,
              cell_id_column: str = "cell_id",
              overlay_keys_used: list | None = None) -> dict:
    import io, tarfile
    from PIL import Image
    if n_out == 0:
        raise ValueError(
            "No cells survived filtering — nothing to validate. "
            "Inspect filter_report.json: drop_counts_by_reason should explain "
            "where the cells went."
        )
    assert len(cells_df) == n_out == adata_out.n_obs
    assert cells_df["sample_key"].nunique() == n_out
    assert cells_df["cell_id"].nunique() == n_out
    assert (cells_df["expr_row"].values == cells_df["sample_index"].values).all(), \
        "expr_row != sample_index"
    assert (cells_df["cell_id"].values == adata_out.obs["cell_id"].values).all(), \
        "manifest cell_id != adata_out.obs cell_id"
    for i in range(min(200, n_out)):
        assert cells_df.iloc[i]["sample_key"] == adata_out.obs.iloc[i]["sample_key"]
        assert cells_df.iloc[i]["cell_id"] == adata_out.obs.iloc[i]["cell_id"]
    resolved_ids = adata_in.obs[cell_id_column].astype(str).values
    for i in range(min(50, n_out)):
        expected = resolved_ids[int(cells_df.iloc[i]["gene_row_index"])]
        assert str(cells_df.iloc[i]["cell_id"]) == expected, \
            "gene_row_index does not point to the cell_id of the resolved table"
    bbox_err = np.abs(cells_df["bbox_x0"].values
                      - (cells_df["center_x_pixel"].values - BASE_HALF))
    assert bbox_err.max() < 1.0
    n_random = min(20, n_out)
    check_idx = rng.choice(n_out, n_random, replace=False)
    bad_image_size = 0
    for si in check_idx:
        row = cells_df.iloc[int(si)]
        with tarfile.open(row["shard_path"], "r") as tf:
            jpg = tf.extractfile(f"{row['sample_key']}.jpg").read()
        img = Image.open(io.BytesIO(jpg))
        if img.size != (patch_size, patch_size):
            bad_image_size += 1
    assert bad_image_size == 0, f"{bad_image_size} JPEGs had wrong size"
    print(f"  [PASS] all validation checks (random JPEG sample size={n_random})")
    return {
        "n_checked": n_random,
        "missing_members": 0,
        "decode_errors": 0,
        "bad_image_size": bad_image_size,
        "overlay_keys_used": overlay_keys_used or [],
    }

if __name__ == "__main__":
    main()
