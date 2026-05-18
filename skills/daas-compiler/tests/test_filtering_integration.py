"""Integration-style tests for the filtering policy layer.

These do not run the full ``extract_sample.py`` CLI (which requires
``wsidata``/``lazyslide`` + a real H&E pyramid). Instead they mirror the
orchestration of Phases 1b–3c so the alignment invariants and the
filter-report contract are verified end-to-end at the Python level.
"""
import json
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import anndata
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from daas.filtering import (
    BiologicalPolicy,
    PatchPolicy,
    build_filter_report,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_biological_policy,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)


# ── small synthetic SpatialData stand-in ─────────────────────────────────
def _adata(cell_ids, n_genes: int = 4) -> anndata.AnnData:
    n = len(cell_ids)
    X = csr_matrix(
        np.arange(n * n_genes, dtype=np.float32).reshape(n, n_genes) + 1
    )
    obs = pd.DataFrame({"cell_id": [str(c) for c in cell_ids]})
    obs.index = [f"{i:07d}" for i in range(n)]
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _shapes(cell_ids, xy_um) -> gpd.GeoDataFrame:
    """GeoDataFrame of point centroids in micron coordinates."""
    return gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in xy_um]},
        index=pd.Index([str(c) for c in cell_ids], name="cell_id"),
    )


def _sdata(tables: dict, shapes: dict):
    return SimpleNamespace(tables=tables, shapes=shapes)


# ── Required test 5: alignment preservation through the full pipeline ───
def test_alignment_preserved_end_to_end(tmp_path):
    """Run the exact orchestration extract_sample.py uses and assert that
    manifest row order ≡ expression.h5ad row order, ``expr_row``
    ≡ ``sample_index``, and ``gene_row_index`` indexes the resolved
    ``adata`` row whose ``cell_id`` matches the manifest's ``cell_id``.
    """
    # Build 6 cells with mixed signs and a few boundary tiles.
    # SLIDE_MPP = 1.0 µm/px, SCALE_SHAPE = 1.0 (microns == pixels), BASE_SIZE = 20.
    cell_ids = ["c0", "c1", "c2", "c3", "c4", "c5"]
    xy_um = [
        (40, 40),    # fully inside  (keep)
        (60, 60),    # fully inside  (keep)
        (-50, 10),   # full_oob      (drop)
        (-5, 10),    # need_pad      (drop strict)
        (0, 10),     # non-positive  (drop)
        (95, 95),    # need_pad      (drop strict)
    ]
    adata_in = _adata(cell_ids)
    gdf_in = _shapes(cell_ids, xy_um)
    sdata = _sdata(tables={"table": adata_in}, shapes={"cell_circles": gdf_in})

    # Phase 1b — biological resolve (none, since no filtered_table)
    bio = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.AUTO,
        table_key_was_default=True,
    )
    assert bio.policy_applied is BiologicalPolicy.NONE
    adata_after_bio = sdata.tables[bio.table_key_used]
    if bio.keep_table_mask is not None:
        adata_after_bio = adata_after_bio[bio.keep_table_mask].copy()
    gdf_full = sdata.shapes[bio.shapes_key_used]

    align = resolve_table_shape_alignment(adata_after_bio, gdf_full)
    assert align.alignment_mode == "exact"
    adata = adata_after_bio[align.adata_row_indices].copy()
    gdf = gdf_full.iloc[align.shape_row_indices].copy()

    # Phase 2 — fixed scales (no real image; this is what extract_sample.py
    # would compute from the transformations).
    SCALE_SHAPE = 1.0
    SLIDE_MPP = 1.0
    BASE_SIZE = 20
    BASE_HALF = BASE_SIZE / 2.0
    IMG_W = IMG_H = 100

    centroids = gdf.geometry
    cx_um = np.array([c.x for c in centroids], dtype=np.float64)
    cy_um = np.array([c.y for c in centroids], dtype=np.float64)
    cx_px = cx_um * SCALE_SHAPE
    cy_px = cy_um * SCALE_SHAPE
    sx0 = cx_px - BASE_HALF
    sy0 = cy_px - BASE_HALF

    # Phase 3 — patch policy
    patch_policy = resolve_patch_policy(PatchPolicy.AUTO, "tile_images")
    assert patch_policy is PatchPolicy.STRICT_NO_PADDING
    pos_mask = mask_positive_centroid(cx_px, cy_px)
    patch_res = mask_patch_policy(
        sx0, sy0,
        base_size=BASE_SIZE, img_w=IMG_W, img_h=IMG_H,
        policy=patch_policy, extract_mode="tile_images",
    )
    final_mask = pos_mask & patch_res.valid_mask
    # Only c0 and c1 survive strict_no_padding.
    np.testing.assert_array_equal(
        final_mask, [True, True, False, False, False, False]
    )

    # Phase 3b — sample everything (no subsample for determinism)
    valid_indices = np.where(final_mask)[0]
    n_out = len(valid_indices)
    sampled_orig = valid_indices

    cx_px_s = cx_px[sampled_orig]
    cy_px_s = cy_px[sampled_orig]
    sx0_s = sx0[sampled_orig]
    sy0_s = sy0[sampled_orig]
    gene_row_s = sampled_orig.copy()

    # Phase 4 — spatial sort
    sort_key = (np.maximum(0, sy0_s) // 4096) * 10000 + (
        np.maximum(0, sx0_s) // 4096
    )
    proc_order = np.argsort(sort_key, kind="stable")
    cx_px_ord = cx_px_s[proc_order]
    sx0_ord = sx0_s[proc_order]
    sy0_ord = sy0_s[proc_order]
    cell_ids_ord = [gdf.index[sampled_orig[i]] for i in proc_order]
    gene_row_ord = gene_row_s[proc_order]

    # Phase 8 — construct manifest + h5ad-like outputs the same way the
    # script does, then assert the row-order invariants.
    cells_rows = [
        {
            "sample_index": i,
            "sample_key": f"{i:07d}",
            "cell_id": cell_ids_ord[i],
            "gene_row_index": int(gene_row_ord[i]),
            "expr_row": i,  # extract_sample.py sets expr_row = sample_index
        }
        for i in range(n_out)
    ]
    cells_df = pd.DataFrame(cells_rows)
    X_out = adata.X[gene_row_ord, :]
    obs_out = pd.DataFrame(
        {"sample_key": cells_df["sample_key"], "cell_id": cells_df["cell_id"]}
    )
    obs_out.index = obs_out["sample_key"].values
    adata_out = anndata.AnnData(X=X_out, obs=obs_out, var=adata.var.copy())
    adata_out.write_h5ad(tmp_path / "expression.h5ad")
    cells_df.to_parquet(tmp_path / "manifest.parquet", index=False)

    # ── Required invariants ─────────────────────────────────────────────
    # 1. expr_row equals sample_index
    assert (cells_df["expr_row"].values == cells_df["sample_index"].values).all()
    # 2. manifest cell_id row order == adata_out.obs.cell_id row order
    assert (cells_df["cell_id"].values == adata_out.obs["cell_id"].values).all()
    # 3. gene_row_index → resolved adata row whose cell_id matches manifest cell_id
    resolved_ids = adata.obs["cell_id"].astype(str).values
    for i in range(n_out):
        assert (
            str(cells_df.iloc[i]["cell_id"])
            == resolved_ids[int(cells_df.iloc[i]["gene_row_index"])]
        )
    # 4. X rows of adata_out match adata.X at gene_row_ord
    np.testing.assert_array_equal(
        adata_out.X.toarray(), adata.X[gene_row_ord, :].toarray()
    )
    # 5. Reload manifest + h5ad and assert again from disk
    on_disk_manifest = pd.read_parquet(tmp_path / "manifest.parquet")
    on_disk_h5ad = anndata.read_h5ad(tmp_path / "expression.h5ad")
    assert (
        on_disk_manifest["cell_id"].values
        == on_disk_h5ad.obs["cell_id"].values
    ).all()


# ── Required test 6: filter_report.json end-to-end ────────────────────────
def test_filter_report_written_with_required_fields(tmp_path):
    """Mirror the script's Phase 3c so the report is built from the
    canonical helpers and contains every field downstream tools rely on."""
    cell_ids = ["c0", "c1", "c2", "c3"]
    xy_um = [(40, 40), (60, 60), (-50, 10), (-5, 10)]
    adata_in = _adata(cell_ids)
    gdf_in = _shapes(cell_ids, xy_um)
    flt = _adata(["c0", "c1", "c2"])  # filtered_table only retains c0..c2
    sdata = _sdata(
        tables={"table": adata_in, "filtered_table": flt},
        shapes={"cell_circles": gdf_in},
    )

    bio = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.AUTO,
        table_key_was_default=True,
    )
    assert bio.policy_applied is BiologicalPolicy.STVISUOME_CANONICAL
    adata_after_bio = sdata.tables[bio.table_key_used]
    gdf_full = sdata.shapes[bio.shapes_key_used]
    align = resolve_table_shape_alignment(adata_after_bio, gdf_full)
    adata = adata_after_bio[align.adata_row_indices].copy()
    gdf = gdf_full.iloc[align.shape_row_indices].copy()

    BASE_SIZE = 20
    BASE_HALF = BASE_SIZE / 2.0
    centroids = gdf.geometry
    cx_px = np.array([c.x for c in centroids], dtype=np.float64)
    cy_px = np.array([c.y for c in centroids], dtype=np.float64)
    sx0 = cx_px - BASE_HALF
    sy0 = cy_px - BASE_HALF
    pos_mask = mask_positive_centroid(cx_px, cy_px)
    patch_res = mask_patch_policy(
        sx0, sy0,
        base_size=BASE_SIZE, img_w=100, img_h=100,
        policy=PatchPolicy.STRICT_NO_PADDING, extract_mode="tile_images",
    )
    final_mask = pos_mask & patch_res.valid_mask
    n_valid = int(final_mask.sum())
    n_out = n_valid  # no subsample

    drop_counts = {}
    drop_counts.update({k: int(v) for k, v in bio.drop_counts.items()})
    drop_counts["non_positive_centroid"] = int((~pos_mask).sum())
    drop_counts.update({k: int(v) for k, v in patch_res.drop_counts.items()})
    drop_counts["requested_subsample"] = int(n_valid - n_out)

    report = build_filter_report(
        sample_id="TEST_INT",
        zarr_path="/x/test.zarr",
        output_dir=str(tmp_path),
        image_key="he_image",
        extract_mode="tile_images",
        source_table_key=bio.table_key_used,
        source_shape_key=bio.shapes_key_used,
        biological_policy_requested=bio.policy_requested.value,
        biological_policy_applied=bio.policy_applied.value,
        patch_policy_requested="auto",
        patch_policy_applied="strict_no_padding",
        n_cells_source=int(bio.n_cells_source),
        n_after_biological_filter=int(adata.n_obs),
        n_after_positive_centroid=int(pos_mask.sum()),
        n_after_patch_policy=n_valid,
        n_out=n_out,
        drop_counts_by_reason=drop_counts,
        patch_size=224, target_mpp=0.5, slide_mpp=1.0, base_size=BASE_SIZE,
        seed=42, warnings=list(bio.warnings),
    )
    report_path = write_filter_report(report, tmp_path)
    assert report_path.exists()
    on_disk = json.loads(report_path.read_text())

    # Required fields are present + carry the right values.
    assert on_disk["source_table_key"] == "filtered_table"
    assert on_disk["source_shape_key"] == "cell_circles"
    assert on_disk["biological_policy_applied"] == "stvisuome_canonical"
    assert on_disk["patch_policy_applied"] == "strict_no_padding"
    assert on_disk["n_cells_source"] == 3
    assert on_disk["n_out"] == n_out
    # filtered_table → c0/c1/c2, of which c0 & c1 survive patch policy
    assert n_out == 2
    assert "non_positive_centroid" in on_disk["drop_counts_by_reason"]
    assert "full_oob" in on_disk["drop_counts_by_reason"]
    assert "need_pad" in on_disk["drop_counts_by_reason"]
    assert isinstance(on_disk["warnings"], list)


# ── Required test 7: backward compatibility — default CLI is strict ──────
def _load_extract_sample_module():
    """Import scripts/extract_sample.py as a module without executing main."""
    skill_root = Path(__file__).resolve().parents[1]
    script_path = skill_root / "scripts" / "extract_sample.py"
    spec = importlib.util.spec_from_file_location("_extract_sample", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_defaults_resolve_to_strict_no_padding(monkeypatch):
    """`auto` policies on the CLI must resolve to ``strict_no_padding`` and
    a non-canonical sdata so existing extractions stay bit-identical."""
    mod = _load_extract_sample_module()
    monkeypatch.setattr(sys, "argv",
                        ["extract_sample.py", "--zarr", "X", "--output", "Y"])
    args = mod.parse_args()
    # The new flags exist and default to AUTO.
    assert args.biological_filter_policy == "auto"
    assert args.patch_filter_policy == "auto"
    # AUTO patch policy collapses to STRICT_NO_PADDING regardless of mode.
    assert (
        resolve_patch_policy(PatchPolicy(args.patch_filter_policy), args.extract_mode)
        is PatchPolicy.STRICT_NO_PADDING
    )
    # Other historic defaults are unchanged.
    assert args.table_key == "table"
    assert args.shapes_key == "cell_circles"
    assert args.image_key == "he_image"
    assert args.extract_mode == "tile_images"
    assert args.cell_id_column == "cell_id"
    assert args.nucleus_boundaries_key == "nucleus_boundaries"
    assert args.filtered_table_key == "filtered_table"
    assert args.filter_report_name == "filter_report.json"


def test_existing_synthetic_manifest_schema_still_loads(synthetic_sample):
    """Smoke check: the per-sample fixture used by test_compile/test_bundle
    still loads via the same column set after the script changes — i.e.
    adding `source_table_key`/`source_shape_key` did not turn the existing
    columns into requirements."""
    manifest = pd.read_parquet(synthetic_sample["dir"] / "manifest.parquet")
    required = {"sample_id", "sample_key", "cell_id",
                "shard_path", "tar_offset", "jpg_size",
                "expr_row", "global_idx"}
    assert required.issubset(set(manifest.columns))
    adata = anndata.read_h5ad(synthetic_sample["dir"] / "expression.h5ad")
    assert adata.n_obs == len(manifest)
