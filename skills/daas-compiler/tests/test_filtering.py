"""Unit tests for daas.filtering (patch policy + alignment + reporting)."""
import json
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from daas.filtering import (
    PatchPolicy,
    AlignmentResult,
    PatchMaskResult,
    build_filter_report,
    get_table_cell_ids,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)


# ── helpers ──────────────────────────────────────────────────────────────
def _adata(cell_ids, n_genes: int = 3) -> anndata.AnnData:
    n = len(cell_ids)
    X = csr_matrix(np.arange(n * n_genes, dtype=np.float32).reshape(n, n_genes))
    obs = pd.DataFrame({"cell_id": list(cell_ids)})
    obs.index = [f"{i:07d}" for i in range(n)]
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _gdf(cell_ids) -> pd.DataFrame:
    """Stand-in for a GeoDataFrame — only `.index` is used by the helpers."""
    return pd.DataFrame(
        {"x": np.arange(len(cell_ids), dtype=float)},
        index=pd.Index(list(cell_ids), name="cell_id"),
    )


# ── get_table_cell_ids ───────────────────────────────────────────────────
def test_get_table_cell_ids_preserves_order_and_str():
    adata = _adata(["c0", "c1", "c2"])
    ids = get_table_cell_ids(adata)
    assert list(ids) == ["c0", "c1", "c2"]


def test_get_table_cell_ids_missing_column_raises():
    adata = _adata(["c0", "c1"])
    del adata.obs["cell_id"]
    with pytest.raises(KeyError, match="cell_id"):
        get_table_cell_ids(adata)


# ── resolve_table_shape_alignment ────────────────────────────────────────
def test_alignment_exact_when_orders_match():
    adata = _adata(["c0", "c1", "c2"])
    gdf = _gdf(["c0", "c1", "c2"])
    align = resolve_table_shape_alignment(adata, gdf)
    assert align.alignment_mode == "exact"
    assert align.n_aligned == 3
    np.testing.assert_array_equal(align.adata_row_indices, [0, 1, 2])
    np.testing.assert_array_equal(align.shape_row_indices, [0, 1, 2])


def test_alignment_intersection_when_orders_differ():
    adata = _adata(["c0", "c1", "c2", "c3"])
    gdf = _gdf(["c3", "c1", "c2"])
    align = resolve_table_shape_alignment(adata, gdf)
    assert align.alignment_mode == "intersection"
    assert align.n_aligned == 3
    np.testing.assert_array_equal(align.adata_row_indices, [1, 2, 3])
    np.testing.assert_array_equal(align.shape_row_indices, [1, 2, 0])


def test_alignment_no_overlap_raises():
    adata = _adata(["c0", "c1"])
    gdf = _gdf(["x0", "x1"])
    with pytest.raises(ValueError, match="No overlapping"):
        resolve_table_shape_alignment(adata, gdf)


# ── mask_positive_centroid ───────────────────────────────────────────────
def test_mask_positive_centroid():
    cx = np.array([10.0, -1.0, 0.0, 5.0])
    cy = np.array([10.0, 5.0, 5.0, -1.0])
    mask = mask_positive_centroid(cx, cy)
    np.testing.assert_array_equal(mask, [True, False, False, False])


def test_mask_positive_centroid_full_combination_matrix():
    """All sign combinations: only (+, +) retained."""
    cx = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    cy = np.array([1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
    mask = mask_positive_centroid(cx, cy)
    np.testing.assert_array_equal(
        mask, [True, False, False, False, False, False, False, False, False]
    )


# ── mask_patch_policy ────────────────────────────────────────────────────
def _patch_inputs():
    # img 100x100, base_size 20
    sx0 = np.array([10.0, -5.0, 95.0, -25.0, 90.0])
    sy0 = np.array([10.0, 10.0, 10.0, 10.0, 90.0])
    return sx0, sy0, 20, 100, 100


def test_patch_policy_strict_no_padding():
    sx0, sy0, bs, w, h = _patch_inputs()
    res = mask_patch_policy(
        sx0, sy0, bs, w, h, PatchPolicy.STRICT_NO_PADDING, "tile_images"
    )
    np.testing.assert_array_equal(res.valid_mask, [True, False, False, False, False])
    assert res.drop_counts["full_oob"] >= 1
    assert res.drop_counts["need_pad"] >= 1
    assert res.policy == "strict_no_padding"


def test_patch_policy_strict_no_padding_minimal_three_cells():
    """Spec case: one fully-inside, one full_oob, one need_pad → only inside kept."""
    # img 100x100, base_size 20:
    # cell 0: sx0=40, sy0=40 → fully inside
    # cell 1: sx0=-50, sy0=10 → full_oob (sx0+20=-30 <= 0)
    # cell 2: sx0=-5,  sy0=10 → need_pad (sx0<0 but sx0+20=15>0)
    sx0 = np.array([40.0, -50.0, -5.0])
    sy0 = np.array([40.0, 10.0, 10.0])
    res = mask_patch_policy(
        sx0, sy0, 20, 100, 100, PatchPolicy.STRICT_NO_PADDING, "tile_images"
    )
    np.testing.assert_array_equal(res.valid_mask, [True, False, False])
    np.testing.assert_array_equal(res.full_oob_mask, [False, True, False])
    np.testing.assert_array_equal(res.need_pad_mask, [False, False, True])
    assert res.drop_counts == {"full_oob": 1, "need_pad": 1}


def test_patch_policy_stvisuome_minimal_keeps_need_pad():
    sx0, sy0, bs, w, h = _patch_inputs()
    res = mask_patch_policy(
        sx0, sy0, bs, w, h, PatchPolicy.STVISUOME_MINIMAL, "tile_images"
    )
    np.testing.assert_array_equal(res.valid_mask, [True, True, True, False, True])
    assert "need_pad" not in res.drop_counts


def test_patch_policy_stvisuome_minimal_rejects_full_modes():
    sx0, sy0, bs, w, h = _patch_inputs()
    for mode in ("full_scale0", "full_ops_level"):
        with pytest.raises(ValueError, match="stvisuome_minimal"):
            mask_patch_policy(
                sx0, sy0, bs, w, h, PatchPolicy.STVISUOME_MINIMAL, mode
            )


def test_patch_policy_strict_with_padding_not_implemented():
    sx0, sy0, bs, w, h = _patch_inputs()
    with pytest.raises(NotImplementedError):
        mask_patch_policy(
            sx0, sy0, bs, w, h, PatchPolicy.STRICT_WITH_PADDING, "tile_images"
        )


def test_resolve_patch_policy_auto_to_strict():
    assert (
        resolve_patch_policy(PatchPolicy.AUTO, "tile_images")
        is PatchPolicy.STRICT_NO_PADDING
    )
    assert (
        resolve_patch_policy(PatchPolicy.STVISUOME_MINIMAL, "tile_images")
        is PatchPolicy.STVISUOME_MINIMAL
    )


# ── filter report ────────────────────────────────────────────────────────
EXPECTED_REPORT_KEYS = {
    "sample_id", "zarr_path", "output_dir", "image_key", "extract_mode",
    "source_table_key", "source_shape_key",
    "patch_policy_requested", "patch_policy_applied",
    "n_cells_source",
    "n_after_shape_alignment",
    "n_after_positive_centroid", "n_after_patch_policy", "n_out",
    "drop_counts_by_reason", "patch_size", "target_mpp",
    "slide_mpp", "base_size",
    "image_width_px", "image_height_px",
    "seed", "warnings",
}


def test_build_filter_report_serializes_to_disk(tmp_path):
    report = build_filter_report(
        sample_id="A_001",
        zarr_path="/x/A_001.zarr",
        output_dir=str(tmp_path),
        image_key="he_image",
        extract_mode="tile_images",
        source_table_key="filtered_table",
        source_shape_key="cell_circles",
        patch_policy_requested="auto",
        patch_policy_applied="strict_no_padding",
        n_cells_source=100,
        n_after_shape_alignment=100,
        n_after_positive_centroid=98,
        n_after_patch_policy=95,
        n_out=95,
        drop_counts_by_reason={"full_oob": 3, "need_pad": 2},
        patch_size=224,
        target_mpp=0.5,
        slide_mpp=0.2125,
        base_size=527,
        image_width_px=38912,
        image_height_px=26624,
        seed=42,
    )
    path = write_filter_report(report, tmp_path)
    assert path.name == "filter_report.json"
    loaded = json.loads(path.read_text())
    assert set(loaded) == EXPECTED_REPORT_KEYS
    assert loaded["sample_id"] == "A_001"
    assert loaded["drop_counts_by_reason"]["full_oob"] == 3


def test_write_filter_report_custom_name(tmp_path):
    report = build_filter_report(
        sample_id="A_001",
        zarr_path="/data/A_001.zarr",
        output_dir="/data/out",
        image_key="he_image",
        extract_mode="tile_images",
        source_table_key="table",
        source_shape_key="cell_circles",
        patch_policy_requested="auto",
        patch_policy_applied="strict_no_padding",
        n_cells_source=100,
        n_after_shape_alignment=100,
        n_after_positive_centroid=98,
        n_after_patch_policy=95,
        n_out=95,
        drop_counts_by_reason={"full_oob": 3, "need_pad": 2},
        patch_size=224,
        target_mpp=0.5,
        slide_mpp=0.2125,
        base_size=527,
        image_width_px=38912,
        image_height_px=26624,
        seed=42,
    )
    path = write_filter_report(report, tmp_path, name="alt.json")
    assert path.name == "alt.json"
    assert json.loads(path.read_text())["n_out"] == 95
