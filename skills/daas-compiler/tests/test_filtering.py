"""Unit tests for daas.filtering."""
import json
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from daas.filtering import (
    BiologicalPolicy,
    PatchPolicy,
    build_filter_report,
    get_table_cell_ids,
    mask_by_nucleus_boundaries,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_biological_policy,
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


def _sdata(tables: dict, shapes: dict):
    return SimpleNamespace(tables=tables, shapes=shapes)


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


# ── mask_by_nucleus_boundaries ───────────────────────────────────────────
def test_mask_by_nucleus_boundaries_abc_keeps_bc():
    """Spec case: table=[a,b,c], nucleus=[b,c] → retain [b,c]."""
    nb = _gdf(["b", "c"])
    mask = mask_by_nucleus_boundaries(["a", "b", "c"], nb)
    np.testing.assert_array_equal(mask, [False, True, True])
    retained = [cid for cid, keep in zip(["a", "b", "c"], mask) if keep]
    assert retained == ["b", "c"]


def test_mask_by_nucleus_boundaries_basic():
    nb = _gdf(["c1", "c3"])
    mask = mask_by_nucleus_boundaries(["c0", "c1", "c2", "c3"], nb)
    np.testing.assert_array_equal(mask, [False, True, False, True])


def test_mask_by_nucleus_boundaries_stringifies():
    nb = _gdf([1, 2, 3])
    mask = mask_by_nucleus_boundaries(["1", "2", "4"], nb)
    np.testing.assert_array_equal(mask, [True, True, False])


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


# ── resolve_biological_policy ────────────────────────────────────────────
def test_biological_auto_picks_canonical_when_filtered_present():
    raw = _adata(["c0", "c1", "c2"])
    flt = _adata(["c0", "c2"])
    sdata = _sdata(
        tables={"table": raw, "filtered_table": flt},
        shapes={"cell_circles": _gdf(["c0", "c1", "c2"])},
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.AUTO,
        table_key_was_default=True,
    )
    assert res.policy_applied is BiologicalPolicy.STVISUOME_CANONICAL
    assert res.table_key_used == "filtered_table"
    assert res.n_cells_source == 2
    assert res.keep_table_mask is None


def test_biological_auto_falls_back_to_none_when_no_filtered():
    raw = _adata(["c0", "c1"])
    sdata = _sdata(
        tables={"table": raw}, shapes={"cell_circles": _gdf(["c0", "c1"])}
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.AUTO,
        table_key_was_default=True,
    )
    assert res.policy_applied is BiologicalPolicy.NONE
    assert res.warnings


def test_biological_auto_does_not_override_explicit_table_key():
    raw = _adata(["c0", "c1"])
    flt = _adata(["c0"])
    sdata = _sdata(
        tables={"table": raw, "filtered_table": flt, "custom": raw},
        shapes={"cell_circles": _gdf(["c0", "c1"])},
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="custom",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.AUTO,
        table_key_was_default=False,
    )
    assert res.policy_applied is BiologicalPolicy.NONE
    assert any("auto" in w for w in res.warnings)


def test_biological_canonical_prefers_filtered_cell_circles():
    """When --shapes-key is default, prefer filtered_cell_circles if present."""
    raw = _adata(["c0", "c1"])
    flt = _adata(["c0", "c1"])
    sdata = _sdata(
        tables={"table": raw, "filtered_table": flt},
        shapes={
            "cell_circles": _gdf(["c0", "c1"]),
            "filtered_cell_circles": _gdf(["c0", "c1"]),
            "filtered_cell_boundaries": _gdf(["c0", "c1"]),
        },
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.STVISUOME_CANONICAL,
        table_key_was_default=True,
        shapes_key_was_default=True,
    )
    assert res.shapes_key_used == "filtered_cell_circles"


def test_biological_canonical_falls_back_to_filtered_cell_boundaries():
    raw = _adata(["c0", "c1"])
    flt = _adata(["c0", "c1"])
    sdata = _sdata(
        tables={"table": raw, "filtered_table": flt},
        shapes={
            "cell_circles": _gdf(["c0", "c1"]),
            "filtered_cell_boundaries": _gdf(["c0", "c1"]),
        },
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.STVISUOME_CANONICAL,
        table_key_was_default=True,
        shapes_key_was_default=True,
    )
    assert res.shapes_key_used == "filtered_cell_boundaries"


def test_biological_canonical_falls_back_to_default_shapes_key():
    raw = _adata(["c0", "c1"])
    flt = _adata(["c0", "c1"])
    sdata = _sdata(
        tables={"table": raw, "filtered_table": flt},
        shapes={"cell_circles": _gdf(["c0", "c1"])},
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.STVISUOME_CANONICAL,
        table_key_was_default=True,
        shapes_key_was_default=True,
    )
    assert res.shapes_key_used == "cell_circles"


def test_biological_canonical_preserves_explicit_shapes_key_with_warning():
    raw = _adata(["c0"])
    flt = _adata(["c0"])
    sdata = _sdata(
        tables={"table": raw, "filtered_table": flt},
        shapes={
            "cell_circles": _gdf(["c0"]),
            "filtered_cell_circles": _gdf(["c0"]),
            "custom_shapes": _gdf(["c0"]),
        },
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="custom_shapes",
        policy=BiologicalPolicy.STVISUOME_CANONICAL,
        table_key_was_default=True,
        shapes_key_was_default=False,
    )
    assert res.shapes_key_used == "custom_shapes"
    assert any("filtered_cell_circles" in w for w in res.warnings)


def test_biological_canonical_raises_when_no_filtered_table():
    raw = _adata(["c0"])
    sdata = _sdata(tables={"table": raw}, shapes={"cell_circles": _gdf(["c0"])})
    with pytest.raises(KeyError, match="filtered_table"):
        resolve_biological_policy(
            sdata=sdata,
            table_key="table",
            shapes_key="cell_circles",
            policy=BiologicalPolicy.STVISUOME_CANONICAL,
            table_key_was_default=True,
        )


def test_biological_nucleus_boundary_filters_by_nucleus_ids():
    raw = _adata(["c0", "c1", "c2", "c3"])
    nucleus = _gdf(["c1", "c3"])
    sdata = _sdata(
        tables={"table": raw},
        shapes={
            "cell_circles": _gdf(["c0", "c1", "c2", "c3"]),
            "nucleus_boundaries": nucleus,
        },
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.STVISUOME_NUCLEUS_BOUNDARY,
        table_key_was_default=True,
    )
    assert res.policy_applied is BiologicalPolicy.STVISUOME_NUCLEUS_BOUNDARY
    assert res.n_after_biological_filter == 2
    np.testing.assert_array_equal(res.keep_table_mask, [False, True, False, True])
    assert res.drop_counts["missing_nucleus_boundary"] == 2


def test_biological_nucleus_boundary_raises_when_no_nucleus_layer():
    raw = _adata(["c0"])
    sdata = _sdata(tables={"table": raw}, shapes={"cell_circles": _gdf(["c0"])})
    with pytest.raises(KeyError, match="nucleus_boundaries"):
        resolve_biological_policy(
            sdata=sdata,
            table_key="table",
            shapes_key="cell_circles",
            policy=BiologicalPolicy.STVISUOME_NUCLEUS_BOUNDARY,
            table_key_was_default=True,
        )


def test_biological_nucleus_boundary_raises_when_all_dropped():
    raw = _adata(["c0", "c1"])
    sdata = _sdata(
        tables={"table": raw},
        shapes={"nucleus_boundaries": _gdf(["x0"])},
    )
    with pytest.raises(ValueError, match="removed all cells"):
        resolve_biological_policy(
            sdata=sdata,
            table_key="table",
            shapes_key="cell_circles",
            policy=BiologicalPolicy.STVISUOME_NUCLEUS_BOUNDARY,
            table_key_was_default=True,
        )


def test_biological_none_passes_through_with_warning():
    raw = _adata(["c0", "c1"])
    sdata = _sdata(
        tables={"table": raw}, shapes={"cell_circles": _gdf(["c0", "c1"])}
    )
    res = resolve_biological_policy(
        sdata=sdata,
        table_key="table",
        shapes_key="cell_circles",
        policy=BiologicalPolicy.NONE,
        table_key_was_default=True,
    )
    assert res.policy_applied is BiologicalPolicy.NONE
    assert res.keep_table_mask is None
    assert any("no biological filtering" in w for w in res.warnings)


# ── filter report ────────────────────────────────────────────────────────
EXPECTED_REPORT_KEYS = {
    "sample_id", "zarr_path", "output_dir", "image_key", "extract_mode",
    "source_table_key", "source_shape_key",
    "biological_policy_requested", "biological_policy_applied",
    "patch_policy_requested", "patch_policy_applied",
    "n_cells_source", "n_after_biological_filter",
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
        biological_policy_requested="auto",
        biological_policy_applied="stvisuome_canonical",
        patch_policy_requested="auto",
        patch_policy_applied="strict_no_padding",
        n_cells_source=100,
        n_after_biological_filter=80,
        n_after_shape_alignment=80,
        n_after_positive_centroid=78,
        n_after_patch_policy=70,
        n_out=50,
        drop_counts_by_reason={
            "missing_nucleus_boundary": 20,
            "non_positive_centroid": 2,
            "full_oob": 5,
            "need_pad": 3,
            "requested_subsample": 20,
        },
        patch_size=224,
        target_mpp=0.5,
        slide_mpp=0.2125,
        base_size=527,
        image_width_px=40000,
        image_height_px=30000,
        seed=42,
        warnings=["example"],
    )
    path = write_filter_report(report, tmp_path)
    assert path.name == "filter_report.json"
    loaded = json.loads(path.read_text())
    assert set(loaded) == EXPECTED_REPORT_KEYS
    assert loaded["sample_id"] == "A_001"
    assert loaded["biological_policy_applied"] == "stvisuome_canonical"
    assert loaded["drop_counts_by_reason"]["missing_nucleus_boundary"] == 20
    assert loaded["warnings"] == ["example"]


def test_write_filter_report_custom_name(tmp_path):
    report = build_filter_report(
        sample_id="x", zarr_path="x", output_dir=str(tmp_path),
        image_key="x", extract_mode="x",
        source_table_key="x", source_shape_key="x",
        biological_policy_requested="none",
        biological_policy_applied="none",
        patch_policy_requested="strict_no_padding",
        patch_policy_applied="strict_no_padding",
        n_cells_source=0, n_after_biological_filter=0,
        n_after_shape_alignment=0,
        n_after_positive_centroid=0, n_after_patch_policy=0, n_out=0,
        drop_counts_by_reason={},
        patch_size=1, target_mpp=0.0, slide_mpp=0.0, base_size=0,
        image_width_px=0, image_height_px=0,
        seed=0,
    )
    path = write_filter_report(report, tmp_path, name="alt.json")
    assert path.name == "alt.json"
    assert json.loads(path.read_text())["n_out"] == 0
