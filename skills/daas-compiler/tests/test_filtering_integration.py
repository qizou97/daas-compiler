# skills/daas-compiler/tests/test_filtering_integration.py
"""Integration tests for alignment + patch filtering (no biological policy)."""
import json
from types import SimpleNamespace

import anndata
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from daas.cli_args import (
    build_extract_sample_parser,
    parse_extract_sample_args,
    validate_policy_combination,
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
    return gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in xy_um]},
        index=pd.Index([str(c) for c in cell_ids], name="cell_id"),
    )


def test_alignment_preserved_end_to_end(tmp_path):
    """Direct table load → alignment → patch filter preserves row order."""
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

    # Direct load — no biological policy
    align = resolve_table_shape_alignment(adata_in, gdf_in)
    assert align.alignment_mode == "exact"
    adata = adata_in[align.adata_row_indices].copy()
    gdf = gdf_in.iloc[align.shape_row_indices].copy()

    SCALE_SHAPE = 1.0
    BASE_SIZE = 20
    BASE_HALF = BASE_SIZE / 2.0
    IMG_W = IMG_H = 100

    cx_um = np.array([c.x for c in gdf.geometry], dtype=np.float64)
    cy_um = np.array([c.y for c in gdf.geometry], dtype=np.float64)
    cx_px = cx_um * SCALE_SHAPE
    cy_px = cy_um * SCALE_SHAPE
    sx0 = cx_px - BASE_HALF
    sy0 = cy_px - BASE_HALF

    patch_policy = resolve_patch_policy(PatchPolicy.AUTO, "tile_images")
    assert patch_policy is PatchPolicy.STRICT_NO_PADDING
    pos_mask = mask_positive_centroid(cx_px, cy_px)
    patch_res = mask_patch_policy(
        sx0, sy0,
        base_size=BASE_SIZE, img_w=IMG_W, img_h=IMG_H,
        policy=patch_policy, extract_mode="tile_images",
    )
    final_mask = pos_mask & patch_res.valid_mask
    np.testing.assert_array_equal(
        final_mask, [True, True, False, False, False, False]
    )

    valid_indices = np.where(final_mask)[0]
    n_out = len(valid_indices)

    # Verify alignment invariants hold
    assert n_out == 2
    survived_ids = [gdf.index[i] for i in valid_indices]
    assert survived_ids == ["c0", "c1"]
    # gene_row_index points to the right cell_id in adata
    for local_i, orig_i in enumerate(valid_indices):
        assert adata.obs.iloc[orig_i]["cell_id"] == survived_ids[local_i]


def test_filter_report_written_with_correct_fields(tmp_path):
    report_dict = build_filter_report(
        sample_id="A_001",
        zarr_path="/data/A_001.zarr",
        output_dir=str(tmp_path),
        image_key="he_image",
        extract_mode="tile_images",
        source_table_key="table_tissue_nucleus",
        source_shape_key="cell_circles",
        patch_policy_requested="auto",
        patch_policy_applied="strict_no_padding",
        n_cells_source=200,
        n_after_shape_alignment=198,
        n_after_positive_centroid=196,
        n_after_patch_policy=190,
        n_out=100,
        drop_counts_by_reason={"full_oob": 6, "need_pad": 4,
                               "requested_subsample": 90},
        patch_size=224,
        target_mpp=0.5,
        slide_mpp=0.2125,
        base_size=527,
        image_width_px=38912,
        image_height_px=26624,
        seed=42,
    )
    assert report_dict["source_table_key"] == "table_tissue_nucleus"
    assert report_dict["n_cells_source"] == 200
    assert report_dict["n_out"] == 100
    # Biological policy fields must NOT be present
    assert "biological_policy_requested" not in report_dict
    assert "biological_policy_applied" not in report_dict
    assert "n_after_biological_filter" not in report_dict

    path = write_filter_report(report_dict, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["n_cells_source"] == 200


def test_cli_parse_no_biological_policy_args():
    """After refactor, --biological-filter-policy no longer exists."""
    import sys
    parser = build_extract_sample_parser()
    # Should not have biological-filter-policy option
    option_strings = [
        action.option_strings
        for action in parser._actions
    ]
    flat = [s for group in option_strings for s in group]
    assert "--biological-filter-policy" not in flat
    assert "--filtered-table-key" not in flat
    assert "--nucleus-boundaries-key" not in flat


def test_cli_parse_table_key_respected():
    args = parse_extract_sample_args(
        ["--zarr", "/data/A.zarr", "--output", "/data/out",
         "--table-key", "table_tissue_nucleus"]
    )
    assert args.table_key == "table_tissue_nucleus"
