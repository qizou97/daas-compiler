from __future__ import annotations

import numpy as np


def _ensure_tissue_shapes(sdata, image_key: str, tissue_key: str) -> str:
    """Return tissue_key if present, else run SOPA and return the created key."""
    if tissue_key in sdata.shapes:
        return tissue_key
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for tissue segmentation. Install with: pip install sopa"
        )
    print(f"  [tissue] {tissue_key!r} not found — running sopa tissue segmentation …")
    sopa.segmentation.tissue(sdata, image_key=image_key)
    if tissue_key not in sdata.shapes:
        raise RuntimeError(
            f"sopa.segmentation.tissue ran but {tissue_key!r} was not created. "
            f"Available shapes: {list(sdata.shapes.keys())}. "
            "Pass --tissue-key with the correct key name."
        )
    return tissue_key


def filter_by_tissue(
    adata,
    cell_shapes,
    tissue_shapes,
    cell_id_column: str = "cell_id",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose centroid is inside
    any tissue polygon.

    cell_shapes: GeoDataFrame with Point or Polygon geometries (cell centroids)
    tissue_shapes: GeoDataFrame with Polygon geometries (tissue regions)
    """
    import geopandas as gpd

    cell_ids = adata.obs[cell_id_column].astype(str)
    # Align cell_shapes to adata row order by cell_id
    cell_shapes_aligned = cell_shapes.loc[
        cell_shapes.index.astype(str).isin(set(cell_ids.tolist()))
    ]
    # Use centroids for point-in-polygon test
    centroids = cell_shapes_aligned.geometry.centroid

    tissue_union = tissue_shapes.geometry.union_all()

    inside = centroids.within(tissue_union)
    # Re-index to adata row order
    inside_series = inside.reindex(cell_ids.values, fill_value=False)
    keep_mask = inside_series.to_numpy(dtype=bool)

    n_dropped = int((~keep_mask).sum())
    return keep_mask, {"outside_tissue": n_dropped}


__all__ = ["filter_by_tissue", "_ensure_tissue_shapes"]
