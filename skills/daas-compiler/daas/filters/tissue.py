from __future__ import annotations

import numpy as np


def run_tissue_segmentation(
    sdata,
    image_key: str,
    allow_holes: bool = False,
    key_added: str | None = None,
) -> str:
    """Run SOPA tissue segmentation and return the shape key.

    Skips segmentation if key_added is given and already present in sdata.shapes.
    Diffs sdata.shapes before/after to discover the new key — never hardcodes
    a key name. Raises RuntimeError if SOPA creates no new shape key.
    """
    if key_added is not None and key_added in sdata.shapes:
        return key_added

    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for tissue segmentation. Install with: pip install sopa"
        )
    shapes_before = set(sdata.shapes.keys())
    kwargs: dict = {"image_key": image_key, "allow_holes": allow_holes}
    if key_added is not None:
        kwargs["key_added"] = key_added
    sopa.segmentation.tissue(sdata, **kwargs)
    new_keys = set(sdata.shapes.keys()) - shapes_before
    if not new_keys:
        raise RuntimeError(
            f"sopa.segmentation.tissue ran but created no new shape key. "
            f"Shapes before: {sorted(shapes_before)}. "
            f"Shapes after: {sorted(sdata.shapes.keys())}."
        )
    if len(new_keys) == 1:
        return new_keys.pop()
    # Multiple new keys: prefer known tissue key names, else take sorted first.
    for candidate in ("region_of_interest", "tissue_boundaries", "tissue"):
        if candidate in new_keys:
            return candidate
    return sorted(new_keys)[0]


def filter_by_tissue(
    adata,
    cell_shapes,
    tissue_shapes,
    cell_id_column: str = "cell_id",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose centroid is inside
    any tissue polygon.

    cell_shapes: GeoDataFrame with Point or Polygon geometries
    tissue_shapes: GeoDataFrame with Polygon geometries (tissue regions)
    """
    cell_ids = adata.obs[cell_id_column].astype(str)
    cell_shapes_aligned = cell_shapes.loc[
        cell_shapes.index.astype(str).isin(set(cell_ids.tolist()))
    ]
    centroids = cell_shapes_aligned.geometry.centroid
    tissue_union = tissue_shapes.geometry.union_all()
    inside = centroids.within(tissue_union)
    inside_series = inside.reindex(cell_ids.values, fill_value=False)
    keep_mask = inside_series.to_numpy(dtype=bool)
    n_dropped = int((~keep_mask).sum())
    return keep_mask, {"outside_tissue": n_dropped}


__all__ = ["run_tissue_segmentation", "filter_by_tissue"]
