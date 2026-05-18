"""Xenium-vs-HE nucleus overlap filter.

Keeps cells whose Xenium nucleus_boundaries polygon overlaps the nearest
HE nucleus polygon with intersection-over-union >= overlap_threshold.

If he_nucleus_boundaries does not exist in sdata.shapes, calls
sopa.segmentation.cellpose on the HE image to create it.
"""
from __future__ import annotations

import numpy as np


def _ensure_he_nucleus_shapes(
    sdata, image_key: str, he_nucleus_key: str
) -> str:
    if he_nucleus_key in sdata.shapes:
        return he_nucleus_key
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for HE nucleus segmentation. "
            "Install with: pip install sopa"
        )
    print(f"  [nucleus_overlap] {he_nucleus_key!r} not found — "
          "running sopa Cellpose HE nucleus segmentation …")
    sopa.segmentation.cellpose(sdata, image_key=image_key)
    if he_nucleus_key not in sdata.shapes:
        raise RuntimeError(
            f"sopa.segmentation.cellpose ran but {he_nucleus_key!r} was not created. "
            f"Available shapes: {list(sdata.shapes.keys())}. "
            "Pass --he-nucleus-key with the correct key."
        )
    return he_nucleus_key


def _iou(poly_a, poly_b) -> float:
    """Intersection-over-union for two shapely geometries."""
    inter = poly_a.intersection(poly_b).area
    if inter == 0:
        return 0.0
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def filter_by_nucleus_overlap(
    adata,
    xenium_nucleus_shapes,
    he_nucleus_shapes,
    cell_id_column: str = "cell_id",
    overlap_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose Xenium nucleus has
    IoU >= overlap_threshold with the nearest HE nucleus polygon.
    """
    import pandas as pd
    from shapely.strtree import STRtree

    cell_ids = adata.obs[cell_id_column].astype(str).values
    xen_ids = pd.Index(xenium_nucleus_shapes.index).astype(str)
    xen_map = {str(cid): geom for cid, geom in
               zip(xen_ids, xenium_nucleus_shapes.geometry)}

    he_geoms = list(he_nucleus_shapes.geometry)
    tree = STRtree(he_geoms)

    keep_mask = np.zeros(len(cell_ids), dtype=bool)
    scores = np.zeros(len(cell_ids), dtype=float)

    for i, cid in enumerate(cell_ids):
        if cid not in xen_map:
            continue
        xen_poly = xen_map[cid]
        candidates = tree.query(xen_poly)
        if len(candidates) == 0:
            continue
        best_iou = max(_iou(xen_poly, he_geoms[j]) for j in candidates)
        scores[i] = best_iou
        if best_iou >= overlap_threshold:
            keep_mask[i] = True

    n_dropped_no_nucleus = int(sum(1 for cid in cell_ids if cid not in xen_map))
    n_dropped_low_overlap = int((~keep_mask).sum()) - n_dropped_no_nucleus

    drop_counts = {}
    if n_dropped_no_nucleus:
        drop_counts["no_xenium_nucleus"] = n_dropped_no_nucleus
    if n_dropped_low_overlap > 0:
        drop_counts["low_he_overlap"] = n_dropped_low_overlap

    return keep_mask, drop_counts


__all__ = [
    "filter_by_nucleus_overlap",
    "_ensure_he_nucleus_shapes",
]
