"""Xenium-vs-HE nucleus overlap filter.

Keeps cells whose Xenium nucleus_boundaries polygon overlaps the nearest
HE nucleus polygon with intersection-over-union >= overlap_threshold.

Requires sopa to be installed. Always runs sopa.segmentation.cellpose
to produce HE nucleus boundaries — no conditional skip.
"""
from __future__ import annotations

import numpy as np


def run_he_nucleus_segmentation(sdata, image_key: str) -> str:
    """Always run SOPA Cellpose HE nucleus segmentation and return the created key.

    Diffs sdata.shapes before/after to discover the new key — never hardcodes
    a key name. Raises RuntimeError if Cellpose creates no new shape key.
    """
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for HE nucleus segmentation. "
            "Install with: pip install sopa"
        )
    shapes_before = set(sdata.shapes.keys())
    sopa.segmentation.cellpose(sdata, image_key=image_key)
    new_keys = set(sdata.shapes.keys()) - shapes_before
    if not new_keys:
        raise RuntimeError(
            f"sopa.segmentation.cellpose ran but created no new shape key. "
            f"Shapes before: {sorted(shapes_before)}. "
            f"Shapes after: {sorted(sdata.shapes.keys())}."
        )
    if len(new_keys) == 1:
        return new_keys.pop()
    for candidate in ("he_nucleus_boundaries", "nucleus_boundaries"):
        if candidate in new_keys:
            return candidate
    return sorted(new_keys)[0]


def _iou(poly_a, poly_b) -> float:
    inter = poly_a.intersection(poly_b).area
    if inter == 0:
        return 0.0
    union = poly_a.union(poly_b).area
    return inter / union


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

    for i, cid in enumerate(cell_ids):
        if cid not in xen_map:
            continue
        xen_poly = xen_map[cid]
        candidates = tree.query(xen_poly)
        if len(candidates) == 0:
            continue
        best_iou = max(_iou(xen_poly, he_geoms[j]) for j in candidates)
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
    "run_he_nucleus_segmentation",
    "filter_by_nucleus_overlap",
]
