from __future__ import annotations

import warnings

import numpy as np


class TissueKeyExistsWarning(UserWarning):
    """Emitted when SOPA creates no new shape key (updated an existing one in-place)."""


def run_tissue_segmentation(
    sdata,
    image_key: str,
    allow_holes: bool = False,
    key_added: str = "tissue",
) -> str:
    """Run SOPA tissue segmentation and return the created (or updated) shape key.

    Parameters
    ----------
    sdata:
        The SpatialData object. Must NOT already contain ``key_added`` — the
        caller (script or agent) is responsible for confirming with the user
        before calling this function when the key exists.
    image_key:
        Key of the H&E image in ``sdata.images``.
    allow_holes:
        Passed through to ``sopa.segmentation.tissue``. Default ``False``.
    key_added:
        Passed through to ``sopa.segmentation.tissue`` as the shape key name.
        Default ``"tissue"``.

    Returns
    -------
    str
        The shape key added or updated by SOPA.

    Warns
    -----
    TissueKeyExistsWarning
        If SOPA creates no new shape key (it updated ``key_added`` in-place).
        The function falls back to ``key_added`` or the first known tissue key.
    """
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for tissue segmentation. Install with: pip install sopa"
        )
    shapes_before = set(sdata.shapes.keys())
    sopa.segmentation.tissue(
        sdata, image_key=image_key, allow_holes=allow_holes, key_added=key_added
    )
    new_keys = set(sdata.shapes.keys()) - shapes_before
    if not new_keys:
        # SOPA updated key_added in-place rather than creating a new key.
        _KNOWN = (key_added, "region_of_interest", "tissue_boundaries", "tissue")
        for candidate in _KNOWN:
            if candidate in sdata.shapes:
                warnings.warn(
                    f"sopa.segmentation.tissue created no new shape key "
                    f"(updated {candidate!r} in-place). Using {candidate!r}.",
                    TissueKeyExistsWarning,
                    stacklevel=2,
                )
                return candidate
        warnings.warn(
            f"sopa.segmentation.tissue created no new shape key and no known "
            f"tissue key found. Shapes: {sorted(sdata.shapes.keys())}.",
            TissueKeyExistsWarning,
            stacklevel=2,
        )
        return key_added
    if len(new_keys) == 1:
        return new_keys.pop()
    # Multiple new keys: prefer key_added, then other known names, else sorted first.
    _KNOWN = (key_added, "region_of_interest", "tissue_boundaries", "tissue")
    for candidate in _KNOWN:
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


__all__ = [
    "TissueKeyExistsWarning",
    "run_tissue_segmentation",
    "filter_by_tissue",
]
