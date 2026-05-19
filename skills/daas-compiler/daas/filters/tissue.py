from __future__ import annotations

import warnings

import numpy as np
import shapely.affinity as _sa


class TissueKeyExistsWarning(UserWarning):
    """Emitted when SOPA creates no new shape key (updated an existing one in-place)."""


def run_tissue_segmentation(
    sdata,
    image_key: str,
    allow_holes: bool = False,
    tissue_key: str = "tissue",
) -> str:
    """Run SOPA tissue segmentation and return the created (or updated) shape key.

    Parameters
    ----------
    sdata:
        The SpatialData object. Must NOT already contain ``tissue_key`` — the
        caller (script or agent) is responsible for confirming with the user
        before calling this function when the key exists.
    image_key:
        Key of the H&E image in ``sdata.images``.
    allow_holes:
        Passed through to ``sopa.segmentation.tissue``. Default ``False``.
    tissue_key:
        Passed through to ``sopa.segmentation.tissue`` as the shape key name.
        Default ``"tissue"``.

    Returns
    -------
    str
        The shape key added or updated by SOPA.

    Warns
    -----
    TissueKeyExistsWarning
        If SOPA creates no new shape key (it updated ``tissue_key`` in-place).
        The function falls back to ``tissue_key`` or the first known tissue key.
    """
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for tissue segmentation. Install with: pip install sopa"
        )
    shapes_before = set(sdata.shapes.keys())
    sopa.segmentation.tissue(
        sdata, image_key=image_key, allow_holes=allow_holes, key_added=tissue_key
    )
    new_keys = set(sdata.shapes.keys()) - shapes_before
    if not new_keys:
        # SOPA updated tissue_key in-place rather than creating a new key.
        warnings.warn(
            f"sopa.segmentation.tissue created no new shape key "
            f"(updated {tissue_key!r} in-place). Using {tissue_key!r}.",
            TissueKeyExistsWarning,
            stacklevel=2,
        )
        return tissue_key
    if len(new_keys) == 1:
        return new_keys.pop()
    # Multiple new keys: prefer tissue_key if present, else sorted first.
    return tissue_key if tissue_key in new_keys else sorted(new_keys)[0]


def _gdf_to_coordinate_system(gdf, coordinate_system: str = "global"):
    """Return a copy of *gdf* with geometries projected to *coordinate_system*.

    Uses the SpatialData transformation stored on *gdf* (if any).  Falls back
    to the identity so the function is safe when shapes carry no registered
    transform.
    """
    try:
        from spatialdata.transformations import get_transformation
        t = get_transformation(gdf, to_coordinate_system=coordinate_system)
        mat = t.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        a, b, xoff = mat[0, 0], mat[0, 1], mat[0, 2]
        d, e, yoff = mat[1, 0], mat[1, 1], mat[1, 2]
        result = gdf.copy()
        result.geometry = gdf.geometry.apply(
            lambda geom: _sa.affine_transform(geom, [a, b, d, e, xoff, yoff])
        )
        return result
    except Exception:
        return gdf


def filter_by_tissue(
    adata,
    cell_shapes,
    tissue_shapes,
    cell_id_column: str = "cell_id",
    coordinate_system: str = "global",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose centroid is inside
    any tissue polygon.

    Both *cell_shapes* and *tissue_shapes* are projected to *coordinate_system*
    (default ``"global"``) before the spatial join, so SOPA tissue ROIs (which
    live in image pixel space) and Xenium cell shapes (in a different pixel
    space) are compared in the same coordinate system.

    cell_shapes: GeoDataFrame with Point or Polygon geometries
    tissue_shapes: GeoDataFrame with Polygon geometries (tissue regions)
    """
    cell_shapes_cs = _gdf_to_coordinate_system(cell_shapes, coordinate_system)
    tissue_shapes_cs = _gdf_to_coordinate_system(tissue_shapes, coordinate_system)

    cell_ids = adata.obs[cell_id_column].astype(str)
    cell_shapes_aligned = cell_shapes_cs.loc[
        cell_shapes_cs.index.astype(str).isin(set(cell_ids.tolist()))
    ]
    centroids = cell_shapes_aligned.geometry.centroid
    tissue_union = tissue_shapes_cs.geometry.union_all()
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
