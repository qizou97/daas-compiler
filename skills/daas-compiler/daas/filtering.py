"""Filtering helpers for cell patch extraction.

Two layered policies:

* **Layer 1 — table/shape alignment.** Validates that the table and shape
  layer are aligned (exact row-order or by intersection), and selects which
  rows to keep.
* **Layer 2 — patch-validity filtering.** Drops cells whose tile would fall
  off the slide or require padding the producer cannot guarantee.

This module is pure Python over ``numpy``/``pandas``. It never imports
``stvisuome_daas``, never runs tissue segmentation, Cellpose, or any
spatial-omics processing — those are upstream responsibilities of the
producer of the canonical SpatialData zarr.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ── Enums ─────────────────────────────────────────────────────────────────
class PatchPolicy(str, Enum):
    AUTO = "auto"
    STRICT_NO_PADDING = "strict_no_padding"
    STVISUOME_MINIMAL = "stvisuome_minimal"
    STRICT_WITH_PADDING = "strict_with_padding"


# ── Result types ──────────────────────────────────────────────────────────
@dataclass
class AlignmentResult:
    adata_row_indices: np.ndarray
    shape_row_indices: np.ndarray
    alignment_mode: str
    n_table_in: int
    n_shape_in: int
    n_aligned: int


@dataclass
class PatchMaskResult:
    valid_mask: np.ndarray
    full_oob_mask: np.ndarray
    need_pad_mask: np.ndarray
    drop_counts: dict
    policy: str


# ── Core helpers ──────────────────────────────────────────────────────────
def get_table_cell_ids(adata, cell_id_column: str = "cell_id") -> pd.Series:
    """Return canonical cell IDs as strings, preserving row order."""
    if cell_id_column not in adata.obs:
        raise KeyError(
            f"Table is missing required obs[{cell_id_column!r}]. "
            "daas-compiler expects a canonical cell-ID column on the table."
        )
    return adata.obs[cell_id_column].astype(str).reset_index(drop=True)


def resolve_table_shape_alignment(
    adata, gdf, cell_id_column: str = "cell_id"
) -> AlignmentResult:
    """Validate exact alignment between an AnnData table and a shape layer.

    Prefers exact row-order match when
    ``list(gdf.index.astype(str)) == list(adata.obs[cell_id_column].astype(str))``.
    Otherwise aligns by ``cell_id``/``index`` overlap, preserving table row
    order. Raises ``ValueError`` if the overlap is empty.
    """
    table_ids = get_table_cell_ids(adata, cell_id_column=cell_id_column)
    shape_ids = pd.Series(gdf.index.astype(str), name="shape_id").reset_index(drop=True)

    if len(table_ids) == len(shape_ids) and (table_ids.values == shape_ids.values).all():
        n = len(table_ids)
        return AlignmentResult(
            adata_row_indices=np.arange(n, dtype=np.int64),
            shape_row_indices=np.arange(n, dtype=np.int64),
            alignment_mode="exact",
            n_table_in=n,
            n_shape_in=n,
            n_aligned=n,
        )

    table_id_set = set(table_ids.tolist())
    shape_id_set = set(shape_ids.tolist())
    overlap = table_id_set & shape_id_set
    if not overlap:
        raise ValueError(
            "No overlapping cell_id between table and shapes. "
            f"table_n={len(table_ids)} shape_n={len(shape_ids)} "
            f"table_examples={list(table_ids[:3])} "
            f"shape_examples={list(shape_ids[:3])}"
        )

    table_pos = {cid: i for i, cid in enumerate(table_ids)}
    shape_pos = {cid: i for i, cid in enumerate(shape_ids)}
    aligned_ids = [cid for cid in table_ids if cid in shape_id_set]  # table order

    adata_idx = np.fromiter(
        (table_pos[cid] for cid in aligned_ids), dtype=np.int64, count=len(aligned_ids)
    )
    shape_idx = np.fromiter(
        (shape_pos[cid] for cid in aligned_ids), dtype=np.int64, count=len(aligned_ids)
    )

    return AlignmentResult(
        adata_row_indices=adata_idx,
        shape_row_indices=shape_idx,
        alignment_mode="intersection",
        n_table_in=len(table_ids),
        n_shape_in=len(shape_ids),
        n_aligned=len(aligned_ids),
    )


def mask_positive_centroid(cx_px, cy_px) -> np.ndarray:
    """Boolean mask keeping cells with strictly positive centroid pixels."""
    cx = np.asarray(cx_px, dtype=float)
    cy = np.asarray(cy_px, dtype=float)
    return (cx > 0) & (cy > 0)


def mask_patch_policy(
    sx0,
    sy0,
    base_size: int,
    img_w: int,
    img_h: int,
    policy: PatchPolicy,
    extract_mode: str,
) -> PatchMaskResult:
    """Return the patch-validity mask for the chosen policy.

    Always computes ``full_oob`` and ``need_pad`` (matching the historic
    inline logic in ``extract_sample.py``). The kept mask depends on policy:

    * ``STRICT_NO_PADDING`` — drop both ``full_oob`` and ``need_pad`` (default).
    * ``STVISUOME_MINIMAL`` — drop only ``full_oob``; keep boundary-crossing
      tiles. Requires ``extract_mode == "tile_images"`` because the
      ``full_*`` extract modes would silently clip those tiles.
    * ``STRICT_WITH_PADDING`` — reserved; raises ``NotImplementedError``.
    """
    sx0 = np.asarray(sx0, dtype=float)
    sy0 = np.asarray(sy0, dtype=float)

    full_oob = (
        (sx0 + base_size <= 0)
        | (sx0 >= img_w)
        | (sy0 + base_size <= 0)
        | (sy0 >= img_h)
    )
    need_pad = (
        (
            (sx0 < 0)
            | (sx0 + base_size > img_w)
            | (sy0 < 0)
            | (sy0 + base_size > img_h)
        )
        & ~full_oob
    )

    if policy is PatchPolicy.STRICT_WITH_PADDING:
        raise NotImplementedError(
            "patch_filter_policy='strict_with_padding' is reserved but not "
            "yet implemented. Use 'strict_no_padding' or 'stvisuome_minimal'."
        )

    if policy is PatchPolicy.STRICT_NO_PADDING:
        valid = ~full_oob & ~need_pad
        return PatchMaskResult(
            valid_mask=valid,
            full_oob_mask=full_oob,
            need_pad_mask=need_pad,
            drop_counts={
                "full_oob": int(full_oob.sum()),
                "need_pad": int(need_pad.sum()),
            },
            policy=policy.value,
        )

    if policy is PatchPolicy.STVISUOME_MINIMAL:
        if extract_mode != "tile_images":
            raise ValueError(
                "patch_filter_policy='stvisuome_minimal' requires "
                f"extract_mode='tile_images' (got {extract_mode!r}). "
                "Full-image modes silently clip boundary-crossing tiles."
            )
        valid = ~full_oob
        return PatchMaskResult(
            valid_mask=valid,
            full_oob_mask=full_oob,
            need_pad_mask=need_pad,
            drop_counts={"full_oob": int(full_oob.sum())},
            policy=policy.value,
        )

    raise ValueError(f"Unknown patch policy: {policy!r}")


def resolve_patch_policy(policy: PatchPolicy, extract_mode: str) -> PatchPolicy:
    """Resolve ``PatchPolicy.AUTO`` to a concrete policy for ``extract_mode``."""
    if policy is PatchPolicy.AUTO:
        return PatchPolicy.STRICT_NO_PADDING
    return policy


# ── Reporting ─────────────────────────────────────────────────────────────
def build_filter_report(
    *,
    sample_id: str,
    zarr_path: str,
    output_dir: str,
    image_key: str,
    extract_mode: str,
    source_table_key: str,
    source_shape_key: str,
    patch_policy_requested: str,
    patch_policy_applied: str,
    n_cells_source: int,
    n_after_shape_alignment: int,
    n_after_positive_centroid: int,
    n_after_patch_policy: int,
    n_out: int,
    drop_counts_by_reason: dict,
    patch_size: int,
    target_mpp: float,
    slide_mpp: float,
    base_size: int,
    image_width_px: int,
    image_height_px: int,
    seed: int,
    warnings: Sequence[str] = (),
) -> dict:
    """Return a JSON-serializable filter report dict.

    Sequential filtering counters:
      n_cells_source
        → n_after_shape_alignment     (table↔shape alignment)
        → n_after_positive_centroid   (cx_px>0 & cy_px>0)
        → n_after_patch_policy        (Layer 2)
        → n_out                        (after optional --n-sample)
    """
    return {
        "sample_id": str(sample_id),
        "zarr_path": str(zarr_path),
        "output_dir": str(output_dir),
        "image_key": str(image_key),
        "extract_mode": str(extract_mode),
        "source_table_key": str(source_table_key),
        "source_shape_key": str(source_shape_key),
        "patch_policy_requested": str(patch_policy_requested),
        "patch_policy_applied": str(patch_policy_applied),
        "n_cells_source": int(n_cells_source),
        "n_after_shape_alignment": int(n_after_shape_alignment),
        "n_after_positive_centroid": int(n_after_positive_centroid),
        "n_after_patch_policy": int(n_after_patch_policy),
        "n_out": int(n_out),
        "drop_counts_by_reason": {k: int(v) for k, v in drop_counts_by_reason.items()},
        "patch_size": int(patch_size),
        "target_mpp": float(target_mpp),
        "slide_mpp": float(slide_mpp),
        "base_size": int(base_size),
        "image_width_px": int(image_width_px),
        "image_height_px": int(image_height_px),
        "seed": int(seed),
        "warnings": list(warnings),
    }


def write_filter_report(
    report: dict, output_dir, name: str = "filter_report.json"
) -> Path:
    """Write ``report`` to ``output_dir/name`` and return the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path


__all__ = [
    "PatchPolicy",
    "AlignmentResult",
    "PatchMaskResult",
    "get_table_cell_ids",
    "resolve_table_shape_alignment",
    "mask_positive_centroid",
    "mask_patch_policy",
    "resolve_patch_policy",
    "build_filter_report",
    "write_filter_report",
]
