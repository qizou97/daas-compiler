from __future__ import annotations

import numpy as np
import pandas as pd


def filter_by_nucleus_presence(
    adata,
    nucleus_boundaries,
    cell_id_column: str = "cell_id",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose cell_id appears in
    nucleus_boundaries.index.

    Raises ValueError if no cells overlap (likely a cell-id format mismatch).
    """
    cell_ids = adata.obs[cell_id_column].astype(str)
    nucleus_ids = set(pd.Index(nucleus_boundaries.index).astype(str).tolist())
    keep_mask = cell_ids.isin(nucleus_ids).to_numpy()
    if keep_mask.sum() == 0:
        raise ValueError(
            f"nucleus_presence: no cells overlap with nucleus_boundaries. "
            f"table has {len(cell_ids)} cells; nucleus_boundaries has "
            f"{len(nucleus_ids)} entries. Check that cell_id formats match."
        )
    n_dropped = int((~keep_mask).sum())
    return keep_mask, {"missing_nucleus_boundary": n_dropped}


__all__ = ["filter_by_nucleus_presence"]
