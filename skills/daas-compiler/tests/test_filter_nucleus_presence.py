# skills/daas-compiler/tests/test_filter_nucleus_presence.py
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from daas.filters.nucleus_presence import filter_by_nucleus_presence
from daas.reports import StageReport, write_stage_report


def _adata(cell_ids):
    n = len(cell_ids)
    X = csr_matrix(np.ones((n, 2), dtype=np.float32))
    obs = pd.DataFrame({"cell_id": list(cell_ids)})
    obs.index = [f"{i:07d}" for i in range(n)]
    return anndata.AnnData(X=X, obs=obs,
                           var=pd.DataFrame(index=["g0", "g1"]))


def _nucleus_gdf(cell_ids):
    return pd.DataFrame({"x": range(len(cell_ids))},
                        index=pd.Index(list(cell_ids)))


def test_filter_keeps_cells_with_nucleus():
    adata = _adata(["c0", "c1", "c2", "c3", "c4"])
    nucleus = _nucleus_gdf(["c0", "c2", "c4"])
    keep, drops = filter_by_nucleus_presence(adata, nucleus)
    assert int(keep.sum()) == 3
    assert drops["missing_nucleus_boundary"] == 2


def test_filter_all_have_nucleus():
    adata = _adata(["c0", "c1", "c2"])
    nucleus = _nucleus_gdf(["c0", "c1", "c2"])
    keep, drops = filter_by_nucleus_presence(adata, nucleus)
    assert keep.all()
    assert drops["missing_nucleus_boundary"] == 0


def test_filter_preserves_row_order():
    adata = _adata(["c0", "c1", "c2", "c3"])
    nucleus = _nucleus_gdf(["c1", "c3"])
    keep, _ = filter_by_nucleus_presence(adata, nucleus)
    # c0=False, c1=True, c2=False, c3=True
    np.testing.assert_array_equal(keep, [False, True, False, True])


def test_empty_nucleus_boundaries_raises():
    adata = _adata(["c0", "c1"])
    nucleus = _nucleus_gdf([])
    with pytest.raises(ValueError, match="no cells overlap"):
        filter_by_nucleus_presence(adata, nucleus)


def test_stage_report_fields_after_filter(tmp_path):
    adata = _adata(["c0", "c1", "c2", "c3", "c4"])
    nucleus = _nucleus_gdf(["c0", "c2", "c4"])
    keep, drops = filter_by_nucleus_presence(adata, nucleus)
    r = StageReport(
        stage="nucleus_presence",
        zarr_path="/data/A.zarr",
        input_table_key="table_tissue",
        output_table_key="table_tissue_nucleus",
        input_shape_key="cell_circles",
        output_shape_key="cell_circles",
        n_cells_in=len(adata),
        n_cells_out=int(keep.sum()),
        drop_counts_by_reason=drops,
    )
    write_stage_report(r, tmp_path)
    assert r.n_cells_out == 3
    assert r.drop_counts_by_reason["missing_nucleus_boundary"] == 2
    assert r.report_path != ""
    data = json.loads((tmp_path / "nucleus_presence_table_tissue.json").read_text())
    assert data["n_cells_out"] == 3
