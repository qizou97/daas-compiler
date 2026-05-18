# tests/test_genes.py
import hashlib, json, tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from daas.genes import (
    gene_panel_sha256,
    resolve_gene_panel,
    validate_gene_panel,
    write_gene_panel,
)


def _adata(gene_names: list[str]) -> anndata.AnnData:
    n = 3
    X = csr_matrix(np.ones((n, len(gene_names)), dtype=np.float32))
    obs = pd.DataFrame({"cell_id": [f"c{i}" for i in range(n)]})
    obs.index = [f"{i:07d}" for i in range(n)]
    var = pd.DataFrame(index=pd.Index(gene_names, name=""))
    return anndata.AnnData(X=X, obs=obs, var=var)


def test_resolve_first_sample_preserves_order():
    # A: genes B,A,C  B: genes A,B (intersection: A,B)
    # first_sample should preserve A's order → [B, A]
    a = _adata(["B", "A", "C"])
    b = _adata(["A", "B"])
    result = resolve_gene_panel([a, b], ["S1", "S2"], "first_sample")
    assert result == ["B", "A"]


def test_resolve_sorted():
    a = _adata(["B", "A", "C"])
    b = _adata(["A", "B", "D"])
    result = resolve_gene_panel([a, b], ["S1", "S2"], "sorted")
    assert result == ["A", "B"]


def test_resolve_explicit():
    a = _adata(["A", "B", "C"])
    b = _adata(["A", "B", "C"])
    result = resolve_gene_panel([a, b], ["S1", "S2"], "explicit", explicit_gene_panel=["C", "A"])
    assert result == ["C", "A"]


def test_resolve_explicit_missing_raises():
    a = _adata(["A", "B"])
    b = _adata(["A", "B"])
    with pytest.raises(ValueError, match="not in the intersection"):
        resolve_gene_panel([a, b], ["S1", "S2"], "explicit", explicit_gene_panel=["A", "Z"])


def test_resolve_empty_intersection_raises():
    a = _adata(["A", "B"])
    b = _adata(["C", "D"])
    with pytest.raises(ValueError, match="empty"):
        resolve_gene_panel([a, b], ["S1", "S2"], "first_sample")


def test_resolve_explicit_no_panel_raises():
    a = _adata(["A", "B"])
    with pytest.raises(ValueError, match="--gene-panel"):
        resolve_gene_panel([a], ["S1"], "explicit", explicit_gene_panel=None)


def test_validate_gene_panel_passes():
    a = _adata(["A", "B"])
    b = _adata(["A", "B"])
    validate_gene_panel([a, b], ["S1", "S2"], ["A", "B"])  # no exception


def test_validate_gene_panel_wrong_order_raises():
    a = _adata(["A", "B"])
    with pytest.raises(ValueError, match="S1"):
        validate_gene_panel([a], ["S1"], ["B", "A"])


def test_gene_panel_sha256_deterministic():
    h1 = gene_panel_sha256(["A", "B", "C"])
    h2 = gene_panel_sha256(["A", "B", "C"])
    assert h1 == h2
    assert len(h1) == 64  # hex sha256
    assert gene_panel_sha256(["A", "B"]) != gene_panel_sha256(["B", "A"])


def test_write_gene_panel_creates_files(tmp_path):
    panel = ["GeneA", "GeneB"]
    write_gene_panel(tmp_path, panel)
    assert (tmp_path / "gene_panel.json").exists()
    assert (tmp_path / "gene_panel.sha256").exists()
    assert json.loads((tmp_path / "gene_panel.json").read_text()) == panel
    sha = (tmp_path / "gene_panel.sha256").read_text().strip()
    assert sha == gene_panel_sha256(panel)
