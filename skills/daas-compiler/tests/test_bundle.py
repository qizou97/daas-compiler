"""Tests for compile_dataset.py --bundle-wds and BundledCellPatchDataset."""
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.sparse import csr_matrix


COMPILE = str(Path(__file__).parent.parent / "scripts/compile_dataset.py")


def _build_two_sample_dir(tmp_path, synthetic_sample):
    """Duplicate synthetic_sample into a second sample so compile has 2 to merge."""
    import shutil
    s2_dir = tmp_path / "TEST_002"
    shutil.copytree(synthetic_sample["dir"], s2_dir)
    mf2 = pd.read_parquet(s2_dir / "manifest.parquet")
    mf2["sample_id"] = "TEST_002"
    mf2["shard_path"] = mf2["shard_path"].astype(str).str.replace(
        synthetic_sample["sample_id"], "TEST_002")
    mf2.to_parquet(s2_dir / "manifest.parquet", index=False)
    a2 = anndata.read_h5ad(s2_dir / "expression.h5ad")
    a2.obs["sample_id"] = "TEST_002"
    a2.write_h5ad(s2_dir / "expression.h5ad")
    return s2_dir


def test_bundle_wds_produces_expected_files(synthetic_sample, tmp_path):
    _build_two_sample_dir(tmp_path, synthetic_sample)
    compiled = tmp_path / "compiled"

    result = subprocess.run(
        [sys.executable, COMPILE,
         "--per-sample-dir", str(tmp_path),
         "--output",         str(compiled),
         "--bundle-wds",
         "--shard-size",     "4"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    wds = compiled / "wds"
    assert wds.exists()
    assert (wds / "manifest.parquet").exists()
    assert (wds / "gene_panel.json").exists()

    shards = sorted(wds.glob("shard-*.tar"))
    assert len(shards) > 0, "Expected at least one bundled shard"

    gene_panel = json.loads((wds / "gene_panel.json").read_text())
    assert isinstance(gene_panel, list)
    assert len(gene_panel) == synthetic_sample["n_genes"]


def test_bundle_wds_tar_members(synthetic_sample, tmp_path):
    """Each cell in a bundled tar must have .jpg + .expr.npz + .json."""
    _build_two_sample_dir(tmp_path, synthetic_sample)
    compiled = tmp_path / "compiled"

    subprocess.run(
        [sys.executable, COMPILE,
         "--per-sample-dir", str(tmp_path),
         "--output",         str(compiled),
         "--bundle-wds"],
        check=True, capture_output=True, text=True
    )

    shards = sorted((compiled / "wds").glob("shard-*.tar"))
    assert shards

    with tarfile.open(shards[0], "r") as tf:
        names = {m.name for m in tf.getmembers()}

    keys = {n.split(".")[0] for n in names}
    for key in keys:
        assert f"{key}.jpg" in names
        assert f"{key}.expr.npz" in names
        assert f"{key}.json" in names


def test_bundled_dataset_returns_image_and_expression(synthetic_sample, tmp_path):
    """BundledCellPatchDataset reads each cell correctly."""
    _build_two_sample_dir(tmp_path, synthetic_sample)
    compiled = tmp_path / "compiled"

    subprocess.run(
        [sys.executable, COMPILE,
         "--per-sample-dir", str(tmp_path),
         "--output",         str(compiled),
         "--bundle-wds"],
        check=True, capture_output=True, text=True
    )

    from daas.dataset import BundledCellPatchDataset
    ds = BundledCellPatchDataset(wds_dir=compiled / "wds")

    assert len(ds) == synthetic_sample["n_cells"] * 2

    sample = ds[0]
    assert "image" in sample
    assert "expression" in sample
    assert "cell_id" in sample
    assert "sample_id" in sample

    assert isinstance(sample["image"], Image.Image)
    assert sample["image"].size == (224, 224)

    assert sample["expression"].shape == (synthetic_sample["n_genes"],)
    assert sample["expression"].dtype == np.float32


def test_bundled_dataset_sparse_mode(synthetic_sample, tmp_path):
    _build_two_sample_dir(tmp_path, synthetic_sample)
    compiled = tmp_path / "compiled"

    subprocess.run(
        [sys.executable, COMPILE,
         "--per-sample-dir", str(tmp_path),
         "--output",         str(compiled),
         "--bundle-wds"],
        check=True, capture_output=True, text=True
    )

    from daas.dataset import BundledCellPatchDataset
    ds = BundledCellPatchDataset(wds_dir=compiled / "wds",
                                  dense_expression=False)
    indices, values, n_genes = ds[0]["expression"]
    assert indices.dtype == np.int32
    assert values.dtype == np.float32
    assert n_genes == synthetic_sample["n_genes"]
    assert len(indices) == len(values)


def test_bundled_dataset_sample_ids_filter(synthetic_sample, tmp_path):
    _build_two_sample_dir(tmp_path, synthetic_sample)
    compiled = tmp_path / "compiled"

    subprocess.run(
        [sys.executable, COMPILE,
         "--per-sample-dir", str(tmp_path),
         "--output",         str(compiled),
         "--bundle-wds"],
        check=True, capture_output=True, text=True
    )

    from daas.dataset import BundledCellPatchDataset
    ds = BundledCellPatchDataset(wds_dir=compiled / "wds",
                                  sample_ids=["TEST_002"])
    assert len(ds) == synthetic_sample["n_cells"]
    assert ds[0]["sample_id"] == "TEST_002"


def test_bundled_expression_matches_compiled_h5ad(synthetic_sample, tmp_path):
    """The sparse expression decoded from .expr.npz must match compiled h5ad row."""
    _build_two_sample_dir(tmp_path, synthetic_sample)
    compiled = tmp_path / "compiled"

    subprocess.run(
        [sys.executable, COMPILE,
         "--per-sample-dir", str(tmp_path),
         "--output",         str(compiled),
         "--bundle-wds"],
        check=True, capture_output=True, text=True
    )

    from daas.dataset import BundledCellPatchDataset
    ds = BundledCellPatchDataset(wds_dir=compiled / "wds")

    h5ad = anndata.read_h5ad(compiled / "expression.h5ad")

    # Compare a handful of rows
    for global_idx in [0, 1, len(ds) // 2, len(ds) - 1]:
        bundled_expr = ds[global_idx]["expression"]
        h5ad_row = h5ad.X[global_idx]
        h5ad_dense = (h5ad_row.toarray() if hasattr(h5ad_row, "toarray")
                      else np.asarray(h5ad_row)).reshape(-1).astype(np.float32)
        np.testing.assert_allclose(bundled_expr, h5ad_dense, rtol=1e-5)
