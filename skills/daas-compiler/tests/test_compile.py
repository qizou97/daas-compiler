import subprocess, sys, shutil
from pathlib import Path

import anndata
import pandas as pd
import pytest

PYTHON  = sys.executable
COMPILE = str(Path(__file__).parent.parent / "scripts/compile_dataset.py")


def test_compile_creates_output(synthetic_sample, tmp_path):
    """compile_dataset.py from a single per-sample dir produces correct output."""
    compiled_dir = tmp_path / "compiled"
    # duplicate synthetic sample as a second sample to test multi-sample merge
    s2_dir = tmp_path / "TEST_002"
    shutil.copytree(synthetic_sample["dir"], s2_dir)
    # fix sample_id in copied manifest
    mf2 = pd.read_parquet(s2_dir / "manifest.parquet")
    mf2["sample_id"] = "TEST_002"
    mf2.to_parquet(s2_dir / "manifest.parquet", index=False)
    a2 = anndata.read_h5ad(s2_dir / "expression.h5ad")
    a2.obs["sample_id"] = "TEST_002"
    a2.write_h5ad(s2_dir / "expression.h5ad")

    # per_sample_dir contains both TEST_001 and TEST_002
    per_sample_dir = tmp_path

    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample_dir),
         "--output",         str(compiled_dir)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    assert (compiled_dir / "manifest.parquet").exists()
    assert (compiled_dir / "expression.h5ad").exists()

    mf = pd.read_parquet(compiled_dir / "manifest.parquet")
    assert "global_idx" in mf.columns
    assert mf["global_idx"].is_monotonic_increasing
    assert len(mf) == synthetic_sample["n_cells"] * 2

    adata = anndata.read_h5ad(compiled_dir / "expression.h5ad")
    assert adata.n_obs == len(mf)
    assert list(mf["global_idx"]) == list(range(len(mf)))


def test_compile_gene_intersection(synthetic_sample, tmp_path):
    """compile takes gene intersection when panels differ."""
    per_sample = tmp_path / "per_sample_inter"
    per_sample.mkdir()
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001")

    # Create second sample with only 7 of 10 genes
    s2_dir = per_sample / "TEST_003"
    s2_dir.mkdir()
    shutil.copy(synthetic_sample["dir"] / "manifest.parquet", s2_dir)
    shutil.copy(synthetic_sample["dir"] / "shard-000000.tar", s2_dir)
    shutil.copy(synthetic_sample["dir"] / "shard-000001.tar", s2_dir)

    import numpy as np
    from scipy.sparse import csr_matrix
    a_orig = anndata.read_h5ad(synthetic_sample["dir"] / "expression.h5ad")
    X2   = csr_matrix(np.random.rand(6, 7).astype(np.float32))
    var2 = pd.DataFrame(index=a_orig.var_names[:7])   # first 7 genes
    obs2 = a_orig.obs.copy(); obs2["sample_id"] = "TEST_003"
    a2   = anndata.AnnData(X=X2, obs=obs2, var=var2)
    a2.write_h5ad(s2_dir / "expression.h5ad")
    mf2 = pd.read_parquet(s2_dir / "manifest.parquet")
    mf2["sample_id"] = "TEST_003"
    mf2.to_parquet(s2_dir / "manifest.parquet", index=False)

    compiled_dir = tmp_path / "compiled2"
    subprocess.run([PYTHON, COMPILE,
                    "--per-sample-dir", str(per_sample),
                    "--output",         str(compiled_dir)],
                   check=True)
    adata = anndata.read_h5ad(compiled_dir / "expression.h5ad")
    assert adata.n_vars == 7   # intersection


def test_compile_samples_flag_filters_dirs(synthetic_sample, tmp_path):
    """--samples restricts which sample dirs are compiled."""
    per_sample = tmp_path / "per_sample_samples_test"
    per_sample.mkdir()

    for name in ["KEEP_A", "KEEP_B", "SKIP_C"]:
        d = per_sample / name
        shutil.copytree(synthetic_sample["dir"], d)
        mf = pd.read_parquet(d / "manifest.parquet")
        mf["sample_id"] = name
        mf.to_parquet(d / "manifest.parquet", index=False)
        a = anndata.read_h5ad(d / "expression.h5ad")
        a.obs["sample_id"] = name
        a.write_h5ad(d / "expression.h5ad")

    compiled = tmp_path / "compiled_samples"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled),
         "--samples", "KEEP_A,KEEP_B"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    mf = pd.read_parquet(compiled / "manifest.parquet")
    assert set(mf["sample_id"].unique()) == {"KEEP_A", "KEEP_B"}
    assert len(mf) == synthetic_sample["n_cells"] * 2


def test_compile_missing_sample_exits_nonzero(synthetic_sample, tmp_path):
    """--samples with a non-existent name exits with code 1."""
    per_sample = tmp_path / "per_sample_missing"
    per_sample.mkdir()

    d = per_sample / "ONLY_ONE"
    shutil.copytree(synthetic_sample["dir"], d)

    compiled = tmp_path / "compiled_missing"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled),
         "--samples", "ONLY_ONE,DOES_NOT_EXIST"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "DOES_NOT_EXIST" in result.stdout + result.stderr
