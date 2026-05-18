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


def test_compile_gene_panel_always_written(synthetic_sample, tmp_path):
    """gene_panel.json and gene_panel.sha256 are written even without --bundle-wds."""
    import shutil, json
    per_sample = tmp_path / "only_one"
    per_sample.mkdir()
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001")
    compiled_dir = tmp_path / "compiled_gp2"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (compiled_dir / "gene_panel.json").exists(), "gene_panel.json missing"
    assert (compiled_dir / "gene_panel.sha256").exists(), "gene_panel.sha256 missing"
    panel = json.loads((compiled_dir / "gene_panel.json").read_text())
    assert len(panel) == synthetic_sample["n_genes"]


def test_compile_report_json_written(synthetic_sample, tmp_path):
    """compile_report.json is written with gene metadata."""
    import shutil, json
    per_sample = tmp_path / "per_sample_report"
    per_sample.mkdir()
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001")
    compiled_dir = tmp_path / "compiled_report"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    report_path = compiled_dir / "compile_report.json"
    assert report_path.exists(), "compile_report.json missing"
    report = json.loads(report_path.read_text())
    assert "gene_panel_sha256" in report
    assert "gene_order_policy" in report
    assert "n_genes" in report
    assert report["n_samples"] == 1


def test_compile_sorted_gene_order(synthetic_sample, tmp_path):
    """--gene-order=sorted produces lexicographically sorted var_names."""
    import shutil
    per_sample = tmp_path / "per_sample_sorted"
    per_sample.mkdir()
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001")
    compiled_dir = tmp_path / "compiled_sorted"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled_dir),
         "--gene-order", "sorted"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    adata = anndata.read_h5ad(compiled_dir / "expression.h5ad")
    assert list(adata.var_names) == sorted(adata.var_names)


def test_compile_gene_order_identical_across_samples_different_input_orders(
    synthetic_sample, tmp_path
):
    """Gene order in output is identical regardless of per-sample input order."""
    import shutil, json
    import numpy as np
    from scipy.sparse import csr_matrix as csr
    per_sample = tmp_path / "per_sample_order"
    per_sample.mkdir()

    # Sample 1: genes in original order gene_0..gene_9
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001")

    # Sample 2: genes in reversed order gene_9..gene_0
    s2_dir = per_sample / "TEST_002"
    s2_dir.mkdir()
    shutil.copy(synthetic_sample["dir"] / "manifest.parquet", s2_dir)
    shutil.copy(synthetic_sample["dir"] / "shard-000000.tar", s2_dir)
    shutil.copy(synthetic_sample["dir"] / "shard-000001.tar", s2_dir)
    a_orig = anndata.read_h5ad(synthetic_sample["dir"] / "expression.h5ad")
    reversed_genes = list(reversed(a_orig.var_names.tolist()))
    a2 = anndata.AnnData(
        X=csr(np.ones((a_orig.n_obs, a_orig.n_vars), dtype=np.float32)),
        obs=a_orig.obs.copy(),
        var=pd.DataFrame(index=pd.Index(reversed_genes)),
    )
    a2.obs["sample_id"] = "TEST_002"
    a2.write_h5ad(s2_dir / "expression.h5ad")
    mf2 = pd.read_parquet(s2_dir / "manifest.parquet")
    mf2["sample_id"] = "TEST_002"
    mf2.to_parquet(s2_dir / "manifest.parquet", index=False)

    compiled_dir = tmp_path / "compiled_order"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    combined = anndata.read_h5ad(compiled_dir / "expression.h5ad")
    panel = json.loads((compiled_dir / "gene_panel.json").read_text())
    assert list(combined.var_names) == panel, "combined var_names != gene_panel"
    n1 = synthetic_sample["n_cells"]
    assert list(combined[:n1].var_names) == panel
    assert list(combined[n1:].var_names) == panel


def test_bundled_wds_json_includes_gene_panel_sha256(synthetic_sample, tmp_path):
    """Each cell .json in bundled shards includes gene_panel_sha256."""
    import shutil, json, tarfile, io
    per_sample = tmp_path / "per_sample_sha"
    per_sample.mkdir()
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001")
    compiled_dir = tmp_path / "compiled_sha"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled_dir),
         "--bundle-wds"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    shards = list((compiled_dir / "TEST_001").glob("shard-*.tar"))
    assert shards
    with tarfile.open(shards[0], "r") as tf:
        json_members = [m for m in tf.getmembers() if m.name.endswith(".json")]
        assert json_members
        meta = json.loads(tf.extractfile(json_members[0]).read())
    assert "gene_panel_sha256" in meta, f"gene_panel_sha256 missing from {meta}"
    expected_sha = (compiled_dir / "gene_panel.sha256").read_text().strip()
    assert meta["gene_panel_sha256"] == expected_sha
