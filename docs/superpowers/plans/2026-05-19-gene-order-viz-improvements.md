# Gene Order Contract + Visualization Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee gene order identity across samples, add tissue overlay to tiles overview, and add post-save tiling validation visualization.

**Architecture:** Three new modules (`daas/genes.py`, `daas/viz.py`) extract reusable logic from scripts; `compile_dataset.py` and `extract_sample.py` become pure orchestrators. Gene panel is always written (`gene_panel.json` + `gene_panel.sha256` + `compile_report.json`). Viz functions take explicit key arguments resolved via helper functions in `daas/viz.py`. Post-save grid reads real JPEGs from tars.

**Tech Stack:** Python 3.10+, anndata, numpy, pandas, matplotlib (extract group), PIL, shapely, tarfile

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `daas/genes.py` | Create | Gene panel resolution, validation, sha256, write |
| `daas/viz.py` | Create | Overlay key resolution, tiles overview, patch grid, saved patch grid |
| `scripts/compile_dataset.py` | Modify | Wire `daas.genes`, add `--gene-order`/`--gene-panel` flags, always write gene_panel + compile_report, include sha256 in bundled json |
| `daas/cli_args.py` | Modify | Add `--tissue-shapes-key`, `--cell-boundaries-key`, `--nucleus-boundaries-key` |
| `scripts/extract_sample.py` | Modify | Use `daas.viz`, resolve overlay keys, call post-save viz, update `_validate` |
| `tests/test_genes.py` | Create | All genes.py tests |
| `tests/test_viz.py` | Create | Key resolver + save_saved_patch_grid tests |
| `tests/test_compile.py` | Modify | gene_panel always written, sorted policy, sha256 in bundled json |
| `pyproject.toml` | Modify | Bump to 0.6.1 |
| `.claude-plugin/plugin.json` | Modify | Bump to 0.6.1 |
| `.claude-plugin/marketplace.json` | Modify | Bump to 0.6.1 |

---

## Task 1: daas/genes.py — Gene panel helpers

**Files:**
- Create: `daas/genes.py`
- Create: `tests/test_genes.py`

- [ ] **Step 1: Write failing tests**

```python
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
    with pytest.raises(AssertionError, match="S1"):
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
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_genes.py -q
```
Expected: ImportError or ModuleNotFoundError (daas.genes does not exist)

- [ ] **Step 3: Implement daas/genes.py**

```python
# daas/genes.py
"""Gene panel resolution for compile_dataset.py.

Ensures that gene order is identical across samples after compile.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def resolve_gene_panel(
    adatas: list,
    sample_ids: list[str],
    policy: str,
    explicit_gene_panel: list[str] | None = None,
) -> list[str]:
    """Return the ordered gene panel list (intersection of all samples).

    policy='first_sample': intersection ordered by first sample's var_names.
    policy='sorted': intersection sorted lexicographically.
    policy='explicit': use explicit_gene_panel (must be a subset of intersection).
    """
    if not adatas:
        raise ValueError("adatas is empty")

    intersection = set(adatas[0].var_names.tolist())
    for a in adatas[1:]:
        intersection &= set(a.var_names.tolist())

    if not intersection:
        raise ValueError(
            "Gene intersection is empty — check that all samples share at least one gene. "
            f"Sample IDs: {sample_ids}"
        )

    if policy == "first_sample":
        return [g for g in adatas[0].var_names.tolist() if g in intersection]

    if policy == "sorted":
        return sorted(intersection)

    if policy == "explicit":
        if explicit_gene_panel is None:
            raise ValueError("--gene-panel path required when --gene-order=explicit")
        missing = [g for g in explicit_gene_panel if g not in intersection]
        if missing:
            raise ValueError(
                f"Explicit gene panel contains genes not in the intersection: {missing[:5]}"
            )
        return list(explicit_gene_panel)

    raise ValueError(
        f"Unknown gene order policy: {policy!r}. "
        "Choose from: first_sample, sorted, explicit"
    )


def validate_gene_panel(
    adatas: list,
    sample_ids: list[str],
    gene_panel: list[str],
) -> None:
    """Assert every adata (already sliced to gene_panel) has var_names == gene_panel."""
    for a, sid in zip(adatas, sample_ids):
        actual = list(a.var_names)
        assert actual == gene_panel, (
            f"Sample {sid!r}: var_names differ from gene_panel. "
            f"len(actual)={len(actual)} len(panel)={len(gene_panel)}"
        )


def gene_panel_sha256(gene_panel: list[str]) -> str:
    """Return SHA-256 hex digest of the canonical JSON-serialised gene list."""
    payload = json.dumps(gene_panel, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_gene_panel(compiled_dir, gene_panel: list[str]) -> None:
    """Write gene_panel.json and gene_panel.sha256 to compiled_dir."""
    compiled_dir = Path(compiled_dir)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / "gene_panel.json").write_text(json.dumps(gene_panel))
    (compiled_dir / "gene_panel.sha256").write_text(gene_panel_sha256(gene_panel))


__all__ = [
    "resolve_gene_panel",
    "validate_gene_panel",
    "gene_panel_sha256",
    "write_gene_panel",
]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_genes.py -q
```
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
git add daas/genes.py tests/test_genes.py
git commit -m "feat: daas/genes.py — gene panel resolution, validation, sha256, write"
```

---

## Task 2: Update compile_dataset.py — gene order contract

**Files:**
- Modify: `scripts/compile_dataset.py`
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write failing tests (append to test_compile.py)**

```python
# Append to tests/test_compile.py

def test_compile_gene_panel_always_written(synthetic_sample, tmp_path):
    """gene_panel.json and gene_panel.sha256 are written even without --bundle-wds."""
    compiled_dir = tmp_path / "compiled_gp"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(tmp_path / "only_one"),
         "--output", str(compiled_dir)],
        capture_output=True, text=True,
    )
    # set up per_sample dir first
    per_sample = tmp_path / "only_one"
    per_sample.mkdir(exist_ok=True)
    import shutil
    shutil.copytree(synthetic_sample["dir"], per_sample / "TEST_001", dirs_exist_ok=True)
    compiled_dir2 = tmp_path / "compiled_gp2"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled_dir2)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    import json
    assert (compiled_dir2 / "gene_panel.json").exists(), "gene_panel.json missing"
    assert (compiled_dir2 / "gene_panel.sha256").exists(), "gene_panel.sha256 missing"
    panel = json.loads((compiled_dir2 / "gene_panel.json").read_text())
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
    from scipy.sparse import csr_matrix as csr
    import numpy as np
    a2 = anndata.AnnData(
        X=csr(np.ones((a_orig.n_obs, a_orig.n_vars), dtype=np.float32)),
        obs=a_orig.obs.copy(),
        var=pd.DataFrame(index=pd.Index(reversed_genes, name="")),
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
    # Both samples' slices should have same gene order as panel
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
    # Find a bundled shard
    shards = list((compiled_dir / "TEST_001").glob("shard-*.tar"))
    assert shards
    with tarfile.open(shards[0], "r") as tf:
        json_members = [m for m in tf.getmembers() if m.name.endswith(".json")]
        assert json_members
        meta = json.loads(tf.extractfile(json_members[0]).read())
    assert "gene_panel_sha256" in meta, f"gene_panel_sha256 missing from {meta}"
    expected_sha = (compiled_dir / "gene_panel.sha256").read_text().strip()
    assert meta["gene_panel_sha256"] == expected_sha
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_compile.py::test_compile_gene_panel_always_written \
       tests/test_compile.py::test_compile_report_json_written \
       tests/test_compile.py::test_compile_sorted_gene_order \
       tests/test_compile.py::test_compile_gene_order_identical_across_samples_different_input_orders \
       tests/test_compile.py::test_bundled_wds_json_includes_gene_panel_sha256 -q
```
Expected: 5 failures

- [ ] **Step 3: Implement compile_dataset.py changes**

Add to `parse_args()`:
```python
    p.add_argument("--gene-order",
                   choices=["first_sample", "sorted", "explicit"],
                   default="first_sample",
                   help="Gene ordering policy. first_sample=intersection ordered by "
                        "first sample's var_names (default). sorted=lexicographic. "
                        "explicit=use --gene-panel file.")
    p.add_argument("--gene-panel", default=None,
                   help="Path to JSON file with explicit ordered gene list. "
                        "Required when --gene-order=explicit.")
```

Replace the `# ── 2: Merge h5ad` block (lines ~217-238) with:
```python
    # ── 2: Merge h5ad with gene order contract ────────────────────────────────
    print("[2/3] Merging expression h5ad …")
    adatas = [anndata.read_h5ad(d / "expression.h5ad") for d in sample_dirs]
    sample_names = [d.name for d in sample_dirs]

    explicit_genes = None
    if args.gene_order == "explicit":
        if args.gene_panel is None:
            print("[compile] ERROR: --gene-panel path required with --gene-order=explicit")
            sys.exit(1)
        explicit_genes = json.loads(Path(args.gene_panel).read_text())

    from daas.genes import resolve_gene_panel, validate_gene_panel, write_gene_panel, gene_panel_sha256
    gene_panel = resolve_gene_panel(adatas, sample_names, args.gene_order,
                                    explicit_gene_panel=explicit_genes)
    n_genes_before = adatas[0].n_vars
    print(f"      genes: {n_genes_before} → {len(gene_panel)} "
          f"(policy={args.gene_order}, intersection across {len(adatas)} samples)")

    sliced = [a[:, gene_panel].copy() for a in adatas]
    validate_gene_panel(sliced, sample_names, gene_panel)

    combined = anndata.concat(sliced, axis=0, merge="same")
    combined.obs_names_make_unique()
    assert list(combined.var_names) == gene_panel, \
        f"combined var_names != gene_panel after concat"

    assert combined.n_obs == len(global_manifest), (
        f"h5ad rows ({combined.n_obs}) != manifest rows ({len(global_manifest)})"
    )
```

After `combined.write_h5ad(compiled / "expression.h5ad")`, add:
```python
    # Always write gene panel and compile report
    write_gene_panel(compiled, gene_panel)
    sha256 = gene_panel_sha256(gene_panel)
    compile_report = {
        "gene_order_policy": args.gene_order,
        "n_samples": len(sample_dirs),
        "sample_ids": [d.name for d in sample_dirs],
        "n_cells": int(combined.n_obs),
        "n_genes": len(gene_panel),
        "gene_panel_sha256": sha256,
        "output_dir": str(compiled),
        "elapsed_s": round(time.time() - t0, 2),
    }
    (compiled / "compile_report.json").write_text(
        json.dumps(compile_report, indent=2)
    )
    print(f"      gene_panel.json + gene_panel.sha256 + compile_report.json → {compiled}/")
```

In `_write_bundled_wds`, update the `meta` dict to include `gene_panel_sha256`. Add parameter `gene_panel_sha: str` to the function signature:

Change `_write_bundled_wds(compiled, global_manifest, combined, common_genes, args.shard_size)` call to:
```python
_write_bundled_wds(compiled, global_manifest, combined, gene_panel, args.shard_size, sha256)
```

Change `_write_bundled_wds` signature from:
```python
def _write_bundled_wds(compiled, global_manifest, combined, common_genes, shard_size):
```
to:
```python
def _write_bundled_wds(compiled, global_manifest, combined, gene_panel, shard_size,
                       gene_panel_sha: str = ""):
```

Inside `_write_bundled_wds`, change `gene_list = list(common_genes)` to `gene_list = list(gene_panel)`.

Remove the `(compiled / "gene_panel.json").write_text(json.dumps(gene_list))` line (now handled by `write_gene_panel` in main).

Update the `meta` dict to add `"gene_panel_sha256": gene_panel_sha`.

Also update the `_write_bundled_wds` call in `main()` to pass `sha256`:
```python
_, n_bundled_shards, n_genes_bundle = _write_bundled_wds(
    compiled, global_manifest, combined, gene_panel, args.shard_size, sha256)
```

And update the wds_info string to reference `gene_panel` instead of `common_genes`.

Also update the final print summary to use `gene_panel` instead of `common_genes`.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_compile.py -q
```
Expected: all passed (4 existing + 5 new = 9 total)

- [ ] **Step 5: Commit**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
git add scripts/compile_dataset.py tests/test_compile.py
git commit -m "feat: gene order contract in compile_dataset — daas.genes, always write gene_panel + compile_report"
```

---

## Task 3: daas/viz.py — overlay key resolvers + save_saved_patch_grid

**Files:**
- Create: `daas/viz.py`
- Create: `tests/test_viz.py`

Key insight: the resolver functions are pure Python (no matplotlib/spatialdata); the `save_saved_patch_grid` function only needs tarfile + PIL + matplotlib so it can be tested with synthetic fixtures.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_viz.py
import io, json, tarfile, tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from daas.viz import (
    resolve_cell_boundaries_key,
    resolve_nucleus_key,
    resolve_tissue_key,
    save_saved_patch_grid,
)


def _mock_sdata(shape_keys):
    """Minimal sdata stub: only .shapes attribute needed for resolvers."""
    return SimpleNamespace(shapes={k: None for k in shape_keys})


# ── resolver tests ────────────────────────────────────────────────────────

def test_resolve_tissue_key_auto_first_candidate():
    sdata = _mock_sdata(["tissue", "cell_circles"])
    assert resolve_tissue_key(sdata) == "tissue"


def test_resolve_tissue_key_auto_fallback():
    sdata = _mock_sdata(["tissue_boundaries", "cell_circles"])
    assert resolve_tissue_key(sdata) == "tissue_boundaries"


def test_resolve_tissue_key_auto_none_when_absent():
    sdata = _mock_sdata(["cell_circles", "cell_boundaries"])
    assert resolve_tissue_key(sdata) is None


def test_resolve_tissue_key_hint_none():
    sdata = _mock_sdata(["tissue"])
    assert resolve_tissue_key(sdata, hint="none") is None


def test_resolve_tissue_key_explicit_present():
    sdata = _mock_sdata(["my_tissue", "tissue"])
    assert resolve_tissue_key(sdata, hint="my_tissue") == "my_tissue"


def test_resolve_tissue_key_explicit_absent():
    sdata = _mock_sdata(["tissue"])
    assert resolve_tissue_key(sdata, hint="missing_key") is None


def test_resolve_cell_boundaries_key_prefers_filtered():
    sdata = _mock_sdata(["filtered_cell_boundaries", "cell_boundaries"])
    assert resolve_cell_boundaries_key(sdata) == "filtered_cell_boundaries"


def test_resolve_nucleus_key_auto():
    sdata = _mock_sdata(["nucleus_boundaries"])
    assert resolve_nucleus_key(sdata) == "nucleus_boundaries"


# ── save_saved_patch_grid tests ───────────────────────────────────────────

def _make_jpg(patch_size=32) -> bytes:
    arr = np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_shard(path: Path, keys: list[str], patch_size=32):
    with tarfile.open(path, "w") as tf:
        for k in keys:
            jpg = _make_jpg(patch_size)
            ti = tarfile.TarInfo(name=f"{k}.jpg")
            ti.size = len(jpg)
            tf.addfile(ti, io.BytesIO(jpg))
            meta = json.dumps({"sample_key": k, "cell_id": k}).encode()
            ti2 = tarfile.TarInfo(name=f"{k}.json")
            ti2.size = len(meta)
            tf.addfile(ti2, io.BytesIO(meta))


def test_save_saved_patch_grid_renders_grid(tmp_path):
    """save_saved_patch_grid reads tars, renders grid, writes png + report."""
    patch_size = 32
    shard = tmp_path / "shard-000000.tar"
    keys = [f"{i:07d}" for i in range(6)]
    _make_shard(shard, keys, patch_size)

    rows = [{"shard_path": str(shard), "sample_key": k,
             "cell_id": k, "bbox_x0": 0.0, "bbox_y0": 0.0} for k in keys]
    manifest = pd.DataFrame(rows)
    sdata = _mock_sdata([])

    viz_dir = tmp_path / "viz"
    result = save_saved_patch_grid(
        manifest_df=manifest,
        sdata=sdata,
        viz_dir=viz_dir,
        sample_id="TEST",
        patch_size=patch_size,
        SCALE_SHAPE=1.0,
        base_size=patch_size,
    )
    assert (viz_dir / "viz_saved_patch_grid.png").exists()
    assert (viz_dir / "viz_saved_patch_grid_report.json").exists()
    assert result["n_rendered"] == 6
    assert result["n_checked"] == 6
    assert result["missing_members"] == 0
    assert result["decode_errors"] == 0
    assert result["bad_image_size"] == 0


def test_save_saved_patch_grid_handles_missing_jpg(tmp_path):
    """Missing .jpg member is counted in missing_members, does not crash."""
    patch_size = 32
    shard = tmp_path / "shard-000000.tar"
    _make_shard(shard, ["0000000"], patch_size)

    rows = [
        {"shard_path": str(shard), "sample_key": "0000000",
         "cell_id": "0000000", "bbox_x0": 0.0, "bbox_y0": 0.0},
        {"shard_path": str(shard), "sample_key": "9999999",  # does not exist
         "cell_id": "9999999", "bbox_x0": 0.0, "bbox_y0": 0.0},
    ]
    manifest = pd.DataFrame(rows)
    sdata = _mock_sdata([])
    viz_dir = tmp_path / "viz"
    result = save_saved_patch_grid(
        manifest_df=manifest, sdata=sdata, viz_dir=viz_dir,
        sample_id="TEST", patch_size=patch_size, SCALE_SHAPE=1.0,
        base_size=patch_size,
    )
    assert result["missing_members"] == 1
    assert result["n_rendered"] == 1


def test_save_saved_patch_grid_bad_image_size(tmp_path):
    """JPEGs with wrong dimensions are counted in bad_image_size."""
    patch_size = 32
    shard = tmp_path / "shard-000000.tar"
    # write a 16x16 JPEG (wrong size)
    arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    wrong_jpg = buf.getvalue()
    with tarfile.open(shard, "w") as tf:
        ti = tarfile.TarInfo(name="0000000.jpg")
        ti.size = len(wrong_jpg)
        tf.addfile(ti, io.BytesIO(wrong_jpg))
        meta = json.dumps({"sample_key": "0000000"}).encode()
        ti2 = tarfile.TarInfo(name="0000000.json")
        ti2.size = len(meta)
        tf.addfile(ti2, io.BytesIO(meta))

    rows = [{"shard_path": str(shard), "sample_key": "0000000",
             "cell_id": "c0", "bbox_x0": 0.0, "bbox_y0": 0.0}]
    manifest = pd.DataFrame(rows)
    sdata = _mock_sdata([])
    viz_dir = tmp_path / "viz"
    result = save_saved_patch_grid(
        manifest_df=manifest, sdata=sdata, viz_dir=viz_dir,
        sample_id="TEST", patch_size=patch_size, SCALE_SHAPE=1.0,
        base_size=patch_size,
    )
    assert result["bad_image_size"] == 1
    assert result["n_rendered"] == 0
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_viz.py -q
```
Expected: ImportError

- [ ] **Step 3: Implement daas/viz.py**

```python
# daas/viz.py
"""Visualization helpers for extract_sample.py.

Overlay key resolution is pure Python. Render functions import matplotlib
lazily so they are only required when the extract group is installed.
"""
from __future__ import annotations

import io
import json as _json
import tarfile
from pathlib import Path

import numpy as np


_TISSUE_CANDIDATES = [
    "tissue",
    "tissue_boundaries",
    "tissue_boundary",
    "tissue_regions",
    "tissue_region",
    "filtered_tissue",
]
_CELL_CANDIDATES = [
    "filtered_cell_boundaries",
    "cell_boundaries",
    "filtered_cell_circles",
    "cell_circles",
]
_NUCLEUS_CANDIDATES = [
    "filtered_nucleus_boundaries",
    "nucleus_boundaries",
]


def _resolve_key(shapes_keys: set, candidates: list[str], hint: str) -> str | None:
    if hint == "none":
        return None
    if hint != "auto":
        return hint if hint in shapes_keys else None
    for k in candidates:
        if k in shapes_keys:
            return k
    return None


def resolve_tissue_key(sdata, hint: str = "auto") -> str | None:
    return _resolve_key(set(sdata.shapes.keys()), _TISSUE_CANDIDATES, hint)


def resolve_cell_boundaries_key(sdata, hint: str = "auto") -> str | None:
    return _resolve_key(set(sdata.shapes.keys()), _CELL_CANDIDATES, hint)


def resolve_nucleus_key(sdata, hint: str = "auto") -> str | None:
    return _resolve_key(set(sdata.shapes.keys()), _NUCLEUS_CANDIDATES, hint)


def save_tiles_overview(
    output_dir,
    wsi,
    sdata=None,
    tissue_key: str | None = None,
    SCALE_SHAPE: float = 1.0,
    dpi: int = 300,
) -> dict:
    """Render lazyslide tiles overview and optionally a tissue overlay variant.

    Returns dict with keys: viz_global_tiles, viz_global_tiles_tissue_overlay (or None), warnings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lazyslide.pl as lpl

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(exist_ok=True)
    warnings = []

    lpl.tiles(wsi, tile_key="cell_tiles")
    fig = plt.gcf()
    out = viz_dir / "viz_global_tiles.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    overlay_path = None
    if tissue_key is not None and sdata is not None:
        try:
            tissue_gdf = sdata.shapes[tissue_key]
            lpl.tiles(wsi, tile_key="cell_tiles")
            fig2 = plt.gcf()
            ax = fig2.axes[0]
            for geom in tissue_gdf.geometry:
                if geom is None:
                    continue
                try:
                    from shapely.geometry import MultiPolygon
                    polys = (list(geom.geoms)
                             if isinstance(geom, MultiPolygon) else [geom])
                    for poly in polys:
                        xs, ys = poly.exterior.xy
                        ax.plot(
                            [x * SCALE_SHAPE for x in xs],
                            [y * SCALE_SHAPE for y in ys],
                            color="lime", linewidth=1.0, alpha=0.8,
                        )
                except Exception as e:
                    warnings.append(f"tissue overlay geometry error: {e}")
            overlay_path = viz_dir / "viz_global_tiles_tissue_overlay.png"
            fig2.savefig(overlay_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig2)
        except Exception as e:
            warnings.append(f"tissue overlay skipped: {e}")
            overlay_path = None

    return {
        "viz_global_tiles": str(out),
        "viz_global_tiles_tissue_overlay": str(overlay_path) if overlay_path else None,
        "warnings": warnings,
    }


def save_patch_grid(
    images,
    cell_ids,
    x0s,
    y0s,
    sdata,
    SCALE_SHAPE: float,
    PATCH_SIZE: int,
    BASE_SIZE: int,
    sample_id: str,
    viz_dir,
    cell_key: str | None = None,
    nucleus_key: str | None = None,
    dpi: int = 300,
) -> Path:
    """Render pre-flight patch grid with optional boundary overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    viz_dir = Path(viz_dir)
    cell_bounds = (sdata.shapes[cell_key]
                   if cell_key and cell_key in sdata.shapes else None)
    nucl_bounds = (sdata.shapes[nucleus_key]
                   if nucleus_key and nucleus_key in sdata.shapes else None)
    nucl_ids = set(nucl_bounds.index) if nucl_bounds is not None else set()
    SCALE = PATCH_SIZE / BASE_SIZE

    def _um_to_px(coords_um, x0, y0):
        arr = np.array(coords_um)
        return np.column_stack([
            (arr[:, 0] * SCALE_SHAPE - x0) * SCALE,
            (arr[:, 1] * SCALE_SHAPE - y0) * SCALE,
        ])

    n_test = len(images)
    n_cols = int(np.ceil(np.sqrt(n_test)))
    n_rows = int(np.ceil(n_test / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_test == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flat

    for i in range(n_test):
        ax = axes_flat[i]
        ax.imshow(images[i])
        cell_id = cell_ids[i]
        x0, y0 = x0s[i], y0s[i]
        if cell_bounds is not None:
            try:
                cb_pts = _um_to_px(
                    list(cell_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(cb_pts, closed=True,
                                        edgecolor="cyan", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        if nucl_bounds is not None and cell_id in nucl_ids:
            try:
                nb_pts = _um_to_px(
                    list(nucl_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(nb_pts, closed=True,
                                        edgecolor="yellow", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        cx = cy = PATCH_SIZE / 2
        arm = PATCH_SIZE * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)
        ax.set_xlim(0, PATCH_SIZE)
        ax.set_ylim(PATCH_SIZE, 0)
        ax.set_title(str(cell_id)[:12], fontsize=5)
        ax.axis("off")

    for j in range(n_test, len(list(np.array(axes).flat))):
        np.array(axes).flat[j].axis("off")

    fig.suptitle(
        f"{sample_id} — patch grid pre-flight "
        f"({n_test} cells, cyan=cell  yellow=nucleus  +=center)",
        fontsize=9, y=0.995,
    )
    plt.tight_layout()
    out_path = viz_dir / "viz_patch_grid.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_saved_patch_grid(
    manifest_df,
    sdata,
    viz_dir,
    sample_id: str,
    patch_size: int,
    SCALE_SHAPE: float,
    x0_col: str = "bbox_x0",
    y0_col: str = "bbox_y0",
    cell_key: str | None = None,
    nucleus_key: str | None = None,
    base_size: int | None = None,
    n_grid: int = 25,
    seed: int = 0,
) -> dict:
    """Read saved JPEGs from manifest shards and render a post-save validation grid.

    Saves viz_dir/viz_saved_patch_grid.png and viz_dir/viz_saved_patch_grid_report.json.
    Returns the report dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from PIL import Image

    if base_size is None:
        base_size = patch_size

    viz_dir = Path(viz_dir)
    viz_dir.mkdir(exist_ok=True)
    overlay_keys_used = [k for k in [cell_key, nucleus_key] if k is not None]

    cell_bounds = (sdata.shapes[cell_key]
                   if cell_key and cell_key in sdata.shapes else None)
    nucl_bounds = (sdata.shapes[nucleus_key]
                   if nucleus_key and nucleus_key in sdata.shapes else None)
    nucl_ids = set(nucl_bounds.index) if nucl_bounds is not None else set()

    rng = np.random.default_rng(seed)
    n_available = len(manifest_df)
    n_sample = min(n_grid, n_available)
    chosen_idx = rng.choice(n_available, n_sample, replace=False)

    images, cell_ids, x0s, y0s = [], [], [], []
    missing_members = 0
    decode_errors = 0
    bad_image_size = 0

    open_tars: dict = {}
    try:
        for ci in chosen_idx:
            row = manifest_df.iloc[int(ci)]
            shard_path = str(row["shard_path"])
            sample_key = str(row["sample_key"])
            cell_id = str(row["cell_id"])

            if shard_path not in open_tars:
                try:
                    open_tars[shard_path] = tarfile.open(shard_path, "r")
                except Exception:
                    missing_members += 1
                    continue

            tf = open_tars[shard_path]
            try:
                member = tf.getmember(f"{sample_key}.jpg")
                jpg = tf.extractfile(member).read()
            except KeyError:
                missing_members += 1
                continue

            try:
                img_arr = np.array(Image.open(io.BytesIO(jpg)).convert("RGB"))
            except Exception:
                decode_errors += 1
                continue

            if img_arr.shape[:2] != (patch_size, patch_size):
                bad_image_size += 1
                continue

            images.append(img_arr)
            cell_ids.append(cell_id)
            x0s.append(float(row[x0_col]) if x0_col in row.index else 0.0)
            y0s.append(float(row[y0_col]) if y0_col in row.index else 0.0)
    finally:
        for tf in open_tars.values():
            tf.close()

    def _um_to_px(coords_um, x0, y0):
        arr = np.array(coords_um)
        SCALE = patch_size / base_size
        return np.column_stack([
            (arr[:, 0] * SCALE_SHAPE - x0) * SCALE,
            (arr[:, 1] * SCALE_SHAPE - y0) * SCALE,
        ])

    n_rendered = len(images)
    if n_rendered == 0:
        report = {
            "n_checked": n_sample, "n_rendered": 0,
            "missing_members": missing_members, "decode_errors": decode_errors,
            "bad_image_size": bad_image_size, "overlay_keys_used": overlay_keys_used,
            "viz_saved_patch_grid": None, "warnings": ["No images could be loaded"],
        }
        (viz_dir / "viz_saved_patch_grid_report.json").write_text(
            _json.dumps(report, indent=2))
        return report

    n_cols = int(np.ceil(np.sqrt(n_rendered)))
    n_rows = int(np.ceil(n_rendered / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rendered == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flat

    for i in range(n_rendered):
        ax = axes_flat[i]
        ax.imshow(images[i])
        cell_id = cell_ids[i]
        x0, y0 = x0s[i], y0s[i]
        if cell_bounds is not None:
            try:
                cb_pts = _um_to_px(
                    list(cell_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(cb_pts, closed=True,
                                        edgecolor="cyan", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        if nucl_bounds is not None and cell_id in nucl_ids:
            try:
                nb_pts = _um_to_px(
                    list(nucl_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(nb_pts, closed=True,
                                        edgecolor="yellow", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        cx = cy = patch_size / 2
        arm = patch_size * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)
        ax.set_xlim(0, patch_size)
        ax.set_ylim(patch_size, 0)
        ax.set_title(str(cell_id)[:12], fontsize=5)
        ax.axis("off")

    for j in range(n_rendered, len(list(np.array(axes).flat))):
        np.array(axes).flat[j].axis("off")

    fig.suptitle(
        f"{sample_id} — saved patch grid ({n_rendered} cells, "
        f"cyan=cell  yellow=nucleus)",
        fontsize=9, y=0.995,
    )
    plt.tight_layout()
    out_path = viz_dir / "viz_saved_patch_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_checked": n_sample, "n_rendered": n_rendered,
        "missing_members": missing_members, "decode_errors": decode_errors,
        "bad_image_size": bad_image_size, "overlay_keys_used": overlay_keys_used,
        "viz_saved_patch_grid": str(out_path), "warnings": [],
    }
    (viz_dir / "viz_saved_patch_grid_report.json").write_text(
        _json.dumps(report, indent=2))
    return report


__all__ = [
    "resolve_tissue_key",
    "resolve_cell_boundaries_key",
    "resolve_nucleus_key",
    "save_tiles_overview",
    "save_patch_grid",
    "save_saved_patch_grid",
]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_viz.py -q
```
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
git add daas/viz.py tests/test_viz.py
git commit -m "feat: daas/viz.py — overlay key resolvers, tiles overview, patch grid, saved patch grid"
```

---

## Task 4: Update extract_sample.py + cli_args.py

**Files:**
- Modify: `daas/cli_args.py`
- Modify: `scripts/extract_sample.py`

No new test files needed (the viz integration is covered by test_viz.py; extract_sample.py is tested via test_filtering_integration.py which stays unchanged).

- [ ] **Step 1: Update daas/cli_args.py**

In `build_extract_sample_parser()`, after the existing `--shapes-key` argument, add:

```python
    p.add_argument("--tissue-shapes-key", default="auto",
                   help="Tissue shape key for overlay viz. 'auto' = probe candidate "
                        "keys (tissue, tissue_boundaries, …). 'none' = skip overlay.")
    p.add_argument("--cell-boundaries-key", default="auto",
                   help="Cell boundary shape key for patch grid overlays. "
                        "'auto' = probe candidate keys.")
    p.add_argument("--nucleus-boundaries-key", default="auto",
                   help="Nucleus boundary shape key for patch grid overlays. "
                        "'auto' = probe candidate keys. 'none' = skip.")
```

Update `__all__` to export the three new constant defaults if needed (no changes required — defaults live in argparse).

- [ ] **Step 2: Update extract_sample.py — imports**

At the top of `scripts/extract_sample.py`, add to the imports:

```python
from daas.viz import (
    resolve_tissue_key,
    resolve_cell_boundaries_key,
    resolve_nucleus_key,
    save_tiles_overview,
    save_patch_grid,
    save_saved_patch_grid,
)
```

Remove the existing `_save_tiles_overview` and `_save_patch_grid` function definitions (they are fully replaced by `daas.viz`).

- [ ] **Step 3: Resolve overlay keys after Phase 1b**

After the `print(f"      {n_after_shape_alignment} cells …")` line at the end of Phase 1b, add:

```python
    # ── Resolve overlay keys (pure Python, no matplotlib) ────────────────────
    tissue_key   = resolve_tissue_key(sdata,   hint=args.tissue_shapes_key)
    cell_key     = resolve_cell_boundaries_key(sdata, hint=args.cell_boundaries_key)
    nucleus_key  = resolve_nucleus_key(sdata,  hint=args.nucleus_boundaries_key)
    print(f"      overlay keys: tissue={tissue_key!r}  cell={cell_key!r}  "
          f"nucleus={nucleus_key!r}")
```

Note: argparse converts `--tissue-shapes-key` → `tissue_shapes_key`, etc.

- [ ] **Step 4: Replace Phase 6 viz block**

Replace the existing Phase 6 block (the `_save_tiles_overview(output_dir, wsi)` call and the patch grid block) with:

```python
    # ── Phase 6: Pre-flight viz (BEFORE writing any shards) ──────────────────
    print("[6/9] Pre-flight viz: global tiles overview + patch grid …")
    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    add_tiles(wsi, key="cell_tiles",
              xys=np.column_stack([sx0_ord, sy0_ord]),
              tile_spec=spec, tissue_ids=np.zeros(n_out, dtype=int))

    # 6a. Global tiles overview (+ tissue overlay if key resolved)
    overview_result = save_tiles_overview(
        output_dir, wsi, sdata=sdata,
        tissue_key=tissue_key, SCALE_SHAPE=SCALE_SHAPE,
    )
    print(f"        → {overview_result['viz_global_tiles']}")
    if overview_result["viz_global_tiles_tissue_overlay"]:
        print(f"        → {overview_result['viz_global_tiles_tissue_overlay']} (tissue overlay)")
    for w in overview_result["warnings"]:
        print(f"        [warn] {w}")

    # 6b. Patch grid: 25 random in-memory test patches with boundary overlays
    n_grid   = min(25, n_out)
    rng_grid = np.random.default_rng(args.seed)
    grid_idx = rng_grid.choice(n_out, n_grid, replace=False)
    grid_xys = np.column_stack([sx0_ord[grid_idx], sy0_ord[grid_idx]])
    add_tiles(wsi, key="patch_grid", xys=grid_xys, tile_spec=spec,
              tissue_ids=np.zeros(n_grid, dtype=int))
    grid_images = []
    for tile in wsi.iter.tile_images("patch_grid"):
        assert tile.image.shape == (PATCH_SIZE, PATCH_SIZE, 3)
        assert tile.image.dtype == np.uint8
        grid_images.append(tile.image)
    grid_out = save_patch_grid(
        grid_images, [cell_ids_ord[i] for i in grid_idx],
        sx0_ord[grid_idx], sy0_ord[grid_idx],
        sdata, SCALE_SHAPE, PATCH_SIZE, BASE_SIZE,
        sample_id, viz_dir,
        cell_key=cell_key, nucleus_key=nucleus_key,
    )
    print(f"        → {grid_out}")

    # 6c. Write viz_report.json
    viz_report = {
        "tissue_key": tissue_key,
        "cell_key": cell_key,
        "nucleus_key": nucleus_key,
        "viz_global_tiles": overview_result["viz_global_tiles"],
        "viz_global_tiles_tissue_overlay": overview_result["viz_global_tiles_tissue_overlay"],
        "viz_patch_grid": str(grid_out),
        "warnings": overview_result["warnings"],
    }
    (viz_dir / "viz_report.json").write_text(json.dumps(viz_report, indent=2))
```

- [ ] **Step 5: Add post-save viz after Phase 8b**

After `cells_df.to_parquet(output_dir / "manifest.parquet", index=False)`, add:

```python
    # ── Phase 8c: Post-save viz (reads from actual tar shards) ───────────────
    print("[8c] Post-save patch grid validation …")
    saved_viz_result = save_saved_patch_grid(
        manifest_df=cells_df,
        sdata=sdata,
        viz_dir=viz_dir,
        sample_id=sample_id,
        patch_size=PATCH_SIZE,
        SCALE_SHAPE=SCALE_SHAPE,
        x0_col="bbox_x0",
        y0_col="bbox_y0",
        cell_key=cell_key,
        nucleus_key=nucleus_key,
        base_size=BASE_SIZE,
        n_grid=25,
        seed=args.seed,
    )
    print(f"      → {saved_viz_result.get('viz_saved_patch_grid')}")
    print(f"      n_checked={saved_viz_result['n_checked']}  "
          f"n_rendered={saved_viz_result['n_rendered']}  "
          f"missing={saved_viz_result['missing_members']}  "
          f"bad_size={saved_viz_result['bad_image_size']}")
```

- [ ] **Step 6: Update _validate signature and return value**

Replace the `_validate` function definition with:

```python
def _validate(cells_df, adata_out, adata_in, BASE_HALF, n_out, rng, patch_size,
              cell_id_column: str = "cell_id",
              overlay_keys_used: list | None = None) -> dict:
    import io, tarfile
    from PIL import Image
    if n_out == 0:
        raise ValueError(
            "No cells survived filtering — nothing to validate. "
            "Inspect filter_report.json: drop_counts_by_reason should explain "
            "where the cells went."
        )
    assert len(cells_df) == n_out == adata_out.n_obs
    assert cells_df["sample_key"].nunique() == n_out
    assert cells_df["cell_id"].nunique() == n_out
    assert (cells_df["expr_row"].values == cells_df["sample_index"].values).all(), \
        "expr_row != sample_index"
    assert (cells_df["cell_id"].values == adata_out.obs["cell_id"].values).all(), \
        "manifest cell_id != adata_out.obs cell_id"
    for i in range(min(200, n_out)):
        assert cells_df.iloc[i]["sample_key"] == adata_out.obs.iloc[i]["sample_key"]
        assert cells_df.iloc[i]["cell_id"] == adata_out.obs.iloc[i]["cell_id"]
    resolved_ids = adata_in.obs[cell_id_column].astype(str).values
    for i in range(min(50, n_out)):
        expected = resolved_ids[int(cells_df.iloc[i]["gene_row_index"])]
        assert str(cells_df.iloc[i]["cell_id"]) == expected, \
            "gene_row_index does not point to the cell_id of the resolved table"
    bbox_err = np.abs(cells_df["bbox_x0"].values
                      - (cells_df["center_x_pixel"].values - BASE_HALF))
    assert bbox_err.max() < 1.0
    n_random = min(20, n_out)
    check_idx = rng.choice(n_out, n_random, replace=False)
    bad_image_size = 0
    for si in check_idx:
        row = cells_df.iloc[int(si)]
        with tarfile.open(row["shard_path"], "r") as tf:
            jpg = tf.extractfile(f"{row['sample_key']}.jpg").read()
        img = Image.open(io.BytesIO(jpg))
        if img.size != (patch_size, patch_size):
            bad_image_size += 1
    assert bad_image_size == 0, f"{bad_image_size} JPEGs had wrong size"
    print(f"  [PASS] all validation checks (random JPEG sample size={n_random})")
    return {
        "n_checked": n_random,
        "missing_members": 0,
        "decode_errors": 0,
        "bad_image_size": bad_image_size,
        "overlay_keys_used": overlay_keys_used or [],
    }
```

Update the `_validate` call in `main()` to:
```python
    validate_report = _validate(
        cells_df, adata_out, adata, BASE_HALF, n_out, rng, PATCH_SIZE,
        cell_id_column=args.cell_id_column,
        overlay_keys_used=[k for k in [tissue_key, cell_key, nucleus_key] if k],
    )
```

- [ ] **Step 7: Run full test suite**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
python -m compileall daas scripts -q
pytest tests -q
```
Expected: all tests pass (original 69 + new genes/viz/compile tests)

- [ ] **Step 8: Commit**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
git add daas/cli_args.py scripts/extract_sample.py
git commit -m "feat: tissue overlay + post-save viz in extract_sample — use daas.viz, add --tissue-shapes-key/--cell-boundaries-key/--nucleus-boundaries-key"
```

---

## Task 5: Final verification + version bump

**Files:**
- Modify: `skills/daas-compiler/pyproject.toml`
- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`

- [ ] **Step 1: compileall**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
python -m compileall daas scripts -q
```
Expected: exit 0

- [ ] **Step 2: Full test suite**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests -q
```
Expected: all passed, 0 errors

- [ ] **Step 3: Import check**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
python -c "
from daas.genes import resolve_gene_panel, validate_gene_panel, write_gene_panel, gene_panel_sha256
from daas.viz import resolve_tissue_key, resolve_cell_boundaries_key, resolve_nucleus_key
from daas.viz import save_tiles_overview, save_patch_grid, save_saved_patch_grid
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 4: Bump version in pyproject.toml**

Change `version = "0.6.0"` to `version = "0.6.1"` in `skills/daas-compiler/pyproject.toml`.

- [ ] **Step 5: Bump version in .claude-plugin files**

In `/home1/zouqi/codes/daas-compiler/.claude-plugin/plugin.json`, change `"0.6.0"` → `"0.6.1"` (all occurrences).
In `/home1/zouqi/codes/daas-compiler/.claude-plugin/marketplace.json`, change `"0.6.0"` → `"0.6.1"` (all occurrences).

- [ ] **Step 6: Commit and tag**

```bash
cd /home1/zouqi/codes/daas-compiler
git add skills/daas-compiler/pyproject.toml .claude-plugin/plugin.json .claude-plugin/marketplace.json
git commit -m "chore(v0.6.1): gene order contract + tissue overlay + post-save viz"
git tag v0.6.1
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|-------------|------|
| A: `daas/genes.py` with resolve/validate/write/sha256 | Task 1 |
| A: `--gene-order`, `--gene-panel` flags | Task 2 |
| A: Always write `gene_panel.json`, `gene_panel.sha256`, `compile_report.json` | Task 2 |
| A: Slice every sample as `adata[:, gene_panel]`, assert after concat | Task 2 |
| A: `gene_panel_sha256` in bundled WDS json | Task 2 |
| B: `daas/viz.py` with tissue/cell/nucleus key candidates | Task 3 |
| B: `--tissue-shapes-key`, `--cell-boundaries-key`, `--nucleus-boundaries-key` CLI | Task 4 |
| B: `viz_global_tiles_tissue_overlay.png` if tissue resolved | Task 3+4 |
| B: Warn and record in `viz_report.json` if tissue absent | Task 4 |
| C: `save_saved_patch_grid` reads JPEGs from tar shards | Task 3 |
| C: Called after manifest.parquet + expression.h5ad written | Task 4 |
| C: Saves `viz_saved_patch_grid.png` + `viz_saved_patch_grid_report.json` | Task 3 |
| C: `_validate` extended with n_checked/missing_members/decode_errors/bad_image_size/overlay_keys_used | Task 4 |
| D: `daas/viz.py` and `daas/genes.py` modules, scripts orchestrate only | Tasks 1,3 |
| E: gene order tests | Task 1+2 |
| E: tissue overlay resolver tests | Task 3 |
| E: post-save viz reads tars, catches missing/bad | Task 3 |
| Version 0.6.1 | Task 5 |

**Type/signature consistency:** `save_saved_patch_grid` uses `sdata.shapes.get(cell_key)` — but `sdata.shapes` for a real SpatialData is a dict-like (supports `.get()`) and for the test stub it's a plain dict. ✅ Both support `.get()`.

**Potential issue:** In the test `_mock_sdata`, `sdata.shapes` is `{k: None}`. The resolver calls `set(sdata.shapes.keys())` — works on a plain dict. The `save_saved_patch_grid` calls `sdata.shapes.get(cell_key)` — also works on plain dict (returns `None` when key absent). ✅

**Placeholder scan:** No TBD, no "implement later", all code blocks are complete. ✅
