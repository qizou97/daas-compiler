# HE2ST Task Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the HE2ST task adapter (`daas/tasks/he2st.py`), split metadata utilities (`daas/splits.py`), loader config helpers (`daas/loaders/configs.py`), and CLI scripts (`scripts/make_task_dataset.py`, `scripts/make_split.py`) that package L3 compiled artifacts into L4 training-ready datasets using sample/group-level split metadata — never physical shard duplication, never random cell-level splits.

**Architecture:** The adapter consumes `compiled/` L3 artifacts (manifest.parquet, expression.h5ad, gene_panel.json, WDS shards) and produces task metadata + split membership files. Splits are pure metadata: `split_membership.parquet` records which `global_idx` belongs to which split, derived from sample/group assignment. The HE2STDataset loader reads `loader_config.yaml` and filters cells by split membership at runtime without touching shards.

**Tech Stack:** Python 3.10+, pandas 2.2+, numpy 1.26+, PyYAML 6+, scipy (sparse npz), anndata 0.10+, pytest 7+

---

## File Map

**New files (create):**
- `skills/daas-compiler/daas/splits.py` — split membership builder/validator/writer
- `skills/daas-compiler/daas/tasks/__init__.py` — empty package marker
- `skills/daas-compiler/daas/tasks/he2st.py` — HE2ST task packaging logic
- `skills/daas-compiler/daas/loaders/__init__.py` — empty package marker
- `skills/daas-compiler/daas/loaders/configs.py` — helpers for writing loader_config.yaml / task_config.yaml
- `skills/daas-compiler/daas/loaders/he2st_dataset.py` — HE2STDataset class
- `skills/daas-compiler/scripts/make_task_dataset.py` — CLI for L3→L4 packaging
- `skills/daas-compiler/scripts/make_split.py` — CLI for standalone split generation
- `skills/daas-compiler/tests/test_splits.py` — tests for daas/splits.py
- `skills/daas-compiler/tests/test_he2st_task.py` — tests for he2st packaging + CLI
- `skills/daas-compiler/tests/test_he2st_dataset.py` — tests for HE2STDataset loader
- `skills/daas-compiler/tests/test_make_split_cli.py` — tests for make_split.py CLI

**Modified files (update):**
- `skills/daas-compiler/daas/__init__.py` — no change required unless re-export needed
- `skills/daas-compiler/README.md` — add L4 section
- `skills/daas-compiler/SKILL.md` — update description to include make_task_dataset.py
- `skills/daas-compiler/usage-guide.md` — add HE2ST training-ready section
- `skills/daas-compiler/references/training-ready-contract.md` — already correct; verify no update needed
- `skills/daas-compiler/references/task-adapters.md` — already correct; verify no update needed
- `skills/daas-compiler/references/agent-contract.md` — already correct; verify no update needed

---

## Shared Test Fixture (conftest additions)

All tasks share a synthetic compiled directory. Add this fixture to `tests/conftest.py`:

```python
@pytest.fixture
def compiled_dir(tmp_path, synthetic_sample):
    """Minimal compiled/ directory with bundled WDS, gene_panel, manifest."""
    import shutil, json, tarfile, io
    import numpy as np
    from scipy.sparse import csr_matrix
    import anndata

    cd = tmp_path / "compiled"
    cd.mkdir()

    # Copy gene panel files
    n_genes = synthetic_sample["n_genes"]
    sample_id = synthetic_sample["sample_id"]
    gene_panel = [f"gene_{i}" for i in range(n_genes)]
    sha = gene_panel_sha256(gene_panel)
    (cd / "gene_panel.json").write_text(json.dumps(gene_panel))
    (cd / "gene_panel.sha256").write_text(sha)

    # Global manifest (6 cells)
    n_cells = synthetic_sample["n_cells"]
    # Build a minimal bundled WDS shard for TEST_001
    sample_dir = cd / sample_id
    sample_dir.mkdir()

    meta_rows = []
    shard_path = sample_dir / "shard-000000.tar"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_cells):
            key = f"{i:09d}"
            jpg = _make_jpg()
            indices = np.array([0, 1], dtype=np.int32)
            values = np.array([1.0, 2.0], dtype=np.float32)
            npz_buf = io.BytesIO()
            np.savez(npz_buf, indices=indices, values=values)
            meta = json.dumps({
                "global_idx": i, "sample_id": sample_id,
                "cell_id": f"cell_{i}", "task": "he2st",
                "n_genes": n_genes, "gene_panel_sha256": sha,
            }).encode()
            for ext, data in [(".jpg", jpg), (".expr.npz", npz_buf.getvalue()), (".json", meta)]:
                ti = tarfile.TarInfo(name=f"{key}{ext}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            meta_rows.append({
                "global_idx": i, "sample_id": sample_id,
                "cell_id": f"cell_{i}", "sample_key": key,
                "global_key": key, "shard_path": str(shard_path),
            })
    shard_path.write_bytes(buf.getvalue())

    # Write bundled_manifest.parquet
    bm = pd.DataFrame(meta_rows)
    bm.to_parquet(cd / "bundled_manifest.parquet", index=False)

    # Write expression.h5ad
    X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))
    obs = pd.DataFrame({"sample_id": [sample_id]*n_cells,
                        "cell_id": [f"cell_{i}" for i in range(n_cells)]})
    obs.index = [f"{i:09d}" for i in range(n_cells)]
    adata = anndata.AnnData(X=X, obs=obs,
                            var=pd.DataFrame(index=gene_panel))
    adata.write_h5ad(cd / "expression.h5ad")

    # Write manifest.parquet
    mf = pd.DataFrame([{"global_idx": i, "sample_id": sample_id,
                         "cell_id": f"cell_{i}", "shard_path": str(shard_path),
                         "sample_key": f"{i:09d}"}
                        for i in range(n_cells)])
    mf.to_parquet(cd / "manifest.parquet", index=False)

    return cd
```

Add `from daas.genes import gene_panel_sha256` to the conftest imports.

---

## Task 1: `daas/splits.py` — Split membership utilities

**Files:**
- Create: `skills/daas-compiler/daas/splits.py`
- Test: `skills/daas-compiler/tests/test_splits.py`

- [ ] **Step 1.1: Write failing tests for `build_split_membership` — sample_holdout**

```python
# tests/test_splits.py
import json
import pandas as pd
import pytest
from daas.splits import build_split_membership, validate_split_membership, write_split_files, summarize_split_report


@pytest.fixture
def manifest_6cells():
    return pd.DataFrame({
        "global_idx": list(range(6)),
        "sample_id":  ["A", "A", "B", "B", "C", "C"],
        "cell_id":    [f"cell_{i}" for i in range(6)],
    })


def test_sample_holdout_basic(manifest_6cells, tmp_path):
    sm = build_split_membership(
        manifest=manifest_6cells,
        policy="sample_holdout",
        task="he2st",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )
    assert set(sm.columns) >= {"global_idx", "sample_id", "cell_id", "split", "task"}
    assert set(sm[sm["split"] == "train"]["sample_id"]) == {"A", "B"}
    assert set(sm[sm["split"] == "val"]["sample_id"]) == {"C"}
    assert len(sm) == 6


def test_sample_holdout_no_leakage(manifest_6cells):
    sm = build_split_membership(
        manifest=manifest_6cells,
        policy="sample_holdout",
        task="he2st",
        train_samples=["A"],
        val_samples=["B"],
        test_samples=["C"],
    )
    for sid in sm["sample_id"].unique():
        splits = sm[sm["sample_id"] == sid]["split"].unique()
        assert len(splits) == 1, f"sample {sid} leaked across splits: {splits}"
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd skills/daas-compiler && python -m pytest tests/test_splits.py::test_sample_holdout_basic -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'daas.splits'`

- [ ] **Step 1.3: Implement `daas/splits.py` — `build_split_membership` for sample_holdout**

```python
# daas/splits.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SUPPORTED_POLICIES = {"sample_holdout", "ratio_by_group", "group_kfold", "existing_file", "defer_split"}
GENERATED_LEVEL_POLICIES = {"sample_holdout", "ratio_by_group", "group_kfold"}


def build_split_membership(
    manifest: pd.DataFrame,
    policy: str,
    task: str,
    *,
    train_samples: list[str] | None = None,
    val_samples: list[str] | None = None,
    test_samples: list[str] | None = None,
    group_column: str | None = None,
    ratios: tuple[float, float, float] | None = None,
    n_folds: int | None = None,
    seed: int | None = None,
    split_file: str | Path | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    """Return a split_membership DataFrame indexed by global_idx.

    Required columns: global_idx, sample_id, cell_id, split, task.
    Optional columns: group_id, fold, split_source, seed, split_policy, leakage_warning.

    Raises ValueError for unsupported or forbidden policies (random_cell).
    """
    if policy == "random_cell":
        raise ValueError(
            "DAAS does not generate random cell-level splits because they can cause "
            "sample/patient leakage. Use sample_holdout, ratio_by_group, group_kfold, "
            "or provide an external benchmark split with existing_file."
        )
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Unknown split policy: {policy!r}. "
            f"Supported: {sorted(SUPPORTED_POLICIES)}"
        )

    required_cols = {"global_idx", "sample_id", "cell_id"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")

    if policy == "sample_holdout":
        return _sample_holdout(manifest, task, train_samples, val_samples, test_samples)
    if policy == "ratio_by_group":
        return _ratio_by_group(manifest, task, group_column, ratios, seed)
    if policy == "group_kfold":
        return _group_kfold(manifest, task, group_column, n_folds, seed, fold)
    if policy == "existing_file":
        return _existing_file(manifest, task, split_file)
    # defer_split — return None sentinel; caller should not call write_split_files
    raise ValueError(f"policy=defer_split is not directly buildable; skip split generation")


def _sample_holdout(manifest, task, train_samples, val_samples, test_samples):
    train_s = set(train_samples or [])
    val_s   = set(val_samples or [])
    test_s  = set(test_samples or [])

    overlap = (train_s & val_s) | (train_s & test_s) | (val_s & test_s)
    if overlap:
        raise ValueError(f"sample_holdout: same sample_id in multiple splits: {overlap}")

    assigned = train_s | val_s | test_s
    unassigned = set(manifest["sample_id"].unique()) - assigned
    if unassigned:
        raise ValueError(
            f"sample_holdout: these sample_ids are not assigned to any split: {unassigned}. "
            "Assign every sample or use ratio_by_group."
        )

    def _split_label(sid):
        if sid in train_s: return "train"
        if sid in val_s:   return "val"
        if sid in test_s:  return "test"
        return "unassigned"

    rows = manifest[["global_idx", "sample_id", "cell_id"]].copy()
    rows["split"] = rows["sample_id"].map(_split_label)
    rows["task"] = task
    rows["split_policy"] = "sample_holdout"
    rows["generated_at_level"] = "sample"
    return rows.reset_index(drop=True)


def _ratio_by_group(manifest, task, group_column, ratios, seed):
    if group_column is None:
        raise ValueError("ratio_by_group requires group_column")
    if group_column not in manifest.columns:
        raise ValueError(f"group_column {group_column!r} not in manifest columns: {list(manifest.columns)}")
    if ratios is None or len(ratios) != 3:
        raise ValueError("ratio_by_group requires ratios=(train_frac, val_frac, test_frac) summing to 1.0")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    groups = sorted(manifest[group_column].unique())
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(ratios[0] * n))
    n_val   = int(round(ratios[1] * n))
    train_groups = set(groups[:n_train])
    val_groups   = set(groups[n_train:n_train + n_val])
    test_groups  = set(groups[n_train + n_val:])

    def _label(g):
        if g in train_groups: return "train"
        if g in val_groups:   return "val"
        return "test"

    rows = manifest[["global_idx", "sample_id", "cell_id"]].copy()
    rows["group_id"] = manifest[group_column].values
    rows["split"] = rows["group_id"].map(_label)
    rows["task"] = task
    rows["split_policy"] = "ratio_by_group"
    rows["seed"] = seed
    rows["generated_at_level"] = "group"
    return rows.reset_index(drop=True)


def _group_kfold(manifest, task, group_column, n_folds, seed, fold):
    if group_column is None:
        raise ValueError("group_kfold requires group_column")
    if group_column not in manifest.columns:
        raise ValueError(f"group_column {group_column!r} not in manifest columns")
    if n_folds is None or n_folds < 2:
        raise ValueError("group_kfold requires n_folds >= 2")

    rng = np.random.default_rng(seed)
    groups = sorted(manifest[group_column].unique())
    rng.shuffle(groups)

    # assign each group to a fold index
    group_fold = {g: i % n_folds for i, g in enumerate(groups)}

    if fold is not None:
        # binary: this fold is val, others are train
        def _label(g):
            return "val" if group_fold[g] == fold else "train"
    else:
        # multi-fold: store fold number as split label for reporting
        def _label(g):
            return f"fold_{group_fold[g]}"

    rows = manifest[["global_idx", "sample_id", "cell_id"]].copy()
    rows["group_id"] = manifest[group_column].values
    rows["fold"] = rows["group_id"].map(group_fold)
    rows["split"] = rows["group_id"].map(_label)
    rows["task"] = task
    rows["split_policy"] = "group_kfold"
    rows["seed"] = seed
    rows["generated_at_level"] = "group"
    return rows.reset_index(drop=True)


def _existing_file(manifest, task, split_file):
    if split_file is None:
        raise ValueError("existing_file policy requires split_file path")
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"split_file not found: {split_file}")

    ext = split_file.suffix.lower()
    if ext == ".parquet":
        sf = pd.read_parquet(split_file)
    elif ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        sf = pd.read_csv(split_file, sep=sep)
    else:
        raise ValueError(f"Unsupported split_file format: {ext!r}. Use .parquet, .csv, or .tsv")

    if "split" not in sf.columns:
        raise ValueError("split_file must contain a 'split' column")

    leakage_warning = None

    # Detect join level
    if "global_idx" in sf.columns and "sample_id" not in sf.columns:
        # pure global_idx-level external split
        leakage_warning = (
            "This split is global_idx-level and was externally provided. "
            "DAAS did not generate a cell-level split. "
            "Validate leakage assumptions before using it."
        )
        rows = manifest[["global_idx", "sample_id", "cell_id"]].merge(
            sf[["global_idx", "split"]], on="global_idx", how="left"
        )
        rows["split_source"] = "external_global_idx"
        rows["generated_at_level"] = "external_global_idx"
    elif "sample_id" in sf.columns and "global_idx" not in sf.columns:
        rows = manifest[["global_idx", "sample_id", "cell_id"]].merge(
            sf[["sample_id", "split"]], on="sample_id", how="left"
        )
        rows["split_source"] = "external_sample"
        rows["generated_at_level"] = "sample"
    elif "global_idx" in sf.columns and "sample_id" in sf.columns:
        rows = manifest[["global_idx", "sample_id", "cell_id"]].merge(
            sf[["global_idx", "split"]], on="global_idx", how="left"
        )
        rows["split_source"] = "external_mixed"
        rows["generated_at_level"] = "external_global_idx"
        leakage_warning = (
            "This split is global_idx-level and was externally provided. "
            "DAAS did not generate a cell-level split. "
            "Validate leakage assumptions before using it."
        )
    else:
        raise ValueError(
            "split_file must contain at least one of: 'global_idx', 'sample_id'"
        )

    if rows["split"].isna().any():
        missing_count = rows["split"].isna().sum()
        raise ValueError(
            f"{missing_count} cells from manifest have no split assignment in split_file"
        )

    rows["task"] = task
    rows["split_policy"] = "existing_file"
    if leakage_warning:
        rows["leakage_warning"] = leakage_warning
    return rows.reset_index(drop=True)


def validate_split_membership(
    sm: pd.DataFrame,
    manifest: pd.DataFrame,
) -> list[str]:
    """Validate split_membership against manifest. Return list of warning strings.

    Raises ValueError on hard violations.
    """
    warnings = []

    # 1. Required columns
    required = {"global_idx", "sample_id", "cell_id", "split", "task"}
    missing = required - set(sm.columns)
    if missing:
        raise ValueError(f"split_membership missing columns: {missing}")

    # 2. Every global_idx in sm must exist in manifest
    manifest_idx = set(manifest["global_idx"].tolist())
    sm_idx = set(sm["global_idx"].tolist())
    unknown = sm_idx - manifest_idx
    if unknown:
        raise ValueError(
            f"split_membership references {len(unknown)} global_idx values "
            f"not in manifest: {list(unknown)[:5]}"
        )

    # 3. Each global_idx appears at most once
    dupes = sm[sm.duplicated("global_idx", keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"{len(dupes)} duplicate global_idx rows in split_membership"
        )

    # 4. No sample_id leakage for sample_holdout / ratio_by_group
    policy = sm.get("split_policy", pd.Series(["unknown"]*len(sm))).iloc[0] if "split_policy" in sm.columns else "unknown"
    if policy in {"sample_holdout", "ratio_by_group"}:
        sample_split = sm.groupby("sample_id")["split"].nunique()
        leaked = sample_split[sample_split > 1]
        if not leaked.empty:
            raise ValueError(
                f"sample_id leakage detected: these samples appear in multiple splits: "
                f"{leaked.index.tolist()}"
            )

    # 5. group_kfold: no group in more than one fold
    if policy == "group_kfold" and "group_id" in sm.columns:
        group_split = sm.groupby("group_id")["fold"].nunique()
        multi_fold = group_split[group_split > 1]
        if not multi_fold.empty:
            raise ValueError(
                f"group_kfold violation: groups in multiple folds: "
                f"{multi_fold.index.tolist()}"
            )

    # 6. Leakage warning for external global_idx splits
    if "leakage_warning" in sm.columns and sm["leakage_warning"].notna().any():
        warnings.append(sm["leakage_warning"].iloc[0])

    return warnings


def write_split_files(
    sm: pd.DataFrame,
    output_dir: Path | str,
    task: str,
    prefix: str = "",
) -> dict[str, Path]:
    """Write split_membership.parquet, train.json, val.json, test.json, split_report.json.

    Returns dict of name → Path for each written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pfx = f"{prefix}_" if prefix else ""

    sm_path = output_dir / f"{pfx}split_membership.parquet"
    sm.to_parquet(sm_path, index=False)

    written = {"split_membership": sm_path}

    for split_name in ["train", "val", "test"]:
        idx_list = sm[sm["split"] == split_name]["global_idx"].tolist()
        p = output_dir / f"{pfx}{split_name}.json"
        p.write_text(json.dumps(idx_list))
        written[split_name] = p

    report = summarize_split_report(sm, task)
    rp = output_dir / f"{pfx}split_report.json"
    rp.write_text(json.dumps(report, indent=2))
    written["split_report"] = rp

    return written


def summarize_split_report(sm: pd.DataFrame, task: str) -> dict:
    """Build a split_report dict from split_membership DataFrame."""
    policy = sm["split_policy"].iloc[0] if "split_policy" in sm.columns else "unknown"
    seed = int(sm["seed"].iloc[0]) if "seed" in sm.columns and sm["seed"].notna().any() else None
    level = sm["generated_at_level"].iloc[0] if "generated_at_level" in sm.columns else "unknown"

    counts_by_split = sm.groupby("split").size().to_dict()
    counts_by_sample = sm.groupby(["sample_id", "split"]).size().reset_index(name="n_cells")

    report = {
        "task": task,
        "split_policy": policy,
        "generated_at_level": level,
        "seed": seed,
        "n_cells_total": len(sm),
        "counts_by_split": counts_by_split,
        "per_sample_counts": counts_by_sample.to_dict(orient="records"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if "leakage_warning" in sm.columns and sm["leakage_warning"].notna().any():
        report["leakage_warning"] = sm["leakage_warning"].iloc[0]
    return report


__all__ = [
    "build_split_membership",
    "validate_split_membership",
    "write_split_files",
    "summarize_split_report",
    "SUPPORTED_POLICIES",
]
```

- [ ] **Step 1.4: Run failing test to verify it passes**

```bash
cd skills/daas-compiler && python -m pytest tests/test_splits.py::test_sample_holdout_basic tests/test_splits.py::test_sample_holdout_no_leakage -v
```

Expected: PASS

- [ ] **Step 1.5: Write remaining split tests**

```python
# Append to tests/test_splits.py

def test_random_cell_rejected(manifest_6cells):
    with pytest.raises(ValueError, match="DAAS does not generate random cell-level splits"):
        build_split_membership(manifest=manifest_6cells, policy="random_cell", task="he2st")


def test_ratio_by_group_no_leakage(manifest_6cells):
    sm = build_split_membership(
        manifest=manifest_6cells,
        policy="ratio_by_group",
        task="he2st",
        group_column="sample_id",
        ratios=(0.5, 0.33, 0.17),
        seed=42,
    )
    # No group should appear in more than one split
    for g in sm["group_id"].unique():
        splits = sm[sm["group_id"] == g]["split"].unique()
        assert len(splits) == 1, f"group {g} leaked across splits: {splits}"
    assert set(sm.columns) >= {"global_idx", "sample_id", "cell_id", "split", "task"}


def test_group_kfold_each_group_one_fold(manifest_6cells):
    sm = build_split_membership(
        manifest=manifest_6cells,
        policy="group_kfold",
        task="he2st",
        group_column="sample_id",
        n_folds=3,
        seed=0,
    )
    # Each group (sample) must be in exactly one fold
    for g in sm["group_id"].unique():
        folds = sm[sm["group_id"] == g]["fold"].unique()
        assert len(folds) == 1, f"group {g} in multiple folds: {folds}"


def test_existing_file_sample_level(manifest_6cells, tmp_path):
    sf = pd.DataFrame({"sample_id": ["A", "B", "C"], "split": ["train", "train", "val"]})
    sf_path = tmp_path / "split.parquet"
    sf.to_parquet(sf_path, index=False)
    sm = build_split_membership(
        manifest=manifest_6cells, policy="existing_file", task="he2st", split_file=sf_path
    )
    assert len(sm) == 6
    assert "leakage_warning" not in sm.columns or sm["leakage_warning"].isna().all()


def test_existing_file_global_idx_level_emits_leakage_warning(manifest_6cells, tmp_path):
    # global_idx-level external split
    sf = pd.DataFrame({
        "global_idx": list(range(6)),
        "split": ["train", "train", "train", "val", "val", "test"],
    })
    sf_path = tmp_path / "split.csv"
    sf.to_csv(sf_path, index=False)
    sm = build_split_membership(
        manifest=manifest_6cells, policy="existing_file", task="he2st", split_file=sf_path
    )
    assert "leakage_warning" in sm.columns
    assert sm["leakage_warning"].notna().all()
    assert "externally provided" in sm["leakage_warning"].iloc[0]


def test_validate_split_membership_detects_leakage(manifest_6cells):
    # Manually create a leaky split (same sample_id in train and val)
    sm = pd.DataFrame({
        "global_idx": list(range(6)),
        "sample_id": ["A", "A", "B", "B", "C", "C"],
        "cell_id": [f"cell_{i}" for i in range(6)],
        "split": ["train", "val", "train", "train", "val", "val"],  # A is in train+val
        "task": ["he2st"] * 6,
        "split_policy": ["sample_holdout"] * 6,
    })
    with pytest.raises(ValueError, match="leakage detected"):
        validate_split_membership(sm, manifest_6cells)


def test_write_split_files_creates_all_outputs(manifest_6cells, tmp_path):
    sm = build_split_membership(
        manifest=manifest_6cells,
        policy="sample_holdout",
        task="he2st",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )
    written = write_split_files(sm, tmp_path / "splits", task="he2st")
    assert (tmp_path / "splits" / "split_membership.parquet").exists()
    assert (tmp_path / "splits" / "train.json").exists()
    assert (tmp_path / "splits" / "val.json").exists()
    assert (tmp_path / "splits" / "test.json").exists()
    assert (tmp_path / "splits" / "split_report.json").exists()

    train_idx = json.loads((tmp_path / "splits" / "train.json").read_text())
    assert set(train_idx) == {0, 1, 2, 3}  # A(0,1) + B(2,3)


def test_generated_split_is_global_idx_level_but_derived_from_sample(manifest_6cells):
    """split_membership is indexed by global_idx but all cells from same sample have same split."""
    sm = build_split_membership(
        manifest=manifest_6cells,
        policy="sample_holdout",
        task="he2st",
        train_samples=["A"],
        val_samples=["B"],
        test_samples=["C"],
    )
    # global_idx-level
    assert "global_idx" in sm.columns
    # but derived from sample assignment
    for sid in ["A", "B", "C"]:
        splits = sm[sm["sample_id"] == sid]["split"].unique()
        assert len(splits) == 1
```

- [ ] **Step 1.6: Run all split tests**

```bash
cd skills/daas-compiler && python -m pytest tests/test_splits.py -v
```

Expected: all PASS

- [ ] **Step 1.7: Commit**

```bash
git add skills/daas-compiler/daas/splits.py skills/daas-compiler/tests/test_splits.py
git commit -m "feat(splits): add daas/splits.py with sample_holdout, ratio_by_group, group_kfold, existing_file policies"
```

---

## Task 2: `daas/loaders/configs.py` — Loader config helpers

**Files:**
- Create: `skills/daas-compiler/daas/loaders/__init__.py`
- Create: `skills/daas-compiler/daas/loaders/configs.py`
- Create: `skills/daas-compiler/daas/tasks/__init__.py`

- [ ] **Step 2.1: Write failing test for loader_config write/read roundtrip**

```python
# tests/test_he2st_task.py  (start this file)
import json
import yaml
import pytest
from pathlib import Path
from daas.loaders.configs import write_loader_config, write_task_config


def test_write_loader_config_training_ready(tmp_path):
    write_loader_config(
        output_path=tmp_path / "loader_config.yaml",
        task="he2st",
        training_ready_status="training_ready",
        shard_path_column="shard_path",
        sample_key_column="global_key",
        manifest_path=str(tmp_path / "bundled_manifest.parquet"),
        shard_glob=str(tmp_path / "data" / "shard-*.tar"),
        gene_panel_path=str(tmp_path / "gene_panel.json"),
        gene_panel_sha256="abc123",
        split_membership_path=str(tmp_path / "splits" / "split_membership.parquet"),
        split_status="available",
        generated_at_level="sample",
        patch_size=224,
        mpp=0.5,
    )
    cfg = yaml.safe_load((tmp_path / "loader_config.yaml").read_text())
    assert cfg["task"] == "he2st"
    assert cfg["training_ready_status"] == "training_ready"
    assert cfg["split"]["status"] == "available"
    assert cfg["split"]["generated_at_level"] == "sample"
    assert cfg["runtime"]["split_argument_required"] is True


def test_write_loader_config_split_pending(tmp_path):
    write_loader_config(
        output_path=tmp_path / "loader_config.yaml",
        task="he2st",
        training_ready_status="split_pending",
        shard_path_column="shard_path",
        sample_key_column="global_key",
        manifest_path=str(tmp_path / "bundled_manifest.parquet"),
        shard_glob=str(tmp_path / "data" / "shard-*.tar"),
        gene_panel_path=str(tmp_path / "gene_panel.json"),
        gene_panel_sha256="abc123",
        split_membership_path=None,
        split_status="missing",
        generated_at_level="missing",
    )
    cfg = yaml.safe_load((tmp_path / "loader_config.yaml").read_text())
    assert cfg["training_ready_status"] == "split_pending"
    assert cfg["split"]["status"] == "missing"
    assert cfg["split"]["generated_at_level"] == "missing"
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_task.py::test_write_loader_config_training_ready -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'daas.loaders'`

- [ ] **Step 2.3: Implement `daas/loaders/__init__.py`, `daas/loaders/configs.py`, `daas/tasks/__init__.py`**

```python
# daas/loaders/__init__.py
```

```python
# daas/tasks/__init__.py
```

```python
# daas/loaders/configs.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml


def write_loader_config(
    output_path: Path | str,
    *,
    task: str,
    training_ready_status: str,
    shard_path_column: str,
    sample_key_column: str,
    manifest_path: str,
    shard_glob: str,
    gene_panel_path: str,
    gene_panel_sha256: str,
    split_membership_path: Optional[str],
    split_status: str,
    generated_at_level: str,
    patch_size: int = 224,
    mpp: Optional[float] = None,
    normalization: str = "raw_counts",
) -> None:
    """Write loader_config.yaml for an HE2ST task package."""
    config = {
        "task": task,
        "training_ready_status": training_ready_status,
        "storage": {
            "format": "webdataset",
            "manifest_path": manifest_path,
            "shard_glob": shard_glob,
            "shard_path_column": shard_path_column,
            "sample_key_column": sample_key_column,
        },
        "gene_panel_path": gene_panel_path,
        "gene_panel_sha256": gene_panel_sha256,
        "split": {
            "required": True,
            "status": split_status,
            "split_membership_path": split_membership_path,
            "split_column": "split",
            "index_column": "global_idx",
            "generated_at_level": generated_at_level,
        },
        "target": {
            "type": "expression",
            "format": "sparse_npz",
            "normalization": normalization,
        },
        "image": {
            "format": "jpg",
            "patch_size": patch_size,
            "mpp": mpp,
        },
        "runtime": {
            "split_argument_required": True,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def write_task_config(
    output_path: Path | str,
    *,
    task: str,
    n_genes: int,
    gene_panel_path: str,
    gene_panel_sha256: str,
    input_modality: str = "he_image",
    target_modality: str = "gene_expression",
    patch_size: int = 224,
    mpp: Optional[float] = None,
    normalization: str = "raw_counts",
) -> None:
    """Write task_config.yaml for an HE2ST task package."""
    config = {
        "task": task,
        "input_modality": input_modality,
        "target_modality": target_modality,
        "n_genes": n_genes,
        "gene_panel_path": gene_panel_path,
        "gene_panel_sha256": gene_panel_sha256,
        "image": {
            "patch_size": patch_size,
            "mpp": mpp,
        },
        "target": {
            "normalization": normalization,
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


__all__ = ["write_loader_config", "write_task_config"]
```

- [ ] **Step 2.4: Run tests**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_task.py::test_write_loader_config_training_ready tests/test_he2st_task.py::test_write_loader_config_split_pending -v
```

Expected: PASS

- [ ] **Step 2.5: Commit**

```bash
git add skills/daas-compiler/daas/loaders/__init__.py skills/daas-compiler/daas/loaders/configs.py skills/daas-compiler/daas/tasks/__init__.py
git commit -m "feat(loaders): add daas/loaders/configs.py with write_loader_config and write_task_config"
```

---

## Task 3: `daas/tasks/he2st.py` — Core HE2ST task packaging logic

**Files:**
- Create: `skills/daas-compiler/daas/tasks/he2st.py`
- Test: add to `skills/daas-compiler/tests/test_he2st_task.py`

This module reads compiled/ artifacts, produces the task output directory, optionally with split metadata.

- [ ] **Step 3.1: Write failing tests for `package_he2st`**

```python
# Append to tests/test_he2st_task.py

import json
import yaml
import pandas as pd
import pytest
from pathlib import Path
from daas.tasks.he2st import package_he2st


def test_package_he2st_training_ready(compiled_dir, tmp_path):
    out = tmp_path / "he2st_task"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy="sample_holdout",
        train_samples=[compiled_dir.name if False else "TEST_001"],  # compiled_dir has TEST_001
        val_samples=[],
        test_samples=[],
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=False,
    )
    # Must produce all required files
    assert (out / "gene_panel.json").exists()
    assert (out / "gene_panel.sha256").exists()
    assert (out / "task_config.yaml").exists()
    assert (out / "loader_config.yaml").exists()
    assert (out / "dataset_card.json").exists()
    assert (out / "validation_report.json").exists()
    assert (out / "splits" / "split_membership.parquet").exists()
    assert (out / "splits" / "train.json").exists()
    assert (out / "splits" / "val.json").exists()
    assert (out / "splits" / "test.json").exists()

    card = json.loads((out / "dataset_card.json").read_text())
    assert card["training_ready"] is True
    assert card["training_ready_status"] == "training_ready"

    report = json.loads((out / "validation_report.json").read_text())
    assert report["training_ready_status"] == "training_ready"


def test_package_he2st_split_pending(compiled_dir, tmp_path):
    out = tmp_path / "he2st_task_pending"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy=None,
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=True,
    )
    assert (out / "task_config.yaml").exists()
    assert (out / "loader_config.yaml").exists()
    assert (out / "dataset_card.json").exists()
    assert (out / "validation_report.json").exists()
    # No split files
    assert not (out / "splits" / "split_membership.parquet").exists()

    card = json.loads((out / "dataset_card.json").read_text())
    assert card["training_ready"] is False
    assert card["training_ready_status"] == "split_pending"

    report = json.loads((out / "validation_report.json").read_text())
    assert report["training_ready_status"] == "split_pending"
    assert "split_membership.parquet" in str(report.get("missing_requirements", []))

    cfg = yaml.safe_load((out / "loader_config.yaml").read_text())
    assert cfg["split"]["status"] == "missing"
    assert cfg["training_ready_status"] == "split_pending"


def test_package_he2st_no_split_no_defer_raises(compiled_dir, tmp_path):
    out = tmp_path / "he2st_task_fail"
    with pytest.raises(ValueError, match="requires sample/group-level split metadata"):
        package_he2st(
            compiled_dir=compiled_dir,
            output_dir=out,
            split_policy=None,
            task="he2st",
            reuse_compiled_storage=False,
            defer_split=False,
        )


def test_package_he2st_gene_panel_copied(compiled_dir, tmp_path):
    out = tmp_path / "he2st_gp"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy="sample_holdout",
        train_samples=["TEST_001"],
        val_samples=[],
        test_samples=[],
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=False,
    )
    expected_sha = (compiled_dir / "gene_panel.sha256").read_text().strip()
    actual_sha   = (out / "gene_panel.sha256").read_text().strip()
    assert actual_sha == expected_sha


def test_package_he2st_no_physical_split_dirs(compiled_dir, tmp_path):
    out = tmp_path / "he2st_no_phys"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy="sample_holdout",
        train_samples=["TEST_001"],
        val_samples=[],
        test_samples=[],
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=False,
    )
    # No physical train/val/test shard dirs
    assert not (out / "train").exists()
    assert not (out / "val").exists()
    assert not (out / "test").exists()
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_task.py::test_package_he2st_split_pending -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'daas.tasks.he2st'`

- [ ] **Step 3.3: Implement `daas/tasks/he2st.py`**

```python
# daas/tasks/he2st.py
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from daas.splits import (
    build_split_membership,
    validate_split_membership,
    write_split_files,
    summarize_split_report,
)
from daas.loaders.configs import write_loader_config, write_task_config


def package_he2st(
    compiled_dir: Path | str,
    output_dir: Path | str,
    *,
    task: str = "he2st",
    split_policy: Optional[str] = None,
    train_samples: Optional[list[str]] = None,
    val_samples: Optional[list[str]] = None,
    test_samples: Optional[list[str]] = None,
    group_column: Optional[str] = None,
    ratios: Optional[tuple[float, float, float]] = None,
    n_folds: Optional[int] = None,
    seed: Optional[int] = None,
    split_file: Optional[Path | str] = None,
    reuse_compiled_storage: bool = False,
    defer_split: bool = False,
    patch_size: int = 224,
    mpp: Optional[float] = None,
    normalization: str = "raw_counts",
) -> None:
    """Package a compiled/ L3 directory into an L4 HE2ST task-ready directory.

    Raises ValueError if split info is absent and defer_split is False.
    """
    compiled_dir = Path(compiled_dir)
    output_dir   = Path(output_dir)

    # Validate split request
    has_split = _has_split_info(
        split_policy, train_samples, val_samples, test_samples,
        group_column, ratios, n_folds, split_file
    )
    if not has_split and not defer_split:
        raise ValueError(
            "HE2ST task-ready output requires sample/group-level split metadata. "
            "Provide --split-policy with sample/group assignment, provide --split-file, "
            "or pass --defer-split to create a split-pending task skeleton."
        )

    # Load compiled artifacts
    manifest = pd.read_parquet(compiled_dir / "manifest.parquet")
    gene_panel_path = compiled_dir / "gene_panel.json"
    gene_panel_sha_path = compiled_dir / "gene_panel.sha256"
    if not gene_panel_path.exists():
        raise FileNotFoundError(f"gene_panel.json not found in {compiled_dir}")

    import json as _json
    gene_panel = _json.loads(gene_panel_path.read_text())
    gene_panel_sha = gene_panel_sha_path.read_text().strip()
    n_genes = len(gene_panel)
    n_cells = len(manifest)
    sample_ids = sorted(manifest["sample_id"].unique().tolist())

    # Set up output directory
    if reuse_compiled_storage:
        task_dir = compiled_dir / "tasks" / task
    else:
        task_dir = output_dir
    task_dir.mkdir(parents=True, exist_ok=True)

    # Copy gene panel files
    shutil.copy2(gene_panel_path, task_dir / "gene_panel.json")
    shutil.copy2(gene_panel_sha_path, task_dir / "gene_panel.sha256")

    # Determine data/shard locations for loader config
    if reuse_compiled_storage:
        shard_glob = str(compiled_dir / "*" / "shard-*.tar")
        manifest_path_for_config = str(compiled_dir / "bundled_manifest.parquet")
        splits_dir = compiled_dir / "splits"
        split_prefix = task
    else:
        # Copy or symlink shards
        data_dir = task_dir / "data"
        data_dir.mkdir(exist_ok=True)
        shard_glob = str(data_dir / "shard-*.tar")
        _copy_shards(compiled_dir, data_dir, manifest)
        manifest_path_for_config = str(task_dir / "data" / "bundled_manifest.parquet")
        splits_dir = task_dir / "splits"
        split_prefix = ""

    # Build split membership
    training_ready_status = "split_pending"
    split_status = "missing"
    generated_at_level = "missing"
    split_membership_path_str = None
    warnings = []

    if has_split:
        sm = build_split_membership(
            manifest=manifest,
            policy=split_policy,
            task=task,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            group_column=group_column,
            ratios=ratios,
            n_folds=n_folds,
            seed=seed,
            split_file=split_file,
        )
        warnings = validate_split_membership(sm, manifest)
        split_files = write_split_files(sm, splits_dir, task=task, prefix=split_prefix)
        sm_path = split_files["split_membership"]
        split_membership_path_str = str(sm_path)
        generated_at_level = sm["generated_at_level"].iloc[0] if "generated_at_level" in sm.columns else "sample"
        training_ready_status = "training_ready"
        split_status = "available"

    # Write task_config.yaml
    write_task_config(
        output_path=task_dir / "task_config.yaml",
        task=task,
        n_genes=n_genes,
        gene_panel_path=str(task_dir / "gene_panel.json"),
        gene_panel_sha256=gene_panel_sha,
        patch_size=patch_size,
        mpp=mpp,
        normalization=normalization,
    )

    # Write loader_config.yaml
    write_loader_config(
        output_path=task_dir / "loader_config.yaml",
        task=task,
        training_ready_status=training_ready_status,
        shard_path_column="shard_path",
        sample_key_column="global_key",
        manifest_path=manifest_path_for_config,
        shard_glob=shard_glob,
        gene_panel_path=str(task_dir / "gene_panel.json"),
        gene_panel_sha256=gene_panel_sha,
        split_membership_path=split_membership_path_str,
        split_status=split_status,
        generated_at_level=generated_at_level,
        patch_size=patch_size,
        mpp=mpp,
        normalization=normalization,
    )

    # Write dataset_card.json
    card = {
        "task": task,
        "training_ready": training_ready_status == "training_ready",
        "training_ready_status": training_ready_status,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "sample_ids": sample_ids,
        "gene_panel_sha256": gene_panel_sha,
        "compiled_dir": str(compiled_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "daas_version": _get_daas_version(),
    }
    if training_ready_status == "training_ready" and has_split:
        card["split_policy"] = split_policy
    (task_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))

    # Write validation_report.json
    report = _build_validation_report(
        training_ready_status=training_ready_status,
        compiled_dir=compiled_dir,
        task=task,
        gene_panel_sha=gene_panel_sha,
        n_cells=n_cells,
        split_status=split_status,
        generated_at_level=generated_at_level,
        split_counts=None if not has_split else _split_counts(sm),
        warnings=warnings,
    )
    (task_dir / "validation_report.json").write_text(json.dumps(report, indent=2))


def _has_split_info(split_policy, train_samples, val_samples, test_samples,
                    group_column, ratios, n_folds, split_file) -> bool:
    if split_policy is None:
        return False
    if split_policy == "defer_split":
        return False
    if split_policy == "sample_holdout":
        return bool(train_samples or val_samples or test_samples)
    if split_policy in {"ratio_by_group", "group_kfold"}:
        return group_column is not None
    if split_policy == "existing_file":
        return split_file is not None
    return False


def _copy_shards(compiled_dir: Path, data_dir: Path, manifest: pd.DataFrame) -> None:
    """Copy all WDS shards referenced in manifest into data_dir, flat."""
    if (compiled_dir / "bundled_manifest.parquet").exists():
        bm = pd.read_parquet(compiled_dir / "bundled_manifest.parquet")
        shards = sorted(set(bm["shard_path"].tolist()))
    else:
        shards = sorted(set(manifest["shard_path"].tolist()))

    for i, src in enumerate(shards):
        dst = data_dir / f"shard-{i:06d}.tar"
        shutil.copy2(src, dst)

    # Copy bundled_manifest if it exists, update shard_path column
    bm_src = compiled_dir / "bundled_manifest.parquet"
    if bm_src.exists():
        bm = pd.read_parquet(bm_src)
        # Remap shard paths
        old_to_new = {s: str(data_dir / f"shard-{i:06d}.tar") for i, s in enumerate(shards)}
        bm["shard_path"] = bm["shard_path"].map(old_to_new)
        bm.to_parquet(data_dir / "bundled_manifest.parquet", index=False)


def _split_counts(sm: pd.DataFrame) -> dict:
    return sm.groupby("split").size().to_dict()


def _build_validation_report(
    training_ready_status, compiled_dir, task, gene_panel_sha, n_cells,
    split_status, generated_at_level, split_counts, warnings
) -> dict:
    report = {
        "training_ready_status": training_ready_status,
        "compiled_dir": str(compiled_dir),
        "task": task,
        "gene_panel_sha256": gene_panel_sha,
        "n_cells_total": n_cells,
        "split_metadata_status": split_status,
        "split_generated_at_level": generated_at_level,
        "counts_by_split": split_counts,
        "storage_validation_status": "ok",
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if training_ready_status == "split_pending":
        report["missing_requirements"] = [
            "split_membership.parquet",
            "train/val/test split definitions",
        ]
    return report


def _get_daas_version() -> str:
    try:
        from importlib.metadata import version
        return version("daas-compiler")
    except Exception:
        return "unknown"


__all__ = ["package_he2st"]
```

- [ ] **Step 3.4: Run HE2ST packaging tests (requires `compiled_dir` fixture)**

First, add the `compiled_dir` fixture and import to `tests/conftest.py` (see Shared Test Fixture section above).

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_task.py -v
```

Expected: all PASS

- [ ] **Step 3.5: Commit**

```bash
git add skills/daas-compiler/daas/tasks/__init__.py skills/daas-compiler/daas/tasks/he2st.py skills/daas-compiler/tests/test_he2st_task.py
git commit -m "feat(he2st): add daas/tasks/he2st.py with package_he2st — L3→L4 packaging"
```

---

## Task 4: `daas/loaders/he2st_dataset.py` — HE2STDataset loader

**Files:**
- Create: `skills/daas-compiler/daas/loaders/he2st_dataset.py`
- Test: `skills/daas-compiler/tests/test_he2st_dataset.py`

- [ ] **Step 4.1: Write failing tests for HE2STDataset**

```python
# tests/test_he2st_dataset.py
import json
import pytest
from pathlib import Path
from daas.loaders.he2st_dataset import HE2STDataset
from daas.tasks.he2st import package_he2st


@pytest.fixture
def he2st_task_dir(compiled_dir, tmp_path):
    out = tmp_path / "he2st_task"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy="sample_holdout",
        train_samples=["TEST_001"],
        val_samples=[],
        test_samples=[],
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=False,
    )
    return out


def test_he2st_dataset_from_config_train(he2st_task_dir):
    ds = HE2STDataset.from_config(he2st_task_dir / "loader_config.yaml", split="train")
    assert len(ds) > 0
    item = ds[0]
    assert "image" in item
    assert "expression" in item
    assert "cell_id" in item
    assert "sample_id" in item
    assert "global_idx" in item


def test_he2st_dataset_split_pending_raises(compiled_dir, tmp_path):
    out = tmp_path / "he2st_pending"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy=None,
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=True,
    )
    with pytest.raises(RuntimeError, match="split_pending"):
        HE2STDataset.from_config(out / "loader_config.yaml", split="train")


def test_he2st_dataset_filters_by_split(compiled_dir, tmp_path):
    """Dataset with val_samples=[] should return 0 items for split='val'."""
    out = tmp_path / "he2st_val"
    package_he2st(
        compiled_dir=compiled_dir,
        output_dir=out,
        split_policy="sample_holdout",
        train_samples=["TEST_001"],
        val_samples=[],
        test_samples=[],
        task="he2st",
        reuse_compiled_storage=False,
        defer_split=False,
    )
    ds = HE2STDataset.from_config(out / "loader_config.yaml", split="val")
    assert len(ds) == 0


def test_he2st_dataset_gene_panel_order_preserved(he2st_task_dir):
    import json
    gene_panel = json.loads((he2st_task_dir / "gene_panel.json").read_text())
    ds = HE2STDataset.from_config(he2st_task_dir / "loader_config.yaml", split="train")
    if len(ds) > 0:
        item = ds[0]
        expr = item["expression"]
        assert len(expr) == len(gene_panel)
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_dataset.py::test_he2st_dataset_from_config_train -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'daas.loaders.he2st_dataset'`

- [ ] **Step 4.3: Implement `daas/loaders/he2st_dataset.py`**

```python
# daas/loaders/he2st_dataset.py
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from PIL import Image


class HE2STDataset:
    """Runtime loader for HE2ST task packages.

    Reads loader_config.yaml, filters cells by split membership, and retrieves
    JPEG + expression from WDS shards. Does not require torch.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        gene_panel: list[str],
        gene_panel_sha256: str,
        transform=None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.gene_panel = gene_panel
        self.n_genes = len(gene_panel)
        self.gene_panel_sha256 = gene_panel_sha256
        self.transform = transform
        self._tar_handles: dict[str, tarfile.TarFile] = {}
        self._tar_members: dict[str, dict] = {}

    @classmethod
    def from_config(
        cls,
        config_path: Path | str,
        split: str,
        transform=None,
    ) -> "HE2STDataset":
        """Construct from loader_config.yaml. Fails clearly if split_pending."""
        config_path = Path(config_path)
        cfg = yaml.safe_load(config_path.read_text())

        if cfg.get("training_ready_status") == "split_pending":
            raise RuntimeError(
                "This HE2ST task package is split_pending. "
                "Generate sample/group-level split metadata with "
                "scripts/make_split.py before training."
            )

        split_cfg = cfg.get("split", {})
        if split_cfg.get("status") == "missing":
            raise RuntimeError(
                "This HE2ST task package is split_pending. "
                "Generate sample/group-level split metadata with "
                "scripts/make_split.py before training."
            )

        # Load split membership
        sm_path = split_cfg.get("split_membership_path")
        if sm_path is None:
            raise RuntimeError(
                "loader_config.yaml does not specify split_membership_path. "
                "Run scripts/make_split.py to generate split metadata."
            )
        sm = pd.read_parquet(sm_path)
        split_indices = set(sm[sm["split"] == split]["global_idx"].tolist())

        # Load manifest
        storage = cfg.get("storage", {})
        manifest_path = storage.get("manifest_path")
        if manifest_path is None:
            raise ValueError("loader_config.yaml storage.manifest_path is required")
        manifest = pd.read_parquet(manifest_path)

        # Filter by split
        manifest = manifest[manifest["global_idx"].isin(split_indices)].copy()

        # Load gene panel
        gene_panel_path = cfg.get("gene_panel_path")
        gene_panel = json.loads(Path(gene_panel_path).read_text())
        gene_panel_sha = cfg.get("gene_panel_sha256", "")

        return cls(
            manifest=manifest,
            gene_panel=gene_panel,
            gene_panel_sha256=gene_panel_sha,
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        shard_path = str(row["shard_path"])
        key = str(row["global_key"])

        tf, members = self._get_tar(shard_path)

        jpg_bytes = tf.extractfile(members[f"{key}.jpg"]).read()
        img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        npz_bytes = tf.extractfile(members[f"{key}.expr.npz"]).read()
        npz = np.load(io.BytesIO(npz_bytes))
        expr = np.zeros(self.n_genes, dtype=np.float32)
        if len(npz["indices"]):
            expr[npz["indices"]] = npz["values"]

        return {
            "image":      img,
            "expression": expr,
            "cell_id":    str(row["cell_id"]),
            "sample_id":  str(row["sample_id"]),
            "global_idx": int(row["global_idx"]),
        }

    def _get_tar(self, shard_path: str):
        if shard_path not in self._tar_handles:
            tf = tarfile.open(shard_path, "r")
            self._tar_handles[shard_path] = tf
            self._tar_members[shard_path] = {m.name: m for m in tf.getmembers()}
        return self._tar_handles[shard_path], self._tar_members[shard_path]

    def close(self):
        for tf in self._tar_handles.values():
            tf.close()
        self._tar_handles.clear()
        self._tar_members.clear()

    def __del__(self):
        if hasattr(self, "_tar_handles"):
            self.close()


__all__ = ["HE2STDataset"]
```

- [ ] **Step 4.4: Run loader tests**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_dataset.py -v
```

Expected: all PASS

- [ ] **Step 4.5: Commit**

```bash
git add skills/daas-compiler/daas/loaders/he2st_dataset.py skills/daas-compiler/tests/test_he2st_dataset.py
git commit -m "feat(loaders): add HE2STDataset with from_config(), split filtering, split_pending guard"
```

---

## Task 5: `scripts/make_task_dataset.py` — CLI for L3→L4 packaging

**Files:**
- Create: `skills/daas-compiler/scripts/make_task_dataset.py`
- Test: add to `skills/daas-compiler/tests/test_he2st_task.py`

- [ ] **Step 5.1: Write failing CLI tests**

```python
# Append to tests/test_he2st_task.py

import subprocess, sys
PYTHON = sys.executable
MAKE_TASK = str(Path(__file__).parent.parent / "scripts/make_task_dataset.py")


def test_make_task_dataset_training_ready(compiled_dir, tmp_path):
    out = tmp_path / "he2st_cli"
    result = subprocess.run([
        PYTHON, MAKE_TASK,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--output", str(out),
        "--format", "webdataset",
        "--split-policy", "sample_holdout",
        "--train-samples", "TEST_001",
        "--val-samples", "",
        "--test-samples", "",
        "--target-normalization", "raw_counts",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (out / "task_config.yaml").exists()
    assert (out / "loader_config.yaml").exists()
    assert (out / "dataset_card.json").exists()
    assert (out / "validation_report.json").exists()
    assert (out / "splits" / "split_membership.parquet").exists()
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["training_ready"] is True


def test_make_task_dataset_split_pending(compiled_dir, tmp_path):
    out = tmp_path / "he2st_cli_pending"
    result = subprocess.run([
        PYTHON, MAKE_TASK,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--output", str(out),
        "--format", "webdataset",
        "--defer-split",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["training_ready"] is False
    assert card["training_ready_status"] == "split_pending"


def test_make_task_dataset_no_split_no_defer_fails(compiled_dir, tmp_path):
    out = tmp_path / "he2st_cli_fail"
    result = subprocess.run([
        PYTHON, MAKE_TASK,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--output", str(out),
        "--format", "webdataset",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires sample/group-level split metadata" in result.stderr + result.stdout


def test_make_task_dataset_ratio_by_group(compiled_dir, tmp_path):
    out = tmp_path / "he2st_cli_ratio"
    result = subprocess.run([
        PYTHON, MAKE_TASK,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--output", str(out),
        "--format", "webdataset",
        "--split-policy", "ratio_by_group",
        "--group-column", "sample_id",
        "--ratios", "1.0,0.0,0.0",
        "--seed", "42",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    card = json.loads((out / "dataset_card.json").read_text())
    assert card["training_ready"] is True
```

- [ ] **Step 5.2: Run tests to verify they fail**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_task.py::test_make_task_dataset_training_ready -v 2>&1 | head -20
```

Expected: `FileNotFoundError` or similar (script doesn't exist yet)

- [ ] **Step 5.3: Implement `scripts/make_task_dataset.py`**

```python
#!/usr/bin/env python3
"""CLI for packaging a compiled/ L3 directory into an L4 HE2ST task-ready dataset.

Usage examples:

  # Sample holdout split
  python3 scripts/make_task_dataset.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --output /data/he2st_task \
      --format webdataset \
      --split-policy sample_holdout \
      --train-samples A_001,A_002 \
      --val-samples A_004 \
      --test-samples A_005 \
      --target-normalization raw_counts

  # Ratio-by-group split
  python3 scripts/make_task_dataset.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --output /data/he2st_task \
      --format webdataset \
      --split-policy ratio_by_group \
      --group-column patient_id \
      --ratios 0.8,0.1,0.1 \
      --seed 42

  # Defer split
  python3 scripts/make_task_dataset.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --output /data/he2st_task \
      --format webdataset \
      --defer-split
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Package a compiled/ directory into an HE2ST training-ready dataset."
    )
    p.add_argument("--compiled-dir", required=True)
    p.add_argument("--task", default="he2st")
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["webdataset"], default="webdataset")
    p.add_argument("--split-policy",
                   choices=["sample_holdout", "ratio_by_group", "group_kfold", "existing_file"],
                   default=None)
    p.add_argument("--train-samples", default=None,
                   help="Comma-separated sample IDs for train (sample_holdout only)")
    p.add_argument("--val-samples", default=None,
                   help="Comma-separated sample IDs for val (sample_holdout only)")
    p.add_argument("--test-samples", default=None,
                   help="Comma-separated sample IDs for test (sample_holdout only)")
    p.add_argument("--group-column", default=None)
    p.add_argument("--ratios", default=None,
                   help="Comma-separated train,val,test ratios (e.g. 0.8,0.1,0.1)")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-file", default=None)
    p.add_argument("--target-normalization", default="raw_counts")
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--mpp", type=float, default=None)
    p.add_argument("--defer-split", action="store_true")
    p.add_argument("--reuse-compiled-storage", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    from daas.tasks.he2st import package_he2st

    train_samples = [s.strip() for s in args.train_samples.split(",") if s.strip()] \
        if args.train_samples else None
    val_samples   = [s.strip() for s in args.val_samples.split(",") if s.strip()] \
        if args.val_samples else None
    test_samples  = [s.strip() for s in args.test_samples.split(",") if s.strip()] \
        if args.test_samples else None

    ratios = None
    if args.ratios:
        parts = [float(x.strip()) for x in args.ratios.split(",")]
        if len(parts) != 3:
            print("ERROR: --ratios must be three comma-separated floats", file=sys.stderr)
            sys.exit(1)
        ratios = tuple(parts)

    try:
        package_he2st(
            compiled_dir=Path(args.compiled_dir),
            output_dir=Path(args.output),
            task=args.task,
            split_policy=args.split_policy,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            group_column=args.group_column,
            ratios=ratios,
            n_folds=args.n_folds,
            seed=args.seed,
            split_file=args.split_file,
            reuse_compiled_storage=args.reuse_compiled_storage,
            defer_split=args.defer_split,
            patch_size=args.patch_size,
            mpp=args.mpp,
            normalization=args.target_normalization,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[make_task_dataset] Done → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.4: Run CLI tests**

```bash
cd skills/daas-compiler && python -m pytest tests/test_he2st_task.py -v -k "make_task_dataset"
```

Expected: all PASS

- [ ] **Step 5.5: Commit**

```bash
git add skills/daas-compiler/scripts/make_task_dataset.py
git commit -m "feat(cli): add scripts/make_task_dataset.py for L3→L4 HE2ST packaging"
```

---

## Task 6: `scripts/make_split.py` — Standalone split generation CLI

**Files:**
- Create: `skills/daas-compiler/scripts/make_split.py`
- Test: `skills/daas-compiler/tests/test_make_split_cli.py`

- [ ] **Step 6.1: Write failing CLI tests**

```python
# tests/test_make_split_cli.py
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PYTHON     = sys.executable
MAKE_SPLIT = str(Path(__file__).parent.parent / "scripts/make_split.py")


def test_make_split_sample_holdout(compiled_dir, tmp_path):
    split_dir = tmp_path / "splits"
    result = subprocess.run([
        PYTHON, MAKE_SPLIT,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--policy", "sample_holdout",
        "--train-samples", "TEST_001",
        "--val-samples", "",
        "--test-samples", "",
        "--output-split-dir", str(split_dir),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert (split_dir / "split_membership.parquet").exists()
    assert (split_dir / "train.json").exists()
    assert (split_dir / "split_report.json").exists()
    # Must NOT rewrite any WDS shards
    assert not any(split_dir.glob("**/*.tar"))


def test_make_split_ratio_by_group(compiled_dir, tmp_path):
    split_dir = tmp_path / "splits_ratio"
    result = subprocess.run([
        PYTHON, MAKE_SPLIT,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--policy", "ratio_by_group",
        "--group-column", "sample_id",
        "--ratios", "1.0,0.0,0.0",
        "--seed", "42",
        "--output-split-dir", str(split_dir),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    sm = pd.read_parquet(split_dir / "split_membership.parquet")
    assert "split" in sm.columns


def test_make_split_group_kfold(compiled_dir, tmp_path):
    split_dir = tmp_path / "splits_kfold"
    result = subprocess.run([
        PYTHON, MAKE_SPLIT,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--policy", "group_kfold",
        "--group-column", "sample_id",
        "--n-folds", "2",
        "--seed", "0",
        "--output-split-dir", str(split_dir),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    sm = pd.read_parquet(split_dir / "split_membership.parquet")
    folds = sm["fold"].unique().tolist()
    assert len(folds) >= 1


def test_make_split_existing_file_global_idx_warns(compiled_dir, tmp_path):
    # Provide a global_idx-level external split file
    n = 6  # compiled_dir has 6 cells
    sf = pd.DataFrame({"global_idx": list(range(n)),
                       "split": ["train"]*4 + ["val"]*2})
    sf_path = tmp_path / "ext_split.csv"
    sf.to_csv(sf_path, index=False)

    split_dir = tmp_path / "splits_ext"
    result = subprocess.run([
        PYTHON, MAKE_SPLIT,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--policy", "existing_file",
        "--split-file", str(sf_path),
        "--output-split-dir", str(split_dir),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "leakage" in result.stdout.lower() or "leakage" in result.stderr.lower()


def test_make_split_random_cell_rejected(compiled_dir, tmp_path):
    split_dir = tmp_path / "splits_random"
    result = subprocess.run([
        PYTHON, MAKE_SPLIT,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--policy", "random_cell",
        "--output-split-dir", str(split_dir),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "random cell-level splits" in result.stderr + result.stdout


def test_make_split_does_not_rewrite_shards(compiled_dir, tmp_path):
    """make_split.py must NOT write any .tar files."""
    split_dir = tmp_path / "splits_no_shards"
    subprocess.run([
        PYTHON, MAKE_SPLIT,
        "--compiled-dir", str(compiled_dir),
        "--task", "he2st",
        "--policy", "sample_holdout",
        "--train-samples", "TEST_001",
        "--output-split-dir", str(split_dir),
    ], check=True)
    assert not any(split_dir.glob("**/*.tar")), "make_split.py must not write shard files"
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
cd skills/daas-compiler && python -m pytest tests/test_make_split_cli.py::test_make_split_sample_holdout -v 2>&1 | head -20
```

Expected: `FileNotFoundError` (script not found yet)

- [ ] **Step 6.3: Implement `scripts/make_split.py`**

```python
#!/usr/bin/env python3
"""Standalone split metadata generation — does NOT rewrite WDS shards.

Usage:

  python3 scripts/make_split.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --policy sample_holdout \
      --train-samples A_001,A_002 \
      --val-samples A_004 \
      --test-samples A_005 \
      --output-split-dir /data/compiled/splits

  python3 scripts/make_split.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --policy ratio_by_group \
      --group-column patient_id \
      --ratios 0.8,0.1,0.1 \
      --seed 42 \
      --output-split-dir /data/compiled/splits

  python3 scripts/make_split.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --policy group_kfold \
      --group-column patient_id \
      --n-folds 5 \
      --seed 42 \
      --output-split-dir /data/compiled/splits

  python3 scripts/make_split.py \
      --compiled-dir /data/compiled \
      --task he2st \
      --policy existing_file \
      --split-file /data/splits/he2st_split.csv \
      --output-split-dir /data/compiled/splits
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Generate split metadata without rewriting WDS shards.")
    p.add_argument("--compiled-dir", required=True)
    p.add_argument("--task", default="he2st")
    p.add_argument("--policy", required=True,
                   help="Split policy: sample_holdout, ratio_by_group, group_kfold, existing_file")
    p.add_argument("--train-samples", default=None)
    p.add_argument("--val-samples", default=None)
    p.add_argument("--test-samples", default=None)
    p.add_argument("--group-column", default=None)
    p.add_argument("--ratios", default=None,
                   help="Comma-separated train,val,test ratios (e.g. 0.8,0.1,0.1)")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split-file", default=None)
    p.add_argument("--output-split-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    from daas.splits import build_split_membership, validate_split_membership, write_split_files

    if args.policy == "random_cell":
        print(
            "ERROR: DAAS does not generate random cell-level splits because they can cause "
            "sample/patient leakage. Use sample_holdout, ratio_by_group, group_kfold, "
            "or provide an external benchmark split with existing_file.",
            file=sys.stderr,
        )
        sys.exit(1)

    compiled_dir = Path(args.compiled_dir)
    manifest_path = compiled_dir / "manifest.parquet"
    if not manifest_path.exists():
        print(f"ERROR: manifest.parquet not found in {compiled_dir}", file=sys.stderr)
        sys.exit(1)
    manifest = pd.read_parquet(manifest_path)

    train_samples = [s.strip() for s in args.train_samples.split(",") if s.strip()] \
        if args.train_samples else None
    val_samples   = [s.strip() for s in args.val_samples.split(",") if s.strip()] \
        if args.val_samples else None
    test_samples  = [s.strip() for s in args.test_samples.split(",") if s.strip()] \
        if args.test_samples else None

    ratios = None
    if args.ratios:
        parts = [float(x.strip()) for x in args.ratios.split(",")]
        if len(parts) != 3:
            print("ERROR: --ratios must be three comma-separated floats", file=sys.stderr)
            sys.exit(1)
        ratios = tuple(parts)

    try:
        sm = build_split_membership(
            manifest=manifest,
            policy=args.policy,
            task=args.task,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            group_column=args.group_column,
            ratios=ratios,
            n_folds=args.n_folds,
            seed=args.seed,
            split_file=args.split_file,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    warnings = validate_split_membership(sm, manifest)
    for w in warnings:
        print(f"WARNING: {w}")

    output_dir = Path(args.output_split_dir)
    write_split_files(sm, output_dir, task=args.task)

    print(f"[make_split] Split files written to {output_dir}")
    counts = sm.groupby("split").size().to_dict()
    for split, n in sorted(counts.items()):
        print(f"  {split}: {n} cells")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run make_split CLI tests**

```bash
cd skills/daas-compiler && python -m pytest tests/test_make_split_cli.py -v
```

Expected: all PASS

- [ ] **Step 6.5: Commit**

```bash
git add skills/daas-compiler/scripts/make_split.py skills/daas-compiler/tests/test_make_split_cli.py
git commit -m "feat(cli): add scripts/make_split.py for standalone split generation without shard rewrite"
```

---

## Task 7: Update `tests/conftest.py` — Add `compiled_dir` fixture

**Files:**
- Modify: `skills/daas-compiler/tests/conftest.py`

This task must be done BEFORE Tasks 3–6 tests can run. Insert the `compiled_dir` fixture shown in the "Shared Test Fixture" section above.

- [ ] **Step 7.1: Read current conftest.py to understand imports**

Look at `tests/conftest.py` — it already imports `io, json, struct, tarfile, tempfile, anndata, numpy, pandas, pytest, PIL, scipy`.

- [ ] **Step 7.2: Add `compiled_dir` fixture and new import**

Append to `tests/conftest.py`:

```python
from daas.genes import gene_panel_sha256 as _gene_panel_sha256


@pytest.fixture
def compiled_dir(tmp_path, synthetic_sample):
    """Minimal compiled/ directory with bundled WDS shard, gene_panel, manifest."""
    import json, tarfile, io
    import numpy as np
    from scipy.sparse import csr_matrix
    import anndata

    cd = tmp_path / "compiled"
    cd.mkdir()

    n_genes   = synthetic_sample["n_genes"]
    n_cells   = synthetic_sample["n_cells"]
    sample_id = synthetic_sample["sample_id"]
    gene_panel = [f"gene_{i}" for i in range(n_genes)]
    sha = _gene_panel_sha256(gene_panel)
    (cd / "gene_panel.json").write_text(json.dumps(gene_panel))
    (cd / "gene_panel.sha256").write_text(sha)

    sample_dir = cd / sample_id
    sample_dir.mkdir()
    shard_path = sample_dir / "shard-000000.tar"

    meta_rows = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_cells):
            key = f"{i:09d}"
            jpg = _make_jpg()
            indices = np.array([0, 1], dtype=np.int32)
            values  = np.array([1.0, 2.0], dtype=np.float32)
            npz_buf = io.BytesIO()
            np.savez(npz_buf, indices=indices, values=values)
            meta_bytes = json.dumps({
                "global_idx": i, "sample_id": sample_id,
                "cell_id": f"cell_{i}", "task": "he2st",
                "n_genes": n_genes, "gene_panel_sha256": sha,
            }).encode()
            for ext, data in [(".jpg", jpg), (".expr.npz", npz_buf.getvalue()), (".json", meta_bytes)]:
                ti = tarfile.TarInfo(name=f"{key}{ext}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            meta_rows.append({
                "global_idx": i, "sample_id": sample_id,
                "cell_id": f"cell_{i}", "sample_key": key,
                "global_key": key, "shard_path": str(shard_path),
            })
    shard_path.write_bytes(buf.getvalue())

    bm = pd.DataFrame(meta_rows)
    bm.to_parquet(cd / "bundled_manifest.parquet", index=False)

    X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))
    obs = pd.DataFrame({"sample_id": [sample_id]*n_cells,
                        "cell_id": [f"cell_{i}" for i in range(n_cells)]})
    obs.index = [f"{i:09d}" for i in range(n_cells)]
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=gene_panel))
    adata.write_h5ad(cd / "expression.h5ad")

    mf = pd.DataFrame(meta_rows)[["global_idx", "sample_id", "cell_id", "shard_path", "sample_key"]]
    mf.to_parquet(cd / "manifest.parquet", index=False)

    return cd
```

- [ ] **Step 7.3: Run all tests to verify nothing broke**

```bash
cd skills/daas-compiler && python -m pytest tests/ -q 2>&1 | tail -20
```

Expected: all pre-existing tests still pass, new tests pass.

- [ ] **Step 7.4: Commit**

```bash
git add skills/daas-compiler/tests/conftest.py
git commit -m "test(conftest): add compiled_dir fixture for HE2ST task adapter tests"
```

---

## Task 8: Documentation updates

**Files:**
- Modify: `skills/daas-compiler/README.md`
- Modify: `skills/daas-compiler/SKILL.md`
- Modify: `skills/daas-compiler/usage-guide.md`
- Verify (no change expected): `references/training-ready-contract.md`, `references/task-adapters.md`, `references/agent-contract.md`

- [ ] **Step 8.1: Read current README.md to find insertion point**

Check the existing README for sections about training-ready or task-ready output.

- [ ] **Step 8.2: Add L4 / HE2ST task-adapter section to README.md**

Find the end of the existing "Usage" or "Compile" section and insert after it:

```markdown
## L4 Task-Ready Packaging: HE2ST

After compiling a dataset (L3), use `make_task_dataset.py` to produce a fully
training-ready (L4) HE2ST package:

```bash
python3 scripts/make_task_dataset.py \
    --compiled-dir /data/compiled \
    --task he2st \
    --output /data/he2st_task \
    --format webdataset \
    --split-policy sample_holdout \
    --train-samples A_001,A_002 \
    --val-samples A_004 \
    --test-samples A_005 \
    --target-normalization raw_counts
```

**Splits are metadata, not physical shard partitions.**
All cells are stored in `data/shard-*.tar`. The loader selects train/val/test cells
at runtime by reading `splits/split_membership.parquet`. Default output does NOT
create physical `train/`, `val/`, or `test/` shard directories.

**DAAS never generates random cell-level splits.**
Generated splits are assigned at the sample level (`sample_holdout`) or group level
(`ratio_by_group`, `group_kfold`). Use `--policy existing_file` to supply an external
benchmark split. If you need to generate or update split metadata later without
rewriting shards, use `scripts/make_split.py`.

### Split-pending mode

If you want to package the task skeleton before deciding on splits:

```bash
python3 scripts/make_task_dataset.py \
    --compiled-dir /data/compiled \
    --task he2st \
    --output /data/he2st_task \
    --format webdataset \
    --defer-split
```

A split-pending package is **not** fully training-ready (`training_ready: false`).
Generate split metadata later with:

```bash
python3 scripts/make_split.py \
    --compiled-dir /data/compiled \
    --task he2st \
    --policy sample_holdout \
    --train-samples A_001,A_002 \
    --val-samples A_004 \
    --output-split-dir /data/he2st_task/splits
```

### Load at training time

```python
from daas.loaders.he2st_dataset import HE2STDataset

ds = HE2STDataset.from_config("/data/he2st_task/loader_config.yaml", split="train")
item = ds[0]
# item["image"] → PIL.Image (224×224 RGB)
# item["expression"] → np.float32 array, shape (n_genes,)
```
```

- [ ] **Step 8.3: Add HE2ST training-ready CLI entry to `SKILL.md`**

In `SKILL.md`, find the scripts section and add `make_task_dataset.py` and `make_split.py` under the list of available scripts.

- [ ] **Step 8.4: Add training-ready section to `usage-guide.md`**

Add a new section (after the existing Compile section):

```markdown
## HE2ST Training-Ready Packaging (L4)

After L3 compile, produce a fully training-ready HE2ST dataset:

### With sample holdout split

```bash
python3 scripts/make_task_dataset.py \
    --compiled-dir /data/compiled --task he2st \
    --output /data/he2st_task --format webdataset \
    --split-policy sample_holdout \
    --train-samples A_001,A_002 --val-samples A_004
```

### With group-level ratio split

```bash
python3 scripts/make_task_dataset.py \
    --compiled-dir /data/compiled --task he2st \
    --output /data/he2st_task --format webdataset \
    --split-policy ratio_by_group \
    --group-column patient_id --ratios 0.8,0.1,0.1 --seed 42
```

### Split-pending (decide splits later)

```bash
python3 scripts/make_task_dataset.py \
    --compiled-dir /data/compiled --task he2st \
    --output /data/he2st_task --defer-split
```

Then later:

```bash
python3 scripts/make_split.py \
    --compiled-dir /data/compiled --task he2st \
    --policy sample_holdout \
    --train-samples A_001,A_002 --val-samples A_004 \
    --output-split-dir /data/he2st_task/splits
```

### Key rules

- WDS shards are **not** physically partitioned by split. The loader reads `splits/split_membership.parquet` at runtime.
- DAAS **never** generates random cell-level splits. Splits are sample-level or group-level.
- Changing split metadata does NOT require rewriting shards.
- A split-pending package has `training_ready: false` in `dataset_card.json`.
```

- [ ] **Step 8.5: Verify reference docs are consistent**

Read `references/training-ready-contract.md`, `references/task-adapters.md`, `references/agent-contract.md` and confirm they already contain the correct split policy language. No edits needed if they are already correct (they appear to be up to date from prior work).

- [ ] **Step 8.6: Commit docs**

```bash
git add skills/daas-compiler/README.md skills/daas-compiler/SKILL.md skills/daas-compiler/usage-guide.md
git commit -m "docs: add L4 HE2ST training-ready section, split policy rules, make_task_dataset.py and make_split.py usage"
```

---

## Task 9: Run full test suite and verify

- [ ] **Step 9.1: Compile check**

```bash
cd skills/daas-compiler && python -m compileall daas/ -q
```

Expected: no output (no errors)

- [ ] **Step 9.2: Run full test suite**

```bash
cd skills/daas-compiler && python -m pytest tests/ -q 2>&1 | tail -30
```

Expected: all tests pass, including pre-existing tests.

- [ ] **Step 9.3: Verify split policy rejection**

```bash
cd skills/daas-compiler && python -c "
from daas.splits import build_split_membership
import pandas as pd
m = pd.DataFrame({'global_idx':[0],'sample_id':['A'],'cell_id':['c0']})
try:
    build_split_membership(m, policy='random_cell', task='he2st')
except ValueError as e:
    print('OK:', e)
"
```

Expected: `OK: DAAS does not generate random cell-level splits...`

- [ ] **Step 9.4: Verify split_pending guard**

```bash
cd skills/daas-compiler && python -c "
import tempfile, json, pathlib, yaml
tmp = pathlib.Path(tempfile.mkdtemp())
cfg = {'task':'he2st','training_ready_status':'split_pending','storage':{'manifest_path':str(tmp/'m.parquet'),'shard_glob':'','shard_path_column':'shard_path','sample_key_column':'global_key'},'gene_panel_path':str(tmp/'g.json'),'gene_panel_sha256':'','split':{'required':True,'status':'missing','split_membership_path':None,'split_column':'split','index_column':'global_idx','generated_at_level':'missing'},'target':{},'image':{},'runtime':{'split_argument_required':True}}
(tmp/'loader_config.yaml').write_text(yaml.dump(cfg))
from daas.loaders.he2st_dataset import HE2STDataset
try:
    HE2STDataset.from_config(tmp/'loader_config.yaml', split='train')
except RuntimeError as e:
    print('OK:', e)
"
```

Expected: `OK: This HE2ST task package is split_pending...`

- [ ] **Step 9.5: Final commit if any cleanup needed**

```bash
git add -p  # review any remaining changes
git commit -m "chore: final cleanup after HE2ST task adapter implementation"
```

---

## Self-Review: Spec Coverage Check

| Spec requirement | Task(s) | Status |
|---|---|---|
| `daas/tasks/he2st.py` — core HE2ST packaging | Task 3 | ✓ |
| `daas/splits.py` — build/validate/write/summarize split | Task 1 | ✓ |
| `daas/loaders/configs.py` — write loader/task config | Task 2 | ✓ |
| `scripts/make_task_dataset.py` CLI | Task 5 | ✓ |
| `scripts/make_split.py` CLI | Task 6 | ✓ |
| split_pending when defer_split | Task 3, 5 | ✓ |
| fail when no split and no defer | Task 3, 5 | ✓ |
| training_ready_status in card/report | Task 3 | ✓ |
| No physical train/val/test shard dirs | Task 3 tests | ✓ |
| random_cell policy rejected | Task 1 | ✓ |
| sample_holdout no leakage | Task 1 tests | ✓ |
| ratio_by_group no group leakage | Task 1 tests | ✓ |
| group_kfold each group one fold | Task 1 tests | ✓ |
| existing_file global_idx-level warns | Task 1 tests | ✓ |
| make_split does not rewrite shards | Task 6 tests | ✓ |
| loader filters by split at runtime | Task 4 | ✓ |
| split_pending → loader raises | Task 4 tests | ✓ |
| gene_panel copied + sha256 referenced | Task 3 tests | ✓ |
| WDS JSON does not include split by default | compile_dataset.py already correct | ✓ (existing) |
| loader_config says split.status=missing | Task 2, 3 | ✓ |
| docs — splits are metadata not shards | Task 8 | ✓ |
| docs — no random cell split | Task 8 | ✓ |
| docs — split_pending not fully training-ready | Task 8 | ✓ |
| validation_report.json fields | Task 3 | ✓ |
| dataset_card.json fields | Task 3 | ✓ |
| generated split_membership derived from sample/group | Task 1 tests | ✓ |

No gaps found.
