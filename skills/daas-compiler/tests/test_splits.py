"""
tests/test_splits.py — TDD tests for daas/splits.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from daas.splits import (
    SUPPORTED_POLICIES,
    build_split_membership,
    summarize_split_report,
    validate_split_membership,
    write_split_files,
)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def manifest_6cells():
    return pd.DataFrame(
        {
            "global_idx": list(range(6)),
            "sample_id": ["A", "A", "B", "B", "C", "C"],
            "cell_id": [f"cell_{i}" for i in range(6)],
        }
    )


@pytest.fixture
def manifest_with_group(manifest_6cells):
    """6-cell manifest with a group column (A and B in group 'g1', C in 'g2')."""
    df = manifest_6cells.copy()
    df["patient"] = ["g1", "g1", "g1", "g1", "g2", "g2"]
    return df


# ── test 1: sample_holdout basic ──────────────────────────────────────────────


def test_sample_holdout_basic(manifest_6cells):
    """A,B→train, C→val; checks columns and split assignments."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )

    # Required columns
    for col in ("global_idx", "sample_id", "cell_id", "split", "task", "split_policy", "generated_at_level"):
        assert col in sm.columns, f"Missing column: {col}"

    assert sm["task"].unique().tolist() == ["gene_pred"]
    assert sm["split_policy"].unique().tolist() == ["sample_holdout"]
    assert sm["generated_at_level"].unique().tolist() == ["sample"]

    # Cells 0,1 (A) and 2,3 (B) → train
    assert set(sm.loc[sm["sample_id"].isin(["A", "B"]), "split"].unique()) == {"train"}
    # Cells 4,5 (C) → val
    assert set(sm.loc[sm["sample_id"] == "C", "split"].unique()) == {"val"}


# ── test 2: sample_holdout no leakage ─────────────────────────────────────────


def test_sample_holdout_no_leakage(manifest_6cells):
    """Each sample_id appears in exactly one split."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A"],
        val_samples=["B"],
        test_samples=["C"],
    )
    for sid in ["A", "B", "C"]:
        splits_for_sample = sm.loc[sm["sample_id"] == sid, "split"].unique()
        assert len(splits_for_sample) == 1, (
            f"sample {sid} found in multiple splits: {splits_for_sample}"
        )


# ── test 3: random_cell rejected ──────────────────────────────────────────────


def test_random_cell_rejected(manifest_6cells):
    """random_cell policy raises ValueError with correct message."""
    with pytest.raises(ValueError, match="DAAS does not generate random cell-level splits"):
        build_split_membership(manifest_6cells, policy="random_cell", task="gene_pred")


# ── test 4: ratio_by_group no leakage ─────────────────────────────────────────


def test_ratio_by_group_no_leakage(manifest_with_group):
    """No group_id appears in multiple splits."""
    sm = build_split_membership(
        manifest_with_group,
        policy="ratio_by_group",
        task="gene_pred",
        group_column="patient",
        ratios=(0.5, 0.25, 0.25),
        seed=42,
    )

    assert "group_id" in sm.columns
    assert sm["generated_at_level"].unique().tolist() == ["group"]

    # Each group_id in exactly one split
    group_split_counts = sm.groupby("group_id")["split"].nunique()
    assert (group_split_counts <= 1).all(), (
        f"Some groups appear in multiple splits: {group_split_counts[group_split_counts > 1]}"
    )


# ── test 5: group_kfold each group in one fold ───────────────────────────────


def test_group_kfold_each_group_one_fold(manifest_with_group):
    """Each group_id appears in exactly one fold."""
    sm = build_split_membership(
        manifest_with_group,
        policy="group_kfold",
        task="gene_pred",
        group_column="patient",
        n_folds=2,
        seed=0,
    )

    assert "group_id" in sm.columns
    assert "fold" in sm.columns
    assert sm["generated_at_level"].unique().tolist() == ["group"]

    # Each group_id → exactly one fold value
    group_fold_counts = sm.groupby("group_id")["fold"].nunique()
    assert (group_fold_counts == 1).all(), (
        f"Some groups appear in multiple folds: {group_fold_counts[group_fold_counts > 1]}"
    )

    # Splits are fold_0, fold_1, ... (no fold specified)
    for s in sm["split"].unique():
        assert s.startswith("fold_"), f"Unexpected split value: {s}"


# ── test 6: existing_file sample-level ────────────────────────────────────────


def test_existing_file_sample_level(manifest_6cells):
    """Sample-level split file: no leakage_warning; generated_at_level='sample'."""
    ext_data = pd.DataFrame(
        {
            "sample_id": ["A", "B", "C"],
            "split": ["train", "train", "val"],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "splits.csv"
        ext_data.to_csv(csv_path, index=False)

        sm = build_split_membership(
            manifest_6cells,
            policy="existing_file",
            task="gene_pred",
            split_file=csv_path,
        )

    assert sm["generated_at_level"].unique().tolist() == ["sample"]
    assert "leakage_warning" not in sm.columns or sm["leakage_warning"].isna().all()
    # A and B → train
    assert set(sm.loc[sm["sample_id"].isin(["A", "B"]), "split"].unique()) == {"train"}
    # C → val
    assert set(sm.loc[sm["sample_id"] == "C", "split"].unique()) == {"val"}


# ── test 7: existing_file global_idx-level emits leakage_warning ─────────────


def test_existing_file_global_idx_level_emits_leakage_warning(manifest_6cells):
    """Global-idx-level split file: leakage_warning is present and non-null."""
    ext_data = pd.DataFrame(
        {
            "global_idx": list(range(6)),
            "split": ["train", "train", "train", "train", "val", "val"],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "splits.csv"
        ext_data.to_csv(csv_path, index=False)

        sm = build_split_membership(
            manifest_6cells,
            policy="existing_file",
            task="gene_pred",
            split_file=csv_path,
        )

    assert "leakage_warning" in sm.columns
    assert sm["leakage_warning"].notna().all()
    assert sm["generated_at_level"].unique().tolist() == ["external_global_idx"]


# ── test 8: validate detects leakage ──────────────────────────────────────────


def test_validate_split_membership_detects_leakage(manifest_6cells):
    """Manually crafted leaky split (same sample in both train and val) → ValueError."""
    # Build a valid sm and then corrupt it
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )
    # Introduce leakage: flip only ONE cell of sample A to "val" while the other stays "train"
    # This means sample A now appears in both train and val → leakage
    sm = sm.copy()
    a_indices = sm.index[sm["sample_id"] == "A"].tolist()
    sm.loc[a_indices[0], "split"] = "val"  # one A cell → val, other stays train

    with pytest.raises(ValueError, match="leakage detected"):
        validate_split_membership(sm, manifest_6cells)


# ── test 9: write_split_files creates all outputs ────────────────────────────


def test_write_split_files_creates_all_outputs(manifest_6cells):
    """All 5 output files are created; train.json has correct global_idx list."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output"
        paths = write_split_files(sm, out_dir, task="gene_pred")

        # 5 files: split_membership, train, val, test, split_report
        assert len(paths) == 5
        for name, p in paths.items():
            assert p.exists(), f"Expected file {p} does not exist (key={name})"

        # train.json has global_idx for A and B cells (0,1,2,3)
        train_idx = json.loads(paths["train"].read_text())
        expected_train = sorted(
            sm.loc[sm["split"] == "train", "global_idx"].tolist()
        )
        assert sorted(train_idx) == expected_train

        # val.json has global_idx for C cells (4,5)
        val_idx = json.loads(paths["val"].read_text())
        expected_val = sorted(sm.loc[sm["split"] == "val", "global_idx"].tolist())
        assert sorted(val_idx) == expected_val

        # test.json is empty
        test_idx = json.loads(paths["test"].read_text())
        assert test_idx == []

        # split_membership.parquet is readable
        sm_read = pd.read_parquet(paths["split_membership"])
        assert len(sm_read) == len(sm)


# ── test 10: generated split is global_idx-level but derived from sample ─────


def test_generated_split_is_global_idx_level_but_derived_from_sample(manifest_6cells):
    """split_membership has global_idx column; all cells from same sample have same split."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A"],
        val_samples=["B"],
        test_samples=["C"],
    )

    # global_idx column present and covers all manifest rows
    assert "global_idx" in sm.columns
    assert set(sm["global_idx"].tolist()) == set(range(6))

    # All cells from same sample have the same split
    for sid in ["A", "B", "C"]:
        sample_splits = sm.loc[sm["sample_id"] == sid, "split"].unique()
        assert len(sample_splits) == 1, (
            f"Cells from sample {sid} have different splits: {sample_splits}"
        )

    # The split is derived from sample (generated_at_level="sample")
    assert sm["generated_at_level"].unique().tolist() == ["sample"]


# ── additional edge-case tests ────────────────────────────────────────────────


def test_sample_holdout_raises_on_overlap(manifest_6cells):
    """ValueError if the same sample_id appears in two split lists."""
    with pytest.raises(ValueError, match="leakage detected|appears in multiple"):
        build_split_membership(
            manifest_6cells,
            policy="sample_holdout",
            task="gene_pred",
            train_samples=["A", "B"],
            val_samples=["B", "C"],  # B is in both train and val
            test_samples=[],
        )


def test_sample_holdout_raises_on_unassigned(manifest_6cells):
    """ValueError if some manifest sample_ids are not assigned."""
    with pytest.raises(ValueError, match="not assigned"):
        build_split_membership(
            manifest_6cells,
            policy="sample_holdout",
            task="gene_pred",
            train_samples=["A"],  # B and C unassigned
            val_samples=[],
            test_samples=[],
        )


def test_unknown_policy_raises(manifest_6cells):
    """Unknown policy raises ValueError with supported policies listed."""
    with pytest.raises(ValueError, match="Unknown split policy"):
        build_split_membership(manifest_6cells, policy="made_up", task="gene_pred")


def test_defer_split_raises(manifest_6cells):
    """defer_split policy raises ValueError with correct message."""
    with pytest.raises(ValueError, match="defer_split is not directly buildable"):
        build_split_membership(manifest_6cells, policy="defer_split", task="gene_pred")


def test_group_kfold_with_fold_specified(manifest_with_group):
    """When fold is specified, split values are 'val' or 'train' only."""
    sm = build_split_membership(
        manifest_with_group,
        policy="group_kfold",
        task="gene_pred",
        group_column="patient",
        n_folds=2,
        seed=0,
        fold=0,
    )
    assert set(sm["split"].unique()).issubset({"train", "val"})


def test_summarize_split_report_structure(manifest_6cells):
    """summarize_split_report returns expected keys."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )
    report = summarize_split_report(sm, "gene_pred")
    for key in ("task", "split_policy", "generated_at_level", "seed", "n_cells_total",
                "counts_by_split", "per_sample_counts", "created_at"):
        assert key in report, f"Missing key in report: {key}"
    assert report["n_cells_total"] == 6
    assert report["counts_by_split"]["train"] == 4
    assert report["counts_by_split"]["val"] == 2


def test_write_split_files_with_prefix(manifest_6cells):
    """Prefix is applied to all output filenames."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A", "B"],
        val_samples=["C"],
        test_samples=[],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        paths = write_split_files(sm, out_dir, task="gene_pred", prefix="myrun")
        for path in paths.values():
            assert path.name.startswith("myrun_"), (
                f"Expected prefix 'myrun_' in {path.name}"
            )


def test_validate_missing_columns_raises(manifest_6cells):
    """Missing required column in sm → ValueError."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A"],
        val_samples=["B"],
        test_samples=["C"],
    )
    sm_bad = sm.drop(columns=["split"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_split_membership(sm_bad, manifest_6cells)


def test_validate_global_idx_not_in_manifest(manifest_6cells):
    """global_idx in sm not in manifest → ValueError."""
    sm = build_split_membership(
        manifest_6cells,
        policy="sample_holdout",
        task="gene_pred",
        train_samples=["A"],
        val_samples=["B"],
        test_samples=["C"],
    )
    sm_bad = sm.copy()
    sm_bad.loc[0, "global_idx"] = 9999
    with pytest.raises(ValueError, match="not in manifest"):
        validate_split_membership(sm_bad, manifest_6cells)


def test_validate_leakage_warning_returned(manifest_6cells):
    """leakage_warning column non-null → warning returned (not raised)."""
    ext_data = pd.DataFrame(
        {
            "global_idx": list(range(6)),
            "split": ["train", "train", "train", "train", "val", "val"],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "splits.csv"
        ext_data.to_csv(csv_path, index=False)
        sm = build_split_membership(
            manifest_6cells,
            policy="existing_file",
            task="gene_pred",
            split_file=csv_path,
        )

    warnings = validate_split_membership(sm, manifest_6cells)
    assert len(warnings) == 1
    assert "DAAS did not generate" in warnings[0]
