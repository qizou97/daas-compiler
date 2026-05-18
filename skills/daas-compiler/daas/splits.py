"""
daas/splits.py — split membership utilities for DaaS compiler.

Supports the following split policies:
  - sample_holdout:  explicit per-sample assignment
  - ratio_by_group:  group-level random ratio split
  - group_kfold:     group-level k-fold cross-validation
  - existing_file:   load pre-built split from parquet/csv/tsv
  - defer_split:     intentionally unsupported (raises ValueError)
  - random_cell:     intentionally rejected (leakage risk)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "build_split_membership",
    "validate_split_membership",
    "write_split_files",
    "summarize_split_report",
    "SUPPORTED_POLICIES",
]

SUPPORTED_POLICIES: set[str] = {
    "sample_holdout",
    "ratio_by_group",
    "group_kfold",
    "existing_file",
    "defer_split",
}

_REQUIRED_MANIFEST_COLS = {"global_idx", "sample_id", "cell_id"}


# ── helpers ──────────────────────────────────────────────────────────────────


def _check_manifest(manifest: pd.DataFrame) -> None:
    missing = _REQUIRED_MANIFEST_COLS - set(manifest.columns)
    if missing:
        raise ValueError(
            f"manifest is missing required columns: {sorted(missing)}"
        )


# ── policy implementations ────────────────────────────────────────────────────


def _build_sample_holdout(
    manifest: pd.DataFrame,
    task: str,
    train_samples: list[str],
    val_samples: list[str],
    test_samples: list[str],
) -> pd.DataFrame:
    all_assigned: list[str] = []
    for lst, name in [
        (train_samples, "train_samples"),
        (val_samples, "val_samples"),
        (test_samples, "test_samples"),
    ]:
        if lst is None:
            raise ValueError(
                f"sample_holdout policy requires {name} to be provided"
            )
        all_assigned.extend(lst)

    # Check for overlaps
    seen: set[str] = set()
    for split_name, lst in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        for s in lst:
            if s in seen:
                raise ValueError(
                    f"sample_id {s!r} appears in multiple splits — leakage detected"
                )
            seen.add(s)

    # Check all manifest samples are assigned
    manifest_samples = set(manifest["sample_id"].unique())
    assigned_samples = set(all_assigned)
    unassigned = manifest_samples - assigned_samples
    if unassigned:
        raise ValueError(
            f"Some manifest sample_ids are not assigned to any split: {sorted(unassigned)}"
        )

    sample_to_split: dict[str, str] = {}
    for sid in train_samples:
        sample_to_split[sid] = "train"
    for sid in val_samples:
        sample_to_split[sid] = "val"
    for sid in test_samples:
        sample_to_split[sid] = "test"

    sm = manifest[["global_idx", "sample_id", "cell_id"]].copy()
    sm["split"] = sm["sample_id"].map(sample_to_split)
    sm["task"] = task
    sm["split_policy"] = "sample_holdout"
    sm["generated_at_level"] = "sample"
    return sm.reset_index(drop=True)


def _build_ratio_by_group(
    manifest: pd.DataFrame,
    task: str,
    group_column: str,
    ratios: tuple[float, float, float],
    seed: int | None,
) -> pd.DataFrame:
    if group_column is None:
        raise ValueError("ratio_by_group policy requires group_column")
    if ratios is None:
        raise ValueError("ratio_by_group policy requires ratios")
    if group_column not in manifest.columns:
        raise ValueError(
            f"group_column {group_column!r} not found in manifest columns"
        )
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    train_frac, val_frac, _test_frac = ratios
    rng = np.random.default_rng(seed)

    groups = np.array(sorted(manifest[group_column].unique()))
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, round(n * train_frac))
    n_val = max(0, round(n * val_frac))
    # ensure we don't exceed total
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    group_to_split: dict[Any, str] = {}
    for i, g in enumerate(groups):
        if i < n_train:
            group_to_split[g] = "train"
        elif i < n_train + n_val:
            group_to_split[g] = "val"
        else:
            group_to_split[g] = "test"

    sm = manifest[["global_idx", "sample_id", "cell_id"]].copy()
    sm["group_id"] = manifest[group_column].values
    sm["split"] = sm["group_id"].map(group_to_split)
    sm["task"] = task
    sm["split_policy"] = "ratio_by_group"
    sm["seed"] = seed
    sm["generated_at_level"] = "group"
    return sm.reset_index(drop=True)


def _build_group_kfold(
    manifest: pd.DataFrame,
    task: str,
    group_column: str,
    n_folds: int,
    seed: int | None,
    fold: int | None,
) -> pd.DataFrame:
    if group_column is None:
        raise ValueError("group_kfold policy requires group_column")
    if n_folds is None or n_folds < 2:
        raise ValueError("group_kfold policy requires n_folds >= 2")
    if group_column not in manifest.columns:
        raise ValueError(
            f"group_column {group_column!r} not found in manifest columns"
        )

    rng = np.random.default_rng(seed)
    groups = np.array(sorted(manifest[group_column].unique()))
    rng.shuffle(groups)

    group_to_fold: dict[Any, int] = {g: i % n_folds for i, g in enumerate(groups)}

    sm = manifest[["global_idx", "sample_id", "cell_id"]].copy()
    sm["group_id"] = manifest[group_column].values
    sm["fold"] = sm["group_id"].map(group_to_fold)

    if fold is not None:
        sm["split"] = sm["fold"].apply(lambda f: "val" if f == fold else "train")
    else:
        sm["split"] = sm["fold"].apply(lambda f: f"fold_{f}")

    sm["task"] = task
    sm["split_policy"] = "group_kfold"
    sm["seed"] = seed
    sm["generated_at_level"] = "group"
    return sm.reset_index(drop=True)


def _build_existing_file(
    manifest: pd.DataFrame,
    task: str,
    split_file: str | Path,
) -> pd.DataFrame:
    if split_file is None:
        raise ValueError("existing_file policy requires split_file")

    path = Path(split_file)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        ext_df = pd.read_parquet(path)
    elif suffix == ".csv":
        ext_df = pd.read_csv(path)
    elif suffix == ".tsv":
        ext_df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(
            f"Unsupported split_file format {suffix!r}. Use .parquet, .csv, or .tsv"
        )

    if "split" not in ext_df.columns:
        raise ValueError("split_file must have a 'split' column")

    has_global_idx = "global_idx" in ext_df.columns
    has_sample_id = "sample_id" in ext_df.columns

    if has_global_idx and not has_sample_id:
        # global_idx join
        merged = manifest[["global_idx", "sample_id", "cell_id"]].merge(
            ext_df[["global_idx", "split"]], on="global_idx", how="left"
        )
        unassigned = merged["split"].isna()
        if unassigned.any():
            n = unassigned.sum()
            raise ValueError(
                f"{n} manifest cells have no split assignment after join on global_idx"
            )
        merged["task"] = task
        merged["split_policy"] = "existing_file"
        merged["generated_at_level"] = "external_global_idx"
        merged["leakage_warning"] = (
            "This split is global_idx-level and was externally provided. "
            "DAAS did not generate a cell-level split. "
            "Validate leakage assumptions before using it."
        )
        return merged.reset_index(drop=True)

    elif has_sample_id:
        # sample_id join
        cols = ["sample_id", "split"]
        if has_global_idx:
            cols = ["global_idx", "sample_id", "split"]
        merged = manifest[["global_idx", "sample_id", "cell_id"]].merge(
            ext_df[cols].drop_duplicates(subset=["sample_id"]),
            on="sample_id",
            how="left",
        )
        # prefer ext_df global_idx only if it's a sample-level file
        if "global_idx_y" in merged.columns:
            merged = merged.drop(columns=["global_idx_y"]).rename(
                columns={"global_idx_x": "global_idx"}
            )
        unassigned = merged["split"].isna()
        if unassigned.any():
            n = unassigned.sum()
            raise ValueError(
                f"{n} manifest cells have no split assignment after join on sample_id"
            )
        merged["task"] = task
        merged["split_policy"] = "existing_file"
        merged["generated_at_level"] = "sample"
        return merged.reset_index(drop=True)

    else:
        raise ValueError(
            "split_file must have either 'global_idx' or 'sample_id' column for joining"
        )


# ── public API ────────────────────────────────────────────────────────────────


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
    """Build a split membership DataFrame from a manifest and a policy.

    Parameters
    ----------
    manifest:
        Must have columns: global_idx, sample_id, cell_id.
    policy:
        One of SUPPORTED_POLICIES (or "random_cell" which is always rejected).
    task:
        Task label string (stored in the 'task' column).

    Returns
    -------
    pd.DataFrame with at minimum columns:
        global_idx, sample_id, cell_id, split, task, split_policy, generated_at_level
    """
    _check_manifest(manifest)

    if policy == "random_cell":
        raise ValueError(
            "DAAS does not generate random cell-level splits because they can cause "
            "sample/patient leakage. Use sample_holdout, ratio_by_group, group_kfold, "
            "or provide an external benchmark split with existing_file."
        )

    if policy == "defer_split":
        raise ValueError(
            "policy=defer_split is not directly buildable; skip split generation"
        )

    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"Unknown split policy: {policy!r}. "
            f"Supported: {sorted(SUPPORTED_POLICIES)}"
        )

    if policy == "sample_holdout":
        return _build_sample_holdout(
            manifest, task, train_samples, val_samples, test_samples
        )

    if policy == "ratio_by_group":
        return _build_ratio_by_group(manifest, task, group_column, ratios, seed)

    if policy == "group_kfold":
        return _build_group_kfold(manifest, task, group_column, n_folds, seed, fold)

    if policy == "existing_file":
        return _build_existing_file(manifest, task, split_file)

    # unreachable
    raise ValueError(f"Unknown split policy: {policy!r}")  # pragma: no cover


def validate_split_membership(
    sm: pd.DataFrame, manifest: pd.DataFrame
) -> list[str]:
    """Validate a split membership DataFrame against the manifest.

    Raises ValueError on hard violations; returns list of warning strings.
    """
    warnings: list[str] = []

    # Required columns
    required = {"global_idx", "sample_id", "cell_id", "split", "task"}
    missing = required - set(sm.columns)
    if missing:
        raise ValueError(
            f"split_membership is missing required columns: {sorted(missing)}"
        )

    # Any global_idx in sm not in manifest → hard error
    manifest_idx = set(manifest["global_idx"])
    sm_idx = set(sm["global_idx"])
    extra = sm_idx - manifest_idx
    if extra:
        raise ValueError(
            f"split_membership contains global_idx values not in manifest: {sorted(extra)}"
        )

    # Duplicate global_idx in sm → hard error
    if sm["global_idx"].duplicated().any():
        dupes = sm.loc[sm["global_idx"].duplicated(keep=False), "global_idx"].unique()
        raise ValueError(
            f"Duplicate global_idx in split_membership: {sorted(dupes)}"
        )

    # Check leakage for sample-level policies
    policy = sm.get("split_policy", pd.Series(dtype=str)).iloc[0] if len(sm) else None

    if policy in {"sample_holdout", "ratio_by_group"}:
        sample_splits = (
            sm.groupby("sample_id")["split"].nunique()
        )
        leaky = sample_splits[sample_splits > 1].index.tolist()
        if leaky:
            raise ValueError(
                f"leakage detected: sample_ids in multiple splits: {sorted(leaky)}"
            )

    # Check leakage for group_kfold
    if policy == "group_kfold" and "group_id" in sm.columns and "fold" in sm.columns:
        group_folds = sm.groupby("group_id")["fold"].nunique()
        leaky_groups = group_folds[group_folds > 1].index.tolist()
        if leaky_groups:
            raise ValueError(
                f"leakage detected: group_ids in multiple folds: {sorted(leaky_groups)}"
            )

    # leakage_warning column → append to warnings
    if "leakage_warning" in sm.columns:
        non_null = sm["leakage_warning"].dropna()
        if len(non_null) > 0:
            unique_warnings = non_null.unique()
            for w in unique_warnings:
                warnings.append(str(w))

    return warnings


def write_split_files(
    sm: pd.DataFrame,
    output_dir: Path | str,
    task: str,
    prefix: str = "",
) -> dict[str, Path]:
    """Write split membership files to output_dir.

    Returns a dict of {name: Path} for all files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_ = f"{prefix}_" if prefix else ""

    paths: dict[str, Path] = {}

    # split_membership.parquet
    parquet_path = output_dir / f"{prefix_}split_membership.parquet"
    sm.to_parquet(parquet_path, index=False)
    paths["split_membership"] = parquet_path

    # train/val/test.json
    for split_name in ("train", "val", "test"):
        idx_list = sm.loc[sm["split"] == split_name, "global_idx"].tolist()
        json_path = output_dir / f"{prefix_}{split_name}.json"
        with json_path.open("w") as f:
            json.dump(idx_list, f)
        paths[split_name] = json_path

    # split_report.json
    report = summarize_split_report(sm, task)
    report_path = output_dir / f"{prefix_}split_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    paths["split_report"] = report_path

    return paths


def summarize_split_report(sm: pd.DataFrame, task: str) -> dict:
    """Build a summary report dict from a split membership DataFrame."""
    policy = sm["split_policy"].iloc[0] if "split_policy" in sm.columns and len(sm) else None
    gen_level = (
        sm["generated_at_level"].iloc[0]
        if "generated_at_level" in sm.columns and len(sm)
        else None
    )
    seed = sm["seed"].iloc[0] if "seed" in sm.columns and len(sm) else None

    counts_by_split: dict[str, int] = (
        sm.groupby("split").size().to_dict()
    )

    per_sample_counts = (
        sm.groupby(["sample_id", "split"])
        .size()
        .reset_index(name="n_cells")
        .to_dict(orient="records")
    )

    report: dict = {
        "task": task,
        "split_policy": policy,
        "generated_at_level": gen_level,
        "seed": seed,
        "n_cells_total": len(sm),
        "counts_by_split": counts_by_split,
        "per_sample_counts": per_sample_counts,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    if "leakage_warning" in sm.columns:
        non_null = sm["leakage_warning"].dropna()
        if len(non_null) > 0:
            report["leakage_warning"] = non_null.iloc[0]

    return report
