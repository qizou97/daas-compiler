from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

__all__ = ["write_loader_config", "write_task_config"]


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
    """
    Write a loader configuration YAML file.

    Args:
        output_path: Path to write the YAML file to.
        task: Task name (e.g., "he2st").
        training_ready_status: Status of training readiness ("training_ready", "split_pending", "validation_failed").
        shard_path_column: Column name for shard paths in manifest.
        sample_key_column: Column name for sample keys.
        manifest_path: Path to the bundled manifest parquet file.
        shard_glob: Glob pattern for shard files.
        gene_panel_path: Path to the gene panel JSON file.
        gene_panel_sha256: SHA256 hash of the gene panel.
        split_membership_path: Path to split membership file, or None if missing.
        split_status: Split status ("available" or "missing").
        generated_at_level: Level at which split was generated ("sample", "group", "external_global_idx", or "missing").
        patch_size: Size of patches (default 224).
        mpp: Microns per pixel, or None.
        normalization: Normalization method (default "raw_counts").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


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
    """
    Write a task configuration YAML file.

    Args:
        output_path: Path to write the YAML file to.
        task: Task name (e.g., "he2st").
        n_genes: Number of genes in the panel.
        gene_panel_path: Path to the gene panel JSON file.
        gene_panel_sha256: SHA256 hash of the gene panel.
        input_modality: Input modality (default "he_image").
        target_modality: Target modality (default "gene_expression").
        patch_size: Size of patches (default 224).
        mpp: Microns per pixel, or None.
        normalization: Normalization method (default "raw_counts").
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
