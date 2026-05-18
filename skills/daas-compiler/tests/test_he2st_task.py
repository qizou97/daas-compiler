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
    assert cfg["split"]["required"] is True
    assert cfg["runtime"]["split_argument_required"] is True
    assert cfg["storage"]["format"] == "webdataset"


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
    assert cfg["split"]["split_membership_path"] is None


def test_write_task_config(tmp_path):
    write_task_config(
        output_path=tmp_path / "task_config.yaml",
        task="he2st",
        n_genes=313,
        gene_panel_path=str(tmp_path / "gene_panel.json"),
        gene_panel_sha256="abc123",
    )
    cfg = yaml.safe_load((tmp_path / "task_config.yaml").read_text())
    assert cfg["task"] == "he2st"
    assert cfg["n_genes"] == 313
    assert cfg["input_modality"] == "he_image"
    assert cfg["target_modality"] == "gene_expression"
