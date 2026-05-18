# tests/test_viz.py
import io, json, tarfile
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
