import io, json, struct, tarfile, tempfile
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.sparse import csr_matrix

from daas.genes import gene_panel_sha256 as _gene_panel_sha256


IDX_MAGIC      = b"CIDX0001"
IDX_RECORD_FMT = "<iQIQII"


def _make_jpg(size=(224, 224)) -> bytes:
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return buf.getvalue()


@pytest.fixture
def synthetic_sample(tmp_path):
    """Creates per-sample directory with 6 cells across 2 shards."""
    n_cells, n_genes, shard_size = 6, 10, 3
    sample_id = "TEST_001"
    sample_dir = tmp_path / sample_id
    sample_dir.mkdir()

    rows, tar_offsets, jpg_sizes = [], [], []

    for shard_no in range(2):
        shard_buf = []
        for i in range(shard_size):
            local_i = shard_no * shard_size + i
            sk       = f"{local_i:07d}"
            jpg      = _make_jpg()
            meta     = {"sample_index": local_i, "sample_key": sk,
                        "sample_id": sample_id, "cell_id": f"cell_{local_i}"}
            shard_buf.append((local_i, sk, jpg, json.dumps(meta).encode()))

        tar_path = sample_dir / f"shard-{shard_no:06d}.tar"
        idx_path = sample_dir / f"shard-{shard_no:06d}.idx"

        with tarfile.open(tar_path, "w") as tf:
            for si, sk, jpg, jmeta in shard_buf:
                for ext, data in [(".jpg", jpg), (".json", jmeta)]:
                    ti = tarfile.TarInfo(name=f"{sk}{ext}")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))

        with tarfile.open(tar_path, "r") as tf:
            members = {m.name: m for m in tf.getmembers()}
            with open(idx_path, "wb") as f:
                f.write(IDX_MAGIC)
                f.write(struct.pack("<I", shard_size))
                for si, sk, jpg, _ in shard_buf:
                    jm = members[f"{sk}.jpg"]
                    nm = members[f"{sk}.json"]
                    f.write(struct.pack(IDX_RECORD_FMT,
                                        si, jm.offset_data, jm.size,
                                        nm.offset_data, nm.size, 0))
                    tar_offsets.append(jm.offset_data)
                    jpg_sizes.append(jm.size)

        for si, sk, jpg, _ in shard_buf:
            rows.append({"sample_id": sample_id,
                         "sample_key": sk,
                         "cell_id": f"cell_{si}",
                         "shard_path": str(tar_path),
                         "tar_offset": tar_offsets[si],
                         "jpg_size": jpg_sizes[si],
                         "expr_row": si,
                         "global_idx": si})

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(sample_dir / "manifest.parquet", index=False)

    X = csr_matrix(np.random.rand(n_cells, n_genes).astype(np.float32))
    obs = pd.DataFrame({"sample_id": [sample_id]*n_cells,
                        "cell_id": [f"cell_{i}" for i in range(n_cells)]})
    obs.index = [f"{i:07d}" for i in range(n_cells)]
    adata = anndata.AnnData(X=X, obs=obs,
                            var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]))
    adata.write_h5ad(sample_dir / "expression.h5ad")

    return {"dir": sample_dir, "manifest": manifest,
            "n_cells": n_cells, "n_genes": n_genes, "sample_id": sample_id}


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
