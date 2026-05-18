import io
import json
import mmap
import tarfile
from collections import OrderedDict
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import issparse


class LRUMmapCache:
    """LRU cache of memory-mapped file handles. Thread-unsafe; use per-worker."""

    def __init__(self, maxsize: int = 128):
        self._cache: OrderedDict[str, mmap.mmap] = OrderedDict()
        self._files: dict[str, object] = {}
        self.maxsize = maxsize

    def get(self, path: str) -> mmap.mmap:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        if len(self._cache) >= self.maxsize:
            evict_path, mm = self._cache.popitem(last=False)
            mm.close()
            self._files.pop(evict_path).close()
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._files[path] = f
        self._cache[path] = mm
        return mm

    def close_all(self):
        for mm in self._cache.values():
            mm.close()
        for f in self._files.values():
            f.close()
        self._cache.clear()
        self._files.clear()


try:
    from torch.utils.data import Dataset as TorchDataset
    _TORCH_AVAILABLE = True
except ImportError:
    TorchDataset = object
    _TORCH_AVAILABLE = False


class CellPatchDataset(TorchDataset):
    """
    PyTorch Dataset for cell-centered HE patches + gene expression.

    Returns dict with keys: image, expression, cell_id, sample_id.
    image is PIL.Image (or transformed Tensor if transform provided).
    expression is np.float32 array of shape (n_genes,).
    """

    def __init__(self, manifest_path, h5ad_path,
                 sample_ids=None, transform=None,
                 mmap_cache_size: int = 128):
        # For large runs set mmap_cache_size ≈ ceil(total_shards / num_workers).
        # Default 128 is safe for ~1k shards; scale up for 72-sample full runs.
        self.manifest = pd.read_parquet(manifest_path,
                                        dtype_backend="numpy_nullable")
        # ensure sample_key stays as string
        self.manifest["sample_key"] = self.manifest["sample_key"].astype(str)

        if sample_ids is not None:
            self.manifest = self.manifest[
                self.manifest["sample_id"].isin(sample_ids)
            ].reset_index(drop=True)

        adata        = anndata.read_h5ad(h5ad_path)
        self.X       = adata.X           # scipy CSR, stays in memory
        self.genes   = adata.var_names.tolist()
        self._mmap   = LRUMmapCache(maxsize=mmap_cache_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]

        # image: zero-copy read from mmap tar
        mm       = self._mmap.get(str(row["shard_path"]))
        offset   = int(row["tar_offset"])
        size     = int(row["jpg_size"])
        jpg      = bytes(mm[offset : offset + size])
        img      = Image.open(io.BytesIO(jpg)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # expression: sparse or dense row → float32 1-D array
        global_idx = int(row["global_idx"])
        row_x = self.X[global_idx]
        expr = (row_x.toarray() if issparse(row_x) else np.array(row_x)).reshape(-1).astype(np.float32)

        return {
            "image":      img,
            "expression": expr,
            "cell_id":    str(row["cell_id"]),
            "sample_id":  str(row["sample_id"]),
        }

    def __del__(self):
        if hasattr(self, "_mmap"):
            self._mmap.close_all()


class BundledCellPatchDataset(TorchDataset):
    """PyTorch Dataset reading from compile_dataset.py's `--bundle-wds` output.

    Each tar entry contains:
      - {key}.jpg       — HE patch
      - {key}.expr.npz  — sparse expression: indices (int32) + values (float32)
      - {key}.json      — metadata (cell_id, sample_id, n_genes, …)

    Compared to CellPatchDataset:
      - No mmap (uses tarfile.extractfile per-cell)
      - No external h5ad — expression travels with the patch
      - Sample filtering via the bundled manifest.parquet
      - Slightly slower per-cell read (~2× vs mmap), but no per-worker
        mmap memory accumulation
    """

    def __init__(self, wds_dir, sample_ids=None, transform=None,
                 dense_expression: bool = True):
        wds_dir = Path(wds_dir)
        self.wds_dir = wds_dir
        self.manifest = pd.read_parquet(wds_dir / "manifest.parquet",
                                        dtype_backend="numpy_nullable")
        self.manifest["sample_key"] = self.manifest["sample_key"].astype(str)
        self.manifest["global_key"] = self.manifest["global_key"].astype(str)

        with open(wds_dir / "gene_panel.json") as f:
            self.genes = json.load(f)
        self.n_genes = len(self.genes)

        if sample_ids is not None:
            self.manifest = self.manifest[
                self.manifest["sample_id"].isin(sample_ids)
            ].reset_index(drop=True)

        self.transform = transform
        self.dense_expression = dense_expression
        # Per-instance (per-worker, when forked) tar handle cache
        self._tar_handles: dict[str, tarfile.TarFile] = {}
        self._tar_members: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.manifest)

    def _get_tar(self, shard_path: str):
        if shard_path not in self._tar_handles:
            tf = tarfile.open(shard_path, "r")
            self._tar_handles[shard_path] = tf
            self._tar_members[shard_path] = {m.name: m for m in tf.getmembers()}
        return self._tar_handles[shard_path], self._tar_members[shard_path]

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
        indices = npz["indices"]
        values = npz["values"]

        if self.dense_expression:
            expr = np.zeros(self.n_genes, dtype=np.float32)
            if len(indices):
                expr[indices] = values
        else:
            expr = (indices.astype(np.int32),
                    values.astype(np.float32),
                    self.n_genes)

        return {
            "image":      img,
            "expression": expr,
            "cell_id":    str(row["cell_id"]),
            "sample_id":  str(row["sample_id"]),
        }

    def close(self):
        for tf in self._tar_handles.values():
            tf.close()
        self._tar_handles.clear()
        self._tar_members.clear()

    def __del__(self):
        if hasattr(self, "_tar_handles"):
            self.close()
