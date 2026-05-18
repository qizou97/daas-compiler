import mmap
from collections import OrderedDict

import io
import numpy as np
import pandas as pd
import anndata
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
