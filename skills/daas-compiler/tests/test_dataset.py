import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from daas.dataset import CellPatchDataset


def test_len(synthetic_sample):
    ds = CellPatchDataset(
        manifest_path = synthetic_sample["dir"] / "manifest.parquet",
        h5ad_path     = synthetic_sample["dir"] / "expression.h5ad",
    )
    assert len(ds) == synthetic_sample["n_cells"]


def test_getitem_shapes(synthetic_sample):
    ds = CellPatchDataset(
        manifest_path = synthetic_sample["dir"] / "manifest.parquet",
        h5ad_path     = synthetic_sample["dir"] / "expression.h5ad",
    )
    item = ds[0]
    assert item["image"].size == (224, 224)    # PIL before transform
    assert item["expression"].shape == (synthetic_sample["n_genes"],)
    assert item["expression"].dtype == np.float32
    assert item["cell_id"] == "cell_0"
    assert item["sample_id"] == synthetic_sample["sample_id"]


def test_sample_id_filter(synthetic_sample):
    ds = CellPatchDataset(
        manifest_path = synthetic_sample["dir"] / "manifest.parquet",
        h5ad_path     = synthetic_sample["dir"] / "expression.h5ad",
        sample_ids    = ["NONEXISTENT"],
    )
    assert len(ds) == 0


def test_transform_applied(synthetic_sample):
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor()])
    ds = CellPatchDataset(
        manifest_path = synthetic_sample["dir"] / "manifest.parquet",
        h5ad_path     = synthetic_sample["dir"] / "expression.h5ad",
        transform     = transform,
    )
    item = ds[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)


def test_dataloader_batch(synthetic_sample):
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor()])
    ds = CellPatchDataset(
        manifest_path = synthetic_sample["dir"] / "manifest.parquet",
        h5ad_path     = synthetic_sample["dir"] / "expression.h5ad",
        transform     = transform,
    )
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
    batch  = next(iter(loader))
    assert batch["image"].shape == (3, 3, 224, 224)
    assert batch["expression"].shape == (3, synthetic_sample["n_genes"])
