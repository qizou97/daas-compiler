"""
Pure-webdataset example: load compiled/wds/ shards with the `webdataset`
library only — no `daas.dataset.BundledCellPatchDataset`, no `anndata`,
no `tarfile` calls of your own.

Prerequisites
-------------
1. Run compile with --bundle-wds, so the shards exist:

       python3 scripts/compile_dataset.py \
           --per-sample-dir /data/out \
           --output         /data/compiled \
           --bundle-wds

2. Install the optional dependency:

       pip install webdataset>=0.2

What this shows
---------------
- Reading multi-shard tars via brace-expansion URL syntax.
- Decoding the three companion files (.jpg, .expr.npz, .json) per cell.
- Reconstructing the dense gene-expression vector from the sparse
  (indices, values) pair stored in .expr.npz.
- Wiring the dataset into a PyTorch DataLoader for training.
"""
import io
import json
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from PIL import Image


# ── 1. Configure paths ────────────────────────────────────────────────────────
COMPILED_DIR = Path("/data/compiled")  # ← change to your --output dir


# ── 2. Read the gene panel once ───────────────────────────────────────────────
# Column order of indices in every .expr.npz matches this list.
with open(COMPILED_DIR / "gene_panel.json") as f:
    GENES = json.load(f)
N_GENES = len(GENES)
print(f"Loaded {N_GENES} genes from {COMPILED_DIR / 'gene_panel.json'}")


# ── 3. Custom decoders for our two non-standard members ───────────────────────
def decode_expr_npz(data: bytes) -> np.ndarray:
    """`.expr.npz` → dense float32 vector of shape (N_GENES,)."""
    npz = np.load(io.BytesIO(data))
    expr = np.zeros(N_GENES, dtype=np.float32)
    idx = npz["indices"]
    if len(idx):
        expr[idx] = npz["values"]
    return expr


def decode_json(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


# ── 4. Build the pipeline ─────────────────────────────────────────────────────
# Shards live in per-sample subdirs: compiled/{sample_id}/shard-NNNNNN.tar
# Collect all shards across all sample subdirs (sorted for reproducibility).
shards = sorted(COMPILED_DIR.rglob("shard-*.tar"))
assert shards, f"No shards found under {COMPILED_DIR}"
n_shards = len(shards)
# Pass the full list of shard paths directly — WebDataset accepts a list.
url_pattern = [str(s) for s in shards]
print(f"Pipeline source: {n_shards} shards across "
      f"{len({s.parent for s in shards})} sample dirs")

# `.decode("pil")` decodes any .jpg / .jpeg / .png to a PIL.Image.
# Custom handlers for our two extra file types are registered via tuples
# of `(extension, handler)`. The library matches by file suffix.
dataset = (
    wds.WebDataset(url_pattern, shardshuffle=False)
    .decode(
        "pil",
        wds.handle_extension("expr.npz", decode_expr_npz),
        wds.handle_extension("json", decode_json),
    )
    .to_tuple("jpg", "expr.npz", "json")
    .map_tuple(
        lambda img: img,                          # PIL.Image; add your transform here
        lambda expr: torch.from_numpy(expr),      # float32 tensor (N_GENES,)
        lambda meta: meta,                        # dict with cell_id, sample_id, ...
    )
)


# ── 5. DataLoader for training ────────────────────────────────────────────────
# `wds.WebLoader` works the same as torch's DataLoader but understands
# WebDataset's worker-sharding scheme.
loader = wds.WebLoader(
    dataset.batched(32),   # batch inside the pipeline so workers stream evenly
    num_workers=4,
    batch_size=None,       # batching already done by `.batched(32)`
)

# Smoke-iterate one batch
images, exprs, metas = next(iter(loader))
print(f"images:  {type(images).__name__}, batch size {len(images)}")
print(f"exprs:   {exprs.shape}  dtype={exprs.dtype}")
print(f"metas[0]: {metas[0]}")


# ── 6. Training loop stub ────────────────────────────────────────────────────
# Uncomment and plug in your model + loss when ready.
#
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torchvision.transforms as T
#
# transform = T.Compose([T.ToTensor(), T.Resize(224)])
#
# def pil_to_tensor(img):
#     return transform(img)
#
# dataset = (
#     wds.WebDataset(url_pattern, shardshuffle=True)
#     .shuffle(1000)
#     .decode(
#         "pil",
#         wds.handle_extension("expr.npz", decode_expr_npz),
#         wds.handle_extension("json", decode_json),
#     )
#     .to_tuple("jpg", "expr.npz", "json")
#     .map_tuple(pil_to_tensor, torch.from_numpy, lambda m: m)
#     .batched(32)
# )
#
# loader = wds.WebLoader(dataset, num_workers=4, batch_size=None)
# model = nn.Linear(3 * 224 * 224, N_GENES).cuda()
# opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
#
# for epoch in range(10):
#     for images, exprs, _ in loader:
#         images = images.cuda(non_blocking=True)
#         exprs  = exprs.cuda(non_blocking=True)
#         pred = model(images.flatten(1))
#         loss = nn.functional.mse_loss(pred, exprs)
#         opt.zero_grad(); loss.backward(); opt.step()
