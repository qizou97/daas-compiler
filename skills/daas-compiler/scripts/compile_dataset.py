"""
Compile per-sample directories into a unified training dataset.
Usage:
  python3 scripts/compile_dataset.py \
      --per-sample-dir /data/out \
      --output         /data/compiled
"""
import argparse, time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-sample-dir", required=True,
                   help="目录，每个子目录为一个样本")
    p.add_argument("--output", required=True,
                   help="compiled 输出目录")
    return p.parse_args()


def main():
    args        = parse_args()
    per_sample  = Path(args.per_sample_dir)
    compiled    = Path(args.output)
    compiled.mkdir(parents=True, exist_ok=True)
    t0          = time.time()

    sample_dirs = sorted(d for d in per_sample.iterdir()
                         if d.is_dir()
                         and (d / "manifest.parquet").exists()
                         and (d / "expression.h5ad").exists())
    assert sample_dirs, f"No valid sample dirs found in {per_sample}"
    print(f"[compile] Found {len(sample_dirs)} samples: "
          f"{[d.name for d in sample_dirs]}")

    # ── 2a: Merge manifests ───────────────────────────────────────────────────
    print("[1/3] Merging manifests …")
    parts = [pd.read_parquet(d / "manifest.parquet") for d in sample_dirs]
    global_manifest = pd.concat(parts, ignore_index=True)
    global_manifest["global_idx"] = np.arange(len(global_manifest),
                                               dtype=np.int64)
    global_manifest.to_parquet(compiled / "manifest.parquet", index=False)
    print(f"      {len(global_manifest)} cells total")

    # ── 2b: Merge h5ad with gene intersection ─────────────────────────────────
    print("[2/3] Merging expression h5ad …")
    adatas = [anndata.read_h5ad(d / "expression.h5ad") for d in sample_dirs]

    common_genes = adatas[0].var_names
    for a in adatas[1:]:
        common_genes = common_genes.intersection(a.var_names)
    n_genes_before = adatas[0].n_vars
    assert len(common_genes) > 0, \
        "Gene intersection is empty — check that all samples share at least one gene"
    print(f"      genes: {n_genes_before} → {len(common_genes)} "
          f"(intersection across {len(adatas)} samples)")

    combined = anndata.concat(
        [a[:, common_genes] for a in adatas],
        axis=0, merge="same"
    )
    combined.obs_names_make_unique()

    # Verify row count matches manifest
    assert combined.n_obs == len(global_manifest), (
        f"h5ad rows ({combined.n_obs}) != manifest rows ({len(global_manifest)})"
    )

    print("[3/3] Writing compiled h5ad …")
    combined.write_h5ad(compiled / "expression.h5ad")

    elapsed = time.time() - t0
    n_shards = sum(1 for d in sample_dirs for _ in d.glob("shard-*.tar"))
    print(f"""
{'='*60}
  COMPILE COMPLETE — {len(sample_dirs)} samples, {len(global_manifest)} cells,
                     {len(common_genes)} genes, {n_shards} shards
{'─'*60}
from daas.dataset import CellPatchDataset
from torch.utils.data import DataLoader

train_samples = [...]   # from splits/train.json

ds = CellPatchDataset(
    manifest_path = "{compiled}/manifest.parquet",
    h5ad_path     = "{compiled}/expression.h5ad",
    sample_ids    = train_samples,
)
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8)
batch  = next(iter(loader))
# batch["image"].shape      → (256, 3, 224, 224)
# batch["expression"].shape → (256, {len(common_genes)})
{'='*60}
  Total time: {elapsed:.1f}s
""")


if __name__ == "__main__":
    main()
