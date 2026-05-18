"""
Compile per-sample directories into a unified training dataset.
Usage:
  python3 scripts/compile_dataset.py \
      --per-sample-dir /data/out \
      --output         /data/compiled \
      [--bundle-wds] [--shard-size 500]

When --bundle-wds is set, also writes a self-contained WebDataset under
{output}/wds/ where each tar entry contains JPEG + sparse expression
(.expr.npz) + JSON metadata for one cell. Training from the bundled
output does not require mmap or the compiled h5ad.
"""
import argparse, io, json, tarfile, time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-sample-dir", required=True,
                   help="目录，每个子目录为一个样本")
    p.add_argument("--output", required=True,
                   help="compiled 输出目录")
    p.add_argument("--bundle-wds", action="store_true",
                   help="Also write {output}/wds/ — each cell bundled as "
                        "jpg + sparse expr.npz + json in a tar shard. No mmap "
                        "needed at training time.")
    p.add_argument("--shard-size", type=int, default=500,
                   help="Cells per bundled WDS tar shard (default: 500)")
    return p.parse_args()


def _flush_bundled_shard(shard_buf, shard_no, wds_dir):
    """Write one bundled tar shard with .jpg + .expr.npz + .json per cell."""
    tar_path = wds_dir / f"shard-{shard_no:06d}.tar"
    with tarfile.open(tar_path, "w") as tf:
        for key, jpg, npz, jsn in shard_buf:
            for ext, data in [(".jpg", jpg), (".expr.npz", npz), (".json", jsn)]:
                ti = tarfile.TarInfo(name=f"{key}{ext}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return tar_path


def _write_bundled_wds(compiled, global_manifest, combined, common_genes, shard_size):
    """Write self-contained bundled shards directly into compiled/.

    Layout: per-sample subdirectories, no extra wds/ nesting.

        {compiled}/
            {sample_id}/
                shard-NNNNNN.tar     jpg + expr.npz + json per cell
            gene_panel.json          gene names in column order
            manifest.parquet         (already written by main; updated here
                                      with bundled shard_path column)
    """
    gene_list = list(common_genes)
    (compiled / "gene_panel.json").write_text(json.dumps(gene_list))
    n_genes = len(gene_list)

    X = combined.X
    n_cells = len(global_manifest)

    # Per-source-shard re-open cache: cells from the same source per-sample
    # shard are contiguous in global_manifest, so one open+getmembers pass
    # per source shard is enough.
    cur_path = None
    cur_tar = None
    cur_members = None

    # Per-sample shard counter: each sample starts at shard-000000.
    sample_shard_state: dict[str, dict] = {}
    bundled_rows = []
    total_shards = 0
    t0 = time.time()

    def _flush(sample_id):
        state = sample_shard_state[sample_id]
        if not state["buf"]:
            return
        sub_dir = compiled / sample_id
        sub_dir.mkdir(exist_ok=True)
        tar_path = _flush_bundled_shard(state["buf"], state["shard_no"], sub_dir)
        # Back-patch shard_path for the rows that just flushed.
        for row in state["pending_rows"]:
            row["shard_path"] = str(tar_path)
        bundled_rows.extend(state["pending_rows"])
        state["buf"] = []
        state["pending_rows"] = []
        state["shard_no"] += 1

    for global_idx in range(n_cells):
        row = global_manifest.iloc[global_idx]
        sample_id = str(row["sample_id"])

        if sample_id not in sample_shard_state:
            sample_shard_state[sample_id] = {
                "shard_no":      0,
                "buf":           [],
                "pending_rows":  [],
            }

        src_path = row["shard_path"]
        if src_path != cur_path:
            if cur_tar is not None:
                cur_tar.close()
            cur_tar = tarfile.open(src_path, "r")
            cur_members = {m.name: m for m in cur_tar.getmembers()}
            cur_path = src_path

        jpg_member = cur_members[f"{row['sample_key']}.jpg"]
        jpg = cur_tar.extractfile(jpg_member).read()

        x_row = X[global_idx]
        if issparse(x_row):
            indices = x_row.indices.astype(np.int32)
            values = x_row.data.astype(np.float32)
        else:
            dense = np.asarray(x_row).reshape(-1)
            nz = np.nonzero(dense)[0]
            indices = nz.astype(np.int32)
            values = dense[nz].astype(np.float32)
        npz_buf = io.BytesIO()
        np.savez(npz_buf, indices=indices, values=values)

        global_key = f"{global_idx:09d}"
        meta = {
            "global_idx": int(global_idx),
            "sample_id":  sample_id,
            "cell_id":    str(row["cell_id"]),
            "sample_key": str(row["sample_key"]),
            "n_genes":    n_genes,
            "n_nonzero":  int(len(indices)),
        }

        state = sample_shard_state[sample_id]
        state["buf"].append((global_key, jpg, npz_buf.getvalue(),
                              json.dumps(meta).encode()))
        state["pending_rows"].append({
            "global_idx":  int(global_idx),
            "sample_id":   sample_id,
            "cell_id":     str(row["cell_id"]),
            "sample_key":  str(row["sample_key"]),
            "global_key":  global_key,
            "shard_path":  None,    # filled in by _flush
        })

        if len(state["buf"]) == shard_size:
            _flush(sample_id)
            total_shards += 1

        if (global_idx + 1) % 5000 == 0:
            rate = (global_idx + 1) / (time.time() - t0)
            print(f"      {global_idx+1}/{n_cells}  {rate:.0f} cells/s")

    # Flush any trailing partial shards (one per sample).
    for sample_id in sample_shard_state:
        if sample_shard_state[sample_id]["buf"]:
            _flush(sample_id)
            total_shards += 1

    if cur_tar is not None:
        cur_tar.close()

    bundled_manifest = pd.DataFrame(bundled_rows)
    # Preserve global_idx order even if multi-sample interleaving in source
    # caused per-sample state to flush at different times.
    bundled_manifest = bundled_manifest.sort_values("global_idx").reset_index(drop=True)
    bundled_manifest.to_parquet(compiled / "bundled_manifest.parquet", index=False)
    return compiled, total_shards, n_genes


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

    # ── 1: Merge manifests ────────────────────────────────────────────────────
    print("[1/3] Merging manifests …")
    parts = [pd.read_parquet(d / "manifest.parquet") for d in sample_dirs]
    global_manifest = pd.concat(parts, ignore_index=True)
    global_manifest["global_idx"] = np.arange(len(global_manifest),
                                               dtype=np.int64)
    global_manifest.to_parquet(compiled / "manifest.parquet", index=False)
    print(f"      {len(global_manifest)} cells total")

    # ── 2: Merge h5ad with gene intersection ──────────────────────────────────
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

    assert combined.n_obs == len(global_manifest), (
        f"h5ad rows ({combined.n_obs}) != manifest rows ({len(global_manifest)})"
    )

    print("[3/3] Writing compiled h5ad …")
    combined.write_h5ad(compiled / "expression.h5ad")

    elapsed = time.time() - t0
    n_shards = sum(1 for d in sample_dirs for _ in d.glob("shard-*.tar"))

    # ── Optional: bundled WebDataset ──────────────────────────────────────────
    wds_info = ""
    if args.bundle_wds:
        print("[bundle] Writing bundled shards …")
        _, n_bundled_shards, n_genes_bundle = _write_bundled_wds(
            compiled, global_manifest, combined, common_genes, args.shard_size)
        print(f"      → {compiled}/{{sample_id}}/shard-NNNNNN.tar "
              f"({n_bundled_shards} shards, "
              f"{len(global_manifest)} cells × {n_genes_bundle} genes)")
        wds_info = f"""

  [bundled] → {compiled}/{{sample_id}}/shard-NNNNNN.tar
  No mmap needed at training time. Each tar entry contains
  {{key}}.jpg + {{key}}.expr.npz + {{key}}.json. Genes list in
  {compiled}/gene_panel.json (n_genes={n_genes_bundle}).

  from daas.dataset import BundledCellPatchDataset
  ds = BundledCellPatchDataset(compiled_dir="{compiled}",
                                sample_ids=train_samples)"""

    print(f"""
{'='*60}
  COMPILE COMPLETE — {len(sample_dirs)} samples, {len(global_manifest)} cells,
                     {len(common_genes)} genes, {n_shards} source shards
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
# batch["expression"].shape → (256, {len(common_genes)}){wds_info}
{'='*60}
  Total time: {time.time() - t0:.1f}s
""")


if __name__ == "__main__":
    main()
