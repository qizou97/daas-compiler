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
import argparse, io, json, sys, tarfile, time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from daas.genes import resolve_gene_panel, validate_gene_panel, write_gene_panel, gene_panel_sha256


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
    p.add_argument("--samples", default=None,
                   help="Comma-separated list of sample IDs to compile. "
                        "Default: all subdirs with manifest + h5ad.")
    p.add_argument("--gene-order",
                   choices=["first_sample", "sorted", "explicit"],
                   default="first_sample",
                   help="Gene ordering policy. first_sample=intersection ordered by "
                        "first sample's var_names (default). sorted=lexicographic. "
                        "explicit=use --gene-panel file.")
    p.add_argument("--gene-panel", default=None,
                   help="Path to JSON file with explicit ordered gene list. "
                        "Required when --gene-order=explicit.")
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


def _write_bundled_wds(compiled, global_manifest, combined, gene_panel, shard_size, gene_panel_sha: str = ""):
    """Write self-contained bundled shards directly into compiled/.

    Layout: per-sample subdirectories, no extra wds/ nesting.

        {compiled}/
            {sample_id}/
                shard-NNNNNN.tar     jpg + expr.npz + json per cell
            manifest.parquet         (already written by main; updated here
                                      with bundled shard_path column)
    """
    gene_list = list(gene_panel)
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
            "global_idx":        int(global_idx),
            "sample_id":         sample_id,
            "cell_id":           str(row["cell_id"]),
            "sample_key":        str(row["sample_key"]),
            "n_genes":           n_genes,
            "n_nonzero":         int(len(indices)),
            "gene_panel_sha256": gene_panel_sha,
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
                         and d.resolve() != compiled.resolve()
                         and (d / "manifest.parquet").exists()
                         and (d / "expression.h5ad").exists())
    assert sample_dirs, f"No valid sample dirs found in {per_sample}"
    print(f"[compile] Found {len(sample_dirs)} samples: "
          f"{[d.name for d in sample_dirs]}")

    if args.samples:
        requested = [s.strip() for s in args.samples.split(",")]
        sample_dir_map = {d.name: d for d in sample_dirs}
        missing = [s for s in requested if s not in sample_dir_map]
        if missing:
            print(f"[compile] ERROR: requested samples not found: {missing}")
            sys.exit(1)
        sample_dirs = [sample_dir_map[s] for s in requested]
        print(f"[compile] --samples filter: using {len(sample_dirs)} of "
              f"{len(sample_dir_map)} available samples")

    # ── 1: Merge manifests ────────────────────────────────────────────────────
    print("[1/3] Merging manifests …")
    parts = [pd.read_parquet(d / "manifest.parquet") for d in sample_dirs]
    global_manifest = pd.concat(parts, ignore_index=True)
    global_manifest["global_idx"] = np.arange(len(global_manifest),
                                               dtype=np.int64)
    global_manifest.to_parquet(compiled / "manifest.parquet", index=False)
    print(f"      {len(global_manifest)} cells total")

    # ── 2: Merge h5ad with gene order contract ────────────────────────────────
    print("[2/3] Merging expression h5ad …")
    adatas = [anndata.read_h5ad(d / "expression.h5ad") for d in sample_dirs]
    sample_names = [d.name for d in sample_dirs]

    explicit_genes = None
    if args.gene_order == "explicit":
        if args.gene_panel is None:
            print("[compile] ERROR: --gene-panel path required with --gene-order=explicit")
            sys.exit(1)
        explicit_genes = json.loads(Path(args.gene_panel).read_text())

    gene_panel = resolve_gene_panel(adatas, sample_names, args.gene_order,
                                    explicit_gene_panel=explicit_genes)
    n_genes_before = adatas[0].n_vars
    print(f"      genes: {n_genes_before} → {len(gene_panel)} "
          f"(policy={args.gene_order}, intersection across {len(adatas)} samples)")

    sliced = [a[:, gene_panel].copy() for a in adatas]
    validate_gene_panel(sliced, sample_names, gene_panel)

    for adata, sample_name in zip(sliced, sample_names):
        adata.obs_names = pd.Index(
            [f"{sample_name}/{name}" for name in adata.obs_names]
        )
    combined = anndata.concat(sliced, axis=0, merge="same")
    assert list(combined.var_names) == gene_panel, \
        "combined var_names != gene_panel after concat"

    assert combined.n_obs == len(global_manifest), (
        f"h5ad rows ({combined.n_obs}) != manifest rows ({len(global_manifest)})"
    )

    print("[3/3] Writing compiled h5ad …")
    combined.write_h5ad(compiled / "expression.h5ad")

    # Always write gene panel and compile report
    write_gene_panel(compiled, gene_panel)
    sha256 = gene_panel_sha256(gene_panel)
    compile_report = {
        "gene_order_policy": args.gene_order,
        "n_samples": len(sample_dirs),
        "sample_ids": [d.name for d in sample_dirs],
        "n_cells": int(combined.n_obs),
        "n_genes": len(gene_panel),
        "gene_panel_sha256": sha256,
        "output_dir": str(compiled),
        "elapsed_s": round(time.time() - t0, 2),
    }
    (compiled / "compile_report.json").write_text(
        json.dumps(compile_report, indent=2)
    )
    print(f"      gene_panel.json + gene_panel.sha256 + compile_report.json → {compiled}/")

    elapsed = time.time() - t0
    n_shards = sum(1 for d in sample_dirs for _ in d.glob("shard-*.tar"))

    # ── Optional: bundled WebDataset ──────────────────────────────────────────
    wds_info = ""
    if args.bundle_wds:
        print("[bundle] Writing bundled shards …")
        _, n_bundled_shards, n_genes_bundle = _write_bundled_wds(
            compiled, global_manifest, combined, gene_panel, args.shard_size, sha256)
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
                     {len(gene_panel)} genes, {n_shards} source shards
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
# batch["expression"].shape → (256, {len(gene_panel)}){wds_info}
{'='*60}
  Total time: {time.time() - t0:.1f}s
""")


if __name__ == "__main__":
    main()
