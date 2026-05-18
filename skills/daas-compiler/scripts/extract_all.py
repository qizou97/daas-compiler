"""
Parallel multi-sample HE patch extraction.
Usage:
  python3 scripts/extract_all.py \
      --zarr-dir /data/spatialdata \
      --output   /data/out \
      --workers  4 \
      [--n-sample 10000] [--pattern "*.zarr"]

Skips samples whose output dir already has manifest.parquet + expression.h5ad.
"""
import argparse, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


EXTRACT_SCRIPT = Path(__file__).parent / "extract_sample.py"


def run_one(zarr_path: str, output: str, extra_args: list[str]):
    """Extract one sample. Returns (sample_id, returncode, tail_log)."""
    sample_id = Path(zarr_path).stem
    out_dir   = Path(output) / sample_id

    if (out_dir / "manifest.parquet").exists() and \
       (out_dir / "expression.h5ad").exists():
        return sample_id, 0, "ALREADY_DONE"

    cmd = [sys.executable, str(EXTRACT_SCRIPT),
           "--zarr", zarr_path, "--output", str(out_dir)] + extra_args
    r   = subprocess.run(cmd, capture_output=True, text=True)
    log = (r.stdout + r.stderr).strip()
    return sample_id, r.returncode, log[-3000:]   # last 3k chars on failure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr-dir",   required=True,
                   help="Directory containing *.zarr samples")
    p.add_argument("--output",     required=True,
                   help="Root output dir; one subdir per sample")
    p.add_argument("--workers",    type=int, default=4,
                   help="Parallel worker processes (default: 4)")
    p.add_argument("--pattern",    default="*.zarr",
                   help="Glob pattern for zarr dirs (default: *.zarr)")
    # forwarded to extract_sample.py
    p.add_argument("--n-sample",   type=int, default=None)
    p.add_argument("--patch-size", type=int, default=224)
    p.add_argument("--mpp",        type=float, default=0.5)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--image-key",  default="he_image")
    p.add_argument("--shapes-key", default="cell_circles")
    p.add_argument("--table-key",  default="table")
    p.add_argument("--extract-mode", default="tile_images",
                   choices=["tile_images", "full_scale0", "full_ops_level"])
    p.add_argument("--patch-filter-policy", default="auto",
                   choices=["auto", "strict_no_padding",
                            "stvisuome_minimal", "strict_with_padding"])
    p.add_argument("--cell-id-column",         default="cell_id")
    p.add_argument("--filter-report-name",     default="filter_report.json")
    return p.parse_args()


def main():
    args = parse_args()

    zarr_paths = sorted(Path(args.zarr_dir).glob(args.pattern))
    assert zarr_paths, \
        f"No zarr files found in {args.zarr_dir} matching '{args.pattern}'"
    print(f"[extract_all] {len(zarr_paths)} samples, {args.workers} workers")

    extra: list[str] = []
    if args.n_sample:
        extra += ["--n-sample", str(args.n_sample)]
    extra += [
        "--patch-size", str(args.patch_size),
        "--mpp",        str(args.mpp),
        "--shard-size", str(args.shard_size),
        "--seed",       str(args.seed),
        "--image-key",  args.image_key,
        "--shapes-key", args.shapes_key,
        "--table-key",  args.table_key,
        "--extract-mode", args.extract_mode,
        "--patch-filter-policy",      args.patch_filter_policy,
        "--cell-id-column",           args.cell_id_column,
        "--filter-report-name",       args.filter_report_name,
    ]

    done, skipped, failed = [], [], []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, str(zp), args.output, extra): zp.stem
            for zp in zarr_paths
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sid, rc, log = fut.result()
            if rc == 0 and log == "ALREADY_DONE":
                tag = "SKIP"
                skipped.append(sid)
            elif rc == 0:
                tag = "OK"
                done.append(sid)
            else:
                tag = "FAIL"
                failed.append(sid)
                print(f"\n  [{sid} FAIL] last output:\n{log}\n")
            print(f"  [{i:>3}/{len(zarr_paths)}] {tag}  {sid}")

    print(f"""
{'='*60}
  EXTRACT ALL COMPLETE
  done={len(done)}  skipped={len(skipped)}  failed={len(failed)}
  failed: {failed or 'none'}
{'='*60}
""")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
