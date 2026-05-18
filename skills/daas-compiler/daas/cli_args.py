"""Lightweight CLI parser for ``scripts/extract_sample.py``.

Lives in the ``daas`` package (not under ``scripts/``) so it can be
imported and exercised by tests without dragging in heavyweight extraction
dependencies (``spatialdata``, ``wsidata``, ``lazyslide``, ``matplotlib``).
``extract_sample.py`` re-uses this parser for its own ``--help`` and
argument parsing.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from daas.filtering import PatchPolicy


DEFAULT_TABLE_KEY = "table"
DEFAULT_SHAPES_KEY = "cell_circles"


def build_extract_sample_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser used by ``scripts/extract_sample.py``."""
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",        required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--sample-id",   default=None,
                   help="样本 ID，默认从 zarr 目录名推断")
    p.add_argument("--n-sample",    type=int, default=None,
                   help="随机采样数量，默认处理全部有效细胞")
    p.add_argument("--patch-size",  type=int, default=224)
    p.add_argument("--mpp",         type=float, default=0.5)
    p.add_argument("--shard-size",  type=int, default=500)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--image-key",   default="he_image")
    p.add_argument("--shapes-key",  default=DEFAULT_SHAPES_KEY)
    p.add_argument("--tissue-shapes-key", default="auto",
                   help="Tissue shape key for overlay viz. 'auto' = probe candidate "
                        "keys (tissue, tissue_boundaries, …). 'none' = skip overlay.")
    p.add_argument("--cell-boundaries-key", default="auto",
                   help="Cell boundary shape key for patch grid overlays. "
                        "'auto' = probe candidate keys.")
    p.add_argument("--nucleus-boundaries-key", default="auto",
                   help="Nucleus boundary shape key for patch grid overlays. "
                        "'auto' = probe candidate keys. 'none' = skip.")
    p.add_argument("--table-key",   default=DEFAULT_TABLE_KEY)
    p.add_argument("--extract-mode", default="tile_images",
                   choices=["tile_images", "full_scale0", "full_ops_level"],
                   help="Patch extraction strategy: tile_images (default, "
                        "low mem), full_scale0 (fast, ~1.6 GB), "
                        "full_ops_level (fastest, ~0.4 GB)")
    p.add_argument("--patch-filter-policy",
                   default=PatchPolicy.AUTO.value,
                   choices=[p.value for p in PatchPolicy],
                   help="Layer 2: patch-validity policy. 'auto' resolves to "
                        "'strict_no_padding' (drop full_oob and need_pad). "
                        "'stvisuome_minimal' keeps boundary-crossing tiles "
                        "and is only valid with --extract-mode tile_images. "
                        "'strict_with_padding' is reserved (raises).")
    p.add_argument("--cell-id-column",         default="cell_id")
    p.add_argument("--filter-report-name",     default="filter_report.json")
    return p


def parse_extract_sample_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parse ``extract_sample.py`` arguments and validate policy combos.

    Validation is run at parse time so misconfigured invocations fail
    before the zarr is loaded.
    """
    args = build_extract_sample_parser().parse_args(argv)
    validate_policy_combination(args)
    return args


def validate_policy_combination(args: argparse.Namespace) -> None:
    """Reject invalid (patch_policy, extract_mode) combinations.

    Raises ``SystemExit`` with a clear message so the CLI exits 2 instead
    of producing a traceback. Test code can catch ``SystemExit``.
    """
    patch_policy = PatchPolicy(args.patch_filter_policy)
    extract_mode = args.extract_mode

    if patch_policy is PatchPolicy.STRICT_WITH_PADDING:
        raise SystemExit(
            "--patch-filter-policy=strict_with_padding is reserved but not "
            "yet implemented. Use 'strict_no_padding' or 'stvisuome_minimal'."
        )

    if (
        patch_policy is PatchPolicy.STVISUOME_MINIMAL
        and extract_mode != "tile_images"
    ):
        raise SystemExit(
            f"--patch-filter-policy=stvisuome_minimal requires "
            f"--extract-mode=tile_images (got {extract_mode!r}). "
            "Full-image modes silently clip boundary-crossing tiles."
        )


__all__ = [
    "DEFAULT_TABLE_KEY",
    "DEFAULT_SHAPES_KEY",
    "build_extract_sample_parser",
    "parse_extract_sample_args",
    "validate_policy_combination",
]
