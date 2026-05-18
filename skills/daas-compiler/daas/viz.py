# daas/viz.py
"""Visualization helpers for extract_sample.py.

Overlay key resolution is pure Python. Render functions import matplotlib
lazily so they are only required when the extract group is installed.
"""
from __future__ import annotations

import io
import json as _json
import tarfile
from pathlib import Path

import numpy as np


_TISSUE_CANDIDATES = [
    "tissue",
    "tissue_boundaries",
    "tissue_boundary",
    "tissue_regions",
    "tissue_region",
    "filtered_tissue",
]
_CELL_CANDIDATES = [
    "filtered_cell_boundaries",
    "cell_boundaries",
    "filtered_cell_circles",
    "cell_circles",
]
_NUCLEUS_CANDIDATES = [
    "filtered_nucleus_boundaries",
    "nucleus_boundaries",
]


def _resolve_key(shapes_keys: set, candidates: list[str], hint: str) -> str | None:
    if hint == "none":
        return None
    if hint != "auto":
        return hint if hint in shapes_keys else None
    for k in candidates:
        if k in shapes_keys:
            return k
    return None


def resolve_tissue_key(sdata, hint: str = "auto") -> str | None:
    return _resolve_key(set(sdata.shapes.keys()), _TISSUE_CANDIDATES, hint)


def resolve_cell_boundaries_key(sdata, hint: str = "auto") -> str | None:
    return _resolve_key(set(sdata.shapes.keys()), _CELL_CANDIDATES, hint)


def resolve_nucleus_key(sdata, hint: str = "auto") -> str | None:
    return _resolve_key(set(sdata.shapes.keys()), _NUCLEUS_CANDIDATES, hint)


def save_tiles_overview(
    output_dir,
    wsi,
    sdata=None,
    tissue_key: str | None = None,
    SCALE_SHAPE: float = 1.0,
    dpi: int = 300,
) -> dict:
    """Render lazyslide tiles overview and optionally a tissue overlay variant.

    Returns dict with keys: viz_global_tiles, viz_global_tiles_tissue_overlay (or None), warnings.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lazyslide.pl as lpl

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(exist_ok=True)
    warnings = []

    lpl.tiles(wsi, tile_key="cell_tiles")
    fig = plt.gcf()
    out = viz_dir / "viz_global_tiles.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    overlay_path = None
    if tissue_key is not None and sdata is not None:
        try:
            tissue_gdf = sdata.shapes[tissue_key]
            lpl.tiles(wsi, tile_key="cell_tiles")
            fig2 = plt.gcf()
            ax = fig2.axes[0]
            for geom in tissue_gdf.geometry:
                if geom is None:
                    continue
                try:
                    from shapely.geometry import MultiPolygon
                    polys = (list(geom.geoms)
                             if isinstance(geom, MultiPolygon) else [geom])
                    for poly in polys:
                        xs, ys = poly.exterior.xy
                        ax.plot(
                            [x * SCALE_SHAPE for x in xs],
                            [y * SCALE_SHAPE for y in ys],
                            color="lime", linewidth=1.0, alpha=0.8,
                        )
                except Exception as e:
                    warnings.append(f"tissue overlay geometry error: {e}")
            overlay_path = viz_dir / "viz_global_tiles_tissue_overlay.png"
            fig2.savefig(overlay_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig2)
        except Exception as e:
            warnings.append(f"tissue overlay skipped: {e}")
            overlay_path = None

    return {
        "viz_global_tiles": str(out),
        "viz_global_tiles_tissue_overlay": str(overlay_path) if overlay_path else None,
        "warnings": warnings,
    }


def save_patch_grid(
    images,
    cell_ids,
    x0s,
    y0s,
    sdata,
    SCALE_SHAPE: float,
    PATCH_SIZE: int,
    BASE_SIZE: int,
    sample_id: str,
    viz_dir,
    cell_key: str | None = None,
    nucleus_key: str | None = None,
    dpi: int = 300,
) -> Path:
    """Render pre-flight patch grid with optional boundary overlays."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    viz_dir = Path(viz_dir)
    cell_bounds = (sdata.shapes[cell_key]
                   if cell_key and cell_key in sdata.shapes else None)
    nucl_bounds = (sdata.shapes[nucleus_key]
                   if nucleus_key and nucleus_key in sdata.shapes else None)
    nucl_ids = set(nucl_bounds.index) if nucl_bounds is not None else set()
    SCALE = PATCH_SIZE / BASE_SIZE

    def _um_to_px(coords_um, x0, y0):
        arr = np.array(coords_um)
        return np.column_stack([
            (arr[:, 0] * SCALE_SHAPE - x0) * SCALE,
            (arr[:, 1] * SCALE_SHAPE - y0) * SCALE,
        ])

    n_test = len(images)
    n_cols = int(np.ceil(np.sqrt(n_test)))
    n_rows = int(np.ceil(n_test / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_test == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flat

    for i in range(n_test):
        ax = axes_flat[i]
        ax.imshow(images[i])
        cell_id = cell_ids[i]
        x0, y0 = x0s[i], y0s[i]
        if cell_bounds is not None:
            try:
                cb_pts = _um_to_px(
                    list(cell_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(cb_pts, closed=True,
                                        edgecolor="cyan", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        if nucl_bounds is not None and cell_id in nucl_ids:
            try:
                nb_pts = _um_to_px(
                    list(nucl_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(nb_pts, closed=True,
                                        edgecolor="yellow", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        cx = cy = PATCH_SIZE / 2
        arm = PATCH_SIZE * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)
        ax.set_xlim(0, PATCH_SIZE)
        ax.set_ylim(PATCH_SIZE, 0)
        ax.set_title(str(cell_id)[:12], fontsize=5)
        ax.axis("off")

    for j in range(n_test, len(list(np.array(axes).flat))):
        np.array(axes).flat[j].axis("off")

    fig.suptitle(
        f"{sample_id} — patch grid pre-flight "
        f"({n_test} cells, cyan=cell  yellow=nucleus  +=center)",
        fontsize=9, y=0.995,
    )
    plt.tight_layout()
    out_path = viz_dir / "viz_patch_grid.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_saved_patch_grid(
    manifest_df,
    sdata,
    viz_dir,
    sample_id: str,
    patch_size: int,
    SCALE_SHAPE: float,
    x0_col: str = "bbox_x0",
    y0_col: str = "bbox_y0",
    cell_key: str | None = None,
    nucleus_key: str | None = None,
    base_size: int | None = None,
    n_grid: int = 25,
    seed: int = 0,
) -> dict:
    """Read saved JPEGs from manifest shards and render a post-save validation grid.

    Saves viz_dir/viz_saved_patch_grid.png and viz_dir/viz_saved_patch_grid_report.json.
    Returns the report dict.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from PIL import Image

    if base_size is None:
        base_size = patch_size

    viz_dir = Path(viz_dir)
    viz_dir.mkdir(exist_ok=True)
    overlay_keys_used = [k for k in [cell_key, nucleus_key] if k is not None]

    cell_bounds = (sdata.shapes[cell_key]
                   if cell_key and cell_key in sdata.shapes else None)
    nucl_bounds = (sdata.shapes[nucleus_key]
                   if nucleus_key and nucleus_key in sdata.shapes else None)
    nucl_ids = set(nucl_bounds.index) if nucl_bounds is not None else set()

    rng = np.random.default_rng(seed)
    n_available = len(manifest_df)
    n_sample = min(n_grid, n_available)
    chosen_idx = rng.choice(n_available, n_sample, replace=False)

    images, cell_ids, x0s, y0s = [], [], [], []
    missing_members = 0
    decode_errors = 0
    bad_image_size = 0

    open_tars: dict = {}
    try:
        for ci in chosen_idx:
            row = manifest_df.iloc[int(ci)]
            shard_path = str(row["shard_path"])
            sample_key = str(row["sample_key"])
            cell_id = str(row["cell_id"])

            if shard_path not in open_tars:
                try:
                    open_tars[shard_path] = tarfile.open(shard_path, "r")
                except Exception:
                    missing_members += 1
                    continue

            tf = open_tars[shard_path]
            try:
                member = tf.getmember(f"{sample_key}.jpg")
                jpg = tf.extractfile(member).read()
            except KeyError:
                missing_members += 1
                continue

            try:
                img_arr = np.array(Image.open(io.BytesIO(jpg)).convert("RGB"))
            except Exception:
                decode_errors += 1
                continue

            if img_arr.shape[:2] != (patch_size, patch_size):
                bad_image_size += 1
                continue

            images.append(img_arr)
            cell_ids.append(cell_id)
            x0s.append(float(row[x0_col]) if x0_col in row.index else 0.0)
            y0s.append(float(row[y0_col]) if y0_col in row.index else 0.0)
    finally:
        for tf in open_tars.values():
            tf.close()

    def _um_to_px(coords_um, x0, y0):
        arr = np.array(coords_um)
        SCALE = patch_size / base_size
        return np.column_stack([
            (arr[:, 0] * SCALE_SHAPE - x0) * SCALE,
            (arr[:, 1] * SCALE_SHAPE - y0) * SCALE,
        ])

    n_rendered = len(images)
    if n_rendered == 0:
        report = {
            "n_checked": n_sample, "n_rendered": 0,
            "missing_members": missing_members, "decode_errors": decode_errors,
            "bad_image_size": bad_image_size, "overlay_keys_used": overlay_keys_used,
            "viz_saved_patch_grid": None, "warnings": ["No images could be loaded"],
        }
        (viz_dir / "viz_saved_patch_grid_report.json").write_text(
            _json.dumps(report, indent=2))
        return report

    n_cols = int(np.ceil(np.sqrt(n_rendered)))
    n_rows = int(np.ceil(n_rendered / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rendered == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flat

    for i in range(n_rendered):
        ax = axes_flat[i]
        ax.imshow(images[i])
        cell_id = cell_ids[i]
        x0, y0 = x0s[i], y0s[i]
        if cell_bounds is not None:
            try:
                cb_pts = _um_to_px(
                    list(cell_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(cb_pts, closed=True,
                                        edgecolor="cyan", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        if nucl_bounds is not None and cell_id in nucl_ids:
            try:
                nb_pts = _um_to_px(
                    list(nucl_bounds.loc[cell_id, "geometry"].exterior.coords), x0, y0)
                ax.add_patch(MplPolygon(nb_pts, closed=True,
                                        edgecolor="yellow", facecolor="none",
                                        linewidth=0.8, alpha=0.9))
            except KeyError:
                pass
        cx = cy = patch_size / 2
        arm = patch_size * 0.08
        ax.plot([cx - arm, cx + arm], [cy, cy], color="red", lw=0.8, alpha=0.9)
        ax.plot([cx, cx], [cy - arm, cy + arm], color="red", lw=0.8, alpha=0.9)
        ax.set_xlim(0, patch_size)
        ax.set_ylim(patch_size, 0)
        ax.set_title(str(cell_id)[:12], fontsize=5)
        ax.axis("off")

    for j in range(n_rendered, len(list(np.array(axes).flat))):
        np.array(axes).flat[j].axis("off")

    fig.suptitle(
        f"{sample_id} — saved patch grid ({n_rendered} cells, "
        f"cyan=cell  yellow=nucleus)",
        fontsize=9, y=0.995,
    )
    plt.tight_layout()
    out_path = viz_dir / "viz_saved_patch_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "n_checked": n_sample, "n_rendered": n_rendered,
        "missing_members": missing_members, "decode_errors": decode_errors,
        "bad_image_size": bad_image_size, "overlay_keys_used": overlay_keys_used,
        "viz_saved_patch_grid": str(out_path), "warnings": [],
    }
    (viz_dir / "viz_saved_patch_grid_report.json").write_text(
        _json.dumps(report, indent=2))
    return report


__all__ = [
    "resolve_tissue_key",
    "resolve_cell_boundaries_key",
    "resolve_nucleus_key",
    "save_tiles_overview",
    "save_patch_grid",
    "save_saved_patch_grid",
]
