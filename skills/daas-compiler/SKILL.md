---
name: daas-compiler
description: Extract cell-centered HE image patches from SpatialData into an indexed WebDataset for ML model training. Covers single-sample extraction, multi-sample parallel batch extraction, compile-step gene-intersection merge, and CellPatchDataset with LRU mmap loader. Use when building HE patch datasets for predicting gene expression from tissue morphology, or when scaling a single-sample pipeline to 10s–100s of zarr samples.
---

## Version Compatibility

Tested with: spatialdata 0.7+, wsidata 0.9+, lazyslide 0.10+, anndata 0.10+, numpy 1.26+, pandas 2.2+, scipy 1.12+

Key API surface to verify on version mismatch:
- `wsidata.open_wsi(sdata, image_key, store)` — SpatialDataImage2DReader creation
- `wsi.set_mpp(value)` — must be called before TileSpec; zarr reader auto-detects None
- `TileSpec.from_wsidata(wsi, tile_px, mpp, slide_mpp)` — base_width/ops_level computation
- `wsidata.io.add_tiles(wsi, key, xys, tile_spec, tissue_ids)` — tile GeoDataFrame creation
- `wsi.iter.tile_images(key)` — tile generator
- `lazyslide.pl.tiles(wsi, tile_key)` — returns None, use `plt.gcf()` after
- `wsi.reader.get_thumbnail(size)` — requires `size` parameter

## Installation

This skill ships its own scripts under `${SKILL_DIR}/scripts/` and a small
`daas` Python package under `${SKILL_DIR}/daas/`. Run once per environment:

```bash
pip install -e "${SKILL_DIR}"
pip install -r "${SKILL_DIR}/requirements.txt"
```

After this, `from daas.dataset import CellPatchDataset` works in user code
and all CLI scripts can be invoked via their full skill-relative path.

`${SKILL_DIR}` denotes the absolute path of the directory containing this
SKILL.md file. If imports fail, ask the user which Python interpreter to
use; never hardcode a path.

---

# Cell Patch Dataset: Single-Sample to Multi-Sample Pipeline

## Three-Phase Architecture

```
N × SpatialData.zarr
        │
        │  [Phase 1: extract]  per-sample, parallelizable
        ▼
per-sample/
  {sample_id}/
    shard-{N:06d}.tar     JPEG patches (WebDataset)
    shard-{N:06d}.idx     binary offset index (CIDX0001)
    expression.h5ad       raw counts sparse matrix
    manifest.parquet      per-cell metadata
    viz/
      viz_preflight_boundary.png   cell+nucleus boundary overlay grid
        │
        │  [Phase 2: compile]  run once all samples done, <2 min
        ▼
compiled/
  manifest.parquet        global_idx → image location + expr location
  expression.h5ad         gene intersection across all samples
        │
        │  [Phase 3: train]
        ▼
CellPatchDataset(manifest, h5ad, sample_ids=[...])
```

**Key invariant:** `compiled/manifest.parquet` row `i` == `compiled/expression.h5ad` row `i` == `global_idx=i`. Both are produced by the same `sorted()` traversal in compile.

---

## Phase 1: Single-Sample Extraction

### CLI

```bash
python3 "${SKILL_DIR}/scripts/extract_sample.py" \
    --zarr   /data/A_002.zarr \
    --output /data/out/A_002 \
    [--sample-id A_002]        # default: inferred from zarr dirname
    [--n-sample 10000]         # default: all valid cells
    [--patch-size 224] [--mpp 0.5] [--shard-size 500] [--seed 42]
    [--image-key he_image] [--shapes-key cell_circles] [--table-key table]
```

### Output

```
{output}/
  shard-000000.tar   500 JPEG patches per shard
  shard-000000.idx   binary offset index (CIDX0001, 32B/record)
  expression.h5ad    AnnData (n_cells, n_genes), obs has sample_id
  manifest.parquet   per-cell metadata (see columns below)
```

### manifest.parquet columns

| column | type | notes |
|--------|------|-------|
| `sample_id` | str | e.g. "A_002" |
| `sample_index` | int | 0..N-1 within this sample |
| `sample_key` | str | zero-padded `f"{i:07d}"` |
| `cell_id` | str | original cell ID |
| `shard_path` | str | absolute path to .tar file |
| `tar_offset` | int64 | byte offset of JPEG in tar |
| `jpg_size` | int32 | JPEG byte count |
| `expr_row` | int32 | = sample_index (local row in per-sample h5ad) |
| `center_x_pixel` | float64 | centroid x in level-0 px |
| `center_y_pixel` | float64 | centroid y in level-0 px |
| `bbox_x0` | float64 | tile top-left x |
| `bbox_y0` | float64 | tile top-left y |

### Pipeline Internals

**MPP derivation** (from affine transforms, no hardcoding):

```python
img_aff   = get_transformation(sdata.images[image_key],
                to_coordinate_system="global").to_affine_matrix(
                input_axes=("y","x"), output_axes=("y","x"))
shape_aff = get_transformation(gdf,
                to_coordinate_system="global").to_affine_matrix(
                input_axes=("y","x"), output_axes=("y","x"))

sx = np.sqrt(img_aff[0,0]**2 + img_aff[1,0]**2)   # rotation-invariant
sy = np.sqrt(img_aff[0,1]**2 + img_aff[1,1]**2)
SCALE_SHAPE = (shape_aff[0,0] + shape_aff[1,1]) / 2.0   # µm → global
SLIDE_MPP   = ((sx + sy) / 2.0) / SCALE_SHAPE
```

| Image tf | Shape tf | SLIDE_MPP | Example |
|----------|----------|-----------|---------|
| Identity | Scale(s) | 1/s | Xenium: s=4.7059 → 0.2125 µm/px |
| Scale(p) | Scale(s) | p/s | Visium: p=0.5, s=1.0 → 0.5 |
| Affine   | Affine   | sqrt(a²+b²) / (sx+sy)/2 | rotated scans |

**OOB filtering:**

```python
BASE_SIZE = round(PATCH_SIZE * MPP_TGT / SLIDE_MPP)
BASE_HALF = BASE_SIZE / 2.0   # MUST be float, not integer division

full_oob = ((sx0+BASE_SIZE<=0)|(sx0>=IMG_W)|(sy0+BASE_SIZE<=0)|(sy0>=IMG_H))
need_pad = (((sx0<0)|(sx0+BASE_SIZE>IMG_W)|(sy0<0)|(sy0+BASE_SIZE>IMG_H)) & ~full_oob)
valid_mask = ~full_oob & ~need_pad
# full_oob: skip entirely; need_pad: skip (zarr clips silently, no zero-fill)
```

**Spatial sort** (zarr cache locality):

```python
sort_key = ((np.maximum(0,sy0)//4096)*10000 + (np.maximum(0,sx0)//4096))
proc_order = np.argsort(sort_key, kind="stable")
```

**TileSpec setup:**

```python
wsi = open_wsi(sdata, image_key=image_key, store=None)
wsi.set_mpp(SLIDE_MPP)
spec = TileSpec.from_wsidata(wsi, tile_px=PATCH_SIZE, mpp=MPP_TGT, slide_mpp=SLIDE_MPP)
assert spec.base_width == BASE_SIZE   # must match
```

**Patch extraction** — only `wsi.iter.tile_images()` is allowed:

```python
add_tiles(wsi, key="cell_tiles",
          xys=np.column_stack([sx0_ord, sy0_ord]),
          tile_spec=spec, tissue_ids=np.zeros(n_out, dtype=int))
for tile in wsi.iter.tile_images("cell_tiles"):
    jpg = io.BytesIO()
    Image.fromarray(tile.image).save(jpg, format="JPEG", quality=95)
```

Never use numpy/dask/zarr/PIL/cv2 to read HE pixel data directly.

### Extraction Strategies

> **Required: ask the user which `--extract-mode` to use before running any extraction.**
> The three modes have very different memory and speed profiles; the right
> choice depends on the user's hardware. Do NOT default silently. Present the
> table below, ask the user to pick one of `tile_images`, `full_scale0`, or
> `full_ops_level`, and pass that choice as `--extract-mode <choice>` on the
> CLI. This applies to both `extract_sample.py` (single sample) and
> `extract_all.py` (batch — forwards the flag to every worker).

| Strategy | `--extract-mode` | Memory | Speed (3000 tiles) | Quality |
|----------|------------------|--------|---------------------|---------|
| Original `tile_images` | `tile_images` | Low (~50 MB) | 224s (1×) | Baseline |
| Full scale0 load | `full_scale0` | scale0 size (~1.6 GB) | 24s (9×) | Good (插值差异) |
| Full ops_level load | `full_ops_level` | ops_level size (~0.4 GB) | 6s (36×) | Good (插值差异) |

**How they work:**

```
tile_images:     add_tiles → iter.tile_images → each tile triggers 3 chunk reads
full_scale0:     sdata["scale0"].values (1.6GB) → memory crop 527×527 → resize 224
full_ops_level:  sdata["scaleN"].values (0.4GB) → memory crop ops_w×ops_w → resize 224
```

**`full_ops_level` — recommended for most cases:**

Reads the entire pyramid level that TileSpec selected (`spec.ops_level`), crops tiles from memory, and resizes. This is the fastest strategy because:
- The ops_level is already the optimal pyramid level (closest to target MPP)
- Only one chunk-read pass (loading `.values` triggers dask compute once)
- Memory scales with the pyramid level: ~0.4 GB for typical Xenium scale1, less for higher levels

```python
lvl_key = f"scale{spec.ops_level}"
full_img = sdata.images[image_key][lvl_key]["image"].values  # (C, H, W)
ds_y = scale0_img.shape[1] / full_img.shape[1]   # level-0 → ops_level ratio
ds_x = scale0_img.shape[2] / full_img.shape[2]

for i in range(n_out):
    x0 = int(sx0_ord[i] / ds_x)
    y0 = int(sy0_ord[i] / ds_y)
    tile = full_img[:, y0:y0+spec.ops_width, x0:x0+spec.ops_width]
    tile_224 = Image.fromarray(tile.transpose(1,2,0)).resize((224,224), LANCZOS)
```

**`full_scale0` — when ops_level is not available:**

Loads the full-resolution scale0 image. Higher memory but same principle.

**`tile_images` — when memory is scarce:**

The original wsidata pipeline. Each tile triggers zarr chunk reads. Slowest but uses minimal memory.

**Choosing the ops_level:**

TileSpec selects `ops_level` to minimize resize magnitude:

| MPP_TGT | ops_level | ops_width | Rationale |
|---------|-----------|-----------|-----------|
| 0.2 | 0 | 211→224 | scale0 MPP closest to target |
| 0.3-0.6 | 1 | 158-316→224 | scale1 best match |
| 0.8-1.2 | 2 | 210-316→224 | scale2 best match |
| 1.5-2.5 | 3 | 197-329→224 | scale3 best match |

The rule: prefer `ops_width >= PATCH_SIZE` (downsample) over upsampling. The code always follows `spec.ops_level` dynamically — no hardcoding.

**Shard writing — two-phase (write then re-read for accurate offsets):**

```python
IDX_MAGIC      = b"CIDX0001"
IDX_RECORD_FMT = "<iQIQII"   # sample_index(i4) jpg_offset(Q) jpg_size(I)
                              # json_offset(Q) json_size(I) reserved(I)
assert struct.calcsize(IDX_RECORD_FMT) == 32   # must be 32, never use hardcoded size

def flush_shard(shard_buf, shard_no, output_dir):
    tar_path = output_dir / f"shard-{shard_no:06d}.tar"
    # Phase 1: write
    with tarfile.open(tar_path, "w") as tf:
        for si, sk, jpg, jsn in shard_buf:
            for ext, data in [(".jpg", jpg), (".json", jsn)]:
                ti = tarfile.TarInfo(name=f"{sk}{ext}"); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    # Phase 2: re-read for accurate offset_data (tell() during write is wrong)
    records = []
    with tarfile.open(tar_path, "r") as tf:
        members = {m.name: m for m in tf.getmembers()}
        for si, sk, jpg, jsn in shard_buf:
            jm, nm = members[f"{sk}.jpg"], members[f"{sk}.json"]
            records.append((si, jm.offset_data, jm.size, nm.offset_data, nm.size))
    with open(output_dir / f"shard-{shard_no:06d}.idx", "wb") as f:
        f.write(IDX_MAGIC)
        f.write(struct.pack("<I", len(records)))
        for rec in records:
            f.write(struct.pack(IDX_RECORD_FMT, *rec, 0))
    return str(tar_path), records
```

After `flush_shard`, back-patch `shard_path`/`tar_offset`/`jpg_size` into `cells_rows`:

```python
tar_path, recs = flush_shard(shard_buf, shard_no, output_dir)
for row, rec in zip(cells_rows[-len(shard_buf):], recs):
    row["shard_path"] = tar_path
    row["tar_offset"] = rec[1]   # jm.offset_data
    row["jpg_size"]   = rec[2]   # jm.size
```

---

## Phase 1b: Parallel Multi-Sample Extraction

### CLI

```bash
python3 "${SKILL_DIR}/scripts/extract_all.py" \
    --zarr-dir /data/spatialdata \
    --output   /data/out \
    --workers  4 \
    [--n-sample 10000] [--pattern "*.zarr"]
    # all other extract_sample.py args forwarded
```

### Behavior

- Discovers all `*.zarr` dirs under `--zarr-dir` (sorted)
- Runs `extract_sample.py` as a subprocess per sample using `ProcessPoolExecutor`
- **Skips** samples whose `{output}/{sample_id}/manifest.parquet` + `expression.h5ad` already exist — safe to re-run after partial failures
- Exits with code 1 if any sample fails; prints last 3k chars of stderr for each failure

### Worker count guidance

| Scenario | `--workers` | Notes |
|----------|-------------|-------|
| 72 samples, fast NFS | 4–8 | Zarr I/O is the bottleneck |
| Local NVMe | 8–16 | CPU/memory bound instead |
| Debug single sample | 1 | Easier to read logs |

Each worker loads one full zarr into memory (~2–4 GB RAM). Set workers ≤ `floor(RAM_GB / 4)`.

### Implementation pattern

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_one(zarr_path, output, extra_args):
    sample_id = Path(zarr_path).stem
    out_dir   = Path(output) / sample_id
    if (out_dir/"manifest.parquet").exists() and (out_dir/"expression.h5ad").exists():
        return sample_id, 0, "ALREADY_DONE"
    cmd = [sys.executable, "scripts/extract_sample.py",
           "--zarr", zarr_path, "--output", str(out_dir)] + extra_args
    r = subprocess.run(cmd, capture_output=True, text=True)
    return sample_id, r.returncode, (r.stdout + r.stderr)[-3000:]

with ProcessPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(run_one, str(zp), output, extra): zp.stem
               for zp in zarr_paths}
    for fut in as_completed(futures):
        sid, rc, log = fut.result()
        print("OK" if rc == 0 else f"FAIL: {log}")
```

---

## Phase 2: Compile

### CLI

```bash
python3 "${SKILL_DIR}/scripts/compile_dataset.py" \
    --per-sample-dir /data/out \
    --output         /data/compiled
```

Scans all subdirs of `--per-sample-dir` that have **both** `manifest.parquet` and `expression.h5ad`. Skips others (e.g. smoke test dirs).

### Implementation

```python
sample_dirs = sorted(d for d in per_sample.iterdir()
                     if d.is_dir()
                     and (d/"manifest.parquet").exists()
                     and (d/"expression.h5ad").exists())

# CRITICAL: same sorted() for both — guarantees manifest row i == h5ad row i
parts  = [pd.read_parquet(d/"manifest.parquet") for d in sample_dirs]
adatas = [anndata.read_h5ad(d/"expression.h5ad") for d in sample_dirs]

global_manifest = pd.concat(parts, ignore_index=True)
global_manifest["global_idx"] = np.arange(len(global_manifest), dtype=np.int64)

common_genes = adatas[0].var_names
for a in adatas[1:]:
    common_genes = common_genes.intersection(a.var_names)
assert len(common_genes) > 0, "Gene intersection is empty"

combined = anndata.concat([a[:, common_genes] for a in adatas],
                          axis=0, merge="same")
combined.obs_names_make_unique()
assert combined.n_obs == len(global_manifest)   # invariant check

global_manifest.to_parquet(compiled/"manifest.parquet", index=False)
combined.write_h5ad(compiled/"expression.h5ad")
```

### Compile speed (72 samples, ~5M cells)

| Step | Time |
|------|------|
| concat 72 parquet | <5s |
| read 72 h5ad | ~30s |
| gene intersection + concat | ~20s |
| write compiled h5ad | ~30s |
| **total** | **<2 min** |

---

## Phase 3: CellPatchDataset

### Usage

```python
from daas.dataset import CellPatchDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Split is injected externally via sample_ids — not hardcoded in dataset
train_samples = json.load(open("splits/train.json"))

ds = CellPatchDataset(
    manifest_path  = "compiled/manifest.parquet",
    h5ad_path      = "compiled/expression.h5ad",
    sample_ids     = train_samples,          # None = all samples
    transform      = T.Compose([T.ToTensor()]),
    mmap_cache_size = 256,   # see sizing note below
)
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8)
batch  = next(iter(loader))
# batch["image"].shape      → (256, 3, 224, 224)
# batch["expression"].shape → (256, n_genes)
# batch["cell_id"]          → list of str
# batch["sample_id"]        → list of str
```

### Three training modes

```python
ds = CellPatchDataset(manifest, h5ad, sample_ids=train, transform=dino_augment)
batch["image"]              # self-supervised patch pretraining

ds = CellPatchDataset(manifest, h5ad, sample_ids=train)
batch["expression"]         # expression foundation model

ds = CellPatchDataset(manifest, h5ad, sample_ids=train, transform=val_tf)
batch["image"], batch["expression"]   # multimodal HE → cell states
```

### LRU mmap cache

Each DataLoader worker inherits an empty cache (Linux fork semantics) and builds its own. No cross-worker sharing.

**mmap_cache_size sizing:**

```python
# Per worker: need to cache ≈ ceil(n_shards / num_workers) handles
# 72 samples × 138 shards/sample = ~10k shards
# 8 workers → ceil(10k/8) = 1250 per worker
# Default 128 causes cache thrash at this scale

ds = CellPatchDataset(..., mmap_cache_size=512)   # safe for most runs
```

| Scale | Shards total | Workers | Recommended cache size |
|-------|-------------|---------|------------------------|
| 2 samples, 40 shards | 40 | 8 | 16 (default 128 fine) |
| 72 samples, ~10k shards | 10000 | 8 | 512–1024 |

### Implementation sketch

```python
class LRUMmapCache:
    def __init__(self, maxsize=128):
        self._cache = OrderedDict()   # path → mmap.mmap
        self._files = {}              # path → file handle
        self.maxsize = maxsize

    def get(self, path):
        if path in self._cache:
            self._cache.move_to_end(path); return self._cache[path]
        if len(self._cache) >= self.maxsize:
            ep, mm = self._cache.popitem(last=False)
            mm.close(); self._files.pop(ep).close()
        f  = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._files[path] = f; self._cache[path] = mm
        return mm

def __getitem__(self, idx):
    row      = self.manifest.iloc[idx]
    mm       = self._mmap.get(str(row["shard_path"]))
    jpg      = bytes(mm[int(row["tar_offset"]) : int(row["tar_offset"]) + int(row["jpg_size"])])
    img      = Image.open(io.BytesIO(jpg)).convert("RGB")
    if self.transform: img = self.transform(img)

    row_x = self.X[int(row["global_idx"])]
    expr  = (row_x.toarray() if issparse(row_x) else np.array(row_x)).reshape(-1).astype(np.float32)
    return {"image": img, "expression": expr,
            "cell_id": str(row["cell_id"]), "sample_id": str(row["sample_id"])}
```

### Memory footprint (5M cells × 5001 genes, ~1% sparsity)

| Component | Size |
|-----------|------|
| CSR expression matrix | ~4 GB |
| manifest DataFrame | ~1 GB |
| LRU mmap (512 shards) | OS page cache, on-demand |
| **Training process total** | **~5 GB** |

---

## Visualization Validation

**Pre-flight boundary viz** runs automatically during `extract_sample.py` before the full extraction (see [Pre-Flight Check + Boundary Viz](#pre-flight-check--boundary-viz-up-to-9-cells)). It saves a grid of up to 9 test patches with cell boundary (cyan), nucleus boundary (yellow), and center crosshair (red) overlays to `{output}/viz/viz_preflight_boundary.png`.

**Post-extraction viz** can be run separately to verify patch quality and spatial alignment.

### CLI

```bash
python3 "${SKILL_DIR}/scripts/viz_sample.py" \
    --zarr   /data/A_002.zarr \
    --output /data/out/A_002    # reads manifest.parquet from here
```

Outputs to `{output}/viz/`:
- `viz_preflight_boundary.png` — cell+nucleus boundary overlay grid (from extract pre-flight)
- `viz_global_tiles.png` — all tiles on whole-slide overview via `lazyslide.pl.tiles` at dpi=300; **always produced**, even with `--skip-viz`
- `viz_centroid_overlay.png` — centroids on thumbnail (skipped with `--skip-viz`)
- `viz_patch_grid.png` — 5×5 grid with cell (cyan) + nucleus (yellow) boundaries, center crosshair (red) (skipped with `--skip-viz`)

### Boundary overlay coordinate transform

```python
SCALE = PATCH_SIZE / BASE_SIZE   # level-0 px → output px (e.g. 224/527)

def shape_um_to_patch_px(coords_um, x0, y0):
    arr  = np.array(coords_um)
    col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
    row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
    return np.column_stack([col, row_])

# cell boundary (cyan)
pts = shape_um_to_patch_px(list(cell_bounds.loc[cell_id].exterior.coords), x0, y0)
ax.add_patch(MplPolygon(pts, closed=True, edgecolor="cyan", facecolor="none", lw=0.8))

# nucleus boundary (yellow, if present)
if cell_id in nucl_ids:
    pts = shape_um_to_patch_px(list(nucl_bounds.loc[cell_id].exterior.coords), x0, y0)
    ax.add_patch(MplPolygon(pts, closed=True, edgecolor="yellow", facecolor="none", lw=0.8))

# crosshair at patch center (red)
cx = cy = PATCH_SIZE / 2; arm = PATCH_SIZE * 0.08
ax.plot([cx-arm, cx+arm], [cy, cy], color="red", lw=0.8)
ax.plot([cx, cx], [cy-arm, cy+arm], color="red", lw=0.8)
```

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `${SKILL_DIR}/scripts/extract_sample.py` | Single-sample extraction (argparse), includes pre-flight boundary viz |
| `${SKILL_DIR}/scripts/extract_all.py` | Parallel multi-sample batch extraction |
| `${SKILL_DIR}/scripts/compile_dataset.py` | Compile per-sample dirs → global dataset |
| `${SKILL_DIR}/scripts/viz_sample.py` | Visualization validation for one sample |
| `${SKILL_DIR}/daas/dataset.py` | `LRUMmapCache` + `CellPatchDataset` |

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `struct.calcsize(fmt) != 32` | Wrong IDX_RECORD_FMT | Use `"<iQIQII"` (32B); assert at script top |
| `lpl.tiles(...)` returns None | lazyslide API | Call `lpl.tiles(...)` then `fig = plt.gcf()` |
| tar random read returns wrong JPEG | `tell()` during write is inaccurate | Two-phase: write-then-reread `offset_data` |
| `'>' not supported between 'tuple' and 'int'` in `get_thumbnail` | SpatialDataImage2DReader API | `except Exception: thumb = sdata.images[k]["scale4"]["image"].values.transpose(1,2,0)` |
| `AttributeError: 'DataTree' has no attribute 'shape'` | Multiscale image is DataTree | Use `sdata.images[k]["scale0"]["image"].shape` |
| `BASE_HALF` causes 0.5px offset | Integer division `BASE_SIZE // 2` | Use `BASE_SIZE / 2.0` (float) |
| subprocess cannot find lazyslide | Python interpreter mismatch | Use `sys.executable` (or ask user for interpreter path); never use `conda run` |
| `sample_key` read as int after parquet round-trip | Old CSV workflow; parquet preserves str | Use `manifest["sample_key"].astype(str)` after `read_parquet` for safety |
| `todense()` AttributeError on dense X | h5ad stored without sparse compression | `row_x.toarray() if issparse(row_x) else np.array(row_x)` |
| `h5ad rows != manifest rows` during compile | Different sort order for manifest vs h5ad | Both must use the same `sorted(sample_dirs)` list |
| mmap cache thrash at large scale | `mmap_cache_size=128` too small for 72 samples | Set `mmap_cache_size ≈ ceil(total_shards / num_workers)` |
| extract_all worker OOM | Too many workers × zarr in memory | Reduce `--workers`; each worker needs ~4 GB RAM |

## Pre-Flight Check + Boundary Viz (up to 9 cells)

Before full-scale processing of a new sample, the pipeline extracts up to 9 test tiles and renders a boundary overlay grid to `{output}/viz/viz_preflight_boundary.png`:

**Steps:**
1. Extract test tiles via `add_tiles` + `iter.tile_images`
2. Assert shape `(PATCH_SIZE, PATCH_SIZE, 3)` and dtype `uint8`
3. Load `cell_boundaries` and `nucleus_boundaries` from `sdata.shapes`
4. Overlay cell boundaries (cyan), nucleus boundaries (yellow), and center crosshair (red) on each patch
5. Save grid to `{output}/viz/viz_preflight_boundary.png`

```python
n_test     = min(9, n_out)
test_xys   = np.column_stack([sx0_ord[:n_test], sy0_ord[:n_test]])
add_tiles(wsi, key="test_tiles", xys=test_xys, tile_spec=spec,
          tissue_ids=np.zeros(n_test, dtype=int))
test_images = []
for tile in wsi.iter.tile_images("test_tiles"):
    assert tile.image.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert tile.image.dtype == np.uint8
    test_images.append(tile.image)

# Render boundary overlay grid
viz_dir = output_dir / "viz"
viz_dir.mkdir(exist_ok=True)
_render_boundary_grid(test_images, cell_ids_ord[:n_test],
                      sx0_ord[:n_test], sy0_ord[:n_test],
                      sdata, SCALE_SHAPE, PATCH_SIZE, BASE_SIZE,
                      sample_id, viz_dir)
```

### Boundary overlay coordinate transform

```python
SCALE = PATCH_SIZE / BASE_SIZE   # level-0 px → output px (e.g. 224/527)

def shape_um_to_patch_px(coords_um, x0, y0):
    arr  = np.array(coords_um)
    col  = (arr[:, 0] * SCALE_SHAPE - x0) * SCALE
    row_ = (arr[:, 1] * SCALE_SHAPE - y0) * SCALE
    return np.column_stack([col, row_])

# cell boundary (cyan)
pts = shape_um_to_patch_px(list(cell_bounds.loc[cell_id].exterior.coords), x0, y0)
ax.add_patch(MplPolygon(pts, closed=True, edgecolor="cyan", facecolor="none", lw=0.8))

# nucleus boundary (yellow)
if cell_id in nucl_ids:
    pts = shape_um_to_patch_px(list(nucl_bounds.loc[cell_id].exterior.coords), x0, y0)
    ax.add_patch(MplPolygon(pts, closed=True, edgecolor="yellow", facecolor="none", lw=0.8))

# crosshair at patch center (red)
cx = cy = PATCH_SIZE / 2; arm = PATCH_SIZE * 0.08
ax.plot([cx-arm, cx+arm], [cy, cy], color="red", lw=0.8)
ax.plot([cx, cx], [cy-arm, cy+arm], color="red", lw=0.8)
```
