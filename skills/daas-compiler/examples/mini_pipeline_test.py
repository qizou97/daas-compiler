# Reference: spatialdata 0.7+, wsidata 0.9+, lazyslide 0.10+, anndata 0.10+, numpy 1.26+
# Verify API if version differs

"""
mini pipeline test: 5-cell end-to-end validation before full-scale extraction.

Run this before any full processing to verify:
1. WSI opens correctly via SpatialDataImage2DReader
2. TileSpec computes expected base_width
3. add_tiles + iter.tile_images yields correct shapes
4. JPEG roundtrip works
5. Tar write + reread gives valid offsets
6. .idx binary format bytes match struct.calcsize
7. Random read from tar via idx offset returns correct data
"""

import io
import json
import struct
import tarfile
from pathlib import Path

import numpy as np
import spatialdata as sd
from PIL import Image
from spatialdata.transformations import get_transformation
from wsidata import TileSpec, open_wsi
from wsidata.io import add_tiles

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_PATH = Path("/path/to/your_sample.zarr")  # adapt to your data
IMAGE_KEY = "he_image"
SHAPES_KEY = "cell_circles"
PATCH_SIZE = 224
MPP_TGT = 0.5
TEST_OUT = Path("/tmp/mini_pipeline_test")
TEST_OUT.mkdir(parents=True, exist_ok=True)

IDX_MAGIC = b"CIDX0001"
IDX_FMT = "<iQQIQI"
IDX_SIZE = struct.calcsize(IDX_FMT)

print(f"IDX record size (struct.calcsize): {IDX_SIZE}")

# ── Load and derive ──────────────────────────────────────────────────────────
sdata = sd.read_zarr(INPUT_PATH)
cell_circles = sdata[SHAPES_KEY]

shape_to_global = get_transformation(
    sdata[SHAPES_KEY], to_coordinate_system="global"
).to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
SCALE_SHAPE = (shape_to_global[0][0] + shape_to_global[1][1]) / 2.0
SLIDE_MPP = 1.0 / SCALE_SHAPE

BASE_SIZE = round(PATCH_SIZE * MPP_TGT / SLIDE_MPP)
BASE_HALF = BASE_SIZE / 2.0

print(f"SLIDE_MPP={SLIDE_MPP}, SCALE_SHAPE={SCALE_SHAPE}")
print(f"BASE_SIZE={BASE_SIZE}, BASE_HALF={BASE_HALF}")

# ── Open WSI ─────────────────────────────────────────────────────────────────
wsi = open_wsi(sdata, image_key=IMAGE_KEY, store=None)
wsi.set_mpp(SLIDE_MPP)

spec = TileSpec.from_wsidata(wsi, tile_px=PATCH_SIZE, mpp=MPP_TGT, slide_mpp=SLIDE_MPP)
assert spec.base_width == BASE_SIZE, \
    f"FAIL: spec.base_width={spec.base_width} != BASE_SIZE={BASE_SIZE}"
print(f"PASS: spec.base_width={spec.base_width}")

# ── Test 5-cell extraction ───────────────────────────────────────────────────
# Use centroids of first 5 valid cells inside image
IMG_H, IMG_W = 23903, 22820
centroids = cell_circles.geometry.centroid
valid_indices = []
for i, c in enumerate(centroids):
    cx_g = c.x * SCALE_SHAPE
    cy_g = c.y * SCALE_SHAPE
    if BASE_HALF <= cx_g <= IMG_W - BASE_HALF and BASE_HALF <= cy_g <= IMG_H - BASE_HALF:
        valid_indices.append(i)
    if len(valid_indices) >= 5:
        break

xys = np.array([
    [centroids[i].x * SCALE_SHAPE - BASE_HALF,
     centroids[i].y * SCALE_SHAPE - BASE_HALF]
    for i in valid_indices
])

add_tiles(wsi, key="test_tiles", xys=xys, tile_spec=spec,
          tissue_ids=np.zeros(5, dtype=int))

# Extract and validate
patches = []
for j, tile in enumerate(wsi.iter.tile_images("test_tiles")):
    assert tile.image.shape == (224, 224, 3), \
        f"FAIL: shape={tile.image.shape}"
    assert tile.image.dtype == np.uint8, \
        f"FAIL: dtype={tile.image.dtype}"

    # JPEG roundtrip
    buf = io.BytesIO()
    Image.fromarray(tile.image).save(buf, format="JPEG", quality=92)
    jpg_bytes = buf.getvalue()
    decoded = Image.open(io.BytesIO(jpg_bytes))
    assert decoded.size == (224, 224), f"FAIL: decoded size={decoded.size}"

    patches.append(jpg_bytes)
    print(f"PASS: tile {j} shape={tile.image.shape} JPEG size={len(jpg_bytes)}")

# ── Test tar + idx write/read ────────────────────────────────────────────────
tar_path = TEST_OUT / "test.tar"
idx_path = TEST_OUT / "test.idx"
sample_keys = [f"TEST_{i:08d}" for i in range(5)]

shard_buf = []
for i in range(5):
    meta = {"sample_index": i, "sample_key": sample_keys[i], "cell_id": "test"}
    json_bytes = json.dumps(meta).encode()
    shard_buf.append((i, sample_keys[i], patches[i], json_bytes, meta))

# Phase 1: write tar
with tarfile.open(str(tar_path), "w") as tf:
    for si, sk, jpg, jsn, _ in shard_buf:
        for ext, data in [(".jpg", jpg), (".json", jsn)]:
            ti = tarfile.TarInfo(name=f"{sk}{ext}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))

# Phase 2: re-read for offsets
records = []
with tarfile.open(str(tar_path), "r") as tf:
    members = {m.name: m for m in tf.getmembers()}
    for si, sk, jpg, jsn, _ in shard_buf:
        jm = members[f"{sk}.jpg"]
        nm = members[f"{sk}.json"]
        records.append((si, jm.offset_data, jm.size, nm.offset_data, nm.size))

# Write idx
with open(str(idx_path), "wb") as f:
    f.write(IDX_MAGIC)
    f.write(struct.pack("<I", len(records)))
    for rec in records:
        f.write(struct.pack(IDX_FMT, *rec, 0))

# Phase 3: random read from idx
with open(str(idx_path), "rb") as f:
    magic = f.read(8)
    assert magic == IDX_MAGIC, f"FAIL: bad magic {magic!r}"
    n = struct.unpack("<I", f.read(4))[0]
    assert n == 5, f"FAIL: n_records={n}"
    for rec_idx in range(n):
        si, jo, js, no, ns, _ = struct.unpack(IDX_FMT, f.read(IDX_SIZE))
        with open(str(tar_path), "rb") as tf:
            tf.seek(jo)
            jpg_read = tf.read(js)
            tf.seek(no)
            json_read = tf.read(ns)
        decoded = Image.open(io.BytesIO(jpg_read))
        meta_read = json.loads(json_read)
        assert decoded.size == (224, 224), f"FAIL read: size={decoded.size}"
        assert meta_read["sample_index"] == si, \
            f"FAIL: sample_index mismatch {meta_read['sample_index']} != {si}"
        print(f"PASS: random read sample_index={si} offset={jo} jpg_size={js}")

# ── Cleanup ──────────────────────────────────────────────────────────────────
import shutil
shutil.rmtree(TEST_OUT)

print("\n" + "=" * 50)
print("ALL CHECKS PASSED — mini pipeline ready for full-scale extraction")
print("=" * 50)
