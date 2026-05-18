# Filtering Recipes Reference

Three filtering recipes are available as stage scripts.

---

## Recipe 1: tissue_inside

**Script:** `scripts/filter_tissue.py`  
**Trigger phrases:** "inside tissue", "out of tissue", "outside tissue", "tissue filter"

**What it does:**
1. Checks `sdata.shapes` for tissue polygon (default key: `tissue_boundaries`).
2. If absent: calls `sopa.segmentation.tissue(sdata, image_key=...)`.
3. Keeps cells whose centroid lies inside any tissue polygon.
4. Writes filtered table to zarr + `StageReport` JSON.

**CLI:**
```bash
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr /data/A_001.zarr \
    --input-table-key table \
    --output-table-key table_tissue \
    [--tissue-key tissue_boundaries] \
    [--image-key he_image]
```

**Output table key:** `{input_table_key}_tissue` (e.g. `table_tissue`)  
**Drop reason key:** `outside_tissue`

---

## Recipe 2: nucleus_presence

**Script:** `scripts/filter_nucleus_presence.py`  
**Trigger phrases:** "with nucleus boundaries", "has nucleus", "only cells with nucleus", "nucleus boundary"

**What it does:**
1. Loads `sdata.shapes[nucleus_boundaries_key]`.
2. Keeps rows where `obs["cell_id"]` appears in `nucleus_boundaries.index`.
3. Writes filtered table to zarr + `StageReport` JSON.
4. No SOPA call — fails if `nucleus_boundaries` does not exist.

**CLI:**
```bash
python3 ${SKILL_DIR}/scripts/filter_nucleus_presence.py \
    --zarr /data/A_001.zarr \
    --input-table-key table_tissue \
    --output-table-key table_tissue_nucleus \
    [--nucleus-boundaries-key nucleus_boundaries]
```

**Output table key:** `{input_table_key}_nucleus`  
**Drop reason key:** `missing_nucleus_boundary`

---

## Recipe 3: xenium_he_nucleus_overlap

**Script:** `scripts/filter_nucleus_overlap.py`  
**Trigger phrases:** "Xenium nucleus overlaps HE nucleus", "HE nucleus", "nucleus overlap", "overlap >"

**What it does:**
1. Checks `sdata.shapes[he_nucleus_boundaries]`.
2. If absent: calls `sopa.segmentation.cellpose(sdata, image_key=...)`.
3. Computes per-cell IoU between Xenium `nucleus_boundaries` and nearest HE nucleus.
4. Keeps cells where `IoU >= --overlap-threshold` (default 0.5).
5. Writes filtered table to zarr + `StageReport` JSON.

**CLI:**
```bash
python3 ${SKILL_DIR}/scripts/filter_nucleus_overlap.py \
    --zarr /data/A_001.zarr \
    --input-table-key table_tissue_nucleus \
    --output-table-key table_tissue_nucleus_he \
    [--overlap-threshold 0.5] \
    [--he-nucleus-key he_nucleus_boundaries]
```

**Output table key:** `{input_table_key}_he`  
**Drop reason keys:** `no_xenium_nucleus`, `low_he_overlap`
