# Task Adapters

Task adapters convert L3 (dataset-compiled) artifacts into L4 (training-ready) artifacts
for a specific training task. Each adapter is task-specific and produces a different
output layout.

**Split policy:** Splits are metadata applied by the loader at runtime. The default
output layout stores all cells in a single canonical `data/` directory; physical
train/val/test shard directories are an optional export mode only (flag:
`--materialize-split-shards`), not the default.

---

## HE2ST (H&E → Spatial Transcriptomics)

Predict gene expression from H&E patch morphology.

| Property | Value |
|---|---|
| Task type | `he2st` |
| Input | H&E image patch |
| Target | Gene expression vector (normalized, gene-panel ordered) |
| Sample unit | Cell / spot / region |
| Default storage | Canonical WebDataset (`data/shard-*.tar`) — not split-partitioned |
| Default split behavior | Runtime filtering from `splits/split_membership.parquet` |
| Default target | Sparse expression vector ordered by `gene_panel.json` |

### Required L4 artifacts

**Canonical storage:**
- `data/shard-NNNNNN.tar` — all cells, not partitioned by split
- `data/bundled_manifest.parquet` — optional per-shard cell index

**Split metadata:**
- `splits/train.json` — ordered list of `global_idx` values → train
- `splits/val.json` — ordered list of `global_idx` values → val
- `splits/test.json` — ordered list of `global_idx` values → test
- `splits/split_membership.parquet` — per-cell: `global_idx`, `sample_id`, `cell_id`, `split`
- `splits/split_report.json` — method, seed, per-sample per-split counts

**Task metadata:**
- `gene_panel.json` — ordered gene names
- `gene_panel.sha256` — SHA-256 of gene_panel.json
- `task_config.yaml` — task_type, n_genes, gene_panel_path, input_modality, target_modality
- `loader_config.yaml` — shard glob pattern(s), split metadata paths, recommended batch_size, num_workers
- `dataset_card.json` — n_cells, n_genes, sample_ids, split counts, daas_version, created_at
- `validation_report.json` — per-split cell counts, gene panel hash check, shard count vs manifest count

### Per-cell shard content

Each tar entry contains:
- `{key}.jpg` — JPEG patch
- `{key}.expr.npz` — sparse expression (`indices`, `values`, `n_genes`)
- `{key}.json` — `{global_idx, sample_id, cell_id, task, n_genes, gene_panel_sha256}`

Note: `split` is **not** baked into per-cell JSON. The loader determines split at runtime
from `splits/split_membership.parquet`.

### Split requirements

- Split assignment must be **sample-level or group-level**.
  All cells from the same `sample_id` must be assigned to the same split.
  DAAS does not generate random cell-level train/val/test splits.
- `split_membership.parquet` is indexed by `global_idx` for loader efficiency,
  but generated rows inherit split from `sample_id` / `patient_id` / `donor_id`
  / `slide_id` / `batch_id` / group column.
- For `sample_holdout`: no `sample_id` may appear in more than one split.
- For `ratio_by_group` and `group_kfold`: no group may appear in more than one split.
- `splits/split_membership.parquet` must cover every cell in the dataset.
- `splits/split_report.json` must record split assignment method, grouping column,
  seed, and per-sample per-split cell counts.
- Physical train/val/test shard directories are an optional export mode, not default.

### Default loader-ready layout

```
{task_output}/
  data/
    shard-000000.tar
    shard-000001.tar
    ...
    bundled_manifest.parquet   (optional)

  splits/
    train.json
    val.json
    test.json
    split_membership.parquet
    split_report.json

  gene_panel.json
  gene_panel.sha256
  task_config.yaml
  loader_config.yaml
  dataset_card.json
  validation_report.json
```

### Alternative: reuse compiled/ as base

```
compiled/
  manifest.parquet
  expression.h5ad
  gene_panel.json
  gene_panel.sha256
  {sample_id}/
    shard-*.tar

  splits/
    he2st_train.json
    he2st_val.json
    he2st_test.json
    he2st_split_membership.parquet
    he2st_split_report.json

  tasks/
    he2st/
      task_config.yaml
      loader_config.yaml
      dataset_card.json
      validation_report.json
```

### Validation checks

- Gene panel consistency: `gene_panel.json` SHA-256 matches `gene_panel_sha256` in each per-cell JSON
- Shard cell count vs `split_membership.parquet` count: must be equal per split
- No cell appears in more than one split
- All required files are present and non-empty
- `loader_config.yaml` references valid shard glob(s) and split metadata paths

---

## Cell Type Classification

Predict cell type label from H&E patch morphology.

| Property | Value |
|---|---|
| Task type | `cell_type_classification` |
| Input | L3 compiled patches + cell type annotations from obs table |
| Target | Cell type label (integer class index) |
| Sample unit | Cell |

### Required L4 artifacts

- `data/shard-*.tar` — canonical storage (not split-partitioned)
- `splits/` — split_membership.parquet, train/val/test JSON, split_report.json
- `label_map.json` — `{class_name: class_index}` mapping
- `task_config.yaml` — task_type, n_classes, label_map_path
- `loader_config.yaml` — shard glob patterns, split metadata paths, batch_size, num_workers
- `dataset_card.json` — n_cells, n_classes, class_distribution, split counts
- `validation_report.json` — class distribution per split, label consistency check

### Per-cell shard content

- `{key}.jpg` — JPEG patch
- `{key}.cls` — 4-byte little-endian int32 (class index)
- `{key}.json` — `{global_idx, sample_id, cell_id, task, class_name, class_idx}`

### Split requirements

Split assignment must be sample-level or group-level (`sample_holdout` or
`ratio_by_group` recommended). All cells from the same sample_id must be
assigned to the same split. DAAS does not generate random cell-level splits.
Stratified split by class distribution is recommended at the group level.

---

## Contrastive / Multimodal Pretraining

Pair H&E patches with gene expression vectors for contrastive learning.

| Property | Value |
|---|---|
| Task type | `contrastive_he_expr` |
| Input | L3 compiled patches + expression.h5ad |
| Target | Paired (image, expression) for contrastive loss |
| Sample unit | Cell — positive pair is same cell's image + expression |

### Required L4 artifacts

- `data/shard-*.tar` — canonical storage
- `splits/` — split_membership.parquet, train/val JSON (test optional for pretraining)
- `gene_panel.json` + `gene_panel.sha256`
- `task_config.yaml` — task_type, contrastive_mode (`"image_expression"`), n_genes
- `loader_config.yaml` — shard glob patterns, recommended batch_size (large for contrastive)
- `dataset_card.json`
- `validation_report.json`

### Per-cell shard content

Same as HE2ST: `{key}.jpg` + `{key}.expr.npz` + `{key}.json`.
The loader pairs image and expression from the same tar entry as the positive pair.

### Split requirements

Split assignment must be sample-level or group-level (`ratio_by_group` recommended).
All cells from the same sample must be assigned to the same split.
DAAS does not generate random cell-level splits for this task either.

---

## Future: Spatial Neighborhood Tasks

Predict cell identity or behavior from the morphological context of its spatial neighborhood.

| Property | Value |
|---|---|
| Task type | `spatial_neighborhood` (placeholder) |
| Input | L3 patches + spatial graph (cell adjacency) |
| Target | Neighbor context embedding or neighborhood composition |
| Sample unit | Cell + k-hop neighborhood |

Contract TBD when implemented. Requires spatial graph export from SpatialData at L3.
