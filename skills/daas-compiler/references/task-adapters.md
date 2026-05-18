# Task Adapters

Task adapters convert L3 (dataset-compiled) artifacts into L4 (training-ready) artifacts
for a specific training task. Each adapter is task-specific and produces a different
output layout.

---

## HE2ST (H&E → Spatial Transcriptomics)

Predict gene expression from H&E patch morphology.

| Property | Value |
|---|---|
| Input | L3 compiled manifest.parquet + expression.h5ad + per-sample JPEG shards |
| Target | Gene expression vector (normalized, gene-panel ordered) |
| Sample unit | Cell |
| Task type | `he2st` |

### Required L4 artifacts

- `train/shard-NNNNNN.tar`, `val/...`, `test/...` — WebDataset shards by split
- `gene_panel.json` — ordered gene names
- `gene_panel.sha256` — SHA-256 of gene_panel.json
- `task_config.yaml` — task_type, n_genes, gene_panel_path, input_modality, target_modality
- `loader_config.yaml` — train/val/test shard glob patterns, recommended batch_size, num_workers
- `dataset_card.json` — n_cells, n_genes, sample_ids, split counts, daas_version, created_at
- `validation_report.json` — per-split cell counts, gene panel hash check, shard count vs manifest count
- `split_report.json` — method, seed, per-sample per-split counts

### Per-cell shard content

Each tar entry contains:
- `{key}.jpg` — JPEG patch
- `{key}.expr.npz` — sparse expression (`indices`, `values`, `n_genes`)
- `{key}.json` — `{global_idx, sample_id, cell_id, split, task, n_genes, gene_panel_sha256}`

### Split requirements

- Train/val/test split materialized into separate shard directories before packaging
- Splits are assigned at the cell level; cell-level splits must not mix samples in val/test
- `split_report.json` must record split assignment method and per-sample counts

### Loader-ready layout

```
{output}/
  train/
    shard-000000.tar
    shard-000001.tar
    ...
  val/
    shard-000000.tar
    ...
  test/
    shard-000000.tar
    ...
  gene_panel.json
  gene_panel.sha256
  task_config.yaml
  loader_config.yaml
  dataset_card.json
  validation_report.json
  split_report.json
```

### Validation checks

- Gene panel consistency: `gene_panel.json` SHA-256 matches `gene_panel_sha256` in each per-cell JSON
- Shard cell count vs manifest cell count: must be equal per split
- No cell appears in more than one split
- All required files are present and non-empty

---

## Cell Type Classification

Predict cell type label from H&E patch morphology.

| Property | Value |
|---|---|
| Input | L3 compiled patches + cell type annotations from obs table |
| Target | Cell type label (integer class index) |
| Sample unit | Cell |
| Task type | `cell_type_classification` |

### Required L4 artifacts

- `train/`, `val/`, `test/` — WebDataset shards by split
- `label_map.json` — `{class_name: class_index}` mapping
- `task_config.yaml` — task_type, n_classes, label_map_path
- `loader_config.yaml` — shard patterns, batch_size, num_workers
- `dataset_card.json` — n_cells, n_classes, class_distribution, split counts
- `validation_report.json` — class distribution per split, label consistency check
- `split_report.json` — method, seed, per-class per-split counts

### Per-cell shard content

- `{key}.jpg` — JPEG patch
- `{key}.cls` — 4-byte little-endian int32 (class index)
- `{key}.json` — `{global_idx, sample_id, cell_id, split, task, class_name, class_idx}`

### Split requirements

Stratified split recommended: each split should preserve the overall class distribution.

---

## Contrastive / Multimodal Pretraining

Pair H&E patches with gene expression vectors for contrastive learning.

| Property | Value |
|---|---|
| Input | L3 compiled patches + expression.h5ad |
| Target | Paired (image, expression) for contrastive loss |
| Sample unit | Cell — positive pair is same cell's image + expression |
| Task type | `contrastive_he_expr` |

### Required L4 artifacts

- `train/`, `val/` — WebDataset shards (test split optional for pretraining)
- `gene_panel.json` + `gene_panel.sha256`
- `task_config.yaml` — task_type, contrastive_mode (`"image_expression"`), n_genes
- `loader_config.yaml` — shard patterns, recommended batch_size (large for contrastive)
- `dataset_card.json`
- `validation_report.json`

### Per-cell shard content

Same as HE2ST: `{key}.jpg` + `{key}.expr.npz` + `{key}.json`.
The loader pairs image and expression from the same tar entry as the positive pair.

### Split requirements

Random split at the cell level. No per-sample stratification required.

---

## Future: Spatial Neighborhood Tasks

Predict cell identity or behavior from the morphological context of its spatial neighborhood.

| Property | Value |
|---|---|
| Input | L3 patches + spatial graph (cell adjacency) |
| Target | Neighbor context embedding or neighborhood composition |
| Sample unit | Cell + k-hop neighborhood |
| Task type | `spatial_neighborhood` (placeholder) |

Contract TBD when implemented. Requires spatial graph export from SpatialData at L3.
