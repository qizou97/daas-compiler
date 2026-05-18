# Training-Ready Contract

## Definition

A dataset is **training-ready** (L4) only when it can be consumed directly by the
intended training loader **without** any of the following additional steps:

- Preprocessing (normalization, filtering, gene reordering)
- Joining (merging manifest with h5ad)
- Splitting (train/val/test materialization)
- Gene reordering (aligning expression vectors to a fixed gene panel)
- Image conversion (JPEG → tensor)
- Artifact conversion (h5ad → tensor, parquet → index)

If the training code must perform any of these steps, the dataset is not training-ready.

## Explicit Non-Examples

| Artifact | Level | Why NOT training-ready |
|---|---|---|
| Per-sample JPEG shards + expression.h5ad + manifest.parquet | L2 | Requires joining, splitting, global gene panel |
| Compiled global manifest.parquet + expression.h5ad | L3 | Requires splitting, task config, loader config |
| Bundled WDS shards without split files or gene_panel.json | L3b | Incomplete: missing split materialization and metadata |

## HE2ST Training-Ready Requirements

A dataset compiled for H&E → Spatial Transcriptomics prediction is training-ready when
ALL of the following artifacts are present and consistent:

### Required files

| File | Description |
|---|---|
| `train/shard-NNNNNN.tar`, `val/...`, `test/...` | WebDataset shards organized by split |
| `gene_panel.json` | Ordered list of gene names matching expression vector indices |
| `gene_panel.sha256` | SHA-256 hash of gene_panel.json for reproducibility |
| `task_config.yaml` | Task type, input/target modalities, gene panel ref, n_genes |
| `loader_config.yaml` | Shard paths, split sizes, batch size recommendations, num_workers |
| `dataset_card.json` | Dataset provenance: n_cells, n_genes, sample_ids, creation date, daas version |
| `validation_report.json` | Per-split cell counts, gene panel consistency check, shard integrity |
| `split_report.json` | Split assignment method, seed, per-sample split counts |

### Required per-cell shard content

Each tar entry must contain:
- `{key}.jpg` — JPEG-encoded patch image
- `{key}.expr.npz` — sparse expression target (`indices`, `values`, `n_genes`)
- `{key}.json` — metadata with required fields:

```json
{
  "global_idx": 12345,
  "sample_id": "A_002",
  "cell_id": "cell_000123",
  "split": "train",
  "task": "he2st",
  "n_genes": 313,
  "gene_panel_sha256": "abc123..."
}
```

### Required json fields

| Field | Type | Description |
|---|---|---|
| `global_idx` | int | Row index in the L3 compiled manifest/h5ad |
| `sample_id` | str | Sample identifier |
| `cell_id` | str | Original cell identifier from SpatialData |
| `split` | str | One of: `train`, `val`, `test` |
| `task` | str | Task type (e.g. `"he2st"`) |
| `n_genes` | int | Length of expression vector |
| `gene_panel_sha256` | str | SHA-256 of `gene_panel.json` for loader-side validation |

## Summary

> Patch extraction (L2) is patch-compiled, not training-ready.
> Compiled h5ad + manifest (L3) is dataset-compiled, not training-ready.
> Training-ready (L4) requires task-specific splits, gene panel, loader config, and validation artifacts.

See `references/artifact-levels.md` for the full level hierarchy.
See `references/task-adapters.md` for task-specific L4 layouts.
