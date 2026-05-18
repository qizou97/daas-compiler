# Agent Contract: Natural Language → Stage Plan

## Overview

When a user makes a natural-language spatial transcriptomics training request,
the agent must produce an explicit **stage plan** before executing any scripts.
The agent must not begin extraction, compilation, or packaging without first
presenting the stage plan for user review.

## Required Stage Plan Fields

A valid stage plan must specify all of the following:

| Field | Description | Example |
|---|---|---|
| `samples` | List of sample identifiers | `["A_001", "A_002", "A_004"]` |
| `zarr_paths` | Absolute path to each sample's zarr | `["/data/A_001.zarr", ...]` |
| `task_type` | Training task | `"he2st"` |
| `input_modality` | Input data type | `"he_image"` |
| `target_modality` | Target data type | `"gene_expression"` |
| `initial_table_key` | Starting table key in zarr | `"table"` |
| `initial_image_key` | H&E image key in zarr | `"he_image"` |
| `initial_shape_key` | Cell shape key in zarr | `"cell_circles"` |
| `filter_stages` | Ordered list of filter stages to apply | `["tissue_inside", "nucleus_presence"]` |
| `final_table_key` | Table key after all filter stages | `"table_tissue_nucleus"` |
| `final_shape_key` | Shape key to use at extraction | `"cell_circles"` |
| `extraction_config` | extract_sample.py parameters | `{mpp: 0.5, patch_size: 224, extract_mode: "full_ops_level", n_sample: 3000}` |
| `compile_config` | compile_dataset.py parameters | `{bundle_wds: true}` |
| `task_ready_config` | Task adapter parameters | `{task: "he2st", split_ratios: [0.8, 0.1, 0.1]}` |
| `split_config` | Split assignment config | `{method: "random", seed: 42, ratios: {train: 0.8, val: 0.1, test: 0.1}}` |
| `loader_config` | Loader-ready output config | `{format: "webdataset", shard_size: 500}` |
| `reports` | Reports and validation outputs to produce | `["filter_report", "compile_report", "validation_report", "split_report"]` |

## Training-Ready Gate

The agent must NOT describe outputs as "training-ready" unless the stage plan includes:

1. A `task_ready_config` specifying a task type and split configuration
2. A `split_config` with explicit train/val/test ratios
3. A `loader_config` specifying the output format
4. `"validation_report"` and `"split_report"` in the `reports` list

If the plan only covers filtering + extraction + compile (L2/L3), the agent must
describe the result as "patch-compiled" or "dataset-compiled" — not "training-ready."

## Stage Plan → CLI Mapping

After presenting and receiving user approval of the stage plan, the agent renders
the corresponding CLI commands using `daas.planning.render_cli()` or equivalent.

See `references/workflow-planning.md` for the stage → script mapping.
See `references/artifact-levels.md` for the L2/L3/L4 distinction.
See `references/training-ready-contract.md` for what L4 requires.
