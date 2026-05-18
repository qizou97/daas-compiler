# Workflow Planning Reference

## How to read a natural-language request

When a user gives you a natural-language data processing request,
call `daas.planning.parse_stage_plan(text)` or read it yourself and map it
to stages using the trigger phrase tables in `references/filtering-recipes.md`.

## Stage ordering

Stages always run in this order (skip stages that are not triggered):

1. `inspect` — always first; print zarr contents so the user can verify keys
2. `tissue_inside` — filter cells outside tissue (SOPA if needed)
3. `nucleus_presence` — keep only cells with nucleus_boundaries entry
4. `xenium_he_nucleus_overlap` — keep cells with IoU >= threshold
5. `extract` — HE patch extraction using the final table key
6. `compile` — merge per-sample outputs into a unified dataset

## render_cli output

`daas.planning.render_cli(plan, zarr_paths, output_dir)` returns a shell
script string. Present it to the user as a code block for review before running.
The user must run each stage script themselves (or you may run them via Bash).

## filtered_table is optional

`filtered_table` (a stVisuome-precomputed table) is NOT required.
It is just one possible `--input-table-key` value. Pass it explicitly:
`--input-table-key filtered_table` on the first stage script if present.
If not present, start from `--input-table-key table`.

## Example: full worked example

Request:
> "Process A_001,A_002,A_004 under /home/zouqi/datasets/mash/spatialdata into
> cell-centered HE patches. Filter out cells outside tissue and only keep cells
> with nucleus boundaries. Target mpp=0.5, patch size=224, use optim_ops_level,
> output to /home/zouqi/datasets/mash/stvisuome, sample 3000 cells per sample,
> compile, and write bundled WebDataset shards."

```python
from daas.planning import parse_stage_plan, render_cli

plan = parse_stage_plan(
    "Process A_001,A_002,A_004 under /home/zouqi/datasets/mash/spatialdata "
    "into cell-centered HE patches. Filter out cells outside tissue and only "
    "keep cells with nucleus boundaries. Target mpp=0.5, patch size=224, "
    "use optim_ops_level, output to /home/zouqi/datasets/mash/stvisuome, "
    "sample 3000 cells per sample, compile, and write bundled WebDataset shards."
)
zarr_paths = [
    "/home/zouqi/datasets/mash/spatialdata/A_001.zarr",
    "/home/zouqi/datasets/mash/spatialdata/A_002.zarr",
    "/home/zouqi/datasets/mash/spatialdata/A_004.zarr",
]
print(render_cli(plan, zarr_paths, "/home/zouqi/datasets/mash/stvisuome"))
```

Resolves to:
- stages: `[tissue_inside, nucleus_presence]`
- `extract_args`: `table_key="table_tissue_nucleus"`, `extract_mode="full_ops_level"`, `mpp=0.5`, `patch_size=224`, `n_sample=3000`
- `compile_args`: `samples=["A_001","A_002","A_004"]`, `bundle_wds=True`
