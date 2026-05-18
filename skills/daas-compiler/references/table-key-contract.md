# Table-Key Contract

## Rule: every stage reads one table key and writes one table key

Each filtering stage script:
- Reads: `sdata.tables[--input-table-key]`
- Writes: `sdata.tables[--output-table-key]` (persisted to zarr)
- Reports: JSON in `--report-dir`

## Naming convention (auto-names)

| Stage script | Suffix appended |
|---|---|
| filter_tissue.py | `_tissue` |
| filter_nucleus_presence.py | `_nucleus` |
| filter_nucleus_overlap.py | `_he` |

Example chain:
```
table  →(tissue)→  table_tissue  →(nucleus)→  table_tissue_nucleus
    →(overlap)→  table_tissue_nucleus_he
```

## extract_sample.py consumes the final key

The final `output_table_key` from the last stage is passed as
`--table-key <FINAL_KEY>` to `extract_sample.py`. The planner does this
automatically. Never extract from a stale earlier key after filtering.

## filtered_table is just a key

If a zarr already contains `filtered_table` (e.g. produced by stVisuome),
pass it as the starting point:
```bash
python3 scripts/filter_nucleus_presence.py \
    --input-table-key filtered_table \
    --output-table-key filtered_table_nucleus ...
```
No special handling needed.

## invariants

- `extract_sample.py` asserts `--table-key` exists in `sdata.tables`. If not,
  it prints the available keys and exits with a clear error.
- Stage report `output_table_key` must match what was actually written to zarr.
- `compile_dataset.py` reads only from per-sample `expression.h5ad` — stage
  table keys are upstream concerns, invisible at compile time.
