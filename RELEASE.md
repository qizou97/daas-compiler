# Release Checklist

Use this checklist when cutting a new release from the `main` branch.

## 1. Pre-Release Validation

Run both commands and confirm exit code 0:

```bash
python -m compileall skills/daas-compiler
```
Expected: `Compiling ...` lines for each .py file, no `SyntaxError` output, exit code 0.

```bash
cd skills/daas-compiler && pytest tests -q
```
Expected: all tests pass. Exit code 0. No failures, errors, or xfail surprises.

## 2. Mini-Pipeline Smoke Test

Run a minimal end-to-end extraction + compile. Use an existing small test zarr or the
test fixtures in `skills/daas-compiler/tests/`:

```bash
# Extraction smoke test (requires pip install -e .[extract])
python3 skills/daas-compiler/scripts/extract_sample.py \
    --zarr <path_to_small_test_zarr> \
    --output /tmp/daas_smoke/sample_A \
    --n-sample 50 \
    --extract-mode tile_images

# Verify L2 outputs exist
ls /tmp/daas_smoke/sample_A/
# Expected: shard-000000.tar, shard-000000.idx, expression.h5ad, manifest.parquet,
#           filter_report.json, viz/viz_global_tiles.png, viz/viz_patch_grid.png

# Compile smoke test
python3 skills/daas-compiler/scripts/compile_dataset.py \
    --per-sample-dir /tmp/daas_smoke \
    --output /tmp/daas_smoke/compiled

# Verify L3 outputs exist
ls /tmp/daas_smoke/compiled/
# Expected: manifest.parquet, expression.h5ad

# Verify gene panel (produced with --bundle-wds or by make_task_dataset)
python3 skills/daas-compiler/scripts/compile_dataset.py \
    --per-sample-dir /tmp/daas_smoke \
    --output /tmp/daas_smoke/compiled_wds \
    --bundle-wds

ls /tmp/daas_smoke/compiled_wds/
# Expected: manifest.parquet, expression.h5ad, gene_panel.json, sample_A/shard-*.tar
```

## 3. HE2ST Task-Ready Smoke Test

If the HE2ST task adapter (`make_task_dataset.py`) is available:

```bash
python3 skills/daas-compiler/scripts/make_task_dataset.py \
    --compiled-dir /tmp/daas_smoke/compiled \
    --output /tmp/daas_smoke/task_ready \
    --task he2st \
    --split-ratios 0.8 0.1 0.1 \
    --seed 42

# Verify canonical L4 layout (data/ + splits/ — NOT train/val/test dirs)
ls /tmp/daas_smoke/task_ready/
# Expected: data/, splits/, gene_panel.json, gene_panel.sha256,
#           task_config.yaml, loader_config.yaml, dataset_card.json,
#           validation_report.json

ls /tmp/daas_smoke/task_ready/data/
# Expected: shard-000000.tar  (all cells, not split-partitioned)

ls /tmp/daas_smoke/task_ready/splits/
# Expected: train.json, val.json, test.json, split_membership.parquet, split_report.json

# Confirm no physical train/val/test shard directories exist by default
test ! -d /tmp/daas_smoke/task_ready/train && echo "OK: no physical train/ dir"
test ! -d /tmp/daas_smoke/task_ready/val   && echo "OK: no physical val/ dir"
test ! -d /tmp/daas_smoke/task_ready/test  && echo "OK: no physical test/ dir"

# Verify gene panel integrity
python3 -c "
import json, hashlib
gp = open('/tmp/daas_smoke/task_ready/gene_panel.json').read()
sha = hashlib.sha256(gp.encode()).hexdigest()
stored = open('/tmp/daas_smoke/task_ready/gene_panel.sha256').read().strip()
assert sha == stored, f'gene_panel.sha256 mismatch: {sha} vs {stored}'
print('gene_panel.sha256 OK')
"

# Verify loader can select splits at runtime from split metadata
python3 -c "
import json, pandas as pd
membership = pd.read_parquet('/tmp/daas_smoke/task_ready/splits/split_membership.parquet')
train_idx = json.load(open('/tmp/daas_smoke/task_ready/splits/train.json'))
val_idx   = json.load(open('/tmp/daas_smoke/task_ready/splits/val.json'))
test_idx  = json.load(open('/tmp/daas_smoke/task_ready/splits/test.json'))
assert set(train_idx).isdisjoint(val_idx), 'train/val overlap'
assert set(train_idx).isdisjoint(test_idx), 'train/test overlap'
assert set(val_idx).isdisjoint(test_idx), 'val/test overlap'
assert len(membership) == len(train_idx) + len(val_idx) + len(test_idx), 'count mismatch'
print(f'Split metadata OK: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}')
"
```

If the task adapter is not yet implemented, mark this step N/A and note it in the
release changelog.

## 4. Version Bump

Update `version` in `skills/daas-compiler/pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

## 5. Changelog Update

In `CHANGELOG.md`:

1. Add a new version section above `[Unreleased]`:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

2. Move all items from `[Unreleased]` into the new version section.
3. Leave `[Unreleased]` empty (with placeholder subsections) for future entries.

## 6. Commit and Tag

```bash
git add skills/daas-compiler/pyproject.toml CHANGELOG.md
git commit -m "release: bump to vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 7. Post-Release

- Verify the tag is visible on GitHub: `git ls-remote --tags origin`
- If the skill is published to the marketplace, update marketplace metadata
- Announce any breaking changes (schema version bumps, contract changes) in the
  GitHub release notes

## Release Checklist (tick all before tagging)

- [ ] `python -m compileall skills/daas-compiler` passed (exit 0)
- [ ] `pytest tests -q` passed (exit 0)
- [ ] Mini extraction smoke test passed: L2 outputs exist
- [ ] Compile smoke test passed: manifest.parquet, expression.h5ad exist
- [ ] `gene_panel.json` exists (from bundle-wds or task adapter)
- [ ] `gene_panel.sha256` exists and matches gene_panel.json content
- [ ] `expression.var_names` exactly matches gene_panel.json gene list
- [ ] Task-ready smoke test: `data/shard-*.tar` present (or N/A)
- [ ] Task-ready smoke test: `splits/split_membership.parquet` present (or N/A)
- [ ] Task-ready smoke test: `splits/train.json`, `val.json`, `test.json` present (or N/A)
- [ ] Task-ready smoke test: no physical `train/`, `val/`, `test/` shard dirs by default
- [ ] Task-ready smoke test: loader selects splits at runtime via split metadata
- [ ] Task-ready smoke test: `loader_config.yaml` present and references split metadata paths
- [ ] Task-ready smoke test: `validation_report.json` present (or N/A)
- [ ] Post-save tile visualization present in viz/ (or explicitly skipped with report)
- [ ] `CHANGELOG.md` updated: unreleased items moved to version section
- [ ] `pyproject.toml` version bumped
