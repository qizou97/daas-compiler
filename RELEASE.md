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

# Verify L4 outputs exist
ls /tmp/daas_smoke/task_ready/
# Expected: train/, val/, test/, gene_panel.json, gene_panel.sha256,
#           task_config.yaml, loader_config.yaml, dataset_card.json,
#           validation_report.json, split_report.json
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
git commit -m "chore(release): bump to vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

## 7. Post-Release

- Verify the tag is visible on GitHub: `git ls-remote --tags origin`
- If the skill is published to the marketplace, update marketplace metadata
- Announce any breaking changes (schema version bumps, contract changes) in the
  GitHub release notes
