# Contributing

## Commit Format

This repo uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <description>

[optional body]

[optional footer: Co-Authored-By, Fixes, etc.]
```

### Types

Standard conventional commit types apply: `feat`, `fix`, `chore`, `docs`,
`refactor`, `test`, `perf`, `ci`.

### Allowed Scopes

| Scope | Covers |
|---|---|
| `skill` | SKILL.md, agent behavior contract |
| `docs` | Documentation files (README, CONTRIBUTING, VERSIONING, RELEASE) |
| `planning` | daas/planning.py, stage plan parsing, CLI rendering |
| `filters` | daas/filters/, filter stage scripts |
| `extract` | scripts/extract_sample.py, scripts/extract_all.py |
| `compile` | scripts/compile_dataset.py |
| `viz` | daas/viz.py, scripts/viz_sample.py |
| `dataset` | daas/dataset.py, CellPatchDataset, BundledCellPatchDataset |
| `wds` | WebDataset integration, bundled shard writing |
| `genes` | Gene panel, gene intersection logic |
| `reports` | daas/reports.py, filter_report, compile_report, validation_report |
| `tasks` | Task adapters, make_task_dataset.py |
| `splits` | Split assignment, split_report |
| `loaders` | loader_config.yaml, loader utilities |
| `deps` | pyproject.toml dependency changes |
| `release` | Version bumps, changelog, tagging |
| `tests` | tests/ |

### Examples

```
feat(extract): add --tissue-shapes-key flag for flexible shape key selection
fix(compile): correct gene intersection when sample has zero common genes
chore(release): bump to v0.7.0
docs(skill): add training-ready contract section
refactor(dataset): extract LRUMmapCache into its own module
```

## PR Checklist

Before opening a pull request:

- [ ] `python -m compileall skills/daas-compiler` passes (no syntax errors)
- [ ] `cd skills/daas-compiler && pytest tests -q` passes (no failures or errors)
- [ ] If an output schema changed: schema version field bumped and documented in `VERSIONING.md`
- [ ] If a task-ready output changed: `references/training-ready-contract.md` and `references/task-adapters.md` updated
- [ ] If a new dependency was added: follows `references/dependency-policy.md` (correct group, lazy import for preprocess)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Commit messages use allowed scopes from the table above

## Test Expectations

- Tests live in `skills/daas-compiler/tests/`
- Tests must not require the `extract` group (spatialdata, wsidata, lazyslide) unless
  explicitly testing extraction. Guard with `pytest.importorskip` where needed.
- Tests that require SOPA must guard with `pytest.importorskip("sopa")`
- The lightweight `[test]` group provides geopandas and shapely for filter integration tests
- New behavior must have tests before the PR merges

## Dependency Rules

See `references/dependency-policy.md` for the full policy. Key points:

- SOPA must be lazy-imported inside functions that use it
- New heavy dependencies must be optional extras, not added to core
- `requirements.txt` covers extraction/viz runtime; do not add SOPA or task extras there

## Output Contract Changes

Any change that alters the shape, fields, or semantics of output files must:

1. Bump the affected schema version(s) — see `VERSIONING.md`
2. Update `references/training-ready-contract.md` if L4 artifacts are affected
3. Update `references/task-adapters.md` if the relevant task adapter's artifact list changes
4. Update or add tests that assert the new schema

Changing output contracts silently is not acceptable, even for "minor" field additions.
