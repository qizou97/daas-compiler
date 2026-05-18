# Contributing

## Commit Format

This repo uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <description>

[optional body]

[optional footer: Co-Authored-By, Fixes, etc.]
```

### Types

| Type | When to use |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | Code restructure without behavior change |
| `perf` | Performance improvement with identical output contract |
| `chore` | Tooling, CI, non-user-facing maintenance |
| `release` | Version bumps, changelog, tagging |
| `deps` | Dependency changes (pyproject.toml extras, requirements files) |

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
| `genes` | Gene panel, gene intersection logic, gene_panel.json |
| `reports` | daas/reports.py, filter_report, compile_report, validation_report |
| `tasks` | Task adapters, make_task_dataset.py |
| `splits` | Split assignment, split_membership.parquet, split metadata |
| `loaders` | loader_config.yaml, loader utilities, runtime split selection |
| `deps` | pyproject.toml dependency changes |
| `release` | Version bumps, changelog, tagging |
| `tests` | tests/ |

### Examples

```
feat(tasks): add HE2ST task-ready adapter with split metadata layout
fix(genes): preserve gene panel order across samples
docs(skill): define DAAS artifact levels
feat(viz): add post-save saved-tile validation grid
deps(preprocess): add SOPA optional dependency group
release: bump version to 0.6.2
feat(splits): add split_membership.parquet as default split representation
fix(loaders): loader_config references correct split metadata paths
```

## Rules

- `feat` requires tests or an explicit documented reason why tests are not practical.
- `fix` requires a regression test when practical.
- Output artifact schema changes must bump the affected schema version and update docs.
- Task-ready layout changes must update `references/training-ready-contract.md` and `references/task-adapters.md`.
- Split behavior changes must update split contract docs (`references/training-ready-contract.md`) and tests.
- Loader behavior changes must update loader contract docs and tests.
- Dependency changes must identify dependency group: `core`, `extract`, `preprocess`, `wds`, `tasks`, or `test`.
- Do not mix unrelated scopes in one PR unless it is a deliberate end-to-end feature.
- Default behavior changes must be documented in `CHANGELOG.md`.

## PR Checklist

Before opening a pull request:

- [ ] `python -m compileall skills/daas-compiler` passes (no syntax errors)
- [ ] `cd skills/daas-compiler && pytest tests -q` passes (no failures or errors)
- [ ] If an output schema changed: schema version field bumped and documented in `VERSIONING.md`
- [ ] If a task-ready output changed: `references/training-ready-contract.md` and `references/task-adapters.md` updated
- [ ] If compile code changed: gene order contract verified (expression.var_names matches gene_panel.json)
- [ ] If task-ready or split code changed: split metadata contract verified (split_membership.parquet exists, no physical shard partitioning by default)
- [ ] If loader config changed: loader smoke test verifies runtime split selection from split metadata
- [ ] If extraction output changed: post-save visualization (viz_global_tiles.png, viz_patch_grid.png) verified or explicitly noted
- [ ] If a new dependency was added: follows `references/dependency-policy.md` (correct group, lazy import for preprocess)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Commit messages use allowed types and scopes from the tables above

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
- `requirements-preprocess.txt` covers SOPA-backed preprocessing only

## Output Contract Changes

Any change that alters the shape, fields, or semantics of output files must:

1. Bump the affected schema version(s) — see `VERSIONING.md`
2. Update `references/training-ready-contract.md` if L4 artifacts are affected
3. Update `references/task-adapters.md` if the relevant task adapter's artifact list changes
4. Update or add tests that assert the new schema

Changing output contracts silently is not acceptable, even for "minor" field additions.
