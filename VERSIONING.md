# Versioning

## Package and Skill Versions

daas-compiler uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

The package version (in `pyproject.toml`) and the skill version (referenced in `SKILL.md`
and marketplace metadata) are kept in sync. When you bump one, bump the other.

**Current baseline:** `0.6.1`

## Pre-1.0 Policy

While the version is below `1.0.0`:
- **Minor version bumps** (`0.x.0`) may include breaking changes to public APIs, output schemas, or artifact contracts. Document breaking changes prominently in `CHANGELOG.md`.
- **Patch version bumps** (`0.0.x`) are backwards-compatible bug fixes or documentation improvements.
- There is no deprecation period requirement before removing or changing APIs.

## Version Types

| Version type | Where it lives | What it governs |
|---|---|---|
| Code version | `skills/daas-compiler/pyproject.toml` → `version` | Python package |
| Skill version | `SKILL.md` frontmatter, marketplace metadata | Agent behavior contract |
| Artifact schema versions | JSON/YAML output field values | Shape of runtime output files |
| Task dataset contract versions | `task_config.yaml`, `dataset_card.json` | L4 training-ready artifact layout |

## Artifact Schema Versions

Each output file format has its own schema version field embedded in the file.
These are independent of the package version and must be incremented when the
format changes in a backwards-incompatible way.

| Schema version key | Output file | Bump when |
|---|---|---|
| `manifest_schema_version` | `manifest.parquet` (as metadata) | Column added, renamed, or removed |
| `filter_report_schema_version` | `filter_report.json` | Field added, renamed, or removed |
| `compile_report_schema_version` | `compile_report.json` | Field added, renamed, or removed |
| `wds_metadata_schema_version` | Per-cell `.json` inside shards | Field added, renamed, or removed |
| `gene_panel_schema_version` | `gene_panel.json` | Structure changes (currently a flat list) |
| `task_dataset_schema_version` | `dataset_card.json` | Field added, renamed, or removed |
| `split_schema_version` | `split_report.json` | Field added, renamed, or removed |
| `loader_config_schema_version` | `loader_config.yaml` | Field added, renamed, or removed |

Schema version fields use integer counters starting at `1`.

## When to Bump the Package Version

| Change | Version to bump |
|---|---|
| Bug fix, no output change | PATCH |
| New optional CLI flag, new optional output field | PATCH (also bump the relevant schema version) |
| New stage, new script, new task adapter | MINOR |
| Output schema change (field renamed or removed) | MINOR (also bump the relevant schema version) |
| API removal or breaking behavior change | MINOR (pre-1.0) or MAJOR (post-1.0) |
| Training-ready contract change (L4 layout) | MINOR — this is a contract change, not a docs change |

**Changing the definition or layout of training-ready task datasets is a contract
change, not just a docs change.** It must be accompanied by:
- A schema version bump for the affected output files
- A `CHANGELOG.md` entry under `[Unreleased]`
- Updated documentation in `references/training-ready-contract.md` and the relevant
  task adapter in `references/task-adapters.md`
- Updated tests

## How to Bump the Version

1. Update `version` in `skills/daas-compiler/pyproject.toml`
2. Update `CHANGELOG.md`: move `[Unreleased]` items to a new version section, add the date
3. Commit: `chore(release): bump to vX.Y.Z`
4. Tag: `git tag vX.Y.Z`

See `RELEASE.md` for the full release checklist.
