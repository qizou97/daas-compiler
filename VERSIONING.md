# Versioning

## Package and Skill Versions

daas-compiler uses [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`).

The package version (in `pyproject.toml`) and the skill version (referenced in `SKILL.md`
and marketplace metadata) are kept in sync. When you bump one, bump the other.

**Current baseline:** `0.7.3`

## Pre-1.0 Policy

While the version is below `1.0.0`:
- **Minor version bumps** (`0.x.0`) may include breaking changes to public APIs, output schemas, or artifact contracts. Document breaking changes prominently in `CHANGELOG.md`.
- **Patch version bumps** (`0.0.x`) are backwards-compatible bug fixes or documentation improvements.
- There is no deprecation period requirement before removing or changing APIs.

## Version Types

| Version type | Where it lives | What it governs |
|---|---|---|
| Package version | `skills/daas-compiler/pyproject.toml` → `version` | Python package |
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
| `split_schema_version` | `splits/split_membership.parquet`, `splits/*.json` | Field added, renamed; split membership format changes |
| `loader_config_schema_version` | `loader_config.yaml` | Field added, renamed, or removed; shard path schema changes |

Schema version fields use integer counters starting at `1`.

## When to Bump the Package Version

### MAJOR

Breaking changes that require consumers to update their pipelines or loaders:

- Breaking output artifact schemas (field renamed or removed)
- Breaking manifest/expression/global_idx alignment
- Breaking WebDataset metadata layout
- Changing gene vector interpretation (gene order, normalization contract)
- Changing split semantics in an incompatible way (e.g., split_membership schema)
- Changing loader_config contract incompatibly

### MINOR

Additive changes that may require new task adapter runs or new optional dependencies:

- New task adapter
- New filtering recipe or filter stage
- New optional dependency group
- New optional report fields (while preserving existing fields)
- New split policy or split export mode
- New validation or visualization stage
- New CLI flag with additive output

### PATCH

Safe changes with no output contract impact:

- Bug fixes with identical output semantics
- Documentation fixes
- Validation improvements that do not change successful output semantics
- Performance improvements with identical output contract
- More robust key detection or error messages

## Split and Loader Contract Rules

- **Split metadata format changes** require bumping `split_schema_version` in affected output files.
- **Loader config changes** require bumping `loader_config_schema_version`.
- **Changing the default split behavior** (e.g., switching from metadata-based to physical
  shard partitioning, or vice versa) is a MINOR version bump with required `CHANGELOG.md` entry.
- The current default: splits are metadata in `splits/split_membership.parquet` — physical
  `train/`, `val/`, `test/` shard directories are an optional export mode only.

## Contract Change Rules

**Changing the definition or layout of training-ready task datasets is a contract
change, not just a docs change.** It must be accompanied by:
- A schema version bump for the affected output files
- A `CHANGELOG.md` entry under `[Unreleased]`
- Updated documentation in `references/training-ready-contract.md` and the relevant
  task adapter in `references/task-adapters.md`
- Updated tests
- Default behavior changes documented in `CHANGELOG.md`

## How to Bump the Version

1. Update `version` in `skills/daas-compiler/pyproject.toml`
2. Update `CHANGELOG.md`: move `[Unreleased]` items to a new version section, add the date
3. Commit: `release: bump to vX.Y.Z`
4. Tag: `git tag vX.Y.Z`

See `RELEASE.md` for the full release checklist.
