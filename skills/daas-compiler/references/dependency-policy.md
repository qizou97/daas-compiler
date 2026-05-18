# Dependency Policy

## Dependency Groups

| Group | pyproject.toml extra | Install command | Purpose |
|---|---|---|---|
| core | (base, always installed) | `pip install -e .` | anndata, numpy, pandas, scipy, Pillow, pyarrow — required for all loading and metadata |
| extract | `[extract]` | `pip install -e .[extract]` | spatialdata, wsidata, lazyslide, matplotlib — required for extraction scripts |
| preprocess | `[preprocess]` | `pip install -e .[preprocess]` | sopa, geopandas, shapely, scikit-image — required for SOPA-backed filtering stages |
| wds | `[wds]` | `pip install -e .[wds]` | webdataset — required for pure webdataset streaming pipeline |
| tasks | `[tasks]` | `pip install -e .[tasks]` | pyyaml — required for writing task_config.yaml and loader_config.yaml |
| test | `[test]` | `pip install -e .[test]` | pytest, geopandas, shapely — required for running the test suite |

## SOPA Rules

SOPA belongs to the `preprocess` group. The following rules are non-negotiable:

**1. SOPA must be lazy-imported.** Any code that uses SOPA must import it inside the
function or block that uses it, not at module level:

```python
# Correct — lazy import inside the function
def filter_with_sopa(sdata, ...):
    import sopa
    ...

# Wrong — breaks installs without the preprocess group
import sopa
def filter_with_sopa(sdata, ...):
    ...
```

**2. Plain extraction must work without SOPA.** `extract_sample.py`, `compile_dataset.py`,
and HE2ST task adapter packaging must complete successfully with only
`pip install -e .` and `pip install -e .[extract]`.

**3. The test suite must not require SOPA.** Tests that exercise SOPA-backed filtering
must guard the import:

```python
pytest.importorskip("sopa")
```

For SOPA integration details — API surface, filter stage implementation, tissue detection
— see `references/sopa-integration.md`.

## Adding New Dependencies

Before adding a new dependency, answer these questions in order:

1. Is it required for core loading or metadata functionality that runs unconditionally?
   → Add to `core`. Requires explicit review — affects all users.

2. Is it required only for extraction (spatialdata, wsidata, lazyslide)?
   → Add to `extract`.

3. Is it required only for SOPA-backed preprocessing?
   → Add to `preprocess`. Must be lazy-imported wherever used.

4. Is it required for task adapter output (task_config.yaml, loader_config.yaml)?
   → Add to `tasks`.

5. Is it a heavy optional dependency used only in one task adapter?
   → Add a new named extra for that task, or add to `tasks` if it's universal across task adapters.

6. Is it only needed for tests?
   → Add to `test`.

**New heavy dependencies must be optional** unless they are required for core
loading or metadata functionality that runs unconditionally at import time.
Do not add SOPA to `requirements.txt` or to `core`. `requirements.txt` covers
extraction/viz runtime only.
