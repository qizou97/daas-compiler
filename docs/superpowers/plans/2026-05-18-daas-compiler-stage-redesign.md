# daas-compiler Stage-Based Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose daas-compiler into composable filtering stages (tissue, nucleus-presence, Xenium-HE overlap) with a natural-language planner, while simplifying `extract_sample.py` to consume a final table key from the last stage.

**Architecture:** Filtering stage scripts write new table keys back into the SpatialData zarr; `extract_sample.py` is simplified to load `--table-key` directly with no inline biological policy. A pure-Python planner (`daas/planning.py`) maps natural-language requests to ordered CLI commands.

**Tech Stack:** Python 3.10+, spatialdata>=0.7, anndata>=0.10, numpy, pandas, geopandas, shapely, sopa (optional, for tissue/HE-nucleus segmentation), pytest

**Spec:** `docs/superpowers/specs/2026-05-18-daas-compiler-stage-redesign.md`

**Run tests throughout with:**
```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
python -m compileall daas scripts
pytest tests -q
```

---

## File Map

| Action | File |
|--------|------|
| Create | `daas/reports.py` |
| Create | `daas/filters/__init__.py` |
| Create | `daas/filters/nucleus_presence.py` |
| Create | `daas/filters/tissue.py` |
| Create | `daas/filters/nucleus_overlap.py` |
| Create | `daas/planning.py` |
| Create | `scripts/inspect_spatialdata.py` |
| Create | `scripts/filter_nucleus_presence.py` |
| Create | `scripts/filter_tissue.py` |
| Create | `scripts/filter_nucleus_overlap.py` |
| Create | `tests/test_reports.py` |
| Create | `tests/test_filter_nucleus_presence.py` |
| Create | `tests/test_planning.py` |
| Create | `references/workflow-planning.md` |
| Create | `references/filtering-recipes.md` |
| Create | `references/table-key-contract.md` |
| Create | `references/sopa-integration.md` |
| Modify | `daas/filtering.py` — remove BiologicalPolicy, resolve_biological_policy, mask_by_nucleus_boundaries, BiologicalResolution; simplify build_filter_report |
| Modify | `daas/cli_args.py` — remove biological policy args from parser |
| Modify | `scripts/extract_sample.py` — remove biological policy phase, load table directly |
| Modify | `scripts/extract_all.py` — remove forwarded biological policy args |
| Modify | `scripts/compile_dataset.py` — add --samples flag |
| Modify | `tests/test_filtering.py` — remove tests for removed functions |
| Modify | `tests/test_filtering_integration.py` — rewrite to not use resolve_biological_policy |
| Modify | `tests/test_compile.py` — add --samples tests |
| Rewrite | `SKILL.md` |

---

## Task 1: daas/reports.py — StageReport dataclass

**Files:**
- Create: `skills/daas-compiler/daas/reports.py`
- Create: `skills/daas-compiler/tests/test_reports.py`

- [ ] **Step 1: Write the failing test**

```python
# skills/daas-compiler/tests/test_reports.py
import json
from pathlib import Path

import pytest

from daas.reports import StageReport, write_stage_report


def _make_report(**kwargs) -> StageReport:
    defaults = dict(
        stage="nucleus_presence",
        zarr_path="/data/A_001.zarr",
        input_table_key="table",
        output_table_key="table_nucleus",
        input_shape_key="cell_circles",
        output_shape_key="cell_circles",
        n_cells_in=100,
        n_cells_out=80,
        drop_counts_by_reason={"missing_nucleus_boundary": 20},
        warnings=[],
    )
    defaults.update(kwargs)
    return StageReport(**defaults)


def test_stage_report_fields():
    r = _make_report()
    assert r.n_cells_in == 100
    assert r.n_cells_out == 80
    assert r.drop_counts_by_reason["missing_nucleus_boundary"] == 20
    assert r.report_path == ""


def test_write_stage_report_creates_file(tmp_path):
    r = _make_report()
    path = write_stage_report(r, tmp_path)
    assert path.exists()
    assert path.name == "nucleus_presence_table.json"
    data = json.loads(path.read_text())
    assert data["stage"] == "nucleus_presence"
    assert data["n_cells_out"] == 80
    assert data["drop_counts_by_reason"]["missing_nucleus_boundary"] == 20


def test_write_stage_report_sets_report_path(tmp_path):
    r = _make_report()
    path = write_stage_report(r, tmp_path)
    assert r.report_path == str(path)


def test_write_stage_report_filename_uses_input_key(tmp_path):
    r = _make_report(stage="tissue_inside", input_table_key="filtered_table")
    path = write_stage_report(r, tmp_path)
    assert path.name == "tissue_inside_filtered_table.json"
```

- [ ] **Step 2: Run test to see it fail**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
pytest tests/test_reports.py -v
```
Expected: `ModuleNotFoundError: No module named 'daas.reports'`

- [ ] **Step 3: Implement daas/reports.py**

```python
# skills/daas-compiler/daas/reports.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StageReport:
    stage: str
    zarr_path: str
    input_table_key: str
    output_table_key: str
    input_shape_key: str
    output_shape_key: str
    n_cells_in: int
    n_cells_out: int
    drop_counts_by_reason: dict = field(default_factory=dict)
    report_path: str = ""
    warnings: list = field(default_factory=list)


def write_stage_report(report: StageReport, report_dir) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    name = f"{report.stage}_{report.input_table_key}.json"
    path = report_dir / name
    data = {
        "stage": report.stage,
        "zarr_path": report.zarr_path,
        "input_table_key": report.input_table_key,
        "output_table_key": report.output_table_key,
        "input_shape_key": report.input_shape_key,
        "output_shape_key": report.output_shape_key,
        "n_cells_in": report.n_cells_in,
        "n_cells_out": report.n_cells_out,
        "drop_counts_by_reason": report.drop_counts_by_reason,
        "warnings": report.warnings,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    report.report_path = str(path)
    return path


__all__ = ["StageReport", "write_stage_report"]
```

- [ ] **Step 4: Run test to see it pass**

```bash
pytest tests/test_reports.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add daas/reports.py tests/test_reports.py
git commit -m "feat: add daas/reports.py — StageReport dataclass and write_stage_report"
```

---

## Task 2: daas/filters/ package + nucleus_presence.py

**Files:**
- Create: `skills/daas-compiler/daas/filters/__init__.py`
- Create: `skills/daas-compiler/daas/filters/nucleus_presence.py`
- Create: `skills/daas-compiler/tests/test_filter_nucleus_presence.py`

- [ ] **Step 1: Write the failing tests**

```python
# skills/daas-compiler/tests/test_filter_nucleus_presence.py
import json

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from daas.filters.nucleus_presence import filter_by_nucleus_presence
from daas.reports import StageReport, write_stage_report


def _adata(cell_ids):
    n = len(cell_ids)
    X = csr_matrix(np.ones((n, 2), dtype=np.float32))
    obs = pd.DataFrame({"cell_id": list(cell_ids)})
    obs.index = [f"{i:07d}" for i in range(n)]
    return anndata.AnnData(X=X, obs=obs,
                           var=pd.DataFrame(index=["g0", "g1"]))


def _nucleus_gdf(cell_ids):
    return pd.DataFrame({"x": range(len(cell_ids))},
                        index=pd.Index(list(cell_ids)))


def test_filter_keeps_cells_with_nucleus():
    adata = _adata(["c0", "c1", "c2", "c3", "c4"])
    nucleus = _nucleus_gdf(["c0", "c2", "c4"])
    keep, drops = filter_by_nucleus_presence(adata, nucleus)
    assert int(keep.sum()) == 3
    assert drops["missing_nucleus_boundary"] == 2


def test_filter_all_have_nucleus():
    adata = _adata(["c0", "c1", "c2"])
    nucleus = _nucleus_gdf(["c0", "c1", "c2"])
    keep, drops = filter_by_nucleus_presence(adata, nucleus)
    assert keep.all()
    assert drops["missing_nucleus_boundary"] == 0


def test_filter_preserves_row_order():
    adata = _adata(["c0", "c1", "c2", "c3"])
    nucleus = _nucleus_gdf(["c1", "c3"])
    keep, _ = filter_by_nucleus_presence(adata, nucleus)
    # c0=False, c1=True, c2=False, c3=True
    np.testing.assert_array_equal(keep, [False, True, False, True])


def test_empty_nucleus_boundaries_raises():
    adata = _adata(["c0", "c1"])
    nucleus = _nucleus_gdf([])
    with pytest.raises(ValueError, match="no cells overlap"):
        filter_by_nucleus_presence(adata, nucleus)


def test_stage_report_fields_after_filter(tmp_path):
    adata = _adata(["c0", "c1", "c2", "c3", "c4"])
    nucleus = _nucleus_gdf(["c0", "c2", "c4"])
    keep, drops = filter_by_nucleus_presence(adata, nucleus)
    r = StageReport(
        stage="nucleus_presence",
        zarr_path="/data/A.zarr",
        input_table_key="table_tissue",
        output_table_key="table_tissue_nucleus",
        input_shape_key="cell_circles",
        output_shape_key="cell_circles",
        n_cells_in=len(adata),
        n_cells_out=int(keep.sum()),
        drop_counts_by_reason=drops,
    )
    write_stage_report(r, tmp_path)
    assert r.n_cells_out == 3
    assert r.drop_counts_by_reason["missing_nucleus_boundary"] == 2
    assert r.report_path != ""
    data = json.loads((tmp_path / "nucleus_presence_table_tissue.json").read_text())
    assert data["n_cells_out"] == 3
```

- [ ] **Step 2: Run test to see it fail**

```bash
pytest tests/test_filter_nucleus_presence.py -v
```
Expected: `ModuleNotFoundError: No module named 'daas.filters'`

- [ ] **Step 3: Implement daas/filters/ package and nucleus_presence.py**

```python
# skills/daas-compiler/daas/filters/__init__.py
```
(empty)

```python
# skills/daas-compiler/daas/filters/nucleus_presence.py
from __future__ import annotations

import numpy as np
import pandas as pd


def filter_by_nucleus_presence(
    adata,
    nucleus_boundaries,
    cell_id_column: str = "cell_id",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose cell_id appears in
    nucleus_boundaries.index.

    Raises ValueError if no cells overlap (likely a cell-id format mismatch).
    """
    cell_ids = adata.obs[cell_id_column].astype(str)
    nucleus_ids = set(pd.Index(nucleus_boundaries.index).astype(str).tolist())
    keep_mask = cell_ids.isin(nucleus_ids).to_numpy()
    if keep_mask.sum() == 0:
        raise ValueError(
            f"nucleus_presence: no cells overlap with nucleus_boundaries. "
            f"table has {len(cell_ids)} cells; nucleus_boundaries has "
            f"{len(nucleus_ids)} entries. Check that cell_id formats match."
        )
    n_dropped = int((~keep_mask).sum())
    return keep_mask, {"missing_nucleus_boundary": n_dropped}


__all__ = ["filter_by_nucleus_presence"]
```

- [ ] **Step 4: Run test to see it pass**

```bash
pytest tests/test_filter_nucleus_presence.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add daas/filters/__init__.py daas/filters/nucleus_presence.py tests/test_filter_nucleus_presence.py
git commit -m "feat: add daas/filters/nucleus_presence — filter_by_nucleus_presence"
```

---

## Task 3: daas/planning.py — NL planner + render_cli

**Files:**
- Create: `skills/daas-compiler/daas/planning.py`
- Create: `skills/daas-compiler/tests/test_planning.py`

- [ ] **Step 1: Write the failing tests**

```python
# skills/daas-compiler/tests/test_planning.py
import pytest

from daas.planning import StagePlan, StageSpec, parse_stage_plan, render_cli


# ── stage detection ───────────────────────────────────────────────────────

def test_tissue_inside_phrase():
    plan = parse_stage_plan("filter out cells outside tissue")
    assert len(plan.stages) == 1
    assert plan.stages[0].name == "tissue_inside"


def test_nucleus_boundary_phrase():
    plan = parse_stage_plan("only keep cells with nucleus boundaries")
    assert len(plan.stages) == 1
    assert plan.stages[0].name == "nucleus_presence"


def test_combined_tissue_and_nucleus():
    plan = parse_stage_plan(
        "filter out cells outside tissue and only keep cells with nucleus boundaries"
    )
    assert [s.name for s in plan.stages] == ["tissue_inside", "nucleus_presence"]


def test_xenium_he_overlap_phrase():
    plan = parse_stage_plan("Xenium nucleus and HE nucleus overlap > 0.5")
    assert len(plan.stages) == 1
    assert plan.stages[0].name == "xenium_he_nucleus_overlap"


def test_no_filter_phrases():
    plan = parse_stage_plan("extract all cells from zarr")
    assert plan.stages == []


# ── extract arg normalization ─────────────────────────────────────────────

def test_optim_ops_level_alias():
    plan = parse_stage_plan("use optim_ops_level for extraction")
    assert plan.extract_args["extract_mode"] == "full_ops_level"


def test_ops_level_alias():
    plan = parse_stage_plan("use ops_level")
    assert plan.extract_args["extract_mode"] == "full_ops_level"


def test_full_ops_level_literal():
    plan = parse_stage_plan("use full_ops_level")
    assert plan.extract_args["extract_mode"] == "full_ops_level"


def test_n_sample_parsed():
    plan = parse_stage_plan("sample 3000 cells per sample")
    assert plan.extract_args["n_sample"] == 3000


def test_n_sample_parsed_alternate_phrasing():
    plan = parse_stage_plan("sampled 3000 cells from each sample")
    assert plan.extract_args["n_sample"] == 3000


def test_mpp_parsed():
    plan = parse_stage_plan("target mpp=0.5")
    assert plan.extract_args["mpp"] == pytest.approx(0.5)


def test_patch_size_parsed():
    plan = parse_stage_plan("patch size 224")
    assert plan.extract_args["patch_size"] == 224


# ── table-key propagation ─────────────────────────────────────────────────

def test_table_key_propagation_two_stages():
    plan = parse_stage_plan(
        "filter outside tissue, keep only cells with nucleus boundaries"
    )
    assert plan.stages[0].input_table_key == "table"
    assert plan.stages[0].output_table_key == "table_tissue"
    assert plan.stages[1].input_table_key == "table_tissue"
    assert plan.stages[1].output_table_key == "table_tissue_nucleus"
    assert plan.final_table_key == "table_tissue_nucleus"
    assert plan.extract_args["table_key"] == "table_tissue_nucleus"


def test_table_key_propagation_three_stages():
    plan = parse_stage_plan(
        "filter tissue, keep nucleus boundaries, Xenium nucleus overlaps HE nucleus"
    )
    keys = [s.output_table_key for s in plan.stages]
    assert keys == ["table_tissue", "table_tissue_nucleus", "table_tissue_nucleus_he"]
    assert plan.final_table_key == "table_tissue_nucleus_he"


def test_no_stages_uses_base_table_key():
    plan = parse_stage_plan("extract all", base_table_key="filtered_table")
    assert plan.final_table_key == "filtered_table"
    assert plan.extract_args["table_key"] == "filtered_table"


def test_custom_base_table_key_propagates_through_stages():
    plan = parse_stage_plan(
        "keep nucleus boundaries",
        base_table_key="filtered_table"
    )
    assert plan.stages[0].input_table_key == "filtered_table"
    assert plan.stages[0].output_table_key == "filtered_table_nucleus"


# ── compile args ──────────────────────────────────────────────────────────

def test_compile_bundle_wds_flag():
    plan = parse_stage_plan("compile and write bundled WebDataset shards")
    assert plan.compile_args.get("bundle_wds") is True


def test_compile_samples_flag():
    plan = parse_stage_plan("Process A_001,A_002,A_004 under /data/spatialdata")
    assert plan.compile_args.get("samples") == ["A_001", "A_002", "A_004"]


# ── render_cli ────────────────────────────────────────────────────────────

def test_render_cli_contains_final_table_key():
    plan = parse_stage_plan(
        "filter outside tissue, keep nucleus boundaries, mpp=0.5, patch size 224"
    )
    cli = render_cli(plan, ["/data/A_001.zarr"], "/data/out")
    assert "--table-key table_tissue_nucleus" in cli


def test_render_cli_contains_extract_mode():
    plan = parse_stage_plan("use optim_ops_level, mpp=0.5")
    cli = render_cli(plan, ["/data/A_001.zarr"], "/data/out")
    assert "--extract-mode full_ops_level" in cli


def test_render_cli_contains_inspect_stage():
    plan = parse_stage_plan("")
    cli = render_cli(plan, ["/data/A_001.zarr"], "/data/out")
    assert "inspect_spatialdata.py" in cli
    assert "/data/A_001.zarr" in cli


def test_render_cli_stage_order():
    plan = parse_stage_plan(
        "filter tissue, keep nucleus boundaries"
    )
    cli = render_cli(plan, ["/data/A_001.zarr"], "/data/out")
    pos_tissue = cli.index("filter_tissue.py")
    pos_nucleus = cli.index("filter_nucleus_presence.py")
    pos_extract = cli.index("extract_sample.py")
    assert pos_tissue < pos_nucleus < pos_extract
```

- [ ] **Step 2: Run test to see it fail**

```bash
pytest tests/test_planning.py -v
```
Expected: `ModuleNotFoundError: No module named 'daas.planning'`

- [ ] **Step 3: Implement daas/planning.py**

```python
# skills/daas-compiler/daas/planning.py
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Stage metadata ────────────────────────────────────────────────────────

_STAGE_ORDER = ["tissue_inside", "nucleus_presence", "xenium_he_nucleus_overlap"]

_STAGE_TRIGGERS: dict[str, list[str]] = {
    "tissue_inside": [
        r"inside tissue", r"out of tissue", r"outside tissue",
        r"tissue filter", r"tissue region", r"filter.*tissue",
    ],
    "nucleus_presence": [
        r"with nucleus boundaries", r"has nucleus",
        r"only cells with nucleus", r"nucleus boundar",
        r"nucleus presence",
    ],
    "xenium_he_nucleus_overlap": [
        r"xenium nucleus.*he nucleus", r"he nucleus",
        r"nucleus overlap", r"overlap\s*>",
    ],
}

_STAGE_SCRIPTS: dict[str, str] = {
    "tissue_inside":              "scripts/filter_tissue.py",
    "nucleus_presence":           "scripts/filter_nucleus_presence.py",
    "xenium_he_nucleus_overlap":  "scripts/filter_nucleus_overlap.py",
}

_STAGE_SUFFIX: dict[str, str] = {
    "tissue_inside":             "tissue",
    "nucleus_presence":          "nucleus",
    "xenium_he_nucleus_overlap": "he",
}

_EXTRACT_MODE_TRIGGERS = [
    "optim_ops_level", "ops_level", "optimized ops level", "full_ops_level",
]


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class StageSpec:
    name: str
    script: str
    input_table_key: str
    output_table_key: str


@dataclass
class StagePlan:
    stages: list[StageSpec] = field(default_factory=list)
    extract_args: dict = field(default_factory=dict)
    compile_args: dict = field(default_factory=dict)
    final_table_key: str = "table"


# ── NL parsing helpers ────────────────────────────────────────────────────

def _detect_stages(text: str) -> list[str]:
    t = text.lower()
    found = []
    for stage in _STAGE_ORDER:
        for trigger in _STAGE_TRIGGERS[stage]:
            if re.search(trigger, t):
                found.append(stage)
                break
    return found


def _detect_extract_args(text: str) -> dict:
    t = text.lower()
    args: dict = {}

    for trigger in _EXTRACT_MODE_TRIGGERS:
        if trigger in t:
            args["extract_mode"] = "full_ops_level"
            break

    m = re.search(r"sampled?\s+(\d+)\s*cell", t)
    if not m:
        m = re.search(r"(\d+)\s*cell[s]?\s+(?:per|from each)\s+sample", t)
    if m:
        args["n_sample"] = int(m.group(1))

    m = re.search(r"mpp[=\s]+([0-9.]+)", t)
    if m:
        args["mpp"] = float(m.group(1))

    m = re.search(r"patch\s+size[=\s]+(\d+)", t)
    if m:
        args["patch_size"] = int(m.group(1))

    return args


def _detect_compile_args(text: str) -> dict:
    t = text.lower()
    args: dict = {}

    if any(p in t for p in ["bundle", "webdataset", "bundled wds", "--bundle-wds"]):
        args["bundle_wds"] = True

    # Sample IDs: look for comma-separated identifiers after "process" or standalone
    m = re.search(r"process\s+([\w]+(?:,[\w]+)+)", text)
    if m:
        args["samples"] = [s.strip() for s in m.group(1).split(",")]

    return args


# ── Public API ────────────────────────────────────────────────────────────

def parse_stage_plan(
    text: str,
    base_table_key: str = "table",
    base_shapes_key: str = "cell_circles",
) -> StagePlan:
    """Map a natural-language request string to a StagePlan.

    Pure Python — no I/O, no zarr access.
    """
    stage_names = _detect_stages(text)
    extract_args = _detect_extract_args(text)
    compile_args = _detect_compile_args(text)

    current_key = base_table_key
    stages: list[StageSpec] = []
    for name in stage_names:
        output_key = f"{current_key}_{_STAGE_SUFFIX[name]}"
        stages.append(StageSpec(
            name=name,
            script=_STAGE_SCRIPTS[name],
            input_table_key=current_key,
            output_table_key=output_key,
        ))
        current_key = output_key

    final_key = current_key
    extract_args["table_key"] = final_key
    extract_args.setdefault("shapes_key", base_shapes_key)

    return StagePlan(
        stages=stages,
        extract_args=extract_args,
        compile_args=compile_args,
        final_table_key=final_key,
    )


def render_cli(
    plan: StagePlan,
    zarr_paths: list[str],
    output_dir: str,
    skill_dir: str = "${SKILL_DIR}",
) -> str:
    """Return a shell-script string of ordered CLI commands to run."""
    lines: list[str] = []

    def _sep(label: str) -> str:
        return f"# ── {label} {'─' * max(0, 60 - len(label))}"

    # Stage 0: inspect
    lines.append(_sep("Stage 0: inspect"))
    for zp in zarr_paths:
        lines.append(
            f"python3 {skill_dir}/scripts/inspect_spatialdata.py \\\n"
            f"    --zarr {zp}"
        )
    lines.append("")

    # Filter stages
    for i, stage in enumerate(plan.stages, 1):
        lines.append(_sep(f"Stage {i}: {stage.name}"))
        for zp in zarr_paths:
            lines.append(
                f"python3 {skill_dir}/{stage.script} \\\n"
                f"    --zarr {zp} \\\n"
                f"    --input-table-key {stage.input_table_key} \\\n"
                f"    --output-table-key {stage.output_table_key}"
            )
        lines.append("")

    # Extract stage
    n_extract = len(plan.stages) + 1
    lines.append(_sep(f"Stage {n_extract}: extract"))
    ea = plan.extract_args
    for zp in zarr_paths:
        sample_id = zp.rstrip("/").split("/")[-1].replace(".zarr", "")
        out = f"{output_dir}/{sample_id}"
        parts = [
            f"python3 {skill_dir}/scripts/extract_sample.py \\",
            f"    --zarr {zp} \\",
            f"    --output {out} \\",
            f"    --table-key {ea.get('table_key', 'table')} \\",
        ]
        if "extract_mode" in ea:
            parts.append(f"    --extract-mode {ea['extract_mode']} \\")
        if "mpp" in ea:
            parts.append(f"    --mpp {ea['mpp']} \\")
        if "patch_size" in ea:
            parts.append(f"    --patch-size {ea['patch_size']} \\")
        if "n_sample" in ea:
            parts.append(f"    --n-sample {ea['n_sample']}")
        # Strip trailing backslash from last line
        parts[-1] = parts[-1].rstrip(" \\")
        lines.append("\n".join(parts))
    lines.append("")

    # Compile stage
    n_compile = n_extract + 1
    lines.append(_sep(f"Stage {n_compile}: compile"))
    compile_parts = [
        f"python3 {skill_dir}/scripts/compile_dataset.py \\",
        f"    --per-sample-dir {output_dir} \\",
        f"    --output {output_dir}/compiled \\",
    ]
    samples = plan.compile_args.get("samples")
    if samples:
        compile_parts.append(f"    --samples {','.join(samples)} \\")
    if plan.compile_args.get("bundle_wds"):
        compile_parts.append("    --bundle-wds")
    compile_parts[-1] = compile_parts[-1].rstrip(" \\")
    lines.append("\n".join(compile_parts))

    return "\n".join(lines)


__all__ = ["StageSpec", "StagePlan", "parse_stage_plan", "render_cli"]
```

- [ ] **Step 4: Run test to see it pass**

```bash
pytest tests/test_planning.py -v
```
Expected: all PASSED (fix any regex edge cases until they do)

- [ ] **Step 5: Commit**

```bash
git add daas/planning.py tests/test_planning.py
git commit -m "feat: add daas/planning.py — NL stage planner with render_cli"
```

---

## Task 4: Refactor daas/filtering.py — remove biological policy

**Files:**
- Modify: `skills/daas-compiler/daas/filtering.py`
- Modify: `skills/daas-compiler/tests/test_filtering.py`
- Modify: `skills/daas-compiler/tests/test_filtering_integration.py`

The following items are **removed** from `daas/filtering.py`:
- `BiologicalPolicy` enum
- `BiologicalResolution` dataclass
- `mask_by_nucleus_boundaries()` function (moved to `daas/filters/nucleus_presence.py`)
- `resolve_biological_policy()` function
- `_pick_canonical_shape_key()` function
- `CANONICAL_SHAPE_PREFERENCE` constant

The `build_filter_report()` signature loses: `biological_policy_requested`, `biological_policy_applied`, `n_after_biological_filter`. Its sequential comment is updated.

- [ ] **Step 1: Update tests/test_filtering.py — remove tests for deleted items**

Open `tests/test_filtering.py` and delete all test functions/classes that import or test:
`BiologicalPolicy`, `BiologicalResolution`, `resolve_biological_policy`, `mask_by_nucleus_boundaries`, `_pick_canonical_shape_key`, `CANONICAL_SHAPE_PREFERENCE`.

Also update the import block at the top of `test_filtering.py`. Replace the full import block with:

```python
# skills/daas-compiler/tests/test_filtering.py  — top of file
"""Unit tests for daas.filtering (patch policy + alignment + reporting)."""
import json
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from daas.filtering import (
    PatchPolicy,
    AlignmentResult,
    PatchMaskResult,
    build_filter_report,
    get_table_cell_ids,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)
```

Also update any `build_filter_report` test calls: remove the two deleted kwargs
(`biological_policy_requested`, `biological_policy_applied`) and the
`n_after_biological_filter` kwarg. The new required kwargs are:

```python
build_filter_report(
    sample_id="A_001",
    zarr_path="/data/A_001.zarr",
    output_dir="/data/out",
    image_key="he_image",
    extract_mode="tile_images",
    source_table_key="table",
    source_shape_key="cell_circles",
    patch_policy_requested="auto",
    patch_policy_applied="strict_no_padding",
    n_cells_source=100,
    n_after_shape_alignment=100,
    n_after_positive_centroid=98,
    n_after_patch_policy=95,
    n_out=95,
    drop_counts_by_reason={"full_oob": 3, "need_pad": 2},
    patch_size=224,
    target_mpp=0.5,
    slide_mpp=0.2125,
    base_size=527,
    image_width_px=38912,
    image_height_px=26624,
    seed=42,
)
```

- [ ] **Step 2: Update tests/test_filtering_integration.py**

The integration test uses `resolve_biological_policy`. Replace the full test content with a version that tests the simplified flow (direct table load → alignment → patch filter):

```python
# skills/daas-compiler/tests/test_filtering_integration.py
"""Integration tests for alignment + patch filtering (no biological policy)."""
import json
from types import SimpleNamespace

import anndata
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from daas.cli_args import (
    build_extract_sample_parser,
    parse_extract_sample_args,
    validate_policy_combination,
)
from daas.filtering import (
    PatchPolicy,
    build_filter_report,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)


def _adata(cell_ids, n_genes: int = 4) -> anndata.AnnData:
    n = len(cell_ids)
    X = csr_matrix(
        np.arange(n * n_genes, dtype=np.float32).reshape(n, n_genes) + 1
    )
    obs = pd.DataFrame({"cell_id": [str(c) for c in cell_ids]})
    obs.index = [f"{i:07d}" for i in range(n)]
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _shapes(cell_ids, xy_um) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in xy_um]},
        index=pd.Index([str(c) for c in cell_ids], name="cell_id"),
    )


def test_alignment_preserved_end_to_end(tmp_path):
    """Direct table load → alignment → patch filter preserves row order."""
    cell_ids = ["c0", "c1", "c2", "c3", "c4", "c5"]
    xy_um = [
        (40, 40),    # fully inside  (keep)
        (60, 60),    # fully inside  (keep)
        (-50, 10),   # full_oob      (drop)
        (-5, 10),    # need_pad      (drop strict)
        (0, 10),     # non-positive  (drop)
        (95, 95),    # need_pad      (drop strict)
    ]
    adata_in = _adata(cell_ids)
    gdf_in = _shapes(cell_ids, xy_um)

    # Direct load — no biological policy
    align = resolve_table_shape_alignment(adata_in, gdf_in)
    assert align.alignment_mode == "exact"
    adata = adata_in[align.adata_row_indices].copy()
    gdf = gdf_in.iloc[align.shape_row_indices].copy()

    SCALE_SHAPE = 1.0
    BASE_SIZE = 20
    BASE_HALF = BASE_SIZE / 2.0
    IMG_W = IMG_H = 100

    cx_um = np.array([c.x for c in gdf.geometry], dtype=np.float64)
    cy_um = np.array([c.y for c in gdf.geometry], dtype=np.float64)
    cx_px = cx_um * SCALE_SHAPE
    cy_px = cy_um * SCALE_SHAPE
    sx0 = cx_px - BASE_HALF
    sy0 = cy_px - BASE_HALF

    patch_policy = resolve_patch_policy(PatchPolicy.AUTO, "tile_images")
    assert patch_policy is PatchPolicy.STRICT_NO_PADDING
    pos_mask = mask_positive_centroid(cx_px, cy_px)
    patch_res = mask_patch_policy(
        sx0, sy0,
        base_size=BASE_SIZE, img_w=IMG_W, img_h=IMG_H,
        policy=patch_policy, extract_mode="tile_images",
    )
    final_mask = pos_mask & patch_res.valid_mask
    np.testing.assert_array_equal(
        final_mask, [True, True, False, False, False, False]
    )

    valid_indices = np.where(final_mask)[0]
    n_out = len(valid_indices)

    # Verify alignment invariants hold
    assert n_out == 2
    survived_ids = [gdf.index[i] for i in valid_indices]
    assert survived_ids == ["c0", "c1"]
    # gene_row_index points to the right cell_id in adata
    for local_i, orig_i in enumerate(valid_indices):
        assert adata.obs.iloc[orig_i]["cell_id"] == survived_ids[local_i]


def test_filter_report_written_with_correct_fields(tmp_path):
    report_dict = build_filter_report(
        sample_id="A_001",
        zarr_path="/data/A_001.zarr",
        output_dir=str(tmp_path),
        image_key="he_image",
        extract_mode="tile_images",
        source_table_key="table_tissue_nucleus",
        source_shape_key="cell_circles",
        patch_policy_requested="auto",
        patch_policy_applied="strict_no_padding",
        n_cells_source=200,
        n_after_shape_alignment=198,
        n_after_positive_centroid=196,
        n_after_patch_policy=190,
        n_out=100,
        drop_counts_by_reason={"full_oob": 6, "need_pad": 4,
                               "requested_subsample": 90},
        patch_size=224,
        target_mpp=0.5,
        slide_mpp=0.2125,
        base_size=527,
        image_width_px=38912,
        image_height_px=26624,
        seed=42,
    )
    assert report_dict["source_table_key"] == "table_tissue_nucleus"
    assert report_dict["n_cells_source"] == 200
    assert report_dict["n_out"] == 100
    # Biological policy fields must NOT be present
    assert "biological_policy_requested" not in report_dict
    assert "biological_policy_applied" not in report_dict
    assert "n_after_biological_filter" not in report_dict

    path = write_filter_report(report_dict, tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["n_cells_source"] == 200


def test_cli_parse_no_biological_policy_args():
    """After refactor, --biological-filter-policy no longer exists."""
    import sys
    parser = build_extract_sample_parser()
    # Should not have biological-filter-policy option
    option_strings = [
        action.option_strings
        for action in parser._actions
    ]
    flat = [s for group in option_strings for s in group]
    assert "--biological-filter-policy" not in flat
    assert "--filtered-table-key" not in flat
    assert "--nucleus-boundaries-key" not in flat


def test_cli_parse_table_key_respected():
    args = parse_extract_sample_args(
        ["--zarr", "/data/A.zarr", "--output", "/data/out",
         "--table-key", "table_tissue_nucleus"]
    )
    assert args.table_key == "table_tissue_nucleus"
```

- [ ] **Step 3: Verify tests still pass before touching implementation**

```bash
pytest tests/test_filtering.py tests/test_filtering_integration.py -v
```
Some tests will fail because we haven't changed the implementation yet. Note which
test functions fail — they should only be ones that reference removed items.

- [ ] **Step 4: Update daas/filtering.py**

Remove the following from `daas/filtering.py`:
1. The `BiologicalPolicy` class and its docstring
2. The `BiologicalResolution` dataclass
3. The `CANONICAL_SHAPE_PREFERENCE` constant
4. The `mask_by_nucleus_boundaries()` function
5. The `_pick_canonical_shape_key()` function
6. The `resolve_biological_policy()` function

Update `build_filter_report()` — replace the full function with this version:

```python
def build_filter_report(
    *,
    sample_id: str,
    zarr_path: str,
    output_dir: str,
    image_key: str,
    extract_mode: str,
    source_table_key: str,
    source_shape_key: str,
    patch_policy_requested: str,
    patch_policy_applied: str,
    n_cells_source: int,
    n_after_shape_alignment: int,
    n_after_positive_centroid: int,
    n_after_patch_policy: int,
    n_out: int,
    drop_counts_by_reason: dict,
    patch_size: int,
    target_mpp: float,
    slide_mpp: float,
    base_size: int,
    image_width_px: int,
    image_height_px: int,
    seed: int,
    warnings: Sequence[str] = (),
) -> dict:
    """Return a JSON-serializable filter report dict.

    Sequential filtering counters:
      n_cells_source
        → n_after_shape_alignment     (table↔shape alignment)
        → n_after_positive_centroid   (cx_px>0 & cy_px>0)
        → n_after_patch_policy        (Layer 2)
        → n_out                        (after optional --n-sample)
    """
    return {
        "sample_id": str(sample_id),
        "zarr_path": str(zarr_path),
        "output_dir": str(output_dir),
        "image_key": str(image_key),
        "extract_mode": str(extract_mode),
        "source_table_key": str(source_table_key),
        "source_shape_key": str(source_shape_key),
        "patch_policy_requested": str(patch_policy_requested),
        "patch_policy_applied": str(patch_policy_applied),
        "n_cells_source": int(n_cells_source),
        "n_after_shape_alignment": int(n_after_shape_alignment),
        "n_after_positive_centroid": int(n_after_positive_centroid),
        "n_after_patch_policy": int(n_after_patch_policy),
        "n_out": int(n_out),
        "drop_counts_by_reason": {k: int(v) for k, v in drop_counts_by_reason.items()},
        "patch_size": int(patch_size),
        "target_mpp": float(target_mpp),
        "slide_mpp": float(slide_mpp),
        "base_size": int(base_size),
        "image_width_px": int(image_width_px),
        "image_height_px": int(image_height_px),
        "seed": int(seed),
        "warnings": list(warnings),
    }
```

Update `__all__` at the bottom of `filtering.py` — remove:
`"BiologicalPolicy"`, `"BiologicalResolution"`, `"mask_by_nucleus_boundaries"`,
`"resolve_biological_policy"`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_filtering.py tests/test_filtering_integration.py -v
```
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add daas/filtering.py tests/test_filtering.py tests/test_filtering_integration.py
git commit -m "refactor: remove biological policy from daas/filtering.py"
```

---

## Task 5: Refactor daas/cli_args.py — remove biological policy args

**Files:**
- Modify: `skills/daas-compiler/daas/cli_args.py`

- [ ] **Step 1: Update cli_args.py**

Remove from `build_extract_sample_parser()`:
- `--biological-filter-policy` argument
- `--cell-id-column` argument (now handled per-stage if needed; keep for now as it's still used for shape alignment)
- `--nucleus-boundaries-key` argument
- `--filtered-table-key` argument

Remove the import of `BiologicalPolicy` from `daas.filtering`.

The updated parser function:

```python
# skills/daas-compiler/daas/cli_args.py
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from daas.filtering import PatchPolicy


DEFAULT_TABLE_KEY = "table"
DEFAULT_SHAPES_KEY = "cell_circles"


def build_extract_sample_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",        required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--sample-id",   default=None)
    p.add_argument("--n-sample",    type=int, default=None)
    p.add_argument("--patch-size",  type=int, default=224)
    p.add_argument("--mpp",         type=float, default=0.5)
    p.add_argument("--shard-size",  type=int, default=500)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--image-key",   default="he_image")
    p.add_argument("--shapes-key",  default=DEFAULT_SHAPES_KEY)
    p.add_argument("--table-key",   default=DEFAULT_TABLE_KEY)
    p.add_argument("--extract-mode", default="tile_images",
                   choices=["tile_images", "full_scale0", "full_ops_level"])
    p.add_argument("--patch-filter-policy",
                   default=PatchPolicy.AUTO.value,
                   choices=[p.value for p in PatchPolicy])
    p.add_argument("--cell-id-column",     default="cell_id")
    p.add_argument("--filter-report-name", default="filter_report.json")
    return p


def parse_extract_sample_args(
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    args = build_extract_sample_parser().parse_args(argv)
    validate_policy_combination(args)
    return args


def validate_policy_combination(args: argparse.Namespace) -> None:
    patch_policy = PatchPolicy(args.patch_filter_policy)
    extract_mode = args.extract_mode

    if patch_policy is PatchPolicy.STRICT_WITH_PADDING:
        raise SystemExit(
            "--patch-filter-policy=strict_with_padding is reserved but not "
            "yet implemented."
        )
    if (
        patch_policy is PatchPolicy.STVISUOME_MINIMAL
        and extract_mode != "tile_images"
    ):
        raise SystemExit(
            f"--patch-filter-policy=stvisuome_minimal requires "
            f"--extract-mode=tile_images (got {extract_mode!r})."
        )


__all__ = [
    "DEFAULT_TABLE_KEY",
    "DEFAULT_SHAPES_KEY",
    "build_extract_sample_parser",
    "parse_extract_sample_args",
    "validate_policy_combination",
]
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests -q
```
Expected: all PASSED. The `test_filtering_integration.py::test_cli_parse_no_biological_policy_args` test should now pass.

- [ ] **Step 3: Commit**

```bash
git add daas/cli_args.py
git commit -m "refactor: remove biological policy args from daas/cli_args.py"
```

---

## Task 6: Refactor scripts/extract_sample.py — load table directly

**Files:**
- Modify: `skills/daas-compiler/scripts/extract_sample.py`

Remove Phase 1b (biological filter resolution) entirely. Directly load the table specified by `--table-key`.

- [ ] **Step 1: Replace Phase 1 + 1b section**

In `extract_sample.py`, find the section between `[1/9] Loading` and `[2/9] Deriving SLIDE_MPP`. Replace everything from after `sdata = sd.read_zarr(zarr_path)` to before `[2/9]` with:

```python
    # ── Phase 1b: Load table + shapes, align ─────────────────────────────────
    print(f"[1b] Loading table={args.table_key!r}  shapes={args.shapes_key!r} …")
    if args.table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.table_key!r}. "
            f"Available tables: {list(sdata.tables.keys())}. "
            "Run filter_tissue.py / filter_nucleus_presence.py first, or "
            "pass --table-key with the correct key."
        )
    adata_source = sdata.tables[args.table_key]
    gdf_full = sdata.shapes[args.shapes_key]
    n_cells_source = int(adata_source.n_obs)

    align = resolve_table_shape_alignment(
        adata_source, gdf_full, cell_id_column=args.cell_id_column,
    )
    if align.alignment_mode != "exact":
        print(f"      [align] non-exact: {align.n_aligned} aligned from "
              f"table={align.n_table_in}, shapes={align.n_shape_in}")
    adata = adata_source[align.adata_row_indices].copy()
    gdf   = gdf_full.iloc[align.shape_row_indices].copy()
    n_after_shape_alignment = int(adata.n_obs)
    print(f"      {n_after_shape_alignment} cells, {adata.n_vars} genes — alignment OK")
```

Also remove the following imports that are no longer needed:
```python
from daas.cli_args import (
    DEFAULT_SHAPES_KEY,
    DEFAULT_TABLE_KEY,
    parse_extract_sample_args,
)
from daas.filtering import (
    BiologicalPolicy,
    PatchPolicy,
    build_filter_report,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_biological_policy,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)
```

Replace with:
```python
from daas.cli_args import parse_extract_sample_args
from daas.filtering import (
    PatchPolicy,
    build_filter_report,
    mask_patch_policy,
    mask_positive_centroid,
    resolve_patch_policy,
    resolve_table_shape_alignment,
    write_filter_report,
)
```

- [ ] **Step 2: Update the build_filter_report call**

Find `report_dict = build_filter_report(` in `extract_sample.py` and update the kwargs to match the new signature (remove `biological_policy_requested`, `biological_policy_applied`, `n_after_biological_filter`; rename `n_cells_source` correctly):

```python
    report_dict = build_filter_report(
        sample_id=sample_id,
        zarr_path=str(zarr_path),
        output_dir=str(output_dir),
        image_key=args.image_key,
        extract_mode=args.extract_mode,
        source_table_key=args.table_key,
        source_shape_key=args.shapes_key,
        patch_policy_requested=requested_patch_policy.value,
        patch_policy_applied=patch_policy_applied.value,
        n_cells_source=n_cells_source,
        n_after_shape_alignment=n_after_shape_alignment,
        n_after_positive_centroid=int(n_after_positive_centroid),
        n_after_patch_policy=int(n_after_patch_policy),
        n_out=int(n_out),
        drop_counts_by_reason=drop_counts_by_reason,
        patch_size=int(PATCH_SIZE),
        target_mpp=float(MPP_TGT),
        slide_mpp=float(SLIDE_MPP),
        base_size=int(BASE_SIZE),
        image_width_px=int(IMG_W),
        image_height_px=int(IMG_H),
        seed=int(args.seed),
        warnings=[],
    )
```

Also update `drop_counts_by_reason` construction — remove the `unaligned_dropped` and `bio.drop_counts` lines. Use:

```python
    drop_counts_by_reason: dict = {}
    unaligned_dropped = n_cells_source - n_after_shape_alignment
    if unaligned_dropped:
        drop_counts_by_reason["unaligned_with_shapes"] = int(unaligned_dropped)
    drop_counts_by_reason["non_positive_centroid"] = int((~pos_mask).sum())
    drop_counts_by_reason.update(
        {k: int(v) for k, v in patch_res.drop_counts.items()}
    )
    drop_counts_by_reason["requested_subsample"] = int(n_valid - n_out)
```

- [ ] **Step 3: Update _validate signature**

The `_validate` function currently references `adata_in` (the resolved table before sampling). Make sure the call passes `adata` (the aligned table) as the `adata_in` argument:

```python
    _validate(cells_df, adata_out, adata, BASE_HALF, n_out, rng, PATCH_SIZE,
              cell_id_column=args.cell_id_column)
```

- [ ] **Step 4: Syntax-check the file**

```bash
python -m compileall skills/daas-compiler/scripts/extract_sample.py
```
Expected: `Compiling ... ok`

- [ ] **Step 5: Run full test suite**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler && pytest tests -q
```
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/extract_sample.py
git commit -m "refactor: extract_sample.py loads table directly — no biological policy"
```

---

## Task 7: Refactor scripts/extract_all.py — remove forwarded args

**Files:**
- Modify: `skills/daas-compiler/scripts/extract_all.py`

- [ ] **Step 1: Remove biological policy args from parse_args() and the extra list**

In `parse_args()`, remove these four `add_argument` calls:
```python
    p.add_argument("--biological-filter-policy", ...)
    p.add_argument("--nucleus-boundaries-key", ...)
    p.add_argument("--filtered-table-key", ...)
```
(Keep `--cell-id-column` since shape alignment still uses it.)

In `main()`, remove from the `extra` list:
```python
        "--biological-filter-policy", args.biological_filter_policy,
        "--nucleus-boundaries-key",   args.nucleus_boundaries_key,
        "--filtered-table-key",       args.filtered_table_key,
```

- [ ] **Step 2: Syntax-check**

```bash
python -m compileall skills/daas-compiler/scripts/extract_all.py
```

- [ ] **Step 3: Run full test suite**

```bash
pytest tests -q
```
Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_all.py
git commit -m "refactor: extract_all.py — remove forwarded biological policy args"
```

---

## Task 8: Add --samples flag to compile_dataset.py

**Files:**
- Modify: `skills/daas-compiler/scripts/compile_dataset.py`
- Modify: `skills/daas-compiler/tests/test_compile.py`

- [ ] **Step 1: Write failing tests**

Add to the bottom of `tests/test_compile.py`:

```python
def test_compile_samples_flag_filters_dirs(synthetic_sample, tmp_path):
    """--samples restricts which sample dirs are compiled."""
    per_sample = tmp_path / "per_sample_samples_test"
    per_sample.mkdir()

    import shutil
    import anndata as _ad
    import pandas as _pd

    for name in ["KEEP_A", "KEEP_B", "SKIP_C"]:
        d = per_sample / name
        shutil.copytree(synthetic_sample["dir"], d)
        mf = _pd.read_parquet(d / "manifest.parquet")
        mf["sample_id"] = name
        mf.to_parquet(d / "manifest.parquet", index=False)
        a = _ad.read_h5ad(d / "expression.h5ad")
        a.obs["sample_id"] = name
        a.write_h5ad(d / "expression.h5ad")

    compiled = tmp_path / "compiled_samples"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled),
         "--samples", "KEEP_A,KEEP_B"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    mf = _pd.read_parquet(compiled / "manifest.parquet")
    assert set(mf["sample_id"].unique()) == {"KEEP_A", "KEEP_B"}
    assert len(mf) == synthetic_sample["n_cells"] * 2


def test_compile_missing_sample_exits_nonzero(synthetic_sample, tmp_path):
    """--samples with a non-existent name exits with code 1."""
    per_sample = tmp_path / "per_sample_missing"
    per_sample.mkdir()

    import shutil
    d = per_sample / "ONLY_ONE"
    shutil.copytree(synthetic_sample["dir"], d)

    compiled = tmp_path / "compiled_missing"
    result = subprocess.run(
        [PYTHON, COMPILE,
         "--per-sample-dir", str(per_sample),
         "--output", str(compiled),
         "--samples", "ONLY_ONE,DOES_NOT_EXIST"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "DOES_NOT_EXIST" in result.stdout + result.stderr
```

- [ ] **Step 2: Run to see them fail**

```bash
pytest tests/test_compile.py::test_compile_samples_flag_filters_dirs \
       tests/test_compile.py::test_compile_missing_sample_exits_nonzero -v
```
Expected: FAILED (no `--samples` flag yet)

- [ ] **Step 3: Add --samples to compile_dataset.py**

In `parse_args()`, add:
```python
    p.add_argument("--samples", default=None,
                   help="Comma-separated list of sample IDs to compile. "
                        "Default: all subdirs with manifest + h5ad.")
```

In `main()`, after building `sample_dirs`, add the filter:
```python
    if args.samples:
        requested = [s.strip() for s in args.samples.split(",")]
        sample_dir_map = {d.name: d for d in sample_dirs}
        missing = [s for s in requested if s not in sample_dir_map]
        if missing:
            print(f"[compile] ERROR: requested samples not found: {missing}")
            sys.exit(1)
        sample_dirs = [sample_dir_map[s] for s in requested]
        print(f"[compile] --samples filter: using {len(sample_dirs)} of "
              f"{len(sample_dir_map)} available samples")
```

Also add `import sys` at the top if not already present.

- [ ] **Step 4: Run tests to see them pass**

```bash
pytest tests/test_compile.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/compile_dataset.py tests/test_compile.py
git commit -m "feat: compile_dataset.py --samples flag to compile a subset of samples"
```

---

## Task 9: scripts/inspect_spatialdata.py

**Files:**
- Create: `skills/daas-compiler/scripts/inspect_spatialdata.py`

No tests (print-only script; correctness verified by running against real data).

- [ ] **Step 1: Create the script**

```python
# skills/daas-compiler/scripts/inspect_spatialdata.py
"""
Print a summary of all tables, shapes, and images in a SpatialData zarr.

Usage:
  python3 scripts/inspect_spatialdata.py --zarr /data/A_001.zarr
"""
import argparse
from pathlib import Path

import spatialdata as sd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = args.zarr
    print(f"[inspect] Loading {zarr_path} …")
    sdata = sd.read_zarr(zarr_path)

    print(f"\n{'='*60}")
    print(f"  SpatialData: {Path(zarr_path).name}")
    print(f"{'='*60}")

    print(f"\n  Tables ({len(sdata.tables)}):")
    for key, tbl in sdata.tables.items():
        print(f"    {key!r:40s}  {tbl.n_obs} cells × {tbl.n_vars} genes")

    print(f"\n  Shapes ({len(sdata.shapes)}):")
    for key, shp in sdata.shapes.items():
        print(f"    {key!r:40s}  {len(shp)} rows")

    print(f"\n  Images ({len(sdata.images)}):")
    for key, img in sdata.images.items():
        try:
            scale0 = img["scale0"]["image"]
            _, h, w = scale0.shape
            print(f"    {key!r:40s}  {h}×{w} px  "
                  f"({len(img)} pyramid levels)")
        except Exception:
            print(f"    {key!r}")

    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

```bash
python -m compileall skills/daas-compiler/scripts/inspect_spatialdata.py
```

- [ ] **Step 3: Commit**

```bash
git add scripts/inspect_spatialdata.py
git commit -m "feat: scripts/inspect_spatialdata.py — print zarr tables/shapes/images"
```

---

## Task 10: scripts/filter_nucleus_presence.py

**Files:**
- Create: `skills/daas-compiler/scripts/filter_nucleus_presence.py`

- [ ] **Step 1: Create the script**

```python
# skills/daas-compiler/scripts/filter_nucleus_presence.py
"""
Filter a SpatialData table to cells that have a nucleus boundary entry.
Writes the filtered table back into the zarr under a new key.

Usage:
  python3 scripts/filter_nucleus_presence.py \
      --zarr /data/A_001.zarr \
      --input-table-key table_tissue \
      --output-table-key table_tissue_nucleus \
      [--nucleus-boundaries-key nucleus_boundaries] \
      [--cell-id-column cell_id] \
      [--report-dir /data/A_001_reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.nucleus_presence import filter_by_nucleus_presence
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",                required=True)
    p.add_argument("--input-table-key",     default="table")
    p.add_argument("--input-shape-key",     default="cell_circles")
    p.add_argument("--output-table-key",    default=None,
                   help="Default: {input_table_key}_nucleus")
    p.add_argument("--nucleus-boundaries-key", default="nucleus_boundaries")
    p.add_argument("--cell-id-column",      default="cell_id")
    p.add_argument("--report-dir",          default=None,
                   help="Default: next to zarr in filter_reports/")
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = Path(args.zarr)
    output_key = args.output_table_key or f"{args.input_table_key}_nucleus"
    report_dir = Path(args.report_dir) if args.report_dir else (
        zarr_path.parent / "filter_reports"
    )

    print(f"[filter_nucleus_presence] {zarr_path.name}")
    print(f"  input_table={args.input_table_key!r}  "
          f"nucleus_boundaries={args.nucleus_boundaries_key!r}  "
          f"output_table={output_key!r}")

    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.input_table_key!r}. "
            f"Available: {list(sdata.tables.keys())}"
        )
    if args.nucleus_boundaries_key not in sdata.shapes:
        raise KeyError(
            f"sdata has no shape {args.nucleus_boundaries_key!r}. "
            f"Available: {list(sdata.shapes.keys())}"
        )

    adata = sdata.tables[args.input_table_key]
    nucleus_gdf = sdata.shapes[args.nucleus_boundaries_key]

    keep_mask, drop_counts = filter_by_nucleus_presence(
        adata, nucleus_gdf, cell_id_column=args.cell_id_column
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  "
          f"(dropped {drop_counts.get('missing_nucleus_boundary', 0)} "
          f"missing nucleus boundary)")

    # Write filtered table back into zarr
    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="nucleus_presence",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.input_shape_key,
        output_shape_key=args.input_shape_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    path = write_stage_report(report, report_dir)
    print(f"  report → {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax-check**

```bash
python -m compileall skills/daas-compiler/scripts/filter_nucleus_presence.py
```

- [ ] **Step 3: Smoke-test against real data (skip if data absent)**

```bash
# Only run if zarr exists
python3 skills/daas-compiler/scripts/filter_nucleus_presence.py \
    --zarr /home/zouqi/datasets/mash/spatialdata/A_001.zarr \
    --input-table-key table \
    --output-table-key table_nucleus_test \
    --report-dir /tmp/daas_reports
```

Expected output:
```
[filter_nucleus_presence] A_001.zarr
  input_table='table'  nucleus_boundaries='nucleus_boundaries'  output_table='table_nucleus_test'
  NNNNN → MMMMM cells  (dropped K missing nucleus boundary)
  wrote 'table_nucleus_test' → ...
  report → /tmp/daas_reports/nucleus_presence_table.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/filter_nucleus_presence.py
git commit -m "feat: scripts/filter_nucleus_presence.py — write filtered table back to zarr"
```

---

## Task 11: daas/filters/tissue.py + scripts/filter_tissue.py

**Files:**
- Create: `skills/daas-compiler/daas/filters/tissue.py`
- Create: `skills/daas-compiler/scripts/filter_tissue.py`

Note: SOPA tissue segmentation API must be verified against the installed version.
Reference: `references/sopa-integration.md` (written in Task 13).

- [ ] **Step 1: Create daas/filters/tissue.py**

```python
# skills/daas-compiler/daas/filters/tissue.py
from __future__ import annotations

import numpy as np


def _ensure_tissue_shapes(sdata, image_key: str, tissue_key: str) -> str:
    """Return tissue_key if present, else run SOPA and return the created key."""
    if tissue_key in sdata.shapes:
        return tissue_key
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for tissue segmentation. Install with: pip install sopa"
        )
    print(f"  [tissue] {tissue_key!r} not found — running sopa tissue segmentation …")
    sopa.segmentation.tissue(sdata, image_key=image_key)
    if tissue_key not in sdata.shapes:
        # SOPA may use a different key; find the first new shapes key
        raise RuntimeError(
            f"sopa.segmentation.tissue ran but {tissue_key!r} was not created. "
            f"Available shapes: {list(sdata.shapes.keys())}. "
            "Pass --tissue-key with the correct key name."
        )
    return tissue_key


def filter_by_tissue(
    adata,
    cell_shapes,
    tissue_shapes,
    cell_id_column: str = "cell_id",
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose centroid is inside
    any tissue polygon.

    cell_shapes: GeoDataFrame with Point or Polygon geometries (cell centroids)
    tissue_shapes: GeoDataFrame with Polygon geometries (tissue regions)
    """
    import geopandas as gpd

    cell_ids = adata.obs[cell_id_column].astype(str)
    # Align cell_shapes to adata row order by cell_id
    cell_shapes_aligned = cell_shapes.loc[
        cell_shapes.index.astype(str).isin(set(cell_ids.tolist()))
    ]
    # Use centroids for point-in-polygon test
    centroids = cell_shapes_aligned.geometry.centroid

    tissue_union = tissue_shapes.geometry.union_all()

    inside = centroids.within(tissue_union)
    # Re-index to adata row order
    inside_series = inside.reindex(cell_ids.values, fill_value=False)
    keep_mask = inside_series.to_numpy(dtype=bool)

    n_dropped = int((~keep_mask).sum())
    return keep_mask, {"outside_tissue": n_dropped}


__all__ = ["filter_by_tissue", "_ensure_tissue_shapes"]
```

- [ ] **Step 2: Create scripts/filter_tissue.py**

```python
# skills/daas-compiler/scripts/filter_tissue.py
"""
Filter a SpatialData table to cells inside tissue regions.
If no tissue polygon exists, runs SOPA tissue segmentation first.

Usage:
  python3 scripts/filter_tissue.py \
      --zarr /data/A_001.zarr \
      --input-table-key table \
      --output-table-key table_tissue \
      [--tissue-key tissue_boundaries] \
      [--input-shape-key cell_circles] \
      [--image-key he_image] \
      [--report-dir /data/reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.tissue import _ensure_tissue_shapes, filter_by_tissue
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",             required=True)
    p.add_argument("--input-table-key",  default="table")
    p.add_argument("--input-shape-key",  default="cell_circles")
    p.add_argument("--output-table-key", default=None,
                   help="Default: {input_table_key}_tissue")
    p.add_argument("--tissue-key",       default="tissue_boundaries",
                   help="Shape key for tissue polygons (created by SOPA if absent)")
    p.add_argument("--image-key",        default="he_image",
                   help="Image key passed to SOPA if tissue segmentation is needed")
    p.add_argument("--cell-id-column",   default="cell_id")
    p.add_argument("--report-dir",       default=None)
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = Path(args.zarr)
    output_key = args.output_table_key or f"{args.input_table_key}_tissue"
    report_dir = Path(args.report_dir) if args.report_dir else (
        zarr_path.parent / "filter_reports"
    )

    print(f"[filter_tissue] {zarr_path.name}")
    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(
            f"sdata has no table {args.input_table_key!r}. "
            f"Available: {list(sdata.tables.keys())}"
        )

    tissue_key = _ensure_tissue_shapes(sdata, args.image_key, args.tissue_key)

    adata = sdata.tables[args.input_table_key]
    cell_shapes = sdata.shapes[args.input_shape_key]
    tissue_shapes = sdata.shapes[tissue_key]

    keep_mask, drop_counts = filter_by_tissue(
        adata, cell_shapes, tissue_shapes,
        cell_id_column=args.cell_id_column,
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  "
          f"(dropped {drop_counts.get('outside_tissue', 0)} outside tissue)")

    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="tissue_inside",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.input_shape_key,
        output_shape_key=args.input_shape_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    path = write_stage_report(report, report_dir)
    print(f"  report → {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax-check both files**

```bash
python -m compileall \
    skills/daas-compiler/daas/filters/tissue.py \
    skills/daas-compiler/scripts/filter_tissue.py
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests -q
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add daas/filters/tissue.py scripts/filter_tissue.py
git commit -m "feat: daas/filters/tissue.py + scripts/filter_tissue.py — tissue-inside filter"
```

---

## Task 12: daas/filters/nucleus_overlap.py + scripts/filter_nucleus_overlap.py

**Files:**
- Create: `skills/daas-compiler/daas/filters/nucleus_overlap.py`
- Create: `skills/daas-compiler/scripts/filter_nucleus_overlap.py`

- [ ] **Step 1: Create daas/filters/nucleus_overlap.py**

```python
# skills/daas-compiler/daas/filters/nucleus_overlap.py
"""Xenium-vs-HE nucleus overlap filter.

Keeps cells whose Xenium nucleus_boundaries polygon overlaps the nearest
HE nucleus polygon with intersection-over-union >= overlap_threshold.

If he_nucleus_boundaries does not exist in sdata.shapes, calls
sopa.segmentation.cellpose on the HE image to create it.
"""
from __future__ import annotations

import numpy as np


def _ensure_he_nucleus_shapes(
    sdata, image_key: str, he_nucleus_key: str
) -> str:
    if he_nucleus_key in sdata.shapes:
        return he_nucleus_key
    try:
        import sopa.segmentation
    except ImportError:
        raise ImportError(
            "sopa is required for HE nucleus segmentation. "
            "Install with: pip install sopa"
        )
    print(f"  [nucleus_overlap] {he_nucleus_key!r} not found — "
          "running sopa Cellpose HE nucleus segmentation …")
    sopa.segmentation.cellpose(sdata, image_key=image_key)
    if he_nucleus_key not in sdata.shapes:
        raise RuntimeError(
            f"sopa.segmentation.cellpose ran but {he_nucleus_key!r} was not created. "
            f"Available shapes: {list(sdata.shapes.keys())}. "
            "Pass --he-nucleus-key with the correct key."
        )
    return he_nucleus_key


def _iou(poly_a, poly_b) -> float:
    """Intersection-over-union for two shapely geometries."""
    inter = poly_a.intersection(poly_b).area
    if inter == 0:
        return 0.0
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def filter_by_nucleus_overlap(
    adata,
    xenium_nucleus_shapes,
    he_nucleus_shapes,
    cell_id_column: str = "cell_id",
    overlap_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Return (keep_mask, drop_counts) keeping cells whose Xenium nucleus has
    IoU >= overlap_threshold with the nearest HE nucleus polygon.
    """
    import pandas as pd
    from shapely.strtree import STRtree

    cell_ids = adata.obs[cell_id_column].astype(str).values
    xen_ids = pd.Index(xenium_nucleus_shapes.index).astype(str)
    xen_map = {str(cid): geom for cid, geom in
               zip(xen_ids, xenium_nucleus_shapes.geometry)}

    he_geoms = list(he_nucleus_shapes.geometry)
    tree = STRtree(he_geoms)

    keep_mask = np.zeros(len(cell_ids), dtype=bool)
    scores = np.zeros(len(cell_ids), dtype=float)

    for i, cid in enumerate(cell_ids):
        if cid not in xen_map:
            continue
        xen_poly = xen_map[cid]
        candidates = tree.query(xen_poly)
        if len(candidates) == 0:
            continue
        best_iou = max(_iou(xen_poly, he_geoms[j]) for j in candidates)
        scores[i] = best_iou
        if best_iou >= overlap_threshold:
            keep_mask[i] = True

    n_dropped_no_nucleus = int(sum(1 for cid in cell_ids if cid not in xen_map))
    n_dropped_low_overlap = int((~keep_mask).sum()) - n_dropped_no_nucleus

    drop_counts = {}
    if n_dropped_no_nucleus:
        drop_counts["no_xenium_nucleus"] = n_dropped_no_nucleus
    if n_dropped_low_overlap > 0:
        drop_counts["low_he_overlap"] = n_dropped_low_overlap

    return keep_mask, drop_counts


__all__ = [
    "filter_by_nucleus_overlap",
    "_ensure_he_nucleus_shapes",
]
```

- [ ] **Step 2: Create scripts/filter_nucleus_overlap.py**

```python
# skills/daas-compiler/scripts/filter_nucleus_overlap.py
"""
Filter cells by Xenium-vs-HE nucleus overlap score.
Runs SOPA Cellpose HE nucleus segmentation if he_nucleus_boundaries absent.

Usage:
  python3 scripts/filter_nucleus_overlap.py \
      --zarr /data/A_001.zarr \
      --input-table-key table_tissue_nucleus \
      --output-table-key table_tissue_nucleus_he \
      [--xenium-nucleus-key nucleus_boundaries] \
      [--he-nucleus-key he_nucleus_boundaries] \
      [--overlap-threshold 0.5] \
      [--image-key he_image] \
      [--report-dir /data/reports]
"""
import argparse
from pathlib import Path

import spatialdata as sd

from daas.filters.nucleus_overlap import (
    _ensure_he_nucleus_shapes,
    filter_by_nucleus_overlap,
)
from daas.reports import StageReport, write_stage_report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",                  required=True)
    p.add_argument("--input-table-key",       default="table")
    p.add_argument("--input-shape-key",       default="cell_circles")
    p.add_argument("--output-table-key",      default=None)
    p.add_argument("--xenium-nucleus-key",    default="nucleus_boundaries")
    p.add_argument("--he-nucleus-key",        default="he_nucleus_boundaries")
    p.add_argument("--overlap-threshold",     type=float, default=0.5)
    p.add_argument("--image-key",             default="he_image")
    p.add_argument("--cell-id-column",        default="cell_id")
    p.add_argument("--report-dir",            default=None)
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = Path(args.zarr)
    output_key = args.output_table_key or f"{args.input_table_key}_he"
    report_dir = Path(args.report_dir) if args.report_dir else (
        zarr_path.parent / "filter_reports"
    )

    print(f"[filter_nucleus_overlap] {zarr_path.name}  "
          f"threshold={args.overlap_threshold}")
    sdata = sd.read_zarr(str(zarr_path))

    if args.input_table_key not in sdata.tables:
        raise KeyError(f"sdata has no table {args.input_table_key!r}.")
    if args.xenium_nucleus_key not in sdata.shapes:
        raise KeyError(
            f"sdata has no shape {args.xenium_nucleus_key!r} "
            "(Xenium nucleus boundaries). Run SOPA nucleus segmentation first."
        )

    he_nucleus_key = _ensure_he_nucleus_shapes(
        sdata, args.image_key, args.he_nucleus_key
    )

    adata = sdata.tables[args.input_table_key]
    xen_nucleus = sdata.shapes[args.xenium_nucleus_key]
    he_nucleus = sdata.shapes[he_nucleus_key]

    keep_mask, drop_counts = filter_by_nucleus_overlap(
        adata, xen_nucleus, he_nucleus,
        cell_id_column=args.cell_id_column,
        overlap_threshold=args.overlap_threshold,
    )

    filtered_adata = adata[keep_mask].copy()
    n_in = int(adata.n_obs)
    n_out = int(filtered_adata.n_obs)

    print(f"  {n_in} → {n_out} cells  drops={drop_counts}")

    sdata[output_key] = filtered_adata
    sdata.write_element(output_key)
    print(f"  wrote {output_key!r} → {zarr_path}")

    report = StageReport(
        stage="xenium_he_nucleus_overlap",
        zarr_path=str(zarr_path),
        input_table_key=args.input_table_key,
        output_table_key=output_key,
        input_shape_key=args.input_shape_key,
        output_shape_key=args.input_shape_key,
        n_cells_in=n_in,
        n_cells_out=n_out,
        drop_counts_by_reason=drop_counts,
    )
    path = write_stage_report(report, report_dir)
    print(f"  report → {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Syntax-check**

```bash
python -m compileall \
    skills/daas-compiler/daas/filters/nucleus_overlap.py \
    skills/daas-compiler/scripts/filter_nucleus_overlap.py
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests -q
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add daas/filters/nucleus_overlap.py scripts/filter_nucleus_overlap.py
git commit -m "feat: nucleus_overlap filter — Xenium-vs-HE IoU with SOPA Cellpose fallback"
```

---

## Task 13: Reference docs

**Files:**
- Create: `skills/daas-compiler/references/workflow-planning.md`
- Create: `skills/daas-compiler/references/filtering-recipes.md`
- Create: `skills/daas-compiler/references/table-key-contract.md`
- Create: `skills/daas-compiler/references/sopa-integration.md`

- [ ] **Step 1: Create references/workflow-planning.md**

```markdown
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
```

- [ ] **Step 2: Create references/filtering-recipes.md**

```markdown
# Filtering Recipes Reference

Three filtering recipes are available as stage scripts.

---

## Recipe 1: tissue_inside

**Script:** `scripts/filter_tissue.py`  
**Trigger phrases:** "inside tissue", "out of tissue", "outside tissue", "tissue filter"

**What it does:**
1. Checks `sdata.shapes` for tissue polygon (default key: `tissue_boundaries`).
2. If absent: calls `sopa.segmentation.tissue(sdata, image_key=...)`.
3. Keeps cells whose centroid lies inside any tissue polygon.
4. Writes filtered table to zarr + `StageReport` JSON.

**CLI:**
```bash
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr /data/A_001.zarr \
    --input-table-key table \
    --output-table-key table_tissue \
    [--tissue-key tissue_boundaries] \
    [--image-key he_image]
```

**Output table key:** `{input_table_key}_tissue` (e.g. `table_tissue`)  
**Drop reason key:** `outside_tissue`

---

## Recipe 2: nucleus_presence

**Script:** `scripts/filter_nucleus_presence.py`  
**Trigger phrases:** "with nucleus boundaries", "has nucleus", "only cells with nucleus", "nucleus boundary"

**What it does:**
1. Loads `sdata.shapes[nucleus_boundaries_key]`.
2. Keeps rows where `obs["cell_id"]` appears in `nucleus_boundaries.index`.
3. Writes filtered table to zarr + `StageReport` JSON.
4. No SOPA call — fails if `nucleus_boundaries` does not exist.

**CLI:**
```bash
python3 ${SKILL_DIR}/scripts/filter_nucleus_presence.py \
    --zarr /data/A_001.zarr \
    --input-table-key table_tissue \
    --output-table-key table_tissue_nucleus \
    [--nucleus-boundaries-key nucleus_boundaries]
```

**Output table key:** `{input_table_key}_nucleus`  
**Drop reason key:** `missing_nucleus_boundary`

---

## Recipe 3: xenium_he_nucleus_overlap

**Script:** `scripts/filter_nucleus_overlap.py`  
**Trigger phrases:** "Xenium nucleus overlaps HE nucleus", "HE nucleus", "nucleus overlap", "overlap >"

**What it does:**
1. Checks `sdata.shapes[he_nucleus_boundaries]`.
2. If absent: calls `sopa.segmentation.cellpose(sdata, image_key=...)`.
3. Computes per-cell IoU between Xenium `nucleus_boundaries` and nearest HE nucleus.
4. Keeps cells where `IoU >= --overlap-threshold` (default 0.5).
5. Writes filtered table to zarr + `StageReport` JSON.

**CLI:**
```bash
python3 ${SKILL_DIR}/scripts/filter_nucleus_overlap.py \
    --zarr /data/A_001.zarr \
    --input-table-key table_tissue_nucleus \
    --output-table-key table_tissue_nucleus_he \
    [--overlap-threshold 0.5] \
    [--he-nucleus-key he_nucleus_boundaries]
```

**Output table key:** `{input_table_key}_he`  
**Drop reason keys:** `no_xenium_nucleus`, `low_he_overlap`
```

- [ ] **Step 3: Create references/table-key-contract.md**

```markdown
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
```

- [ ] **Step 4: Create references/sopa-integration.md**

```markdown
# SOPA Integration Reference

SOPA (Spatial Omics Pipeline Architecture) is an optional dependency used by
two stage scripts:

- `filter_tissue.py` — `sopa.segmentation.tissue()`
- `filter_nucleus_overlap.py` — `sopa.segmentation.cellpose()`

## Installation

```bash
pip install sopa
# or, if using extras:
pip install "sopa[cellpose]"
```

## Tissue segmentation (filter_tissue.py)

Called when `--tissue-key` (default: `tissue_boundaries`) is absent from
`sdata.shapes`:

```python
import sopa.segmentation
sopa.segmentation.tissue(sdata, image_key="he_image")
```

Expected result: a new polygon GeoDataFrame in `sdata.shapes` named
`tissue_boundaries` (or similar — check `sopa` docs for exact key name).

**Verify the actual API against your installed sopa version:**
```python
import sopa.segmentation
help(sopa.segmentation.tissue)
```

## HE nucleus segmentation (filter_nucleus_overlap.py)

Called when `--he-nucleus-key` (default: `he_nucleus_boundaries`) is absent:

```python
import sopa.segmentation
sopa.segmentation.cellpose(sdata, image_key="he_image")
```

Expected result: a new polygon GeoDataFrame in `sdata.shapes` named
`he_nucleus_boundaries` (or similar — check `sopa` docs).

**If the created key differs from the default, pass `--he-nucleus-key`
explicitly.**

## Error if sopa missing

Both scripts raise `ImportError` with installation instructions when
`import sopa.segmentation` fails. The scripts do NOT require sopa if the
relevant shape keys already exist in the zarr.
```

- [ ] **Step 5: Commit**

```bash
git add references/workflow-planning.md references/filtering-recipes.md \
        references/table-key-contract.md references/sopa-integration.md
git commit -m "docs: add four reference docs for stage-based workflow"
```

---

## Task 14: Rewrite SKILL.md

**Files:**
- Modify: `skills/daas-compiler/SKILL.md`

The SKILL.md rewrite is the largest documentation task. Replace the entire file. Key sections to preserve from the current file (copy verbatim or adapt):

- Version Compatibility table
- Installation block
- Phase 1: Single-Sample Extraction (CLI section)
- manifest.parquet columns table
- Patch-validity filtering (Layer 2 only — biological policy section removed)
- filter_report.json schema (updated: remove biological_policy_* fields)
- Pipeline Internals (MPP derivation, OOB filtering, spatial sort, TileSpec, shard writing)
- Extraction Strategies table
- Phase 1b: Parallel Multi-Sample Extraction
- Phase 2: Compile (add --samples flag documentation)
- Phase 3: CellPatchDataset, BundledCellPatchDataset, LRU mmap cache
- Visualization Validation
- Common Errors table

New sections to add (replace the old "Filtering policy" section):

- [ ] **Step 1: Add "Stage-Based Workflow" section at the top after Installation**

```markdown
## Stage-Based Workflow

Extract cell-centered patches by composing stages. Each stage writes a new
table key back into the zarr; the final key flows into `extract_sample.py`.

```
sdata.zarr
    │
    │  [Stage 0: inspect]  always first — verify keys
    │
    │  [Stage 1–N: filter]  optional, any combination, in order:
    │    filter_tissue.py          → writes table_tissue
    │    filter_nucleus_presence.py → writes table_tissue_nucleus
    │    filter_nucleus_overlap.py  → writes table_tissue_nucleus_he
    │
    │  [Stage N+1: extract]  per-sample, parallelizable
    │    extract_sample.py --table-key <FINAL_KEY>
    │
    │  [Stage N+2: compile]  once all samples done
    │    compile_dataset.py [--samples A,B,C] [--bundle-wds]
    ▼
compiled/  or  per-sample bundled WebDataset shards
```

### filtered_table is optional

`filtered_table` is not required. It is one possible `--input-table-key` value.
If your zarr has a pre-existing `filtered_table`, pass it:
`--input-table-key filtered_table` to the first stage script you run.

### Building a stage plan from natural language

Use `daas.planning.parse_stage_plan()` and `render_cli()`:

```python
from daas.planning import parse_stage_plan, render_cli

plan = parse_stage_plan(
    "filter outside tissue, keep nucleus boundaries, "
    "mpp=0.5, patch size 224, use optim_ops_level, sample 3000 cells per sample"
)
print(render_cli(plan, ["/data/A_001.zarr"], "/data/out"))
```

Trigger phrase → stage mapping:

| Phrase | Stage |
|---|---|
| "inside tissue", "out of tissue", "outside tissue" | tissue_inside |
| "with nucleus boundaries", "only cells with nucleus" | nucleus_presence |
| "Xenium nucleus overlaps HE nucleus", "overlap >" | xenium_he_nucleus_overlap |
| "optim_ops_level", "ops_level" | extract_mode=full_ops_level |
| "sample N cells per sample" | --n-sample N |

### Stage report contract

Every stage script writes a JSON to `--report-dir`:

```json
{
  "stage": "nucleus_presence",
  "input_table_key": "table_tissue",
  "output_table_key": "table_tissue_nucleus",
  "n_cells_in": 184523,
  "n_cells_out": 173210,
  "drop_counts_by_reason": {"missing_nucleus_boundary": 11313},
  "warnings": []
}
```

### Worked example

Request:
> "Process A_001,A_002,A_004 under /home/zouqi/datasets/mash/spatialdata into
> cell-centered HE patches. Filter out cells outside tissue and only keep cells
> with nucleus boundaries. Target mpp=0.5, patch size=224, use optim_ops_level,
> output to /home/zouqi/datasets/mash/stvisuome, sample 3000 cells per sample,
> compile, and write bundled WebDataset shards."

Stage plan resolves to:
- stages: `tissue_inside` → `nucleus_presence`
- final table key: `table_tissue_nucleus`
- extract: `--extract-mode full_ops_level --mpp 0.5 --patch-size 224 --n-sample 3000`
- compile: `--samples A_001,A_002,A_004 --bundle-wds`

Generated CLI (run in order):
```bash
# Stage 0: inspect
python3 ${SKILL_DIR}/scripts/inspect_spatialdata.py --zarr .../A_001.zarr
# (repeat for A_002, A_004)

# Stage 1: tissue_inside
python3 ${SKILL_DIR}/scripts/filter_tissue.py \
    --zarr .../A_001.zarr \
    --input-table-key table --output-table-key table_tissue
# (repeat for A_002, A_004)

# Stage 2: nucleus_presence
python3 ${SKILL_DIR}/scripts/filter_nucleus_presence.py \
    --zarr .../A_001.zarr \
    --input-table-key table_tissue --output-table-key table_tissue_nucleus
# (repeat for A_002, A_004)

# Stage 3: extract
python3 ${SKILL_DIR}/scripts/extract_sample.py \
    --zarr .../A_001.zarr \
    --output .../stvisuome/A_001 \
    --table-key table_tissue_nucleus \
    --extract-mode full_ops_level --mpp 0.5 --patch-size 224 --n-sample 3000
# (repeat for A_002, A_004)

# Stage 4: compile
python3 ${SKILL_DIR}/scripts/compile_dataset.py \
    --per-sample-dir .../stvisuome \
    --output .../stvisuome/compiled \
    --samples A_001,A_002,A_004 \
    --bundle-wds
```
```

- [ ] **Step 2: Add the new Scripts Reference table**

In the Scripts Reference section at the bottom, replace the old table with:

```markdown
| Script | Purpose |
|--------|---------|
| `${SKILL_DIR}/scripts/inspect_spatialdata.py` | Print zarr tables/shapes/images |
| `${SKILL_DIR}/scripts/filter_tissue.py` | Tissue-inside filter (SOPA if needed) |
| `${SKILL_DIR}/scripts/filter_nucleus_presence.py` | Keep cells with nucleus boundary |
| `${SKILL_DIR}/scripts/filter_nucleus_overlap.py` | Xenium-vs-HE nucleus IoU filter |
| `${SKILL_DIR}/scripts/extract_sample.py` | Single-sample HE patch extraction |
| `${SKILL_DIR}/scripts/extract_all.py` | Parallel multi-sample extraction |
| `${SKILL_DIR}/scripts/compile_dataset.py` | Compile per-sample dirs; --samples flag |
| `${SKILL_DIR}/scripts/viz_sample.py` | Re-render viz outputs |
| `${SKILL_DIR}/daas/dataset.py` | LRUMmapCache + CellPatchDataset |
| `${SKILL_DIR}/daas/planning.py` | NL → StagePlan → render_cli |
| `${SKILL_DIR}/daas/reports.py` | StageReport + write_stage_report |
| `${SKILL_DIR}/daas/filters/nucleus_presence.py` | nucleus_presence filter logic |
| `${SKILL_DIR}/daas/filters/tissue.py` | tissue_inside filter logic |
| `${SKILL_DIR}/daas/filters/nucleus_overlap.py` | xenium_he_nucleus_overlap logic |
```

- [ ] **Step 3: Run final tests to confirm nothing broke**

```bash
pytest tests -q
```
Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add SKILL.md
git commit -m "docs: rewrite SKILL.md for stage-based workflow"
```

---

## Task 15: Final verification

- [ ] **Step 1: compileall**

```bash
cd /home1/zouqi/codes/daas-compiler/skills/daas-compiler
python -m compileall daas scripts -q
```
Expected: no errors

- [ ] **Step 2: Full test suite**

```bash
pytest tests -q
```
Expected: all PASSED, 0 errors

- [ ] **Step 3: Verify new modules importable**

```bash
python -c "
from daas.reports import StageReport, write_stage_report
from daas.filters.nucleus_presence import filter_by_nucleus_presence
from daas.filters.tissue import filter_by_tissue
from daas.filters.nucleus_overlap import filter_by_nucleus_overlap
from daas.planning import parse_stage_plan, render_cli, StagePlan
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 4: Smoke-test the planner with the worked example**

```bash
python -c "
from daas.planning import parse_stage_plan, render_cli
plan = parse_stage_plan(
    'Process A_001,A_002,A_004 under /data/spatialdata. '
    'Filter out cells outside tissue and only keep cells with nucleus boundaries. '
    'Target mpp=0.5, patch size=224, use optim_ops_level, '
    'sample 3000 cells per sample, compile, and write bundled WebDataset shards.'
)
print('stages:', [s.name for s in plan.stages])
print('final_table_key:', plan.final_table_key)
print('extract_mode:', plan.extract_args.get('extract_mode'))
print('n_sample:', plan.extract_args.get('n_sample'))
print('bundle_wds:', plan.compile_args.get('bundle_wds'))
zarr_paths = ['/data/A_001.zarr', '/data/A_002.zarr', '/data/A_004.zarr']
print(render_cli(plan, zarr_paths, '/data/stvisuome'))
"
```
Expected:
```
stages: ['tissue_inside', 'nucleus_presence']
final_table_key: table_tissue_nucleus
extract_mode: full_ops_level
n_sample: 3000
bundle_wds: True
# ── Stage 0: inspect ...
```

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "chore(v0.6.0): stage-based daas-compiler redesign complete"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by task(s) |
|---|---|
| filter_tissue.py script | Task 11 |
| filter_nucleus_presence.py script | Task 10 |
| filter_nucleus_overlap.py script | Task 12 |
| inspect_spatialdata.py script | Task 9 |
| daas/planning.py | Task 3 |
| daas/filters/tissue.py | Task 11 |
| daas/filters/nucleus_presence.py | Task 2 |
| daas/filters/nucleus_overlap.py | Task 12 |
| daas/reports.py | Task 1 |
| References (4 files) | Task 13 |
| extract_sample.py — remove biological policy | Task 6 |
| daas/filtering.py — remove biological policy | Task 4 |
| daas/cli_args.py — remove biological policy args | Task 5 |
| extract_all.py — remove forwarded args | Task 7 |
| compile_dataset.py --samples | Task 8 |
| SKILL.md rewrite | Task 14 |
| tests/test_planning.py | Task 3 |
| tests/test_filter_nucleus_presence.py | Task 2 |
| tests/test_compile.py --samples tests | Task 8 |
| Stage report contract (StageReport) | Task 1 |
| Table-key propagation | Task 3 (planner) |
| NL normalization: optim_ops_level alias | Task 3 |
| NL normalization: n_sample | Task 3 |
| Write filtered tables back to zarr | Tasks 10, 11, 12 |
| SOPA integration | Tasks 11, 12, 13 |

**Type/name consistency verified:**
- `StageReport` defined in Task 1; used in Tasks 10, 11, 12
- `write_stage_report(report, report_dir)` consistent throughout
- `filter_by_nucleus_presence(adata, nucleus_gdf, cell_id_column)` defined Task 2; called Task 10
- `filter_by_tissue(adata, cell_shapes, tissue_shapes, cell_id_column)` defined Task 11; called Task 11
- `filter_by_nucleus_overlap(adata, xen, he, cell_id_column, overlap_threshold)` defined Task 12; called Task 12
- `parse_stage_plan(text, base_table_key, base_shapes_key)` defined Task 3; tested Task 3
- `render_cli(plan, zarr_paths, output_dir, skill_dir)` defined Task 3; tested Task 3
- `build_filter_report(...)` updated Task 4; call site updated Task 6 — new signature used consistently
