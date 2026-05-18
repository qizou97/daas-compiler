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

    # Sample IDs: look for comma-separated identifiers after "process" or "Process"
    # Match both "Process A_001,A_002" and similar patterns
    m = re.search(r"[Pp]rocess\s+([\w]+(?:,[\w]+)+)", text)
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
