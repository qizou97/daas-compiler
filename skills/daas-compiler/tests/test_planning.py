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
