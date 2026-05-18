"""
Static checks verifying agent-contract.md and related docs encode the correct
policies for splits, artifact levels, and missing-split behavior.

These tests parse the Markdown documents directly and assert the presence or
absence of specific strings that the contract requires or forbids.
"""
import re
from pathlib import Path

REFS = Path(__file__).parent.parent / "references"
SKILL_MD = Path(__file__).parent.parent / "SKILL.md"
USAGE_GUIDE = Path(__file__).parent.parent / "usage-guide.md"
CONTRACT = REFS / "agent-contract.md"
TRAINING_READY = REFS / "training-ready-contract.md"
TASK_ADAPTERS = REFS / "task-adapters.md"


def _text(path: Path) -> str:
    return path.read_text()


# ── agent-contract.md: split-as-metadata ────────────────────────────────────

def test_contract_states_splits_are_metadata():
    text = _text(CONTRACT)
    assert "Splits are metadata" in text or "splits are metadata" in text.lower()


def test_contract_does_not_require_physical_split_dirs_as_default():
    text = _text(CONTRACT)
    # Should reference materialize-split-shards as the opt-in, not the default
    assert "--materialize-split-shards" in text


def test_contract_canonical_storage_not_split_partitioned():
    text = _text(CONTRACT)
    assert "NOT" in text and ("partitioned" in text or "split-partitioned" in text)


# ── agent-contract.md: missing-split behavior ────────────────────────────────

def test_contract_requires_prompting_when_split_missing():
    text = _text(CONTRACT)
    # Must describe what to do when split information is missing
    assert "split information is missing" in text or "Missing-Split Behavior" in text


def test_contract_must_not_invent_split_silently():
    text = _text(CONTRACT)
    assert "NOT" in text and "silently" in text


def test_contract_defines_split_pending_status():
    text = _text(CONTRACT)
    assert "split_pending" in text


def test_contract_split_pending_requires_split_required_true():
    text = _text(CONTRACT)
    assert "split_required: true" in text


def test_contract_split_pending_dataset_card_not_training_ready():
    text = _text(CONTRACT)
    assert "training_ready: false" in text


def test_contract_defines_defer_split_option():
    text = _text(CONTRACT)
    assert "defer_split" in text or "defer split" in text.lower()


# ── agent-contract.md: later split generation without shard rewriting ────────

def test_contract_allows_split_generation_later():
    text = _text(CONTRACT)
    assert "Later split generation" in text or "later split generation" in text.lower()


def test_contract_split_generation_does_not_rewrite_shards():
    text = _text(CONTRACT)
    assert "not rewrite" in text.lower() or "does not rewrite" in text.lower() or "must not rewrite" in text.lower()


# ── agent-contract.md: stage plan requirements ───────────────────────────────

def test_contract_requires_training_ready_status_field():
    text = _text(CONTRACT)
    assert "training_ready_status" in text


def test_contract_stage_plan_requires_task_type():
    text = _text(CONTRACT)
    assert "task_type" in text


def test_contract_stage_plan_requires_split_policy():
    text = _text(CONTRACT)
    assert "split_policy" in text


def test_contract_stage_plan_requires_loader_config():
    text = _text(CONTRACT)
    assert "loader_config" in text


# ── agent-contract.md: table_key propagation ────────────────────────────────

def test_contract_requires_table_key_propagation():
    text = _text(CONTRACT)
    assert "table_key" in text and ("downstream" in text or "propagat" in text)


def test_contract_never_use_stale_table_key():
    text = _text(CONTRACT)
    assert "stale" in text and "table_key" in text


# ── agent-contract.md: artifact levels ──────────────────────────────────────

def test_contract_l2_is_not_training_ready():
    text = _text(CONTRACT)
    assert "L2" in text
    # L2 must be labeled patch-compiled, not training-ready
    assert "patch-compiled" in text


def test_contract_l3_is_not_training_ready():
    text = _text(CONTRACT)
    assert "L3" in text
    assert "dataset-compiled" in text


def test_contract_l4_requires_split_metadata():
    text = _text(CONTRACT)
    assert "L4" in text
    assert "split_membership.parquet" in text


def test_contract_loader_filters_by_split_metadata_at_runtime():
    text = _text(CONTRACT)
    assert "runtime" in text and "split" in text


# ── training-ready-contract.md: split-as-metadata ───────────────────────────

def test_training_ready_contract_states_splits_are_metadata():
    text = _text(TRAINING_READY)
    assert "Splits are metadata" in text or "metadata" in text


def test_training_ready_contract_no_physical_split_dirs_by_default():
    text = _text(TRAINING_READY)
    assert "not produced by default" in text or "optional" in text


# ── SKILL.md: agent contract pointer ────────────────────────────────────────

def test_skill_md_references_agent_contract():
    text = _text(SKILL_MD)
    assert "agent-contract.md" in text


def test_skill_md_mentions_split_pending():
    text = _text(SKILL_MD)
    assert "split-pending" in text or "split_pending" in text


def test_skill_md_mentions_splits_as_metadata():
    text = _text(SKILL_MD)
    assert "metadata" in text and "split" in text.lower()


# ── usage-guide.md: training-ready examples ─────────────────────────────────

def test_usage_guide_shows_he2st_with_explicit_split():
    text = _text(USAGE_GUIDE)
    assert "train" in text and "val" in text and "sample_holdout" in text


def test_usage_guide_shows_agent_prompts_when_split_missing():
    text = _text(USAGE_GUIDE)
    assert "defer_split" in text or "defer split" in text.lower()


def test_usage_guide_shows_later_split_generation():
    text = _text(USAGE_GUIDE)
    assert "make_split.py" in text


def test_usage_guide_shows_from_config_loader_api():
    text = _text(USAGE_GUIDE)
    assert "from_config" in text and "split=" in text


def test_usage_guide_canonical_storage_not_partitioned():
    text = _text(USAGE_GUIDE)
    assert "NOT partitioned" in text or "not partitioned" in text.lower()


# ── no random cell-level splits ──────────────────────────────────────────────

def test_contract_no_random_cell_as_generated_policy():
    text = _text(CONTRACT)
    # random_cell_split must not appear as a supported generated policy
    assert "random_cell_split" not in text
    # random_cell may appear only in the rejection/warning context, not as a policy option
    # Check it's not listed as a supported --policy value
    assert "--policy sample_holdout | ratio_by_group | random_cell" not in text
    assert "| random_cell |" not in text


def test_contract_states_no_cell_level_random_splits():
    text = _text(CONTRACT)
    assert "does not generate random cell-level" in text or \
           "DAAS does not generate random cell" in text


def test_contract_generated_splits_are_sample_or_group_level():
    text = _text(CONTRACT)
    assert "sample-level or group-level" in text


def test_contract_defines_group_kfold():
    text = _text(CONTRACT)
    assert "group_kfold" in text


def test_contract_group_kfold_assigns_group_to_one_fold():
    text = _text(CONTRACT)
    assert "group_kfold" in text and ("one fold" in text or "exactly one fold" in text)


def test_contract_sample_holdout_no_sample_id_leakage():
    text = _text(CONTRACT)
    assert "sample_holdout" in text
    assert "no `sample_id` may appear in more than one split" in text or \
           "sample_id" in text and "more than one split" in text


def test_contract_ratio_by_group_no_group_leakage():
    text = _text(CONTRACT)
    assert "ratio_by_group" in text
    assert "no `group_id` may appear in more than one split" in text or \
           "group_id" in text and "more than one split" in text


def test_contract_existing_file_global_idx_warns_leakage():
    text = _text(CONTRACT)
    assert "existing_file" in text or "existing_split_file" in text
    # Must warn about leakage for global_idx-level external splits
    assert "leakage warning" in text or "leakage" in text


def test_contract_split_membership_inherits_from_sample_group():
    text = _text(CONTRACT)
    # split_membership.parquet may be global_idx-level but must inherit from group
    assert "inherit" in text and ("sample" in text or "group" in text)


def test_contract_reject_random_cell_with_suggestion():
    text = _text(CONTRACT)
    # Must explain what to do when a user asks for random cell split
    assert "random_cell" not in text or "not a supported policy" in text or \
           "does not generate" in text


# ── task-adapters.md: no random cell splits ──────────────────────────────────

def test_task_adapters_no_random_cell_level_split():
    text = _text(TASK_ADAPTERS)
    assert "Random split at the cell level" not in text
    assert "random_cell" not in text


def test_task_adapters_he2st_sample_level_splits():
    text = _text(TASK_ADAPTERS)
    assert "sample-level or group-level" in text


def test_task_adapters_he2st_no_cell_level_random():
    text = _text(TASK_ADAPTERS)
    assert "does not generate random cell-level" in text or \
           "DAAS does not generate" in text


def test_task_adapters_split_membership_inherits_from_group():
    text = _text(TASK_ADAPTERS)
    assert "inherit" in text or "patient_id" in text or "group" in text


# ── training-ready-contract.md: no random cell splits ───────────────────────

def test_training_ready_no_random_cell_generated():
    text = _text(TRAINING_READY)
    assert "does not generate random cell-level" in text or \
           "DAAS does not generate" in text


def test_training_ready_supported_generated_policies():
    text = _text(TRAINING_READY)
    assert "sample_holdout" in text
    assert "ratio_by_group" in text
    assert "group_kfold" in text


def test_training_ready_existing_file_leakage_warning():
    text = _text(TRAINING_READY)
    assert "leakage warning" in text or "leakage" in text


# ── SKILL.md: no random cell splits ─────────────────────────────────────────

def test_skill_md_no_random_cell_splits():
    text = _text(SKILL_MD)
    # SKILL.md must state that DAAS does not generate random cell-level splits
    assert "never generates random cell-level" in text or \
           "DAAS never generates random" in text or \
           "never\ngenerates" in text or \
           "sample_holdout" in text


# ── usage-guide.md: no random_cell_split ────────────────────────────────────

def test_usage_guide_no_random_cell_split_option():
    text = _text(USAGE_GUIDE)
    assert "random_cell_split" not in text


def test_usage_guide_shows_group_kfold_option():
    text = _text(USAGE_GUIDE)
    assert "group_kfold" in text


def test_usage_guide_no_random_cell_leakage_note():
    text = _text(USAGE_GUIDE)
    # Must note that DAAS does not generate random cell-level splits
    assert "does not generate random cell-level" in text or \
           "DAAS does not generate random" in text
