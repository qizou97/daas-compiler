# DAAS Agent Design Specification

This document defines the expected behavior, architecture, state model, tool contract, and execution rules for the DAAS agent layer.

The goal of the agent is not to be a general-purpose autonomous assistant. The goal is to turn natural-language spatial transcriptomics training requests into auditable, reproducible, task-specific, loader-ready artifacts by orchestrating deterministic DAAS tools.

## Scope

The DAAS agent is responsible for:

- interpreting a user request;
- producing an explicit stage plan;
- selecting and parameterizing registered tools;
- executing the plan through deterministic tools;
- validating intermediate artifacts;
- repairing recoverable failures;
- producing a reproducible final report.

The DAAS agent must not directly mutate datasets, invent benchmark results, bypass validation, or silently change split policies.

## Design Principles

1. **LLM as planner, tools as executors**

   The language model may interpret intent, plan stages, choose tools, explain results, and propose repairs. All data processing, splitting, exporting, training, evaluation, and profiling must be performed by deterministic tools.

2. **Artifact-first execution**

   Every meaningful step must produce explicit artifact references. The agent state must refer to artifacts by path or URI, not by implicit in-memory assumptions.

3. **Reproducibility over autonomy**

   The agent must preserve configurations, random seeds, schema versions, environment information, tool versions, source commit hashes, and validation reports.

4. **No benchmark numbers without tool output**

   The agent must never report metrics, throughput, GPU utilization, or biological conclusions unless they were produced by the relevant benchmark, profiler, or analysis tool.

5. **No split without leakage checks**

   Any train/validation/test split must produce a split report and a leakage check report. Group-aware split policies such as patient-level, slide-level, batch-level, and study-level splits must be explicit.

6. **Training-ready means loader-ready**

   A dataset is training-ready only when it can be consumed by the target loader without additional preprocessing, joining, splitting, gene reordering, image conversion, target conversion, or artifact conversion.

7. **Simple execution loop first**

   The first implementation should use a controlled `plan -> execute -> observe -> repair -> report` loop. Multi-agent delegation, long-term memory, and open-ended reflection are optional later extensions.

## Relationship to DAAS Artifact Levels

The agent should operate over the existing DAAS artifact levels:

| Level | Name | Agent expectation |
|---|---|---|
| L0 | Raw | Input data is not modified in place. |
| L1 | Canonical | Optional normalization and filtering stages may produce canonical SpatialData. |
| L2 | Patch-compiled | Per-sample extraction artifacts are produced by extraction tools. |
| L3 | Dataset-compiled | Cross-sample manifests and expression matrices are produced by compile tools. |
| L4 | Task-ready / Training-ready | Task adapters produce loader-ready datasets with split metadata and validation reports. |
| L5 | Benchmark-ready | Benchmark tooling freezes L4 artifacts with provenance, fixed splits, configs, and metrics. |

The agent should treat L4 as the default target for training data construction and L5 as the default target for benchmark reporting.

## Supported Task Families

The initial DAAS agent design should support two broad supervised learning task families:

1. **Image(s) to table row**

   Example: H&E patch or multiple H&E patches mapped to one cell-level or spot-level omics vector.

2. **Table row(s) to table row**

   Example: one or more omics feature rows mapped to another omics feature row, such as RNA-to-protein or one spatial omics representation to another.

Both task families must be represented through explicit task schemas rather than custom ad hoc code paths.

## Agent Roles

The system may eventually expose multiple logical agents, but the first implementation should keep a single orchestrator with modular responsibilities.

### Workflow Planner

Converts the user request into a task DAG.

Responsibilities:

- identify task type;
- identify modalities;
- identify requested split policy;
- identify target artifact level;
- identify benchmark and analysis requirements;
- produce a structured stage plan.

### Execution Controller

Executes ready tasks from the DAG in deterministic order.

Responsibilities:

- choose the next runnable task;
- call the registered tool;
- capture outputs, logs, metrics, and errors;
- update agent state;
- stop on unrecoverable errors.

### Repair Controller

Handles recoverable failures.

Allowed repair categories:

- missing optional file;
- schema mismatch with clear mapping;
- invalid or empty split;
- empty shard;
- CUDA out-of-memory;
- NaN or diverging loss;
- missing dependency with install instruction;
- unsupported model adapter.

The repair controller must not fabricate missing data, silently change biological targets, silently switch split policies, or hide failed validation.

### Report Writer

Produces a final user-facing report grounded only in state and artifacts.

The report must include:

- request summary;
- executed plan;
- produced artifacts;
- validation results;
- split policy and leakage summary;
- benchmark metrics if available;
- system profiling summary if available;
- biological analysis if available;
- warnings and limitations;
- reproduction instructions.

## Default Workflow DAG

The default workflow is:

```text
inspect_data
   |
   v
build_dataset
   |
   v
split_dataset
   |
   v
export_training_ready_dataset
   |
   +--------------------------+
   |                          |
   v                          v
generate_loader_examples   resolve_models
   |                          |
   +------------+-------------+
                |
                v
          run_benchmark
                |
        +-------+-------+
        |               |
        v               v
  analyze_metrics   profile_system
        |
        v
  analyze_biology
        |
        v
  write_report
```

The planner may remove stages that are irrelevant to the user request, but it must not bypass required validation stages for produced artifacts.

## Minimal MVP Workflow

The first implementation should support:

- H&E patch or patches to omics vector;
- local filesystem inputs;
- L4 training-ready export;
- metadata-based train/validation/test split;
- patient-level and sample-level split policies;
- WebDataset-compatible streaming shards;
- one ViT-like image baseline;
- one simple MLP or linear baseline for tabular inputs;
- gene-wise Pearson and Spearman metrics;
- Markdown report generation.

The first implementation should defer:

- automatic GitHub model mining;
- multi-agent collaboration;
- long-term cross-project memory;
- open-ended web research;
- complex biological prior database integration;
- automatic paper reading;
- arbitrary shell access.

## Agent State Schema

The agent state should be serializable as JSON and persisted under the run directory.

```python
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel


class TaskSpec(BaseModel):
    task_id: str
    task_type: str
    inputs: Dict[str, Any]
    depends_on: List[str] = []
    status: Literal["pending", "running", "success", "failed", "skipped"] = "pending"
    output_refs: Dict[str, str] = {}
    error: Optional[str] = None


class ToolResult(BaseModel):
    ok: bool
    summary: str
    artifact_refs: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    logs: List[str] = []
    warnings: List[str] = []
    error: Optional[str] = None


class AgentState(BaseModel):
    run_id: str
    user_request: str
    project_root: str
    target_artifact_level: str = "L4"
    tasks: List[TaskSpec]
    artifacts: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    warnings: List[str] = []
    final_report: Optional[str] = None
```

Recommended persisted layout:

```text
.runs/{run_id}/
  state.json
  plan.json
  tool_calls.jsonl
  logs/
  artifacts/
  reports/
```

## Tool Registry Contract

Every tool must declare:

- `name`;
- `description`;
- `input_schema`;
- `output_schema`;
- required artifact level inputs;
- produced artifact level outputs;
- recoverable error types;
- side-effect boundaries.

A tool must return a `ToolResult`. It must not communicate success only through stdout.

Example:

```python
class SplitDatasetTool:
    name = "split_dataset"
    description = "Create train/validation/test split metadata and leakage diagnostics."

    input_schema = {
        "manifest_path": "Path to L3 or L4 manifest.",
        "strategy": "random | patient_level | sample_level | slide_level | batch_level | spatial_block | leave_one_group_out",
        "ratios": "Three floats such as [0.8, 0.1, 0.1].",
        "group_key": "Metadata key for grouped split, such as patient_id.",
        "seed": "Integer random seed."
    }

    output_schema = {
        "split_membership": "Path to split_membership.parquet.",
        "train_json": "Path to train.json.",
        "val_json": "Path to val.json.",
        "test_json": "Path to test.json.",
        "split_report": "Path to split_report.json.",
        "leakage_report": "Path to leakage_report.json."
    }
```

## Core Tool Set

### DataInspectionTool

Purpose: inspect raw or intermediate input data before planning destructive or expensive operations.

Must detect when available:

- modality names;
- sample IDs;
- patient IDs;
- slide IDs;
- image layers;
- spatial coordinates;
- cell or spot identifiers;
- expression matrix shape;
- gene or feature names;
- coordinate system assumptions;
- missing required metadata.

Required outputs:

- `schema_report.json`;
- `inspection_summary.md`;
- warnings for missing or ambiguous metadata.

### DatasetBuildTool

Purpose: transform inspected inputs into DAAS-compatible sample records.

Must support:

- image(s) to target row;
- row(s) to target row;
- explicit metadata propagation;
- stable sample IDs;
- deterministic ordering.

Required outputs:

- sample manifest;
- feature metadata;
- build report;
- validation report.

### SplitPlannerTool

Purpose: produce split metadata and verify leakage.

Must support:

- random split;
- patient-level split;
- sample-level split;
- slide-level split;
- batch-level split;
- spatial block split;
- leave-one-group-out split.

Required outputs:

- split membership file;
- train/validation/test JSON files;
- split report;
- leakage diagnostics.

### TrainingReadyExportTool

Purpose: produce L4 task-ready artifacts.

Required outputs:

```text
data/
  shard-000000.tar
  shard-000001.tar
  ...
splits/
  train.json
  val.json
  test.json
  split_membership.parquet
  split_report.json
  leakage_report.json
gene_panel.json
gene_panel.sha256
task_config.yaml
loader_config.yaml
dataset_card.json
validation_report.json
```

Physical `train/`, `val/`, and `test/` shard directories are optional export mode only. The default is split metadata over a single canonical data directory.

### LoaderExampleTool

Purpose: generate minimal examples for consuming produced artifacts.

Must output:

- PyTorch `Dataset` example;
- PyTorch `DataLoader` example;
- optional pure `webdataset` pipeline;
- notes about split membership and gene ordering.

### ModelResolverTool

Purpose: resolve model implementations into DAAS-compatible adapters.

Must record:

- model name;
- source;
- version or commit hash;
- license if known;
- adapter path;
- expected input schema;
- expected output schema;
- default configuration;
- unsupported assumptions.

Initial MVP may use only in-repo baseline models.

### BenchmarkRunnerTool

Purpose: run training and evaluation over frozen L4 or L5 artifacts.

Must record:

- model adapter;
- dataset version;
- split version;
- config;
- seed;
- hardware summary;
- environment summary;
- logs;
- metrics;
- failure reason if failed.

### MetricsAnalyzerTool

Purpose: compute and summarize benchmark metrics.

Recommended metrics:

- gene-wise Pearson correlation;
- gene-wise Spearman correlation;
- MSE;
- MAE;
- highly variable gene subset metrics;
- sample-level aggregate metrics;
- per-split metrics;
- calibration or residual summaries when applicable.

### BiologyPriorAnalyzerTool

Purpose: analyze whether model predictions preserve biologically meaningful structure.

Possible analyses:

- marker gene performance;
- gene co-expression preservation;
- mutual exclusivity preservation;
- pathway score correlation;
- spatial autocorrelation consistency;
- cell-type-specific performance.

This tool must clearly distinguish measured analysis from biological speculation.

### SystemProfilerTool

Purpose: record system-level performance of dataset loading and training.

Recommended outputs:

- samples per second;
- shard read throughput;
- CPU utilization;
- GPU utilization;
- GPU memory usage;
- I/O wait;
- epoch time;
- data loading time versus compute time.

### ReportWriterTool

Purpose: generate the final report from state and artifact references.

The report must not invent missing outputs. Missing metrics or artifacts must be listed as missing or not run.

## Execution Loop

Reference loop:

```python
class DAASAgent:
    def __init__(self, planner, registry, reporter):
        self.planner = planner
        self.registry = registry
        self.reporter = reporter

    def run(self, user_request: str, project_root: str) -> AgentState:
        state = self.planner.plan(user_request, project_root)
        persist_state(state)

        while has_pending_tasks(state):
            task = next_runnable_task(state)
            task.status = "running"
            persist_state(state)

            tool = self.registry.get(task.task_type)
            result = tool.run(**task.inputs)

            state = observe_result(state, task, result)
            persist_state(state)

            if not result.ok:
                state = attempt_repair_or_fail(state, task, result)
                persist_state(state)

            if has_unrecoverable_failure(state):
                break

        state.final_report = self.reporter.write(state)
        persist_state(state)
        return state
```

`next_runnable_task` should be deterministic and based on DAG dependencies. The LLM should not freely choose arbitrary next actions during normal execution.

## Planning Rules

The planner must:

- inspect data before assuming schema;
- ask for or infer only safe defaults;
- make split policy explicit;
- make target artifact level explicit;
- include validation stages for generated artifacts;
- include benchmark stages only when requested or when the workflow target is L5;
- include profiling stages only when requested or when comparing system efficiency;
- record assumptions in the plan.

The planner must not:

- assume patient-level metadata exists before inspection;
- use random split when the user requested patient-level or slide-level isolation;
- silently collapse multiple modalities;
- choose a model implementation without recording source and adapter assumptions;
- skip validation to save time.

## Repair Rules

Allowed repairs:

- retry with a smaller batch size after CUDA OOM;
- enable mixed precision if supported and recorded;
- reduce number of data loader workers if worker crashes;
- regenerate split with a valid group key when the requested key is absent and a clear equivalent exists;
- stop and report missing required metadata when no safe equivalent exists;
- install or report missing optional dependencies according to dependency policy.

Forbidden repairs:

- changing target genes without user-visible record;
- switching from patient-level split to random split silently;
- removing failed samples silently;
- reporting partial benchmark metrics as full metrics;
- overwriting raw input data;
- editing generated artifacts without updating provenance.

## Memory Model

### Run Memory

Per-run state stored under `.runs/{run_id}`.

Includes:

- task graph;
- tool calls;
- tool outputs;
- logs;
- errors;
- metrics;
- final report.

### Project Memory

Project-level facts stored under `.daas/project_memory.json` when available.

May include:

- known dataset roots;
- known modality mappings;
- preferred split policy;
- previous benchmark artifact references;
- known model adapters.

Project memory must not override the current inspection report when they conflict.

### Knowledge Memory

Reusable general knowledge may be added later through curated references or RAG. It must not be used as a substitute for inspecting project artifacts.

## Reproducibility Requirements

Every L5 benchmark-ready result must include:

- run ID;
- dataset artifact hash or manifest hash;
- split artifact hash;
- gene panel hash if applicable;
- model adapter source and version;
- config file;
- random seed;
- package/environment summary;
- hardware summary;
- metric outputs;
- logs;
- report generation timestamp.

## Safety and Audit Requirements

The agent must maintain an audit trail of:

- user request;
- generated plan;
- assumptions;
- tool calls;
- command-line arguments if tools shell out internally;
- artifacts produced;
- validation results;
- errors and repairs;
- final report.

The agent must never require arbitrary shell access for normal workflows. If a tool internally runs scripts, the script name, arguments, return code, stdout path, and stderr path must be recorded.

## Prompt Contract

The orchestrator prompt should enforce the following behavior:

```text
You are the DAAS Orchestrator Agent.

You do not directly modify data or train models.
You operate only through registered tools.

Your job:
1. Understand the user's biomedical ML dataset request.
2. Produce a task DAG.
3. Select tools and provide structured arguments.
4. Inspect tool results.
5. Retry only when the error is recoverable.
6. Produce a reproducible final report.

Rules:
- Never invent file paths.
- Never assume dataset schema without an inspection result.
- Never create train/validation/test splits without leakage checks.
- Never report benchmark numbers unless benchmark tools produced them.
- Always preserve artifact references.
- Always include manifest, config, random seed, source versions, and environment information when reporting benchmark results.
```

## Implementation Roadmap

### Phase 1: Deterministic Tool Foundation

- Define `TaskSpec`, `ToolResult`, and `AgentState`.
- Implement the tool registry.
- Ensure each tool returns structured outputs.
- Persist run state.

### Phase 2: Minimal Agent Loop

- Implement planner for the default workflow.
- Implement deterministic DAG execution.
- Implement report writer.
- Support MVP HE-to-omics workflow.

### Phase 3: Validation and Repair

- Add recoverable error taxonomy.
- Add limited repair policies.
- Add validation gates before L4 and L5 claims.

### Phase 4: Benchmark Registry

- Add baseline model adapters.
- Freeze benchmark configs.
- Record provenance and environment metadata.

### Phase 5: System and Biology Analysis

- Add profiling tools.
- Add gene-level and biology-aware analysis tools.
- Add richer benchmark reports.

### Phase 6: Optional Multi-Agent Expansion

Only after the single orchestrator is stable, split the system into logical agents:

- Workflow Agent;
- Execution Agent;
- Benchmark Agent;
- Analysis Agent.

The external behavior should remain artifact-driven and auditable.

## Acceptance Criteria

A DAAS agent implementation is acceptable only if:

- it can serialize and resume run state;
- it never reports nonexistent artifacts;
- it records every tool call;
- it validates generated training-ready artifacts;
- it produces split leakage diagnostics;
- it can generate a minimal loader example for L4 artifacts;
- it can run at least one baseline benchmark or explicitly report that benchmarking was not requested;
- it generates a final report grounded in artifacts and metrics;
- it preserves enough provenance to reproduce the workflow.
