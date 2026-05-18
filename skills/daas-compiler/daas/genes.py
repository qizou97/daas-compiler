# daas/genes.py
"""Gene panel resolution for compile_dataset.py.

Ensures that gene order is identical across samples after compile.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def resolve_gene_panel(
    adatas: list,
    sample_ids: list[str],
    policy: str,
    explicit_gene_panel: list[str] | None = None,
) -> list[str]:
    """Return the ordered gene panel list (intersection of all samples).

    policy='first_sample': intersection ordered by first sample's var_names.
    policy='sorted': intersection sorted lexicographically.
    policy='explicit': use explicit_gene_panel (must be a subset of intersection).
    """
    if not adatas:
        raise ValueError("adatas is empty")

    intersection = set(adatas[0].var_names.tolist())
    for a in adatas[1:]:
        intersection &= set(a.var_names.tolist())

    if not intersection:
        raise ValueError(
            "Gene intersection is empty — check that all samples share at least one gene. "
            f"Sample IDs: {sample_ids}"
        )

    if policy == "first_sample":
        return [g for g in adatas[0].var_names.tolist() if g in intersection]

    if policy == "sorted":
        return sorted(intersection)

    if policy == "explicit":
        if explicit_gene_panel is None:
            raise ValueError("--gene-panel path required when --gene-order=explicit")
        missing = [g for g in explicit_gene_panel if g not in intersection]
        if missing:
            raise ValueError(
                f"Explicit gene panel contains genes not in the intersection: {missing[:5]}"
            )
        return list(explicit_gene_panel)

    raise ValueError(
        f"Unknown gene order policy: {policy!r}. "
        "Choose from: first_sample, sorted, explicit"
    )


def validate_gene_panel(
    adatas: list,
    sample_ids: list[str],
    gene_panel: list[str],
) -> None:
    """Assert every adata (already sliced to gene_panel) has var_names == gene_panel."""
    for a, sid in zip(adatas, sample_ids):
        actual = list(a.var_names)
        assert actual == gene_panel, (
            f"Sample {sid!r}: var_names differ from gene_panel. "
            f"len(actual)={len(actual)} len(panel)={len(gene_panel)}"
        )


def gene_panel_sha256(gene_panel: list[str]) -> str:
    """Return SHA-256 hex digest of the canonical JSON-serialised gene list."""
    payload = json.dumps(gene_panel, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_gene_panel(compiled_dir, gene_panel: list[str]) -> None:
    """Write gene_panel.json and gene_panel.sha256 to compiled_dir."""
    compiled_dir = Path(compiled_dir)
    compiled_dir.mkdir(parents=True, exist_ok=True)
    (compiled_dir / "gene_panel.json").write_text(json.dumps(gene_panel))
    (compiled_dir / "gene_panel.sha256").write_text(gene_panel_sha256(gene_panel))


__all__ = [
    "resolve_gene_panel",
    "validate_gene_panel",
    "gene_panel_sha256",
    "write_gene_panel",
]
