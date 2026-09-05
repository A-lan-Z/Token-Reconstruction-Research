"""Public geometry diagnostics for TRR-P02.

The helpers in this package are deliberately model agnostic.  They operate on
small, fully known public teacher-prefix activation panels and provide the
deterministic offset, ranking, and separation summaries used by the P02
diagnostic runner.
"""

from .geometry import (
    ContextSpec,
    GeometryDiagnosticError,
    pairwise_token_deformation,
    rank_metrics,
    reference_corrected_query,
    separation_summary,
    summarize_offsets,
)

__all__ = [
    "ContextSpec",
    "GeometryDiagnosticError",
    "pairwise_token_deformation",
    "rank_metrics",
    "reference_corrected_query",
    "separation_summary",
    "summarize_offsets",
]
