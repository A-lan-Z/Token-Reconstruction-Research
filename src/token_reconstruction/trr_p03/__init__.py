"""Task-local primitives for the TRR-P03 natural-sequence readout study.

The package is intentionally split into three small boundaries:

* :mod:`io` validates the opaque observation and frozen-prediction formats;
* :mod:`ranking` implements chunked, deterministic full-vocabulary ranking;
* :mod:`readouts` supplies the raw, projected, and historical A1 score paths.

None of these modules accepts source token IDs during reconstruction.  Truth
is consumed only by the post-freeze scorer in ``scripts/trr_p03/score.py``.
"""

from .ranking import RankResult, rank_queries, score_block
from .readouts import (
    A1ReadoutResult,
    ProjectedReadout,
    rank_a1,
    rank_projected,
    rank_raw,
)

__all__ = [
    "A1ReadoutResult",
    "ProjectedReadout",
    "RankResult",
    "rank_a1",
    "rank_projected",
    "rank_queries",
    "rank_raw",
    "score_block",
]
