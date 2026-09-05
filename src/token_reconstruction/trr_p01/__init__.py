"""Task-local primitives for the TRR-P01 boundary-prototype pilot."""

from .boundary_prototype import (
    BOUNDARY_TABLE_SCHEMA,
    BOS_TOKEN_ID,
    DEFAULT_PROTOTYPE_BATCH_SIZE,
    DEFAULT_PROTOTYPE_CHUNK_SIZE,
    DEFAULT_QUERY_CHUNK_SIZE,
    CorrectionResult,
    NearestResult,
    PrototypeBuildStats,
    PrototypeTable,
    PrototypeError,
    apply_reference_correction,
    nearest_embedding,
)

__all__ = [
    "BOUNDARY_TABLE_SCHEMA",
    "BOS_TOKEN_ID",
    "DEFAULT_PROTOTYPE_BATCH_SIZE",
    "DEFAULT_PROTOTYPE_CHUNK_SIZE",
    "DEFAULT_QUERY_CHUNK_SIZE",
    "CorrectionResult",
    "NearestResult",
    "PrototypeBuildStats",
    "PrototypeTable",
    "PrototypeError",
    "apply_reference_correction",
    "nearest_embedding",
]
