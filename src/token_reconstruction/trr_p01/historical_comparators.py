"""Frozen historical A1+A2 control for the TRR-P01 geometry port.

This module is a small, inference-only port of the published owner-R1 policy
``a1a2_43ea0bb737bc075531ca``.  The policy first ranks the full public
vocabulary with the published frozen Alpaca affine lens, retains the ordered
top 512 candidates, and then evaluates exactly the first 256 candidates with a
public prefix simulation.  The A2 score is direct cosine against the observed
activation.  The highest scoring candidate is always committed; there is no
confidence shortcut, fitting, centering, adaptive routing, or abstention.  The
A1 proposal order is the published ``torch.topk(..., sorted=True)`` order, and
A2 ``argmax`` therefore selects the first proposal on an exact score tie.

The historical source of the constants is ``research/A1_A2_CONFIGURATION_PROTOCOL.md``
(the owner-R1 fixed K=256 direct-cosine policy).  The public lens and raw
``torch.topk(..., sorted=True)`` proposal path are in
``reference/strict_bos/round001_teacher.py``; fixed-policy cache simulation and
``scores.argmax`` commit semantics are in
``src/token_reconstruction/a1a2_configuration_search.py``.
The published native records use a different 128x128 geometry.  This module
therefore labels its output as a fixed-rule geometry port: it adapts only the
record shape and invokes the same frozen rule, without training or target-model
calls.  It accepts observations only; no truth or condition label is visible
here.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F


# These values are part of the published control identity.  Keep them local so
# a caller cannot silently change the historical decision rule through a
# generic framework option.
HISTORICAL_POLICY_ID = "a1a2_43ea0bb737bc075531ca"
HISTORICAL_LENS_ARTIFACT_SHA256 = (
    "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
)
PUBLIC_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PUBLIC_MODEL_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
BOS_TOKEN_ID = 128000
CUT_DEPTH = 4
VOCAB_SIZE = 128256
HIDDEN_SIZE = 2048
SEQUENCE_TOKENS = 40
SCORED_TOKENS = SEQUENCE_TOKENS - 1
A1_TOP_K = 512
A1_CHUNK = 256
A2_BUDGET = 256
DEFAULT_RECORD_BATCH_SIZE = 8
MAX_RECORD_BATCH_SIZE = 8


class HistoricalComparatorError(RuntimeError):
    """Raised when the frozen historical control contract is violated."""


@dataclass(frozen=True)
class HistoricalComparatorResult:
    """CPU result and execution receipt for one fixed-rule comparator run.

    ``predictions`` is ``[records, 40]`` with the fixed BOS value in column
    zero.  The remaining tensors are diagnostic proposal/selection outputs;
    they contain ``-1`` or ``-inf`` in the BOS column.  All returned tensors
    are detached CPU tensors so the runner can serialize them before any truth
    access.
    """

    predictions: torch.Tensor
    candidates: torch.Tensor
    proposal_scores: torch.Tensor
    proposal_confidence: torch.Tensor
    selection_scores: torch.Tensor
    winner_margins: torch.Tensor
    proposal_elapsed_seconds: float
    selection_elapsed_seconds: float
    elapsed_seconds: float
    a1_forward_calls: int
    a1_input_token_evaluations: int
    candidate_simulations: int
    executed_candidate_simulations: int
    persistent_cache_commits: int
    candidate_cache_commits: int
    public_prefix_calls: int
    record_batch_size: int
    policy_id: str = HISTORICAL_POLICY_ID
    lens_artifact_sha256: str = HISTORICAL_LENS_ARTIFACT_SHA256

    @property
    def prefix_cache_commits(self) -> int:
        """Alias used by preparation/runner receipts."""

        return self.persistent_cache_commits

    @property
    def prefix_commit_tokens(self) -> int:
        """Published selector terminology for persistent cache writes."""

        return self.persistent_cache_commits


def load_published_frozen_lens(
    path: str | Path, *, device: torch.device
) -> torch.nn.Module:
    """Load the published frozen Alpaca lens with its strict checkpoint checks.

    The implementation is intentionally delegated to the published
    ``reference/strict_bos/round001_teacher.py`` loader.  Keeping this helper
    lazy avoids making the task-local package depend on the reference package
    merely to import the comparator.
    """

    try:
        from reference.strict_bos.round001_teacher import load_frozen_lens
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - packaging failure
        raise HistoricalComparatorError(
            "published strict-BOS lens loader is unavailable"
        ) from exc
    try:
        lens = load_frozen_lens(Path(path), device=torch.device(device))
    except Exception as exc:  # preserve a task-local exception boundary
        raise HistoricalComparatorError("published frozen lens failed to load") from exc
    _validate_frozen_lens(lens)
    return lens


def _validate_frozen_lens(lens: Any) -> None:
    if not isinstance(lens, torch.nn.Module):
        raise HistoricalComparatorError("lens must be a torch.nn.Module")
    if bool(getattr(lens, "training", False)):
        raise HistoricalComparatorError("historical lens must be in eval mode")
    parameters = list(lens.parameters())
    if any(parameter.requires_grad for parameter in parameters):
        raise HistoricalComparatorError("historical lens parameters must be frozen")
    for parameter in parameters:
        if not torch.isfinite(parameter.detach()).all().item():
            raise HistoricalComparatorError("historical lens parameters are non-finite")


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError) as exc:
        raise HistoricalComparatorError(
            "public prefix has no parameters; pass an explicit device"
        ) from exc


def _public_embedding(
    public_prefix: Any,
    *,
    normalized_embeddings: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    embed_tokens = getattr(public_prefix, "embed_tokens", None)
    weight = getattr(embed_tokens, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise HistoricalComparatorError(
            "public prefix must expose a two-dimensional embed_tokens.weight"
        )
    if tuple(weight.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
        raise HistoricalComparatorError(
            "historical comparator requires the pinned public vocabulary geometry"
        )
    if not torch.isfinite(weight.detach()).all().item():
        raise HistoricalComparatorError("public embedding table is non-finite")
    if normalized_embeddings is None:
        normalized = F.normalize(weight.detach().float(), dim=-1)
    else:
        normalized = normalized_embeddings.detach()
        if tuple(normalized.shape) != (VOCAB_SIZE, HIDDEN_SIZE):
            raise HistoricalComparatorError(
                "normalized public embeddings have the wrong geometry"
            )
        if not normalized.dtype.is_floating_point:
            raise HistoricalComparatorError("normalized public embeddings must be floating point")
        if not torch.isfinite(normalized).all().item():
            raise HistoricalComparatorError("normalized public embeddings are non-finite")
        normalized = normalized.float()
    if not torch.isfinite(normalized).all().item():
        raise HistoricalComparatorError("public embedding normalization is non-finite")
    return normalized.to(device=device)



def _last_hidden(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        if not value:
            raise HistoricalComparatorError("public prefix returned an empty tuple")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise HistoricalComparatorError("public prefix returned a non-tensor hidden state")
    if value.ndim == 3:
        return value[:, -1, :]
    if value.ndim == 2:
        return value
    raise HistoricalComparatorError("public prefix returned invalid hidden geometry")


def _repeat_cache(cache: Any, repeats: int) -> Any:
    try:
        candidate_cache = copy.deepcopy(cache)
    except Exception as exc:  # pragma: no cover - backend-specific failure
        raise HistoricalComparatorError(
            "public prefix cache cannot be copied for candidate simulation"
        ) from exc
    repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise HistoricalComparatorError("public prefix cache cannot repeat candidates")
    repeat(int(repeats))
    return candidate_cache


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_observations(observations: torch.Tensor) -> None:
    if not isinstance(observations, torch.Tensor):
        raise HistoricalComparatorError("observations must be a torch.Tensor")
    if tuple(observations.shape) != (
        int(observations.shape[0]) if observations.ndim >= 1 else 0,
        SEQUENCE_TOKENS,
        HIDDEN_SIZE,
    ):
        raise HistoricalComparatorError(
            "historical comparator requires [records,40,2048] observations"
        )
    if observations.shape[0] <= 0:
        raise HistoricalComparatorError("observations must contain at least one record")
    if not observations.dtype.is_floating_point:
        raise HistoricalComparatorError("observations must be floating point")
    if not torch.isfinite(observations).all().item():
        raise HistoricalComparatorError("observations contain non-finite values")


def _validate_prefix(public_prefix: Any, *, device: torch.device) -> None:
    embed_tokens = getattr(public_prefix, "embed_tokens", None)
    if embed_tokens is None:
        raise HistoricalComparatorError("public prefix must expose embed_tokens")
    cut_depth = getattr(public_prefix, "cut_depth", CUT_DEPTH)
    if int(cut_depth) != CUT_DEPTH:
        raise HistoricalComparatorError("historical comparator requires public cut depth 4")
    for name in ("new_cache", "run_cached"):
        if not callable(getattr(public_prefix, name, None)):
            raise HistoricalComparatorError(f"public prefix lacks {name} cache API")
    prefix_device = _module_device(public_prefix)
    if prefix_device != device:
        raise HistoricalComparatorError(
            f"public prefix device {prefix_device} differs from requested device {device}"
        )


def _validate_lens_device(lens: torch.nn.Module, *, device: torch.device) -> None:
    for parameter in lens.parameters():
        if parameter.device != device:
            raise HistoricalComparatorError(
                f"lens device {parameter.device} differs from requested device {device}"
            )


@torch.inference_mode()
def run_fixed_k256_a1_a2(
    *,
    observations: torch.Tensor,
    public_prefix: Any,
    frozen_lens: torch.nn.Module,
    normalized_embeddings: torch.Tensor | None = None,
    device: torch.device | None = None,
    record_batch_size: int = DEFAULT_RECORD_BATCH_SIZE,
    lens_artifact_sha256: str = HISTORICAL_LENS_ARTIFACT_SHA256,
) -> HistoricalComparatorResult:
    """Run the published fixed-K256 direct-cosine A1+A2 control.

    Args:
        observations: Public arm activations with exact shape ``[R,40,2048]``.
            Column zero is the BOS observation and is never scored.
        public_prefix: Frozen public embedding plus layers 0--3.  It must
            expose ``new_cache`` and ``run_cached``; the cache is copied for
            every candidate simulation.
        frozen_lens: The published, already loaded and frozen Alpaca affine
            lens.  Its full-vocabulary logits are used only to form top-512
            proposals.
        normalized_embeddings: Optional precomputed normalized public input
            embedding-table.  If omitted, they are derived from
            ``public_prefix.embed_tokens.weight`` in float32.
        device: Device of the public prefix and lens.  It is inferred from the
            public prefix when omitted.
        record_batch_size: Number of records processed together by the A2
            simulator.  The pilot freezes 8; values 1--8 are allowed for an
            implementation-equivalence check and do not alter the rule.
        lens_artifact_sha256: Expected published lens artifact identity.  The
            default is the recorded historical public lens hash; callers may
            pass an already verified identical value explicitly.

    Returns:
        A CPU-only result.  No target model, condition label, truth tensor, or
        source record is accepted or consulted by this interface.
    """

    _validate_observations(observations)
    if not isinstance(frozen_lens, torch.nn.Module):
        raise HistoricalComparatorError("frozen_lens must be a torch.nn.Module")
    _validate_frozen_lens(frozen_lens)
    if not isinstance(record_batch_size, int) or isinstance(record_batch_size, bool):
        raise HistoricalComparatorError("record_batch_size must be an integer")
    if not 0 < record_batch_size <= MAX_RECORD_BATCH_SIZE:
        raise HistoricalComparatorError("record_batch_size must be between 1 and 8")
    if not isinstance(lens_artifact_sha256, str) or lens_artifact_sha256 != HISTORICAL_LENS_ARTIFACT_SHA256:
        raise HistoricalComparatorError("historical lens artifact identity changed")

    if device is None:
        device = _module_device(public_prefix)
    device = torch.device(device)
    _validate_prefix(public_prefix, device=device)
    _validate_lens_device(frozen_lens, device=device)
    normalized = _public_embedding(
        public_prefix,
        normalized_embeddings=normalized_embeddings,
        device=device,
    )

    records = int(observations.shape[0])
    # Proposal preparation uses exactly the post-BOS observations in the
    # declared 256-row chunks.  Keep all proposal arrays on CPU for artifact
    # isolation and to avoid retaining full-vocabulary logits on the device.
    flat_observations = observations[:, 1:, :].reshape(-1, HIDDEN_SIZE)
    proposal_candidates: list[torch.Tensor] = []
    proposal_scores: list[torch.Tensor] = []
    proposal_confidence: list[torch.Tensor] = []
    _synchronize(device)
    proposal_started = time.perf_counter()
    a1_forward_calls = 0
    for start in range(0, int(flat_observations.shape[0]), A1_CHUNK):
        activation = flat_observations[start : start + A1_CHUNK].to(device=device)
        logits = frozen_lens(activation, normalized)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
            int(activation.shape[0]),
            VOCAB_SIZE,
        ):
            raise HistoricalComparatorError("frozen lens returned invalid vocabulary logits")
        logits = logits.float()
        if not torch.isfinite(logits).all().item():
            raise HistoricalComparatorError("frozen lens logits are non-finite")
        top_scores, top_ids = torch.topk(
            logits, k=A1_TOP_K, dim=1, largest=True, sorted=True
        )
        # Preserve the published proposal order exactly.  In particular, do
        # not add a post-topk token-ID tie sort: the historical selector uses
        # the first candidate returned by torch.topk on an A2 score tie.
        top_ids = top_ids.detach().cpu()
        top_scores = top_scores.detach().float().cpu()
        confidence = torch.exp(
            top_scores[:, 0]
            - torch.logsumexp(logits, dim=1).detach().float().cpu()
        )
        if not torch.isfinite(confidence).all().item():
            raise HistoricalComparatorError("frozen lens confidence is non-finite")
        proposal_candidates.append(top_ids.to(dtype=torch.int32))
        proposal_scores.append(top_scores.to(dtype=torch.float32))
        proposal_confidence.append(confidence.to(dtype=torch.float32))
        a1_forward_calls += 1
    _synchronize(device)
    proposal_elapsed = time.perf_counter() - proposal_started

    candidates = torch.full(
        (records, SEQUENCE_TOKENS, A1_TOP_K),
        -1,
        dtype=torch.int32,
    )
    candidate_score_tensor = torch.full(
        (records, SEQUENCE_TOKENS, A1_TOP_K),
        -float("inf"),
        dtype=torch.float32,
    )
    confidence_tensor = torch.full(
        (records, SEQUENCE_TOKENS), float("nan"), dtype=torch.float32
    )
    candidates[:, 1:] = torch.cat(proposal_candidates, dim=0).reshape(
        records, SCORED_TOKENS, A1_TOP_K
    )
    candidate_score_tensor[:, 1:] = torch.cat(proposal_scores, dim=0).reshape(
        records, SCORED_TOKENS, A1_TOP_K
    )
    confidence_tensor[:, 1:] = torch.cat(proposal_confidence, dim=0).reshape(
        records, SCORED_TOKENS
    )

    predictions = torch.full(
        (records, SEQUENCE_TOKENS), -1, dtype=torch.long
    )
    predictions[:, 0] = BOS_TOKEN_ID
    selection_scores = torch.full(
        (records, SEQUENCE_TOKENS, A2_BUDGET),
        -float("inf"),
        dtype=torch.float32,
    )
    winner_margins = torch.full(
        (records, SEQUENCE_TOKENS), float("nan"), dtype=torch.float32
    )
    persistent_cache_commits = 0
    candidate_cache_commits = 0
    public_prefix_calls = 0
    _synchronize(device)
    selection_started = time.perf_counter()

    for record_start in range(0, records, record_batch_size):
        record_end = min(record_start + record_batch_size, records)
        record_count = record_end - record_start
        cache = public_prefix.new_cache()
        bos = torch.full(
            (record_count, 1), BOS_TOKEN_ID, dtype=torch.long, device=device
        )
        bos_hidden = _last_hidden(public_prefix.run_cached(bos, cache, 0))
        public_prefix_calls += 1
        if tuple(bos_hidden.shape) != (record_count, HIDDEN_SIZE):
            raise HistoricalComparatorError("public BOS cache output has invalid geometry")
        persistent_cache_commits += record_count

        for position in range(1, SEQUENCE_TOKENS):
            ids = candidates[record_start:record_end, position].to(
                device=device, dtype=torch.long
            )
            if ids.shape != (record_count, A1_TOP_K):
                raise HistoricalComparatorError("A1 candidate row geometry changed")
            ids = ids[:, :A2_BUDGET]
            if ids.lt(0).any().item() or ids.ge(VOCAB_SIZE).any().item():
                raise HistoricalComparatorError("A1 candidate IDs are outside the public vocabulary")
            if torch.unique(ids, dim=1).shape[1] != A2_BUDGET:
                raise HistoricalComparatorError("A1 proposal contains duplicate candidate IDs")

            candidate_cache = _repeat_cache(cache, A2_BUDGET)
            simulated = _last_hidden(
                public_prefix.run_cached(
                    ids.reshape(-1, 1), candidate_cache, position
                )
            )
            public_prefix_calls += 1
            if tuple(simulated.shape) != (
                record_count * A2_BUDGET,
                HIDDEN_SIZE,
            ):
                raise HistoricalComparatorError("candidate simulation output has invalid geometry")
            simulated = simulated.reshape(record_count, A2_BUDGET, HIDDEN_SIZE).float()
            target = observations[record_start:record_end, position].to(
                device=device, dtype=torch.float32
            )
            score = F.cosine_similarity(simulated, target[:, None, :], dim=-1)
            if not torch.isfinite(score).all().item():
                raise HistoricalComparatorError("direct-cosine candidate scores are non-finite")
            selection_scores[record_start:record_end, position] = score.cpu()
            top_two = torch.topk(score, k=2, dim=1, largest=True, sorted=True).values
            winner_margins[record_start:record_end, position] = (
                top_two[:, 0] - top_two[:, 1]
            ).cpu()
            # Preserve the published A1 proposal order.  The historical
            # selector uses argmax directly, so an exact A2 score tie chooses
            # the first candidate returned by torch.topk above.
            choice = score.argmax(dim=1)
            chosen = ids.gather(1, choice[:, None]).squeeze(1)
            predictions[record_start:record_end, position] = chosen.cpu()

            commit = chosen[:, None]
            commit_hidden = _last_hidden(
                public_prefix.run_cached(commit, cache, position)
            )
            public_prefix_calls += 1
            if tuple(commit_hidden.shape) != (record_count, HIDDEN_SIZE):
                raise HistoricalComparatorError("public commit output has invalid geometry")
            persistent_cache_commits += record_count
            candidate_cache_commits += record_count * A2_BUDGET
            del candidate_cache, simulated, target, score, top_two
        del cache

    _synchronize(device)
    selection_elapsed = time.perf_counter() - selection_started
    elapsed = proposal_elapsed + selection_elapsed
    logical_simulations = records * SCORED_TOKENS * A2_BUDGET
    # All 39 positions are valid by contract, so every candidate row is
    # executed.  Keep both fields to make skipped/padded future ports visible.
    executed_simulations = candidate_cache_commits
    if executed_simulations != logical_simulations:
        raise HistoricalComparatorError("all-valid fixed comparator did not execute K256 candidates")
    expected_persistent = records * SEQUENCE_TOKENS
    if persistent_cache_commits != expected_persistent:
        raise HistoricalComparatorError("persistent prefix commit count changed")

    return HistoricalComparatorResult(
        predictions=predictions,
        candidates=candidates,
        proposal_scores=candidate_score_tensor,
        proposal_confidence=confidence_tensor,
        selection_scores=selection_scores,
        winner_margins=winner_margins,
        proposal_elapsed_seconds=float(proposal_elapsed),
        selection_elapsed_seconds=float(selection_elapsed),
        elapsed_seconds=float(elapsed),
        a1_forward_calls=a1_forward_calls,
        a1_input_token_evaluations=records * SCORED_TOKENS,
        candidate_simulations=logical_simulations,
        executed_candidate_simulations=executed_simulations,
        persistent_cache_commits=persistent_cache_commits,
        candidate_cache_commits=candidate_cache_commits,
        public_prefix_calls=public_prefix_calls,
        record_batch_size=record_batch_size,
        policy_id=HISTORICAL_POLICY_ID,
        lens_artifact_sha256=lens_artifact_sha256,
    )


# A shorter name for runners that already namespace historical controls.
run_historical_a1_a2 = run_fixed_k256_a1_a2


__all__ = [
    "A1_CHUNK",
    "A1_TOP_K",
    "A2_BUDGET",
    "BOS_TOKEN_ID",
    "CUT_DEPTH",
    "DEFAULT_RECORD_BATCH_SIZE",
    "HIDDEN_SIZE",
    "HISTORICAL_LENS_ARTIFACT_SHA256",
    "HISTORICAL_POLICY_ID",
    "HistoricalComparatorError",
    "HistoricalComparatorResult",
    "PUBLIC_MODEL_ID",
    "PUBLIC_MODEL_REVISION",
    "SEQUENCE_TOKENS",
    "VOCAB_SIZE",
    "load_published_frozen_lens",
    "run_fixed_k256_a1_a2",
    "run_historical_a1_a2",
]
