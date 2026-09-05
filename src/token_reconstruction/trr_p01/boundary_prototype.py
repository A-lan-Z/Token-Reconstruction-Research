"""Deterministic full-vocabulary boundary-prototype lookup for TRR-P01.

The prototype construction is deliberately simple:

``b_v = public_prefix([BOS, v])[1]``

The table is public preparation state, not a fitted inverse.  Reconstruction
queries only the frozen table and (for the optional correction arm) the public
prefix with a reconstructed prefix.  No source truth is accepted by this
module.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from pathlib import Path
import time
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F


BOUNDARY_TABLE_SCHEMA = "token-reconstruction.trr-p01-boundary-prototypes.v1"
BOS_TOKEN_ID = 128000
DEFAULT_PROTOTYPE_BATCH_SIZE = 256
DEFAULT_QUERY_CHUNK_SIZE = 256
DEFAULT_PROTOTYPE_CHUNK_SIZE = 8192
_METRICS = ("cosine", "l2")


class PrototypeError(RuntimeError):
    """Raised when a prototype table or lookup violates its contract."""


@dataclass(frozen=True)
class PrototypeBuildStats:
    """Preparation evidence for one public prototype-table build."""

    vocab_size: int
    hidden_size: int
    batch_size: int
    forward_calls: int
    input_token_evaluations: int
    elapsed_seconds: float
    output_dtype: str
    @property
    def committed_tokens(self) -> int:
        """Backward-compatible name for the 2*V input-token evaluation count."""
        return self.input_token_evaluations



@dataclass(frozen=True)
class NearestResult:
    """Full-vocabulary nearest-neighbour output in deterministic ID order."""

    predictions: torch.Tensor
    scores: torch.Tensor
    runner_up_scores: torch.Tensor
    margins: torch.Tensor


@dataclass(frozen=True)
class CorrectionResult:
    """Output and cost evidence for the fixed-reference causal arm."""

    predictions: torch.Tensor
    scores: torch.Tensor
    margins: torch.Tensor
    offsets: torch.Tensor
    reference_evaluations: int
    persistent_cache_commits: int
    probe_cache_commits: int


def _require_metric(metric: str) -> str:
    if metric not in _METRICS:
        raise PrototypeError(f"metric must be one of {_METRICS}, got {metric!r}")
    return metric


def _regular_create_only(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise PrototypeError(f"artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_matrix(value: torch.Tensor, *, name: str) -> None:
    if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
        raise PrototypeError(f"{name} must be a non-empty [rows, hidden] matrix")
    if not value.dtype.is_floating_point:
        raise PrototypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all().item():
        raise PrototypeError(f"{name} contains non-finite values")


def _validate_queries(value: torch.Tensor, *, hidden_size: int | None = None) -> None:
    _validate_matrix(value, name="queries")
    if hidden_size is not None and int(value.shape[1]) != int(hidden_size):
        raise PrototypeError("query hidden size differs from the prototype table")


def _stable_top2(
    current_scores: torch.Tensor,
    current_ids: torch.Tensor,
    new_scores: torch.Tensor,
    new_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge candidates by score descending and token ID ascending.

    Two stable sorts avoid the arbitrary tie order of ``torch.topk``.  The
    first sort establishes ID order, and the second sort establishes descending
    score while preserving the ID order for equal scores.
    """

    scores = torch.cat((current_scores, new_scores), dim=1)
    ids = torch.cat((current_ids, new_ids), dim=1)
    by_id = torch.argsort(ids, dim=1, descending=False, stable=True)
    ids = ids.gather(1, by_id)
    scores = scores.gather(1, by_id)
    by_score = torch.argsort(scores, dim=1, descending=True, stable=True)
    by_score = by_score[:, :2]
    return scores.gather(1, by_score), ids.gather(1, by_score)


def _score_block(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    metric: str,
) -> torch.Tensor:
    """Compute higher-is-better scores without materializing pairwise deltas."""

    if metric == "cosine":
        return F.normalize(queries, dim=1, eps=1e-12) @ F.normalize(
            prototypes, dim=1, eps=1e-12
        ).transpose(0, 1)
    # -squared L2 is used so all lookup arms share a higher-is-better score
    # convention.  The norm identity keeps the workspace at [Q, P].
    query_sq = queries.square().sum(dim=1, keepdim=True)
    prototype_sq = prototypes.square().sum(dim=1).view(1, -1)
    return -(query_sq + prototype_sq - 2.0 * (queries @ prototypes.transpose(0, 1)))


def _nearest_matrix(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    metric: str,
    query_chunk_size: int,
    prototype_chunk_size: int,
) -> NearestResult:
    _require_metric(metric)
    _validate_matrix(prototypes, name="prototypes")
    _validate_queries(queries, hidden_size=int(prototypes.shape[1]))
    if query_chunk_size <= 0 or prototype_chunk_size <= 0:
        raise PrototypeError("lookup chunk sizes must be positive")

    rows = int(queries.shape[0])
    vocab_size = int(prototypes.shape[0])
    all_predictions: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_runner_up: list[torch.Tensor] = []
    for q_start in range(0, rows, query_chunk_size):
        q = queries[q_start : q_start + query_chunk_size].float()
        # The sentinel ID is larger than every valid ID and remains behind all
        # real IDs if every score is tied (including all-zero L2 queries).
        best_scores = torch.full((q.shape[0], 2), -float("inf"), dtype=torch.float32)
        best_ids = torch.full((q.shape[0], 2), vocab_size, dtype=torch.long)
        for p_start in range(0, vocab_size, prototype_chunk_size):
            p = prototypes[p_start : p_start + prototype_chunk_size].float()
            block = _score_block(q, p, metric).float()
            ids = torch.arange(
                p_start,
                p_start + p.shape[0],
                dtype=torch.long,
            ).view(1, -1).expand(q.shape[0], -1)
            block_scores, block_ids = _stable_top2(
                torch.full((q.shape[0], 2), -float("inf")),
                torch.full((q.shape[0], 2), vocab_size, dtype=torch.long),
                block,
                ids,
            )
            best_scores, best_ids = _stable_top2(
                best_scores, best_ids, block_scores, block_ids
            )
        if (best_ids[:, 0] >= vocab_size).any().item():
            raise PrototypeError("prototype lookup produced no valid candidate")
        all_predictions.append(best_ids[:, 0].cpu())
        all_scores.append(best_scores[:, 0].cpu())
        all_runner_up.append(best_scores[:, 1].cpu())
    scores = torch.cat(all_scores)
    runner_up = torch.cat(all_runner_up)
    return NearestResult(
        predictions=torch.cat(all_predictions),
        scores=scores,
        runner_up_scores=runner_up,
        margins=scores - runner_up,
    )


class PrototypeTable:
    """A frozen full-vocabulary table of public boundary activations."""

    def __init__(
        self,
        prototypes: torch.Tensor,
        *,
        model_id: str = "",
        model_revision: str = "",
        cut_depth: int = 4,
        bos_token_id: int = BOS_TOKEN_ID,
        construction: str = "public_prefix([BOS,v])[1]",
    ) -> None:
        _validate_matrix(prototypes, name="prototypes")
        if int(bos_token_id) != BOS_TOKEN_ID:
            raise PrototypeError("declared BOS token changed")
        if not isinstance(cut_depth, int) or isinstance(cut_depth, bool) or cut_depth < 0:
            raise PrototypeError("cut_depth must be a non-negative integer")
        if not isinstance(construction, str) or not construction:
            raise PrototypeError("construction description is required")
        self.prototypes = prototypes.detach().cpu().contiguous()
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.cut_depth = int(cut_depth)
        self.bos_token_id = int(bos_token_id)
        self.construction = construction

    @property
    def vocab_size(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.prototypes.shape[1])

    def nearest(
        self,
        queries: torch.Tensor,
        *,
        metric: str = "cosine",
        query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
        prototype_chunk_size: int = DEFAULT_PROTOTYPE_CHUNK_SIZE,
    ) -> NearestResult:
        """Find nearest full-vocabulary prototypes with stable ties."""

        return _nearest_matrix(
            queries.detach().cpu(),
            self.prototypes,
            metric=metric,
            query_chunk_size=query_chunk_size,
            prototype_chunk_size=prototype_chunk_size,
        )

    @classmethod
    @torch.inference_mode()
    def build(
        cls,
        public_prefix: Any,
        *,
        vocab_size: int | None = None,
        bos_token_id: int = BOS_TOKEN_ID,
        cut_depth: int | None = None,
        model_id: str = "",
        model_revision: str = "",
        batch_size: int = DEFAULT_PROTOTYPE_BATCH_SIZE,
        storage_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        return_stats: bool = False,
    ) -> "PrototypeTable | tuple[PrototypeTable, PrototypeBuildStats]":
        """Build ``[BOS,v]`` prototypes using only the public prefix.

        ``storage_dtype=None`` preserves the prefix output dtype.  The pilot
        stores bfloat16 outputs from the pinned Llama model and promotes to
        float32 only during lookup.  ``return_stats`` is convenient for the
        evaluator's preparation receipt while keeping the default API simple.
        """

        if int(bos_token_id) != BOS_TOKEN_ID:
            raise PrototypeError("declared BOS token changed")
        if batch_size <= 0:
            raise PrototypeError("prototype batch size must be positive")
        embed = getattr(public_prefix, "embed_tokens", None)
        if embed is None or not hasattr(embed, "num_embeddings"):
            raise PrototypeError("public prefix must expose embed_tokens.num_embeddings")
        inferred_vocab = int(embed.num_embeddings)
        if vocab_size is None:
            vocab_size = inferred_vocab
        if int(vocab_size) != inferred_vocab or int(vocab_size) <= 0:
            raise PrototypeError("vocab size must equal the public embedding table")
        if device is None:
            try:
                device = next(public_prefix.parameters()).device
            except StopIteration as exc:
                raise PrototypeError("public prefix has no parameters; pass device") from exc
        if cut_depth is None:
            cut_depth = int(getattr(public_prefix, "cut_depth", 4))

        started = time.perf_counter()
        table: torch.Tensor | None = None
        forward_calls = 0
        bos = torch.full((batch_size,), BOS_TOKEN_ID, dtype=torch.long, device=device)
        for start in range(0, int(vocab_size), batch_size):
            stop = min(start + batch_size, int(vocab_size))
            ids = torch.arange(start, stop, dtype=torch.long, device=device)
            input_ids = torch.stack((bos[: ids.shape[0]], ids), dim=1)
            output = public_prefix.forward_full(input_ids)
            if output.ndim != 3 or tuple(output.shape[:2]) != (ids.shape[0], 2):
                raise PrototypeError("public prefix returned invalid prototype geometry")
            values = output[:, 1, :].detach()
            _validate_matrix(values, name="public prefix prototypes")
            if table is None:
                dtype = storage_dtype or values.dtype
                if not dtype.is_floating_point:
                    raise PrototypeError("prototype storage dtype must be floating point")
                table = torch.empty(
                    (int(vocab_size), int(values.shape[1])), dtype=dtype, device="cpu"
                )
            elif int(values.shape[1]) != int(table.shape[1]):
                raise PrototypeError("public prefix hidden size changed during table build")
            table[start:stop].copy_(values.to(device="cpu", dtype=table.dtype))
            forward_calls += 1
        assert table is not None
        result = cls(
            table,
            model_id=model_id,
            model_revision=model_revision,
            cut_depth=int(cut_depth),
            bos_token_id=bos_token_id,
        )
        stats = PrototypeBuildStats(
            vocab_size=int(vocab_size),
            hidden_size=result.hidden_size,
            batch_size=int(batch_size),
            forward_calls=forward_calls,
            input_token_evaluations=int(vocab_size) * 2,
            elapsed_seconds=time.perf_counter() - started,
            output_dtype=str(table.dtype).replace("torch.", ""),
        )
        return (result, stats) if return_stats else result

    def save(self, path: str | os.PathLike[str]) -> str:
        """Create a self-describing, create-only safetensors artifact."""

        destination = Path(path)
        _regular_create_only(destination)
        metadata = {
            "schema": BOUNDARY_TABLE_SCHEMA,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "cut_depth": str(self.cut_depth),
            "bos_token_id": str(self.bos_token_id),
            "vocab_size": str(self.vocab_size),
            "hidden_size": str(self.hidden_size),
            "dtype": str(self.prototypes.dtype).replace("torch.", ""),
            "construction": self.construction,
            "truth_opened": "false",
        }
        save_file({"prototypes": self.prototypes}, destination, metadata=metadata)
        import hashlib

        digest = hashlib.sha256()
        with destination.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_model_id: str | None = None,
        expected_model_revision: str | None = None,
        expected_cut_depth: int | None = None,
        expected_vocab_size: int | None = None,
        expected_hidden_size: int | None = None,
    ) -> "PrototypeTable":
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise PrototypeError(f"prototype artifact must be a regular file: {source}")
        try:
            with safe_open(source, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"prototypes"}:
                    raise PrototypeError("prototype artifact tensor fields changed")
                metadata = handle.metadata() or {}
                required = {
                    "schema",
                    "model_id",
                    "model_revision",
                    "cut_depth",
                    "bos_token_id",
                    "vocab_size",
                    "hidden_size",
                    "dtype",
                    "construction",
                    "truth_opened",
                }
                if set(metadata) != required:
                    raise PrototypeError("prototype artifact metadata fields changed")
                if metadata["schema"] != BOUNDARY_TABLE_SCHEMA or metadata["truth_opened"] != "false":
                    raise PrototypeError("prototype artifact schema or truth state changed")
                if int(metadata["bos_token_id"]) != BOS_TOKEN_ID:
                    raise PrototypeError("prototype artifact BOS changed")
                prototypes = handle.get_tensor("prototypes")
        except PrototypeError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise PrototypeError(f"invalid prototype artifact: {source}") from exc

        if tuple(prototypes.shape) != (
            int(metadata["vocab_size"]),
            int(metadata["hidden_size"]),
        ):
            raise PrototypeError("prototype tensor shape disagrees with metadata")
        if expected_model_id is not None and metadata["model_id"] != expected_model_id:
            raise PrototypeError("prototype model identity changed")
        if expected_model_revision is not None and metadata["model_revision"] != expected_model_revision:
            raise PrototypeError("prototype model revision changed")
        if expected_cut_depth is not None and int(metadata["cut_depth"]) != int(expected_cut_depth):
            raise PrototypeError("prototype cut depth changed")
        if expected_vocab_size is not None and int(metadata["vocab_size"]) != int(expected_vocab_size):
            raise PrototypeError("prototype vocabulary changed")
        if expected_hidden_size is not None and int(metadata["hidden_size"]) != int(expected_hidden_size):
            raise PrototypeError("prototype hidden size changed")
        dtype = str(prototypes.dtype).replace("torch.", "")
        if metadata["dtype"] != dtype:
            raise PrototypeError("prototype dtype metadata changed")
        return cls(
            prototypes,
            model_id=metadata["model_id"],
            model_revision=metadata["model_revision"],
            cut_depth=int(metadata["cut_depth"]),
            bos_token_id=int(metadata["bos_token_id"]),
            construction=metadata["construction"],
        )


def nearest_embedding(
    queries: torch.Tensor,
    embedding_table: torch.Tensor,
    *,
    metric: str = "cosine",
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    prototype_chunk_size: int = DEFAULT_PROTOTYPE_CHUNK_SIZE,
) -> NearestResult:
    """Run the same deterministic lookup against raw public embeddings."""

    return _nearest_matrix(
        queries.detach().cpu(),
        embedding_table.detach().cpu(),
        metric=metric,
        query_chunk_size=query_chunk_size,
        prototype_chunk_size=prototype_chunk_size,
    )


def _last_hidden(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        return value[:, -1, :]
    if value.ndim == 2:
        return value
    raise PrototypeError("public prefix probe returned invalid hidden geometry")


@torch.inference_mode()
def apply_reference_correction(
    *,
    observations: torch.Tensor,
    public_prefix: Any,
    prototypes: PrototypeTable,
    metric: str = "cosine",
    reference_token: int = 220,
    bos_token_id: int = BOS_TOKEN_ID,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    prototype_chunk_size: int = DEFAULT_PROTOTYPE_CHUNK_SIZE,
    device: torch.device | None = None,
) -> CorrectionResult:
    """Apply a single public reference probe per position.

    At position ``i``, only the already reconstructed prefix is committed to a
    persistent cache.  A deep-copied cache receives the public reference token;
    its output is compared with the static reference prototype and subtracted
    from the observed activation.  The probe token is never committed to the
    persistent reconstruction state.
    """

    _require_metric(metric)
    if observations.ndim != 3 or observations.shape[0] <= 0 or observations.shape[1] <= 1:
        raise PrototypeError("observations must be [records, sequence, hidden] with sequence > 1")
    if int(observations.shape[2]) != prototypes.hidden_size:
        raise PrototypeError("observation hidden size differs from prototypes")
    if int(reference_token) < 0 or int(reference_token) >= prototypes.vocab_size:
        raise PrototypeError("reference token is outside the prototype vocabulary")
    if int(bos_token_id) != BOS_TOKEN_ID:
        raise PrototypeError("declared BOS token changed")
    if device is None:
        try:
            device = next(public_prefix.parameters()).device
        except StopIteration as exc:
            raise PrototypeError("public prefix has no parameters; pass device") from exc

    records, sequence, hidden = map(int, observations.shape)
    predictions = torch.full((records, sequence), -1, dtype=torch.long)
    scores = torch.full((records, sequence), float("nan"), dtype=torch.float32)
    margins = torch.full((records, sequence), float("nan"), dtype=torch.float32)
    offsets = torch.zeros((records, sequence, hidden), dtype=torch.float32)
    reference = prototypes.prototypes[int(reference_token)].float().to(device)
    reference_evaluations = 0
    persistent_cache_commits = 0
    probe_cache_commits = 0

    for record_index in range(records):
        cache = public_prefix.new_cache()
        bos = torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device)
        public_prefix.run_cached(bos, cache, 0)
        persistent_cache_commits += 1
        predictions[record_index, 0] = BOS_TOKEN_ID
        for position in range(1, sequence):
            try:
                probe_cache = copy.deepcopy(cache)
            except Exception as exc:  # pragma: no cover - backend-specific failure
                raise PrototypeError("public prefix cache cannot be copied for reference probing") from exc
            probe = torch.tensor([[int(reference_token)]], dtype=torch.long, device=device)
            probe_hidden = _last_hidden(public_prefix.run_cached(probe, probe_cache, position))
            probe_cache_commits += 1
            reference_evaluations += 1
            offset = probe_hidden[0].float() - reference
            offsets[record_index, position] = offset.cpu()
            query = observations[record_index, position].float().to(device) - offset
            nearest = prototypes.nearest(
                query.view(1, -1).cpu(),
                metric=metric,
                query_chunk_size=query_chunk_size,
                prototype_chunk_size=prototype_chunk_size,
            )
            token = int(nearest.predictions[0])
            predictions[record_index, position] = token
            scores[record_index, position] = nearest.scores[0]
            margins[record_index, position] = nearest.margins[0]
            selected = torch.tensor([[token]], dtype=torch.long, device=device)
            public_prefix.run_cached(selected, cache, position)
            persistent_cache_commits += 1
        del cache
    return CorrectionResult(
        predictions=predictions,
        scores=scores,
        margins=margins,
        offsets=offsets,
        reference_evaluations=reference_evaluations,
        persistent_cache_commits=persistent_cache_commits,
        probe_cache_commits=probe_cache_commits,
    )

