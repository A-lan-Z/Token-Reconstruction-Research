"""Geometry pipeline for frozen TRR-P12 B1 stage artifacts.

The input contract is limited to four runner-produced artifacts per stage:
int64 ``expanded_fixed`` predictions, FP32 projected hidden rows, and the
runner's compact geometry JSON.  A public FP32 embedding table is the only
other input.  Stage 0 and 64 are streamed first; the 128/256 projected files
are opened only after the frozen 128/256 direction forecast JSON is written.
Full-vocabulary scores are ephemeral and compact winner/competitor rows are
retained for the signed boundary report.  The default score backend is CPU;
an optional ``cuda:0`` backend moves only the public embedding table and one
projected row at a time to the GPU, then returns scores to CPU before any
boundary analysis.  Exact full-vocabulary argmax alignment remains a hard
failure in either mode.  No source text, truth, target prefix, or target
weights are accepted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import sys
from typing import Any, Iterator, Mapping

import torch
from safetensors import safe_open

from scripts.trr_p12.boundary import BoundaryRows, geometric_boundary_metrics, analyze_score_pair
from scripts.trr_p12.predict import DirectionForecast, direction_forecast


TASK_ID = "TRR-P12"
PIPELINE_SCHEMA = "token-reconstruction.trr-p12-geometry-pipeline.v1"
FORECAST_SCHEMA = "token-reconstruction.trr-p12-direction-forecast-artifact.v1"
PRIVATE_SCHEMA = "token-reconstruction.trr-p12-private-boundary-rows.v1"
PUBLIC_SCHEMA = "token-reconstruction.trr-p12-public-geometry-aggregates.v1"
STAGE_GEOMETRY_SCHEMA = "token-reconstruction.trr-p12-b1-compact-geometry.v1"
PREDICTION_KEY = "expanded_fixed"
PROJECTED_KEY = "projected_hidden"
# B1 stores base.s as FP32.  Keep the exact package scalar operation for
# score margins and signed-distance reconstruction; the positive scale does
# not change argmax, but its rounding matters to reported margins.
B1_BASE_S = 4.275415420532227
DEFAULT_LOGIT_SCALE = float(torch.exp(torch.tensor(B1_BASE_S, dtype=torch.float32)).item())
VOCABULARY_SIZE = 128256
HIDDEN_SIZE = 2048
STORED_SEQUENCE_TOKENS = 128
BOS_TOKEN_ID = 128000
DEFAULT_MIN_FREE_GPU_BYTES = 8 * 1024**3


class GeometryPipelineError(RuntimeError):
    """Raised when a frozen stage geometry or boundary result is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _regular_path(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise GeometryPipelineError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise GeometryPipelineError(f"{label} must be a regular file: {resolved}")
    return resolved


def file_binding(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_path(path, label=label)
    return {"label": label, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def create_only(path: Path, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise GeometryPipelineError(f"{label} is create-only: {raw}")
    path = raw.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_path(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeometryPipelineError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise GeometryPipelineError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True)
class PipelineConfig:
    """Fixed decoder geometry; small values are used only by CPU unit tests."""

    vocabulary_size: int = VOCABULARY_SIZE
    hidden_size: int = HIDDEN_SIZE
    stored_sequence_tokens: int = STORED_SEQUENCE_TOKENS
    bos_token_id: int = BOS_TOKEN_ID
    logit_scale: float = DEFAULT_LOGIT_SCALE

    @property
    def scored_positions(self) -> int:
        return self.stored_sequence_tokens - 1


@dataclass(frozen=True)
class StageInput:
    stage: int
    predictions_path: Path
    projected_path: Path
    geometry_path: Path
    geometry: dict[str, Any]
    record_order: tuple[str, ...]

    @classmethod
    def from_files(
        cls,
        stage: int,
        predictions_path: Path,
        projected_path: Path,
        geometry_path: Path,
        *,
        config: PipelineConfig,
    ) -> "StageInput":
        geometry = _json_object(geometry_path, label=f"stage {stage} geometry")
        if geometry.get("schema") != STAGE_GEOMETRY_SCHEMA:
            raise GeometryPipelineError(f"stage {stage} geometry schema differs")
        count = geometry.get("record_count")
        order = geometry.get("record_order")
        pred_shape = geometry.get("prediction_output_geometry")
        proj_shape = geometry.get("projected_output_geometry")
        expected_pred = [int(count), config.stored_sequence_tokens] if isinstance(count, int) else None
        expected_proj = [int(count), config.scored_positions, config.hidden_size] if isinstance(count, int) else None
        if not isinstance(count, int) or count <= 0 or not isinstance(order, list) or len(order) != count:
            raise GeometryPipelineError(f"stage {stage} geometry record order is malformed")
        if len(set(str(item) for item in order)) != count:
            raise GeometryPipelineError(f"stage {stage} geometry record order is not unique")
        if pred_shape != expected_pred or proj_shape != expected_proj:
            raise GeometryPipelineError(f"stage {stage} geometry output shapes differ from config")
        pred = _regular_path(predictions_path, label=f"stage {stage} predictions")
        proj = _regular_path(projected_path, label=f"stage {stage} projected features")
        geom = _regular_path(geometry_path, label=f"stage {stage} geometry")
        return cls(stage, pred, proj, geom, geometry, tuple(str(item) for item in order))

    def bindings(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "predictions": file_binding(self.predictions_path, label=f"stage {self.stage} predictions"),
            "projected_hidden": file_binding(self.projected_path, label=f"stage {self.stage} projected features"),
            "geometry": file_binding(self.geometry_path, label=f"stage {self.stage} compact geometry"),
            "record_count": len(self.record_order),
            "record_order_sha256": hashlib.sha256("\n".join(self.record_order).encode("utf-8")).hexdigest(),
        }

    @contextmanager
    def records(self, *, config: PipelineConfig) -> Iterator[Iterator[tuple[int, str, torch.Tensor, torch.Tensor]]]:
        """Yield an iterator that reads one projected/prediction row at a time."""

        try:
            with safe_open(str(self.predictions_path), framework="pt", device="cpu") as pred_handle:
                with safe_open(str(self.projected_path), framework="pt", device="cpu") as projected_handle:
                    if set(pred_handle.keys()) != {PREDICTION_KEY}:
                        raise GeometryPipelineError(f"stage {self.stage} prediction keys differ")
                    if set(projected_handle.keys()) != {PROJECTED_KEY}:
                        raise GeometryPipelineError(f"stage {self.stage} projected keys differ")
                    pred_slice = pred_handle.get_slice(PREDICTION_KEY)
                    projected_slice = projected_handle.get_slice(PROJECTED_KEY)
                    pred_shape = tuple(int(item) for item in pred_slice.get_shape())
                    projected_shape = tuple(int(item) for item in projected_slice.get_shape())
                    expected_pred = (len(self.record_order), config.stored_sequence_tokens)
                    expected_projected = (len(self.record_order), config.scored_positions, config.hidden_size)
                    if pred_shape != expected_pred or projected_shape != expected_projected:
                        raise GeometryPipelineError(f"stage {self.stage} tensor geometry differs")
                    yield self._iter_slices(pred_slice, projected_slice, config=config)
        except GeometryPipelineError:
            raise
        except Exception as exc:
            raise GeometryPipelineError(f"stage {self.stage} tensors could not be streamed") from exc

    def _iter_slices(self, pred_slice: Any, projected_slice: Any, *, config: PipelineConfig) -> Iterator[tuple[int, str, torch.Tensor, torch.Tensor]]:
        for index, slot in enumerate(self.record_order):
            prediction = pred_slice[index].to(device="cpu", dtype=torch.long).contiguous()
            projected = projected_slice[index].to(device="cpu", dtype=torch.float32).contiguous()
            if prediction.numel() != config.stored_sequence_tokens or projected.shape != (config.scored_positions, config.hidden_size):
                raise GeometryPipelineError(f"stage {self.stage} record {index} row geometry changed")
            if int(prediction[0].item()) != config.bos_token_id or bool(prediction.lt(0).any().item()) or bool(prediction.ge(config.vocabulary_size).any().item()):
                raise GeometryPipelineError(f"stage {self.stage} record {index} prediction IDs violate the frozen contract")
            if not bool(torch.isfinite(projected).all().item()):
                raise GeometryPipelineError(f"stage {self.stage} record {index} projected rows are non-finite")
            yield index, slot, prediction, projected


def load_embedding(path: Path, *, config: PipelineConfig) -> torch.Tensor:
    path = _regular_path(path, label="public embedding table")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"embeddings"}:
                raise GeometryPipelineError("public embedding key differs")
            embedding = handle.get_tensor("embeddings").to(device="cpu", dtype=torch.float32).contiguous()
    except GeometryPipelineError:
        raise
    except Exception as exc:
        raise GeometryPipelineError("public embedding table could not be loaded") from exc
    if tuple(embedding.shape) != (config.vocabulary_size, config.hidden_size) or not bool(torch.isfinite(embedding).all().item()):
        raise GeometryPipelineError("public embedding geometry or finiteness differs")
    return embedding


def _initialize_cuda(device: torch.device) -> None:
    """Initialize CUDA before allocator or free-memory queries."""

    if device.type != "cuda":
        return
    if str(device) != "cuda:0":
        raise GeometryPipelineError("only cuda:0 is registered for the optional score backend")
    if not torch.cuda.is_available():
        raise GeometryPipelineError("CUDA score backend requested but CUDA is unavailable")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.cuda.synchronize(device)


def _score_backend_resources(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(
            {
                "cuda_free_bytes": int(free),
                "cuda_total_bytes": int(total),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


class ScoreBackend:
    """Compute full-vocabulary scores and return compact CPU rows.

    ``embedding`` is always retained as a CPU FP32 tensor.  For ``cuda:0``
    the table is copied once to the device and each projected feature row is
    copied independently; the resulting score matrix is immediately copied
    back to CPU.  Callers therefore retain no device panel and continue to
    use the exact CPU boundary/predictor code.  The full-vocabulary argmax
    check is deliberately performed after this copy, so CUDA numerical drift
    cannot silently pass into production diagnostics.
    """

    def __init__(
        self,
        embedding: torch.Tensor,
        *,
        config: PipelineConfig,
        device: str | torch.device = "cpu",
        min_free_gpu_bytes: int = DEFAULT_MIN_FREE_GPU_BYTES,
    ) -> None:
        table = torch.as_tensor(embedding).detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(table.shape) != (config.vocabulary_size, config.hidden_size):
            raise GeometryPipelineError("public embedding geometry differs from score backend config")
        if not bool(torch.isfinite(table).all().item()):
            raise GeometryPipelineError("public embedding contains non-finite values")
        try:
            requested = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise GeometryPipelineError(f"invalid score backend device: {device}") from exc
        if requested.type not in {"cpu", "cuda"}:
            raise GeometryPipelineError("score backend device must be cpu or cuda:0")
        self.config = config
        self.device = requested
        self.embedding_cpu = table
        self.embedding_device: torch.Tensor | None = None
        if requested.type == "cuda":
            _initialize_cuda(requested)
        self.resource_before = _score_backend_resources(requested)
        if requested.type == "cuda":
            free, _total = torch.cuda.mem_get_info(requested)
            if int(free) < int(min_free_gpu_bytes):
                raise GeometryPipelineError(
                    f"CUDA free-memory guard failed before embedding upload: {int(free)} < {int(min_free_gpu_bytes)}"
                )
            torch.cuda.reset_peak_memory_stats(requested)
            try:
                self.embedding_device = table.to(device=requested, dtype=torch.float32).contiguous()
                torch.cuda.synchronize(requested)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                raise GeometryPipelineError("public embedding upload to CUDA failed") from exc
        self.resource_after_upload = _score_backend_resources(requested)

    def metadata(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "compute_dtype": "torch.float32",
            "embedding_shape": list(self.embedding_cpu.shape),
            "embedding_on_device": self.embedding_device is not None,
            "recordwise_projected_upload": self.device.type == "cuda",
            "scores_returned_to": "cpu",
            "full_vocabulary_argmax_checked_after_device_copy": True,
            "resource_before": self.resource_before,
            "resource_after_embedding_upload": self.resource_after_upload,
            "resource_after": _score_backend_resources(self.device),
        }

    def scores(self, projected: torch.Tensor) -> torch.Tensor:
        compact = torch.as_tensor(projected).detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(compact.shape) != (self.config.scored_positions, self.config.hidden_size):
            raise GeometryPipelineError("projected feature geometry differs")
        if not bool(torch.isfinite(compact).all().item()):
            raise GeometryPipelineError("projected features are non-finite")
        with torch.inference_mode():
            if self.device.type == "cuda":
                assert self.embedding_device is not None
                device_rows = compact.to(device=self.device, dtype=torch.float32)
                scores = torch.matmul(device_rows, self.embedding_device.transpose(0, 1))
                scores = scores.mul(float(self.config.logit_scale)).to(device="cpu", dtype=torch.float32).contiguous()
                torch.cuda.synchronize(self.device)
            else:
                scores = torch.matmul(compact, self.embedding_cpu.transpose(0, 1))
                scores = scores.mul(float(self.config.logit_scale)).to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(scores).all().item()):
            raise GeometryPipelineError("computed full-vocabulary scores are non-finite")
        return scores


def scores_from_projected(projected: torch.Tensor, embedding: torch.Tensor, *, config: PipelineConfig) -> torch.Tensor:
    """Backward-compatible CPU score helper used by synthetic callers."""

    return ScoreBackend(embedding, config=config, device="cpu").scores(projected)


def verify_prediction_alignment(scores: torch.Tensor, prediction: torch.Tensor, *, config: PipelineConfig, stage: int, record_index: int) -> None:
    prediction = torch.as_tensor(prediction).to(dtype=torch.long, device="cpu")
    if tuple(prediction.shape) != (config.stored_sequence_tokens,) or int(prediction[0].item()) != config.bos_token_id:
        raise GeometryPipelineError(f"stage {stage} record {record_index} prediction shape/BOS differs")
    computed = torch.argmax(scores, dim=-1).to(dtype=torch.long, device="cpu")
    if not torch.equal(computed, prediction[1:]):
        mismatch = int(torch.nonzero(computed.ne(prediction[1:]), as_tuple=False)[0].item())
        raise GeometryPipelineError(f"stage {stage} record {record_index} full-vocabulary argmax mismatch at post-BOS position {mismatch + 1}")


def _forecast_record(forecast: DirectionForecast, *, record_index: int, slot: str) -> dict[str, Any]:
    compact = forecast.compact()
    compact["record_index"] = record_index
    compact["slot"] = slot
    return compact


def _forecast_aggregate(records: list[DirectionForecast]) -> dict[str, Any]:
    if not records:
        raise GeometryPipelineError("forecast has no records")
    result: dict[str, Any] = {"rows": int(sum(int(item.valid.sum().item()) for item in records)), "stages": {}}
    for stage in (128, 256):
        margins = torch.cat([item.forecast_margin[stage].reshape(-1)[item.valid.reshape(-1)] for item in records])
        risk = torch.cat([item.risk_by_stage[stage].reshape(-1)[item.valid.reshape(-1)] for item in records])
        strict = margins.lt(0.0)
        ties = margins.eq(0.0)
        result["stages"][str(stage)] = {
            "rows": int(margins.numel()),
            "strict_predicted_changes": int(strict.sum().item()),
            "predicted_ties": int(ties.sum().item()),
            "strict_change_rate": float(strict.float().mean().item()),
            "tie_rate": float(ties.float().mean().item()),
            "risk_mean": float(risk.mean().item()),
            "risk_max": float(risk.max().item()),
        }
    return result


def _geometry_compact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    tensor_names = (
        "signed_distance_before",
        "signed_distance_after",
        "signed_distance_displacement",
        "boundary_normal_norm",
        "projected_movement_norm",
    )
    result = {name: [float(item) for item in metrics[name].reshape(-1).tolist()] for name in tensor_names}
    result["strict_geometric_crossing"] = [bool(item) for item in metrics["strict_geometric_crossing"].reshape(-1).tolist()]
    result["score_margin_reconstruction_error_max"] = float(metrics["score_margin_reconstruction_error_max"])
    return result


def _boundary_aggregate(entries: list[tuple[BoundaryRows, Mapping[str, Any]]]) -> dict[str, Any]:
    if not entries:
        raise GeometryPipelineError("boundary stage has no records")
    valid = torch.cat([rows.valid.reshape(-1) for rows, _ in entries])
    def cat(name: str) -> torch.Tensor:
        return torch.cat([getattr(rows, name).reshape(-1) for rows, _ in entries])[valid]
    def count(name: str) -> int:
        return int(cat(name).to(dtype=torch.bool).sum().item())
    def mean(name: str) -> float:
        return float(cat(name).float().mean().item())
    geometric_crossings = torch.cat([metrics["strict_geometric_crossing"].reshape(-1) for _, metrics in entries])[valid]
    distances = torch.cat([metrics["signed_distance_displacement"].reshape(-1) for _, metrics in entries])[valid]
    return {
        "rows": int(valid.sum().item()),
        "argmax_changed": count("argmax_changed"),
        "strict_crossings": count("strict_crossing"),
        "tie_transitions": count("tie_transition"),
        "transition_margin_before_mean": mean("transition_margin_before"),
        "transition_margin_after_mean": mean("transition_margin_after"),
        "signed_displacement_mean": mean("signed_displacement"),
        "unsigned_pair_movement_mean": mean("unsigned_pair_movement"),
        "strict_geometric_crossings": int(geometric_crossings.sum().item()),
        "signed_distance_displacement_mean": float(distances.float().mean().item()),
        "score_margin_reconstruction_error_max": max(float(metrics["score_margin_reconstruction_error_max"]) for _, metrics in entries),
        "interpretation": "retrospective full-vocabulary boundary diagnostic; unsigned movement is not a crossing proof",
    }


def _relative_output_bindings(paths: Mapping[str, Path]) -> dict[str, Any]:
    # The receipt is create-only and cannot contain its own digest.
    return {
        name: file_binding(path, label=f"geometry pipeline {name} output")
        for name, path in paths.items()
        if name != "receipt"
    }


def run_pipeline(
    stages: Mapping[int, StageInput],
    embedding_path: Path,
    *,
    forecast_output: Path,
    private_output: Path,
    public_output: Path,
    receipt_output: Path,
    config: PipelineConfig = PipelineConfig(),
    score_device: str | torch.device = "cpu",
    min_free_gpu_bytes: int = DEFAULT_MIN_FREE_GPU_BYTES,
) -> dict[str, Any]:
    """Run forecast freeze then actual 0→64/128/256 boundary diagnostics."""

    required = {0, 64, 128, 256}
    if set(stages) != required:
        raise GeometryPipelineError(f"stage set must be exactly {sorted(required)}")
    reference_order = stages[0].record_order
    for stage in sorted(required):
        if stages[stage].record_order != reference_order:
            raise GeometryPipelineError(f"stage {stage} record order differs from stage 0")
    outputs = {
        "forecast": create_only(forecast_output, label="forecast output"),
        "private_boundary": create_only(private_output, label="private boundary output"),
        "public_aggregates": create_only(public_output, label="public aggregate output"),
        "receipt": create_only(receipt_output, label="pipeline receipt"),
    }
    started_utc = utc_now()
    started = time.perf_counter()
    embedding = load_embedding(embedding_path, config=config)
    embedding_path = Path(embedding_path).expanduser().resolve()
    embedding_binding = file_binding(embedding_path, label="public embedding table")
    scorer = ScoreBackend(
        embedding,
        config=config,
        device=score_device,
        min_free_gpu_bytes=min_free_gpu_bytes,
    )

    stage_bindings: dict[str, Any] = {}
    stage0_64_started = time.perf_counter()
    forecast_rows: list[dict[str, Any]] = []
    forecast_objects: list[DirectionForecast] = []
    iter0 = stages[0].records(config=config)
    iter64 = stages[64].records(config=config)
    with iter0 as records0, iter64 as records64:
        for left, right in zip(records0, records64):
            index0, slot0, prediction0, projected0 = left
            index64, slot64, prediction64, projected64 = right
            if index0 != index64 or slot0 != slot64:
                raise GeometryPipelineError("stage 0 and 64 record orders differ")
            score0 = scorer.scores(projected0)
            score64 = scorer.scores(projected64)
            verify_prediction_alignment(score0, prediction0, config=config, stage=0, record_index=index0)
            verify_prediction_alignment(score64, prediction64, config=config, stage=64, record_index=index64)
            forecast = direction_forecast(score0, score64)
            forecast_objects.append(forecast)
            forecast_rows.append(_forecast_record(forecast, record_index=index0, slot=slot0))
    if len(forecast_rows) != len(stages[0].record_order):
        raise GeometryPipelineError("stage 0/64 record counts differ")
    early_forecast_seconds = time.perf_counter() - stage0_64_started
    stage_bindings["0"] = stages[0].bindings()
    stage_bindings["64"] = stages[64].bindings()
    early_bindings = {"0": stage_bindings["0"], "64": stage_bindings["64"]}
    forecast_payload = {
        "schema": FORECAST_SCHEMA,
        "task_id": TASK_ID,
        "status": "FORECAST_FROZEN_BEFORE_LATER_FEATURE_LOAD",
        "created_utc": utc_now(),
        "forecast_rule": "u_t = u_0 + (t/64) * (u_64-u_0), full-vocabulary margin of stage-0 winner against all other IDs",
        "tie_rule": "torch.argmax; margin == 0 is reported separately",
        "later_stages_not_loaded": [128, 256],
        "embedding": embedding_binding,
        "early_stage_inputs": {"0": stage_bindings["0"], "64": stage_bindings["64"]},
        "aggregate": _forecast_aggregate(forecast_objects),
        "rows": forecast_rows,
        "truth_opened": False,
        "target_weights_loaded": False,
        "source_text_loaded": False,
    }
    outputs["forecast"].write_text(json.dumps(forecast_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    forecast_written_utc = utc_now()
    later_load_started_utc = utc_now()

    pair_entries: dict[str, list[tuple[BoundaryRows, Mapping[str, Any]]]] = {"64": [], "128": [], "256": []}
    private_rows: dict[str, list[dict[str, Any]]] = {"64": [], "128": [], "256": []}
    stage0 = stages[0]
    iter0_actual = stage0.records(config=config)
    iter64_actual = stages[64].records(config=config)
    iter128 = stages[128].records(config=config)
    iter256 = stages[256].records(config=config)
    with iter0_actual as records0, iter64_actual as records64, iter128 as records128, iter256 as records256:
        for left, early, middle, right in zip(records0, records64, records128, records256):
            index0, slot0, prediction0, projected0 = left
            index64, slot64, prediction64, projected64 = early
            index128, slot128, prediction128, projected128 = middle
            index256, slot256, prediction256, projected256 = right
            if not (index0 == index64 == index128 == index256 and slot0 == slot64 == slot128 == slot256):
                raise GeometryPipelineError("stage 0/64/128/256 record orders differ")
            score0 = scorer.scores(projected0)
            score64 = scorer.scores(projected64)
            score128 = scorer.scores(projected128)
            score256 = scorer.scores(projected256)
            verify_prediction_alignment(score0, prediction0, config=config, stage=0, record_index=index0)
            verify_prediction_alignment(score64, prediction64, config=config, stage=64, record_index=index64)
            verify_prediction_alignment(score128, prediction128, config=config, stage=128, record_index=index128)
            verify_prediction_alignment(score256, prediction256, config=config, stage=256, record_index=index256)
            for stage, score, prediction, projected in (
                (64, score64, prediction64, projected64),
                (128, score128, prediction128, projected128),
                (256, score256, prediction256, projected256),
            ):
                rows = analyze_score_pair(
                    score0,
                    score,
                    baseline_predictions=prediction0[1:],
                    updated_predictions=prediction[1:],
                )
                metrics = geometric_boundary_metrics(rows, projected0, projected, embedding, logit_scale=config.logit_scale)
                pair_key = str(stage)
                pair_entries[pair_key].append((rows, metrics))
                private_rows[pair_key].append({
                    "record_index": index0,
                    "slot": slot0,
                    "boundary": rows.compact(include_scores=True),
                    "geometry": _geometry_compact(metrics),
                })
    if any(len(pair_entries[key]) != len(stages[0].record_order) for key in pair_entries):
        raise GeometryPipelineError("actual stage record counts differ")
    if stages[0].bindings() != early_bindings["0"]:
        raise GeometryPipelineError("stage 0 input changed between forecast freeze and boundary pass")
    if stages[64].bindings() != early_bindings["64"]:
        raise GeometryPipelineError("stage 64 input changed between forecast freeze and boundary pass")
    for stage in (128, 256):
        stage_bindings[str(stage)] = stages[stage].bindings()
    actual_finished_utc = utc_now()

    public_payload = {
        "schema": PUBLIC_SCHEMA,
        "task_id": TASK_ID,
        "status": "ACTUAL_BOUNDARY_DIAGNOSTICS_COMPLETE",
        "created_utc": utc_now(),
        "forecast": forecast_payload["aggregate"],
        "actual_pairs": {stage: _boundary_aggregate(pair_entries[stage]) for stage in ("64", "128", "256")},
        "later_stage_load_order": {"forecast_written_utc": forecast_written_utc, "later_features_opened_after": forecast_written_utc},
        "truth_opened": False,
        "target_weights_loaded": False,
        "source_text_loaded": False,
    }
    outputs["public_aggregates"].write_text(json.dumps(public_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    private_payload = {
        "schema": PRIVATE_SCHEMA,
        "task_id": TASK_ID,
        "status": "ACTUAL_BOUNDARY_COMPACT_ROWS_COMPLETE",
        "created_utc": utc_now(),
        "stage_pairs": {"64": "0_to_64", "128": "0_to_128", "256": "0_to_256"},
        "rows": private_rows,
        "truth_opened": False,
        "target_weights_loaded": False,
        "source_text_loaded": False,
    }
    outputs["private_boundary"].write_text(json.dumps(private_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = {
        "schema": PIPELINE_SCHEMA,
        "task_id": TASK_ID,
        "status": "FORECAST_AND_BOUNDARY_COMPLETE",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
        "package_math": "score = logit_scale * projected_hidden @ public_E.T; torch.argmax over all vocabulary IDs",
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "hidden_size": config.hidden_size,
            "stored_sequence_tokens": config.stored_sequence_tokens,
            "scored_positions": config.scored_positions,
            "bos_token_id": config.bos_token_id,
            "base_s": B1_BASE_S,
            "logit_scale": config.logit_scale,
            "logit_scale_source": "torch.float32 exp(base.s)",
        },
        "embedding": embedding_binding,
        "score_backend": scorer.metadata(),
        "stage_inputs": stage_bindings,
        "code": {
            "pipeline": file_binding(Path(__file__), label="geometry pipeline code"),
            "boundary": file_binding(Path(__file__).with_name("boundary.py"), label="boundary code"),
            "predict": file_binding(Path(__file__).with_name("predict.py"), label="predictor code"),
        },
        "outputs": _relative_output_bindings(outputs),
        "phases": {
            "early_forecast_seconds": early_forecast_seconds,
            "forecast_written_utc": forecast_written_utc,
            "later_features_opened_after_forecast": True,
            "later_features_opened_utc": later_load_started_utc,
            "actual_boundary_finished_utc": actual_finished_utc,
        },
        "verification": {
            "full_vocabulary_argmax_checked": True,
            "prediction_alignment_stages": [0, 64, 128, 256],
            "boundary_pairs": ["0_to_64", "0_to_128", "0_to_256"],
            "raw_logits_retained": False,
            "private_rows_public_aggregates_separate": True,
        },
        "access_boundary": {
            "truth_opened": False,
            "source_text_loaded": False,
            "token_ids_loaded": False,
            "target_weights_loaded": False,
            "target_prefix_queried": False,
        },
    }
    outputs["receipt"].write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt_path"] = str(outputs["receipt"])
    receipt["receipt_sha256"] = sha256_file(outputs["receipt"])
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in (0, 64, 128, 256):
        parser.add_argument(f"--stage{stage}-predictions", type=Path, required=True)
        parser.add_argument(f"--stage{stage}-projected", type=Path, required=True)
        parser.add_argument(f"--stage{stage}-geometry", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--logit-scale", type=float, default=DEFAULT_LOGIT_SCALE)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu", help="score matmul device; scores return to CPU")
    parser.add_argument("--min-free-gpu-gib", type=float, default=8.0)
    parser.add_argument("--forecast-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = PipelineConfig(logit_scale=float(args.logit_scale))
        stages = {
            stage: StageInput.from_files(
                stage,
                getattr(args, f"stage{stage}_predictions"),
                getattr(args, f"stage{stage}_projected"),
                getattr(args, f"stage{stage}_geometry"),
                config=config,
            )
            for stage in (0, 64, 128, 256)
        }
        receipt = run_pipeline(
            stages,
            args.embedding,
            forecast_output=args.forecast_output,
            private_output=args.private_output,
            public_output=args.public_output,
            receipt_output=args.receipt_output,
            config=config,
            score_device=args.device,
            min_free_gpu_bytes=int(float(args.min_free_gpu_gib) * 1024**3),
        )
    except (GeometryPipelineError, OSError, RuntimeError, ValueError) as exc:
        print(f"TRR-P12 geometry pipeline error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": receipt["status"], "receipt_path": receipt["receipt_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
