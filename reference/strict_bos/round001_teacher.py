"""Truthless public-P0 teacher and rank-8 A1 update for Round 001.

This module deliberately defines its own passive row type, public Llama prefix,
strict-BOS cascade, and adapter training loop.  It has no target-token field and
does not import either the trace reader or any historical training-pair helper.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import gc
import hashlib
import inspect
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


INVALID_TOKEN_ID = -1
BOS_TOKEN_ID = 128000
TIER_K = (32, 128, 512)
A1_FAST_PATH_MIN_CONFIDENCE = 0.999
NORMALIZED_WINNER_MIN = 2.0

ROUTE_PADDING = 0
ROUTE_BOS = 1
ROUTE_A1 = 2
ROUTE_A2_K32 = 3
ROUTE_A2_K128 = 4
ROUTE_A2_K512 = 5
ROUTE_ABSTAIN_K512 = 6
ROUTE_ABSTAINED_SUFFIX = 7


class Round001TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class PassiveRow:
    """One intercepted row.  No trusted-token member exists by construction."""

    row_index: int
    activation: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    def validate(self) -> None:
        if self.activation.ndim != 2 or self.activation.shape[1] != 2048:
            raise Round001TeacherError("passive row activation must be [positions,2048]")
        if self.attention_mask.shape != (self.activation.shape[0],):
            raise Round001TeacherError("passive row mask shape changed")
        if self.position_ids.shape != self.attention_mask.shape:
            raise Round001TeacherError("passive row position shape changed")
        mask = self.attention_mask.to(torch.bool)
        if not mask.any().item():
            raise Round001TeacherError("passive row is empty")
        if (self.attention_mask[1:] > self.attention_mask[:-1]).any().item():
            raise Round001TeacherError("passive row is not right padded")
        expected = self.attention_mask.cumsum(0).sub(1).clamp_min(0)
        if not torch.equal(self.position_ids.to(torch.long), expected.to(torch.long)):
            raise Round001TeacherError("passive row positions disagree with its mask")
        if not torch.isfinite(self.activation).all().item():
            raise Round001TeacherError("passive row activation is non-finite")


@dataclass(frozen=True)
class TeacherRowResult:
    row_index: int
    token_ids: torch.Tensor
    route_codes: torch.Tensor
    candidate_simulations: int


@dataclass(frozen=True)
class AdapterSpec:
    rank: int
    learning_rate: float
    steps: int
    batch_size: int
    correction_weight: float
    gradient_clip: float
    weight_decay: float

    def validate(self) -> None:
        if self.rank != 8:
            raise Round001TeacherError("Round 001 adapter rank must be 8")
        if self.learning_rate != 1e-3 or self.steps != 5 or self.batch_size != 256:
            raise Round001TeacherError("Round 001 optimizer schedule changed")
        if self.correction_weight != 8.0 or self.gradient_clip != 1.0:
            raise Round001TeacherError("Round 001 weighting or clipping changed")
        if self.weight_decay != 0.0:
            raise Round001TeacherError("Round 001 weight decay must remain zero")


class FrozenAffineLens(nn.Module):
    """Pinned public A1 affine lens, with no auxiliary state or data access."""

    def __init__(self, width: int) -> None:
        super().__init__()
        if width != 2048:
            raise Round001TeacherError("Round 001 lens width changed")
        self.W = nn.Parameter(torch.eye(width, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(width, dtype=torch.float32))
        self.s = nn.Parameter(torch.tensor(3.0, dtype=torch.float32))

    def projected(self, activation: torch.Tensor) -> torch.Tensor:
        value = activation.float()
        return value @ self.W.float().T + self.b.float()

    def forward(
        self, activation: torch.Tensor, normalized_embeddings: torch.Tensor
    ) -> torch.Tensor:
        projected = F.normalize(self.projected(activation), dim=-1)
        logits = projected.to(normalized_embeddings.dtype) @ normalized_embeddings.T
        return logits.float() * self.s.float().exp()


class ResidualA1Adapter(nn.Module):
    """Frozen affine A1 plus a zero-initialized rank-8 residual map."""

    def __init__(self, frozen_lens: FrozenAffineLens, rank: int, *, seed: int) -> None:
        super().__init__()
        if rank != 8:
            raise Round001TeacherError("Round 001 residual rank must be 8")
        width = int(frozen_lens.W.shape[0])
        self.register_buffer("base_W", frozen_lens.W.detach().float().clone())
        self.register_buffer("base_b", frozen_lens.b.detach().float().clone())
        self.register_buffer("base_s", frozen_lens.s.detach().float().clone())
        self.down = nn.Linear(width, rank, bias=False, dtype=torch.float32)
        self.up = nn.Linear(rank, width, bias=False, dtype=torch.float32)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            initial = torch.randn(
                self.down.weight.shape, generator=generator, dtype=torch.float32
            ) / math.sqrt(width)
            self.down.weight.copy_(initial)
            self.up.weight.zero_()

    def projected(self, activation: torch.Tensor) -> torch.Tensor:
        value = activation.float()
        return value @ self.base_W.T + self.base_b + self.up(self.down(value))

    def forward(
        self, activation: torch.Tensor, normalized_embeddings: torch.Tensor
    ) -> torch.Tensor:
        projected = F.normalize(self.projected(activation), dim=-1)
        logits = projected.to(normalized_embeddings.dtype) @ normalized_embeddings.T
        return logits.float() * self.base_s.exp()


class PublicP0Precut(nn.Module):
    """Public Llama embedding and decoder layers 0--3 with checked cache length."""

    def __init__(self, full_model: nn.Module, prefix_layers: Sequence[int]) -> None:
        super().__init__()
        if tuple(prefix_layers) != (0, 1, 2, 3):
            raise Round001TeacherError("public P0 prefix must be decoder layers 0--3")
        inner = full_model.model
        self.embed_tokens = inner.embed_tokens
        self.layers = nn.ModuleList([inner.layers[index] for index in prefix_layers])
        self.rotary_emb = inner.rotary_emb
        self.config = full_model.config
        self.cut = len(prefix_layers)
        parameters = inspect.signature(self.layers[0].forward).parameters
        if "past_key_values" in parameters:
            self.cache_keyword = "past_key_values"
        elif "past_key_value" in parameters:
            self.cache_keyword = "past_key_value"
        else:
            raise Round001TeacherError("public Llama prefix exposes no supported cache API")
        for layer in self.layers[1:]:
            if self.cache_keyword not in inspect.signature(layer.forward).parameters:
                raise Round001TeacherError("public prefix layers disagree on cache API")
        self.checked_cache_transitions = 0

    def new_cache(self) -> Any:
        from transformers.cache_utils import DynamicCache

        try:
            return DynamicCache(config=self.config)
        except TypeError:  # pragma: no cover - transformers 4.x
            return DynamicCache()

    def cache_layer_lengths(self, cache: Any) -> tuple[int, ...]:
        getter = getattr(cache, "get_seq_length", None)
        if not callable(getter):
            raise Round001TeacherError("public P0 cache lacks get_seq_length")
        values: list[int] = []
        for layer_index in range(self.cut):
            try:
                value = getter(layer_index)
            except TypeError:  # pragma: no cover - legacy global-length cache
                value = values[0] if layer_index else getter()
            except IndexError:
                value = 0
            values.append(int(value))
        return tuple(values)

    def require_cache_length(self, cache: Any, expected: int, where: str) -> None:
        observed = self.cache_layer_lengths(cache)
        if any(value != expected for value in observed):
            raise Round001TeacherError(
                f"public P0 cache length failed {where}: {observed} != {expected}"
            )

    @staticmethod
    def _causal_mask(
        hidden: torch.Tensor, *, start_pos: int, total_tokens: int
    ) -> torch.Tensor | None:
        query_tokens = int(hidden.shape[1])
        if query_tokens == 1:
            return None
        minimum = torch.finfo(hidden.dtype).min
        mask = torch.full(
            (query_tokens, total_tokens), minimum, dtype=hidden.dtype, device=hidden.device
        )
        mask = torch.triu(mask, diagonal=1 + start_pos)
        return mask.view(1, 1, query_tokens, total_tokens).expand(
            hidden.shape[0], 1, query_tokens, total_tokens
        )

    @torch.inference_mode()
    def run_cached(self, input_ids: torch.Tensor, cache: Any, start_pos: int) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] <= 0 or start_pos < 0:
            raise Round001TeacherError("cached public input must be non-empty [batch,time]")
        self.require_cache_length(cache, start_pos, "before commit")
        hidden = self.embed_tokens(input_ids)
        batch, tokens, _ = hidden.shape
        position_ids = torch.arange(
            start_pos, start_pos + tokens, device=hidden.device, dtype=torch.long
        ).view(1, -1).expand(batch, -1)
        cache_position = torch.arange(
            start_pos, start_pos + tokens, device=hidden.device, dtype=torch.long
        )
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = self._causal_mask(
            hidden, start_pos=start_pos, total_tokens=start_pos + tokens
        )
        for layer in self.layers:
            output = layer(
                hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **{self.cache_keyword: cache},
            )
            hidden = output[0] if isinstance(output, tuple) else output
        self.require_cache_length(cache, start_pos + tokens, "after commit")
        self.checked_cache_transitions += 1
        return hidden


def state_sha256(state: Mapping[str, torch.Tensor], *, domain: bytes) -> str:
    """Byte-for-byte identity domain shared with AUDIT-0004."""

    digest = hashlib.sha256(domain + b"\0")
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(dtype).to_bytes(8, "big"))
        digest.update(dtype)
        digest.update(len(tensor.shape).to_bytes(8, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


def normalize_public_embeddings(embedding: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(embedding.detach().float(), dim=-1)
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(normalized).all().item():
        raise Round001TeacherError("public embedding normalization is non-finite")
    return normalized


def load_frozen_lens(path: Path, *, device: torch.device) -> FrozenAffineLens:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"sd", "hidden", "corpus"}:
        raise Round001TeacherError("public lens checkpoint fields changed")
    if checkpoint["hidden"] != 0 or checkpoint["corpus"] != "alpaca":
        raise Round001TeacherError("public lens architecture/corpus changed")
    state = checkpoint["sd"]
    if not isinstance(state, dict) or set(state) != {"W", "b", "s"}:
        raise Round001TeacherError("public lens state fields changed")
    lens = FrozenAffineLens(2048)
    lens.load_state_dict(state, strict=True)
    lens.requires_grad_(False)
    return lens.to(device).eval()


def load_public_teacher(
    model_spec: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    lens_path: Path,
) -> tuple[PublicP0Precut, FrozenAffineLens, torch.Tensor, torch.device, dict[str, Any]]:
    """Load and independently re-hash the exact public P0 teacher state."""

    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise Round001TeacherError("Round 001 public teacher requires CUDA")
    device = torch.device("cuda")
    if model_spec != {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "revision": "9213176726f574b556790deb65791e0c5aa438b6",
        "prefix_layers": [0, 1, 2, 3],
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "local_files_only": True,
    }:
        raise Round001TeacherError("public teacher model specification changed")
    full = AutoModelForCausalLM.from_pretrained(
        model_spec["id"],
        revision=model_spec["revision"],
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    full.requires_grad_(False)
    commit = getattr(full.config, "_commit_hash", None)
    if commit not in (None, model_spec["revision"]):
        raise Round001TeacherError("loaded public model reports a conflicting revision")
    if int(full.config.vocab_size) != 128256 or int(full.config.hidden_size) != 2048:
        raise Round001TeacherError("public model dimensions changed")
    precut = PublicP0Precut(full, model_spec["prefix_layers"]).to(device).eval()
    normalized = normalize_public_embeddings(precut.embed_tokens.weight).to(device)
    lens = load_frozen_lens(lens_path, device=device)
    del full
    gc.collect()
    torch.cuda.empty_cache()

    observed = {
        "model_id": model_spec["id"],
        "model_revision": model_spec["revision"],
        "prefix_layers_sha256": state_sha256(
            dict(precut.layers.state_dict()), domain=b"ersoy-public-p0-layers-v1"
        ),
        "embedding_weight_sha256": state_sha256(
            {"embed_tokens.weight": precut.embed_tokens.weight},
            domain=b"ersoy-public-p0-embedding-v1",
        ),
        "normalized_embedding_sha256": state_sha256(
            {"normalized_embedding": normalized},
            domain=b"ersoy-public-p0-normalized-embedding-v1",
        ),
        "lens_state_sha256": state_sha256(
            dict(lens.state_dict()), domain=b"ersoy-a1-lens-state-v1"
        ),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
    }
    for key in (
        "model_id",
        "model_revision",
        "prefix_layers_sha256",
        "embedding_weight_sha256",
        "normalized_embedding_sha256",
        "lens_state_sha256",
        "runtime",
    ):
        if observed[key] != identity[key]:
            raise Round001TeacherError(
                f"public teacher identity mismatch for {key}: {observed[key]!r}"
            )
    return precut, lens, normalized, device, observed


@torch.inference_mode()
def rank_topk(
    activation: torch.Tensor,
    *,
    lens: nn.Module,
    normalized_embeddings: torch.Tensor,
    top_k: int = 512,
    chunk: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Established batched-topk A1 path, returning CPU IDs and top-1 confidence."""

    if activation.ndim != 2 or activation.shape[1] != 2048:
        raise Round001TeacherError("A1 activation must be [positions,2048]")
    if top_k != 512 or chunk != 256:
        raise Round001TeacherError("Round 001 A1 top-k/chunk changed")
    candidates: list[torch.Tensor] = []
    confidence: list[torch.Tensor] = []
    for start in range(0, activation.shape[0], chunk):
        logits = lens(
            activation[start : start + chunk].to(normalized_embeddings.device),
            normalized_embeddings,
        ).float()
        top_score, selected = torch.topk(
            logits, k=top_k, dim=-1, largest=True, sorted=True
        )
        candidates.append(selected.to(device="cpu", dtype=torch.int32))
        confidence.append(
            torch.exp(top_score[:, 0] - torch.logsumexp(logits, dim=1)).cpu()
        )
    return torch.cat(candidates, dim=0), torch.cat(confidence, dim=0)


def _centered_cosine_scores(simulated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    simulated = simulated.float()
    target = target.to(simulated.device).float().view(1, -1)
    mean = simulated.mean(dim=0, keepdim=True)
    scores = F.cosine_similarity(simulated - mean, target - mean, dim=-1)
    if not torch.isfinite(scores).all().item():
        raise Round001TeacherError("public candidate scores are non-finite")
    return scores


@torch.inference_mode()
def _simulate_candidates(
    precut: PublicP0Precut,
    *,
    cache: Any,
    candidate_ids: torch.Tensor,
    position: int,
    device: torch.device,
) -> torch.Tensor:
    if candidate_ids.ndim != 1 or candidate_ids.numel() <= 0:
        raise Round001TeacherError("candidate simulator received an empty vector")
    candidate_cache = copy.deepcopy(cache)
    repeat = getattr(candidate_cache, "batch_repeat_interleave", None)
    if not callable(repeat):
        raise Round001TeacherError("public P0 cache cannot repeat candidate prefixes")
    repeat(int(candidate_ids.numel()))
    hidden = precut.run_cached(
        candidate_ids.to(device=device, dtype=torch.long).view(-1, 1),
        candidate_cache,
        position,
    )
    return hidden[:, -1].detach().float()


def _progressive_decision(
    *,
    precut: PublicP0Precut,
    cache: Any,
    proposals: torch.Tensor,
    target: torch.Tensor,
    position: int,
    device: torch.device,
) -> tuple[int | None, int, int]:
    if proposals.shape != (512,):
        raise Round001TeacherError("strict cascade requires an ordered top-512 vector")
    parts: list[torch.Tensor] = []
    previous = 0
    for k in TIER_K:
        parts.append(
            _simulate_candidates(
                precut,
                cache=cache,
                candidate_ids=proposals[previous:k],
                position=position,
                device=device,
            )
        )
        hidden = torch.cat(parts, dim=0)
        if hidden.shape[0] != k:
            raise Round001TeacherError("progressive candidate reuse changed ordering")
        scores = _centered_cosine_scores(hidden, target)
        posterior = torch.softmax(scores.float(), dim=0)
        winner = int(scores.argmax().item())
        normalized = float(k * posterior[winner].item())
        if not math.isfinite(normalized) or normalized < 0.0:
            raise Round001TeacherError("normalized winner is invalid")
        if normalized >= NORMALIZED_WINNER_MIN:
            return int(proposals[winner].item()), k, k
        previous = k
    return None, 512, 512


@torch.inference_mode()
def decode_teacher_row(
    row: PassiveRow,
    *,
    candidates: torch.Tensor,
    a1_confidence: torch.Tensor,
    precut: PublicP0Precut,
    device: torch.device,
) -> TeacherRowResult:
    """Strict BOS-only cascade; no target-token comparison is possible here."""

    row.validate()
    if candidates.shape != (row.attention_mask.numel(), 512):
        raise Round001TeacherError("teacher candidates changed shape")
    if a1_confidence.shape != row.attention_mask.shape:
        raise Round001TeacherError("teacher A1 confidence changed shape")
    valid = torch.nonzero(row.attention_mask.bool(), as_tuple=False).flatten().tolist()
    token_ids = torch.full_like(row.attention_mask, INVALID_TOKEN_ID, dtype=torch.long)
    routes = torch.full_like(row.attention_mask, ROUTE_PADDING, dtype=torch.int8)
    first = valid[0]
    token_ids[first] = BOS_TOKEN_ID
    routes[first] = ROUTE_BOS
    cache = precut.new_cache()
    precut.run_cached(
        torch.tensor([[BOS_TOKEN_ID]], dtype=torch.long, device=device), cache, 0
    )
    simulations = 0
    stopped = False
    for logical, physical in enumerate(valid[1:], start=1):
        if stopped:
            routes[physical] = ROUTE_ABSTAINED_SUFFIX
            continue
        confidence = float(a1_confidence[physical].item())
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise Round001TeacherError("teacher A1 confidence is outside [0,1]")
        if confidence >= A1_FAST_PATH_MIN_CONFIDENCE:
            chosen = int(candidates[physical, 0].item())
            route = ROUTE_A1
        else:
            chosen_or_none, final_k, used = _progressive_decision(
                precut=precut,
                cache=cache,
                proposals=candidates[physical],
                target=row.activation[physical],
                position=logical,
                device=device,
            )
            simulations += used
            if chosen_or_none is None:
                routes[physical] = ROUTE_ABSTAIN_K512
                stopped = True
                continue
            chosen = chosen_or_none
            route = {
                32: ROUTE_A2_K32,
                128: ROUTE_A2_K128,
                512: ROUTE_A2_K512,
            }[final_k]
        token_ids[physical] = chosen
        routes[physical] = route
        precut.run_cached(
            torch.tensor([[chosen]], dtype=torch.long, device=device), cache, logical
        )
    return TeacherRowResult(
        row_index=row.row_index,
        token_ids=token_ids.cpu().contiguous(),
        route_codes=routes.cpu().contiguous(),
        candidate_simulations=simulations,
    )


def passive_rows(
    activation: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor
) -> list[PassiveRow]:
    if activation.shape != (8, 16, 128, 2048):
        raise Round001TeacherError("source activation geometry changed")
    if attention_mask.shape != (8, 16, 128) or position_ids.shape != attention_mask.shape:
        raise Round001TeacherError("source mask/position geometry changed")
    flat_activation = activation.reshape(128, 128, 2048)
    flat_mask = attention_mask.reshape(128, 128)
    flat_positions = position_ids.reshape(128, 128)
    rows = [
        PassiveRow(
            row_index=index,
            activation=flat_activation[index],
            attention_mask=flat_mask[index].to(torch.long),
            position_ids=flat_positions[index].to(torch.long),
        )
        for index in range(128)
    ]
    for row in rows:
        row.validate()
    return rows


def decode_teacher_source(
    *,
    activation: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    frozen_lens: FrozenAffineLens,
    normalized_embeddings: torch.Tensor,
    precut: PublicP0Precut,
    device: torch.device,
) -> tuple[list[TeacherRowResult], torch.Tensor, torch.Tensor, dict[str, Any]]:
    rows = passive_rows(activation, attention_mask, position_ids)
    started = time.perf_counter()
    candidates, confidence = rank_topk(
        activation.reshape(-1, 2048),
        lens=frozen_lens,
        normalized_embeddings=normalized_embeddings,
    )
    candidate_seconds = time.perf_counter() - started
    candidates = candidates.reshape(128, 128, 512)
    confidence = confidence.reshape(128, 128)
    torch.cuda.synchronize(device)
    decode_started = time.perf_counter()
    results = [
        decode_teacher_row(
            row,
            candidates=candidates[row.row_index],
            a1_confidence=confidence[row.row_index],
            precut=precut,
            device=device,
        )
        for row in rows
    ]
    torch.cuda.synchronize(device)
    decode_seconds = time.perf_counter() - decode_started
    covered = sum(int(result.token_ids.ge(0).sum().item()) for result in results)
    simulations = sum(result.candidate_simulations for result in results)
    return results, candidates, confidence, {
        "rows": len(rows),
        "candidate_seconds": candidate_seconds,
        "decode_seconds": decode_seconds,
        "covered_positions_including_bos": covered,
        "candidate_simulations": simulations,
        "checked_cache_transitions": precut.checked_cache_transitions,
    }


def build_suffix_training_pairs(
    *,
    activation: torch.Tensor,
    attention_mask: torch.Tensor,
    teacher_results: Sequence[TeacherRowResult],
    frozen_candidates: torch.Tensor,
    minimum_physical_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """Build only teacher-labelled suffix pairs; no correctness audit is possible."""

    if minimum_physical_position != 25:
        raise Round001TeacherError("Round 001 suffix position rule changed")
    if len(teacher_results) != 128 or frozen_candidates.shape != (128, 128, 512):
        raise Round001TeacherError("teacher source output geometry changed")
    flat_activation = activation.reshape(128, 128, 2048)
    flat_mask = attention_mask.reshape(128, 128).bool()
    xs: list[torch.Tensor] = []
    labels: list[int] = []
    corrections: list[bool] = []
    linear_indices: list[int] = []
    routes: list[int] = []
    route_counts: dict[str, int] = {}
    for row_index, result in enumerate(teacher_results):
        if result.row_index != row_index:
            raise Round001TeacherError("teacher row ordering changed")
        for physical in range(minimum_physical_position, 128):
            if not bool(flat_mask[row_index, physical].item()):
                continue
            label = int(result.token_ids[physical].item())
            if label == INVALID_TOKEN_ID:
                continue
            route = int(result.route_codes[physical].item())
            route_counts[str(route)] = route_counts.get(str(route), 0) + 1
            xs.append(flat_activation[row_index, physical].view(1, -1))
            labels.append(label)
            corrections.append(
                int(frozen_candidates[row_index, physical, 0].item()) != label
            )
            linear_indices.append(row_index * 128 + physical)
            routes.append(route)
    if not xs:
        raise Round001TeacherError("strict teacher produced no suffix training pairs")
    label_tensor = torch.tensor(labels, dtype=torch.long)
    correction_tensor = torch.tensor(corrections, dtype=torch.bool)
    provenance = {
        "pairs": len(labels),
        "minimum_physical_position": minimum_physical_position,
        "corrections_to_frozen_a1": int(correction_tensor.sum().item()),
        "route_counts": dict(sorted(route_counts.items())),
    }
    retained = {
        "linear_indices": torch.tensor(linear_indices, dtype=torch.int32),
        "teacher_token_ids": label_tensor.to(torch.int32),
        "frozen_top1_token_ids": frozen_candidates.reshape(-1, 512)[
            torch.tensor(linear_indices, dtype=torch.long), 0
        ].to(torch.int32),
        "correction_mask": correction_tensor.to(torch.uint8),
        "route_codes": torch.tensor(routes, dtype=torch.int8),
    }
    return (
        torch.cat(xs, dim=0).to(torch.bfloat16),
        label_tensor,
        correction_tensor,
        provenance,
        retained,
    )


def train_adapter(
    *,
    frozen_lens: FrozenAffineLens,
    normalized_embeddings: torch.Tensor,
    activations: torch.Tensor,
    labels: torch.Tensor,
    corrections: torch.Tensor,
    spec: AdapterSpec,
    device: torch.device,
    adapter_seed: int,
    sampler_seed: int,
) -> tuple[ResidualA1Adapter, dict[str, Any]]:
    spec.validate()
    if activations.ndim != 2 or activations.shape[1] != 2048:
        raise Round001TeacherError("adapter activations are malformed")
    if labels.shape != (len(activations),) or corrections.shape != labels.shape:
        raise Round001TeacherError("adapter labels/corrections are malformed")
    model = ResidualA1Adapter(frozen_lens, spec.rank, seed=adapter_seed).to(device)
    optimizer = torch.optim.AdamW(
        list(model.down.parameters()) + list(model.up.parameters()),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    sampler = torch.Generator(device="cpu").manual_seed(sampler_seed)
    losses: list[float] = []
    minibatch_correct: list[int] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _step in range(spec.steps):
        chosen = torch.randint(
            0, len(labels), (spec.batch_size,), generator=sampler, device="cpu"
        )
        x = activations[chosen].to(device=device, dtype=torch.float32)
        y = labels[chosen].to(device=device)
        is_correction = corrections[chosen].to(device=device)
        logits = model(x, normalized_embeddings)
        per_example = F.cross_entropy(logits, y, reduction="none")
        weights = torch.where(
            is_correction,
            torch.full_like(per_example, spec.correction_weight),
            torch.ones_like(per_example),
        )
        loss = (per_example * weights).sum() / weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), spec.gradient_clip, error_if_nonfinite=True
        )
        if not torch.isfinite(norm).item():
            raise Round001TeacherError("adapter gradient norm is non-finite")
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        minibatch_correct.append(int(logits.argmax(dim=-1).eq(y).sum().item()))
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    model.eval()
    return model, {
        "spec": asdict(spec),
        "training_pairs": int(len(labels)),
        "adapter_seed": adapter_seed,
        "sampler_seed": sampler_seed,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "seconds": seconds,
        "losses": losses,
        "minibatch_correct_counts": minibatch_correct,
        "minibatch_denominator": spec.batch_size,
    }


def adapter_state(model: ResidualA1Adapter) -> dict[str, torch.Tensor]:
    return {
        "down.weight": model.down.weight.detach().cpu().contiguous(),
        "up.weight": model.up.weight.detach().cpu().contiguous(),
    }


__all__ = [
    "A1_FAST_PATH_MIN_CONFIDENCE",
    "AdapterSpec",
    "BOS_TOKEN_ID",
    "FrozenAffineLens",
    "INVALID_TOKEN_ID",
    "NORMALIZED_WINNER_MIN",
    "PassiveRow",
    "PublicP0Precut",
    "ResidualA1Adapter",
    "Round001TeacherError",
    "TIER_K",
    "adapter_state",
    "build_suffix_training_pairs",
    "decode_teacher_source",
    "load_public_teacher",
    "passive_rows",
    "rank_topk",
    "state_sha256",
    "train_adapter",
]
