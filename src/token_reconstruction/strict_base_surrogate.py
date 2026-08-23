"""Checkpoint-only candidate proposal and intuitive reconstruction metrics.

The strict proposer has no fitted state.  It normalizes an observed cut
activation and compares it directly with the untouched public checkpoint's
normalized input-embedding table.  Its optional temperature exists only to
report the historical A0 confidence diagnostic; fixed-budget selectors never
use that confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .component_crossover import ProposalResult, _pack_proposal, _stable_score_order
from .dual_benchmark import BOS_TOKEN_ID, scored_mask


IDENTITY_TEMPERATURE_LOG = 3.0


class StrictBaseSurrogateError(RuntimeError):
    """Raised when the checkpoint-only proposer contract is violated."""


@torch.inference_mode()
def propose_checkpoint_identity(
    *,
    observations: torch.Tensor,
    attention_mask: torch.Tensor,
    normalized_embeddings: torch.Tensor,
    max_k: int = 512,
    chunk: int = 256,
) -> ProposalResult:
    """Rank tokens without a fitted lens or auxiliary training data."""

    if observations.ndim != 3 or observations.shape[:2] != attention_mask.shape:
        raise StrictBaseSurrogateError("observation and mask geometry differ")
    if observations.shape[2] != normalized_embeddings.shape[1]:
        raise StrictBaseSurrogateError("observation and embedding widths differ")
    if max_k != 512 or chunk != 256:
        raise StrictBaseSurrogateError("strict identity proposal constants changed")
    if normalized_embeddings.ndim != 2 or normalized_embeddings.shape[0] < max_k:
        raise StrictBaseSurrogateError("normalized embedding table is invalid")
    if not torch.isfinite(normalized_embeddings).all().item():
        raise StrictBaseSurrogateError("normalized embeddings are non-finite")

    mask = scored_mask(attention_mask)
    flat = observations[mask]
    all_ids: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_confidence: list[torch.Tensor] = []
    device = normalized_embeddings.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    import time

    wall_started = time.perf_counter()
    scale = float(torch.exp(torch.tensor(IDENTITY_TEMPERATURE_LOG)).item())
    for offset in range(0, flat.shape[0], chunk):
        query = F.normalize(flat[offset : offset + chunk].to(device).float(), dim=-1)
        cosine = (
            query.to(normalized_embeddings.dtype) @ normalized_embeddings.T
        ).float()
        top_values, top_ids = torch.topk(
            cosine, k=max_k, dim=1, largest=True, sorted=True
        )
        confidence = torch.exp(
            top_values[:, 0] * scale - torch.logsumexp(cosine * scale, dim=1)
        )
        ids, values = _stable_score_order(top_ids.cpu(), top_values.cpu())
        all_ids.append(ids)
        all_scores.append(values)
        all_confidence.append(confidence.float().cpu())
        del query, cosine, top_values, top_ids, values, ids, confidence
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - wall_started

    candidates = torch.cat(all_ids, dim=0)
    scores = torch.cat(all_scores, dim=0)
    confidence = torch.cat(all_confidence, dim=0)
    packed = _pack_proposal(
        attention_mask=attention_mask,
        candidate_ids=candidates,
        candidate_scores=scores,
        confidence=confidence,
    )
    return ProposalResult(*packed, elapsed)


def canonical_mapping_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    """Encode the private ordered selection for an HMAC commitment."""

    encoded: list[str] = []
    for row in rows:
        encoded.append(
            "|".join(
                (
                    str(row["opaque_id"]),
                    str(int(row["dataset_index"])),
                    str(row["source_sha256"]),
                    str(int(row["token_length"])),
                    str(row["length_bin"]),
                )
            )
        )
    return ("\n".join(encoded) + "\n").encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExactInputSummary:
    records: int
    exact_token_records: int
    exact_decoded_text_records: int
    exact_source_text_records: int | None
    failed_records: int
    errors_in_failed_records: int
    mean_errors_per_failed_record: float
    median_errors_per_failed_record: float
    maximum_errors_in_failed_record: int

    def as_dict(self) -> dict[str, Any]:
        exact_source_rate = (
            None
            if self.exact_source_text_records is None
            else self.exact_source_text_records / self.records
        )
        return {
            "records": self.records,
            "exact_token_records": self.exact_token_records,
            "exact_token_record_rate": self.exact_token_records / self.records,
            "exact_decoded_text_records": self.exact_decoded_text_records,
            "exact_decoded_text_record_rate": (
                self.exact_decoded_text_records / self.records
            ),
            "exact_source_text_records": self.exact_source_text_records,
            "exact_source_text_record_rate": exact_source_rate,
            "failed_records": self.failed_records,
            "errors_in_failed_records": self.errors_in_failed_records,
            "mean_errors_per_failed_record": self.mean_errors_per_failed_record,
            "median_errors_per_failed_record": self.median_errors_per_failed_record,
            "maximum_errors_in_failed_record": self.maximum_errors_in_failed_record,
        }


def exact_input_summary(
    *,
    predictions: torch.Tensor,
    truth: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer: Any,
    source_texts: Sequence[str] | None = None,
) -> tuple[ExactInputSummary, list[dict[str, Any]]]:
    """Report whole-input recovery and error concentration per record."""

    if predictions.shape != truth.shape or truth.shape != attention_mask.shape:
        raise StrictBaseSurrogateError("prediction, truth, and mask shapes differ")
    records = int(truth.shape[0])
    if source_texts is not None and len(source_texts) != records:
        raise StrictBaseSurrogateError("source-text count differs from records")
    if not truth[:, 0].eq(BOS_TOKEN_ID).all().item():
        raise StrictBaseSurrogateError("truth does not begin with the declared BOS")

    per_record: list[dict[str, Any]] = []
    errors_in_failures: list[int] = []
    token_exact_count = 0
    decoded_exact_count = 0
    source_exact_count = 0
    for index in range(records):
        length = int(attention_mask[index].sum().item())
        if length <= 1:
            raise StrictBaseSurrogateError("record has no scored token")
        true_ids = truth[index, :length].to(torch.long)
        predicted_ids = predictions[index, :length].to(torch.long)
        wrong = predicted_ids[1:].ne(true_ids[1:])
        errors = int(wrong.sum().item())
        token_exact = errors == 0
        true_text = tokenizer.decode(
            true_ids[1:].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        predicted_text = tokenizer.decode(
            predicted_ids[1:].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        decoded_exact = predicted_text == true_text
        source_exact: bool | None = None
        if source_texts is not None:
            source_exact = predicted_text == source_texts[index]
            source_exact_count += int(source_exact)
        first_error = (
            None
            if token_exact
            else int(torch.nonzero(wrong, as_tuple=False)[0].item()) + 1
        )
        token_exact_count += int(token_exact)
        decoded_exact_count += int(decoded_exact)
        if errors:
            errors_in_failures.append(errors)
        per_record.append(
            {
                "record_index": index,
                "valid_tokens_including_bos": length,
                "scored_tokens": length - 1,
                "correct_tokens": length - 1 - errors,
                "token_errors": errors,
                "exact_tokens": token_exact,
                "exact_decoded_text": decoded_exact,
                "exact_source_text": source_exact,
                "first_error_position": first_error,
            }
        )

    sorted_errors = sorted(errors_in_failures)
    failed = len(sorted_errors)
    if failed:
        middle = failed // 2
        median = (
            float(sorted_errors[middle])
            if failed % 2
            else (sorted_errors[middle - 1] + sorted_errors[middle]) / 2
        )
    else:
        median = 0.0
    summary = ExactInputSummary(
        records=records,
        exact_token_records=token_exact_count,
        exact_decoded_text_records=decoded_exact_count,
        exact_source_text_records=(source_exact_count if source_texts is not None else None),
        failed_records=failed,
        errors_in_failed_records=sum(sorted_errors),
        mean_errors_per_failed_record=(
            sum(sorted_errors) / failed if failed else 0.0
        ),
        median_errors_per_failed_record=median,
        maximum_errors_in_failed_record=max(sorted_errors, default=0),
    )
    return summary, per_record


def length_stratified_summary(
    per_record: Sequence[dict[str, Any]],
    *,
    bins: Sequence[tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    """Aggregate exact-input and token metrics over preregistered length bins."""

    output: dict[str, dict[str, Any]] = {}
    for lower, upper in bins:
        selected = [
            row
            for row in per_record
            if lower <= int(row["valid_tokens_including_bos"]) <= upper
        ]
        key = f"{lower}-{upper}"
        scored = sum(int(row["scored_tokens"]) for row in selected)
        correct = sum(int(row["correct_tokens"]) for row in selected)
        exact = sum(bool(row["exact_tokens"]) for row in selected)
        output[key] = {
            "records": len(selected),
            "scored_tokens": scored,
            "correct_tokens": correct,
            "token_accuracy": correct / scored if scored else None,
            "exact_token_records": exact,
            "exact_token_record_rate": exact / len(selected) if selected else None,
        }
    return output
