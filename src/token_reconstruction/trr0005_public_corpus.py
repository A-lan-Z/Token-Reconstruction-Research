"""Public fitting-corpus design and preparation contracts for TRR-0005.

The TRR-0005 question is deliberately a small crossed comparison: the
TRR-0004 Alpaca fitting stream is compared with one fixed, coverage-enriched
public stream while geometry and optimizer exposure are held constant.  This
module contains the content-blind parts of that contract and the small public
data helpers used by the preparation command.

The module never accesses an evaluator-private source.  In particular, the
``fit_frequency`` ranges are separate from the declared future holdout ranges
and are checked before a caller may inspect a row.  Model activation material-
ization is intentionally outside this module: synthetic rows must be sent
through a real public-prefix forward by the producer, rather than inheriting
or splicing an activation from a parent row.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Literal


TASK_ID = "TRR-0005"
CORPUS_SCHEMA = "token-reconstruction.trr0005-public-corpus.v1"
PLAN_SCHEMA = "token-reconstruction.trr0005-public-corpus-plan.v1"

BOS_TOKEN_ID = 128000
PAD_TOKEN_ID = 128001
VOCAB_SIZE = 128256
MAX_SEQUENCE_LENGTH = 192
FIT_RECORD_COUNT = 1200
POST_BOS_POSITION_COUNT = 124371
STORED_ROW_COUNT = FIT_RECORD_COUNT + POST_BOS_POSITION_COUNT

ORIGINAL_ARM = "original_like_alpaca_v1"
ENRICHED_ARM = "coverage_mix_v1"
ARMS = (ORIGINAL_ARM, ENRICHED_ARM)

ENRICHED_COUNTS: Mapping[str, int] = {
    "alpaca_instruction": 600,
    "pile_natural": 300,
    "finance_instruction": 180,
    "controlled_pile_context": 60,
    "controlled_finance_context": 60,
}
CONTROLLED_RECORD_COUNT = 120
CONTROLLED_REPLACEMENTS_PER_RECORD = 30
CONTROLLED_REPLACEMENT_COUNT = CONTROLLED_RECORD_COUNT * CONTROLLED_REPLACEMENTS_PER_RECORD
CONTROLLED_TOKEN_ID_TARGET = 2000
CONTROLLED_LEGACY_MAX_FREQUENCY = 1
MIN_LEGACY_ABSENT_CONTROLLED_IDS = 1800
MIN_ENRICHED_DISTINCT_TOKEN_IDS = 13000

FIT_BATCH_SIZE = 512
FIT_STEPS = 3000
FIT_EXPOSURE_DRAWS = FIT_BATCH_SIZE * FIT_STEPS
FIT_SAMPLER_SEED = 4005
PREPARATION_MAX_SECONDS = 300.0

# These are data partitions, not paths to the future holdout contents.  The
# producer is allowed to inspect only the fit/frequency half in this task.
SOURCE_PARTITIONS: Mapping[str, Mapping[str, Any]] = {
    "pile": {
        "dataset_id": "NeelNanda/pile-10k",
        "split": "train",
        "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
        "fit_frequency_start": 2000,
        "fit_frequency_stop": 7000,
        "holdout_reserve_start": 7000,
        "holdout_reserve_stop": 10000,
    },
    "finance": {
        "dataset_id": "Josephgflowers/Finance-Instruct-500k",
        "split": "train",
        "revision": "583a98fb0ec14d904e9423b671d9d0fea88891b6",
        "fit_frequency_start": 2000,
        "fit_frequency_stop": 12000,
        "holdout_reserve_start": 12000,
        "holdout_reserve_stop": 20000,
    },
}

# Alpaca is reused from the registered TRR-0004 fitting bank rather than
# selected from a new frequency/holdout partition. Keep its immutable public
# identity here so source IDs are constructed by the same helper for all
# three public datasets. The partition validator intentionally remains
# limited to Pile and Finance, whose future reserve ranges are part of this
# task's blind-selection contract.
SOURCE_DATASETS: Mapping[str, Mapping[str, Any]] = {
    "alpaca": {
        "dataset_id": "tatsu-lab/alpaca",
        "split": "train",
        "revision": "dce01c9b08f87459cf36a430d809084718273017",
    },
    "pile": SOURCE_PARTITIONS["pile"],
    "finance": SOURCE_PARTITIONS["finance"],
}


class TRR0005CorpusError(ValueError):
    """Raised when a public corpus plan violates the frozen TRR-0005 design."""


@dataclass(frozen=True)
class LengthSlot:
    """One target record length copied from the TRR-0004 public fit stream."""

    slot: int
    legacy_record_id: str
    legacy_row_index: int
    post_bos_token_count: int

    @property
    def full_token_count(self) -> int:
        return self.post_bos_token_count + 1

    def as_dict(self) -> dict[str, int | str]:
        return {
            "slot": self.slot,
            "legacy_record_id": self.legacy_record_id,
            "legacy_row_index": self.legacy_row_index,
            "full_token_count": self.full_token_count,
            "post_bos_token_count": self.post_bos_token_count,
        }


@dataclass(frozen=True)
class PublicSourceRow:
    """A public source row after tokenization, without retaining its text."""

    dataset_key: str
    dataset_id: str
    split: str
    revision: str
    row_index: int
    record_id: str
    domain: str
    rendered_sha256: str
    full_token_count: int
    token_ids: tuple[int, ...]

    @property
    def post_bos_token_count(self) -> int:
        return self.full_token_count - 1

    def as_metadata(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "revision": self.revision,
            "row_index": self.row_index,
            "record_id": self.record_id,
            "domain": self.domain,
            "rendered_sha256": self.rendered_sha256,
            "full_token_count": self.full_token_count,
            "post_bos_token_count": self.post_bos_token_count,
        }


@dataclass(frozen=True)
class PlannedRecord:
    """One record in a corpus plan.

    ``replacement_positions`` are offsets after BOS.  ``replacement_token_ids``
    is populated only for controlled records.  Natural records are reconstructed
    from the source cache at capture time and independently checked against the
    recorded rendered digest.
    """

    slot: int
    record_id: str
    source_record_id: str
    dataset_key: str
    domain: str
    target_post_bos_token_count: int
    source_full_token_count: int
    rendered_sha256: str
    synthetic: bool = False
    replacement_positions: tuple[int, ...] = ()
    replacement_token_ids: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "slot": self.slot,
            "record_id": self.record_id,
            "source_record_id": self.source_record_id,
            "dataset_key": self.dataset_key,
            "domain": self.domain,
            "target_post_bos_token_count": self.target_post_bos_token_count,
            "target_full_token_count": self.target_post_bos_token_count + 1,
            "source_full_token_count": self.source_full_token_count,
            "rendered_sha256": self.rendered_sha256,
            "synthetic": self.synthetic,
        }
        if self.synthetic:
            result.update(
                {
                    "replacement_positions": list(self.replacement_positions),
                    "replacement_token_ids": list(self.replacement_token_ids),
                    "replacement_count": len(self.replacement_positions),
                }
            )
        return result


def sha256_bytes(payload: bytes) -> str:
    """Hash public rendered bytes for a stable source binding."""

    return hashlib.sha256(payload).hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    """Hash an ordered public identity stream using newline-delimited UTF-8."""

    digest = hashlib.sha256()
    for line in lines:
        digest.update(str(line).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_record_id(
    dataset_id: str,
    split: str,
    revision: str,
    row_index: int,
) -> str:
    """Return an ID bound to the public dataset revision and row index."""

    if not dataset_id or not split or not revision:
        raise TRR0005CorpusError("dataset ID, split, and revision are required")
    if not isinstance(row_index, int) or row_index < 0:
        raise TRR0005CorpusError("source row index must be a non-negative integer")
    return f"{dataset_id}/{split}@{revision}:row-{row_index:06d}"


def controlled_record_id(source_id: str, ordinal: int) -> str:
    """Return a stable synthetic identity tied to one unique public parent row."""

    if not source_id:
        raise TRR0005CorpusError("controlled rows require a parent source ID")
    if not isinstance(ordinal, int) or ordinal < 0:
        raise TRR0005CorpusError("controlled ordinal must be non-negative")
    return f"{TASK_ID}/controlled-v1/{source_id}::row-{ordinal:03d}"


def _canonical_hash_key(seed: int, dataset_key: str, row_index: int) -> str:
    return hashlib.sha256(
        f"{TASK_ID}|{seed}|{dataset_key}|row:{row_index}".encode("utf-8")
    ).hexdigest()


def deterministic_row_order(
    row_indices: Iterable[int], *, dataset_key: str, seed: int
) -> list[int]:
    """Return a deterministic content-blind order for public row indices."""

    values = [int(value) for value in row_indices]
    if any(value < 0 for value in values):
        raise TRR0005CorpusError("row indices must be non-negative")
    if len(set(values)) != len(values):
        raise TRR0005CorpusError("duplicate row index in candidate pool")
    return sorted(values, key=lambda value: (_canonical_hash_key(seed, dataset_key, value), value))


def validate_partition_index(dataset_key: str, row_index: int, *, role: Literal["fit", "holdout"] = "fit") -> None:
    """Reject any attempt to inspect a row outside its declared partition."""

    if dataset_key not in SOURCE_PARTITIONS:
        raise TRR0005CorpusError(f"unknown source partition: {dataset_key}")
    if not isinstance(row_index, int) or row_index < 0:
        raise TRR0005CorpusError("partition row index must be a non-negative integer")
    spec = SOURCE_PARTITIONS[dataset_key]
    if role == "fit":
        start, stop = int(spec["fit_frequency_start"]), int(spec["fit_frequency_stop"])
    elif role == "holdout":
        start, stop = int(spec["holdout_reserve_start"]), int(spec["holdout_reserve_stop"])
    else:  # pragma: no cover - Literal protects callers; defensive at runtime.
        raise TRR0005CorpusError(f"unknown partition role: {role}")
    if not start <= row_index < stop:
        raise TRR0005CorpusError(
            f"{dataset_key} row {row_index} is outside the declared {role} partition [{start}, {stop})"
        )


def validate_fit_only_indices(dataset_key: str, row_indices: Iterable[int]) -> tuple[int, ...]:
    """Validate and return a fit/frequency index tuple without touching holdout rows."""

    values = tuple(int(value) for value in row_indices)
    for value in values:
        validate_partition_index(dataset_key, value, role="fit")
    if len(set(values)) != len(values):
        raise TRR0005CorpusError(f"duplicate {dataset_key} fit/frequency row")
    return values


def load_trr4_length_slots(path: Path) -> tuple[LengthSlot, ...]:
    """Read only TRR-0004 metadata and return its exact target length vector."""

    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise TRR0005CorpusError(f"TRR-0004 length metadata must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TRR0005CorpusError(f"cannot parse TRR-0004 length metadata: {path}") from exc
    try:
        records = value["registration"]["fit"]["records"]
    except (KeyError, TypeError) as exc:
        raise TRR0005CorpusError("TRR-0004 metadata has no registration.fit.records") from exc
    if not isinstance(records, list) or len(records) != FIT_RECORD_COUNT:
        raise TRR0005CorpusError(
            f"TRR-0004 length vector must contain {FIT_RECORD_COUNT} records"
        )
    result: list[LengthSlot] = []
    for slot, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TRR0005CorpusError("TRR-0004 fit record metadata must be objects")
        try:
            record_id = str(record["record_id"])
            row_index = int(record["row_index"])
            count = int(record["post_bos_token_count"])
            full_count = int(record["full_token_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TRR0005CorpusError("TRR-0004 fit length record is malformed") from exc
        if not record_id or row_index < 0 or count < 31 or full_count != count + 1:
            raise TRR0005CorpusError(f"invalid TRR-0004 target length at slot {slot}")
        if full_count > MAX_SEQUENCE_LENGTH:
            raise TRR0005CorpusError(f"TRR-0004 target exceeds {MAX_SEQUENCE_LENGTH} tokens")
        result.append(LengthSlot(slot, record_id, row_index, count))
    total = sum(record.post_bos_token_count for record in result)
    if total != POST_BOS_POSITION_COUNT:
        raise TRR0005CorpusError(
            f"TRR-0004 target has {total} post-BOS positions; expected {POST_BOS_POSITION_COUNT}"
        )
    return tuple(result)


def length_vector_digest(slots: Sequence[LengthSlot]) -> str:
    """Hash the ordered target geometry without source text or token labels."""

    return sha256_lines(
        f"{slot.slot}\t{slot.post_bos_token_count}" for slot in slots
    )


def length_multiset(slots: Sequence[LengthSlot]) -> Counter[int]:
    """Return the post-BOS length multiset used for geometry comparison."""

    return Counter(int(slot.post_bos_token_count) for slot in slots)


def assert_length_vector_match(
    slots: Sequence[LengthSlot], planned_lengths: Sequence[int]
) -> None:
    """Fail closed unless a prepared arm keeps every ordered target slot length."""

    expected = tuple(int(slot.post_bos_token_count) for slot in slots)
    actual = tuple(int(value) for value in planned_lengths)
    if expected != actual:
        first_difference = next(
            (index for index, (left, right) in enumerate(zip(expected, actual)) if left != right),
            min(len(expected), len(actual)),
        )
        raise TRR0005CorpusError(
            "ordered target length vector changed at slot "
            f"{first_difference}; expected={expected[first_difference:first_difference + 1]} "
            f"actual={actual[first_difference:first_difference + 1]}"
        )


def assert_length_multiset_match(
    slots: Sequence[LengthSlot], planned_lengths: Sequence[int]
) -> None:
    """Fail closed unless a prepared arm has exactly the frozen length multiset."""

    expected = length_multiset(slots)
    actual = Counter(int(value) for value in planned_lengths)
    if expected != actual:
        missing = expected - actual
        extra = actual - expected
        raise TRR0005CorpusError(
            f"length multiset changed; missing={dict(missing)} extra={dict(extra)}"
        )


def stable_public_text_digest(rendered_text: str) -> str:
    """Return a digest for public rendered text without retaining the text."""

    if not isinstance(rendered_text, str):
        raise TRR0005CorpusError("rendered public text must be a string")
    return sha256_bytes(rendered_text.encode("utf-8"))


def token_ids_from_encoding(encoded: Any) -> tuple[int, ...]:
    """Normalize a tokenizer output to an integer tuple."""

    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise TRR0005CorpusError("tokenizer did not return a one-dimensional token sequence")
    values = tuple(int(value) for value in ids)
    if any(value < 0 for value in values):
        raise TRR0005CorpusError("tokenizer returned a negative token ID")
    return values


def ensure_bos(ids: Sequence[int], *, bos_token_id: int = BOS_TOKEN_ID) -> tuple[int, ...]:
    """Require a leading declared BOS, adding it only for raw public text."""

    values = tuple(int(value) for value in ids)
    if not values or values[0] != bos_token_id:
        raise TRR0005CorpusError(
            f"public sequence must begin with BOS {bos_token_id}; got {values[:1]}"
        )
    return values


def validate_source_row(
    row: PublicSourceRow,
    *,
    target_post_bos_token_count: int,
    bos_token_id: int = BOS_TOKEN_ID,
    vocab_size: int = VOCAB_SIZE,
) -> None:
    """Check a public row can fill one frozen-length slot."""

    if target_post_bos_token_count < 31 or target_post_bos_token_count > MAX_SEQUENCE_LENGTH - 1:
        raise TRR0005CorpusError("target post-BOS length is outside the public geometry")
    if row.full_token_count != len(row.token_ids) or row.full_token_count != row.post_bos_token_count + 1:
        raise TRR0005CorpusError(f"source row {row.record_id} has inconsistent token geometry")
    ensure_bos(row.token_ids, bos_token_id=bos_token_id)
    if row.full_token_count < target_post_bos_token_count + 1:
        raise TRR0005CorpusError(f"source row {row.record_id} is shorter than its target slot")
    if any(int(value) >= vocab_size for value in row.token_ids):
        raise TRR0005CorpusError(f"source row {row.record_id} exceeds the public vocabulary")


def select_public_token_ids(
    legacy_frequency: Mapping[int, int],
    public_frequency: Mapping[int, int],
    *,
    target_count: int = CONTROLLED_TOKEN_ID_TARGET,
    max_legacy_frequency: int = CONTROLLED_LEGACY_MAX_FREQUENCY,
    min_legacy_absent: int = MIN_LEGACY_ABSENT_CONTROLLED_IDS,
    special_token_ids: Iterable[int] = (BOS_TOKEN_ID, PAD_TOKEN_ID),
    vocab_size: int = VOCAB_SIZE,
) -> tuple[int, ...]:
    """Select ordinary IDs by public frequency, prioritizing legacy-absent IDs.

    The legacy and public maps are derived exclusively from public fitting
    rows.  Sorting is deterministic and depends on no evaluation records.
    """

    if target_count <= 0 or max_legacy_frequency < 0 or min_legacy_absent < 0:
        raise TRR0005CorpusError("token selection limits must be non-negative and positive")
    special = {int(value) for value in special_token_ids}
    candidates = [
        int(token_id)
        for token_id, frequency in public_frequency.items()
        if 0 <= int(token_id) < vocab_size
        and int(token_id) not in special
        and int(frequency) > 0
        and int(legacy_frequency.get(int(token_id), 0)) <= max_legacy_frequency
    ]
    absent = [token_id for token_id in candidates if int(legacy_frequency.get(token_id, 0)) == 0]
    if len(absent) < min_legacy_absent:
        raise TRR0005CorpusError(
            f"public fit frequencies expose only {len(absent)} legacy-absent IDs; "
            f"need {min_legacy_absent}"
        )
    if len(candidates) < target_count:
        raise TRR0005CorpusError(
            f"public fit frequencies expose only {len(candidates)} eligible IDs; need {target_count}"
        )
    ordered = sorted(
        candidates,
        key=lambda token_id: (
            int(legacy_frequency.get(token_id, 0)),
            -int(public_frequency.get(token_id, 0)),
            token_id,
        ),
    )
    selected = tuple(ordered[:target_count])
    selected_absent = sum(int(legacy_frequency.get(token_id, 0)) == 0 for token_id in selected)
    if selected_absent < min_legacy_absent:
        # This should only be possible if target_count is smaller than the
        # absence requirement; keeping the check makes the contract explicit.
        raise TRR0005CorpusError(
            f"selected only {selected_absent} legacy-absent IDs; need {min_legacy_absent}"
        )
    return selected


def replacement_positions(
    source_token_ids: Sequence[int],
    *,
    target_post_bos_token_count: int,
    count: int = CONTROLLED_REPLACEMENTS_PER_RECORD,
    structural_token_ids: Iterable[int] = (BOS_TOKEN_ID, PAD_TOKEN_ID),
) -> tuple[int, ...]:
    """Choose deterministic post-BOS positions for controlled replacements."""

    if count <= 0:
        raise TRR0005CorpusError("replacement count must be positive")
    if target_post_bos_token_count <= 0:
        raise TRR0005CorpusError("target post-BOS length must be positive")
    ids = tuple(int(value) for value in source_token_ids)
    ensure_bos(ids)
    if len(ids) < target_post_bos_token_count + 1:
        raise TRR0005CorpusError("source sequence is shorter than its target replacement window")
    structural = {int(value) for value in structural_token_ids}
    eligible = [
        offset
        for offset in range(target_post_bos_token_count)
        if ids[offset + 1] not in structural
    ]
    if len(eligible) < count:
        raise TRR0005CorpusError(
            f"only {len(eligible)} ordinary target positions are available; need {count}"
        )
    # Even spacing preserves natural contexts around the controlled labels and
    # avoids concentrating all interventions at one sequence boundary.
    chosen: list[int] = []
    for index in range(count):
        candidate_index = min(
            len(eligible) - 1,
            int(math.floor((index + 0.5) * len(eligible) / count)),
        )
        candidate = eligible[candidate_index]
        if candidate in chosen:
            # The half-open spacing above is unique for ordinary lengths, but
            # retain a defensive fallback for very small test geometries.
            candidate = next(value for value in eligible if value not in chosen)
        chosen.append(candidate)
    return tuple(chosen)


def apply_replacements(
    source_token_ids: Sequence[int],
    positions: Sequence[int],
    replacement_token_ids: Sequence[int],
    *,
    target_post_bos_token_count: int,
    structural_token_ids: Iterable[int] = (BOS_TOKEN_ID, PAD_TOKEN_ID),
    vocab_size: int = VOCAB_SIZE,
) -> tuple[int, ...]:
    """Return one constructed public sequence with ordinary replacement IDs."""

    if len(positions) != len(replacement_token_ids) or not positions:
        raise TRR0005CorpusError("replacement positions and IDs must have equal non-zero length")
    ids = list(int(value) for value in source_token_ids[: target_post_bos_token_count + 1])
    ensure_bos(ids)
    structural = {int(value) for value in structural_token_ids}
    seen: set[int] = set()
    for offset, token_id in zip(positions, replacement_token_ids):
        offset = int(offset)
        token_id = int(token_id)
        if offset < 0 or offset >= target_post_bos_token_count:
            raise TRR0005CorpusError("replacement position is outside the post-BOS window")
        if offset in seen:
            raise TRR0005CorpusError("replacement positions must be unique")
        if token_id < 0 or token_id >= vocab_size or token_id in structural:
            raise TRR0005CorpusError("replacement token ID is not an ordinary public vocabulary ID")
        if ids[offset + 1] in structural:
            raise TRR0005CorpusError("replacement position targets a structural token")
        ids[offset + 1] = token_id
        seen.add(offset)
    return tuple(ids)


def token_frequency_summary(
    token_ids: Iterable[int],
    *,
    bos_token_id: int = BOS_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID,
    exclude_special_values: bool = True,
) -> dict[str, Any]:
    """Summarize token frequencies over an explicitly selected position stream.

    Callers that already selected valid post-BOS positions should pass
    ``exclude_special_values=False``. Position masks define whether BOS or
    padding is included; filtering by token value could discard a legitimate
    special ID occurring inside a public record. The default remains compatible
    with older callers that pass an undifferentiated token stream.
    """

    counts: Counter[int] = Counter()
    total = 0
    for token_id in token_ids:
        token_id = int(token_id)
        if exclude_special_values and token_id in (bos_token_id, pad_token_id):
            continue
        counts[token_id] += 1
        total += 1
    buckets: dict[str, dict[str, int | float]] = {}
    ranges = (
        ("seen_1", 1, 1),
        ("seen_2_4", 2, 4),
        ("seen_5_16", 5, 16),
        ("seen_17_64", 17, 64),
        ("seen_65_plus", 65, None),
    )
    for name, low, high in ranges:
        values = [frequency for frequency in counts.values() if frequency >= low and (high is None or frequency <= high)]
        rows = sum(values)
        buckets[name] = {
            "distinct_token_ids": len(values),
            "token_rows": rows,
            "mean_frequency": float(mean(values)) if values else 0.0,
        }
    return {
        "token_rows": total,
        "distinct_token_ids": len(counts),
        "frequency_buckets": buckets,
        "frequency_quantiles": _frequency_quantiles(tuple(counts.values())),
        "token_frequency_by_id": {str(token_id): int(frequency) for token_id, frequency in sorted(counts.items())},
    }


def _frequency_quantiles(values: Sequence[int]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(int(value) for value in values)

    def percentile(q: float) -> float:
        index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered))) - 1))
        return float(ordered[index])

    return {
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": float(ordered[-1]),
    }


def domain_length_summary(records: Sequence[PlannedRecord]) -> dict[str, Any]:
    """Summarize source-domain and target-length composition."""

    by_domain: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_domain[record.domain].append(int(record.target_post_bos_token_count))
    result: dict[str, Any] = {}
    for domain, lengths in sorted(by_domain.items()):
        result[domain] = {
            "records": len(lengths),
            "post_bos_positions": sum(lengths),
            "mean_post_bos_length": float(mean(lengths)) if lengths else 0.0,
            "min_post_bos_length": min(lengths) if lengths else None,
            "max_post_bos_length": max(lengths) if lengths else None,
            "length_histogram": {
                str(length): count for length, count in sorted(Counter(lengths).items())
            },
        }
    return result


def coverage_contrast(
    legacy_summary: Mapping[str, Any],
    enriched_summary: Mapping[str, Any],
    *,
    minimum_distinct: int = MIN_ENRICHED_DISTINCT_TOKEN_IDS,
    minimum_legacy_absent: int = MIN_LEGACY_ABSENT_CONTROLLED_IDS,
    selected_controlled_ids: Sequence[int] = (),
    legacy_frequency: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Check the preregistered coverage contrast and return an auditable receipt."""

    enriched_distinct = int(enriched_summary.get("distinct_token_ids", 0))
    frequencies = legacy_frequency or {}
    selected_absent = sum(int(frequencies.get(int(token_id), 0)) == 0 for token_id in selected_controlled_ids)
    checks = {
        "enriched_distinct_at_least_minimum": enriched_distinct >= minimum_distinct,
        "controlled_legacy_absent_at_least_minimum": selected_absent >= minimum_legacy_absent,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL_INSUFFICIENT_CONTRAST",
        "minimum_distinct_token_ids": minimum_distinct,
        "enriched_distinct_token_ids": enriched_distinct,
        "minimum_legacy_absent_controlled_ids": minimum_legacy_absent,
        "selected_controlled_ids": len(selected_controlled_ids),
        "selected_controlled_legacy_absent_ids": selected_absent,
        "checks": checks,
        "legacy_distinct_token_ids": int(legacy_summary.get("distinct_token_ids", 0)),
    }


def expected_sampler_exposure(
    *,
    post_bos_positions: int = POST_BOS_POSITION_COUNT,
    batch_size: int = FIT_BATCH_SIZE,
    steps: int = FIT_STEPS,
) -> dict[str, Any]:
    """Return exact exposure for the joint trainer's post-BOS draw schedule."""

    if post_bos_positions <= 0 or batch_size <= 0 or steps <= 0:
        raise TRR0005CorpusError("sampler geometry must be positive")
    draws = int(batch_size) * int(steps)
    return {
        "position_scope": "post_bos_only",
        "post_bos_positions": int(post_bos_positions),
        "bos_positions_drawn": 0,
        "batch_size": int(batch_size),
        "steps": int(steps),
        "draws": draws,
        "expected_draws_per_post_bos_position": draws / int(post_bos_positions),
        "sampler": "one shared deterministic schedule over valid post-BOS positions; BOS excluded",
        "seed": FIT_SAMPLER_SEED,
    }


def validate_planned_records(
    records: Sequence[PlannedRecord],
    slots: Sequence[LengthSlot],
    *,
    expected_domain_counts: Mapping[str, int] | None = None,
) -> None:
    """Validate one arm's record count, IDs, lengths, and synthetic lineage."""

    if len(records) != len(slots):
        raise TRR0005CorpusError(f"planned record count {len(records)} != {len(slots)}")
    if [record.slot for record in records] != list(range(len(slots))):
        raise TRR0005CorpusError("planned slots must be contiguous and ordered")
    if len({record.record_id for record in records}) != len(records):
        raise TRR0005CorpusError("planned record IDs must be unique")
    assert_length_vector_match(slots, [record.target_post_bos_token_count for record in records])
    if sum(record.target_post_bos_token_count for record in records) != POST_BOS_POSITION_COUNT:
        raise TRR0005CorpusError("planned post-BOS positions changed")
    synthetic_parents: set[str] = set()
    for record in records:
        if record.target_post_bos_token_count < 31 or record.target_post_bos_token_count >= MAX_SEQUENCE_LENGTH:
            raise TRR0005CorpusError("planned record length is outside the fixed geometry")
        if record.synthetic:
            if not record.source_record_id or record.source_record_id in synthetic_parents:
                raise TRR0005CorpusError("controlled parent rows must be unique")
            synthetic_parents.add(record.source_record_id)
            if len(record.replacement_positions) != CONTROLLED_REPLACEMENTS_PER_RECORD:
                raise TRR0005CorpusError("controlled record replacement count changed")
            if len(record.replacement_positions) != len(set(record.replacement_positions)):
                raise TRR0005CorpusError("controlled replacement positions must be unique")
            if len(record.replacement_token_ids) != len(record.replacement_positions):
                raise TRR0005CorpusError("controlled replacement IDs do not align")
            if any(position < 0 or position >= record.target_post_bos_token_count for position in record.replacement_positions):
                raise TRR0005CorpusError("controlled replacement position is out of range")
    if expected_domain_counts is not None:
        actual = Counter(record.domain for record in records)
        expected = Counter({str(key): int(value) for key, value in expected_domain_counts.items()})
        if actual != expected:
            raise TRR0005CorpusError(f"domain record counts changed; expected={dict(expected)} actual={dict(actual)}")


def public_source_file_record(path: Path, *, role: str) -> dict[str, Any]:
    """Record a cache path without hashing or reading its large contents."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise TRR0005CorpusError(f"public source path does not exist: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": int(path.stat().st_size) if path.is_file() else None,
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
        "content_hash": "DEFERRED_NO_GBHASH_IN_METADATA_PHASE",
    }


def read_exclusion_record_ids(paths: Iterable[Path]) -> set[str]:
    """Read explicit public metadata manifests for source-ID exclusions.

    This function accepts only caller-named files.  It does not recursively
    search prior outputs and it never treats missing optional manifests as a
    reason to inspect another directory.
    """

    result: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_symlink() or not path.is_file():
            raise TRR0005CorpusError(f"exclusion manifest must be a regular file: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TRR0005CorpusError(f"cannot parse exclusion manifest: {path}") from exc
        _collect_record_ids(value, result)
    return result


def _collect_record_ids(value: Any, result: set[str]) -> None:
    if isinstance(value, Mapping):
        record_id = value.get("record_id")
        if isinstance(record_id, str) and record_id:
            result.add(record_id)
        for key, child in value.items():
            if key in {"source_text", "text", "token_ids", "truth", "labels"}:
                continue
            _collect_record_ids(child, result)
    elif isinstance(value, list):
        for child in value:
            _collect_record_ids(child, result)


__all__ = [
    "ARMS",
    "BOS_TOKEN_ID",
    "CONTROLLED_REPLACEMENT_COUNT",
    "CONTROLLED_REPLACEMENTS_PER_RECORD",
    "CORPUS_SCHEMA",
    "ENRICHED_ARM",
    "ENRICHED_COUNTS",
    "FIT_BATCH_SIZE",
    "FIT_EXPOSURE_DRAWS",
    "FIT_RECORD_COUNT",
    "FIT_STEPS",
    "LengthSlot",
    "MAX_SEQUENCE_LENGTH",
    "MIN_ENRICHED_DISTINCT_TOKEN_IDS",
    "MIN_LEGACY_ABSENT_CONTROLLED_IDS",
    "ORIGINAL_ARM",
    "PAD_TOKEN_ID",
    "PLAN_SCHEMA",
    "POST_BOS_POSITION_COUNT",
    "PREPARATION_MAX_SECONDS",
    "PlannedRecord",
    "PublicSourceRow",
    "SOURCE_PARTITIONS",
    "SOURCE_DATASETS",
    "STORED_ROW_COUNT",
    "TASK_ID",
    "TRR0005CorpusError",
    "apply_replacements",
    "assert_length_multiset_match",
    "assert_length_vector_match",
    "controlled_record_id",
    "coverage_contrast",
    "deterministic_row_order",
    "domain_length_summary",
    "ensure_bos",
    "expected_sampler_exposure",
    "length_multiset",
    "length_vector_digest",
    "load_trr4_length_slots",
    "public_source_file_record",
    "read_exclusion_record_ids",
    "replacement_positions",
    "select_public_token_ids",
    "sha256_bytes",
    "sha256_lines",
    "source_record_id",
    "stable_public_text_digest",
    "token_frequency_summary",
    "token_ids_from_encoding",
    "validate_fit_only_indices",
    "validate_partition_index",
    "validate_planned_records",
    "validate_source_row",
]
