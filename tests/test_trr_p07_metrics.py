from __future__ import annotations

import numpy as np
import pytest

from token_reconstruction.trr_p07_metrics import (
    CONTRASTS,
    P07MetricsError,
    aggregate_replicate_comparisons,
    paired_cluster_bootstrap,
    paired_metrics_from_scores,
    score_method,
)


def _scores(
    method_id: str,
    record_ids: tuple[str, ...],
    truth: np.ndarray,
    wrong: dict[int, tuple[int, int]],
) -> dict[str, object]:
    predictions = truth.copy()
    for row, (start, stop) in wrong.items():
        predictions[row, start:stop] += 1
    return score_method(
        predictions,
        truth,
        record_ids=record_ids,
        method_id=method_id,
    )


def test_replicate_aggregation_keeps_fractional_joint_counts_and_source_n() -> None:
    record_ids = ("source-0", "source-1")
    truth = np.zeros((2, 128), dtype=np.int64)
    truth[:, 0] = 128000
    truth[:, 1:] = 1000 + np.arange(127)

    # The fit replicates reverse which method is correct.  Aggregation must
    # average the paired events within each source before any resampling.
    left_6106 = _scores("left", record_ids, truth, {})
    right_6106 = _scores("right", record_ids, truth, {0: (1, 2), 1: (1, 2)})
    left_6107 = _scores("left", record_ids, truth, {0: (1, 2), 1: (1, 2)})
    right_6107 = _scores("right", record_ids, truth, {})
    first = paired_metrics_from_scores(left_6106, right_6106, contrast_id="left_minus_right")
    second = paired_metrics_from_scores(left_6107, right_6107, contrast_id="left_minus_right")

    averaged = aggregate_replicate_comparisons({"6106": first, "6107": second})
    assert averaged["records"] == 2
    assert averaged["replicate_count"] == 2
    assert averaged["metrics"]["scored_tokens"] == 254
    assert averaged["metrics"]["token_gains"] == pytest.approx(1.0)
    assert averaged["metrics"]["token_losses"] == pytest.approx(1.0)
    assert averaged["metrics"]["token_delta_pp"] == pytest.approx(0.0)
    assert averaged["metrics"]["left_exact_records"] == pytest.approx(1.0)
    assert averaged["metrics"]["right_exact_records"] == pytest.approx(1.0)
    assert averaged["per_record"][0]["left_correct_tokens"] == pytest.approx(126.5)
    assert averaged["per_record"][0]["right_correct_tokens"] == pytest.approx(126.5)
    assert averaged["per_record"][0]["token_gains"] == pytest.approx(0.5)
    assert averaged["per_record"][0]["token_losses"] == pytest.approx(0.5)


def _bootstrap_cells(*, reverse_target_order: bool = False) -> dict[str, dict[str, object]]:
    records = 3
    record_ids = tuple(f"source-{index}" for index in range(records))
    truth = np.zeros((records, 128), dtype=np.int64)
    truth[:, 0] = 128000
    truth[:, 1:] = 2000 + np.arange(127)

    def method_scores(seed: int, *, target: str) -> dict[str, dict[str, object]]:
        # Keep all four method families present in the fixture.  P06 methods
        # have two replicate seeds; retained methods use one retained key.
        flip = (seed == 6107) ^ (target == "public_lora_2601")
        past_wrong = {0: (1, 2)} if not flip else {}
        diagonal_wrong = {1: (1, 2)}
        return {
            "p06_past_only": {
                "6106": _scores("p06_past_only", record_ids, truth, past_wrong),
                "6107": _scores("p06_past_only", record_ids, truth, past_wrong),
            },
            "p06_positionwise_diagonal": {
                "6106": _scores("p06_positionwise_diagonal", record_ids, truth, diagonal_wrong),
                "6107": _scores("p06_positionwise_diagonal", record_ids, truth, diagonal_wrong),
            },
            "trr0006_positionwise_reference": {
                "retained": _scores("reference", record_ids, truth, {}),
            },
            "trr0006_causal_enriched": {
                "retained": _scores("causal", record_ids, truth, {2: (1, 2)}),
            },
        }

    cells: dict[str, dict[str, object]] = {}
    for panel in ("p06_panel", "trr0006_subset"):
        for domain in ("pile", "finance"):
            for target in ("public_base", "public_lora_2601"):
                ids = tuple(reversed(record_ids)) if reverse_target_order and target == "public_lora_2601" else record_ids
                # All arrays remain in the original order; changing only the
                # declared IDs exercises the paired source-order guard.
                scores = method_scores(6106, target=target)
                if ids != record_ids:
                    for methods in scores.values():
                        for score in methods.values():
                            score["record_ids"] = list(ids)
                            for row, record_id in zip(score["per_record"], ids):
                                row["record_id"] = record_id
                cells[f"{panel}/{domain}/{target}"] = {
                    "panel": panel,
                    "domain": domain,
                    "target": target,
                    "scores": scores,
                }
    return cells


def test_shared_seeded_source_bootstrap_is_deterministic_and_target_paired() -> None:
    first = paired_cluster_bootstrap(_bootstrap_cells(), draws=64, seed=7007)
    second = paired_cluster_bootstrap(_bootstrap_cells(), draws=64, seed=7007)
    assert first == second
    assert first["draws"] == 64
    assert first["seed"] == 7007
    assert first["unit"] == "source-record cluster"
    assert set(first["contrasts"]) == set(CONTRASTS)
    assert len(first["cells"]) == 8

    by_panel_domain: dict[tuple[str, str], list[dict[str, object]]] = {}
    for cell in first["cells"].values():
        by_panel_domain.setdefault((cell["panel"], cell["domain"]), []).append(cell)
        for contrast in cell["contrasts"].values():
            assert contrast["bootstrap"]["records"] == 3
            assert contrast["bootstrap"]["draws_with_exact_observation"] == 64
            assert contrast["replicate_averaged"]["records"] == 3
    for cells in by_panel_domain.values():
        assert len({cell["schedule_sha256"] for cell in cells}) == 1
        assert len({cell["contrasts"]["past_minus_reference"]["schedule_sha256"] for cell in cells}) == 1


def test_source_order_change_is_rejected_even_with_same_shape() -> None:
    with pytest.raises(P07MetricsError, match="source-record order"):
        paired_cluster_bootstrap(_bootstrap_cells(reverse_target_order=True), draws=8, seed=7007)


def test_padded_only_exact_interval_is_explicitly_undefined() -> None:
    records = 3
    ids = tuple(f"source-{index}" for index in range(records))
    truth = np.zeros((records, 128), dtype=np.int64)
    truth[:, 0] = 128000
    truth[:, 1:] = 3000 + np.arange(127)
    mask = np.zeros((records, 128), dtype=bool)
    mask[:, :4] = True
    short = _scores("short", ids, truth, {})
    # score_method does not accept a score-side mask, so recreate it with the
    # same public interface and ensure no full-record exact observation exists.
    pred = truth.copy()
    short = score_method(pred, truth, record_ids=ids, attention_mask=mask, method_id="short")
    methods = {
        "p06_past_only": {"6106": short, "6107": short},
        "p06_positionwise_diagonal": {"6106": short, "6107": short},
        "trr0006_positionwise_reference": {"retained": short},
        "trr0006_causal_enriched": {"retained": short},
    }
    cells = {
        f"{panel}/{domain}/{target}": {"panel": panel, "domain": domain, "target": target, "scores": methods}
        for panel in ("p06_panel", "trr0006_subset")
        for domain in ("pile", "finance")
        for target in ("public_base", "public_lora_2601")
    }
    result = paired_cluster_bootstrap(cells, draws=16, seed=7007)
    summary = next(iter(result["cells"].values()))["contrasts"]["past_minus_reference"]["bootstrap"]
    assert summary["exact_denominator"] == 0
    assert summary["ci95_percentile"]["exact_delta_pp"] == [None, None]
    assert summary["draws_with_exact_observation"] == 0
