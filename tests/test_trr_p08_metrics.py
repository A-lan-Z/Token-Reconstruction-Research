from __future__ import annotations

import copy
import numpy as np
import pytest

from token_reconstruction.trr_p08_metrics import (
    INTERACTION_METHODS,
    P08MetricsError,
    aggregate_replicate_comparisons,
    bootstrap_interaction,
    classify_interaction_gate,
    interaction_from_replicates,
    paired_cluster_bootstrap,
    paired_metrics_from_scores,
    score_method,
)


METHODS = tuple(INTERACTION_METHODS.values())


def _truth(records: int = 3) -> np.ndarray:
    values = np.zeros((records, 128), dtype=np.int64)
    values[:, 0] = 128000
    values[:, 1:] = 1000 + np.arange(127, dtype=np.int64)
    return values


def _score(method_id: str, truth: np.ndarray, *, wrong: dict[int, tuple[int, int]] | None = None, mask: np.ndarray | None = None) -> dict[str, object]:
    predictions = truth.copy()
    for row, (start, stop) in (wrong or {}).items():
        predictions[row, start:stop] += 1
    return score_method(
        predictions,
        truth,
        record_ids=tuple(f"source-{i}" for i in range(len(truth))),
        attention_mask=mask,
        method_id=method_id,
    )


def _four_methods(*, reverse_seed: bool = False) -> dict[str, dict[str, object]]:
    truth = _truth(3)
    result: dict[str, dict[str, object]] = {method: {} for method in METHODS}
    for seed in (6106, 6107):
        reverse = reverse_seed and seed == 6107
        # Use one-token, source-specific errors so the interaction is paired
        # and deterministic while remaining small enough for a CPU fixture.
        for method in METHODS:
            wrong = {}
            if method == "p08_positionwise_joint":
                wrong = {0: (1, 2)}
            elif method == "p08_positionwise_staged":
                wrong = {1: (1, 2)}
            elif method == "p08_past_only_joint":
                wrong = {2: (1, 2)}
            elif method == "p08_past_only_staged":
                wrong = {0: (1, 2)}
            if reverse:
                wrong = {} if wrong else {0: (1, 2)}
            result[method][str(seed)] = _score(method, truth, wrong=wrong)
    return result


def _cell_matrix() -> dict[str, dict[str, object]]:
    methods = _four_methods()
    return {
        f"{domain}/{target}": {
            "domain": domain,
            "target": target,
            "method_scores": copy.deepcopy(methods),
        }
        for domain in ("pile", "finance")
        for target in ("public_base", "public_lora_2601")
    }


def test_unequal_length_fixture_keeps_micro_and_macro_distinct() -> None:
    truth = _truth(2)
    mask = np.zeros((2, 128), dtype=bool)
    mask[0, :2] = True  # one scored post-BOS token
    mask[1, :8] = True  # seven scored post-BOS tokens
    scored = _score("p08_positionwise_joint", truth, wrong={0: (1, 2)}, mask=mask)

    assert scored["metrics"]["scored_tokens"] == 8
    assert scored["metrics"]["correct_tokens"] == 7
    assert scored["metrics"]["token_accuracy"] == pytest.approx(7 / 8)
    assert scored["metrics"]["macro_token_accuracy"] == pytest.approx(0.5)
    assert scored["metrics"]["exact_denominator"] == 0


def test_replicate_aggregation_preserves_fractional_gains_exact_and_source_n() -> None:
    truth = _truth(2)
    left_6106 = _score("p08_past_only_staged", truth)
    right_6106 = _score("p08_positionwise_staged", truth, wrong={0: (1, 2), 1: (1, 2)})
    left_6107 = _score("p08_past_only_staged", truth, wrong={0: (1, 2), 1: (1, 2)})
    right_6107 = _score("p08_positionwise_staged", truth)
    first = paired_metrics_from_scores(left_6106, right_6106, contrast_id="past_staged_minus_positionwise_staged")
    second = paired_metrics_from_scores(left_6107, right_6107, contrast_id="past_staged_minus_positionwise_staged")

    aggregate = aggregate_replicate_comparisons({"6106": first, "6107": second})
    assert aggregate["records"] == 2
    assert aggregate["replicate_count"] == 2
    assert aggregate["metrics"]["scored_tokens"] == 254
    assert aggregate["metrics"]["token_gains"] == pytest.approx(1.0)
    assert aggregate["metrics"]["token_losses"] == pytest.approx(1.0)
    assert aggregate["metrics"]["token_delta_pp"] == pytest.approx(0.0)
    assert aggregate["metrics"]["left_exact_records"] == pytest.approx(1.0)
    assert aggregate["metrics"]["right_exact_records"] == pytest.approx(1.0)
    assert aggregate["per_record"][0]["left_correct_tokens"] == pytest.approx(126.5)
    assert aggregate["per_record"][0]["token_gains"] == pytest.approx(0.5)


def test_interaction_averages_each_seed_before_source_bootstrap() -> None:
    interaction = interaction_from_replicates(_four_methods(reverse_seed=True))
    assert interaction["records"] == 3
    assert interaction["replicate_count"] == 2
    assert interaction["aggregation"].startswith("average each seed")
    assert all(isinstance(row["interaction_token_delta"], float) for row in interaction["per_record"])
    summary = bootstrap_interaction(interaction, draws=128, seed=8080)
    assert summary["draws"] == 128
    assert summary["records"] == 3
    assert summary["ci95_percentile"]["token_delta_pp"][0] is not None
    assert summary["schedule_sha256"] == bootstrap_interaction(interaction, draws=128, seed=8080)["schedule_sha256"]


def test_shared_domain_schedule_is_reused_across_targets_and_seeded() -> None:
    first = paired_cluster_bootstrap(_cell_matrix(), draws=64, seed=8080)
    second = paired_cluster_bootstrap(_cell_matrix(), draws=64, seed=8080)
    assert first == second
    assert len(first["cells"]) == 4
    assert first["cells"]["pile/public_base"]["schedule_sha256"] == first["cells"]["pile/public_lora_2601"]["schedule_sha256"]
    assert first["cells"]["finance/public_base"]["schedule_sha256"] == first["cells"]["finance/public_lora_2601"]["schedule_sha256"]
    assert first["cells"]["pile/public_base"]["schedule_sha256"] != first["cells"]["finance/public_base"]["schedule_sha256"]


def test_source_order_mismatch_across_targets_is_rejected() -> None:
    cells = _cell_matrix()
    changed = cells["pile/public_lora_2601"]
    methods = changed["method_scores"]
    for seed_scores in methods.values():
        for score in seed_scores.values():
            score["record_ids"] = list(reversed(score["record_ids"]))
            for row, record_id in zip(score["per_record"], score["record_ids"]):
                row["record_id"] = record_id
    with pytest.raises(P08MetricsError, match="source-record order"):
        paired_cluster_bootstrap(cells, draws=8, seed=8080)


def _summary(token_ci: tuple[float, float], exact_ci: tuple[float, float], *, token_point: float = 0.0, exact_point: float = 0.0) -> dict[str, object]:
    return {
        "point": {"token_delta_pp": token_point, "exact_delta_pp": exact_point},
        "ci95_percentile": {"token_delta_pp": list(token_ci), "exact_delta_pp": list(exact_ci)},
    }


def test_gate_applies_practical_ruled_out_before_mixed_sign_label() -> None:
    summaries = {
        "pile": _summary((-0.2, 0.2), (-1.0, 4.0), token_point=-0.05, exact_point=0.2),
        "finance": _summary((-0.15, 0.25), (-2.0, 4.5), token_point=0.04, exact_point=-0.3),
    }
    result = classify_interaction_gate(
        summaries,
        seed_interactions={"pile": (-0.1, 0.1), "finance": (-0.1, 0.1)},
    )
    assert result["disposition"] == "USEFUL_CONTEXTUAL_BENEFIT_RULED_OUT"


def test_gate_rejects_ruled_out_when_one_seed_reaches_positive_margin() -> None:
    summaries = {
        "pile": _summary((-0.2, 0.2), (-1.0, 4.0)),
        "finance": _summary((-0.15, 0.25), (-2.0, 4.5)),
    }
    result = classify_interaction_gate(
        summaries,
        seed_interactions={"pile": (0.5, -0.1), "finance": (0.1, -0.1)},
    )
    assert result["disposition"] == "INCONCLUSIVE"
    assert result["ruled_out_seed_ok"]["pile"] is False


def test_gate_rejects_support_when_one_seed_is_materially_negative() -> None:
    summaries = {
        "pile": _summary((0.6, 1.4), (6.0, 9.0), token_point=1.0, exact_point=7.0),
        "finance": _summary((0.7, 1.5), (6.0, 9.0), token_point=1.1, exact_point=7.0),
    }
    result = classify_interaction_gate(
        summaries,
        seed_interactions={"pile": (-0.5, 1.0), "finance": (1.0, 1.1)},
    )
    assert result["disposition"] == "INCONCLUSIVE"
    assert result["support_seed_ok"]["pile"] is False
