from pathlib import Path

import pytest

import trr0005_prepare_public_activations as adapter


ROOT = Path(__file__).resolve().parents[1]
TRR4_ROOT = ROOT.parent / "TRR-0004"
CORPUS_PLAN = ROOT / "experiments" / "TRR-0005" / "corpus" / "corpus_plan.json"
ORIGINAL_ARTIFACT = TRR4_ROOT / "outputs" / "TRR-0004" / "public_activation_v2" / "train_large_cut4.safetensors"
ENRICHED_TOKENS = ROOT / "experiments" / "TRR-0005" / "corpus" / "coverage_mix_v1" / "constructed_public_tokens.safetensors"
COMMON_ARTIFACT = TRR4_ROOT / "experiments" / "TRR-0004" / "fit" / "adapter_v2" / "validation_mixed_cut4.safetensors"
COMMON_RECORDS = TRR4_ROOT / "experiments" / "TRR-0004" / "fit" / "adapter_v2" / "affine_validation_records.json"


@pytest.mark.skipif(not CORPUS_PLAN.is_file(), reason="prepared TRR-0005 corpus is unavailable")
def test_plan_has_correct_post_bos_exposure_and_ordered_geometry() -> None:
    plan = adapter._plan(CORPUS_PLAN)
    assert plan["design"]["post_bos_positions"] == 124371
    assert plan["joint_training_exposure"] == {
        **plan["joint_training_exposure"],
        "batch_size": 512,
        "steps": 3000,
        "seed": 4005,
        "post_bos_positions": 124371,
    }
    original = adapter._arm_records(plan, "original_like_alpaca_v1")
    enriched = adapter._arm_records(plan, "coverage_mix_v1")
    assert [row["slot"] for row in original] == list(range(1200))
    assert [row["slot"] for row in enriched] == list(range(1200))


@pytest.mark.skipif(not ORIGINAL_ARTIFACT.is_file() or not ENRICHED_TOKENS.is_file(), reason="prepared token artifacts are unavailable")
def test_original_and_enriched_masks_match_exactly() -> None:
    original, _, original_mask = adapter._batch_from_artifact(ORIGINAL_ARTIFACT, label="original")
    enriched, _, enriched_mask = adapter._batch_from_artifact(ENRICHED_TOKENS, label="enriched")
    assert original.post_bos_positions == 124371
    assert enriched.post_bos_positions == 124371
    assert original_mask.sum(dim=1).equal(enriched_mask.sum(dim=1))


@pytest.mark.skipif(not COMMON_ARTIFACT.is_file() or not COMMON_RECORDS.is_file(), reason="TRR4 validation artifacts are unavailable")
def test_common_validation_maps_24_records_per_style_and_3133_positions() -> None:
    _, records, grouping = adapter._common_validation(COMMON_ARTIFACT, COMMON_RECORDS)
    assert len(records) == 48
    assert grouping["style_counts"] == {"alpaca": 24, "pile": 24}
    assert grouping["post_bos_positions"] == 3133
    assert grouping["post_bos_positions_by_style"] == {"alpaca": 2197, "pile": 936}
    assert grouping["selection_metric"].startswith("unweighted mean")


@pytest.mark.skipif(not CORPUS_PLAN.is_file() or not ORIGINAL_ARTIFACT.is_file() or not ENRICHED_TOKENS.is_file(), reason="prepared corpus artifacts are unavailable")
def test_coverage_diagnostics_are_descriptive_and_bos_excluded() -> None:
    plan = adapter._plan(CORPUS_PLAN)
    original_rows = adapter._arm_records(plan, "original_like_alpaca_v1")
    enriched_rows = adapter._arm_records(plan, "coverage_mix_v1")
    _, original_tokens, original_mask = adapter._batch_from_artifact(ORIGINAL_ARTIFACT, label="original")
    _, enriched_tokens, enriched_mask = adapter._batch_from_artifact(ENRICHED_TOKENS, label="enriched")
    diagnostics = adapter._coverage_diagnostics(
        original_tokens,
        original_mask,
        enriched_tokens,
        enriched_mask,
        enriched_rows,
        plan["controlled_token_selection"]["selected_token_ids"],
    )
    assert diagnostics["primary_sets_exclude_bos"] is True
    assert diagnostics["original_like_distinct_post_bos"] == 11265
    assert diagnostics["enriched_distinct_post_bos"] == 15602
    assert diagnostics["newly_covered_by_enriched_distinct_post_bos"] == 7630
    assert diagnostics["lost_from_original_distinct_post_bos"] == 3293
    assert diagnostics["natural_only"]["records"] == 1080
    assert diagnostics["controlled_supplement"]["replacement_occurrences"] == 3600
    assert diagnostics["descriptive_only"] is True
    assert diagnostics["selection_changed_after_preparation"] is False


def test_parser_defaults_to_manifest_without_capture() -> None:
    args = adapter._parser().parse_args(
        [
            "--corpus-plan", "plan.json",
            "--original-artifact", "original.safetensors",
            "--original-records", "original.json",
            "--common-validation-artifact", "validation.safetensors",
            "--common-validation-records", "validation.json",
            "--embedding-table", "embedding.safetensors",
            "--output-root", "output",
        ]
    )
    assert args.mode == "manifest"
    assert args.batch_records == 8
    assert args.cut_depth == 4
