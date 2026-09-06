from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.trr_p04 import freeze_predictions as freezer
from scripts.trr_p04 import prepare_panel as prep
from scripts.trr_p04 import score_predictions as score


class _FakeTokenizer:
    bos_token_id = prep.BOS_TOKEN_ID

    @staticmethod
    def _ids(value: str) -> list[int]:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return [prep.BOS_TOKEN_ID] + [1 + ((digest[i % len(digest)] + i) % 128000) for i in range(128)]

    def __call__(self, value: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return {"input_ids": self._ids(value)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if not tokenize:
            return rendered
        return self._ids(rendered)


def _fake_rows(style: str, count: int = 190) -> list[dict[str, str]]:
    if style == "pile_plain":
        return [{"text": f"pile unique public record {i} " + ("word " * 140)} for i in range(count)]
    if style == "finance_chat":
        return [
            {"system": "finance", "user": f"finance question {i}", "assistant": f"finance answer {i} " + ("word " * 140)}
            for i in range(count)
        ]
    return [
        {"instruction": f"instruction {i}", "input": "", "output": f"answer {i} " + ("word " * 140)}
        for i in range(count)
    ]


def test_public_selector_assigns_disjoint_quotas_and_anchor_geometry() -> None:
    tokenizer = _FakeTokenizer()
    seen_hashes: set[str] = set()
    seen_sequences: set[str] = set()
    exclusions = prep.Exclusions(ids=set(), hashes=set(), indices={})
    pools: dict[str, list[dict[str, object]]] = {"correction": [], "validation": [], "panel": []}
    for style in prep.STYLES:
        selected, skipped = prep._select_style(
            style,
            _fake_rows(style),
            tokenizer,
            seed=20260906,
            exclusions=exclusions,
            seen_hashes=seen_hashes,
            seen_sequences=seen_sequences,
            correction_count=85,
            validation_count=64,
            panel_count=24,
        )
        assert sum(skipped.values()) >= 0
        pools["correction"].extend(row.metadata(pool="public_correction") for row in selected["correction"])
        pools["validation"].extend(row.metadata(pool="public_validation") for row in selected["validation"])
        panel_rows, anchors = prep._panel_rows(selected["panel"])
        pools["panel"].extend(panel_rows)
        assert len(anchors) == 4
        assert [row["length_stratum"] for row in panel_rows] == [16] * 6 + [32] * 6 + [64] * 6 + [128] * 6
    all_ids = [row["record_id"] for rows in pools.values() for row in rows]
    assert len(all_ids) == len(set(all_ids)) == 85 * 3 + 64 * 3 + 24 * 3
    assert len({row["truncated_sequence_sha256"] for rows in pools.values() for row in rows}) == len(all_ids)
    assert sum(row["anchor"] for row in pools["panel"]) == 12


def _write_panel(path: Path) -> dict[str, object]:
    records = []
    for style_index, style in enumerate(("pile_plain", "finance_chat", "alpaca_instruction")):
        for length in score.PANEL_LENGTHS:
            for offset in range(6):
                records.append(
                    {
                        "pool": "fresh_evaluation",
                        "style": style,
                        "record_id": f"{style}-l{length}-r{offset}",
                        "length_stratum": length,
                        "anchor": length == 32 and offset < 4,
                    }
                )
    panel = {
        "schema": score.SELECTION_SCHEMA,
        "task_id": score.TASK_ID,
        "status": "PUBLIC_SELECTION_READY_NO_MODEL_NO_EVALUATION_TRUTH",
        "pools": {"fresh_evaluation": {"records": records}},
    }
    path.write_text(json.dumps(panel) + "\n", encoding="utf-8")
    return panel


def _write_predictions(path: Path, panel: dict[str, object]) -> None:
    records = panel["pools"]["fresh_evaluation"]["records"]
    with path.open("w", encoding="utf-8") as handle:
        for condition in score.DEFAULT_CONDITIONS:
            for seed in score.DEFAULT_SEEDS:
                for method in score.DEFAULT_METHODS:
                    for row in records:
                        length = int(row["length_stratum"])
                        handle.write(
                            json.dumps(
                                {
                                    "schema": score.PREDICTION_SCHEMA,
                                    "method_id": method,
                                    "seed": seed,
                                    "condition": condition,
                                    "record_id": row["record_id"],
                                    "predicted_token_ids": [0] * length,
                                    "anchor": False,
                                }
                            )
                            + "\n"
                        )
        for condition in score.DEFAULT_CONDITIONS:
            for row in records:
                if not row["anchor"]:
                    continue
                length = int(row["length_stratum"])
                handle.write(
                    json.dumps(
                        {
                            "schema": score.PREDICTION_SCHEMA,
                            "method_id": "native_a1_a2",
                            "seed": None,
                            "condition": condition,
                            "record_id": row["record_id"],
                            "predicted_token_ids": [0] * length,
                            "anchor": True,
                        }
                    )
                    + "\n"
                )


def _write_truth(path: Path, panel: dict[str, object]) -> None:
    records = panel["pools"]["fresh_evaluation"]["records"]
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(
                json.dumps(
                    {
                        "record_id": row["record_id"],
                        "token_ids": [0] * int(row["length_stratum"]),
                    }
                )
                + "\n"
            )


def _valid_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    panel_path = tmp_path / "panel.json"
    panel = _write_panel(panel_path)
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(pred_path, panel)
    pred_descriptor = {
        "path": str(pred_path),
        "bytes": pred_path.stat().st_size,
        "sha256": score._sha256_file(pred_path),
    }
    state_dir = tmp_path / "states"
    state_dir.mkdir()
    state_rows = []
    for method in score.DEFAULT_METHODS:
        for seed in score.DEFAULT_SEEDS:
            state_path = state_dir / f"{method}-{seed}.state"
            state_path.write_bytes(f"{method}:{seed}".encode())
            state_rows.append(
                {
                    "method_id": method,
                    "seed": seed,
                    "path": str(state_path),
                    "bytes": state_path.stat().st_size,
                    "sha256": score._sha256_file(state_path),
                }
            )
    state_manifest = tmp_path / "state_manifest.json"
    state_manifest.write_text(json.dumps({"states": state_rows}) + "\n")
    freeze = {
        "schema": score.FREEZE_SCHEMA,
        "task_id": score.TASK_ID,
        "status": "FROZEN_BEFORE_TRUTH",
        "panel_frozen": True,
        "predictions_frozen": True,
        "all_states_frozen": True,
        "truth_open_allowed": True,
        "panel": {"path": str(panel_path), "sha256": score._sha256_file(panel_path)},
        "prediction_files": [pred_descriptor],
        "state_manifest": {
            "path": str(state_manifest),
            "bytes": state_manifest.stat().st_size,
            "sha256": score._sha256_file(state_manifest),
        },
        "state_files": state_rows,
        "prediction_groups": [
            {
                "method_id": method,
                "seed": seed,
                "condition": condition,
                "anchor": False,
            }
            for condition in score.DEFAULT_CONDITIONS
            for seed in score.DEFAULT_SEEDS
            for method in score.DEFAULT_METHODS
        ] + [
            {
                "method_id": "native_a1_a2",
                "seed": None,
                "condition": condition,
                "anchor": True,
            }
            for condition in score.DEFAULT_CONDITIONS
        ],
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze) + "\n", encoding="utf-8")
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    for condition in score.DEFAULT_CONDITIONS:
        _write_truth(truth_dir / f"{condition}.jsonl", panel)
    return panel_path, freeze_path, pred_path, truth_dir


def test_scorer_rejects_before_truth_when_freeze_is_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    panel_path, freeze_path, pred_path, truth_dir = _valid_bundle(tmp_path)
    freeze = json.loads(freeze_path.read_text())
    freeze["predictions_frozen"] = False
    freeze_path.write_text(json.dumps(freeze) + "\n")

    def fail_if_truth(*args, **kwargs):
        raise AssertionError("private truth was opened before the public gate")

    monkeypatch.setattr(score, "_load_truth_after_gate", fail_if_truth)
    with pytest.raises(score.ScoreError, match="predictions_frozen"):
        score.score_predictions(
            panel_path=panel_path,
            freeze_path=freeze_path,
            prediction_paths=[pred_path],
            truth_dir=truth_dir,
            output_path=tmp_path / "score.json",
            argv=["fixture"],
        )


def test_scorer_reads_truth_only_after_gate_and_reports_dh_ds_hs_s_affine(tmp_path: Path) -> None:
    panel_path, freeze_path, pred_path, truth_dir = _valid_bundle(tmp_path)
    output = tmp_path / "score.json"
    result = score.score_predictions(
        panel_path=panel_path,
        freeze_path=freeze_path,
        prediction_paths=[pred_path],
        truth_dir=truth_dir,
        output_path=output,
        argv=["fixture"],
    )
    assert result["truth_gate"]["verified_before_truth"] is True
    assert result["truth_gate"]["truth_opened_after_gate"] is True
    assert result["truth_gate"]["prediction_files_rewritten"] is False
    pairs = {(row["left"], row["right"]) for row in result["pairwise_record_cluster_bootstrap"]}
    assert pairs == {("D", "H"), ("D", "S"), ("H", "S"), ("S", "affine")}
    assert all(row["bootstrap"]["cluster_unit"] == "source_record_id" for row in result["pairwise_record_cluster_bootstrap"])
    assert output.is_file()

def test_freezer_requires_joint_student_anchor_and_state_bindings(tmp_path: Path) -> None:
    panel_path, freeze_path, combined_path, truth_dir = _valid_bundle(tmp_path)
    rows = [json.loads(line) for line in combined_path.read_text().splitlines() if line.strip()]
    student_path = tmp_path / "student_predictions.jsonl"
    anchor_path = tmp_path / "anchor_predictions.jsonl"
    with student_path.open("w", encoding="utf-8") as student, anchor_path.open("w", encoding="utf-8") as anchor:
        for row in rows:
            (anchor if row["anchor"] else student).write(json.dumps(row) + "\n")
    prior = json.loads(freeze_path.read_text())
    state_manifest_path = Path(prior["state_manifest"]["path"])
    output = tmp_path / "joint_freeze.json"
    frozen = freezer.build_freeze(
        panel_path=panel_path,
        prediction_paths=[student_path],
        anchor_prediction_paths=[anchor_path],
        state_manifest_path=state_manifest_path,
        truth_dir=truth_dir,
        output_path=output,
        argv=["fixture"],
    )
    assert frozen["status"] == "FROZEN_BEFORE_TRUTH"
    assert frozen["truth_accessed"] is False
    assert len(frozen["state_files"]) == 8
    assert len(frozen["prediction_groups"]) == 18

