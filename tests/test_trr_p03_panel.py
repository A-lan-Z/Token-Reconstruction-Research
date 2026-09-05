"""Focused truth-free tests for the TRR-P03 public panel selector."""

from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trr_p03" / "prepare_panel.py"
_SPEC = spec_from_file_location("trr_p03_prepare_panel", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
panel = module_from_spec(_SPEC)
_SPEC.loader.exec_module(panel)


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _Tokenizer:
    def __init__(self, ids_by_prompt):
        self.ids_by_prompt = ids_by_prompt
        self.bos_token_id = panel.BOS_TOKEN_ID

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": list(self.ids_by_prompt[text])}


def _audit():
    return {"rejections": [], "stage_counts": {}}


def test_long_record_with_early_p02_prefix_is_skipped(monkeypatch):
    monkeypatch.setattr(panel, "LENGTHS", (2,))
    monkeypatch.setattr(panel, "PER_STYLE_PER_LENGTH", 1)
    monkeypatch.setattr(panel, "STYLE_ORDER", ("coding",))
    monkeypatch.setattr(panel, "STYLE_CATEGORIES", {"coding": ("Coding",)})
    monkeypatch.setattr(panel, "_selection_key", lambda stage, length, style, index, seed: f"{index:02d}")
    rows = _Dataset(
        [
            {"prompt": "collision", "prompt_id": "c", "category": "Coding"},
            {"prompt": "same-endpoint-different-context", "prompt_id": "ok", "category": "Coding"},
        ]
    )
    tokenizer = _Tokenizer(
        {
            "collision": [13, 99],
            # Endpoint 13 is still eligible when its preceding context differs.
            "same-endpoint-different-context": [55, 13],
        }
    )
    audit = _audit()
    selected = panel._select_panel(
        rows,
        tokenizer,
        p02={(panel.BOS_TOKEN_ID, 13): ["p02-C0-v13"]},
        prior_hashes=set(),
        prior_indices=set(),
        stage="s1",
        seed=1,
        excluded_indices=set(),
        audit=audit,
    )
    assert [row["dataset_index"] for row in selected] == [1]
    assert audit["rejections"][0]["reason"] == "p02_exact_prefix_collision"
    assert audit["rejections"][0]["collisions"][0]["endpoint_id"] == 13
    assert panel._prefix_collisions(
        [panel.BOS_TOKEN_ID, 55, 13],
        {(panel.BOS_TOKEN_ID, 13): ["p02-C0-v13"]},
    ) == []


def test_stage_panels_have_disjoint_rows_and_exact_quotas(monkeypatch):
    monkeypatch.setattr(panel, "LENGTHS", (2, 3))
    monkeypatch.setattr(panel, "PER_STYLE_PER_LENGTH", 1)
    monkeypatch.setattr(
        panel,
        "STYLE_ORDER",
        ("coding", "question_answer", "creative_generation"),
    )
    monkeypatch.setattr(
        panel,
        "STYLE_CATEGORIES",
        {
            "coding": ("Coding",),
            "question_answer": ("Open QA",),
            "creative_generation": ("Generation",),
        },
    )
    rows = []
    ids_by_prompt = {}
    for index in range(12):
        category = ("Coding", "Open QA", "Generation")[index % 3]
        prompt = f"row-{index}"
        rows.append({"prompt": prompt, "prompt_id": prompt, "category": category})
        ids_by_prompt[prompt] = [100 + index, 200 + index, 300 + index]
    dataset = _Dataset(rows)
    tokenizer = _Tokenizer(ids_by_prompt)
    stage1_audit = _audit()
    stage1 = panel._select_panel(
        dataset,
        tokenizer,
        p02={},
        prior_hashes=set(),
        prior_indices=set(),
        stage="s1",
        seed=11,
        excluded_indices=set(),
        audit=stage1_audit,
    )
    stage1_indices = {row["dataset_index"] for row in stage1}
    stage2_audit = _audit()
    stage2 = panel._select_panel(
        dataset,
        tokenizer,
        p02={},
        prior_hashes=set(),
        prior_indices=set(),
        stage="s2",
        seed=12,
        excluded_indices=stage1_indices,
        audit=stage2_audit,
    )
    assert len(stage1) == len(stage2) == 6
    assert stage1_indices.isdisjoint(row["dataset_index"] for row in stage2)
    for selected in (stage1, stage2):
        assert Counter(row["scored_tokens"] for row in selected) == Counter({2: 3, 3: 3})
        assert Counter(row["style"] for row in selected) == Counter(
            {"coding": 2, "question_answer": 2, "creative_generation": 2}
        )
