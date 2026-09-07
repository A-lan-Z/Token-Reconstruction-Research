from types import SimpleNamespace

import torch

from scripts.trr_p09.prepare_public_validation import (
    _build_rows_and_joins,
    _validate_public_batches,
)


def _declared(style: str, index: int, record_id: str) -> dict[str, object]:
    return {
        "dataset_key": style,
        "row_index": index,
        "record_id": record_id,
        "public_record_sha256": f"{index + 1:064x}",
        "final_sequence_sha256": f"{index + 100:064x}",
    }


def test_public_validation_join_is_global_and_domain_partitioned() -> None:
    selected = {
        "finance": [_declared("finance", 12000, "finance-0"), _declared("finance", 12001, "finance-1")],
        "pile": [_declared("pile", 7000, "pile-0")],
    }
    cells = {
        "Finance": {"records": 2, "record_ids_sha256": "a" * 64, "cell_id": "finance__public_base", "h": {"path": "/tmp/finance.safetensors", "bytes": 1, "sha256": "b" * 64}, "activations_key": "activations", "attention_mask_key": "attention_mask", "position_ids_key": "position_ids"},
        "Pile": {"records": 1, "record_ids_sha256": "c" * 64, "cell_id": "pile__public_base", "h": {"path": "/tmp/pile.safetensors", "bytes": 1, "sha256": "d" * 64}, "activations_key": "activations", "attention_mask_key": "attention_mask", "position_ids_key": "position_ids"},
    }
    # Bind the fixture cells to the trusted record-ID digest expected by the helper.
    from scripts import trr0009_eval_capture as capture

    digests = capture._digest_record_ids(selected)
    cells["Finance"]["record_ids_sha256"] = digests["finance"]
    cells["Pile"]["record_ids_sha256"] = digests["pile"]
    rows, by_domain, joins, digest, _ = _build_rows_and_joins(selected, cells)

    assert rows == [
        {"global_row": 0, "record_id": "finance-0", "domain": "Finance"},
        {"global_row": 1, "record_id": "finance-1", "domain": "Finance"},
        {"global_row": 2, "record_id": "pile-0", "domain": "Pile"},
    ]
    assert by_domain == {"Finance": [0, 1], "Pile": [2]}
    assert [join["observation_row"] for join in joins] == [0, 1, 0]
    assert digest == _build_rows_and_joins(selected, cells)[3]


def test_validation_batch_crop_preserves_public_h128_alignment() -> None:
    tokens = torch.full((2, 192), 128001, dtype=torch.int32)
    tokens[:, 0] = 128000
    tokens[:, 1:] = torch.arange(1, 192, dtype=torch.int32)
    mask = torch.ones((2, 192), dtype=torch.uint8)
    positions = torch.arange(192, dtype=torch.int64).expand(2, -1).clone()
    batches = {"finance": SimpleNamespace(token_ids=tokens, attention_mask=mask, position_ids=positions)}
    prepared = _validate_public_batches(batches, {"finance": [{}, {}]})

    assert prepared["finance"]["token_ids"].shape == (2, 128)
    assert prepared["finance"]["token_ids"].dtype == torch.int32
    assert prepared["finance"]["attention_mask"].eq(1).all().item()
    assert torch.equal(prepared["finance"]["position_ids"], torch.arange(128, dtype=torch.int64).expand(2, -1))
