"""Synthetic fail-closed tests for the TRR-0010 registration adapter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import trr0010_eval_gate as gate
from scripts import trr0010_eval_register as register


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_draft_design_is_rejected_before_selection_or_capture(tmp_path: Path) -> None:
    design = _write(
        tmp_path / "design.json",
        {
            "schema": register.DESIGN_SCHEMA,
            "task_id": gate.TASK_ID,
            "status": "PROPOSAL_V4_READY_FOR_ROOT_REVIEW",
            "code_commit": "a" * 40,
        },
    )
    with pytest.raises(register.RegisterError, match="owner-frozen"):
        register._require_design(design, root=tmp_path)


def test_frozen_design_requires_all_six_contenders(tmp_path: Path) -> None:
    design = _write(
        tmp_path / "design.json",
        {
            "schema": register.DESIGN_SCHEMA,
            "task_id": gate.TASK_ID,
            "status": register.DESIGN_STATUS,
            "code_commit": "a" * 40,
            "methods": {
                method_id: gate.METHOD_ROLES[method_id]
                for method_id in gate.METHOD_ORDER[:-1]
            },
        },
    )
    with pytest.raises(register.RegisterError, match="all six contender"):
        register._require_design(design, root=tmp_path)


def test_producer_wiring_constants_are_inherited_geometry() -> None:
    assert register.gate.RECORDS_BY_DOMAIN == {"finance": 128, "pile": 128}
    assert register.gate.CELL_ORDER == (
        "pile__public_base",
        "pile__public_lora_2601",
        "finance__public_base",
        "finance__public_lora_2601",
    )
    assert register.gate.STORED_SEQUENCE_TOKENS == 128
    assert register.gate.OBSERVATION_HIDDEN_SIZE == 2048


def test_frequency_reference_binding_freezes_both_named_banks(tmp_path: Path) -> None:
    frequency_path = _write(
        tmp_path / "frequency.json",
        {
            "schema": "token-reconstruction.trr0010-frequency-reference.v1",
            "task_id": gate.TASK_ID,
            "frequency_references": {
                "B0": {"0": 3, "7": 2},
                "B1": {"0": 30, "7": 20, "9": 1},
            },
            "truth_opened": False,
        },
    )
    bindings, metadata = register._frequency_reference_bindings(
        root=tmp_path,
        frequency_reference_path=frequency_path,
        frequency_reference_paths=None,
    )
    assert set(bindings) == {"frequency_reference_B0", "frequency_reference_B1"}
    assert metadata["bank_order"] == ["B0", "B1"]
    assert metadata["all_methods_each_bank"] is True
    assert metadata["banks"]["B0"]["support_token_count"] == 2
    assert metadata["banks"]["B1"]["support_token_count"] == 3


def test_frequency_reference_binding_rejects_missing_bank(tmp_path: Path) -> None:
    frequency_path = _write(
        tmp_path / "frequency.json",
        {
            "frequency_references": {"B0": {"0": 1}},
            "truth_opened": False,
        },
    )
    with pytest.raises(register.RegisterError, match="named B1"):
        register._frequency_reference_bindings(
            root=tmp_path,
            frequency_reference_path=frequency_path,
            frequency_reference_paths=None,
        )


def _truth_descriptor_fixture(tmp_path: Path) -> tuple[Path, dict]:
    def record(name: str) -> dict:
        path = tmp_path / name
        path.write_bytes(name.encode('utf-8'))
        return gate.file_record(path, root=tmp_path)

    registration = record('registration.json')
    run_manifest = record('run_manifest.json')
    contract = record('contract.json')
    inputs = {
        name: record(f'{name}.json')
        for name in (
            'source_selection',
            'panel',
            'public_observations',
            'capture',
            'frequency_reference_B0',
            'frequency_reference_B1',
        )
    }
    digests = {'pile': 'a' * 64, 'finance': 'b' * 64}
    observations = {
        cell: {'records': gate.RECORDS_PER_CELL, 'record_ids_sha256': digests[cell.split('__', 1)[0]]}
        for cell in gate.CELL_ORDER
    }
    freeze = {
        'truth_opened': False,
        'registration': registration,
        'run_manifest': run_manifest,
        'contract_binding': contract,
        'input_bindings': inputs,
        'observation_bindings': observations,
    }
    descriptor = {
        'schema': register.TRUTH_BINDING_SCHEMA,
        'task_id': gate.TASK_ID,
        'status': register.TRUTH_BINDING_STATUS,
        'truth_opened': False,
        'prepared_after_public_freeze': True,
        'registration': registration,
        'run_manifest': run_manifest,
        'contract_binding': contract,
        'input_bindings': inputs,
        'records_by_domain': dict(gate.RECORDS_BY_DOMAIN),
        'cell_order': list(gate.CELL_ORDER),
        'target_conditions': list(gate.TARGET_ORDER),
        'labels_shared_across_target_conditions': True,
        'truth_shape': [gate.RECORDS_PER_CELL, gate.STORED_SEQUENCE_TOKENS],
        'truth_tensor_keys': [f'{cell}__token_ids' for cell in gate.CELL_ORDER],
        'cells': [
            {
                'cell_id': cell,
                'records': gate.RECORDS_PER_CELL,
                'record_ids_sha256': digests[cell.split('__', 1)[0]],
            }
            for cell in gate.CELL_ORDER
        ],
        'truth_payload': {
            'path': str(tmp_path / 'sealed-truth.safetensors'),
            'bytes': 123,
            'sha256': 'c' * 64,
        },
    }
    descriptor_path = _write(tmp_path / 'truth_descriptor.json', descriptor)
    return descriptor_path, freeze


def test_truth_descriptor_validates_public_order_without_opening_payload(tmp_path: Path) -> None:
    descriptor_path, freeze = _truth_descriptor_fixture(tmp_path)
    checked = register.validate_truth_descriptor(
        descriptor_path,
        repository_root=tmp_path,
        freeze=freeze,
    )
    assert checked['truth_payload']['sha256'] == 'c' * 64
    assert checked['cells'][0]['cell_id'] == gate.CELL_ORDER[0]


def test_truth_descriptor_rejects_swapped_source_order_before_payload(tmp_path: Path) -> None:
    descriptor_path, freeze = _truth_descriptor_fixture(tmp_path)
    descriptor = json.loads(descriptor_path.read_text(encoding='utf-8'))
    descriptor['cells'][0]['record_ids_sha256'] = 'd' * 64
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True) + '\n', encoding='utf-8')
    with pytest.raises(register.RegisterError, match='source order'):
        register.validate_truth_descriptor(
            descriptor_path,
            repository_root=tmp_path,
            freeze=freeze,
        )
