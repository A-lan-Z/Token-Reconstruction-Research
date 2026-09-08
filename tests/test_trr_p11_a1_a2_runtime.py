"""Pure-Python tests for the P11 native A1+A2 runtime binding."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.trr_p11 import a1_a2_runtime as runtime
from scripts.trr_p11 import source_selector as selector


ROOT = Path(__file__).resolve().parents[1]


def _selection_context() -> SimpleNamespace:
    rows = {}
    for domain in runtime.DOMAIN_ORDER:
        rows[domain] = [
            {
                "record_id": f"{domain}-{index}",
                "h128_sequence_sha256": hashlib.sha256(f"{domain}-h128-{index}".encode()).hexdigest(),
            }
            for index in range(selector.RECORDS_PER_DOMAIN)
        ]
    return SimpleNamespace(rows=rows)


def test_first128_binding_is_ordered_and_separate_per_domain() -> None:
    subset = runtime.first128_subset_binding(_selection_context())
    assert subset["method"] == runtime.METHOD_ID
    assert subset["records_per_domain"] == 128
    assert subset["record_ids"]["pile"][0] == "pile-0"
    assert subset["record_ids"]["finance"][-1] == "finance-127"
    assert subset["token_positions_per_cell"] == 16256
    assert subset["record_ids_sha256"]["pile"] != subset["record_ids_sha256"]["finance"]


def test_native_policy_is_immutable() -> None:
    policy = {
        "candidate_budget": 256,
        "proposal_budget": 512,
        "proposal_chunk": 256,
        "schedule": [256],
        "fast_path_id": "off",
        "score_rule": "direct_cosine",
        "terminal_action": "commit_last_winner",
        "reconstructed_prefix": True,
    }
    assert runtime.validate_native_policy(policy)["candidate_budget"] == 256
    policy["candidate_budget"] = 128
    with pytest.raises(runtime.A1A2RuntimeError, match="candidate_budget"):
        runtime.validate_native_policy(policy)


def test_resource_plan_requires_largest_cell_qualification() -> None:
    plan = runtime.resource_plan()
    assert plan["device_required"] == "cuda"
    assert plan["candidate_proposal_budget"] == 512
    assert plan["candidate_budget"] == 256
    assert plan["qualification_required_before_full_run"] is True
    assert plan["external_live_watchdog_required"] is True
    assert plan["largest_representative_cell"] == "finance__public_base"
    assert plan["maximum_wall_seconds"] is None
    assert plan["candidate_arrays_persisted"] is True
    assert plan["preflight_basis"]["known_transient_bytes_per_qualified_cell"] > 0
    assert plan["preflight_basis"]["peak_memory_measurement"].startswith("required")


def test_execution_cells_are_explicit_and_restart_safe() -> None:
    assert runtime.execution_cells(qualification_only=True) == ("finance__public_base",)
    remaining = runtime.execution_cells(qualification_cell="finance__public_base", reuse_qualification=True)
    assert remaining == tuple(cell for cell in runtime.CELL_ORDER if cell != "finance__public_base")
    with pytest.raises(runtime.A1A2RuntimeError, match="unknown qualification cell"):
        runtime.execution_cells(qualification_cell="finance__unknown", qualification_only=True)
    with pytest.raises(runtime.A1A2RuntimeError, match="cannot reuse"):
        runtime.execution_cells(qualification_only=True, reuse_qualification=True)


def test_runtime_requires_explicit_execute_before_cuda_import(tmp_path: Path) -> None:
    with pytest.raises(runtime.A1A2RuntimeError, match="explicit execute"):
        runtime.run_native_a1_a2(
            binding_path=tmp_path / "binding.json",
            observations_path=tmp_path / "observations.json",
            output_root=tmp_path / "experiments" / "TRR-P11" / "evaluation" / "a1_a2",
            repository_root=tmp_path,
            execute=False,
        )


def test_runtime_binding_revalidates_code_and_resource_hashes(tmp_path: Path) -> None:
    resources = {}
    for key, content in (("public_embedding_table", b"E"), ("retained_a1_lens", b"L"), ("public_reference", b"R")):
        path = tmp_path / f"{key}.bin"
        path.write_bytes(content)
        resources[key] = {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    code = {}
    for key, (relative, digest) in runtime.CODE_BINDINGS.items():
        path = ROOT / relative
        code[key] = {
            "path": str(path.resolve()),
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    snapshot_file = snapshot / "config.json"
    snapshot_file.write_bytes(b"C")
    resources["model_snapshot"] = {
        "path": str(snapshot),
        "files": {
            "config.json": {
                "path": str(snapshot_file),
                "bytes": 1,
                "sha256": hashlib.sha256(b"C").hexdigest(),
            }
        },
    }
    subset = runtime.first128_subset_binding(_selection_context())
    binding = {
        "schema": runtime.RUNTIME_SCHEMA,
        "task_id": runtime.TASK_ID,
        "status": runtime.RUNTIME_STATUS,
        "method_id": runtime.METHOD_ID,
        "policy": {
            "candidate_budget": 256,
            "proposal_budget": 512,
            "proposal_chunk": 256,
            "schedule": [256],
            "fast_path_id": "off",
            "score_rule": "direct_cosine",
            "terminal_action": "commit_last_winner",
            "reconstructed_prefix": True,
        },
        "subset": subset,
        "code_bindings": code,
        "resources": resources,
        "source_text_written": False,
        "token_ids_written": False,
        "target_labels_loaded": False,
        "truth_opened": False,
        "p03_holdout_accessed": False,
    }
    assert runtime.validate_runtime_binding(binding, root=ROOT)["method_id"] == runtime.METHOD_ID
    resources["public_reference"]["sha256"] = "0" * 64
    with pytest.raises(runtime.A1A2RuntimeError, match="hash binding changed"):
        runtime.validate_runtime_binding(binding, root=ROOT)
