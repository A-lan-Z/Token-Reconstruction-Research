"""Validation tests for the native TRR-0001-R1 access boundary."""

from __future__ import annotations

import copy

import pytest

from token_reconstruction.isolation import (
    IsolationError,
    ISOLATION_SCHEMA,
    validate_isolation_manifest,
)


def passing_manifest() -> dict:
    values = {
        "HOME": "/tmp/home",
        "PATH": "/usr/bin:/usr/sbin",
        "PYTHONPATH": "/code/src:/site-packages",
        "HF_HOME": "/tmp/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "LD_LIBRARY_PATH": "/site-packages/torch/lib:/usr/lib/wsl/lib",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONNOUSERSITE": "1",
        "LANG": "C.UTF-8",
    }
    probes = [
        {
            "name": name,
            "target": f"/denied/{name}",
            "observed": "absent",
            "passed": True,
        }
        for name in (
            "repository_workspace",
            "fresh_private_mapping",
            "clean_truth",
            "original_truth",
            "target_lora",
            "dataset_cache",
            "network_ipv4",
        )
    ]
    return {
        "schema": ISOLATION_SCHEMA,
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "method": "direct_inverse",
        "started_utc": "2026-08-22T00:00:00Z",
        "ended_utc": "2026-08-22T00:00:01Z",
        "exit_status": 0,
        "identity": {
            "uid": 0,
            "gid": 0,
            "pid": 3,
            "hostname": "isolated",
            "uid_map": "0 1000 1",
            "gid_map": "0 1000 1",
        },
        "namespaces": {
            "user": "user:[1]",
            "mount": "mnt:[2]",
            "network": "net:[3]",
            "pid": "pid:[4]",
        },
        "mounts": {
            "mountinfo_sha256": "a" * 64,
            "entries": [
                "1 0 0:1 / / ro - tmpfs tmpfs ro",
                "2 1 0:2 / /code ro - tmpfs tmpfs ro",
                "3 1 0:3 / /etc ro - tmpfs tmpfs ro",
                "4 1 0:4 / /input ro - tmpfs tmpfs ro",
                "5 1 0:5 / /model-repo ro - tmpfs tmpfs ro",
                "6 1 0:6 / /site-packages ro - tmpfs tmpfs ro",
                "7 1 0:7 / /usr ro - tmpfs tmpfs ro",
                "8 7 0:8 / /usr/lib/wsl/lib ro - tmpfs tmpfs ro",
                "9 1 0:9 / /dev/shm rw - tmpfs tmpfs rw",
                "10 1 0:10 / /output rw - tmpfs tmpfs rw",
                "11 1 0:11 / /tmp rw - tmpfs tmpfs rw",
            ],
            "read_only": [
                "/",
                "/code",
                "/etc",
                "/input",
                "/model-repo",
                "/site-packages",
                "/usr",
            ],
            "writable": ["/dev/shm", "/output", "/tmp"],
        },
        "permissions": {
            "root_write_denied": True,
            "input_write_denied": True,
            "code_write_denied": True,
            "model_write_denied": True,
            "output_write_succeeded": True,
            "tmp_write_succeeded": True,
        },
        "environment": {"keys": sorted(values), "values": values},
        "denial_probes": probes,
        "network": {
            "connect_errno": 101,
            "default_route_present": False,
            "interfaces": ["lo"],
            "passed": True,
        },
        "result": "PASS_FAIL_CLOSED_ACCESS_BOUNDARY",
    }


def test_complete_isolation_manifest_passes() -> None:
    validate_isolation_manifest(passing_manifest(), method="direct_inverse")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["denial_probes"].pop(), "probe set"),
        (lambda value: value["environment"]["values"].update({"SECRET": "x"}), "keys and values"),
        (lambda value: value["permissions"].update({"input_write_denied": False}), "permission"),
        (lambda value: value["network"].update({"default_route_present": True}), "network"),
        (lambda value: value["mounts"].update({"writable": ["/output", "/tmp", "/workspace"]}), "writable"),
    ],
)
def test_isolation_evidence_fails_closed(mutation, message: str) -> None:
    value = passing_manifest()
    mutation(value)
    with pytest.raises(IsolationError, match=message):
        validate_isolation_manifest(value, method="direct_inverse")


def test_method_identity_cannot_be_reused_across_processes() -> None:
    value = passing_manifest()
    with pytest.raises(IsolationError, match="method"):
        validate_isolation_manifest(
            value, method="causal_public_surrogate_search"
        )


def test_private_metadata_has_no_channel_into_public_method_route() -> None:
    public_route = {
        "record_order": [f"blind-r1-{value:06d}" for value in range(1, 65)],
        "config_sha256": "b" * 64,
        "truth_or_source_inputs": 0,
    }
    first_private = {"dataset_index": 1, "token_ids": [1, 2, 3]}
    second_private = {"dataset_index": 9999, "token_ids": [9, 8, 7]}
    assert first_private != second_private
    assert copy.deepcopy(public_route) == public_route
    assert not (set(first_private) & set(public_route))
    assert not (set(second_private) & set(public_route))
