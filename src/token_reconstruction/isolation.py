"""Strict validation for TRR-0001-R1 process-isolation evidence."""

from __future__ import annotations

from typing import Any, Mapping

from token_reconstruction.blind_commitment import require_exact_keys


ISOLATION_SCHEMA = "token-reconstruction.trr0001-r1-isolation-manifest.v1"
DENIAL_PROBES = {
    "repository_workspace",
    "fresh_private_mapping",
    "clean_truth",
    "original_truth",
    "target_lora",
    "dataset_cache",
    "network_ipv4",
}
_ALLOWED_ENVIRONMENT = {
    "HOME",
    "PATH",
    "PYTHONPATH",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "LD_LIBRARY_PATH",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONNOUSERSITE",
    "LANG",
}


class IsolationError(RuntimeError):
    """Raised when access-boundary evidence is absent, incomplete, or failed."""


def validate_isolation_manifest(value: Mapping[str, Any], *, method: str) -> None:
    require_exact_keys(
        value,
        {
            "schema",
            "task_id",
            "revision_id",
            "method",
            "started_utc",
            "ended_utc",
            "exit_status",
            "identity",
            "namespaces",
            "mounts",
            "permissions",
            "environment",
            "denial_probes",
            "network",
            "result",
        },
        label="isolation manifest",
    )
    if value["schema"] != ISOLATION_SCHEMA:
        raise IsolationError("isolation schema changed")
    if value["task_id"] != "TRR-0001" or value["revision_id"] != "TRR-0001-R1":
        raise IsolationError("isolation task identity changed")
    if method not in {"direct_inverse", "causal_public_surrogate_search"}:
        raise IsolationError("unknown reconstruction method")
    if value["method"] != method or value["exit_status"] != 0:
        raise IsolationError("isolation method or exit status changed")

    identity = value["identity"]
    require_exact_keys(
        identity,
        {"uid", "gid", "pid", "hostname", "uid_map", "gid_map"},
        label="isolation identity",
    )
    if identity["uid"] != 0 or identity["gid"] != 0:
        raise IsolationError("process is not rooted inside the private user namespace")
    if not identity["uid_map"] or not identity["gid_map"]:
        raise IsolationError("user namespace maps are absent")

    namespaces = value["namespaces"]
    require_exact_keys(
        namespaces, {"user", "mount", "network", "pid"}, label="namespace identities"
    )
    namespace_labels = {"user": "user", "mount": "mnt", "network": "net", "pid": "pid"}
    if any(not str(namespaces[name]).startswith(f"{namespace_labels[name]}:[") for name in namespaces):
        raise IsolationError("namespace identity is malformed")

    mounts = value["mounts"]
    require_exact_keys(
        mounts,
        {"mountinfo_sha256", "entries", "read_only", "writable"},
        label="mount evidence",
    )
    if mounts["read_only"] != ["/", "/code", "/etc", "/input", "/model-repo", "/site-packages", "/usr"]:
        raise IsolationError("read-only mount contract changed")
    if mounts["writable"] != ["/output", "/tmp"]:
        raise IsolationError("writable mount contract changed")
    if not isinstance(mounts["entries"], list) or not mounts["entries"]:
        raise IsolationError("exact mount table is absent")

    permissions = value["permissions"]
    require_exact_keys(
        permissions,
        {
            "root_write_denied", "input_write_denied", "code_write_denied",
            "model_write_denied", "output_write_succeeded", "tmp_write_succeeded",
        },
        label="permission probes",
    )
    if permissions != {
        "root_write_denied": True,
        "input_write_denied": True,
        "code_write_denied": True,
        "model_write_denied": True,
        "output_write_succeeded": True,
        "tmp_write_succeeded": True,
    }:
        raise IsolationError("mount permission probe failed")

    environment = value["environment"]
    require_exact_keys(environment, {"keys", "values"}, label="isolated environment")
    if set(environment["keys"]) != set(environment["values"]):
        raise IsolationError("environment keys and values disagree")
    if set(environment["keys"]) != _ALLOWED_ENVIRONMENT:
        raise IsolationError("environment allowlist changed")
    if environment["values"].get("HF_HUB_OFFLINE") != "1":
        raise IsolationError("offline environment is not enforced")

    probes = value["denial_probes"]
    if not isinstance(probes, list):
        raise IsolationError("denial probes are absent")
    by_name = {probe.get("name"): probe for probe in probes}
    if set(by_name) != DENIAL_PROBES or len(by_name) != len(probes):
        raise IsolationError("denial probe set changed")
    for name, probe in by_name.items():
        require_exact_keys(
            probe,
            {"name", "target", "observed", "passed"},
            label=f"denial probe {name}",
        )
        if probe["passed"] is not True:
            raise IsolationError(f"denial probe failed: {name}")

    network = value["network"]
    require_exact_keys(
        network,
        {"connect_errno", "default_route_present", "interfaces", "passed"},
        label="network probe",
    )
    if network["passed"] is not True or network["default_route_present"] is not False:
        raise IsolationError("network isolation failed")
    if value["result"] != "PASS_FAIL_CLOSED_ACCESS_BOUNDARY":
        raise IsolationError("isolation result is not passing")
