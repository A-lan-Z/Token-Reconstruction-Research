#!/usr/bin/env python3
"""Run inside the minimal R1 reconstruction root and prove denied access."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import socket

from token_reconstruction.experiment_runtime import utc_now, write_json_exclusive
from token_reconstruction.isolation import ISOLATION_SCHEMA, validate_isolation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=("direct_inverse", "causal_public_surrogate_search"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def namespace_identity(name: str) -> str:
    return os.readlink(f"/proc/self/ns/{name}")


def write_probe(directory: Path) -> bool:
    target = directory / ".trr0001-r1-write-probe"
    try:
        with target.open("xb") as handle:
            handle.write(b"probe\n")
    except OSError as exc:
        return exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM}
    else:
        target.unlink()
        return False


def absent_probe(name: str, target: str) -> dict:
    path = Path(target)
    absent = not path.exists() and not path.is_symlink()
    return {
        "name": name,
        "target": target,
        "observed": "absent" if absent else "unexpectedly_visible",
        "passed": absent,
    }


def network_probe() -> tuple[dict, dict]:
    connect_errno: int | None = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.25)
        connect_errno = sock.connect_ex(("1.1.1.1", 53))
        sock.close()
    except OSError as exc:
        connect_errno = exc.errno
    route_lines = Path("/proc/net/route").read_text(encoding="utf-8").splitlines()
    default_route = any(
        len(parts := line.split()) >= 2 and parts[1] == "00000000"
        for line in route_lines[1:]
    )
    interfaces = []
    for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]:
        if ":" in line:
            interfaces.append(line.split(":", 1)[0].strip())
    passed = connect_errno not in {0, None} and not default_route
    network = {
        "connect_errno": connect_errno,
        "default_route_present": default_route,
        "interfaces": sorted(interfaces),
        "passed": passed,
    }
    denial = {
        "name": "network_ipv4",
        "target": "1.1.1.1:53",
        "observed": f"connect_errno={connect_errno};default_route={default_route}",
        "passed": passed,
    }
    return network, denial


def main() -> int:
    args = parse_args()
    started_utc = utc_now()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("access manifest is create-only")
    mountinfo_bytes = Path("/proc/self/mountinfo").read_bytes()
    mount_entries = mountinfo_bytes.decode("utf-8").splitlines()
    network, network_denial = network_probe()
    denials = [
        absent_probe(
            "repository_workspace",
            "/home/alanz/spartan/punim2939/Token-Reconstruction-Research",
        ),
        absent_probe(
            "fresh_private_mapping",
            "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0001-R1/evaluator_private/fresh_selection_private.json",
        ),
        absent_probe(
            "clean_truth",
            "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0001-R1/clean/evaluator_private/blind_truth.jsonl",
        ),
        absent_probe(
            "original_truth",
            "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0001/evaluator_private/blind_truth.jsonl",
        ),
        absent_probe(
            "target_lora",
            "/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0001/evaluator_private/target_lora.safetensors",
        ),
        absent_probe(
            "dataset_cache",
            "/home/alanz/.cache/huggingface/datasets",
        ),
        network_denial,
    ]
    environment = {key: value for key, value in sorted(os.environ.items())}
    manifest = {
        "schema": ISOLATION_SCHEMA,
        "task_id": "TRR-0001",
        "revision_id": "TRR-0001-R1",
        "method": args.method,
        "started_utc": started_utc,
        "ended_utc": utc_now(),
        "exit_status": 0,
        "identity": {
            "uid": os.getuid(),
            "gid": os.getgid(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "uid_map": Path("/proc/self/uid_map").read_text(encoding="utf-8").strip(),
            "gid_map": Path("/proc/self/gid_map").read_text(encoding="utf-8").strip(),
        },
        "namespaces": {
            "user": namespace_identity("user"),
            "mount": namespace_identity("mnt"),
            "network": namespace_identity("net"),
            "pid": namespace_identity("pid"),
        },
        "mounts": {
            "mountinfo_sha256": hashlib.sha256(mountinfo_bytes).hexdigest(),
            "entries": mount_entries,
            "read_only": ["/", "/code", "/etc", "/input", "/model-repo", "/site-packages", "/usr"],
            "writable": ["/output", "/tmp"],
        },
        "permissions": {
            "root_write_denied": write_probe(Path("/")),
            "input_write_denied": write_probe(Path("/input")),
            "code_write_denied": write_probe(Path("/code")),
            "model_write_denied": write_probe(Path("/model-repo")),
            "output_write_succeeded": not write_probe(Path("/output")),
            "tmp_write_succeeded": not write_probe(Path("/tmp")),
        },
        "environment": {
            "keys": sorted(environment),
            "values": environment,
        },
        "denial_probes": denials,
        "network": network,
        "result": "PASS_FAIL_CLOSED_ACCESS_BOUNDARY",
    }
    validate_isolation_manifest(manifest, method=args.method)
    write_json_exclusive(args.output, manifest)
    print(
        {
            "status": "access_boundary_verified",
            "method": args.method,
            "denial_probes": len(denials),
            "network_connect_errno": network["connect_errno"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
