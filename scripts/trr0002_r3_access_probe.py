#!/usr/bin/env python3
"""Prove the R3 reconstructor sees only sanitized inputs and the public base."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import socket

from token_reconstruction.experiment_runtime import utc_now, write_json_exclusive


LENS_SHA256 = "33b825dff8eb13cfe877a55bb14e3404c4e3f66355e271fb29004b2d49f4a742"
PROPOSERS = ("checkpoint_identity", "alpaca_affine_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposer", choices=PROPOSERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def namespace_identity(name: str) -> str:
    return os.readlink(f"/proc/self/ns/{name}")


def write_probe(directory: Path) -> bool:
    target = directory / ".trr0002-r3-write-probe"
    try:
        with target.open("xb") as handle:
            handle.write(b"probe\n")
    except OSError as exc:
        return exc.errno in {errno.EROFS, errno.EACCES, errno.EPERM}
    else:
        target.unlink()
        return False


def network_probe() -> dict[str, object]:
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
    interfaces = sorted(
        line.split(":", 1)[0].strip()
        for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]
        if ":" in line
    )
    return {
        "connect_errno": connect_errno,
        "default_route_present": default_route,
        "interfaces": interfaces,
        "passed": connect_errno not in {0, None} and not default_route,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError("R3 access manifest is create-only")
    input_entries = sorted(
        path.relative_to("/input").as_posix()
        for path in Path("/input").rglob("*")
        if path.is_file() or path.is_symlink()
    )
    if input_entries != ["config.json", "observations.safetensors"]:
        raise RuntimeError(f"sanitized input registry changed: {input_entries}")

    forbidden_truth_paths = (
        "/truth",
        "/evaluator-private",
        "/input/truth.json",
        "/input/blind_truth.jsonl",
        "/input/evaluator_private",
        "/home/alanz/spartan/punim2939/Token-Reconstruction-Research",
        "/home/alanz/spartan/punim2939/backdoor_lora/ersoy2026",
    )
    forbidden_dataset_paths = (
        "/dataset",
        "/home/alanz/.cache/huggingface/datasets",
    )
    forbidden_target_paths = (
        "/target-model",
        "/heavy-model",
        "/home/alanz/.cache/huggingface/hub/models--Vikhrmodels--Vikhr-Llama-3.2-1B-Instruct",
    )
    truth_visible = [path for path in forbidden_truth_paths if Path(path).exists()]
    dataset_visible = [path for path in forbidden_dataset_paths if Path(path).exists()]
    target_visible = [path for path in forbidden_target_paths if Path(path).exists()]

    lens_path = Path("/assets/lens_alpaca.pt")
    lens_available = lens_path.is_file() and not lens_path.is_symlink()
    lens_hash = sha256_file(lens_path) if lens_available else None
    expected_lens = args.proposer == "alpaca_affine_control"
    if lens_available != expected_lens:
        raise RuntimeError("proposer lens mount differs from declared control")
    if lens_available and lens_hash != LENS_SHA256:
        raise RuntimeError("mounted Alpaca lens hash changed")
    if args.proposer == "checkpoint_identity" and Path("/assets").exists():
        raise RuntimeError("strict proposer unexpectedly received an assets mount")

    network = network_probe()
    public_model_visible = (
        Path("/model-repo/config.json").is_file()
        and Path("/model-repo/model.safetensors").is_file()
        and Path("/model-repo/tokenizer.json").is_file()
    )
    permissions = {
        "root_write_denied": write_probe(Path("/")),
        "input_write_denied": write_probe(Path("/input")),
        "code_write_denied": write_probe(Path("/code")),
        "model_write_denied": write_probe(Path("/model-repo")),
        "output_write_succeeded": not write_probe(Path("/output")),
        "tmp_write_succeeded": not write_probe(Path("/tmp")),
    }
    expected_permissions = {
        "root_write_denied": True,
        "input_write_denied": True,
        "code_write_denied": True,
        "model_write_denied": True,
        "output_write_succeeded": True,
        "tmp_write_succeeded": True,
    }
    checks = {
        "sanitized_input_registry": input_entries
        == ["config.json", "observations.safetensors"],
        "truth_absent": not truth_visible,
        "dataset_absent": not dataset_visible,
        "target_model_absent": not target_visible,
        "public_model_visible": public_model_visible,
        "network_isolated": network["passed"] is True,
        "lens_mount_matches_proposer": lens_available == expected_lens,
        "permissions": permissions == expected_permissions,
    }
    if not all(checks.values()):
        raise RuntimeError(
            "R3 fail-closed access check failed: "
            + ",".join(sorted(name for name, passed in checks.items() if not passed))
        )

    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    environment = {key: value for key, value in sorted(os.environ.items())}
    manifest = {
        "schema": "token-reconstruction.trr0002-owner-r3-isolation.v1",
        "task_id": "TRR-0002",
        "revision_id": "TRR-0002-OWNER-REVISION-R3",
        "status": "PASS",
        "proposer_id": args.proposer,
        "started_utc": utc_now(),
        "ended_utc": utc_now(),
        "exit_status": 0,
        "truth_paths_visible": len(truth_visible),
        "dataset_content_visible": bool(dataset_visible),
        "target_model_visible": bool(target_visible),
        "network_default_route": bool(network["default_route_present"]),
        "public_model_visible": public_model_visible,
        "lens_available": lens_available,
        "lens_sha256": lens_hash,
        "input_entries": input_entries,
        "checks": checks,
        "forbidden_path_observations": {
            "truth": truth_visible,
            "dataset": dataset_visible,
            "target_model": target_visible,
        },
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
            "mountinfo_sha256": hashlib.sha256(mountinfo).hexdigest(),
            "entries": mountinfo.decode("utf-8").splitlines(),
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
        "permissions": permissions,
        "environment": {"keys": sorted(environment), "values": environment},
        "network": network,
    }
    write_json_exclusive(args.output, manifest)
    print(
        json.dumps(
            {
                "status": "ACCESS_BOUNDARY_VERIFIED",
                "proposer": args.proposer,
                "truth_paths_visible": 0,
                "target_model_visible": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
