#!/usr/bin/env python3
"""Bind and run the TRR-0012 paired fixed-readout fit through native P09.

The task launcher is standard-library-only.  It hashes the actual public
inputs and P09 source, resolves a captured B1 bank, computes the native
combined-bank digest, writes a create-only plan, arms the native fail-closed
watchdog, invokes ``fixed_control_cli.py`` directly, and audits every durable
checkpoint after success.  The native CLI owns all tensor/model/embedding/bank
loading, optimization, validation, and checkpoint selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

TASK_ID = "TRR-0012"
TASK_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = TASK_ROOT.parent.parent
TASK_EXPERIMENT = TASK_ROOT / "experiments" / TASK_ID
TASK_OUTPUT = TASK_ROOT / "outputs" / TASK_ID
P10_ROOT = TASK_ROOT.parent / "TRR-P10"
P09_CLI = P10_ROOT / "scripts" / "trr_p09" / "fixed_control_cli.py"
P09_GUARD = P10_ROOT / "scripts" / "trr_p09" / "fixed_control_guarded_launch.py"
SOURCE_COMMIT = "26e08098135c47a5607aff915e8025b57a747e20"
P10_HEAD = "cab7d455c305aa01c3d3bfac225fcffa3e66640b"

STARTING_STATE = CANONICAL_ROOT / ".worktrees" / "TRR-0009" / "experiments" / "TRR-0009" / "training" / "run_v1" / "continued_fixed_readout" / "selected.safetensors"
PUBLIC_EMBEDDING = CANONICAL_ROOT / "outputs" / "TRR-0003" / "track_b" / "public_fit_v2" / "public_normalized_embeddings.safetensors"
B0_BINDING = P10_ROOT / "experiments" / "TRR-P09" / "setup" / "b0-immutable-loader-binding-r1.json"
FIXED_DIAGNOSTIC = P10_ROOT / "experiments" / "TRR-P09" / "setup" / "fixed-diagnostic-binding-r1.json"
# P09's signed digest is the original public TRR-0010 supplement.  The later
# P09 worktree copy includes a metadata amendment and has hash 7150c2..., so
# it is not interchangeable with the frozen CLI gate 760adf...f7.
SELECTION_SUPPLEMENT = CANONICAL_ROOT / ".worktrees" / "TRR-0010" / "experiments" / "TRR-0010" / "planning" / "stage1_template_compatible_selection_supplement_r1.json"
SHARED_CONTRACT = TASK_EXPERIMENT / "shared_contract_v1.json"
VALIDATION_MANIFEST = TASK_EXPERIMENT / "preparation" / "public_validation_r1" / "validation_manifest.json"
VALIDATION_ROWS = TASK_EXPERIMENT / "preparation" / "public_validation_r1" / "validation_rows.json"
CONFIGURATION_DRY_RUN_RECEIPT = TASK_EXPERIMENT / "execution" / "configuration_dry_run_r2" / "configuration_dry_run_receipt.json"
SCHEDULE_ROOT = TASK_EXPERIMENT / "preparation" / "schedules_r4"
B1_CAPTURE_ROOT = TASK_EXPERIMENT / "capture"

START_SHA = "5cada4a3d04bb5477eaf0be25ed8d8ac25a89283223e9ba14b18fa10416bee14"
EMBEDDING_SHA = "ad4201381ec062f0ece1ed007f6a003503e57ef4384271361059f0cc781fdcf1"
B0_SCHEDULE_SHA = "abe3fbb72c6639316243ec522c2126952ceac17096c04f940c3427fe00966483"
B1_SCHEDULE_SHA = "f334697433746814b672e91f1a08cdca8e4810511b6d6648a0d73432e9072e61"
DIAGNOSTIC_SHA = "8b899bc168b55570e4cfd526ff4e8656b70050e840386cba9390ad6adb1f54c3"
SELECTION_SHA = "760adf507438111b03c7308740e6666b0961088e6fc8868f999240d58d2814f7"
CHECKPOINT_GRID = (0, 1000, 2000, 4000, 8000, 12000, 13000)
VALID_POSITIONS = {"B0": 124371, "B1": 1243710}
MODEL_IDS = {"B0": "TRR-0012/current_fixed_replication_1", "B1": "TRR-0012/expanded_fixed_replication_1"}
ARM_NAMES = {"B0": "current_fixed_replication_1", "B1": "expanded_fixed_replication_1"}
GIB = 2**30

P09_SOURCE_HASHES = {
    "scripts/trr_p09/fixed_control_cli.py": "707848724cff8bdaadbbd287cf61b2eafebf32bd9f40add7645e25bf2800234e",
    "scripts/trr_p09/fixed_control_caller.py": "13410a03b825467479001fa97112ed05c9784b9ad64dd46880bcd8076de53f6b",
    "scripts/trr_p09/fixed_control_runner.py": "fde3db3c9655afca1678f3d2d0641fba5d4130a73fdb77cc2287a38b1741a98e",
    "scripts/trr_p09/b0_immutable_loader.py": "915c53781c24d997eccc7311553c2d80ff0d7e14688d22bb64c41db2f5e43f70",
    "scripts/trr_p09/prepare_streamed_bank.py": "e76a27f94328648f8fe3da6c6fae48fffd27006b213f9b70b976bcb5f4077c4b",
    "scripts/trr_p09/public_validation_loader.py": "3705ff2a58d34512c7f30d9eb2657206fca3621eb9f6fb685b82c12ee008cd0f",
    "scripts/trr_p09/fixed_control_guarded_launch.py": "80779d154a6735c84451f5a978b22820bd76dc1d157a1422b114f0a2b1acd311",
    "scripts/trr_p09/resource_watchdog.py": "07691ea3d8c86e2bea56a85a304ac6952ad14272a146af9cea5f0e5401a42b22",
    "src/token_reconstruction/trr_p09_fixed_control_adapter.py": "aa6a8b21fcc03d55583e69940b722bc9a1e4f4caa05e888f7dab8bf02d3e1a41",
    "src/token_reconstruction/trr0007_positionwise.py": "89fcd036a57407c8c49294aa5fd15ccef46f02b6fabcc35435d097f1213de485",
}


class LauncherError(RuntimeError):
    """Raised for a changed, missing, or unauthorized fit input."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, *, label: str, reject_tmp: bool = True) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise LauncherError(f"{label} is a symlink: {raw}")
    resolved = raw.resolve()
    if reject_tmp and any(part.lower() == "tmp" for part in resolved.parts):
        raise LauncherError(f"{label} may not use temporary storage: {resolved}")
    if not resolved.is_file() or resolved.is_symlink():
        raise LauncherError(f"{label} is not a regular file: {resolved}")
    return resolved


def record(path: Path, *, label: str, reject_tmp: bool = True) -> dict[str, Any]:
    path = regular_file(path, label=label, reject_tmp=reject_tmp)
    return {"label": label, "path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    path = regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LauncherError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"{label} is not a JSON object: {path}")
    return value


def write_create_only(path: Path, value: Any) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise LauncherError(f"create-only artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def git_head(root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LauncherError(f"cannot resolve git head for {root}") from exc
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise LauncherError(f"malformed git head for {root}: {value!r}")
    return value


def source_bindings() -> dict[str, Any]:
    if not P10_ROOT.is_dir() or P10_ROOT.is_symlink():
        raise LauncherError(f"persistent P09 source worktree is unavailable: {P10_ROOT}")
    actual_head = git_head(P10_ROOT)
    if actual_head != P10_HEAD:
        raise LauncherError(f"P10 source head changed: expected {P10_HEAD}, got {actual_head}")
    files: dict[str, Any] = {}
    for relative, expected in P09_SOURCE_HASHES.items():
        path = regular_file(P10_ROOT / relative, label=f"P09 source {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise LauncherError(f"P09 source hash changed for {relative}: {actual} != {expected}")
        files[relative] = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": actual}
    task_files: dict[str, Any] = {}
    for relative in ("scripts/trr0012/run_fixed_pair.py", "scripts/trr0012/plan_fixed_control.py", "scripts/trr0012/fixed_control_wrapper.py"):
        path = regular_file(TASK_ROOT / relative, label=f"task source {relative}")
        task_files[relative] = {"path": str(path), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    return {"authoritative_p09_commit": SOURCE_COMMIT, "actual_source_worktree_head": actual_head, "p09_files": files, "task_files": task_files, "imported_cli": str(P09_CLI), "imported_guard": str(P09_GUARD)}


def resolve_b1_manifest(argument: Path | None) -> Path:
    if argument is not None:
        return regular_file(argument, label="B1 captured manifest")
    candidates = [
        B1_CAPTURE_ROOT / "b1_activation_date07" / "bank_manifest.json",
        B1_CAPTURE_ROOT / "b1_activation_date07" / "bank_manifest-qualified-r1.json",
        B1_CAPTURE_ROOT / "activation_date07" / "bank_manifest.json",
        B1_CAPTURE_ROOT / "full_capture" / "bank_manifest.json",
        B1_CAPTURE_ROOT / "final_b1_manifest.json",
    ]
    candidates = [path for path in candidates if path.is_file() and not path.is_symlink()]
    if len(candidates) == 1:
        return regular_file(candidates[0], label="B1 captured manifest")
    if not candidates:
        raise LauncherError("B1 captured fit-addition manifest is absent; pass --b1-manifest <bank_manifest.json>")
    raise LauncherError("multiple B1 manifest candidates exist; pass --b1-manifest explicitly: " + ", ".join(map(str, candidates)))


def validate_b1_manifest(path: Path) -> dict[str, Any]:
    value = load_json(path, label="B1 captured manifest")
    schema = str(value.get("schema", ""))
    if not schema.startswith("token-reconstruction.trr-p09-streamed-public-bank"):
        raise LauncherError(f"B1 manifest schema is not native streamed-bank: {schema!r}")
    bank = value.get("bank")
    geometry = value.get("geometry")
    if not isinstance(bank, Mapping) or not isinstance(geometry, Mapping):
        raise LauncherError("B1 manifest lacks bank/geometry objects")
    if int(bank.get("record_count", -1)) != 10800 or int(bank.get("expanded_row_origin", -1)) != 1200:
        raise LauncherError("B1 manifest must bind 10,800 addition rows at expanded origin 1,200")
    if int(geometry.get("sequence_tokens", -1)) != 192 or int(geometry.get("hidden_size", -1)) != 2048:
        raise LauncherError("B1 manifest geometry differs from the signed P09 contract")
    shards = value.get("sharding", {}).get("shards") if isinstance(value.get("sharding"), Mapping) else None
    if not isinstance(shards, list) or not shards:
        raise LauncherError("B1 manifest has no streamed shard list")
    sidecars: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, Mapping) or not isinstance(shard.get("sidecar"), Mapping):
            raise LauncherError(f"B1 shard {index} descriptor is malformed")
        side = shard["sidecar"]
        raw_path = Path(str(side.get("path", "")))
        side_path = raw_path if raw_path.is_absolute() else path.parent / raw_path
        side_record = record(side_path, label=f"B1 shard {index} sidecar")
        for key in ("bytes", "sha256"):
            if key in side and str(side[key]) != str(side_record[key]):
                raise LauncherError(f"B1 shard {index} sidecar {key} differs from manifest")
        sidecars.append(side_record)
    return {"manifest": record(path, label="B1 fit bank manifest"), "status": value.get("status"), "schema": schema, "bank": {"record_count": 10800, "expanded_row_origin": 1200}, "geometry": {key: geometry.get(key) for key in ("sequence_tokens", "hidden_size", "hidden_dtype", "loader_batch_records", "shard_records")}, "shard_count": len(shards), "sidecars": sidecars}


def schedule(bank: str) -> tuple[Path, str, str]:
    path = SCHEDULE_ROOT / f"schedule-{bank.lower()}-seed4010.safetensors"
    expected = B0_SCHEDULE_SHA if bank == "B0" else B1_SCHEDULE_SHA
    semantic = "9aaad9c030f2f9b801f91f956c97f858966580edd21229cebb350dcde358f2c3" if bank == "B0" else "8acdb2c4f8e5afba546ad01cbe0adae340eb841c8e3322c919e976434d29ede5"
    return regular_file(path, label=f"{bank} common schedule"), expected, semantic


def fit_binding(b1_path: Path) -> tuple[dict[str, Any], str]:
    payload = {"prefix": record(B0_BINDING, label="B0 fit bank manifest"), "addition": record(b1_path, label="B1 fit bank manifest")}
    return payload, canonical_digest(payload)


def b0_artifact_bindings() -> dict[str, Any]:
    """Bind every public artifact named by the immutable B0 descriptor."""

    binding = load_json(B0_BINDING, label="B0 immutable fit binding")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise LauncherError("B0 immutable binding lacks its artifact descriptors")
    result: dict[str, Any] = {}
    for role in ("payload", "records", "published_manifest", "position_contract", "corpus_plan"):
        descriptor = artifacts.get(role)
        if not isinstance(descriptor, Mapping):
            raise LauncherError(f"B0 immutable binding lacks artifact {role}")
        raw_path = descriptor.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise LauncherError(f"B0 artifact {role} has no path")
        current = record(Path(raw_path), label=f"B0 {role}", reject_tmp=False)
        for key in ("bytes", "sha256"):
            if str(current[key]) != str(descriptor.get(key)):
                raise LauncherError(f"B0 {role} differs from its immutable descriptor: {key}")
        result[role] = current
    return result


def gate_record(path: Path, *, label: str, kind: str) -> dict[str, Any]:
    value = load_json(path, label=label)
    status = str(value.get("status", "")).upper()
    if kind == "qualification":
        # A generic capture or synthetic PASS is not a largest-cell fixed-fit
        # qualification.  Require the delegated native P09 discarded-probe
        # receipt and its explicit no-contender contract.
        if status != "QUALIFICATION_PASS":
            raise LauncherError(f"{label} does not carry an accepted qualification status: {status!r}")
        if value.get("schema") != "token-reconstruction.trr0012-fixed-native-qualification.v1":
            raise LauncherError(f"{label} is not the fixed-native qualification schema")
        if value.get("mode") != "fixed_public_readout_native_runner" or value.get("bank") != "B1":
            raise LauncherError(f"{label} is not the B1 fixed-readout qualification")
        if value.get("discarded_updates") is not True or value.get("all_updates_discarded") is not True:
            raise LauncherError(f"{label} does not prove all probe updates were discarded")
        if value.get("contender_selection") is not False or value.get("retained_fitted_arm") is not False:
            raise LauncherError(f"{label} permits a retained contender")
        if value.get("checkpoint_callback") is not None or value.get("checkpoint_written") is not False:
            raise LauncherError(f"{label} permits checkpoint retention")
        training = value.get("training_contract")
        if not isinstance(training, Mapping):
            raise LauncherError(f"{label} lacks its fixed training contract")
        if int(training.get("probe_steps", -1)) <= 0 or int(training.get("probe_steps", -1)) > 8 or int(training.get("full_fit_steps", -1)) != 13000:
            raise LauncherError(f"{label} probe/full step binding is not the coordinator contract")
        if training.get("compute_base_logits") is not True or training.get("selection_metric") != "domain_balanced_token_accuracy":
            raise LauncherError(f"{label} does not bind full-vocabulary fixed-readout settings")
        if not isinstance(value.get("inputs"), Mapping) or not isinstance(value["inputs"].get("binding"), Mapping) or not isinstance(value["inputs"].get("preflight"), Mapping):
            raise LauncherError(f"{label} lacks fixed input-binding and live-preflight evidence")
    if kind == "agreement" and not (value.get("fit_authorized") is True or ("STAGE3" in status and "AGREE" in status)):
        raise LauncherError(f"{label} is not an accepted stage3 agreement: {status!r}")
    if value.get("truth_opened") is True or value.get("evaluation_truth_opened") is True:
        raise LauncherError(f"{label} crosses the evaluation-truth boundary")
    return {"file": record(path, label=label), "status": status, "schema": value.get("schema"), "fit_authorized": value.get("fit_authorized"), "qualification_pass": value.get("qualification_pass")}


def make_stage3_agreement(*, fit_release: Path, run_root: Path, contract: dict[str, Any], fit_sha: str, b1: dict[str, Any]) -> Path:
    release = load_json(fit_release, label="root fit release")
    status = str(release.get("status", release.get("release", ""))).upper()
    if status not in {"AUTHORIZED", "RELEASED", "FIT_AUTHORIZED", "SUBSTANTIVE_FIT_AUTHORIZED"}:
        raise LauncherError(f"root fit release is not authorized: {status!r}")
    if release.get("substantive_fit") is not True:
        raise LauncherError("root fit release must explicitly set substantive_fit=true")
    if release.get("paid_compute") is True or release.get("evaluation_truth_access") is True:
        raise LauncherError("root fit release crosses a prohibited boundary")
    declared_contract = release.get("shared_contract_sha256", release.get("contract_sha256"))
    if declared_contract is not None and str(declared_contract) != contract["sha256"]:
        raise LauncherError("root fit release is bound to a different shared contract")
    payload = {"schema": "token-reconstruction.trr0012-stage3-agreement.v1", "task_id": TASK_ID, "status": "TRR0012_STAGE3_AGREED_FIT_AUTHORIZED", "fit_authorized": True, "truth_opened": False, "evaluation_truth_opened": False, "created_utc": utc_now(), "root_fit_release": record(fit_release, label="root fit release"), "shared_contract": contract, "combined_fit_binding_sha256": fit_sha, "b1_manifest": b1["manifest"], "new_model_ids": list(MODEL_IDS.values()), "serialized_method_id": "continued_fixed_readout", "source_commit": SOURCE_COMMIT}
    return write_create_only(run_root / "gates" / "stage3-agreement.json", payload)


def child_argv(*, bank: str, b1_path: Path, fit_sha: str, stage3: Path, qualification: Path, watchdog: Path, output: Path) -> list[str]:
    sched, sched_sha, _ = schedule(bank)
    env = ["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "TOKENIZERS_PARALLELISM=false", "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1", f"PYTHONPATH={P10_ROOT}:{P10_ROOT / 'src'}"]
    return env + [sys.executable, str(P09_CLI), "--output-root", str(output), "--common-schedule", str(sched), "--expected-schedule-sha256", sched_sha, "--schedule-bank", bank, "--repository-root", str(P10_ROOT), "--fit-prefix-manifest", str(B0_BINDING), "--fit-addition-manifest", str(b1_path), "--validation-manifest", str(VALIDATION_MANIFEST), "--validation-rows", str(VALIDATION_ROWS), "--fixed-diagnostic-binding", str(FIXED_DIAGNOSTIC), "--expected-fixed-diagnostic-sha256", DIAGNOSTIC_SHA, "--embedding", str(PUBLIC_EMBEDDING), "--state", str(STARTING_STATE), "--selection-supplement", str(SELECTION_SUPPLEMENT), "--stage3-agreement", str(stage3), "--qualification-receipt", str(qualification), "--watchdog-receipt", str(watchdog), "--device", "cuda", "--steps", "13000", "--seed", "4010", "--learning-rate", "0.0002", "--weight-decay", "0.0", "--gradient-clip-norm", "1.0", "--validation-every", "1000", "--validation-sequence-tokens", "128", "--deadline-seconds", "7200", "--maximum-host-rss-gib", "12", "--minimum-host-available-gib", "8", "--minimum-gpu-free-gib", "2", "--maximum-gpu-reserved-gib", "8", "--expected-fit-binding-sha256", fit_sha]


def build_binding(*, run_id: str, b1_arg: Path | None, stage3_arg: Path | None, qualification_arg: Path | None, fit_release_arg: Path | None) -> dict[str, Any]:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in run_id):
        raise LauncherError("run-id must contain only letters, digits, '.', '_' or '-'")
    b1_path = resolve_b1_manifest(b1_arg)
    b1 = validate_b1_manifest(b1_path)
    source = source_bindings()
    contract = record(SHARED_CONTRACT, label="TRR-0012 shared contract")
    start = record(STARTING_STATE, label="exact common starting state")
    if start["sha256"] != START_SHA:
        raise LauncherError("common starting state SHA differs from the sole approved 5cada... file")
    embedding = record(PUBLIC_EMBEDDING, label="public normalized embedding")
    if embedding["sha256"] != EMBEDDING_SHA:
        raise LauncherError("public embedding SHA differs from the signed fixed E")
    b0 = record(B0_BINDING, label="B0 immutable fit binding")
    b0_artifacts = b0_artifact_bindings()
    validation_manifest = record(VALIDATION_MANIFEST, label="public validation manifest")
    validation_rows = record(VALIDATION_ROWS, label="public validation rows")
    configuration_dry_run = record(CONFIGURATION_DRY_RUN_RECEIPT, label="native configuration dry-run receipt")
    configuration_value = load_json(CONFIGURATION_DRY_RUN_RECEIPT, label="native configuration dry-run receipt")
    if configuration_value.get("status") != "PASS_B0_AND_B1_NATIVE_CPU_CONFIGURATION":
        raise LauncherError("native configuration dry-run receipt is not a paired PASS")
    diagnostic = record(FIXED_DIAGNOSTIC, label="fixed diagnostic binding")
    if diagnostic["sha256"] != DIAGNOSTIC_SHA:
        raise LauncherError("fixed diagnostic binding SHA differs from native P09")
    supplement = record(SELECTION_SUPPLEMENT, label="signed selection supplement")
    if supplement["sha256"] != SELECTION_SHA:
        raise LauncherError("signed selection supplement is not the original P09 760adf...f7 file")
    fit_asset, fit_sha = fit_binding(b1_path)
    run_root = TASK_EXPERIMENT / "execution" / "fixed_fits" / run_id
    if run_root.exists() or run_root.is_symlink():
        raise LauncherError(f"run root is create-only; choose a fresh run-id: {run_root}")
    run_root.mkdir(parents=True)
    gates: dict[str, Any] = {"stage3": None, "qualification": None, "root_fit_release": None}
    blockers: list[str] = []
    if stage3_arg is not None:
        stage3 = regular_file(stage3_arg, label="stage3 agreement")
        gates["stage3"] = gate_record(stage3, label="stage3 agreement", kind="agreement")
    elif fit_release_arg is not None:
        release = regular_file(fit_release_arg, label="root fit release")
        gates["root_fit_release"] = record(release, label="root fit release")
        stage3 = make_stage3_agreement(fit_release=release, run_root=run_root, contract=contract, fit_sha=fit_sha, b1=b1)
        gates["stage3"] = gate_record(stage3, label="stage3 agreement", kind="agreement")
    else:
        stage3 = run_root / "gates" / "stage3-agreement.json"
        blockers.append("root fit release/stage3 agreement is not supplied")
    if qualification_arg is not None:
        qualification = regular_file(qualification_arg, label="largest-cell qualification")
        gates["qualification"] = gate_record(qualification, label="largest-cell qualification", kind="qualification")
    else:
        qualification = run_root / "gates" / "largest-cell-qualification.json"
        blockers.append("largest-cell qualification receipt is not supplied")
    schedules: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    for bank in ("B0", "B1"):
        sched, sched_sha, semantic = schedule(bank)
        schedules[bank] = {"file": record(sched, label=f"{bank} common schedule"), "expected_sha256": sched_sha, "semantic_sha256": semantic, "valid_positions": VALID_POSITIONS[bank]}
        output = TASK_OUTPUT / "fixed_fits" / run_id / ARM_NAMES[bank]
        watchdog_root = run_root / "watchdog" / bank
        armed = run_root / "receipts" / f"watchdog-{bank.lower()}-armed.json"
        if blockers:
            child = guard = None
        else:
            child = child_argv(bank=bank, b1_path=b1_path, fit_sha=fit_sha, stage3=stage3, qualification=qualification, watchdog=armed, output=output)
            # The guard imports resource_watchdog from the P10 source tree.
            # Bind that import path explicitly so the hashed module recorded in
            # source_bindings() is the module executed by the real process.
            guard = ["env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "PYTHONPATH=" + str(P10_ROOT) + ":" + str(P10_ROOT / "src"), sys.executable, str(P09_GUARD), "--output-root", str(watchdog_root), "--child-output-root", str(output), "--timeout-seconds", "7200", "--poll-seconds", "0.5", "--max-rss-bytes", str(12 * GIB), "--min-available-bytes", str(8 * GIB), "--min-disk-free-bytes", str(20 * GIB), "--max-output-bytes", str(5 * GIB), "--kill-grace-seconds", "2", "--cwd", str(P10_ROOT), "--label", f"TRR-0012-{bank}", "--watchdog-source", str(P09_GUARD), "--", *child]
        arms[bank] = {"bank": bank, "model_id": MODEL_IDS[bank], "output_root": str(output), "watchdog_root": str(watchdog_root), "watchdog_receipt": str(armed), "child_argv": child, "guard_argv": guard}
    plan = {"schema": "token-reconstruction.trr0012-native-fixed-pair-run-plan.v1", "task_id": TASK_ID, "status": "READY_FOR_ROOT_FIT_RELEASE_AND_QUALIFICATION" if blockers else "READY_TO_ARM", "created_utc": utc_now(), "run_id": run_id, "run_root": str(run_root), "source": source, "contract": contract, "starting_state": start, "public_embedding": embedding, "fit_manifests": fit_asset, "combined_fit_binding_sha256": fit_sha, "b1_capture": b1, "b0_binding": b0, "b0_artifacts": b0_artifacts, "validation": {"manifest": validation_manifest, "rows": validation_rows}, "configuration_dry_run": configuration_dry_run, "fixed_diagnostic": diagnostic, "signed_selection_supplement": supplement, "gates": gates, "schedules": schedules, "arms": arms, "training_contract": {"seed": 4010, "steps": 13000, "record_batch_size": 8, "position_budget": 512, "train_sequence_tokens": 192, "validation_sequence_tokens": 128, "validation_every_metadata": 1000, "checkpoint_grid": list(CHECKPOINT_GRID), "selection_metric": "domain_balanced_token_accuracy", "selection_rule": "earliest strict maximum over the seven grid validations, including step 0", "optimizer": "AdamW foreach=False", "learning_rate": 0.0002, "weight_decay": 0.0, "gradient_clip_norm": 1.0, "scheduler": "CosineAnnealingLR(T_max=13000)", "serialized_method_id": "continued_fixed_readout", "task_model_ids": list(MODEL_IDS.values())}, "resource_guard": {"timeout_seconds": 7200, "max_group_rss_bytes": 12 * GIB, "min_host_available_bytes": 8 * GIB, "min_gpu_free_bytes": 2 * GIB, "max_gpu_reserved_bytes": 8 * GIB, "min_disk_free_bytes": 20 * GIB, "max_output_bytes": 5 * GIB, "process_group_fail_closed": True, "env_threads": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}}, "blockers": blockers, "truth_boundary": {"model_loaded": False, "public_embedding_loaded": False, "fit_started": False, "evaluation_truth_opened": False, "P03_holdout_touched": False}}
    write_create_only(run_root / "run_plan.json", plan)
    return plan


def arm_watchdog(plan: Mapping[str, Any], bank: str) -> Path:
    arm = plan["arms"][bank]
    child = arm.get("child_argv")
    guard = arm.get("guard_argv")
    if not isinstance(child, list) or not isinstance(guard, list):
        raise LauncherError("run plan has no complete native command; gates are still pending")
    path = Path(str(arm["watchdog_receipt"]))
    payload = {"schema": "token-reconstruction.trr0012-watchdog-armed.v1", "task_id": TASK_ID, "status": "ARMED", "created_utc": utc_now(), "bank": bank, "model_id": arm["model_id"], "command": guard, "child_command": child, "output_root": arm["output_root"], "watchdog_root": arm["watchdog_root"], "enforcement": {"process_group_fail_closed": True, "disk_and_output_enforced_during_live_process": True, "kill_grace_seconds": 2}, "truth_opened": False, "evaluation_truth_opened": False}
    return write_create_only(path, payload)



def _verify_frozen_record(expected: Mapping[str, Any], *, label: str, reject_tmp: bool = True) -> dict[str, Any]:
    """Re-read one plan-bound file and require byte/hash identity."""

    raw_path = expected.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise LauncherError(f"{label} has no frozen path")
    actual = record(Path(raw_path), label=label, reject_tmp=reject_tmp)
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise LauncherError(
                f"{label} changed after plan construction at {raw_path}: "
                f"{key} {actual.get(key)!r} != {expected.get(key)!r}"
            )
    return actual


def verify_frozen_inputs(plan: Mapping[str, Any]) -> None:
    """Fail closed if any source, gate, or public fit input changed.

    This runs immediately before every native child launch.  The plan is
    create-only, but checking the current bytes again closes the gap between
    plan construction and process launch and proves that the guard imports
    are still the hashed P09 files recorded in the plan.
    """

    expected_source = plan.get("source")
    actual_source = source_bindings()
    if actual_source != expected_source:
        raise LauncherError("task/P09 source bindings changed after plan construction")

    for key, label in (
        ("contract", "TRR-0012 shared contract"),
        ("starting_state", "exact common starting state"),
        ("public_embedding", "public normalized embedding"),
        ("b0_binding", "B0 immutable fit binding"),
        ("fixed_diagnostic", "fixed diagnostic binding"),
        ("signed_selection_supplement", "signed selection supplement"),
        ("configuration_dry_run", "native configuration dry-run receipt"),
    ):
        expected = plan.get(key)
        if not isinstance(expected, Mapping):
            raise LauncherError(f"plan lacks frozen {key} record")
        _verify_frozen_record(expected, label=label)
    expected_b0_artifacts = plan.get("b0_artifacts")
    if not isinstance(expected_b0_artifacts, Mapping):
        raise LauncherError("plan lacks nested B0 artifact records")
    for role, expected in expected_b0_artifacts.items():
        if not isinstance(expected, Mapping):
            raise LauncherError(f"B0 artifact record is malformed: {role}")
        if role == "position_contract":
            # The unmodified historical loader binding names this read-only
            # metadata in /tmp. Its exact bytes also survive durably. This is
            # not a model, bank payload, output, or deployment dependency.
            durable = record(TASK_EXPERIMENT / "capture" / "b0-position-contract-r1.json", label="durable B0 position contract")
            if any(str(durable[key]) != str(expected[key]) for key in ("bytes", "sha256")):
                raise LauncherError("B0 position contract differs from its durable exact-byte copy")
            _verify_frozen_record(expected, label=f"B0 {role}", reject_tmp=False)
        else:
            _verify_frozen_record(expected, label=f"B0 {role}")
    validation = plan.get("validation")
    if not isinstance(validation, Mapping):
        raise LauncherError("plan lacks frozen validation records")
    for key, label in (("manifest", "public validation manifest"), ("rows", "public validation rows")):
        expected = validation.get(key)
        if not isinstance(expected, Mapping):
            raise LauncherError(f"plan lacks frozen validation {key} record")
        _verify_frozen_record(expected, label=label)

    fit_manifests = plan.get("fit_manifests")
    if not isinstance(fit_manifests, Mapping):
        raise LauncherError("plan lacks frozen fit-manifest records")
    prefix = fit_manifests.get("prefix")
    addition = fit_manifests.get("addition")
    if not isinstance(prefix, Mapping) or not isinstance(addition, Mapping):
        raise LauncherError("plan lacks frozen B0/B1 fit-manifest records")
    _verify_frozen_record(prefix, label="B0 fit bank manifest")
    b1_path = Path(str(addition.get("path")))
    _verify_frozen_record(addition, label="B1 fit bank manifest")
    b1_now = validate_b1_manifest(b1_path)
    expected_b1 = plan.get("b1_capture")
    if not isinstance(expected_b1, Mapping):
        raise LauncherError("plan lacks frozen B1 capture metadata")
    expected_manifest = expected_b1.get("manifest")
    actual_manifest = b1_now.get("manifest")
    if not isinstance(expected_manifest, Mapping) or not isinstance(actual_manifest, Mapping):
        raise LauncherError("B1 capture metadata lacks manifest records")
    _verify_frozen_record(expected_manifest, label="B1 captured bank manifest")
    expected_sidecars = expected_b1.get("sidecars")
    actual_sidecars = b1_now.get("sidecars")
    if not isinstance(expected_sidecars, list) or not isinstance(actual_sidecars, list) or len(expected_sidecars) != len(actual_sidecars):
        raise LauncherError("B1 shard sidecar inventory changed after plan construction")
    for index, expected_sidecar in enumerate(expected_sidecars):
        if not isinstance(expected_sidecar, Mapping):
            raise LauncherError(f"B1 shard {index} frozen sidecar record is malformed")
        _verify_frozen_record(expected_sidecar, label=f"B1 shard {index} sidecar")
        actual_sidecar = actual_sidecars[index]
        if not isinstance(actual_sidecar, Mapping) or any(str(actual_sidecar.get(key)) != str(expected_sidecar.get(key)) for key in ("path", "bytes", "sha256")):
            raise LauncherError(f"B1 shard {index} sidecar changed after plan construction")
    expected_fit, expected_fit_sha = fit_binding(b1_path)
    if expected_fit != fit_manifests or expected_fit_sha != str(plan.get("combined_fit_binding_sha256")):
        raise LauncherError("combined B0+B1 fit binding changed after plan construction")

    schedules = plan.get("schedules")
    if not isinstance(schedules, Mapping):
        raise LauncherError("plan lacks frozen common schedules")
    for bank in ("B0", "B1"):
        expected_schedule = schedules.get(bank)
        if not isinstance(expected_schedule, Mapping) or not isinstance(expected_schedule.get("file"), Mapping):
            raise LauncherError(f"plan lacks frozen {bank} schedule")
        current_path, current_sha, current_semantic = schedule(bank)
        _verify_frozen_record(expected_schedule["file"], label=f"{bank} common schedule")
        if str(expected_schedule["expected_sha256"]) != current_sha or str(expected_schedule["semantic_sha256"]) != current_semantic or str(expected_schedule["file"]["path"]) != str(current_path.resolve()):
            raise LauncherError(f"{bank} schedule binding changed after plan construction")

    gates = plan.get("gates")
    if not isinstance(gates, Mapping):
        raise LauncherError("plan lacks fit gate records")
    for key, kind, label in (("stage3", "agreement", "stage3 agreement"), ("qualification", "qualification", "largest-cell qualification")):
        expected_gate = gates.get(key)
        if not isinstance(expected_gate, Mapping):
            raise LauncherError(f"plan lacks {label}")
        expected_file = expected_gate.get("file")
        if not isinstance(expected_file, Mapping):
            raise LauncherError(f"plan lacks frozen {label} file record")
        _verify_frozen_record(expected_file, label=label)
        actual_gate = gate_record(Path(str(expected_file["path"])), label=label, kind=kind)
        actual_gate_file = actual_gate.get("file")
        if not isinstance(actual_gate_file, Mapping) or any(str(actual_gate_file.get(field)) != str(expected_file.get(field)) for field in ("path", "bytes", "sha256")):
            raise LauncherError(f"{label} changed after plan construction")
    root_release = gates.get("root_fit_release")
    if isinstance(root_release, Mapping):
        _verify_frozen_record(root_release, label="root fit release")


def copy_exclusive(source: Path, destination: Path) -> dict[str, Any]:
    source = regular_file(source, label="selected checkpoint")
    destination = destination.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise LauncherError(f"selected-state destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    return record(destination, label="selected decoder state copy")


def audit_output(plan: Mapping[str, Any], bank: str) -> dict[str, Any]:
    arm = plan["arms"][bank]
    receipt_path = regular_file(Path(str(arm["output_root"])) / "receipt.json", label=f"{bank} native fit receipt")
    receipt = load_json(receipt_path, label=f"{bank} native fit receipt")
    if receipt.get("status") != "COMPLETED":
        raise LauncherError(f"{bank} native receipt is not COMPLETED: {receipt.get('status')!r}")
    state = receipt.get("state")
    if not isinstance(state, Mapping) or state.get("method_id") != "continued_fixed_readout":
        raise LauncherError(f"{bank} native receipt serialized method identity changed")
    bindings = state.get("checkpoint_state_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(CHECKPOINT_GRID):
        raise LauncherError(f"{bank} native receipt does not contain exactly seven checkpoint bindings")
    output_root = Path(str(arm["output_root"])).resolve()
    by_step: dict[int, Mapping[str, Any]] = {}
    checkpoints: list[dict[str, Any]] = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or not isinstance(binding.get("fitting_diagnostics"), Mapping):
            raise LauncherError(f"{bank} checkpoint binding is malformed")
        step = int(binding["fitting_diagnostics"]["step"])
        if step in by_step or step not in CHECKPOINT_GRID:
            raise LauncherError(f"{bank} checkpoint step is outside/duplicated in the exact grid: {step}")
        checkpoint = binding.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise LauncherError(f"{bank} checkpoint file binding is absent at step {step}")
        cp_path = regular_file(Path(str(checkpoint.get("path"))), label=f"{bank} checkpoint step {step}")
        try:
            cp_path.relative_to(output_root)
        except ValueError as exc:
            raise LauncherError(f"{bank} checkpoint escaped its output root: {cp_path}") from exc
        actual = record(cp_path, label=f"{bank} checkpoint step {step}")
        if int(checkpoint.get("bytes", -1)) != actual["bytes"] or str(checkpoint.get("sha256")) != actual["sha256"]:
            raise LauncherError(f"{bank} checkpoint binding changed at step {step}")
        by_step[step] = binding
        checkpoints.append({"step": step, "file": actual, "logical_state_sha256": binding.get("state_sha256")})
    if set(by_step) != set(CHECKPOINT_GRID):
        raise LauncherError(f"{bank} checkpoint grid differs: {sorted(by_step)}")
    selected_step = int(receipt.get("selection", {}).get("selected_step", -1))
    if selected_step not in by_step:
        raise LauncherError(f"{bank} selected step is not on the exact grid: {selected_step}")
    selected_sha = str(state.get("selected_state_sha256", ""))
    if not selected_sha or selected_sha != str(by_step[selected_step].get("state_sha256", "")):
        raise LauncherError(f"{bank} selected logical state digest does not match checkpoint binding")
    selected = copy_exclusive(Path(str(by_step[selected_step]["checkpoint"]["path"])), output_root / "selected.safetensors")
    watchdog_root = Path(str(arm["watchdog_root"])).resolve()
    guard_path = regular_file(watchdog_root / "resource_guard.json", label=f"{bank} watchdog resource guard")
    guard = load_json(guard_path, label=f"{bank} watchdog resource guard")
    if guard.get("status") != "PASS" or guard.get("process_group_fail_closed") is not True:
        raise LauncherError(f"{bank} watchdog guard did not PASS fail-closed: {guard.get('status')!r}")
    binding = {"schema": "token-reconstruction.trr0012-selected-state-binding.v1", "task_id": TASK_ID, "status": "PASS_NATIVE_P09_CHECKPOINT_AUDIT", "bank": bank, "model_id": arm["model_id"], "serialized_method_id": "continued_fixed_readout", "selected_step": selected_step, "selected_state_sha256": selected_sha, "selected_state": selected, "native_receipt": record(receipt_path, label=f"{bank} native fit receipt"), "watchdog_guard": record(guard_path, label=f"{bank} watchdog resource guard"), "checkpoint_grid": checkpoints, "truth_opened": False, "evaluation_truth_opened": False}
    binding_path = output_root / "selected-state-binding.json"
    write_create_only(binding_path, binding)
    audit = {"schema": "token-reconstruction.trr0012-native-fixed-fit-audit.v1", "task_id": TASK_ID, "status": "PASS", "bank": bank, "model_id": arm["model_id"], "serialized_method_id": "continued_fixed_readout", "native_receipt": record(receipt_path, label=f"{bank} native fit receipt"), "selected_state_binding": record(binding_path, label=f"{bank} selected-state binding"), "checkpoint_grid": checkpoints, "selected_step": selected_step, "selected_state_sha256": selected_sha, "truth_opened": False, "evaluation_truth_opened": False}
    audit_path = output_root / "fit-audit.json"
    write_create_only(audit_path, audit)
    return {"status": "PASS", "selected_step": selected_step, "selected_state": selected, "audit": record(audit_path, label=f"{bank} fit audit"), "watchdog": record(guard_path, label=f"{bank} watchdog resource guard")}


def run_arm(plan: Mapping[str, Any], bank: str) -> dict[str, Any]:
    # Check once before creating the armed receipt and again immediately before
    # spawning the guarded native child.  The second check is the launch gate.
    verify_frozen_inputs(plan)
    armed = arm_watchdog(plan, bank)
    verify_frozen_inputs(plan)
    command = plan["arms"][bank]["guard_argv"]
    if not isinstance(command, list):
        raise LauncherError(f"{bank} guard command is incomplete")
    started = utc_now()
    completed = subprocess.run(command, cwd=str(TASK_ROOT), check=False)
    result: dict[str, Any] = {"schema": "token-reconstruction.trr0012-native-fixed-run-result.v1", "task_id": TASK_ID, "bank": bank, "model_id": plan["arms"][bank]["model_id"], "started_utc": started, "finished_utc": utc_now(), "wrapper_return_code": int(completed.returncode), "armed_watchdog": record(armed, label=f"{bank} armed watchdog")}
    result_path = Path(str(plan["run_root"])) / f"{bank.lower()}-run-result.json"
    if completed.returncode != 0:
        result["status"] = "FAILED_PRESERVED"
        write_create_only(result_path, result)
        raise LauncherError(f"{bank} native fit failed with guarded return code {completed.returncode}; output preserved")
    result["audit"] = audit_output(plan, bank)
    result["status"] = "PASS"
    write_create_only(result_path, result)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--b1-manifest", type=Path)
    parser.add_argument("--stage3-agreement", type=Path)
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--fit-release", type=Path)
    parser.add_argument("--arm", choices=("B0", "B1"))
    parser.add_argument("--pair", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    if bool(args.arm) == bool(args.pair):
        raise LauncherError("select exactly one of --arm B0|B1 or --pair")
    plan = build_binding(run_id=args.run_id, b1_arg=args.b1_manifest, stage3_arg=args.stage3_agreement, qualification_arg=args.qualification_receipt, fit_release_arg=args.fit_release)
    print(json.dumps({"status": plan["status"], "run_root": plan["run_root"], "combined_fit_binding_sha256": plan["combined_fit_binding_sha256"], "blockers": plan["blockers"], "arms": {bank: {"child_argv": arm["child_argv"], "guard_argv": arm["guard_argv"]} for bank, arm in plan["arms"].items()}}, indent=2, sort_keys=True))
    if args.mode == "plan":
        return 0
    if plan["blockers"]:
        raise LauncherError("fit execution is blocked: " + "; ".join(plan["blockers"]))
    results = []
    for bank in ("B0", "B1") if args.pair else (args.arm,):
        results.append(run_arm(plan, bank))
    write_create_only(Path(str(plan["run_root"])) / "pair-run-result.json", {"schema": "token-reconstruction.trr0012-native-fixed-pair-result.v1", "task_id": TASK_ID, "status": "PASS", "run_id": args.run_id, "results": results, "truth_opened": False, "evaluation_truth_opened": False})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LauncherError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"TRR-0012 fixed launcher refused: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
