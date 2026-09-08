#!/usr/bin/env python3
"""TRR-P11 metadata-only exclusion audit.

This module is a narrow continuation of the P10 identity union.  It reuses the
P10 generic metadata walker, but replaces its broad P04 walker with a strict
producer-aware loader, binds every previously opened aggregate panel to its
selection ledger, and provides one optional public-token H128 curator pass.

The curator pass may read only a public fit payload's ``token_ids`` and
``attention_mask`` tensors.  It never reads activations, model weights, source
text, labels, or evaluation truth, and it never serializes token values.
Coverage remains explicitly partial until every missing producer/row mapping is
resolved.  This module never selects new sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from scripts.trr_p10 import build_exclusion_audit as p10


TASK_ID = "TRR-P11"
SCHEMA = "token-reconstruction.trr-p11-identity-exclusion-audit.v1"
IDENTITY_UNION_SCHEMA = "token-reconstruction.trr-p11-identity-union.v1"
IDENTITY_UNION_PARTIAL_STATUS = "PARTIAL_IDENTITY_UNION_NO_SELECTION_RELEASE"
IDENTITY_UNION_COMPLETE_STATUS = "IDENTITY_UNION_COMPLETE_NO_PAYLOAD"
IDENTITY_UNION_FIELDS = frozenset({
    "record_id",
    "rendered_sha256",
    "tokenized_record_sha256",
    "h128_sequence_sha256",
    "h129_sequence_sha256",
    # TRR-0001's public 40-token rows use raw signed-int32 H40.  This is a
    # distinct namespace from TRR-0002's tensor-header/signed-int64 H40.
    "h40_sequence_sha256",
    "trr0002_active_token_ids_sha256",
    "trr0002_h40_token_ids_sha256",
    "source_index",
})
P04_PATH = "experiments/TRR-0006/coordination/p04_reservation_hashes.json"
P04_SHA256 = "98f8dfcab0977b4bcafa47d97a86a410ab37359b897b9b553746afa7df5c7904"
P04_PRODUCER_SHA256 = "a622db694e0efeac0eb8cfe4ab2004e0490fe21e0c03b5ac1e11de603f1b53eb"
P04_PRODUCER_COMMIT = "561fc5ce9913af63824b9e4ee9c22063b147df20"
P04_TARGET_PLAN_COMMIT = "1aefc307ebdd4cd5002ac6ac0cdc5a1fc696aa68"
P04_TARGET_PLAN_PATH = "experiments/TRR-P04/setup/evaluator_target_plan.json"
P04_TARGET_PLAN_SHA256 = "55f5cc5ecc90599d8983ea1fa23d81a5a062178fd5dde6c9d19359c9fcc54fc2"
P04_HELPER_OBJECTS: tuple[tuple[str, str, str], ...] = (
    ("scripts/trr_p04/prepare_panel.py", "f423ef596a718a7c8a8480e6211295b97bdfd806", "26c003fc37a80c549ca04ebbf0dd629ae09026fad5f4afc21af0adcca72db97f"),
    ("scripts/trr_p04/prepare_evaluator_target.py", "63d4016f2e555e8460ea566cf967237dc94fb0c3", "bbf4fe9ac443f49f11c12f4a2fd3a5e5eef2cd67bf46573384eb13ffb2cbbd04"),
    ("src/token_reconstruction/p04_training.py", "15d847c40469adbcbd1688014703b204848f880f", "f214dc02d4b3ba5854cd87174c61dd24f02a75a486ce7a297dd3d1d9cabe7486"),
)
P04_H128_IDENTITY_PATH = "experiments/TRR-P11/exclusions/p04_h128_identity_rows_r1.json"
P04_H128_IDENTITY_SHA256 = "01758b410dc11c487cabc0eee22160d32c3d72abb2e2d0202b57a60f8dab4f6b"
P04_H128_RECOVERY_PATH = "experiments/TRR-P11/exclusions/p04_h128_recovery_r2.json"
P04_H128_RECOVERY_SHA256 = "d5284aa0e8e643adbc9d15bc67f90c6733ca535bf69d8ae52606dde5c559a649"
P04_H128_RECOVERY_SCRIPT_SHA256 = "ff079eb11e4099dd337bef4096b644453f89697461b46266154a0a509417aff7"
P04_RECIPE_MIGRATION_PATH = "experiments/TRR-P11/exclusions/p04_recipe_migration_r1.json"
P04_RECIPE_MIGRATION_SHA256 = "7b6b26f6ae6f94f5c34b96647d6df6eb73be39b911f52ee8ef7a2f10847c4a64"
P04_PERSISTENT_RECOVERY_SCRIPT_SHA256 = "b92cf740472894f568725df36801511a824b683aa4d1761d0e71d131cf8da91c"
P04_PERSISTENT_SELECTION_PATH = "experiments/TRR-P11/exclusions/p04_public_selection_r2.json"
P04_PERSISTENT_SELECTION_SHA256 = "05f941e0dbcf29ea3efc47c7bc8abb3a7146a266eeea770f05052bb7728cde6a"
P04_PERSISTENT_ALPACA_HELPER_PATH = "src/token_reconstruction/alpaca_split.py"
P04_PERSISTENT_ALPACA_HELPER_SHA256 = "fa9a15fd4cf92ffa06be3bd77888324536180e8a2fd2c43ae14f20e470a3626a"

P11_CANONICAL_PASS_BINDING_PATH = "experiments/TRR-P11/exclusions/canonical_pass_binding_r1.json"
P11_CANONICAL_PASS_BINDING_SHA256 = "0a9812e26d7953c8c53910af9737dfea31a6f4662b90b54753bec84ff5a697ec"
P11_P05_IDENTITY_PATH = "experiments/TRR-P11/exclusions/p05_h128_identity_rows_r1.json"
P11_P05_IDENTITY_SHA256 = "894359966b7af396ad841da13e3ff1caa1854db6cc81ae3d7382be10d60cdcad"
P11_P05_RECOVERY_PATH = "experiments/TRR-P11/exclusions/p05_h128_recovery_r1.json"
P11_P05_RECOVERY_SHA256 = "3c5251076a51a78aa1bae1da304098aa7fba8d9865e85e6e6e5092f6ac0ba1f9"
P11_P05_SCRIPT_SHA256 = "ea17fe280c2a739ad5b680a9f56b1cec08a662919fc282cac2282f4e31fdb5aa"
P11_P04_TARGET_IDENTITY_PATH = "experiments/TRR-P11/exclusions/p04_targetfit_identity_rows_r1.json"
P11_P04_TARGET_IDENTITY_SHA256 = "2a800aaec35e8dbc2437f5bb1c5a5c4e0a3ca67f5b68ed2b092b53ee6f6e40a4"
P11_P04_TARGET_RECOVERY_PATH = "experiments/TRR-P11/exclusions/p04_targetfit_recovery_r1.json"
P11_P04_TARGET_RECOVERY_SHA256 = "e83095caf3cc50356b896911dc13e9cd666fbaff7bdb3f6063bcc6feb32dc955"
P11_P04_TARGET_SCRIPT_SHA256 = "73a2b8cf6caa0e60660a9ab465f3ff5a5956d7cd68070476be90b3a464c33b34"

# TRR-0003's public fit ledger records the exact 40-token Pile geometry but
# did not retain the producer H40 digest. This bounded, public-only recovery
# re-rendered the pinned rows and emits opaque H40 identities. Keep it as a
# separate verified overlay: the recovery receipt has a producer-specific
# schema and intentionally does not masquerade as an H128/H129 export.
P11_TRR0003_H40_IDENTITY_PATH = "experiments/TRR-P11/exclusions/trr0003_h40_identity_rows_r1.json"
P11_TRR0003_H40_IDENTITY_SHA256 = "7889bfd955f1b6202370058b6168cb49a707f62a8a1518f7ab31c568fd5c4b3c"
P11_TRR0003_H40_RECOVERY_PATH = "experiments/TRR-P11/exclusions/trr0003_h40_recovery_r1.json"
P11_TRR0003_H40_RECOVERY_SHA256 = "a91165b5f1a73d2557cc7f9645729ad586895b43b05bf9d7dc899c42313bb2e5"
P11_TRR0003_H40_SCRIPT_PATH = "experiments/TRR-P11/exclusions/recover_trr0003_h40_r1.py"
P11_TRR0003_H40_SCRIPT_SHA256 = "f1f639f9aa39f2057b44dd4613bb5c961bc848991f795718215f7d4d936fe0fc"
P11_TRR0003_H40_FAILURE_PATH = "experiments/TRR-P11/exclusions/trr0003_h40_recovery_attempt1_failure.json"
P11_TRR0003_H40_FAILURE_SHA256 = "fbfa33ca082dbaf00a5c0a725bcdc8d62570164096b6750f37b3ab8a168ce086"

# Gitworker's already-completed public token/mask-only recoveries. The single
# original-fit export is producer-bound to the byte-identical TRR-0004 fit
# ledgers and to the exact record lineage reused by TRR-0005/TRR-0007.
P11_ORIGINAL_RECOVERY_HANDOFF_PATH = "experiments/TRR-P11/exclusions/original_payload_recovery_handoff_r1.json"
P11_ORIGINAL_RECOVERY_HANDOFF_SHA256 = "1b14607df868bd9c98ca86eb233833d7e739679c89f78dd62bbd32d482841b11"
P11_ORIGINAL_RECOVERY_HANDOFF_HISTORICAL_SHA256 = "af52b27c503a2e935a7beb0978fd5fe3b07d27bebe0893aa6ecf9c34d91a5986"
P11_ORIGINAL_FIT_IDENTITY_PATH = "experiments/TRR-P11/exclusions/original_fit_1200_identity_r1.json"
P11_ORIGINAL_FIT_IDENTITY_SHA256 = "45700bb02bbb89e7c8006e0bc6cff60b8f3afaeea533e1d9c4c41cc19531a9da"
P11_ORIGINAL_FIT_RECOVERY_PATH = "experiments/TRR-P11/exclusions/original_fit_1200_recovery_r1.json"
P11_ORIGINAL_FIT_RECOVERY_SHA256 = "3f06a5287beaccb39efc3bf6b0ad2da9dbf09e376eb4d58fb2406152b0bd95a1"
P11_VALIDATION_IDENTITY_PATH = "experiments/TRR-P11/exclusions/validation_48_identity_r1.json"
P11_VALIDATION_IDENTITY_SHA256 = "d68d98aecdfd0cf041be1e508074ffbf4ded24e0bc0a401362f1eb262e5e17ab"
P11_VALIDATION_RECOVERY_PATH = "experiments/TRR-P11/exclusions/validation_48_recovery_r1.json"
P11_VALIDATION_RECOVERY_SHA256 = "9473f1b378add851456dc1acf120cb530738ddbb055695c46613fcb0311efc0e"
P11_ORIGINAL_FIT_METADATA_SHA256 = "3f733212efd62b19e71f60f002f4e9756b4975ebfb4dcabf0ac27935929f94ff"
P11_ORIGINAL_FIT_METADATA_BYTES = 463510
P11_ORIGINAL_ALPACA_METADATA_SHA256 = "9c577f4ecb54ccb61e114aa741859e1bc045a4fd9d302479b2a49d17050dcf81"
P11_ORIGINAL_ALPACA_METADATA_BYTES = 754638
P11_VALIDATION_METADATA_SHA256 = "30b422b681bef5e7af4c26d339e57dfb3571ecef8077bdc4be5d960ef05c9777"
P11_VALIDATION_METADATA_BYTES = 16456
P11_ORIGINAL_FIT_PAYLOAD_SHA256 = "d1c78fcf1acc91b57d51355ee11f267bf4c12f1bc7d5160164b3b6ea11b45344"
P11_VALIDATION_PAYLOAD_SHA256 = "a8e7633ffb369864af33754c5ebb2d9a4ca9d6e7d4550731e8ff26e20c8200cf"
P11_ORIGINAL_LINEAGE_METADATA = {
    "trr0004_affine_fit": ("experiments/TRR-0004/fit/affine_fit_records.json", P11_ORIGINAL_FIT_METADATA_SHA256),
    "trr0004_adapter_v2_fit": ("experiments/TRR-0004/fit/adapter_v2/affine_fit_records.json", P11_ORIGINAL_FIT_METADATA_SHA256),
    "trr0005_original_fit": ("experiments/TRR-0005/public_activation_v1/original_fit_records.json", P11_ORIGINAL_ALPACA_METADATA_SHA256),
    "trr0007_original_fit": ("experiments/TRR-0007/support/broader_capture_v2/original_fit_records.json", P11_ORIGINAL_ALPACA_METADATA_SHA256),
}

# Gitworker's completed TRR-0001/TRR-0002 public H40 recovery.  TRR-0001 is
# raw signed-int32 H40; TRR-0002 is the historical tensor-header plus
# signed-int64 H40.  Keep their identity fields separate in the union.
P11_H40_RECOVERY_HANDOFF_PATH = "experiments/TRR-P11/exclusions/trr0001_trr0002_h40_recovery_handoff_r1.json"
P11_H40_RECOVERY_HANDOFF_SHA256 = "9ab88d1abadeebb22038936b26672a7ecca337c7c93f206e5708c00ef0f2f1aa"
P11_H40_RECOVERY_HANDOFF_BYTES = 3860
P11_H40_RECOVERY_ATTEMPTS_PATH = "experiments/TRR-P11/exclusions/trr0001_trr0002_h40_attempts_r1.json"
P11_H40_RECOVERY_ATTEMPTS_SHA256 = "cf5cb6474c90a540a13687dad08e606aae546e0f9afc217a642e8d8bbe75053d"
P11_H40_RECOVERY_ATTEMPTS_BYTES = 4696
P11_H40_RECOVERY_SCRIPT_PATH = "experiments/TRR-P11/exclusions/recover_trr0001_trr0002_h40_r1.py"
P11_H40_RECOVERY_SCRIPT_SHA256 = "3d43b2a93950be035c6889182139d696165c0fa8b62e2634a85ff13b643c0c04"
P11_TRR0001_H40_IDENTITY_PATH = "experiments/TRR-P11/exclusions/trr0001_h40_identity_rows_r1.json"
P11_TRR0001_H40_IDENTITY_SHA256 = "4268eefa65fc36c94255cb7a8b4b8fc8993eb38985081c5b115c969454c3bab3"
P11_TRR0001_H40_IDENTITY_BYTES = 150792
P11_TRR0001_H40_RECOVERY_PATH = "experiments/TRR-P11/exclusions/trr0001_h40_recovery_r1.json"
P11_TRR0001_H40_RECOVERY_SHA256 = "b26e1e984fd975a2b158081515c205fd4d410c5e649f756a4912738f9763e83c"
P11_TRR0001_H40_RECOVERY_BYTES = 6656
P11_TRR0002_H40_IDENTITY_PATH = "experiments/TRR-P11/exclusions/trr0002_h40_identity_rows_r1.json"
P11_TRR0002_H40_IDENTITY_SHA256 = "e938f89f3acc00cf13a096c911de9a94083b56f2821e8984860a656a8d3ecc11"
P11_TRR0002_H40_IDENTITY_BYTES = 42509
P11_TRR0002_H40_RECOVERY_PATH = "experiments/TRR-P11/exclusions/trr0002_h40_recovery_r1.json"
P11_TRR0002_H40_RECOVERY_SHA256 = "d3416cdc3aa76e4081bccc7ecb40a1d9c919556bb39e93dead355edd5b58589a"
P11_TRR0002_H40_RECOVERY_BYTES = 6576
P11_VALIDATION_LINEAGE_METADATA = {
    "trr0004_affine_validation": ("experiments/TRR-0004/fit/affine_validation_records.json", P11_VALIDATION_METADATA_SHA256),
    "trr0004_adapter_v2_validation": ("experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json", P11_VALIDATION_METADATA_SHA256),
}

# Agent 1's exact hash-only replication inputs. These are metadata/manifest
# bindings only; the exclusion audit never opens the B0/B1 activation tensors.
P11_REPLICATION_INPUTS: dict[str, Any] = {
    "status": "BOUND_EXACT_BANK_AND_SELECTION_IDENTITIES_AUDIT_COMPLETE",
    "banks": {
        "B0": {
            "bank": "B0",
            "bank_manifest": {
                "path": "../TRR-0007/experiments/TRR-0007/support/broader_capture_v2/enriched_manifest.json",
                "sha256": "c7a857e545a2f252ce8b3ab71bb2336e552fd212c7c434e39cef66f233b77a08",
                "bytes": 8156,
            },
            "ordered_identity": {
                "path": "../TRR-0007/experiments/TRR-0007/support/broader_capture_v2/enriched_fit_records.json",
                "sha256": "808cfd0f95ea3ee66c1d4094f3c10f1e346ba986a6d77138a61b8f8f13c4a738",
                "bytes": 918427,
            },
        },
        "B1": {
            "bank": "B1",
            "bank_manifest": {
                "path": "../TRR-0012/experiments/TRR-0012/capture/b1_activation_date07/bank_manifest.json",
                "sha256": "8265df60998fd23a11db560de661e65d972aa8823df09101303f4bab44732c7c",
                "bytes": 144117,
            },
            "ordered_identity": {
                "path": "../TRR-0012/experiments/TRR-0012/preparation/b1_cpu_r4_date07/records.json",
                "sha256": "367cfba0ffe78f59454861a76f23830f480a8745cc6b35e6b4a0d05eca53638b",
                "bytes": 10767794,
            },
            "preparation_manifest": {
                "path": "../TRR-0012/experiments/TRR-0012/preparation/b1_cpu_r4_date07/preparation_manifest.json",
                "sha256": "d1097b467e4abe4ec9f39dfca27f34f1d481527d2de79ca610751a8fe260620e",
                "bytes": 42163,
            },
        },
    },
    "development_selection": {
        "path": "../TRR-0012/experiments/TRR-0012/preparation/public_validation_r1/validation_manifest.json",
        "sha256": "b47114d86d80d704c331f50693ae2bc79741147cc01cc692cacba0c5e0697f65",
        "bytes": 468777,
    },
}
P11_HISTORICAL_SELECTION_BINDING = {
    "path": "../TRR-0010/experiments/TRR-0009/selection_v2/source_selection.json",
    "sha256": "c2e996514f7f45e55d7bfadc27fc8048bdb07a2c979b208de8716972d8d1def2",
    "bytes": 287719,
}
P11_TRR0005_PRODUCER_PATH = "scripts/trr0005_produce_confirmation.py"
P11_TRR0005_PRODUCER_SHA256 = "7289ee5db30bc0b2d3c1b50b1d5b6ca1ac235001371f61b92adf10e2dd979e2b"
P11_TRR0005_SELECTION_PLAN_PATH = "experiments/TRR-0005/fresh_confirmation_v1/selection_plan.json"
P11_REPLICATION_SIDECAR_PATH = "experiments/TRR-P11/exclusions/replication_inputs_binding_r1.json"
P11_REPLICATION_SIDECAR_SHA256 = "6b1945a065d7a737b40ca0623bc5bedc909e8721836c95eed4742b64ae171c98"

P04_LEDGER_OBJECTS: dict[str, tuple[str, str, int]] = {
    "correction": ("experiments/TRR-P04/setup/public-pools-r2/correction_records.json", "2dad95db8f1f796a476e812e6d7eb0a232a325852e6a17ddc3187088c72c2ea2", 256),
    "validation": ("experiments/TRR-P04/setup/public-pools-r2/validation_records.json", "e9aca619f9973f5131c60086f525ba899ec4788561fce68480427e9cdc26f3fc", 192),
    "fresh_evaluation": ("experiments/TRR-P04/setup/public-pools-r2/fresh_panel_index.json", "e3eb40cd0d18529c98d274eb102880f6480ddf055d3da52422a304a0ff6935f1", 72),
}
BOS_TOKEN_ID = 128000

# Six aggregate panels whose individual rows are represented by an immutable
# selection ledger.  PR20 has one extra binding layer and is handled below.
AGGREGATE_BINDINGS: tuple[dict[str, str], ...] = (
    {
        "label": "trr0006_panel",
        "panel": "experiments/TRR-0006/panel_capture_v1/panel.json",
        "selection": "experiments/TRR-0006/source_selection.json",
    },
    {
        "label": "trr0006_public_observation_panel",
        "panel": "experiments/TRR-0006/public_observations_v1/panel.json",
        "selection": "experiments/TRR-0006/source_selection.json",
    },
    {
        "label": "trr0007_eval_panel",
        "panel": "experiments/TRR-0007/evaluation/public_observations/panel.json",
        "selection": "experiments/TRR-0007/selection/source_selection.json",
    },
    {
        "label": "trr0008_eval_panel",
        "panel": "experiments/TRR-0008/evaluation/public_observations_v1/panel.json",
        "selection": "experiments/TRR-0008/selection/source_selection.json",
    },
    {
        "label": "trr0009_eval_panel",
        "panel": "experiments/TRR-0009/evaluation/public_observations_v2/panel.json",
        "selection": "experiments/TRR-0009/selection_v2/source_selection.json",
    },
    {
        "label": "pr20_source_panel",
        "panel": "@pr20/experiments/TRR-0010/evaluation/public_capture_watchdog_r5/observations_v1/panel.json",
        "selection": "@pr20/experiments/TRR-0010/evaluation/source_selection.json",
    },
)

_HEX = set("0123456789abcdefABCDEF")


class ExclusionAuditError(p10.AuditError):
    """Raised when a P11 exclusion invariant cannot be established."""


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExclusionAuditError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExclusionAuditError(f"{label} root is not an object: {path}")
    return value


def _read_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExclusionAuditError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ExclusionAuditError(f"{label} root is not an object")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _digest_lines(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve(path: str, *, root: Path, pr20_root: Path | None) -> Path:
    if path.startswith("@pr20/"):
        if pr20_root is None:
            raise ExclusionAuditError("@pr20 path requested without --pr20-root")
        return (pr20_root / path.removeprefix("@pr20/")).resolve()
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _descriptor_path(
    descriptor: Mapping[str, Any], *, root: Path, pr20_root: Path | None, label: str
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ExclusionAuditError(f"{label} descriptor is malformed")
    raw = descriptor.get("path")
    if not isinstance(raw, str) or not raw:
        raise ExclusionAuditError(f"{label} descriptor has no path")
    path = _resolve(raw, root=root, pr20_root=pr20_root)
    if path.is_symlink() or not path.is_file():
        raise ExclusionAuditError(f"{label} descriptor target is unavailable: {path}")
    actual_bytes = path.stat().st_size
    actual_sha = p10.sha256_file(path)
    try:
        declared_bytes = int(descriptor["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExclusionAuditError(f"{label} descriptor bytes are malformed") from exc
    if declared_bytes != actual_bytes or descriptor.get("sha256") != actual_sha:
        raise ExclusionAuditError(f"{label} descriptor changed")
    return path


def _new_bundle(spec: p10.SourceSpec, path: Path, value: Mapping[str, Any]) -> p10.IdentityBundle:
    return p10.IdentityBundle(
        label=spec.label,
        role=spec.role,
        path=path,
        sha256=p10.sha256_file(path),
        bytes=path.stat().st_size,
        schema=value.get("schema") if isinstance(value.get("schema"), str) else None,
        status=value.get("status") if isinstance(value.get("status"), str) else None,
    )


def _verify_p04_opaque_summary(info: Mapping[str, Any], *, label: str) -> list[str]:
    """Check the producer's ordered/set commitments and return its values.

    Values remain in memory only long enough to insert them into the identity
    set.  Receipts expose counts/commitments, never the opaque arrays.
    """
    if info.get("available") is not True:
        return []
    values = info.get("ordered_values")
    unique = info.get("unique_values")
    if not isinstance(values, list) or not isinstance(unique, list):
        raise ExclusionAuditError(f"P04 {label} lacks ordered/unique opaque values")
    if any(not _is_sha256(value) for value in values + unique):
        raise ExclusionAuditError(f"P04 {label} contains a malformed opaque SHA")
    expected_unique = sorted(set(value.casefold() for value in values))
    actual_unique = [value.casefold() for value in unique]
    if actual_unique != expected_unique:
        raise ExclusionAuditError(f"P04 {label} unique values are not sorted/deduplicated")
    if info.get("ordered_count") != len(values) or info.get("distinct_count") != len(expected_unique):
        raise ExclusionAuditError(f"P04 {label} count commitment mismatch")
    if info.get("ordered_canonical_json_sha256") != _json_digest(values):
        raise ExclusionAuditError(f"P04 {label} ordered canonical commitment mismatch")
    if info.get("ordered_newline_sha256") != _digest_lines(values):
        raise ExclusionAuditError(f"P04 {label} ordered newline commitment mismatch")
    if info.get("unique_set_canonical_json_sha256") != _json_digest(expected_unique):
        raise ExclusionAuditError(f"P04 {label} unique canonical commitment mismatch")
    return [value.casefold() for value in values]


@dataclass(frozen=True)
class P04Load:
    bundle: p10.IdentityBundle
    proof: dict[str, Any]
    gaps: list[dict[str, Any]]


def _git_object_bytes(root: Path, commit: str, relative_path: str) -> bytes | None:
    """Read a retained Git object without materializing it in the worktree."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "show", f"{commit}:{relative_path}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _p04_producer_git_proof(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    producer_bytes = _git_object_bytes(root, P04_PRODUCER_COMMIT, "scripts/trr0004_p04_reservation_hashes.py")
    records.append({
        "path": "scripts/trr0004_p04_reservation_hashes.py",
        "commit": P04_PRODUCER_COMMIT,
        "expected_sha256": P04_PRODUCER_SHA256,
        "available": producer_bytes is not None,
        "sha256_match": producer_bytes is not None and hashlib.sha256(producer_bytes).hexdigest() == P04_PRODUCER_SHA256,
    })
    for relative_path, commit, expected in P04_HELPER_OBJECTS:
        data = _git_object_bytes(root, commit, relative_path)
        records.append({
            "path": relative_path,
            "commit": commit,
            "expected_sha256": expected,
            "available": data is not None,
            "sha256_match": data is not None and hashlib.sha256(data).hexdigest() == expected,
        })
    # These two helpers are part of the current P11 checkout and are also
    # listed by the P04 producer as convention sources.
    for relative_path, expected in (
        ("src/token_reconstruction/public_activation.py", "3be43dd5b6a5918a48f289971ab2a833977a7f9d87094e6001c5c7cdc60ab76f"),
        ("src/token_reconstruction/alpaca_split.py", "fa9a15fd4cf92ffa06be3bd77888324536180e8a2fd2c43ae14f20e470a3626a"),
    ):
        path = root / relative_path
        available = path.is_file() and not path.is_symlink()
        records.append({
            "path": relative_path,
            "source": "current_worktree",
            "expected_sha256": expected,
            "available": available,
            "sha256_match": available and p10.sha256_file(path) == expected,
        })
    return {
        "records": records,
        "all_expected_source_bytes_verified": all(item["available"] and item["sha256_match"] for item in records),
    }



def _p04_target_plan_proof(root: Path) -> dict[str, Any]:
    """Bind the retained target-selection rule while preserving counts-only status."""
    data = _git_object_bytes(root, P04_TARGET_PLAN_COMMIT, P04_TARGET_PLAN_PATH)
    if data is None or hashlib.sha256(data).hexdigest() != P04_TARGET_PLAN_SHA256:
        raise ExclusionAuditError("retained P04 target plan is unavailable or changed")
    value = _read_json_bytes(data, label="P04 evaluator target plan")
    corpus = value.get("update_corpus")
    selection = corpus.get("selection") if isinstance(corpus, Mapping) else None
    schedule = value.get("schedule")
    if (
        value.get("schema") != "token-reconstruction.trr-p04-evaluator-target-plan.v1"
        or value.get("task_id") != "TRR-P04"
        or value.get("condition_id") != "p04_evaluator_target_update_v1"
        or not isinstance(corpus, Mapping)
        or corpus.get("dataset_id") != "HuggingFaceH4/no_robots"
        or corpus.get("dataset_revision") != "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b"
        or corpus.get("split") != "train"
        or corpus.get("expected_source_rows") != 9500
        or not isinstance(selection, Mapping)
        or selection.get("seed") != 20260910
        or selection.get("records") != 256
        or selection.get("order") != "sha256(TRR-P04|target-update-v1|row:index|seed:20260910), then row index"
        or selection.get("row_ids_serialized_in_public_metadata") is not False
        or selection.get("selected_row_order_sha256") != "42fb5bb7dfc58dba8ccf9e3b288e787fd88cc1788e936ee934cfbbdc86de2fd2"
        or not isinstance(schedule, Mapping)
        or schedule.get("formula") != "row=(step*record_batch_size+offset) mod records"
        or schedule.get("batch_order") != "cyclic sequential selected rows"
    ):
        raise ExclusionAuditError("retained P04 target selection rule changed")
    if any(corpus.get(key) is not False for key in ("student_training_access", "teacher_access", "fresh_panel_access")):
        raise ExclusionAuditError("retained P04 target access boundary changed")
    return {
        "status": "PASS_TARGET_RULE_COUNTS_ONLY",
        "plan_path": P04_TARGET_PLAN_PATH,
        "plan_commit": P04_TARGET_PLAN_COMMIT,
        "plan_sha256": P04_TARGET_PLAN_SHA256,
        "dataset_id": corpus["dataset_id"],
        "dataset_revision": corpus["dataset_revision"],
        "split": corpus["split"],
        "source_rows": corpus["expected_source_rows"],
        "selected_rows": selection["records"],
        "selection_seed": selection["seed"],
        "selection_order": selection["order"],
        "selected_row_order_sha256": selection["selected_row_order_sha256"],
        "row_ids_serialized_in_public_metadata": False,
        "individual_target_hashes_available": False,
        "targetfit_recovery_requirement": "render the pinned no_robots rows using the exact producer/tokenizer and compare public/rendered and first-129/first-128 identities before admitting them",
    }

def _recover_p04_ledger(root: Path, pool: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    relative_path, expected_sha, expected_count = P04_LEDGER_OBJECTS[pool]
    data = _git_object_bytes(root, "81542a6ac87b22ca3a1b9a48dacf1e0e0afb1bf3", relative_path)
    if data is None or hashlib.sha256(data).hexdigest() != expected_sha:
        return None, {"pool": pool, "available": False, "expected_sha256": expected_sha, "record_count": expected_count}
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExclusionAuditError(f"recovered P04 {pool} ledger is invalid JSON") from exc
    rows = value.get("records")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ExclusionAuditError(f"recovered P04 {pool} ledger record count changed")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("record_id"), str)
            or not row["record_id"]
            or not _is_sha256(row.get("public_record_sha256"))
            or not _is_sha256(row.get("truncated_sequence_sha256"))
            or not isinstance(row.get("row_index"), int)
            or not isinstance(row.get("full_token_count"), int)
            or not isinstance(row.get("post_bos_token_count"), int)
        ):
            raise ExclusionAuditError(f"recovered P04 {pool} ledger row {index} is malformed")
    return value, {
        "pool": pool,
        "available": True,
        "commit": "81542a6ac87b22ca3a1b9a48dacf1e0e0afb1bf3",
        "path": relative_path,
        "sha256": expected_sha,
        "record_count": len(rows),
        "per_record_identity_fields": ["record_id", "public_record_sha256", "row_index", "dataset_id", "dataset_revision", "truncated_sequence_sha256"],
    }




def _load_canonical_pass_binding(root: Path) -> dict[str, Any]:
    path = (root / P11_CANONICAL_PASS_BINDING_PATH).resolve()
    if not path.is_file() or path.is_symlink() or p10.sha256_file(path) != P11_CANONICAL_PASS_BINDING_SHA256:
        raise ExclusionAuditError("canonical CPU-pass binding receipt is unavailable or changed")
    value = _read_json(path, label="canonical CPU-pass binding")
    if value.get("schema") != "token-reconstruction.trr-p11-canonical-pass-binding.v1" or value.get("task_id") != TASK_ID or value.get("status") != "PASS_P05_TOKEN_MASK_AND_P04_TARGETFIT_CANONICAL_RECOVERY":
        raise ExclusionAuditError("canonical CPU-pass binding status changed")
    steps = value.get("steps")
    if not isinstance(steps, list) or {item.get("label") for item in steps if isinstance(item, Mapping)} != {"p05_enriched_fit_token_mask", "p04_no_robots_targetfit"}:
        raise ExclusionAuditError("canonical CPU-pass binding steps are incomplete")
    expected = {
        "p05_enriched_fit_token_mask": (P11_P05_SCRIPT_SHA256, P11_P05_IDENTITY_PATH, P11_P05_IDENTITY_SHA256, P11_P05_RECOVERY_PATH, P11_P05_RECOVERY_SHA256),
        "p04_no_robots_targetfit": (P11_P04_TARGET_SCRIPT_SHA256, P11_P04_TARGET_IDENTITY_PATH, P11_P04_TARGET_IDENTITY_SHA256, P11_P04_TARGET_RECOVERY_PATH, P11_P04_TARGET_RECOVERY_SHA256),
    }
    for step in steps:
        if not isinstance(step, Mapping) or step.get("label") not in expected:
            raise ExclusionAuditError("canonical CPU-pass step is malformed")
        script_sha, identity_path, identity_sha, recovery_path, recovery_sha = expected[step["label"]]
        if step.get("script_sha256") != script_sha or step.get("identity_path") != identity_path or step.get("identity_sha256") != identity_sha or step.get("recovery_path") != recovery_path or step.get("recovery_sha256") != recovery_sha or step.get("status") not in {"PASS_PUBLIC_TOKEN_IDENTITY_RECOVERY", "PASS_P04_TARGETFIT_EXACT_RENDERED_H129_H128_RECOVERY"}:
            raise ExclusionAuditError(f"canonical CPU-pass binding changed: {step.get('label')}")
        identity = (root / identity_path).resolve()
        recovery = (root / recovery_path).resolve()
        if not identity.is_file() or identity.is_symlink() or p10.sha256_file(identity) != identity_sha or not recovery.is_file() or recovery.is_symlink() or p10.sha256_file(recovery) != recovery_sha:
            raise ExclusionAuditError(f"canonical CPU-pass output changed: {step.get('label')}")
    resource = value.get("resource_contract")
    access = value.get("access_boundary")
    if not isinstance(resource, Mapping) or resource.get("threads") != 1 or resource.get("timeout_seconds") != 240 or resource.get("model_loaded") is not False or resource.get("gpu_used") is not False:
        raise ExclusionAuditError("canonical CPU-pass resource contract changed")
    if not isinstance(access, Mapping) or any(access.get(key) is True for key in ("source_text_serialized", "source_tokens_serialized", "token_values_emitted", "activations_read", "model_loaded", "gpu_used", "evaluation_truth_opened", "p03_holdout_accessed", "new_selection_started")):
        raise ExclusionAuditError("canonical CPU-pass access boundary changed")
    return {"path": P11_CANONICAL_PASS_BINDING_PATH, "sha256": P11_CANONICAL_PASS_BINDING_SHA256, "status": value["status"], "steps": steps, "resource_contract": dict(resource)}


def _load_public_identity_export(
    root: Path,
    *,
    identity_path: str,
    identity_sha256: str,
    recovery_path: str,
    recovery_sha256: str,
    expected_identity_schema: str,
    expected_recovery_schema: str,
    expected_identity_status: str,
    label: str,
    role: str,
    binding: Mapping[str, Any],
) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    identity_file = (root / identity_path).resolve()
    recovery_file = (root / recovery_path).resolve()
    if not identity_file.is_file() or identity_file.is_symlink() or p10.sha256_file(identity_file) != identity_sha256:
        raise ExclusionAuditError(f"{label} identity export is unavailable or changed")
    if not recovery_file.is_file() or recovery_file.is_symlink() or p10.sha256_file(recovery_file) != recovery_sha256:
        raise ExclusionAuditError(f"{label} recovery receipt is unavailable or changed")
    identity = _read_json(identity_file, label=f"{label} identity export")
    recovery = _read_json(recovery_file, label=f"{label} recovery receipt")
    if identity.get("schema") != expected_identity_schema or identity.get("task_id") != TASK_ID or identity.get("status") != expected_identity_status:
        raise ExclusionAuditError(f"{label} identity export status changed")
    if recovery.get("schema") != expected_recovery_schema or recovery.get("task_id") != TASK_ID or recovery.get("status") != expected_identity_status or recovery.get("mismatch_count") != 0:
        raise ExclusionAuditError(f"{label} recovery receipt status changed")
    if recovery.get("identity_export", {}).get("sha256") != identity_sha256:
        raise ExclusionAuditError(f"{label} recovery receipt does not bind identity export")
    rows = identity.get("records")
    if not isinstance(rows, list) or identity.get("record_count") != len(rows):
        raise ExclusionAuditError(f"{label} identity row count is malformed")
    forbidden = {"source_text", "source_tokens", "token_ids", "input_ids", "labels", "truth", "oracle", "target_weights", "activations"}
    if forbidden.intersection(identity):
        raise ExclusionAuditError(f"{label} identity export contains forbidden payload")
    access = identity.get("access_boundary")
    if not isinstance(access, Mapping) or any(access.get(key) is True for key in ("source_text_serialized", "source_tokens_serialized", "token_values_emitted", "activations_read", "model_loaded", "gpu_used", "evaluation_truth_opened", "p03_holdout_accessed", "new_selection_started")):
        raise ExclusionAuditError(f"{label} identity access boundary is unsafe")
    bundle = p10.IdentityBundle(label, role, identity_file, identity_sha256, identity_file.stat().st_size, schema=expected_identity_schema, status=expected_identity_status)
    h128 = h129 = rendered = source_indices = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row["record_id"]:
            raise ExclusionAuditError(f"{label} row {index} record_id is malformed")
        style = p10._style(row.get("dataset_key") or row.get("domain") or row.get("style")) or "*"
        dataset_id_value = row.get("dataset_id")
        dataset_id = dataset_id_value if isinstance(dataset_id_value, str) and dataset_id_value else "*"
        split_value = row.get("split")
        split = split_value if isinstance(split_value, str) and split_value else "*"
        revision_value = row.get("revision") or row.get("dataset_revision")
        revision = revision_value if isinstance(revision_value, str) and revision_value else "*"
        namespace = p10.Namespace(style, dataset_id, split, revision)
        bundle.add("record_id", row["record_id"], namespace)
        for field in ("rendered_sha256", "h128_sequence_sha256", "h129_sequence_sha256"):
            field_value = row.get(field)
            if field_value is None:
                continue
            if not _is_sha256(field_value):
                raise ExclusionAuditError(f"{label} row {index} {field} is malformed")
            bundle.add(field, field_value.casefold(), namespace)
            if field == "rendered_sha256": rendered += 1
            elif field == "h128_sequence_sha256": h128 += 1
            else: h129 += 1
        source_index = row.get("source_index")
        if source_index is None:
            source_index = row.get("row_index")
        if isinstance(source_index, int) and not isinstance(source_index, bool) and source_index >= 0 and style != "*":
            bundle.add("source_index", source_index, namespace); source_indices += 1
        full = row.get("full_token_count")
        post = row.get("post_bos_token_count")
        if not isinstance(full, int) or not isinstance(post, int) or full != post + 1:
            raise ExclusionAuditError(f"{label} row {index} geometry is malformed")
    bundle.metadata_notes.append("Opaque identity export; no source/token/truth payload emitted")
    proof = {"label": label, "role": role, "identity_path": identity_path, "identity_sha256": identity_sha256, "recovery_path": recovery_path, "recovery_sha256": recovery_sha256, "identity_schema": expected_identity_schema, "record_count": len(rows), "identity_counts": bundle.counts(), "h128_rows": h128, "h129_rows": h129, "rendered_rows": rendered, "source_index_rows": source_indices, "binding": dict(binding)}
    return bundle, proof


def _load_trr0003_h40_identity_export(
    root: Path,
) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    """Load the producer-specific TRR-0003 public H40 overlay.

    TRR-0003's fit metadata retained the pinned public rows and their 40-token
    geometry, while its original receipt did not retain H40 digests.  The
    companion recovery was rerun from the immutable Pile revision and emits
    only opaque rendered/H40 identities.  Keep this loader strict and
    producer-specific: a generic ``truncated_sequence_sha256`` alias must
    never turn an H129 or unknown-width value into H40.
    """
    identity_file = (root / P11_TRR0003_H40_IDENTITY_PATH).resolve()
    recovery_file = (root / P11_TRR0003_H40_RECOVERY_PATH).resolve()
    failure_file = (root / P11_TRR0003_H40_FAILURE_PATH).resolve()
    recipe_file = (root / P11_TRR0003_H40_SCRIPT_PATH).resolve()
    for path, expected, label in (
        (identity_file, P11_TRR0003_H40_IDENTITY_SHA256, "TRR-0003 H40 identity export"),
        (recovery_file, P11_TRR0003_H40_RECOVERY_SHA256, "TRR-0003 H40 recovery receipt"),
        (failure_file, P11_TRR0003_H40_FAILURE_SHA256, "TRR-0003 H40 failed-attempt receipt"),
        (recipe_file, P11_TRR0003_H40_SCRIPT_SHA256, "TRR-0003 H40 recovery recipe"),
    ):
        if not path.is_file() or path.is_symlink() or p10.sha256_file(path) != expected:
            raise ExclusionAuditError(f"{label} is unavailable or changed")

    identity = _read_json(identity_file, label="TRR-0003 H40 identity export")
    recovery = _read_json(recovery_file, label="TRR-0003 H40 recovery receipt")
    failure = _read_json(failure_file, label="TRR-0003 H40 failed-attempt receipt")
    if (
        identity.get("schema") != "token-reconstruction.trr-p11-public-h40-identity-rows.v1"
        or identity.get("task_id") != TASK_ID
        or identity.get("status") != "PASS_TRR0003_PUBLIC_H40_RECOVERY"
    ):
        raise ExclusionAuditError("TRR-0003 H40 identity export status changed")
    if (
        recovery.get("schema") != "token-reconstruction.trr-p11-trr0003-h40-recovery.v1"
        or recovery.get("task_id") != TASK_ID
        or recovery.get("status") != "PASS_TRR0003_PUBLIC_H40_RECOVERY"
    ):
        raise ExclusionAuditError("TRR-0003 H40 recovery receipt status changed")
    coverage = recovery.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("rows_seen") != 128
        or coverage.get("h40_rows") != 128
        or coverage.get("h128_rows") != 0
        or coverage.get("h129_rows") != 0
        or coverage.get("mismatch_count") != 0
    ):
        raise ExclusionAuditError("TRR-0003 H40 recovery coverage changed")
    output = recovery.get("output")
    if (
        not isinstance(output, Mapping)
        or output.get("sha256") != P11_TRR0003_H40_IDENTITY_SHA256
        or output.get("bytes") != identity_file.stat().st_size
        or Path(str(output.get("path", ""))).name != identity_file.name
    ):
        raise ExclusionAuditError("TRR-0003 H40 recovery does not bind identity export")
    access = identity.get("access_boundary")
    if (
        not isinstance(access, Mapping)
        or any(
            access.get(key) is True
            for key in (
                "source_text_serialized",
                "token_values_emitted",
                "activations_read",
                "model_loaded",
                "gpu_used",
                "truth_opened",
                "p03_holdout_accessed",
                "new_selection_started",
            )
        )
    ):
        raise ExclusionAuditError("TRR-0003 H40 identity access boundary is unsafe")
    if (
        identity.get("contains_source_text") is not False
        or identity.get("contains_token_ids") is not False
        or identity.get("contains_truth") is not False
    ):
        raise ExclusionAuditError("TRR-0003 H40 identity export contains payload")
    if (
        failure.get("schema") != "token-reconstruction.trr-p11-h40-recovery-failure.v1"
        or failure.get("task_id") != TASK_ID
        or failure.get("status") != "EXCLUDED_METADATA_SCHEMA_MISMATCH"
        or failure.get("failure", {}).get("mismatch_count") != 128
    ):
        raise ExclusionAuditError("TRR-0003 failed-attempt provenance changed")

    producer = identity.get("producer")
    if (
        not isinstance(producer, Mapping)
        or producer.get("dataset_id") != "NeelNanda/pile-10k"
        or producer.get("revision") != "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
        or producer.get("sequence_convention") != "first 40 BOS-inclusive IDs, SHA-256 little-endian signed-int32 bytes"
        or producer.get("fit_metadata_sha256") != "7aee0f6cb452bb1df401c920ca2a628d32fb204d144d72e4419fc8bd34a3a08e"
        or producer.get("trr0001_plan_sha256") != "b498a5db5b14ae8dde19f3ae4f519f86fdf0a67572a8f78747f7921e4f9e7269"
    ):
        raise ExclusionAuditError("TRR-0003 H40 producer binding changed")
    rows = identity.get("records")
    if not isinstance(rows, list) or identity.get("record_count") != len(rows) or len(rows) != 128:
        raise ExclusionAuditError("TRR-0003 H40 identity row count is malformed")

    bundle = p10.IdentityBundle(
        "trr0003_fit_records_h40_public_identity",
        "fitting_bank",
        identity_file,
        P11_TRR0003_H40_IDENTITY_SHA256,
        identity_file.stat().st_size,
        schema=str(identity["schema"]),
        status=str(identity["status"]),
    )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExclusionAuditError(f"TRR-0003 H40 row {index} is malformed")
        record_id = row.get("record_id")
        rendered = row.get("rendered_sha256")
        h40 = row.get("trr0002_h40_token_ids_sha256")
        source_index = row.get("source_index")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not _is_sha256(rendered)
            or not _is_sha256(h40)
            or not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or source_index < 0
            or row.get("full_token_count") != 40
            or row.get("post_bos_token_count") != 39
            or row.get("dataset_id") != "NeelNanda/pile-10k"
            or row.get("split") != "train"
            or row.get("revision") != "127bfedcd5047750df5ccf3a12979a47bfa0bafa"
        ):
            raise ExclusionAuditError(f"TRR-0003 H40 row {index} violates the producer identity contract")
        namespace = p10.Namespace(
            "pile",
            "NeelNanda/pile-10k",
            "train",
            "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
        )
        bundle.add("record_id", record_id, namespace)
        bundle.add("rendered_sha256", rendered.casefold(), namespace)
        bundle.add("trr0002_h40_token_ids_sha256", h40.casefold(), namespace)
        bundle.add("source_index", source_index, namespace)
    bundle.metadata_notes.append(
        "Producer-verified TRR-0003 H40 overlay; public source/token IDs were transiently used for opaque hashing only"
    )
    proof = {
        "label": bundle.label,
        "role": bundle.role,
        "identity_path": P11_TRR0003_H40_IDENTITY_PATH,
        "identity_sha256": P11_TRR0003_H40_IDENTITY_SHA256,
        "recovery_path": P11_TRR0003_H40_RECOVERY_PATH,
        "recovery_sha256": P11_TRR0003_H40_RECOVERY_SHA256,
        "failed_attempt_path": P11_TRR0003_H40_FAILURE_PATH,
        "failed_attempt_sha256": P11_TRR0003_H40_FAILURE_SHA256,
        "recipe_path": P11_TRR0003_H40_SCRIPT_PATH,
        "recipe_sha256": P11_TRR0003_H40_SCRIPT_SHA256,
        "identity_schema": identity["schema"],
        "record_count": len(rows),
        "identity_counts": bundle.counts(),
        "h40_rows": 128,
        "h128_rows": 0,
        "h129_rows": 0,
        "rendered_rows": 128,
        "source_index_rows": 128,
        "producer": dict(producer),
        "resource": dict(recovery.get("resource", {})) if isinstance(recovery.get("resource"), Mapping) else {},
        "access_boundary": dict(access),
        "binding": {
            "fit_metadata_sha256": producer["fit_metadata_sha256"],
            "trr0001_plan_sha256": producer["trr0001_plan_sha256"],
            "dataset_revision": producer["revision"],
        },
    }
    return bundle, proof


def _verified_file_value(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    expected_bytes: int,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Read a hash-bound JSON metadata file for lineage validation."""
    path = (root / relative_path).resolve()
    if (
        not path.is_file()
        or path.is_symlink()
        or p10.sha256_file(path) != expected_sha256
        or path.stat().st_size != expected_bytes
    ):
        raise ExclusionAuditError(f"{label} is unavailable or changed")
    value = _read_json(path, label=label)
    return path, value


def _load_original_recovery_lineage(
    root: Path,
) -> tuple[list[p10.IdentityBundle], list[dict[str, Any]], dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]]]:
    """Bind the completed original-fit/validation exports to all consumers.

    Gitworker's exports are derived from public ``token_ids``/masks and carry
    exact H128/H129 commitments.  The TRR-0004 fit ledgers are byte-identical
    to each other; TRR-0005 and TRR-0007 preserve the same ordered original
    record IDs, geometry, source lineage, and non-synthetic markers.  This
    function verifies those producer facts before allowing record-ID joins for
    those four ledgers.  The join is deliberately local to this verified
    lineage; ordinary record-ID aliases remain insufficient elsewhere.
    """
    handoff_path, handoff = _verified_file_value(
        root,
        P11_ORIGINAL_RECOVERY_HANDOFF_PATH,
        P11_ORIGINAL_RECOVERY_HANDOFF_SHA256,
        9830,
        label="original payload recovery handoff",
    )
    if (
        handoff.get("schema") != "token-reconstruction.trr-p11-original-payload-recovery-handoff.v1"
        or handoff.get("task_id") != TASK_ID
        or handoff.get("status") != "PASS_OPAQUE_EXPORTS_READY_FOR_EXCLUSION_WORKER"
    ):
        raise ExclusionAuditError("original payload recovery handoff status changed")
    handoff_access = handoff.get("access_boundary")
    if (
        not isinstance(handoff_access, Mapping)
        or any(
            handoff_access.get(key) is True
            for key in (
                "source_text_serialized",
                "source_tokens_serialized",
                "token_values_emitted",
                "activations_read",
                "model_loaded",
                "gpu_used",
                "evaluation_truth_opened",
                "p03_holdout_accessed",
                "new_selection_started",
            )
        )
    ):
        raise ExclusionAuditError("original payload recovery access boundary is unsafe")
    recovery_program = handoff.get("recovery_program")
    if (
        not isinstance(recovery_program, Mapping)
        or recovery_program.get("path") != "scripts/trr_p11/recover_public_token_payload.py"
        or recovery_program.get("sha256") != P11_P05_SCRIPT_SHA256
    ):
        raise ExclusionAuditError("original payload recovery program binding changed")

    expected_jobs = {
        "trr4_original_fit_1200": {
            "identity_path": P11_ORIGINAL_FIT_IDENTITY_PATH,
            "identity_sha256": P11_ORIGINAL_FIT_IDENTITY_SHA256,
            "identity_bytes": 487363,
            "recovery_path": P11_ORIGINAL_FIT_RECOVERY_PATH,
            "recovery_sha256": P11_ORIGINAL_FIT_RECOVERY_SHA256,
            "metadata_suffix": "experiments/TRR-0004/fit/affine_fit_records.json",
            "metadata_sha256": P11_ORIGINAL_FIT_METADATA_SHA256,
            "metadata_bytes": P11_ORIGINAL_FIT_METADATA_BYTES,
            "payload_suffix": "outputs/TRR-0004/public_activation_v2/train_large_cut4.safetensors",
            "payload_sha256": P11_ORIGINAL_FIT_PAYLOAD_SHA256,
            "payload_bytes": 947176648,
            "rows": 1200,
            "h128_rows": 350,
            "h129_rows": 343,
            "short_rows": 850,
        },
        "trr4_original_validation_48": {
            "identity_path": P11_VALIDATION_IDENTITY_PATH,
            "identity_sha256": P11_VALIDATION_IDENTITY_SHA256,
            "identity_bytes": 16614,
            "recovery_path": P11_VALIDATION_RECOVERY_PATH,
            "recovery_sha256": P11_VALIDATION_RECOVERY_SHA256,
            "metadata_suffix": "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json",
            "metadata_sha256": P11_VALIDATION_METADATA_SHA256,
            "metadata_bytes": P11_VALIDATION_METADATA_BYTES,
            "payload_suffix": "experiments/TRR-0004/fit/adapter_v2/validation_mixed_cut4.safetensors",
            "payload_sha256": P11_VALIDATION_PAYLOAD_SHA256,
            "payload_bytes": 37795064,
            "rows": 48,
            "h128_rows": 6,
            "h129_rows": 5,
            "short_rows": 42,
        },
    }
    jobs = handoff.get("jobs")
    if not isinstance(jobs, list):
        raise ExclusionAuditError("original payload recovery handoff has no jobs")
    jobs_by_label = {job.get("label"): job for job in jobs if isinstance(job, Mapping)}
    if set(jobs_by_label) != set(expected_jobs):
        raise ExclusionAuditError("original payload recovery job set changed")

    job_proofs: dict[str, dict[str, Any]] = {}
    for job_label, expected in expected_jobs.items():
        job = jobs_by_label[job_label]
        identity = job.get("identity_export")
        recovery = job.get("recovery_receipt")
        metadata = job.get("metadata")
        payload = job.get("payload")
        coverage = job.get("coverage")
        if not all(isinstance(item, Mapping) for item in (identity, recovery, metadata, payload, coverage)):
            raise ExclusionAuditError(f"{job_label} handoff binding is malformed")
        if (
            identity.get("path") != expected["identity_path"]
            or identity.get("sha256") != expected["identity_sha256"]
            or identity.get("bytes") != expected["identity_bytes"]
            or recovery.get("path") != expected["recovery_path"]
            or recovery.get("sha256") != expected["recovery_sha256"]
            or metadata.get("sha256") != expected["metadata_sha256"]
            or metadata.get("bytes") != expected["metadata_bytes"]
            or not str(metadata.get("path", "")).endswith(expected["metadata_suffix"])
            or payload.get("sha256") != expected["payload_sha256"]
            or payload.get("bytes") != expected["payload_bytes"]
            or not str(payload.get("path", "")).endswith(expected["payload_suffix"])
            or coverage.get("rows_seen") != expected["rows"]
            or coverage.get("h128_rows") != expected["h128_rows"]
            or coverage.get("h129_rows") != expected["h129_rows"]
            or coverage.get("short_rows_h128_inapplicable") != expected["short_rows"]
            or job.get("mismatch_count") != 0
        ):
            raise ExclusionAuditError(f"{job_label} handoff evidence changed")
        job_proofs[job_label] = {
            "label": job_label,
            "identity_export": dict(identity),
            "recovery_receipt": dict(recovery),
            "metadata": dict(metadata),
            "payload": dict(payload),
            "coverage": dict(coverage),
            "mismatch_count": job.get("mismatch_count"),
            "ordered_record_ids_newline_sha256": job.get("ordered_record_ids_newline_sha256"),
            "ordered_record_ids_canonical_sha256": job.get("ordered_record_ids_canonical_sha256"),
            "guard": dict(job.get("guard", {})) if isinstance(job.get("guard"), Mapping) else {},
        }

    fit_bundle, fit_proof = _load_public_identity_export(
        root,
        identity_path=P11_ORIGINAL_FIT_IDENTITY_PATH,
        identity_sha256=P11_ORIGINAL_FIT_IDENTITY_SHA256,
        recovery_path=P11_ORIGINAL_FIT_RECOVERY_PATH,
        recovery_sha256=P11_ORIGINAL_FIT_RECOVERY_SHA256,
        expected_identity_schema="token-reconstruction.trr-p11-public-token-identity-rows.v1",
        expected_recovery_schema="token-reconstruction.trr-p11-public-token-identity-recovery.v1",
        expected_identity_status="PASS_PUBLIC_TOKEN_IDENTITY_RECOVERY",
        label="trr0004_original_fit_public_token_identity",
        role="fitting_bank",
        binding={"handoff_path": P11_ORIGINAL_RECOVERY_HANDOFF_PATH, "handoff_sha256": P11_ORIGINAL_RECOVERY_HANDOFF_SHA256, "job": job_proofs["trr4_original_fit_1200"]},
    )
    validation_bundle, validation_proof = _load_public_identity_export(
        root,
        identity_path=P11_VALIDATION_IDENTITY_PATH,
        identity_sha256=P11_VALIDATION_IDENTITY_SHA256,
        recovery_path=P11_VALIDATION_RECOVERY_PATH,
        recovery_sha256=P11_VALIDATION_RECOVERY_SHA256,
        expected_identity_schema="token-reconstruction.trr-p11-public-token-identity-rows.v1",
        expected_recovery_schema="token-reconstruction.trr-p11-public-token-identity-recovery.v1",
        expected_identity_status="PASS_PUBLIC_TOKEN_IDENTITY_RECOVERY",
        label="trr0004_original_validation_public_token_identity",
        role="calibration",
        binding={"handoff_path": P11_ORIGINAL_RECOVERY_HANDOFF_PATH, "handoff_sha256": P11_ORIGINAL_RECOVERY_HANDOFF_SHA256, "job": job_proofs["trr4_original_validation_48"]},
    )

    fit_identity = _read_json((root / P11_ORIGINAL_FIT_IDENTITY_PATH).resolve(), label="original fit identity export")
    validation_identity = _read_json((root / P11_VALIDATION_IDENTITY_PATH).resolve(), label="validation identity export")
    if (
        fit_identity.get("source_label") != "trr4_original_fit_1200"
        or fit_identity.get("metadata_sha256") != P11_ORIGINAL_FIT_METADATA_SHA256
        or fit_identity.get("payload_sha256") != P11_ORIGINAL_FIT_PAYLOAD_SHA256
        or validation_identity.get("source_label") != "trr4_original_validation_48"
        or validation_identity.get("metadata_sha256") != P11_VALIDATION_METADATA_SHA256
        or validation_identity.get("payload_sha256") != P11_VALIDATION_PAYLOAD_SHA256
    ):
        raise ExclusionAuditError("original identity export producer bindings changed")
    fit_rows = fit_identity.get("records")
    validation_rows = validation_identity.get("records")
    if not isinstance(fit_rows, list) or not isinstance(validation_rows, list):
        raise ExclusionAuditError("original identity exports lack records")

    def source_records(relative_path: str, expected_sha: str, expected_bytes: int, label: str) -> list[Mapping[str, Any]]:
        _path, value = _verified_file_value(root, relative_path, expected_sha, expected_bytes, label=label)
        rows = value.get("records")
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise ExclusionAuditError(f"{label} lacks a records list")
        return list(rows)

    fit_targets = {
        **P11_ORIGINAL_LINEAGE_METADATA,
    }
    fit_source_rows = {
        label: source_records(path, expected_sha, P11_ORIGINAL_FIT_METADATA_BYTES if expected_sha == P11_ORIGINAL_FIT_METADATA_SHA256 else P11_ORIGINAL_ALPACA_METADATA_BYTES, f"{label} metadata")
        for label, (path, expected_sha) in fit_targets.items()
    }
    validation_source_rows = {
        label: source_records(path, expected_sha, P11_VALIDATION_METADATA_BYTES, f"{label} metadata")
        for label, (path, expected_sha) in P11_VALIDATION_LINEAGE_METADATA.items()
    }

    def geometry(row: Mapping[str, Any]) -> tuple[int | None, int | None]:
        full = row.get("full_token_count")
        post = row.get("post_bos_token_count")
        if not isinstance(full, int) or not isinstance(post, int) or full != post + 1:
            raise ExclusionAuditError("recovered/source row geometry is malformed")
        return full, post

    def compare_order(
        exported: list[Mapping[str, Any]],
        source_rows: list[Mapping[str, Any]],
        *,
        label: str,
        alpaca_lineage: bool,
        compare_rendered: bool,
    ) -> dict[str, Any]:
        if len(exported) != len(source_rows):
            raise ExclusionAuditError(f"{label} row count differs from recovery export")
        ids_match = geometry_match = rendered_match = source_index_match = True
        synthetic_ok = target_geometry_ok = slot_order_ok = True
        for index, (export_row, source_row) in enumerate(zip(exported, source_rows)):
            if export_row.get("record_id") != source_row.get("record_id"):
                ids_match = False
            if geometry(export_row) != geometry(source_row):
                geometry_match = False
            if compare_rendered and export_row.get("rendered_sha256") != source_row.get("rendered_sha256"):
                rendered_match = False
            source_index = export_row.get("source_index")
            if source_index is not None and source_row.get("row_index") is not None and source_index != source_row.get("row_index"):
                source_index_match = False
            if alpaca_lineage:
                synthetic_ok = synthetic_ok and source_row.get("synthetic") is False and source_row.get("source_record_id") == source_row.get("record_id")
                target_geometry_ok = target_geometry_ok and source_row.get("target_full_token_count") == source_row.get("full_token_count") and source_row.get("target_post_bos_token_count") == source_row.get("post_bos_token_count")
                slot_order_ok = slot_order_ok and source_row.get("slot") == index
        if not (ids_match and geometry_match and rendered_match and source_index_match and synthetic_ok and target_geometry_ok and slot_order_ok):
            raise ExclusionAuditError(f"{label} producer lineage does not match recovery export")
        return {
            "label": label,
            "rows": len(exported),
            "record_id_order_match": ids_match,
            "geometry_match": geometry_match,
            "rendered_commitment_match": rendered_match,
            "source_index_match": source_index_match,
            "non_synthetic_source_lineage": synthetic_ok,
            "target_geometry_match": target_geometry_ok,
            "slot_order_match": slot_order_ok,
            "canonical_join": "exact ordered record_id + geometry + producer lineage; sequence proof comes from the recovered row itself",
        }

    lineage_checks = []
    for label, rows in fit_source_rows.items():
        lineage_checks.append(compare_order(fit_rows, rows, label=label, alpaca_lineage=label in {"trr0005_original_fit", "trr0007_original_fit"}, compare_rendered=label.startswith("trr0004_")))
    for label, rows in validation_source_rows.items():
        lineage_checks.append(compare_order(validation_rows, rows, label=label, alpaca_lineage=False, compare_rendered=True))

    def lineage_for(exported: list[Mapping[str, Any]], canonical_label: str) -> dict[str, set[tuple[str, tuple[str, ...]]]]:
        result: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
        for index, row in enumerate(exported):
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or record_id in result:
                raise ExclusionAuditError(f"{canonical_label} recovery record IDs are not unique")
            full, _post = geometry(row)
            h128 = row.get("h128_sequence_sha256")
            if full >= 128:
                if not _is_sha256(h128):
                    raise ExclusionAuditError(f"{canonical_label} eligible row lacks H128 proof")
                proof = ("h128",)
            else:
                proof = ("short",)
            result[record_id] = {(canonical_label, proof)}
        return result

    lineage: dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]] = {}
    for label in P11_ORIGINAL_LINEAGE_METADATA:
        lineage[label] = lineage_for(fit_rows, fit_bundle.label)
    for label in P11_VALIDATION_LINEAGE_METADATA:
        lineage[label] = lineage_for(validation_rows, validation_bundle.label)

    fit_proof.update({
        "handoff_path": P11_ORIGINAL_RECOVERY_HANDOFF_PATH,
        "handoff_sha256": P11_ORIGINAL_RECOVERY_HANDOFF_SHA256,
        "pre_amendment_handoff_sha256": P11_ORIGINAL_RECOVERY_HANDOFF_HISTORICAL_SHA256,
        "job": job_proofs["trr4_original_fit_1200"],
        "lineage_checks": [item for item in lineage_checks if item["label"] in fit_targets],
        "covered_source_labels": sorted(fit_targets),
        "canonical_lineage_join": "verified ordered record IDs, geometry, and source producer invariants",
    })
    validation_proof.update({
        "handoff_path": P11_ORIGINAL_RECOVERY_HANDOFF_PATH,
        "handoff_sha256": P11_ORIGINAL_RECOVERY_HANDOFF_SHA256,
        "pre_amendment_handoff_sha256": P11_ORIGINAL_RECOVERY_HANDOFF_HISTORICAL_SHA256,
        "job": job_proofs["trr4_original_validation_48"],
        "lineage_checks": [item for item in lineage_checks if item["label"] in P11_VALIDATION_LINEAGE_METADATA],
        "covered_source_labels": sorted(P11_VALIDATION_LINEAGE_METADATA),
        "canonical_lineage_join": "verified ordered record IDs and geometry; rendered commitment where present",
    })
    return [fit_bundle, validation_bundle], [fit_proof, validation_proof], lineage

def _load_h40_recovery_exports(
    root: Path,
) -> tuple[
    list[p10.IdentityBundle],
    list[dict[str, Any]],
    dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]],
    list[dict[str, Any]],
]:
    """Load the completed TRR-0001/TRR-0002 public H40 overlays.

    The two producers are intentionally separate.  TRR-0001 hashes the first
    40 BOS-inclusive IDs as little-endian signed-int32 bytes.  TRR-0002's
    historical Pile receipt hashes a tensor header followed by signed-int64
    bytes.  Neither value is treated as H128/H129, and the opaque exports do
    not retain source text or token values.
    """
    handoff_path, handoff = _verified_file_value(
        root,
        P11_H40_RECOVERY_HANDOFF_PATH,
        P11_H40_RECOVERY_HANDOFF_SHA256,
        P11_H40_RECOVERY_HANDOFF_BYTES,
        label="TRR-0001/TRR-0002 H40 recovery handoff",
    )
    if (
        handoff.get("schema") != "token-reconstruction.trr-p11-h40-recovery-handoff.v1"
        or handoff.get("task_id") != TASK_ID
        or handoff.get("status") != "PASS_TRR0001_AND_TRR0002_PUBLIC_H40_RECOVERY"
    ):
        raise ExclusionAuditError("TRR-0001/TRR-0002 H40 handoff status changed")
    access = handoff.get("access_boundary")
    if (
        not isinstance(access, Mapping)
        or access.get("source_text_read_transiently") is not True
        or any(
            access.get(key) is True
            for key in (
                "activations_read",
                "gpu_used",
                "model_loaded",
                "new_selection_started",
                "p03_holdout_accessed",
                "source_text_serialized",
                "source_tokens_serialized",
                "token_values_emitted",
                "truth_opened",
            )
        )
    ):
        raise ExclusionAuditError("H40 recovery handoff access boundary is unsafe")

    attempts_path, attempts = _verified_file_value(
        root,
        P11_H40_RECOVERY_ATTEMPTS_PATH,
        P11_H40_RECOVERY_ATTEMPTS_SHA256,
        P11_H40_RECOVERY_ATTEMPTS_BYTES,
        label="H40 recovery attempts receipt",
    )
    if (
        attempts.get("schema") != "token-reconstruction.trr-p11-h40-recovery-attempts.v1"
        or attempts.get("task_id") != TASK_ID
        or attempts.get("status") != "PASS_FINAL_ATTEMPT_R2_WITH_PRIOR_FAIL_PRESERVED"
    ):
        raise ExclusionAuditError("H40 recovery attempts provenance changed")
    attempt_rows = attempts.get("attempts")
    if (
        not isinstance(attempt_rows, list)
        or len(attempt_rows) != 2
        or attempt_rows[0].get("attempt") != "r1"
        or attempt_rows[0].get("status") != "CHILD_EXITED_NONZERO"
        or attempt_rows[1].get("attempt") != "r2"
        or attempt_rows[1].get("status") != "PASS"
    ):
        raise ExclusionAuditError("H40 prior failed attempt was not preserved")
    recipe_path = (root / P11_H40_RECOVERY_SCRIPT_PATH).resolve()
    if (
        not recipe_path.is_file()
        or recipe_path.is_symlink()
        or p10.sha256_file(recipe_path) != P11_H40_RECOVERY_SCRIPT_SHA256
    ):
        raise ExclusionAuditError("H40 recovery recipe is unavailable or changed")

    expected_sources = {
        "trr0001_manifest": ("experiments/TRR-0001/manifest.json", "80b6e7bae818729e06685b329f7270524623cb365817f8f318a3dac2bdac4dc0", 245971),
        "trr0001_plan": ("experiments/TRR-0001/plan.json", "b498a5db5b14ae8dde19f3ae4f519f86fdf0a67572a8f78747f7921e4f9e7269", 104718),
        "trr0001_revision_r1_manifest": ("experiments/TRR-0001/revision-r1/manifest.json", "1abdfe97ca9066f00ae30bfe3519da612bb288bb0358a904663c0518e3808c72", 99882),
        "trr0001_revision_r1_plan": ("experiments/TRR-0001/revision-r1/plan.json", "59944cb1e01ec2e88e04109e46db0088eece74500fe599eb6a62ead038b6fe14", 18970),
        "trr0001_revision_r1_reveal": ("experiments/TRR-0001/revision-r1/selection_reveal.json", "d96cae6d6c9beb29d99be34ecb597f4986f9e107212657f4ad98268676251b41", 10971),
        "trr0002_configuration_winner": ("experiments/TRR-0002/configuration-search/causal-selection/winner.json", "a75a5220647b0dea019cabb09be7c99a82f0d8142bf5476ffff6be5099fcb4f3", 18525),
        "trr0002_fresh_observation_index": ("experiments/TRR-0002/configuration-search/fresh-blind/observation-index.json", "df7248f4d6f6566bc9d9b687c8da10a84ca08cb94bc4f23a9449b5e83ee103d2", 3523),
        "trr0002_public_pile_records": ("experiments/TRR-0002/configuration-search/public-pile/records.json", "37be9bb6d621873047a05544e4e2d8ed7c07f1160ab878093ad912306ae1dd7f", 73510),
    }
    provenance = handoff.get("provenance_bindings")
    if not isinstance(provenance, Mapping) or set(provenance) != set(expected_sources):
        raise ExclusionAuditError("H40 source provenance binding set changed")
    source_values: dict[str, dict[str, Any]] = {}
    for label, (relative, expected_sha, expected_bytes) in expected_sources.items():
        declared = provenance.get(label)
        if not isinstance(declared, Mapping):
            raise ExclusionAuditError(f"H40 source provenance for {label} is malformed")
        declared_path = Path(str(declared.get("path", ""))).resolve()
        expected_path = (root / relative).resolve()
        if declared_path != expected_path:
            raise ExclusionAuditError(f"H40 source provenance path changed for {label}")
        if declared.get("sha256") != expected_sha or declared.get("bytes") != expected_bytes:
            raise ExclusionAuditError(f"H40 source provenance digest changed for {label}")
        _path, source_values[label] = _verified_file_value(
            root,
            relative,
            expected_sha,
            expected_bytes,
            label=f"H40 source {label}",
        )

    expected_receipts = {
        "trr0001": (P11_TRR0001_H40_RECOVERY_PATH, P11_TRR0001_H40_RECOVERY_SHA256, P11_TRR0001_H40_RECOVERY_BYTES, "token-reconstruction.trr-p11-trr0001-h40-recovery.v1", "PASS_TRR0001_PUBLIC_H40_RECOVERY"),
        "trr0002": (P11_TRR0002_H40_RECOVERY_PATH, P11_TRR0002_H40_RECOVERY_SHA256, P11_TRR0002_H40_RECOVERY_BYTES, "token-reconstruction.trr-p11-trr0002-h40-recovery.v1", "PASS_TRR0002_PUBLIC_PILE_H40_RECOVERY"),
    }
    handoff_receipts = handoff.get("receipts")
    handoff_records = handoff.get("records")
    if not isinstance(handoff_receipts, Mapping) or not isinstance(handoff_records, Mapping):
        raise ExclusionAuditError("H40 handoff lacks receipt/output bindings")
    identity_specs = {
        "trr0001": (P11_TRR0001_H40_IDENTITY_PATH, P11_TRR0001_H40_IDENTITY_SHA256, P11_TRR0001_H40_IDENTITY_BYTES, "token-reconstruction.trr-p11-trr0001-h40-identity-rows.v1", "PASS_TRR0001_PUBLIC_H40_RECOVERY", "trr0001_plan_and_revision_r1"),
        "trr0002": (P11_TRR0002_H40_IDENTITY_PATH, P11_TRR0002_H40_IDENTITY_SHA256, P11_TRR0002_H40_IDENTITY_BYTES, "token-reconstruction.trr-p11-trr0002-h40-identity-rows.v1", "PASS_TRR0002_PUBLIC_PILE_H40_RECOVERY", "trr0002_public_pile"),
    }
    identity_values: dict[str, dict[str, Any]] = {}
    recovery_values: dict[str, dict[str, Any]] = {}
    for key, (recovery_rel, recovery_sha, recovery_bytes, schema, status) in expected_receipts.items():
        receipt_binding = handoff_receipts.get(key)
        if not isinstance(receipt_binding, Mapping):
            raise ExclusionAuditError(f"H40 {key} receipt binding is missing")
        receipt_path = Path(str(receipt_binding.get("path", ""))).resolve()
        if receipt_path != (root / recovery_rel).resolve() or receipt_binding.get("sha256") != recovery_sha or receipt_binding.get("bytes") != recovery_bytes:
            raise ExclusionAuditError(f"H40 {key} receipt binding changed")
        _receipt_path, recovery = _verified_file_value(root, recovery_rel, recovery_sha, recovery_bytes, label=f"H40 {key} recovery receipt")
        if recovery.get("schema") != schema or recovery.get("task_id") != TASK_ID or recovery.get("status") != status:
            raise ExclusionAuditError(f"H40 {key} recovery status changed")
        coverage = recovery.get("coverage")
        if not isinstance(coverage, Mapping) or coverage.get("mismatch_count") != 0:
            raise ExclusionAuditError(f"H40 {key} recovery mismatch evidence changed")
        identity_rel, identity_sha, identity_bytes, identity_schema, identity_status, record_key = identity_specs[key]
        record_binding = handoff_records.get(record_key)
        record_path = Path(str(record_binding.get("path", ""))).resolve() if isinstance(record_binding, Mapping) else Path("/")
        if (
            not isinstance(record_binding, Mapping)
            or record_path != (root / identity_rel).resolve()
            or record_binding.get("sha256") != identity_sha
            or record_binding.get("bytes") != identity_bytes
        ):
            raise ExclusionAuditError(f"H40 {key} identity output binding changed")
        output = recovery.get("output")
        if not isinstance(output, Mapping) or output.get("sha256") != identity_sha or output.get("bytes") != identity_bytes or Path(str(output.get("path", ""))).name != Path(identity_rel).name:
            raise ExclusionAuditError(f"H40 {key} recovery output does not bind identity export")
        _identity_path, identity = _verified_file_value(root, identity_rel, identity_sha, identity_bytes, label=f"H40 {key} identity export")
        if identity.get("schema") != identity_schema or identity.get("task_id") != TASK_ID or identity.get("status") != identity_status:
            raise ExclusionAuditError(f"H40 {key} identity status changed")
        identity_access = identity.get("access_boundary")
        if (
            not isinstance(identity_access, Mapping)
            or any(identity_access.get(field) is True for field in ("source_text_serialized", "source_tokens_serialized", "token_values_emitted", "activations_read", "model_loaded", "gpu_used", "truth_opened", "p03_holdout_accessed", "new_selection_started"))
            or identity.get("contains_source_text") is not False
            or identity.get("contains_token_ids") is not False
            or identity.get("contains_truth") is not False
        ):
            raise ExclusionAuditError(f"H40 {key} identity access boundary is unsafe")
        identity_values[key] = identity
        recovery_values[key] = recovery

    producer_expectations = {
        "trr0001": {
            "source_label": "trr0001_plan_and_revision_r1_reveal_public_rerender",
            "dataset_id": "NeelNanda/pile-10k",
            "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
            "split": "train",
            "bos_token_id": 128000,
            "sequence_convention": "first 40 BOS-inclusive IDs, SHA-256 of little-endian signed-int32 bytes",
            "rows": 352,
            "groups": {"blind_evaluation": 64, "development": 32, "inverse_train": 128, "revision_r1_blind_evaluation": 64, "target_update_train": 64},
        },
        "trr0002": {
            "source_label": "trr0002_public_pile",
            "dataset_id": "NeelNanda/pile-10k",
            "revision": "127bfedcd5047750df5ccf3a12979a47bfa0bafa",
            "split": "train",
            "bos_token_id": 128000,
            "sequence_convention": "first 40 BOS-inclusive IDs; SHA-256 of canonical JSON tensor header {dtype: torch.int64, shape: [40]} followed by little-endian signed-int64 bytes",
            "rows": 96,
            "groups": {"development": 32, "update_train": 64},
        },
    }
    bundles: list[p10.IdentityBundle] = []
    proofs: list[dict[str, Any]] = []
    parsed_rows: dict[str, list[Mapping[str, Any]]] = {}
    for key, expected in producer_expectations.items():
        identity = identity_values[key]
        producer = identity.get("producer")
        if not isinstance(producer, Mapping) or any(producer.get(field) != value for field, value in expected.items() if field in {"dataset_id", "revision", "split", "bos_token_id", "sequence_convention"}) or producer.get("source_text_transient_only") is not True or producer.get("source_tokenization") != "pinned tokenizer, add_special_tokens=False":
            raise ExclusionAuditError(f"H40 {key} producer convention changed")
        rows = identity.get("records")
        if not isinstance(rows, list) or len(rows) != expected["rows"] or identity.get("record_count") != expected["rows"]:
            raise ExclusionAuditError(f"H40 {key} row count changed")
        groups = identity.get("groups")
        if groups != expected["groups"]:
            raise ExclusionAuditError(f"H40 {key} group counts changed")
        parsed_rows[key] = rows
        label = "trr0001_public_h40_identity" if key == "trr0001" else "trr0002_public_pile_h40_recovery_identity"
        bundle = p10.IdentityBundle(
            label,
            "inherited_fitting",
            (root / identity_specs[key][0]).resolve(),
            identity_specs[key][1],
            identity_specs[key][2],
            schema=str(identity["schema"]),
            status=str(identity["status"]),
        )
        namespace = p10.Namespace("pile", expected["dataset_id"], expected["split"], expected["revision"])
        seen_ids: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ExclusionAuditError(f"H40 {key} row {index} is malformed")
            record_id = row.get("record_id")
            rendered = row.get("rendered_sha256")
            source_index = row.get("source_index")
            h40_key = "h40_sequence_sha256" if key == "trr0001" else "trr0002_h40_token_ids_sha256"
            h40 = row.get(h40_key)
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in seen_ids
                or not _is_sha256(rendered)
                or not _is_sha256(h40)
                or not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or source_index < 0
                or row.get("dataset_id") != expected["dataset_id"]
                or row.get("revision") != expected["revision"]
                or row.get("split") != expected["split"]
                or row.get("full_token_count") != 40
                or row.get("post_bos_token_count") != 39
            ):
                raise ExclusionAuditError(f"H40 {key} row {index} violates the producer identity contract")
            if "h128_sequence_sha256" in row or "h129_sequence_sha256" in row:
                raise ExclusionAuditError(f"H40 {key} row {index} is mislabelled as H128/H129")
            seen_ids.add(record_id)
            bundle.add("record_id", record_id, namespace)
            bundle.add("rendered_sha256", rendered.casefold(), namespace)
            bundle.add(h40_key, h40.casefold(), namespace)
            bundle.add("source_index", source_index, namespace)
        bundle.metadata_notes.append(
            "Producer-verified public H40 overlay; source text/token IDs were transiently used for opaque hashing only"
        )
        bundles.append(bundle)
        coverage = recovery_values[key].get("coverage", {})
        proofs.append({
            "label": label,
            "role": bundle.role,
            "identity_path": identity_specs[key][0],
            "identity_sha256": identity_specs[key][1],
            "recovery_path": expected_receipts[key][0],
            "recovery_sha256": expected_receipts[key][1],
            "identity_schema": identity["schema"],
            "record_count": len(rows),
            "identity_counts": bundle.counts(),
            "h40_rows": len(rows),
            "h128_rows": 0,
            "h129_rows": 0,
            "groups": dict(groups),
            "producer": dict(producer),
            "coverage": dict(coverage),
            "handoff_path": P11_H40_RECOVERY_HANDOFF_PATH,
            "handoff_sha256": P11_H40_RECOVERY_HANDOFF_SHA256,
            "attempts_path": P11_H40_RECOVERY_ATTEMPTS_PATH,
            "attempts_sha256": P11_H40_RECOVERY_ATTEMPTS_SHA256,
            "recipe_path": P11_H40_RECOVERY_SCRIPT_PATH,
            "recipe_sha256": P11_H40_RECOVERY_SCRIPT_SHA256,
            "access_boundary": dict(identity.get("access_boundary", {})),
        })

    # Exact row-level producer checks against the source ledgers bound by the
    # handoff.  The checks compare only IDs, rendered commitments, indices and
    # counts; no source text or token value is emitted.
    trr1_plan = source_values["trr0001_plan"]
    plan_splits = trr1_plan.get("data", {}).get("selection", {}).get("splits", {})
    if not isinstance(plan_splits, Mapping):
        raise ExclusionAuditError("TRR-0001 plan selection splits are unavailable")
    trr1_reveal = source_values["trr0001_revision_r1_reveal"]
    reveal_rows = trr1_reveal.get("records")
    if not isinstance(reveal_rows, list):
        raise ExclusionAuditError("TRR-0001 revision reveal rows are unavailable")
    trr1_by_id = {row.get("record_id"): row for row in parsed_rows["trr0001"]}
    if len(trr1_by_id) != 352:
        raise ExclusionAuditError("TRR-0001 H40 record IDs are not unique")
    group_sources = {
        "blind_evaluation": (plan_splits.get("blind_evaluation"), "index"),
        "development": (plan_splits.get("development"), "index"),
        "inverse_train": (plan_splits.get("inverse_train"), "index"),
        "target_update_train": (plan_splits.get("target_update_train"), "index"),
    }
    lineage_checks: list[dict[str, Any]] = []
    for group, (source_split, index_key) in group_sources.items():
        source_rows = source_split.get("records") if isinstance(source_split, Mapping) else None
        if not isinstance(source_rows, list):
            raise ExclusionAuditError(f"TRR-0001 plan split {group} is unavailable")
        matched = 0
        for source_row in source_rows:
            row = trr1_by_id.get(source_row.get("record_id"))
            if row is None or row.get("rendered_sha256") != source_row.get("text_sha256") or row.get("source_index") != source_row.get(index_key):
                raise ExclusionAuditError(f"TRR-0001 H40 row does not match plan split {group}")
            if row.get("source_group") != group:
                raise ExclusionAuditError(f"TRR-0001 H40 group label changed for {group}")
            matched += 1
        lineage_checks.append({"source": "trr0001_plan", "group": group, "rows": matched, "ids_rendered_and_indices_match": True})
    matched = 0
    for source_row in reveal_rows:
        row = trr1_by_id.get(source_row.get("record_id"))
        if row is None or row.get("rendered_sha256") != source_row.get("text_sha256") or row.get("source_index") != source_row.get("dataset_index") or row.get("source_group") != "revision_r1_blind_evaluation":
            raise ExclusionAuditError("TRR-0001 H40 row does not match revision-r1 reveal")
        matched += 1
    lineage_checks.append({"source": "trr0001_revision_r1_reveal", "group": "revision_r1_blind_evaluation", "rows": matched, "ids_rendered_and_indices_match": True})

    trr2_public = source_values["trr0002_public_pile_records"]
    trr2_by_group = {group: trr2_public.get(group) for group in ("development", "update_train")}
    trr2_by_id = {row.get("record_id"): row for rows in trr2_by_group.values() if isinstance(rows, list) for row in rows if isinstance(row, Mapping)}
    trr2_identity_by_id = {row.get("record_id"): row for row in parsed_rows["trr0002"]}
    if len(trr2_by_id) != 96 or len(trr2_identity_by_id) != 96:
        raise ExclusionAuditError("TRR-0002 public Pile row count changed")
    for group, source_rows in trr2_by_group.items():
        if not isinstance(source_rows, list):
            raise ExclusionAuditError(f"TRR-0002 public Pile group {group} is unavailable")
        matched = 0
        for source_row in source_rows:
            row = trr2_identity_by_id.get(source_row.get("record_id"))
            if row is None or row.get("rendered_sha256") != source_row.get("text_sha256") or row.get("source_index") != source_row.get("dataset_index") or row.get("source_group") != group:
                raise ExclusionAuditError(f"TRR-0002 H40 row does not match public Pile group {group}")
            matched += 1
        lineage_checks.append({"source": "trr0002_public_pile_records", "group": group, "rows": matched, "ids_rendered_and_indices_match": True})

    # TRR-0002's fresh observation index is an opaque view of the same ordered
    # revision-r1 blind IDs.  The exact ordered-ID equality plus the recovered
    # TRR-0001 rendered/H40 commitments is the producer proof; an arbitrary
    # record-ID collision would not pass this check.
    observation = source_values["trr0002_fresh_observation_index"]
    observation_rows = observation.get("records")
    reveal_ids = [row.get("record_id") for row in reveal_rows]
    observation_ids = [row.get("record_id") for row in observation_rows] if isinstance(observation_rows, list) else []
    revision_rows = [row for row in parsed_rows["trr0001"] if row.get("source_group") == "revision_r1_blind_evaluation"]
    revision_ids = [row.get("record_id") for row in revision_rows]
    if observation.get("source_material_included") is not False or observation_ids != reveal_ids or set(observation_ids) != set(revision_ids):
        raise ExclusionAuditError("TRR-0002 fresh observation index is not exactly bound to the recovered revision-r1 blind rows")
    fresh_bundle = bundles[0]
    verified_lineage: dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]] = {
        "trr0002_fresh_observation_index": {
            record_id: {(fresh_bundle.label, ("h40",))}
            for record_id in observation_ids
            if isinstance(record_id, str)
        }
    }
    lineage_proofs = [{
        "label": "trr0002_fresh_observation_index",
        "status": "PASS_EXACT_ORDERED_ID_AND_RECOVERED_H40_LINEAGE",
        "source_path": "experiments/TRR-0002/configuration-search/fresh-blind/observation-index.json",
        "source_sha256": expected_sources["trr0002_fresh_observation_index"][1],
        "covered_identity_source": P11_TRR0001_H40_IDENTITY_PATH,
        "rows": len(observation_ids),
        "ordered_ids_match_revision_r1_reveal": True,
        "ordered_ids_match_recovered_trr0001_h40": True,
        "source_material_included": False,
    }]
    proofs[0]["lineage_checks"] = lineage_checks
    proofs[0]["fresh_observation_lineage"] = lineage_proofs[0]
    return bundles, proofs, verified_lineage, lineage_proofs


def _load_trr0002_winner_lineage(
    root: Path,
    bundles: Sequence[p10.IdentityBundle],
    h40_bundles: Sequence[p10.IdentityBundle],
) -> tuple[dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]], dict[str, Any]]:
    """Bind method-prefixed winner IDs to their own canonical public rows.

    TRR-0002's frozen winner uses ``public_lora_2602:<public-record-id>``.
    The prefix is accepted only after the immutable preregistration rule and
    the exact records 16--31 in each public domain are checked.  The target
    row must itself carry a verified H128 (Finance) or producer-bound H40
    (Pile) proof.
    """
    winner_path = (root / "experiments/TRR-0002/configuration-search/causal-selection/winner.json").resolve()
    winner = _read_json(winner_path, label="TRR-0002 frozen winner")
    plan_binding = winner.get("plan")
    if not isinstance(plan_binding, Mapping):
        raise ExclusionAuditError("TRR-0002 winner plan binding is missing")
    plan_rel = str(plan_binding.get("path", ""))
    if plan_rel != "experiments/TRR-0002/configuration-search/preregistration/plan.json" or not _is_sha256(plan_binding.get("sha256")):
        raise ExclusionAuditError("TRR-0002 winner plan binding changed")
    plan_path = (root / plan_rel).resolve()
    if not plan_path.is_file() or plan_path.is_symlink() or p10.sha256_file(plan_path) != plan_binding.get("sha256") or plan_path.stat().st_size != plan_binding.get("bytes"):
        raise ExclusionAuditError("TRR-0002 winner plan is unavailable or changed")
    plan = _read_json(plan_path, label="TRR-0002 preregistration plan")
    public = plan.get("public_data")
    roles = public.get("roles") if isinstance(public, Mapping) else None
    finance = public.get("finance") if isinstance(public, Mapping) else None
    pile = public.get("pile") if isinstance(public, Mapping) else None
    if (
        winner.get("selection_condition") != "public_lora_2602"
        or winner.get("selection_records") != [16, 32]
        or not isinstance(roles, Mapping)
        or roles.get("causal_selection") != "records 16-31 in both domains under public_lora_2602"
        or not isinstance(finance, Mapping)
        or finance.get("raw_cursor_start") != 38978
        or finance.get("selection") != "take the first 32 nonempty user/assistant rows at or after raw cursor 38978, use the historical 06 Aug 2026 Llama chat template, truncate/right-pad to 128, and record every raw index and content/token hash"
        or not isinstance(pile, Mapping)
        or pile.get("geometry") != "32 records x 40 tokens including BOS"
    ):
        raise ExclusionAuditError("TRR-0002 winner selection rule changed")

    by_label = {bundle.label: bundle for bundle in bundles}
    finance_bundle = by_label.get("trr0002_public_finance_records")
    pile_bundle = next((bundle for bundle in h40_bundles if bundle.label == "trr0002_public_pile_h40_recovery_identity"), None)
    if finance_bundle is None or pile_bundle is None:
        raise ExclusionAuditError("TRR-0002 winner canonical source bundles are unavailable")
    finance_value = _read_json(finance_bundle.path, label="TRR-0002 Finance public rows")
    pile_value = _read_json((root / "experiments/TRR-0002/configuration-search/public-pile/records.json").resolve(), label="TRR-0002 Pile public rows")
    finance_rows = finance_value.get("records")
    pile_rows = pile_value.get("development")
    if not isinstance(finance_rows, list) or len(finance_rows) != 32 or not isinstance(pile_rows, list) or len(pile_rows) != 32:
        raise ExclusionAuditError("TRR-0002 winner public source row counts changed")
    metric_ids: dict[str, list[str]] = {}
    selection_metrics = winner.get("selection_metrics")
    if not isinstance(selection_metrics, Mapping):
        raise ExclusionAuditError("TRR-0002 winner selection metrics are missing")
    for domain in ("finance", "pile"):
        metrics = selection_metrics.get(domain)
        rows = metrics.get("per_record") if isinstance(metrics, Mapping) else None
        ids = [row.get("record_id") for row in rows] if isinstance(rows, list) else []
        if len(ids) != 16 or not all(isinstance(item, str) and item.startswith("public_lora_2602:") for item in ids) or len(set(ids)) != 16:
            raise ExclusionAuditError(f"TRR-0002 winner {domain} rows changed")
        metric_ids[domain] = ids
    canonical_finance = {row.get("record_id"): row for row in finance_rows if isinstance(row, Mapping)}
    canonical_pile = {row.get("record_id"): row for row in pile_rows if isinstance(row, Mapping)}
    pile_identity_rows = {row.get("record_id"): row for row in _record_rows(_read_json(pile_bundle.path, label="TRR-0002 H40 identity export"))}
    lineage: dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]] = {"trr0002_configuration_winner": {}}
    source_counts = Counter()
    for domain, ids in metric_ids.items():
        canonical = canonical_finance if domain == "finance" else canonical_pile
        for winner_id in ids:
            suffix = winner_id.split(":", 1)[1]
            row = canonical.get(suffix)
            if row is None:
                raise ExclusionAuditError(f"TRR-0002 winner ID lacks canonical public source row: {domain}")
            if domain == "pile":
                identity_row = pile_identity_rows.get(suffix)
                if identity_row is None or identity_row.get("rendered_sha256") != row.get("text_sha256") or identity_row.get("source_index") != row.get("dataset_index"):
                    raise ExclusionAuditError("TRR-0002 Pile winner row does not match recovered H40 identity")
                reference = (pile_bundle.label, ("h40",))
            else:
                _namespace, fields = _row_identity_fields(row, source_label=finance_bundle.label)
                proof = _row_canonical_proof(finance_bundle, row, fields)
                if not ({"h128", "short"} & set(proof)):
                    raise ExclusionAuditError("TRR-0002 Finance winner row lacks own H128 or verified-short proof")
                reference = (finance_bundle.label, tuple(sorted(proof)))
            lineage["trr0002_configuration_winner"][winner_id] = {reference}
            source_counts[domain] += 1
    proof = {
        "label": "trr0002_configuration_winner",
        "status": "PASS_EXACT_METHOD_PREFIX_LINEAGE",
        "winner_path": "experiments/TRR-0002/configuration-search/causal-selection/winner.json",
        "winner_sha256": p10.sha256_file(winner_path),
        "plan_path": plan_rel,
        "plan_sha256": plan_binding.get("sha256"),
        "prefix": "public_lora_2602:",
        "selection_rule": "exact frozen records 16-31 in each 32-row public domain",
        "rows": sum(source_counts.values()),
        "finance_rows": source_counts["finance"],
        "pile_rows": source_counts["pile"],
        "canonical_sources": [finance_bundle.label, pile_bundle.label],
        "canonical_sequence_proof": {"finance": "own H128 or verified short geometry", "pile": "recovered producer-bound TRR2 H40"},
    }
    return lineage, proof

def _load_p04_recipe_migration(root: Path) -> dict[str, Any]:
    """Verify the persistent entrypoint migration without rerunning P04."""
    path = (root / P04_RECIPE_MIGRATION_PATH).resolve()
    if not path.is_file() or path.is_symlink() or p10.sha256_file(path) != P04_RECIPE_MIGRATION_SHA256:
        raise ExclusionAuditError("P04 persistent recipe migration receipt is unavailable or changed")
    value = _read_json(path, label="P04 recipe migration")
    if value.get("schema") != "token-reconstruction.trr-p11-p04-recovery-recipe-migration.v1" or value.get("task_id") != TASK_ID or value.get("status") != "PASS_RECIPE_PATH_MIGRATION_NO_RERUN":
        raise ExclusionAuditError("P04 recipe migration status changed")
    historical = value.get("historical_execution")
    persistent = value.get("persistent_entrypoint")
    checks = value.get("migration_checks")
    if not isinstance(historical, Mapping) or not isinstance(persistent, Mapping) or not isinstance(checks, Mapping):
        raise ExclusionAuditError("P04 recipe migration blocks are malformed")
    if historical.get("receipt_path") != P04_H128_RECOVERY_PATH or historical.get("receipt_sha256") != P04_H128_RECOVERY_SHA256 or historical.get("identity_export_path") != P04_H128_IDENTITY_PATH or historical.get("identity_export_sha256") != P04_H128_IDENTITY_SHA256 or historical.get("executed_recovery_script_sha256") != P04_H128_RECOVERY_SCRIPT_SHA256:
        raise ExclusionAuditError("P04 historical recovery binding changed")
    script_path = (root / str(persistent.get("path", ""))).resolve()
    selection_path = (root / P04_PERSISTENT_SELECTION_PATH).resolve()
    helper_path = (root / P04_PERSISTENT_ALPACA_HELPER_PATH).resolve()
    if script_path != (root / "scripts/trr_p11/recover_p04_public.py").resolve() or not script_path.is_file() or script_path.is_symlink() or p10.sha256_file(script_path) != P04_PERSISTENT_RECOVERY_SCRIPT_SHA256:
        raise ExclusionAuditError("P04 persistent recovery script binding changed")
    if "/tmp/" in script_path.read_text(encoding="utf-8"):
        raise ExclusionAuditError("P04 persistent recovery script still depends on /tmp")
    if not selection_path.is_file() or selection_path.is_symlink() or p10.sha256_file(selection_path) != P04_PERSISTENT_SELECTION_SHA256:
        raise ExclusionAuditError("P04 persistent selection metadata binding changed")
    if not helper_path.is_file() or helper_path.is_symlink() or p10.sha256_file(helper_path) != P04_PERSISTENT_ALPACA_HELPER_SHA256:
        raise ExclusionAuditError("P04 persistent Alpaca helper binding changed")
    expected = {
        "selection_bytes_equal_expected": True,
        "alpaca_helper_bytes_equal_expected": True,
        "historical_receipt_and_identity_preserved": True,
        "new_tokenization_or_source_scan": False,
        "new_selection_started": False,
        "p03_holdout_accessed": False,
    }
    if any(checks.get(key) is not expected_value for key, expected_value in expected.items()):
        raise ExclusionAuditError("P04 recipe migration access checks changed")
    return {
        "status": value["status"],
        "path": P04_RECIPE_MIGRATION_PATH,
        "sha256": P04_RECIPE_MIGRATION_SHA256,
        "historical_execution": dict(historical),
        "persistent_entrypoint": dict(persistent),
        "verified_selection_sha256": P04_PERSISTENT_SELECTION_SHA256,
        "verified_alpaca_helper_sha256": P04_PERSISTENT_ALPACA_HELPER_SHA256,
        "new_tokenization_or_source_scan": False,
    }

def _load_p04_h128_identity(root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load the sanitized, producer-verified P04 H128 row export.

    The export contains only opaque identity fields and geometry.  Its paired
    recovery receipt and exact script hash are bound before any value enters
    the union.  Source text and token IDs are deliberately absent.
    """
    migration_proof = _load_p04_recipe_migration(root)
    identity_path = (root / P04_H128_IDENTITY_PATH).resolve()
    recovery_path = (root / P04_H128_RECOVERY_PATH).resolve()
    if not identity_path.is_file() or identity_path.is_symlink():
        raise ExclusionAuditError(f"P04 H128 identity export is unavailable: {identity_path}")
    if p10.sha256_file(identity_path) != P04_H128_IDENTITY_SHA256:
        raise ExclusionAuditError("P04 H128 identity export SHA-256 changed")
    if not recovery_path.is_file() or recovery_path.is_symlink():
        raise ExclusionAuditError(f"P04 H128 recovery receipt is unavailable: {recovery_path}")
    if p10.sha256_file(recovery_path) != P04_H128_RECOVERY_SHA256:
        raise ExclusionAuditError("P04 H128 recovery receipt SHA-256 changed")
    identity = _read_json(identity_path, label="P04 H128 identity export")
    recovery = _read_json(recovery_path, label="P04 H128 recovery receipt")
    if identity.get("schema") != "token-reconstruction.trr-p11-p04-h128-identity-rows.v1" or identity.get("task_id") != TASK_ID:
        raise ExclusionAuditError("P04 H128 identity export schema changed")
    if identity.get("status") != "PASS_P04_EXACT_RENDERED_H129_H128_RECOVERY" or identity.get("record_count") != 520:
        raise ExclusionAuditError("P04 H128 identity export is not a complete 520-row pass")
    if identity.get("source_selection_sha256") != "05f941e0dbcf29ea3efc47c7bc8abb3a7146a266eeea770f05052bb7728cde6a":
        raise ExclusionAuditError("P04 H128 identity export source selection binding changed")
    if identity.get("recovery_script_sha256") != P04_H128_RECOVERY_SCRIPT_SHA256:
        raise ExclusionAuditError("P04 H128 identity export recovery script binding changed")
    forbidden = {"source_text", "source_tokens", "token_ids", "input_ids", "labels", "truth", "oracle", "target_weights"}
    if any(key in identity for key in forbidden):
        raise ExclusionAuditError("P04 H128 identity export contains forbidden payload")
    if recovery.get("status") != "PASS_P04_EXACT_RENDERED_H129_H128_RECOVERY" or recovery.get("mismatch_count") != 0:
        raise ExclusionAuditError("P04 H128 recovery receipt is not a zero-mismatch pass")
    if recovery.get("source_code", {}).get("recovery_script_sha256") != P04_H128_RECOVERY_SCRIPT_SHA256:
        raise ExclusionAuditError("P04 H128 recovery receipt script binding changed")
    identity_descriptor = recovery.get("identity_export")
    if not isinstance(identity_descriptor, Mapping) or identity_descriptor.get("sha256") != P04_H128_IDENTITY_SHA256:
        raise ExclusionAuditError("P04 H128 recovery receipt does not bind identity export")
    access = identity.get("access_boundary")
    if not isinstance(access, Mapping) or any(access.get(key) is True for key in ("source_text_serialized", "source_tokens_serialized", "token_values_emitted", "evaluation_truth_opened", "target_update_opened", "model_loaded", "gpu_used", "p03_holdout_accessed", "new_selection_started")):
        raise ExclusionAuditError("P04 H128 identity export access boundary is unsafe")
    rows = identity.get("records")
    if not isinstance(rows, list) or len(rows) != 520:
        raise ExclusionAuditError("P04 H128 identity export rows are incomplete")
    grouped: dict[str, list[dict[str, Any]]] = {"correction": [], "validation": [], "fresh_evaluation": []}
    required = {"pool", "style", "dataset_id", "dataset_revision", "row_index", "record_id", "public_record_sha256", "h128_sequence_sha256", "h129_sequence_sha256", "rendered_char_count", "full_token_count", "post_bos_token_count"}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != required:
            raise ExclusionAuditError(f"P04 H128 identity row {index} fields changed")
        pool = row.get("pool")
        if pool == "public_correction":
            pool_key = "correction"
        elif pool == "public_validation":
            pool_key = "validation"
        elif pool == "fresh_evaluation":
            pool_key = "fresh_evaluation"
        else:
            raise ExclusionAuditError(f"P04 H128 identity row {index} pool is malformed")
        if not all(isinstance(row.get(key), str) and row.get(key) for key in ("style", "dataset_id", "dataset_revision", "record_id")):
            raise ExclusionAuditError(f"P04 H128 identity row {index} metadata is malformed")
        if not isinstance(row.get("row_index"), int) or not isinstance(row.get("rendered_char_count"), int) or not isinstance(row.get("full_token_count"), int) or not isinstance(row.get("post_bos_token_count"), int):
            raise ExclusionAuditError(f"P04 H128 identity row {index} geometry is malformed")
        if int(row["full_token_count"]) != int(row["post_bos_token_count"]) + 1 or int(row["full_token_count"]) < 128:
            raise ExclusionAuditError(f"P04 H128 identity row {index} geometry is not eligible")
        for field in ("public_record_sha256", "h128_sequence_sha256", "h129_sequence_sha256"):
            if not _is_sha256(row.get(field)):
                raise ExclusionAuditError(f"P04 H128 identity row {index} {field} is malformed")
        grouped[pool_key].append(dict(row))
    expected_counts = {"correction": 256, "validation": 192, "fresh_evaluation": 72}
    if {key: len(value) for key, value in grouped.items()} != expected_counts:
        raise ExclusionAuditError("P04 H128 identity pool counts changed")
    proof = {
        "available": True,
        "path": P04_H128_IDENTITY_PATH,
        "sha256": P04_H128_IDENTITY_SHA256,
        "recovery_receipt_path": P04_H128_RECOVERY_PATH,
        "recovery_receipt_sha256": P04_H128_RECOVERY_SHA256,
        "recovery_script_sha256": P04_H128_RECOVERY_SCRIPT_SHA256,
        "record_count": len(rows),
        "pool_counts": expected_counts,
        "fields": sorted(required),
        "source_text_or_tokens_emitted": False,
        "recipe_migration": migration_proof,
    }
    return grouped, proof


def _consumer_proof(root: Path) -> dict[str, Any]:
    """Record surviving downstream proof for the H129 convention."""
    files = (
        "scripts/trr0005_produce_confirmation.py",
        "scripts/trr0008_select_public.py",
        "scripts/trr0009_select_public.py",
    )
    records: list[dict[str, Any]] = []
    for relative in files:
        path = root / relative
        if not path.is_file():
            records.append({"path": relative, "available": False})
            continue
        text = path.read_text(encoding="utf-8")
        if relative.endswith("produce_confirmation.py"):
            convention = "_sequence_digest" in text and "torch.int32" in text
        else:
            convention = "SEQUENCE_TOKENS + 1" in text and "_p06_sequence_digest" in text
        records.append(
            {
                "path": relative,
                "available": True,
                "sha256": p10.sha256_file(path),
                "h129_consumer_pattern_verified": convention,
            }
        )
    return {"files": records, "all_h129_consumer_patterns_verified": all(item.get("h129_consumer_pattern_verified") is True for item in records)}


def load_p04(root: Path) -> P04Load:
    path = (root / P04_PATH).resolve()
    if not path.is_file() or path.is_symlink():
        raise ExclusionAuditError(f"required P04 exchange is unavailable: {path}")
    actual_sha = p10.sha256_file(path)
    if actual_sha != P04_SHA256:
        raise ExclusionAuditError("P04 exchange SHA-256 changed")
    value = _read_json(path, label="P04 exchange")
    if value.get("schema") != "token-reconstruction.trr-p04-reservation-hashes.v1":
        raise ExclusionAuditError("P04 exchange schema changed")
    if value.get("task_id") != "TRR-P04":
        raise ExclusionAuditError("P04 exchange task identity changed")
    if not _is_sha256(value.get("exchange_digest_sha256")):
        raise ExclusionAuditError("P04 exchange digest is malformed")
    conventions = value.get("hash_conventions")
    if not isinstance(conventions, Mapping):
        raise ExclusionAuditError("P04 hash conventions are absent")
    required_fragments = {
        "canonical_json_sha256": "json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)",
        "ordered_opaque_value_sha256": "SHA256 of each declared opaque UTF-8 string followed by LF",
        "ordered_pair_sha256": "canonical JSON {'public_record_sha256': value, 'truncated_sequence_sha256': value}",
        "unique_set_canonical_json_sha256": "SHA256 of canonical JSON for the sorted unique opaque-string set",
    }
    convention_checks = {
        key: isinstance(conventions.get(key), str) and fragment in conventions[key]
        for key, fragment in required_fragments.items()
    }
    seq = conventions.get("sequence_fingerprint_generation")
    binary = conventions.get("sequence_fingerprint_generation", {}).get("binary_encoding", "") if isinstance(seq, Mapping) else ""
    panel = conventions.get("sequence_fingerprint_generation", {}).get("public_panel", "") if isinstance(seq, Mapping) else ""
    convention_checks.update(
        {
            "signed_int32_binary": "struct.pack('<' + 'i'*N" in binary,
            "bos_plus_128_h129": "token_ids[:1+128]" in panel and "BOS" in panel,
        }
    )
    if not all(convention_checks.values()):
        raise ExclusionAuditError("P04 declared hash convention changed")

    spec = p10.SourceSpec("trr0006_p04_opaque", "opaque_reservation", P04_PATH)
    bundle = _new_bundle(spec, path, value)
    gaps: list[dict[str, Any]] = []
    pool_proof: dict[str, Any] = {}
    ledger_proofs: dict[str, Any] = {}
    p04_h128_rows, p04_h128_export_proof = _load_p04_h128_identity(root)
    reservations = value.get("reservations")
    if not isinstance(reservations, Mapping):
        raise ExclusionAuditError("P04 reservations are absent")
    for pool in ("correction", "validation", "fresh_evaluation"):
        reservation = reservations.get(pool)
        if not isinstance(reservation, Mapping):
            raise ExclusionAuditError(f"P04 {pool} reservation is absent")
        hashes = reservation.get("hashes", {}).get("individual_opaque_hashes") if isinstance(reservation.get("hashes"), Mapping) else None
        if not isinstance(hashes, Mapping):
            raise ExclusionAuditError(f"P04 {pool} individual opaque hashes are absent")
        public = _verify_p04_opaque_summary(hashes.get("public_record_sha256", {}), label=f"{pool}.public_record_sha256")
        truncated = _verify_p04_opaque_summary(hashes.get("truncated_sequence_sha256", {}), label=f"{pool}.truncated_sequence_sha256")
        pair = _verify_p04_opaque_summary(hashes.get("ordered_public_sequence_pair_sha256", {}), label=f"{pool}.ordered_pair")
        if len(public) != len(truncated) or len(public) != len(pair):
            raise ExclusionAuditError(f"P04 {pool} paired opaque counts disagree")
        reservation_hashes = reservation.get("hashes")
        if not isinstance(reservation_hashes, Mapping):
            raise ExclusionAuditError(f"P04 {pool} hash block is malformed")
        individual = reservation_hashes.get("individual_opaque_hashes")
        if not isinstance(individual, Mapping):
            raise ExclusionAuditError(f"P04 {pool} individual hash block is malformed")
        individual_digest = reservation_hashes.get("individual_opaque_hashes_digest_sha256")
        if individual_digest != _json_digest(individual):
            raise ExclusionAuditError(f"P04 {pool} individual hash digest mismatch")
        aggregate = {
            key: reservation_hashes.get(key)
            for key in ("opaque_record_ids", "opaque_public_record_fingerprints", "opaque_truncated_sequence_fingerprints")
        }
        reservation_digest = reservation_hashes.get("reservation_digest_sha256")
        if reservation_digest != _json_digest({"aggregate": aggregate, "individual_opaque_hashes_digest_sha256": individual_digest}):
            raise ExclusionAuditError(f"P04 {pool} reservation digest mismatch")
        namespace = p10.Namespace("*", "*", "*", "*")
        for digest in public:
            bundle.add("rendered_sha256", digest, namespace)
        for digest in truncated:
            bundle.add("h129_sequence_sha256", digest, namespace)
        ledger, ledger_proof = _recover_p04_ledger(root, pool)
        ledger_proofs[pool] = ledger_proof
        if ledger is None:
            gaps.append({"field": f"{pool}.individual_record_identity", "reason": "historical P04 ledger Git object is unavailable"})
        else:
            rows = ledger["records"]
            row_public = [str(row["public_record_sha256"]).casefold() for row in rows]
            row_truncated = [str(row["truncated_sequence_sha256"]).casefold() for row in rows]
            if row_public != public or row_truncated != truncated:
                raise ExclusionAuditError(f"P04 {pool} opaque arrays do not bind recovered ledger order")
            row_pairs = [_json_digest({"public_record_sha256": a, "truncated_sequence_sha256": b}) for a, b in zip(row_public, row_truncated)]
            if row_pairs != pair:
                raise ExclusionAuditError(f"P04 {pool} ordered pair array does not bind recovered ledger order")
            eligible_by_length = 0
            for row in rows:
                if int(row["full_token_count"]) != int(row["post_bos_token_count"]) + 1:
                    raise ExclusionAuditError(f"P04 {pool} full/post-BOS length identity changed")
                style = p10._style(row.get("style")) or "*"
                row_namespace = p10.Namespace(style, str(row.get("dataset_id", "*")), "train", str(row.get("dataset_revision", "*")))
                bundle.add("record_id", str(row["record_id"]), row_namespace)
                # rendered/H129 are global fields; the wildcard entries above
                # already carry them.  Do not duplicate global values per row
                # namespace in the receipt counts.
                bundle.add("source_index", int(row["row_index"]), row_namespace)
                if int(row["full_token_count"]) >= 128:
                    eligible_by_length += 1
            ledger_proof["eligible_rows_by_length_for_h128"] = eligible_by_length
            recovered_h128 = p04_h128_rows.get(pool)
            if recovered_h128 is None or len(recovered_h128) != len(rows):
                raise ExclusionAuditError(f"P04 {pool} H128 identity export count does not bind recovered ledger")
            for row, recovered in zip(rows, recovered_h128):
                expected_identity = (str(row["record_id"]), int(row["row_index"]), str(row.get("dataset_id", "*")), str(row.get("dataset_revision", "*")))
                actual_identity = (str(recovered["record_id"]), int(recovered["row_index"]), str(recovered["dataset_id"]), str(recovered["dataset_revision"]))
                if expected_identity != actual_identity:
                    raise ExclusionAuditError(f"P04 {pool} H128 identity export does not bind ledger row")
                if str(recovered["public_record_sha256"]).casefold() != str(row["public_record_sha256"]).casefold() or str(recovered["h129_sequence_sha256"]).casefold() != str(row["truncated_sequence_sha256"]).casefold():
                    raise ExclusionAuditError(f"P04 {pool} H128 identity export does not bind published H129/rendered row")
                bundle.add("h128_sequence_sha256", str(recovered["h128_sequence_sha256"]).casefold(), p10.Namespace())
            if len({str(row["h128_sequence_sha256"]).casefold() for row in recovered_h128}) != len(recovered_h128):
                raise ExclusionAuditError(f"P04 {pool} H128 identity export contains duplicate sequence fingerprints")
            ledger_proof["h128_values_present"] = True
            ledger_proof["h128_identity_export_bound"] = True
        pool_proof[pool] = {
            "public_record_sha256": {"available": True, "count": len(public)},
            "truncated_sequence_sha256": {"available": True, "count": len(truncated), "mapping": "H129 = BOS plus 128 post-BOS IDs"},
            "h128_sequence_sha256": {"available": ledger is not None, "count": len(p04_h128_rows[pool]) if ledger is not None else 0, "mapping": "H128 = first 128 active BOS-inclusive IDs after exact public rerender and H129/rendered checks"},
            "ordered_pair": {"available": True, "count": len(pair)},
            "recovered_individual_ledger": ledger_proof,
        }
    fit_replay = reservations.get("fit_replay")
    fit_hashes = fit_replay.get("individual_opaque_hashes") if isinstance(fit_replay, Mapping) else None
    if not isinstance(fit_hashes, Mapping):
        raise ExclusionAuditError("P04 fit_replay individual hashes are absent")
    fit_rendered = fit_hashes.get("rendered_sha256")
    if not isinstance(fit_rendered, Mapping):
        raise ExclusionAuditError("P04 fit_replay rendered hash descriptor is absent")
    fit_values = _verify_p04_opaque_summary(fit_rendered, label="fit_replay.rendered_sha256")
    for digest in fit_values:
        bundle.add("rendered_sha256", digest, p10.Namespace())
    gaps.extend(
        [
            {"field": "fit_replay.public_record_sha256", "reason": "replay metadata exposes rendered_sha256 under a different field; no per-record public field mapping is emitted"},
            {"field": "fit_replay.truncated_sequence_sha256", "reason": "immutable replay metadata contains no per-record sequence array"},
            {"field": "targetfit.public_record_sha256", "reason": "target preparation/audit serialize counts only"},
            {"field": "targetfit.truncated_sequence_sha256", "reason": "target preparation/audit serialize counts only"},
        ]
    )
    bundle.metadata_notes.extend(
        [
            "P04 public_record_sha256 opaque values admitted as rendered_sha256 only for correction/validation/fresh_evaluation, after ordered/set commitment checks",
            "P04 truncated_sequence_sha256 opaque values admitted as H129 only after the declared BOS+128 and signed-int32 convention checks",
            "fit_replay rendered_sha256 admitted separately; unavailable public/sequence aliases remain explicit gaps",
            "P04 public pool Git-object ledgers were recovered and bound by exact file SHA, preserving individual record IDs/rendered hashes/row indices without source payload",
        ]
    )
    producer_git_proof = _p04_producer_git_proof(root)
    target_plan_proof = _p04_target_plan_proof(root)
    recomputed_exchange_digest = _json_digest({
        "reservations": value["reservations"],
        "overlap_counts": value["overlap_counts"],
        "selection_fresh_panel_reconciliation": value["selection_fresh_panel_reconciliation"],
    })
    exchange_digest_match = recomputed_exchange_digest == value["exchange_digest_sha256"]
    if not exchange_digest_match:
        raise ExclusionAuditError("P04 top-level exchange digest mismatch")
    proof = {
        "status": "PASS_PRODUCER_CONVENTION_VERIFIED_H128_TARGETFIT_PARTIAL",
        "exchange_path": str(path),
        "exchange_sha256": actual_sha,
        "producer_descriptor_sha256": P04_PRODUCER_SHA256,
        "producer_source_bytes_available": producer_git_proof["all_expected_source_bytes_verified"],
        "producer_git_object_proof": producer_git_proof,
        "declared_convention_checks": convention_checks,
        "pool_counts": pool_proof,
        "recovered_ledger_proof": ledger_proofs,
        "h128_identity_export": p04_h128_export_proof,
        "fit_replay_rendered_count": len(fit_values),
        "consumer_proof": _consumer_proof(root),
        "individual_record_ids_available": all(item.get("available") is True for item in ledger_proofs.values()),
        "targetfit_individual_hashes_available": False,
        "targetfit_plan_proof": target_plan_proof,
        "h128_individual_hashes_available": all(item.get("h128_values_present") is True for item in ledger_proofs.values()),
        "top_level_exchange_digest_recomputed": True,
        "top_level_exchange_digest_match": exchange_digest_match,
        "top_level_exchange_digest_equation": "SHA256(canonical JSON of reservations, overlap_counts, selection_fresh_panel_reconciliation)",
    }
    return P04Load(bundle=bundle, proof=proof, gaps=gaps)


def _augment_trr0002_finance_h128(bundle: p10.IdentityBundle, path: Path) -> p10.IdentityBundle:
    """Derive canonical H128 only for exact historical Finance inputs.

    TRR-0002's producer stored an active-token digest and a padded input row.
    For rows whose declared active width is at least 128, the surviving
    metadata itself is an exact public token input, so the canonical first-128
    digest can be derived transiently without opening source text or truth.
    Short rows remain explicitly H128-inapplicable.
    """
    value = _read_json(path, label="TRR2 Finance records")
    rows = value.get("records")
    dataset_id = value.get("dataset")
    if not isinstance(rows, list) or not isinstance(dataset_id, str) or not dataset_id:
        raise ExclusionAuditError("TRR2 Finance metadata cannot support H128 derivation")
    namespace = p10.Namespace("finance", dataset_id, "train", "*")
    eligible = 0
    short = 0
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} is malformed")
        ids = row.get("input_ids")
        valid = row.get("valid_tokens")
        if not isinstance(ids, list) or not isinstance(valid, int) or valid <= 0 or valid > len(ids):
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} lacks exact active input width")
        if valid < 128:
            short += 1
            continue
        if ids[0] != BOS_TOKEN_ID:
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} does not begin with BOS")
        if any(not isinstance(token, int) or isinstance(token, bool) for token in ids[:valid]):
            raise ExclusionAuditError(f"TRR2 Finance row {row_number} contains a non-integer active token")
        digest = p10._raw_int32_digest(ids[:128])
        bundle.add("h128_sequence_sha256", digest, namespace)
        eligible += 1
    bundle.metadata_notes.append(
        f"TRR2 Finance canonical H128 derived transiently from exact active input_ids for {eligible} rows; {short} rows are verified shorter than 128"
    )
    return bundle


def load_bundle(spec: p10.SourceSpec, *, root: Path, pr20_root: Path | None = None) -> p10.IdentityBundle | None:
    """Load a P11 source with strict P04 and bounded historical H128 paths."""
    if spec.label == "trr0005_selection_plan":
        # The retained TRR-0005 producer writes final_sequence_sha256 as the
        # first 128 BOS-inclusive int32 IDs (SEQUENCE_TOKENS == 128).  Keep
        # this mapping local to P11 so the generic P10 inventory remains
        # unchanged and the producer proof is recorded in this audit.
        spec = p10.SourceSpec(
            spec.label, spec.role, spec.path, required=spec.required,
            hash_aliases=tuple(sorted({**spec.aliases, "final_sequence_sha256": "h128_sequence_sha256"}.items())),
            metadata_mode=spec.metadata_mode, expected_sha256=spec.expected_sha256,
        )
    if spec.label == "trr0006_p04_opaque":
        return load_p04(root).bundle
    bundle = p10.load_bundle(spec, root=root, pr20_root=pr20_root)
    if bundle is not None and spec.label == "trr0002_public_finance_records":
        bundle = _augment_trr0002_finance_h128(bundle, bundle.path)
    return bundle


def _load_specs(root: Path, pr20_root: Path | None) -> tuple[list[p10.IdentityBundle], P04Load]:
    specs = list(p10.source_specs())
    p04 = load_p04(root)
    bundles: list[p10.IdentityBundle] = []
    for spec in specs:
        if spec.label == "trr0006_p04_opaque":
            bundles.append(p04.bundle)
            continue
        bundle = load_bundle(spec, root=root, pr20_root=pr20_root)
        if bundle is not None:
            bundles.append(bundle)
    if len(bundles) != len(specs):
        present = {bundle.label for bundle in bundles}
        raise ExclusionAuditError(f"required P11 identity sources missing: {sorted({spec.label for spec in specs} - present)}")
    return bundles, p04


def _selection_rows(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    rule = selection.get("selection_rule")
    if not isinstance(rule, Mapping) or not isinstance(rule.get("records"), Mapping):
        raise ExclusionAuditError("selection ledger has no selection_rule.records mapping")
    return rule["records"]


def validate_panel_selection(panel: Mapping[str, Any], selection: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Strictly bind an aggregate panel's declared row digest/counts to rows."""
    if panel.get("public_material_only") is not True or panel.get("truth_opened") is not False or panel.get("same_sources_across_targets") is not True:
        raise ExclusionAuditError(f"{label} access flags do not satisfy the public pre-truth contract")
    declared_panel_digest = panel.get("record_ids_sha256")
    declared_selection_digest = selection.get("selection_rule", {}).get("record_ids_sha256")
    if not isinstance(declared_panel_digest, Mapping) or not isinstance(declared_selection_digest, Mapping):
        raise ExclusionAuditError(f"{label} record digest map is absent")
    rows = _selection_rows(selection)
    declared_counts = panel.get("records_by_domain")
    if declared_counts is None:
        scalar = panel.get("records_per_domain")
        declared_counts = {domain: scalar for domain in rows} if isinstance(scalar, int) else None
    if not isinstance(declared_counts, Mapping):
        raise ExclusionAuditError(f"{label} domain counts are absent")
    domain_results: dict[str, Any] = {}
    for domain, domain_rows in rows.items():
        if not isinstance(domain, str) or not isinstance(domain_rows, list):
            raise ExclusionAuditError(f"{label} domain rows are malformed")
        expected_count = declared_counts.get(domain)
        if not isinstance(expected_count, int) or len(domain_rows) != expected_count:
            raise ExclusionAuditError(f"{label} {domain} row count mismatch")
        record_ids: list[str] = []
        required_hashes = ("public_record_sha256", "final_sequence_sha256")
        for row in domain_rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row["record_id"]:
                raise ExclusionAuditError(f"{label} {domain} contains a malformed record_id")
            if any(not _is_sha256(row.get(key)) for key in required_hashes):
                raise ExclusionAuditError(f"{label} {domain} contains a malformed identity hash")
            record_ids.append(row["record_id"])
        if len(record_ids) != len(set(record_ids)):
            raise ExclusionAuditError(f"{label} {domain} contains duplicate record IDs")
        digest = _json_digest(record_ids)
        if digest != declared_selection_digest.get(domain) or digest != declared_panel_digest.get(domain):
            raise ExclusionAuditError(f"{label} {domain} record digest mismatch")
        domain_results[domain] = {
            "records": len(record_ids),
            "computed_record_ids_sha256": digest,
            "panel_digest_match": True,
            "selection_digest_match": True,
        }
    extra_panel_domains = set(declared_panel_digest) - set(rows)
    if extra_panel_domains:
        raise ExclusionAuditError(f"{label} declares unbound panel domains")
    return {
        "label": label,
        "status": "PASS",
        "domains": domain_results,
        "public_material_only": True,
        "same_sources_across_targets": True,
        "truth_opened": False,
    }


def _verify_panel_selection_descriptor(
    panel_path: Path,
    selection_path: Path,
    *,
    root: Path,
    pr20_root: Path | None,
    label: str,
) -> dict[str, Any]:
    panel = _read_json(panel_path, label=f"{label} panel")
    selection = _read_json(selection_path, label=f"{label} selection")
    descriptor = panel.get("selection_plan")
    if not isinstance(descriptor, Mapping):
        raise ExclusionAuditError(f"{label} panel selection_plan descriptor is absent")
    pointed = _descriptor_path(descriptor, root=root, pr20_root=pr20_root, label=f"{label} panel selection_plan")
    if pointed != selection_path.resolve():
        # Historical panel descriptors may carry a sibling worktree absolute
        # path; byte identity is the binding, while the expected path is the
        # explicit inventory path.
        if p10.sha256_file(pointed) != p10.sha256_file(selection_path):
            raise ExclusionAuditError(f"{label} panel selection pointer does not bind the expected ledger")
    return validate_panel_selection(panel, selection, label=label)


def _verify_pr20_binding(root: Path, pr20_root: Path) -> dict[str, Any]:
    panel_path = _resolve(AGGREGATE_BINDINGS[-1]["panel"], root=root, pr20_root=pr20_root)
    panel = _read_json(panel_path, label="PR20 panel")
    panel_descriptor = panel.get("selection_plan")
    binding_path = _descriptor_path(panel_descriptor, root=root, pr20_root=pr20_root, label="PR20 panel→binding")
    binding = _read_json(binding_path, label="PR20 source-selection binding")
    ledger_descriptor = binding.get("selection_ledger")
    ledger_path = _descriptor_path(ledger_descriptor, root=root, pr20_root=pr20_root, label="PR20 binding→selection")
    selection = _read_json(ledger_path, label="PR20 selection ledger")
    binding_counts = binding.get("records_by_domain")
    selection_counts = selection.get("records_by_domain")
    if binding_counts != selection_counts:
        raise ExclusionAuditError("PR20 binding and selection domain counts differ")
    binding_digest = binding.get("record_ids_sha256")
    selection_digest = selection.get("selection_rule", {}).get("record_ids_sha256")
    if binding_digest != selection_digest:
        raise ExclusionAuditError("PR20 binding and selection record digests differ")
    result = validate_panel_selection(panel, selection, label="pr20_source_panel")
    result["panel_to_binding"] = {"status": "PASS", "binding_sha256": p10.sha256_file(binding_path)}
    result["binding_to_selection"] = {"status": "PASS", "selection_sha256": p10.sha256_file(ledger_path)}
    return result


def validate_aggregate_bindings(*, root: Path, pr20_root: Path | None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in AGGREGATE_BINDINGS[:-1]:
        panel_path = _resolve(item["panel"], root=root, pr20_root=pr20_root)
        selection_path = _resolve(item["selection"], root=root, pr20_root=pr20_root)
        if not panel_path.is_file() or not selection_path.is_file():
            raise ExclusionAuditError(f"aggregate binding source unavailable: {item['label']}")
        results.append(_verify_panel_selection_descriptor(panel_path, selection_path, root=root, pr20_root=pr20_root, label=item["label"]))
    if pr20_root is None:
        raise ExclusionAuditError("PR20 root is required for aggregate binding audit")
    results.append(_verify_pr20_binding(root, pr20_root))
    return {
        "status": "PASS_ALL_SIX",
        "count": len(results),
        "bindings": results,
        "pr20_final_sources_are_development_overlap": True,
    }


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ExclusionAuditError("public token payload tensor row is not list-like")
    return value


def _row_namespace(row: Mapping[str, Any]) -> p10.Namespace:
    style = row.get("dataset_key") or row.get("dataset") or "*"
    style_text = p10._style(style) or "*"
    dataset_id = row.get("dataset_id") or "*"
    split = row.get("split") or "train"
    revision = row.get("revision") or "*"
    return p10.Namespace(str(style_text), str(dataset_id), str(split), str(revision))


def rehash_public_token_rows(
    rows: Sequence[Mapping[str, Any]],
    token_rows: Sequence[Any],
    mask_rows: Sequence[Any],
    *,
    payload_path: Path,
    payload_sha256: str | None = None,
) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    """Derive canonical H128 from a public token payload without retaining IDs."""
    if len(rows) != len(token_rows) or len(rows) != len(mask_rows):
        raise ExclusionAuditError("public payload and metadata row counts differ")
    bundle = p10.IdentityBundle(
        "agent1_b0_public_token_payload",
        "inherited_fitting",
        payload_path,
        payload_sha256 or p10.sha256_file(payload_path),
        payload_path.stat().st_size,
        status="public_token_ids_hashed_transiently",
    )
    short = 0
    eligible = 0
    seen_ids: set[str] = set()
    h128_count = 0
    for index, (row, raw_ids, raw_mask) in enumerate(zip(rows, token_rows, mask_rows)):
        if not isinstance(row, Mapping) or not isinstance(row.get("record_id"), str) or not row["record_id"]:
            raise ExclusionAuditError(f"public payload metadata row {index} lacks record_id")
        record_id = row["record_id"]
        if record_id in seen_ids:
            raise ExclusionAuditError("public payload metadata contains duplicate record_id")
        seen_ids.add(record_id)
        ids = [int(value) for value in _as_list(raw_ids)]
        mask = [int(value) for value in _as_list(raw_mask)]
        if len(ids) != len(mask) or not ids or any(value not in (0, 1) for value in mask):
            raise ExclusionAuditError(f"public payload row {index} mask geometry is malformed")
        active_count = sum(mask)
        if active_count <= 0 or any(mask[pos] == 0 and any(mask[pos + 1 :]) for pos in range(len(mask))):
            raise ExclusionAuditError(f"public payload row {index} mask is not a prefix")
        if ids[0] != BOS_TOKEN_ID:
            raise ExclusionAuditError(f"public payload row {index} BOS identity changed")
        bundle.add("record_id", record_id, _row_namespace(row))
        rendered = row.get("rendered_sha256")
        if _is_sha256(rendered):
            bundle.add("rendered_sha256", rendered.casefold(), _row_namespace(row))
        if active_count < 128:
            short += 1
            continue
        eligible += 1
        digest = p10._raw_int32_digest(ids[:128])
        bundle.add("h128_sequence_sha256", digest, _row_namespace(row))
        h128_count += 1
    return bundle, {
        "status": "PASS_PUBLIC_TOKEN_H128_REHASH",
        "rows_seen": len(rows),
        "h128_rows": h128_count,
        "short_rows_h128_inapplicable": short,
        "eligible_rows": eligible,
        "token_values_emitted": False,
        "source_text_read": False,
        "activations_read": False,
        "model_loaded": False,
        "payload_path": str(payload_path),
        "payload_sha256": bundle.sha256,
        "payload_bytes": bundle.bytes,
        "tensors_read": ["token_ids", "attention_mask"],
        "tensors_not_read": ["activations", "position_ids", "post_bos_selector_large", "post_bos_selector_small"],
        "sequence_convention": "first 128 active BOS-inclusive IDs, SHA-256 little-endian signed-int32 bytes",
    }


def rehash_public_token_payload(*, payload_path: Path, metadata_path: Path) -> tuple[p10.IdentityBundle, dict[str, Any]]:
    """Read only ``token_ids`` and ``attention_mask`` from a safetensors payload."""
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ExclusionAuditError("safetensors is required for the bounded public-token pass") from exc
    metadata = _read_json(metadata_path, label="public token metadata")
    rows = metadata.get("records")
    if not isinstance(rows, list):
        raise ExclusionAuditError("public token metadata lacks records list")
    payload_path = payload_path.resolve()
    if not payload_path.is_file() or payload_path.is_symlink():
        raise ExclusionAuditError(f"public token payload is unavailable: {payload_path}")
    payload_sha = p10.sha256_file(payload_path)
    with safe_open(str(payload_path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required = {"token_ids", "attention_mask"}
        if not required.issubset(keys):
            raise ExclusionAuditError("public token payload lacks token_ids/attention_mask")
        token_tensor = handle.get_tensor("token_ids")
        mask_tensor = handle.get_tensor("attention_mask")
        if tuple(token_tensor.shape) != tuple(mask_tensor.shape) or len(token_tensor.shape) != 2:
            raise ExclusionAuditError("public token payload tensor geometry differs")
        token_rows = token_tensor.tolist()
        mask_rows = mask_tensor.tolist()
    bundle, receipt = rehash_public_token_rows(rows, token_rows, mask_rows, payload_path=payload_path, payload_sha256=payload_sha)
    receipt.update(
        {
            "tensor_keys_present": sorted(keys),
            "token_tensor_shape": [int(value) for value in token_tensor.shape],
            "token_tensor_dtype": str(token_tensor.dtype),
            "mask_tensor_dtype": str(mask_tensor.dtype),
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": p10.sha256_file(metadata_path),
        }
    )
    return bundle, receipt


def _record_rows(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield record mappings from a metadata JSON tree without reading payloads."""
    if isinstance(value, Mapping):
        if isinstance(value.get("record_id"), str):
            yield value
        for child in value.values():
            yield from _record_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _record_rows(child)



REPLICATION_METADATA_SPECS: tuple[tuple[str, str, str, str], ...] = (
    (
        "agent1_b0_public_metadata",
        "../TRR-0007/experiments/TRR-0007/support/broader_capture_v2/enriched_fit_records.json",
        "808cfd0f95ea3ee66c1d4094f3c10f1e346ba986a6d77138a61b8f8f13c4a738",
        "inherited_fitting",
    ),
    (
        "agent1_b1_public_metadata",
        "../TRR-0012/experiments/TRR-0012/preparation/b1_cpu_r4_date07/records.json",
        "367cfba0ffe78f59454861a76f23830f480a8745cc6b35e6b4a0d05eca53638b",
        "inherited_fitting",
    ),
)


def _load_replication_metadata(root: Path) -> tuple[list[p10.IdentityBundle], list[dict[str, Any]]]:
    bundles: list[p10.IdentityBundle] = []
    descriptors: list[dict[str, Any]] = []
    for label, relative_path, expected_sha, role in REPLICATION_METADATA_SPECS:
        path = (root / relative_path).resolve()
        descriptor: dict[str, Any] = {
            "label": label,
            "role": role,
            "path": relative_path,
            "expected_sha256": expected_sha,
            "available": False,
            "identity_fields_loaded": [],
            "payload_opened": False,
        }
        if not path.is_file() or path.is_symlink():
            descriptors.append(descriptor)
            continue
        actual_sha = p10.sha256_file(path)
        descriptor.update({"available": True, "bytes": path.stat().st_size, "sha256": actual_sha, "sha256_match": actual_sha == expected_sha})
        if actual_sha != expected_sha:
            raise ExclusionAuditError(f"{label} metadata SHA-256 changed")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ExclusionAuditError(f"{label} metadata is invalid JSON: {path}") from exc
        raw_rows = value.get("records") if isinstance(value, Mapping) else value
        if not isinstance(raw_rows, list):
            raise ExclusionAuditError(f"{label} metadata has no records list")
        bundle = p10.IdentityBundle(label, role, path, actual_sha, path.stat().st_size, status="PUBLIC_METADATA_ONLY")
        row_count = 0
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise ExclusionAuditError(f"{label} metadata row is malformed")
            record_id = row.get("record_id") or row.get("source_record_id")
            if isinstance(record_id, str) and record_id:
                bundle.add("record_id", record_id, _row_namespace(row))
                row_count += 1
            rendered = row.get("rendered_sha256")
            if _is_sha256(rendered):
                bundle.add("rendered_sha256", rendered.casefold(), _row_namespace(row))
            index = row.get("source_row_index")
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                namespace = _row_namespace(row)
                if namespace.style != "*":
                    bundle.add("source_index", index, namespace)
        bundle.metadata_notes.append("Agent1 handoff metadata only; no token payload, activations, or model opened")
        descriptor.update({"rows": row_count, "identity_counts": bundle.counts(), "identity_fields_loaded": sorted(bundle.values)})
        bundles.append(bundle)
        descriptors.append(descriptor)
    return bundles, descriptors


def _verify_bound_file(root: Path, binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path_value = binding.get("path")
    expected_sha = binding.get("sha256")
    expected_bytes = binding.get("bytes")
    if not isinstance(path_value, str) or not _is_sha256(expected_sha):
        raise ExclusionAuditError(f"{label} binding is malformed")
    path = (root / path_value).resolve()
    if not path.is_file() or path.is_symlink():
        raise ExclusionAuditError(f"{label} file is unavailable: {path_value}")
    actual_bytes = path.stat().st_size
    actual_sha = p10.sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha != expected_sha:
        raise ExclusionAuditError(f"{label} file binding changed")
    return {"path": path_value, "sha256": expected_sha, "bytes": actual_bytes}


def _load_replication_input_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the sidecar's exact B0/B1 metadata without reading tensors."""
    sidecar = (root / P11_REPLICATION_SIDECAR_PATH).resolve()
    if not sidecar.is_file() or sidecar.is_symlink() or p10.sha256_file(sidecar) != P11_REPLICATION_SIDECAR_SHA256:
        raise ExclusionAuditError("replication input sidecar is unavailable or changed")
    value = _read_json(sidecar, label="replication input sidecar")
    raw = value.get("replication_inputs")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("banks"), Mapping):
        raise ExclusionAuditError("replication input sidecar lacks bank bindings")
    normalized: dict[str, Any] = {
        "status": "BOUND_EXACT_BANK_AND_SELECTION_IDENTITIES_AUDIT_COMPLETE",
        "banks": {},
    }
    for bank_name in ("B0", "B1"):
        raw_bank = raw["banks"].get(bank_name)
        if not isinstance(raw_bank, Mapping) or raw_bank.get("bank") != bank_name:
            raise ExclusionAuditError(f"replication input sidecar lacks {bank_name}")
        bank: dict[str, Any] = {"bank": bank_name}
        for key in ("bank_manifest", "ordered_identity", "preparation_manifest"):
            if key not in raw_bank:
                if bank_name == "B0" and key == "preparation_manifest":
                    continue
                raise ExclusionAuditError(f"replication input sidecar lacks {bank_name}/{key}")
            bank[key] = _verify_bound_file(root, raw_bank[key], label=f"{bank_name}/{key}")
        normalized["banks"][bank_name] = bank
    development = raw.get("development_selection")
    if not isinstance(development, Mapping):
        raise ExclusionAuditError("replication input sidecar lacks development selection")
    normalized["development_selection"] = _verify_bound_file(root, development, label="development selection")
    historical = raw.get("historical_trr0009_selection")
    if not isinstance(historical, Mapping):
        raise ExclusionAuditError("replication input sidecar lacks TRR-0009 selection")
    historical_verified = _verify_bound_file(root, historical, label="historical TRR-0009 selection")
    proof = {
        "sidecar": {"path": P11_REPLICATION_SIDECAR_PATH, "sha256": P11_REPLICATION_SIDECAR_SHA256, "status": value.get("status")},
        "replication_inputs": normalized,
        "historical_trr0009_selection": historical_verified,
        "tensor_payload_opened": False,
    }
    return normalized, proof


def _row_identity_fields(row: Mapping[str, Any], *, source_label: str) -> tuple[p10.Namespace, dict[str, set[str | int]]]:
    namespace = p10._namespace_from_mapping(p10.Namespace(), row, source_label)
    aliases = {
        **p10.DEFAULT_ALIASES,
        "final_sequence_sha256": "h128_sequence_sha256",
        "h128_sequence_sha256": "h128_sequence_sha256",
        "sequence_h128_sha256": "h128_sequence_sha256",
        "truncated_sequence_sha256": "h129_sequence_sha256",
        "h129_sequence_sha256": "h129_sequence_sha256",
        "sequence_h129_sha256": "h129_sequence_sha256",
        # TRR-0001's raw signed-int32 H40 is distinct from TRR-0002's
        # tensor-header/signed-int64 H40 field below.
        "h40_sequence_sha256": "h40_sequence_sha256",
        "sequence_h40_sha256": "h40_sequence_sha256",
        "trr0001_h40_sequence_sha256": "h40_sequence_sha256",
        "trr0002_active_token_ids_sha256": "trr0002_active_token_ids_sha256",
        "trr0002_h40_token_ids_sha256": "trr0002_h40_token_ids_sha256",
        # TRR-0002 Finance producer field; this is not a global alias.
        "token_ids_sha256": "trr0002_active_token_ids_sha256",
    }
    # TRR-0004 used one field name for two producer-bound prefix widths:
    # the Pile selection stores exactly 40 BOS-inclusive IDs, while Finance
    # stores exactly 128. Do not apply the historical generic H129 alias to
    # this source; infer the canonical namespace only from the producer's
    # explicit valid_tokens geometry. Unknown geometry remains unclassified.
    if source_label == "trr0004_selection_plan" and "truncated_sequence_sha256" in aliases:
        width = row.get("valid_tokens")
        if width == 40:
            aliases["truncated_sequence_sha256"] = "trr0002_h40_token_ids_sha256"
        elif width == 128:
            aliases["truncated_sequence_sha256"] = "h128_sequence_sha256"
        else:
            aliases.pop("truncated_sequence_sha256", None)
    fields: dict[str, set[str | int]] = {}
    for raw_key, raw_value in row.items():
        canonical = aliases.get(str(raw_key).casefold().replace("-", "_"))
        if canonical == "record_id" and isinstance(raw_value, str) and raw_value:
            fields.setdefault(canonical, set()).add(raw_value)
        elif canonical == "source_index" and isinstance(raw_value, int) and not isinstance(raw_value, bool) and raw_value >= 0 and namespace.style != "*":
            fields.setdefault(canonical, set()).add(raw_value)
        elif canonical in IDENTITY_UNION_FIELDS and canonical != "source_index" and _is_sha256(raw_value):
            fields.setdefault(canonical, set()).add(raw_value.casefold())
    return namespace, fields


def _row_key(namespace: p10.Namespace, fields: Mapping[str, set[str | int]]) -> tuple[Any, ...] | None:
    for field in ("record_id", "rendered_sha256", "h128_sequence_sha256", "h129_sequence_sha256", "h40_sequence_sha256", "trr0002_active_token_ids_sha256", "trr0002_h40_token_ids_sha256"):
        values = fields.get(field)
        if values:
            return (field, sorted(str(value) for value in values)[0])
    values = fields.get("source_index")
    if values:
        return ("source_index", namespace.as_string(), int(sorted(values)[0]))
    return None


def _row_matches_other_bundles(
    namespace: p10.Namespace,
    fields: Mapping[str, set[str | int]],
    bundles: Sequence[p10.IdentityBundle],
) -> set[str]:
    matches: set[str] = set()
    for field, values in fields.items():
        for value in values:
            if field == "source_index":
                if any(value in source_values and namespace.compatible(source_namespace) for bundle in bundles for source_namespace, source_values in bundle.values.get(field, {}).items()):
                    matches.add(field)
            elif any(value in source_values for bundle in bundles for source_values in bundle.values.get(field, {}).values()):
                matches.add(field)
    return matches


_CANONICAL_ROW_SOURCE_LABELS = frozenset({
    "trr0002_public_finance_records",
    "trr0002_public_pile_records",
    "trr0003_fit_records_h40_public_identity",
    "trr0001_public_h40_identity",
    "trr0002_public_pile_h40_recovery_identity",
    "trr0004_original_fit_public_token_identity",
    "trr0004_original_validation_public_token_identity",
    "trr0004_selection_plan",
    "trr0005_selection_plan",
    "trr0005_enriched_fit_public_token_identity",
    "trr0006_selection",
    "trr0006_p04_targetfit_public_identity",
    "trr0007_selection",
    "trr0008_selection",
    "trr0009_original_selection",
    "trr0009_selection_v2",
    "trr_p09_nested_b1",
    "pr20_source_selection",
})
_CANONICAL_SUMMARY_LABELS = frozenset({
    "trr0006_p04_opaque",
    "trr0007_p06_opaque",
    "trr0008_p06_opaque",
    "trr0008_selection_opaque",
    "trr0009_opaque_reservation",
    "trr0009_p08_opaque",
    "trr0007_prefix_exclusions",
})


def _row_geometry(row: Mapping[str, Any]) -> int | None:
    """Return an explicit active sequence length, never padded length."""
    for key in ("valid_tokens", "full_token_count", "active_token_count"):
        value = row.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    value = row.get("post_bos_token_count")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value + 1
    return None


def _iter_identity_rows(bundle: p10.IdentityBundle) -> Iterable[tuple[Mapping[str, Any], p10.Namespace, dict[str, set[str | int]]]]:
    """Yield scalar row identities; aggregate opaque arrays are not rows."""
    if bundle.label in _CANONICAL_SUMMARY_LABELS:
        return
    try:
        raw = _read_json(bundle.path, label=bundle.label)
    except ExclusionAuditError:
        return
    for row in _record_rows(raw):
        if not isinstance(row, Mapping):
            continue
        namespace, fields = _row_identity_fields(row, source_label=bundle.label)
        if fields:
            yield row, namespace, fields


def _row_canonical_proof(
    bundle: p10.IdentityBundle,
    row: Mapping[str, Any],
    fields: Mapping[str, set[str | int]],
) -> frozenset[str]:
    """Return proof carried by this canonical row itself.

    This never infers a sequence hash from a record ID, duplicate alias, or
    H129.  TRR2's loader has verified its active digest, so eligible Finance
    rows prove H128 while Pile rows retain the producer-bound H40 convention.
    """
    proof: set[str] = set()
    if fields.get("h128_sequence_sha256"):
        proof.add("h128")
    if fields.get("h40_sequence_sha256") or fields.get("trr0002_h40_token_ids_sha256"):
        proof.add("h40")
    geometry = _row_geometry(row)
    if bundle.label == "trr0002_public_finance_records":
        if fields.get("trr0002_active_token_ids_sha256") and geometry is not None and geometry >= 128:
            proof.add("h128")
    if bundle.label == "trr0002_public_pile_records":
        tokens = row.get("token_ids")
        if isinstance(tokens, list) and len(tokens) == 40 and all(isinstance(item, int) and not isinstance(item, bool) for item in tokens):
            proof.add("h40")
    if geometry is not None and geometry < 128:
        proof.add("short")
    return frozenset(proof)


def _build_canonical_row_anchor_index(
    bundles: Sequence[p10.IdentityBundle],
) -> dict[str, Any]:
    """Build an index whose entries each carry their own canonical proof."""
    global_index: dict[str, dict[str, list[tuple[str, tuple[str, ...]]]]] = {}
    source_index: dict[str, list[tuple[p10.Namespace, tuple[str, tuple[str, ...]]]]] = {}
    proof_counts: dict[str, Counter[str]] = {}
    row_counts: Counter[str] = Counter()
    for bundle in bundles:
        if bundle.label not in _CANONICAL_ROW_SOURCE_LABELS:
            continue
        for row, namespace, fields in _iter_identity_rows(bundle):
            proof = _row_canonical_proof(bundle, row, fields)
            if not proof:
                continue
            row_counts[bundle.label] += 1
            counts = proof_counts.setdefault(bundle.label, Counter())
            for item in proof:
                counts[item] += 1
            reference = (bundle.label, tuple(sorted(proof)))
            for field, values in fields.items():
                if field == "h129_sequence_sha256":
                    continue
                for value in values:
                    if field == "source_index":
                        source_index.setdefault(str(value), []).append((namespace, reference))
                    else:
                        global_index.setdefault(field, {}).setdefault(str(value), []).append(reference)
    return {
        "global": global_index,
        "source_index": source_index,
        "proof_counts": {label: dict(sorted(counts.items())) for label, counts in sorted(proof_counts.items())},
        "row_counts": dict(sorted(row_counts.items())),
        "source_labels": sorted(_CANONICAL_ROW_SOURCE_LABELS),
    }


def _canonical_anchor_matches(
    namespace: p10.Namespace,
    fields: Mapping[str, set[str | int]],
    index: Mapping[str, Any],
) -> set[tuple[str, tuple[str, ...]]]:
    matches: set[tuple[str, tuple[str, ...]]] = set()
    global_index = index.get("global", {})
    source_index = index.get("source_index", {})
    # Record IDs and namespace-scoped indices are useful diagnostics, but do
    # not prove that an original/control row has the same rendered sequence as
    # an enriched/replacement row.  Require a shared producer-bound content or
    # sequence commitment for a canonical join.
    strong_fields = {
        "rendered_sha256",
        "tokenized_record_sha256",
        "h128_sequence_sha256",
        "h40_sequence_sha256",
        "trr0002_active_token_ids_sha256",
        "trr0002_h40_token_ids_sha256",
    }
    for field, values in fields.items():
        if field not in strong_fields:
            continue
        for value in values:
            matches.update(global_index.get(field, {}).get(str(value), ()))
    return matches


def _verified_lineage_anchor_matches(
    bundle_label: str,
    fields: Mapping[str, set[str | int]],
    verified_lineage: Mapping[str, Mapping[str, set[tuple[str, tuple[str, ...]]]]],
) -> set[tuple[str, tuple[str, ...]]]:
    """Return joins backed by an explicitly verified producer lineage proof."""
    rows = verified_lineage.get(bundle_label, {})
    matches: set[tuple[str, tuple[str, ...]]] = set()
    for record_id in fields.get("record_id", set()):
        if isinstance(record_id, str):
            matches.update(rows.get(record_id, set()))
    return matches


def _reconcile_legacy_aliases(
    bundles: Sequence[p10.IdentityBundle],
    *,
    root: Path,
    verified_lineage: Mapping[str, Mapping[str, set[tuple[str, tuple[str, ...]]]]] | None = None,
) -> dict[str, Any]:
    """Report aliases and fail-closed row-level canonical coverage.

    Historical alias counters remain for provenance, but closure uses only
    row-level canonical proofs and all eligible/unresolved rows.  A duplicate
    in two noncanonical ledgers cannot close a row.
    """
    replication, replication_descriptors = _load_replication_metadata(root)
    all_bundles = [*bundles, *replication]
    verified_lineage = verified_lineage or {}
    anchor_index = _build_canonical_row_anchor_index(all_bundles)
    per_source: list[dict[str, Any]] = []
    unique_uncovered_keys: set[tuple[Any, ...]] = set()
    global_uncovered_keys: set[tuple[Any, ...]] = set()
    total_rows = total_matched = total_canonical = total_without_canonical = 0
    total_short = total_h40 = total_eligible_without = total_unresolved = 0
    for bundle in all_bundles:
        row_count = matched_count = rows_with_canonical_anchor = rows_without_canonical_anchor = 0
        field_matches: Counter[str] = Counter()
        anchor_source_matches: Counter[str] = Counter()
        uncovered_keys: set[tuple[Any, ...]] = set()
        verified_short_rows = verified_h40_rows = eligible_without_canonical_anchor = 0
        unresolved_without_canonical_anchor = short_without_canonical_anchor = 0
        unknown_without_canonical_anchor = uncovered_h128_eligible = uncovered_short = uncovered_geometry_unknown = 0
        uncovered_direct_h128 = uncovered_with_own_bound_identity = uncovered_without_own_bound_identity = 0
        for row, namespace, fields in _iter_identity_rows(bundle):
            row_count += 1
            geometry = _row_geometry(row)
            if geometry is not None and geometry < 128:
                verified_short_rows += 1
            own_proof = _row_canonical_proof(bundle, row, fields)
            if "h40" in own_proof:
                verified_h40_rows += 1
            if "h128" in own_proof:
                uncovered_direct_h128 += 1
            anchor_matches = _canonical_anchor_matches(namespace, fields, anchor_index)
            lineage_matches = _verified_lineage_anchor_matches(bundle.label, fields, verified_lineage)
            if lineage_matches:
                anchor_matches.update(lineage_matches)
                field_matches["verified_producer_lineage"] += 1
            if anchor_matches:
                rows_with_canonical_anchor += 1
                anchor_source_matches.update(label for label, _proof in anchor_matches)
            else:
                rows_without_canonical_anchor += 1
                if geometry is not None and geometry >= 128:
                    eligible_without_canonical_anchor += 1
                    uncovered_h128_eligible += 1
                elif geometry is not None and geometry < 128:
                    short_without_canonical_anchor += 1
                    uncovered_short += 1
                else:
                    unknown_without_canonical_anchor += 1
                    uncovered_geometry_unknown += 1
                if "h40" not in own_proof and not (geometry is not None and geometry < 128):
                    unresolved_without_canonical_anchor += 1
                key = _row_key(namespace, fields)
                if key is not None:
                    unique_uncovered_keys.add((bundle.label, *key))
                    global_uncovered_keys.add(key)
                    uncovered_keys.add(key)
                    if fields:
                        uncovered_with_own_bound_identity += 1
                    else:
                        uncovered_without_own_bound_identity += 1
            others = [candidate for candidate in all_bundles if candidate is not bundle]
            matches = _row_matches_other_bundles(namespace, fields, others)
            if matches:
                matched_count += 1
                field_matches.update(matches)
        total_rows += row_count
        total_matched += matched_count
        total_canonical += rows_with_canonical_anchor
        total_without_canonical += rows_without_canonical_anchor
        total_short += verified_short_rows
        total_h40 += verified_h40_rows
        total_eligible_without += eligible_without_canonical_anchor
        total_unresolved += unresolved_without_canonical_anchor
        per_source.append({
            "label": bundle.label,
            "role": bundle.role,
            "rows_with_identity": row_count,
            "rows_with_alias_in_other_bound_source": matched_count,
            "rows_without_alias_in_other_bound_source": len(uncovered_keys),
            "unique_uncovered_row_keys": len(uncovered_keys),
            "unique_uncovered_h128_eligible_rows": uncovered_h128_eligible,
            "unique_uncovered_short_rows_h128_inapplicable": uncovered_short,
            "unique_uncovered_rows_geometry_unknown": uncovered_geometry_unknown,
            "unique_uncovered_rows_with_direct_h128": uncovered_direct_h128,
            "unique_uncovered_rows_with_own_bound_identity": uncovered_with_own_bound_identity,
            "unique_uncovered_rows_without_own_bound_identity": uncovered_without_own_bound_identity,
            "rows_with_verified_canonical_anchor": rows_with_canonical_anchor,
            "rows_without_verified_canonical_anchor": rows_without_canonical_anchor,
            "verified_short_rows_h128_inapplicable": verified_short_rows,
            "verified_h40_rows": verified_h40_rows,
            "eligible_rows_without_verified_canonical_anchor": eligible_without_canonical_anchor,
            "unresolved_rows_without_verified_canonical_anchor": unresolved_without_canonical_anchor,
            "short_rows_without_verified_canonical_anchor": short_without_canonical_anchor,
            "unknown_geometry_rows_without_verified_canonical_anchor": unknown_without_canonical_anchor,
            "canonical_anchor_labels": sorted(anchor_source_matches),
            "alias_match_fields": dict(sorted(field_matches.items())),
            "status": (
                "PASS_ALL_ELIGIBLE_ROWS_ANCHORED"
                if row_count and eligible_without_canonical_anchor == 0 and unresolved_without_canonical_anchor == 0
                else ("UNRESOLVED_CANONICAL_ROWS" if unresolved_without_canonical_anchor else ("SHORT_ROWS_UNANCHORED" if row_count else "NO_INDIVIDUAL_ROWS"))
            ),
        })
    return {
        "status": "PASS_ROW_CANONICAL_ANCHOR_RECONCILIATION" if total_unresolved == 0 and total_eligible_without == 0 else "PARTIAL_ROW_CANONICAL_ANCHOR_RECONCILIATION",
        "source_count_including_replication_metadata": len(all_bundles),
        "primary_source_count": len(bundles),
        "rows_with_identity_across_sources": total_rows,
        "rows_with_alias_in_other_bound_source": total_matched,
        "rows_with_verified_canonical_anchor": total_canonical,
        "rows_without_verified_canonical_anchor": total_without_canonical,
        "verified_short_rows_h128_inapplicable": total_short,
        "verified_h40_rows": total_h40,
        "eligible_rows_without_verified_canonical_anchor": total_eligible_without,
        "unresolved_rows_without_verified_canonical_anchor": total_unresolved,
        "unique_uncovered_source_row_keys": len(unique_uncovered_keys),
        "unique_uncovered_identity_keys_across_sources": len(global_uncovered_keys),
        "unique_uncovered_identity_keys_without_verified_canonical_anchor": len(unique_uncovered_keys),
        "replication_metadata": replication_descriptors,
        "canonical_row_anchor_index": {
            "source_labels": anchor_index["source_labels"],
            "row_counts_by_source": anchor_index["row_counts"],
            "proof_counts_by_source": anchor_index["proof_counts"],
        },
        "verified_producer_lineage": {
            label: {"rows": len(rows), "canonical_labels": sorted({ref[0] for refs in rows.values() for ref in refs})}
            for label, rows in sorted(verified_lineage.items())
        },
        "per_source": per_source,
        "interpretation": "Only a row-level match to a canonical record carrying its own H128, verified short geometry, or producer-bound H40 counts as an anchor. H129, record-ID membership in a noncanonical bundle, duplicate aliases, and aggregate opaque arrays do not close an eligible row. The six original-ledger joins are admitted only through the separately recorded exact producer-lineage proof.",
    }

def _sequence_gap_report(
    bundles: Sequence[p10.IdentityBundle], *, root: Path, p04: P04Load
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for bundle in bundles:
        direct = bundle.counts().get("h128_sequence_sha256", 0)
        raw: Any = None
        try:
            raw = _read_json(bundle.path, label=bundle.label)
        except ExclusionAuditError:
            raw = None
        rows = list(_record_rows(raw)) if raw is not None else []
        full_lengths: list[int] = []
        for row in rows:
            if isinstance(row.get("full_token_count"), int):
                full_lengths.append(int(row["full_token_count"]))
            elif isinstance(row.get("active_token_count"), int):
                full_lengths.append(int(row["active_token_count"]))
            elif isinstance(row.get("post_bos_token_count"), int):
                full_lengths.append(int(row["post_bos_token_count"]) + 1)
        # TRR-0002 Pile was opened under the historical tensor-header H40
        # convention. It is a known 40-token observation, so requiring a
        # later H128 identity would misstate a producer limitation as a
        # recoverable missing hash. Keep H40 in the union and make this
        # convention override explicit in the receipt.
        if bundle.label == "trr0002_public_pile_records":
            short = None
            eligible = 0
            missing = 0
            status = "H40_ONLY_H128_INAPPLICABLE_TO_OPENED_40_TOKEN_OBSERVATION"
        # Finance retains the TRR2 producer's active-token digest. Where its
        # exact active input is 128 tokens, the same producer bytes also prove
        # canonical H128; shorter rows remain H128-inapplicable.
        elif bundle.label == "trr0002_public_finance_records":
            short = sum(isinstance(row.get("valid_tokens"), int) and row["valid_tokens"] < 128 for row in rows)
            eligible = sum(isinstance(row.get("valid_tokens"), int) and row["valid_tokens"] >= 128 for row in rows)
            missing = max(0, eligible - direct)
            status = "PARTIAL_H128_DERIVED_FROM_TRR2_ACTIVE_INPUT_PLUS_VERIFIED_SHORT_ROWS" if missing == 0 else "MISSING_TRR2_FINANCE_H128_FOR_ELIGIBLE_ROWS"
        elif bundle.label == "trr_p09_nested_b1":
            short = sum(row.get("final_sequence_eligibility") == "ineligible_shorter_than_128_active_tokens_no_padding_hash" for row in rows)
            eligible = sum(row.get("final_sequence_eligibility") == "eligible_first_128_active_int32_ids_including_bos" for row in rows)
            missing = max(0, eligible - direct)
            status = "PARTIAL_DIRECT_H128_PLUS_VERIFIED_SHORT_ROWS" if missing == 0 else "PARTIAL_DIRECT_H128_WITH_UNMATCHED_ELIGIBLE_ROWS"
        elif full_lengths:
            eligible = sum(length >= 128 for length in full_lengths)
            short = sum(length < 128 for length in full_lengths)
            missing = max(0, eligible - direct)
            status = "PARTIAL_H128_DIRECT_PLUS_SHORT_CLASSIFICATION" if missing == 0 else "MISSING_RECOVERABLE_H128_FOR_ELIGIBLE_ROWS"
        elif direct:
            short = None
            eligible = None
            missing = None
            status = "DIRECT_H128_WITHOUT_COMPLETE_ROW_LENGTH_CLASSIFICATION"
        elif bundle.label == "trr0006_p04_opaque":
            short = None
            eligible = None
            missing = None
            status = "H129_ONLY_H128_NOT_DERIVABLE_FROM_OPAQUE_VALUES"
        elif rows:
            short = None
            eligible = None
            missing = len(rows)
            status = "MISSING_RECOVERABLE_H128_EXACT_RENDERER_OR_TOKEN_INPUT"
        else:
            short = None
            eligible = None
            missing = None
            status = "NO_INDIVIDUAL_ROWS_OR_H128"
        report.append(
            {
                "label": bundle.label,
                "role": bundle.role,
                "record_rows_seen": len(rows),
                "direct_h128_count": direct,
                "verified_short_rows_h128_inapplicable": short,
                "eligible_rows_by_declared_length": eligible,
                "eligible_rows_without_h128": missing,
                "status": status,
            }
        )
    # P04's explicit unavailable targetfit fields are kept alongside the
    # per-source sequence diagnosis to prevent an aggregate count from being
    # misread as complete individual coverage.
    report.append(
        {
            "label": "trr0006_p04_targetfit",
            "role": "opaque_reservation",
            "status": "INDIVIDUAL_IDENTITIES_UNAVAILABLE",
            "public_record_sha256": "unavailable_counts_only",
            "h129_sequence_sha256": "unavailable_counts_only",
        }
    )
    return report


def _synthetic_probe() -> dict[str, Any]:
    values = list(range(192))
    fingerprints = p10.candidate_sequence_fingerprints(values)
    ns = p10.Namespace("pile", "fixture", "train", "fixture")
    bundle = p10.IdentityBundle("fixture", "regression", Path("."), "", 0)
    bundle.add("record_id", "reused-validation-record", ns)
    bundle.add("rendered_sha256", "a" * 64, ns)
    bundle.add("h128_sequence_sha256", fingerprints["h128_sequence_sha256"], ns)
    bundle.add("h129_sequence_sha256", fingerprints["h129_sequence_sha256"], ns)
    reasons = p10.check_candidate({**fingerprints, "record_id": "reused-validation-record", "style": "pile", "dataset_id": "fixture", "split": "train", "revision": "fixture"}, bundle)
    return {
        "long_candidate_prefixes_distinct": fingerprints["h128_sequence_sha256"] != fingerprints["h129_sequence_sha256"],
        "helper_fields_consumed_by_checker": {reason["field"] for reason in reasons} == {"record_id", "h128_sequence_sha256", "h129_sequence_sha256"},
        "h128_label_does_not_match_h129": p10.check_candidate({"h128_sequence_sha256": fingerprints["h129_sequence_sha256"], "style": "pile", "dataset_id": "fixture"}, bundle) == [],
    }



def _unknown_identity_row_fields(
    bundles: Sequence[p10.IdentityBundle],
) -> list[dict[str, Any]]:
    """Report unknown identity-like keys that occur on an individual row.

    Aggregate commitments such as ``record_ids_sha256`` remain visible in the
    source inventory but do not block completion when no row-level unknown key
    exists.  This check is metadata-only and never retains the associated
    values.
    """
    special_aliases = {
        "final_sequence_sha256": "h128_sequence_sha256",
        **{field: field for field in IDENTITY_UNION_FIELDS},
        "public_record_fingerprint": "rendered_sha256",
        "truncated_sequence_fingerprint": "h129_sequence_sha256",
    }
    result: Counter[tuple[str, str]] = Counter()

    def visit(value: Any, *, label: str, aliases: Mapping[str, str]) -> None:
        if isinstance(value, Mapping):
            has_record = any(
                isinstance(value.get(key), str) and value.get(key)
                for key in ("record_id", "source_record_id", "public_record_id")
            )
            if has_record:
                for raw_key in value:
                    lowered = str(raw_key).casefold().replace("-", "_")
                    canonical = aliases.get(lowered, p10.DEFAULT_ALIASES.get(lowered))
                    if canonical is None and p10._looks_like_unknown_identity_key(lowered):
                        result[(label, lowered)] += 1
            for child in value.values():
                visit(child, label=label, aliases=aliases)
        elif isinstance(value, list):
            for child in value:
                visit(child, label=label, aliases=aliases)

    spec_aliases = {spec.label: spec.aliases for spec in p10.source_specs()}
    for bundle in bundles:
        try:
            raw = _read_json(bundle.path, label=bundle.label)
        except ExclusionAuditError:
            continue
        aliases = dict(spec_aliases.get(bundle.label, {}))
        if bundle.label in {"trr0005_selection_plan", "trr0006_p04_opaque"} or bundle.label.startswith("trr0006_p04_"):
            aliases.update(special_aliases)
        elif (
            bundle.label.startswith("trr0005_enriched_fit_public_token_identity")
            or bundle.label in {
                "trr0003_fit_records_h40_public_identity",
                "trr0001_public_h40_identity",
                "trr0002_public_pile_h40_recovery_identity",
                "trr0004_original_fit_public_token_identity",
                "trr0004_original_validation_public_token_identity",
            }
        ):
            aliases.update(special_aliases)
        visit(raw, label=bundle.label, aliases=aliases)
    return [
        {"label": label, "key": key, "row_occurrences": count}
        for (label, key), count in sorted(result.items())
    ]


def _descriptor_pointer(root: Path, pointer: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    path_value = pointer.get("path")
    expected_sha = pointer.get("sha256")
    if not isinstance(path_value, str) or not _is_sha256(expected_sha):
        return {"label": label, "status": "MALFORMED_POINTER"}
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        return {"label": label, "status": "UNAVAILABLE_POINTER", "path": path_value, "expected_sha256": expected_sha}
    actual_sha = p10.sha256_file(path)
    actual_bytes = path.stat().st_size
    expected_bytes = pointer.get("bytes")
    if actual_sha != expected_sha or (expected_bytes is not None and actual_bytes != expected_bytes):
        return {"label": label, "status": "POINTER_HASH_MISMATCH", "path": path_value, "expected_sha256": expected_sha, "actual_sha256": actual_sha, "bytes": actual_bytes}
    return {"label": label, "status": "PASS", "path": path_value, "sha256": actual_sha, "bytes": actual_bytes}


def _pointer_target(root: Path, checked: Mapping[str, Any]) -> Path | None:
    raw = checked.get("path")
    if not isinstance(raw, str):
        return None
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()



def _load_selection_exclusion_descriptor_proof(root: Path) -> list[dict[str, Any]]:
    """Verify TRR7/8/9 aggregate exclusion receipts against used ledgers.

    These files are producer receipts for exclusions applied while constructing
    later banks.  They contain aggregate counts and hash-bound sidecar
    pointers, not an additional row list.  The proof therefore binds each
    receipt to its already-loaded selection ledger and records explicitly that
    no new used-record identities are introduced by the descriptor itself.
    """
    specs = (
        {
            "label": "trr0007_selection_exclusions",
            "path": "experiments/TRR-0007/selection/source_exclusions.json",
            "sha256": "7547ac0b85052955d355b58ab83bdf5ba24f9d621f514da480a336b01858760e",
            "bytes": 22555,
            "schema": "token-reconstruction.trr0007-source-exclusions.v1",
            "task_id": "TRR-0007",
            "selection_label": "trr0006_selection",
            "selection_suffix": "experiments/TRR-0006/source_selection.json",
            "selection_path": "experiments/TRR-0006/source_selection.json",
            "selection_sha256": "75909aaf0f9e40176c197d86c09651097010a11519855f1db3dc50fe5e754f43",
            "selection_bytes": 2135542,
            "selection_counts": {"finance": 1536, "pile": 1536},
        },
        {
            "label": "trr0008_selection_exclusions",
            "path": "experiments/TRR-0008/selection/source_exclusions.json",
            "sha256": "acdbabb1a923ed9acccc17076f615201baca0a2d0fb6a40177bf41bb96bd09c4",
            "bytes": 18344,
            "schema": "token-reconstruction.trr0008-source-exclusions.v1",
            "task_id": "TRR-0008",
            "selection_label": "trr0007_selection",
            "selection_suffix": "experiments/TRR-0007/selection/source_selection.json",
            "selection_path": "experiments/TRR-0007/selection/source_selection.json",
            "selection_sha256": "0adf45078ee017ab1877e5d8b905261583d9916cf59c9237591166d0b39c431c",
            "selection_bytes": 198090,
            "selection_counts": {"finance": 128, "pile": 128},
        },
        {
            "label": "trr0009_selection_v2_exclusions",
            "path": "experiments/TRR-0009/selection_v2/source_exclusions.json",
            "sha256": "bba988428e6e1d13c2916b5e9c001acd86aa37d838bda553744eecd0f88d806f",
            "bytes": 20664,
            "schema": "token-reconstruction.trr0009-source-exclusions.v1",
            "task_id": "TRR-0009",
            "selection_label": "trr0008_selection",
            "selection_suffix": "experiments/TRR-0008/selection/source_selection.json",
            "selection_path": "experiments/TRR-0008/selection/source_selection.json",
            "selection_sha256": "ea9a7bf2edcc22eee1a8e791a331d423ccd940e0fcab3079b6b43cc456ee8e57",
            "selection_bytes": 997800,
            "selection_counts": {"finance": 1024, "pile": 384},
        },
    )
    results: list[dict[str, Any]] = []
    for spec in specs:
        descriptor_path, descriptor = _verified_file_value(
            root,
            spec["path"],
            spec["sha256"],
            spec["bytes"],
            label=f"{spec['label']} descriptor",
        )
        if (
            descriptor.get("schema") != spec["schema"]
            or descriptor.get("task_id") != spec["task_id"]
            or descriptor.get("status") != "PUBLIC_IDENTITY_EXCLUSIONS_COMPLETE_NO_TRUTH"
            or descriptor.get("private_or_truth_payload_read") is not False
            or descriptor.get("source_text_or_token_ids_written") is not False
            or descriptor.get("truth_opened") is not False
        ):
            raise ExclusionAuditError(f"{spec['label']} descriptor status or access boundary changed")
        sources = descriptor.get("sources")
        if not isinstance(sources, list) or not sources or any(
            not isinstance(item, Mapping)
            or item.get("available") is not True
            or not isinstance(item.get("path"), str)
            or not _is_sha256(item.get("sha256"))
            or not isinstance(item.get("new_identity_count"), int)
            or item.get("new_identity_count") < 0
            for item in sources
        ):
            raise ExclusionAuditError(f"{spec['label']} source-pointer inventory changed")
        selection_pointer = next(
            (
                item
                for item in sources
                if str(item.get("path", "")).replace("\\", "/").endswith(spec["selection_suffix"])
                and item.get("sha256") == spec["selection_sha256"]
            ),
            None,
        )
        if selection_pointer is None:
            raise ExclusionAuditError(f"{spec['label']} does not bind its required selection ledger")
        selection_path, selection = _verified_file_value(
            root,
            spec["selection_path"],
            spec["selection_sha256"],
            spec["selection_bytes"],
            label=f"{spec['label']} selection ledger",
        )
        record_groups = selection.get("selection_rule", {}).get("records") if isinstance(selection.get("selection_rule"), Mapping) else None
        if not isinstance(record_groups, Mapping):
            raise ExclusionAuditError(f"{spec['label']} selection ledger lacks row groups")
        counts = {domain: len(rows) for domain, rows in record_groups.items() if isinstance(rows, list)}
        if counts != spec["selection_counts"]:
            raise ExclusionAuditError(f"{spec['label']} selection ledger counts changed")
        if any(
            key in descriptor
            for key in ("records", "rows", "record_ids", "final_sequence_sha256", "public_record_sha256")
        ):
            raise ExclusionAuditError(f"{spec['label']} unexpectedly contains an individual used-record list")

        sidecar_checks: list[dict[str, Any]] = []
        if isinstance(descriptor.get("final_bank_ledgers"), Mapping):
            ledger = descriptor["final_bank_ledgers"]
            if ledger.get("source_and_sequence_ledgers_verified") is not True or not isinstance(ledger.get("selected_parent_rows"), Mapping) or ledger["selected_parent_rows"].get("rows") != 120:
                raise ExclusionAuditError(f"{spec['label']} final-bank producer proof changed")
            for key, pointer in ledger.get("files", {}).items():
                checked = _descriptor_pointer(root, pointer, label=f"{spec['label']}.final_bank_ledgers.{key}") if isinstance(pointer, Mapping) else {"status": "MALFORMED_POINTER"}
                sidecar_checks.append({"key": key, **checked})
                if checked.get("status") != "PASS":
                    raise ExclusionAuditError(f"{spec['label']} final-bank pointer is unavailable")
        for key in ("decision_contract", "identity_inventory"):
            pointer = descriptor.get(key)
            if isinstance(pointer, Mapping):
                checked = _descriptor_pointer(root, pointer, label=f"{spec['label']}.{key}")
                sidecar_checks.append({"key": key, **checked})
                if checked.get("status") != "PASS":
                    raise ExclusionAuditError(f"{spec['label']} {key} pointer is unavailable")
        opaque_checks: list[dict[str, Any]] = []
        for key in ("p06_opaque_reservation", "p08_opaque_reservation"):
            opaque = descriptor.get(key)
            if not isinstance(opaque, Mapping):
                continue
            privacy = opaque.get("privacy")
            if not isinstance(privacy, Mapping):
                raise ExclusionAuditError(f"{spec['label']} opaque privacy block changed")
            if key == "p06_opaque_reservation":
                if opaque.get("status") != "OPAQUE_HASH_RESERVATION_FOR_FUTURE_EXCLUSION" or any(privacy.get(field) is True for field in ("labels_or_answers_present", "record_ids_present", "row_indices_present", "source_text_present", "token_ids_present")):
                    raise ExclusionAuditError(f"{spec['label']} P06 opaque reservation is unsafe")
            else:
                if opaque.get("status") != "READY_FOR_HASH_ONLY_EXCHANGE" or any(privacy.get(field) is True for field in ("contains_record_ids", "contains_source_indices", "contains_source_text", "contains_token_ids", "contains_target_labels", "contains_truth")):
                    raise ExclusionAuditError(f"{spec['label']} P08 opaque reservation is unsafe")
            opaque_checks.append({"label": key, "status": opaque.get("status"), "hash_only": True})
        results.append({
            "label": spec["label"],
            "status": "PASS",
            "descriptor": {"path": spec["path"], "sha256": spec["sha256"], "bytes": spec["bytes"], "schema": spec["schema"]},
            "selection_ledger": {"label": spec["selection_label"], "path": spec["selection_path"], "sha256": spec["selection_sha256"], "bytes": selection_path.stat().st_size, "record_counts": counts, "pointer_new_identity_count": selection_pointer["new_identity_count"]},
            "source_pointers_declared": len(sources),
            "source_pointers_marked_available": sum(item.get("available") is True for item in sources),
            "aggregate_identity_counts": descriptor.get("identity_counts"),
            "individual_used_record_rows_in_descriptor": False,
            "new_used_record_rows": False,
            "producer_rule": "aggregate source-exclusion receipt; used records are exactly the hash-bound selection ledger rows",
            "sidecar_checks": sidecar_checks,
            "opaque_reservation_checks": opaque_checks,
        })
    return results

def _load_descriptor_pointer_proof(root: Path) -> dict[str, Any]:
    """Bind descriptor-only sources to concrete producer rules and counts.

    A hash-valid pointer is necessary but insufficient.  The proof records the
    source rule, expected row counts, and the exact identity ledger covered by
    that rule.  Unavailable pointers therefore remain failures and are never
    treated as harmless external assets.
    """
    proof: dict[str, Any] = {
        "schema": "token-reconstruction.trr-p11-descriptor-pointer-proof.v2",
        "sources": [],
    }
    selection_exclusion_proofs = _load_selection_exclusion_descriptor_proof(root)
    frozen_path = root / "experiments/TRR-0002/calibration/frozen_calibration.json"
    frozen = _read_json(frozen_path, label="TRR-0002 frozen calibration")
    frozen_pointers = []
    for key in ("fit_result", "plan", "selector_source"):
        pointer = frozen.get(key)
        if isinstance(pointer, Mapping):
            frozen_pointers.append(_descriptor_pointer(root, pointer, label=f"trr0002_calibration.{key}"))
    calibration_rule: dict[str, Any] = {"status": "FAIL"}
    by_label = {str(item.get("label")): item for item in frozen_pointers}
    fit_check = by_label.get("trr0002_calibration.fit_result", {})
    plan_check = by_label.get("trr0002_calibration.plan", {})
    selector_check = by_label.get("trr0002_calibration.selector_source", {})
    # The fit-result artifact contains outcomes, not source identity.  Its
    # absence is recorded but may be nonblocking when the frozen plan,
    # selector source, and covered source ledger prove the actual rule/counts.
    calibration_pointer_ok = (
        plan_check.get("status") == "PASS"
        and selector_check.get("status") == "PASS"
        and fit_check.get("status") in {"PASS", "UNAVAILABLE_POINTER"}
    )
    if frozen_pointers and calibration_pointer_ok:
        fit_path = _pointer_target(root, fit_check)
        plan_path = _pointer_target(root, plan_check)
        fit_value = _read_json(fit_path, label="TRR-0002 calibration fit") if fit_path else {}
        plan_value = _read_json(plan_path, label="TRR-0002 calibration plan") if plan_path else {}
        public = plan_value.get("public_development") if isinstance(plan_value, Mapping) else None
        source_plan_path = root / "experiments/TRR-0001/plan.json"
        try:
            source_plan = _read_json(source_plan_path, label="TRR-0001 public source plan")
        except ExclusionAuditError:
            source_plan = {}
        splits = source_plan.get("data", {}).get("selection", {}).get("splits", {}) if isinstance(source_plan.get("data"), Mapping) else {}
        development = splits.get("development", {}) if isinstance(splits, Mapping) else {}
        update = splits.get("target_update_train", {}) if isinstance(splits, Mapping) else {}
        def split_proof(split: Any, expected_count: int) -> bool:
            if not isinstance(split, Mapping) or split.get("count") != expected_count or not isinstance(split.get("records"), list):
                return False
            ids = [row.get("record_id") for row in split["records"] if isinstance(row, Mapping)]
            digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest() if ids else None
            return len(ids) == expected_count and all(isinstance(item, str) and item for item in ids) and split.get("record_ids_sha256") == digest

        record_commitments_ok = split_proof(development, 32) and split_proof(update, 64)
        fit_conditions_ok = fit_check.get("status") == "UNAVAILABLE_POINTER" or fit_value.get("fit_conditions") == frozen.get("fit_conditions")
        source_rule_ok = (
            isinstance(public, Mapping)
            and public.get("source") == "the 32 public development records fixed in experiments/TRR-0001/plan.json"
            and isinstance(public.get("update_training_source"), str)
            and public.get("update_training_source", "").startswith("the 64 public target_update_train records fixed in experiments/TRR-0001/plan.json")
            and public.get("sequence_geometry") == "32 records by 40 tokens including BOS"
            and fit_conditions_ok
            and record_commitments_ok
        )
        calibration_rule = {
            "status": "PASS" if source_rule_ok else "FAIL",
            "covered_identity_sources": ["trr0001_plan"] if source_rule_ok else [],
            "source_plan": {"path": "experiments/TRR-0001/plan.json", "sha256": p10.sha256_file(source_plan_path) if source_plan_path.is_file() else None},
            "development_count": development.get("count"),
            "target_update_train_count": update.get("count"),
            "fit_result_pointer_status": fit_check.get("status"),
            "fit_conditions": fit_value.get("fit_conditions") if fit_value else frozen.get("fit_conditions"),
            "record_commitments_ok": record_commitments_ok,
            "rule": public.get("source") if isinstance(public, Mapping) else None,
            "update_rule": public.get("update_training_source") if isinstance(public, Mapping) else None,
        }
    proof["sources"].append({
        "label": "trr0002_calibration",
        "pointer_checks": frozen_pointers,
        "identity_rows": "none_declared; source rule is bound to TRR-0001 plan",
        "covered_identity_sources": calibration_rule.get("covered_identity_sources", []),
        "source_rule_proof": calibration_rule,
        "status": "PASS" if calibration_rule.get("status") == "PASS" else "FAIL",
    })

    validation_path = root / "experiments/TRR-0005/public_validation_selection.json"
    validation = _read_json(validation_path, label="TRR-0005 public validation selection")
    validation_pointers = []
    fit_evidence = validation.get("fit_evidence")
    if isinstance(fit_evidence, Mapping):
        validation_pointers.append(_descriptor_pointer(root, fit_evidence, label="trr0005_public_validation_selection.fit_evidence"))
    distributions = validation.get("distributions")
    if isinstance(distributions, Mapping):
        for distribution, entry in distributions.items():
            if not isinstance(entry, Mapping):
                continue
            for index, candidate in enumerate(entry.get("candidates", [])):
                if isinstance(candidate, Mapping) and isinstance(candidate.get("curve_file"), Mapping):
                    validation_pointers.append(_descriptor_pointer(root, candidate["curve_file"], label=f"trr0005_public_validation_selection.{distribution}.curve_{index}"))
    validation_rule: dict[str, Any] = {"status": "FAIL"}
    evidence_checked = next((item for item in validation_pointers if item.get("label", "").endswith("fit_evidence")), None)
    evidence_path = _pointer_target(root, evidence_checked or {})
    if validation_pointers and all(item.get("status") == "PASS" for item in validation_pointers) and evidence_path:
        evidence = _read_json(evidence_path, label="TRR-0005 fit evidence")
        source_paths = {
            "original": root / "experiments/TRR-0005/public_activation_v1/original_fit_records.json",
            "enriched": root / "experiments/TRR-0005/public_activation_v1/enriched_fit_records.json",
            "validation": root / "experiments/TRR-0004/fit/adapter_v2/affine_validation_records.json",
        }
        expected_sha = {label: p10.sha256_file(path) if path.is_file() else None for label, path in source_paths.items()}
        dist_results: dict[str, Any] = {}
        for distribution in ("original", "enriched"):
            entry = evidence.get("distributions", {}).get(distribution, {}) if isinstance(evidence.get("distributions"), Mapping) else {}
            metadata = entry.get("data_metadata", {}) if isinstance(entry, Mapping) else {}
            payload = metadata.get("fit_payload", {}) if isinstance(metadata, Mapping) else {}
            resources = payload.get("resources", {}) if isinstance(payload, Mapping) else {}
            fit_resource = resources.get("fit_records", {}) if isinstance(resources, Mapping) else {}
            validation_resource = resources.get("validation_records", {}) if isinstance(resources, Mapping) else {}
            dist_results[distribution] = {
                "fit_record_count": payload.get("fit_record_count"),
                "fit_geometry": payload.get("geometry", {}).get("fit") if isinstance(payload.get("geometry"), Mapping) else None,
                "validation_geometry": payload.get("geometry", {}).get("validation") if isinstance(payload.get("geometry"), Mapping) else None,
                "fit_records_sha256": fit_resource.get("sha256"),
                "validation_records_sha256": validation_resource.get("sha256"),
                "fit_records_match": fit_resource.get("sha256") == expected_sha.get(distribution),
                "validation_records_match": validation_resource.get("sha256") == expected_sha.get("validation"),
            }
        rule_text = validation.get("selection_rule")
        rule_ok = isinstance(rule_text, str) and "maximum public validation-style accuracy mean" in rule_text and "ties use the earliest" in rule_text
        selection_ok = all(
            isinstance(entry, Mapping)
            and isinstance(entry.get("selected_method_id"), str)
            and isinstance(entry.get("selected_step"), int)
            for entry in (distributions or {}).values()
        ) if isinstance(distributions, Mapping) else False
        counts_ok = all(
            result.get("fit_record_count") == 1200
            and result.get("fit_geometry") == [1200, 192, 2048]
            and result.get("validation_geometry") == [48, 192, 2048]
            and result.get("fit_records_match") is True
            and result.get("validation_records_match") is True
            for result in dist_results.values()
        ) and set(dist_results) == {"original", "enriched"}
        validation_rule = {
            "status": "PASS" if rule_ok and selection_ok and counts_ok and evidence.get("status") == "JOINT_FIT_COMPLETE_NO_FINAL_EVALUATION" else "FAIL",
            "covered_identity_sources": ["trr0005_original_fit", "trr0005_enriched_fit", "trr0004_adapter_v2_validation"] if rule_ok and selection_ok and counts_ok else [],
            "selection_rule": rule_text,
            "distribution_proof": dist_results,
        }
    proof["sources"].append({
        "label": "trr0005_public_validation_selection",
        "pointer_checks": validation_pointers,
        "identity_rows": "none_declared; fit evidence is bound to both 1200-row ledgers and 48-row validation metadata",
        "covered_identity_sources": validation_rule.get("covered_identity_sources", []),
        "source_rule_proof": validation_rule,
        "status": "PASS" if validation_rule.get("status") == "PASS" else "FAIL",
    })

    p09_path = root / "experiments/TRR-P09/setup/public-validation-r1-audit.json"
    p09 = _read_json(p09_path, label="TRR-P09 public validation audit")
    p09_input = p09.get("inputs", {}).get("source_selection") if isinstance(p09.get("inputs"), Mapping) else None
    p09_check = _descriptor_pointer(root, p09_input, label="trr_p09_public_validation_audit.source_selection") if isinstance(p09_input, Mapping) else {"label": "trr_p09_public_validation_audit.source_selection", "status": "MALFORMED_POINTER"}
    p09_rule: dict[str, Any] = {"status": "FAIL"}
    p09_target = _pointer_target(root, p09_check)
    if p09_check.get("status") == "PASS" and p09_target:
        selection = _read_json(p09_target, label="TRR-0009 selection ledger")
        record_groups = selection.get("selection_rule", {}).get("records") if isinstance(selection.get("selection_rule"), Mapping) else None
        rows = [row for group in record_groups.values() for row in group] if isinstance(record_groups, Mapping) and all(isinstance(group, list) for group in record_groups.values()) else []
        fields_ok = bool(rows) and all(isinstance(row, Mapping) and isinstance(row.get("record_id"), str) and _is_sha256(row.get("public_record_sha256")) and _is_sha256(row.get("final_sequence_sha256")) for row in rows)
        p09_rule = {
            "status": "PASS" if fields_ok and len(rows) == 384 else "FAIL",
            "covered_identity_sources": ["trr0009_selection_v2"] if fields_ok and len(rows) == 384 else [],
            "record_count": len(rows),
            "required_fields": ["record_id", "public_record_sha256", "final_sequence_sha256"],
            "selection_rule_records_present": isinstance(record_groups, Mapping),
        }
    proof["sources"].append({
        "label": "trr_p09_public_validation_audit",
        "pointer_checks": [p09_check],
        "identity_rows": "selection ledger rows are individually hash-bound",
        "covered_identity_sources": p09_rule.get("covered_identity_sources", []),
        "source_rule_proof": p09_rule,
        "status": "PASS" if p09_rule.get("status") == "PASS" else "FAIL",
    })
    for descriptor_proof in selection_exclusion_proofs:
        proof["sources"].append({
            "label": descriptor_proof["label"],
            "pointer_checks": descriptor_proof["sidecar_checks"],
            "identity_rows": "aggregate descriptor only; no individual used-record rows",
            "covered_identity_sources": [descriptor_proof["selection_ledger"]["label"]],
            "source_rule_proof": descriptor_proof,
            "status": descriptor_proof["status"],
        })
    proof["proven_descriptor_labels"] = sorted(
        item["label"] for item in proof["sources"] if item.get("status") == "PASS"
    )
    proof["status"] = "PASS_DESCRIPTOR_POINTER_BINDINGS" if len(proof["proven_descriptor_labels"]) == len(proof["sources"]) else "PARTIAL_DESCRIPTOR_POINTER_BINDINGS"
    proof["source_identity_pointer_bindings_complete"] = proof["status"] == "PASS_DESCRIPTOR_POINTER_BINDINGS"
    return proof

def _build_closure_assessment(
    *,
    root: Path,
    bundles: Sequence[p10.IdentityBundle],
    recovered_bundles: Sequence[p10.IdentityBundle],
    alias_reconciliation: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    p04: P04Load,
    recovered_identity_proofs: Sequence[Mapping[str, Any]],
    replication_inputs: Mapping[str, Any],
    replication_proof: Mapping[str, Any],
    descriptor_pointer_proof: Mapping[str, Any],
    identity_gaps: Sequence[Mapping[str, Any]],
    descriptor_only_labels: set[str],
) -> dict[str, Any]:
    labels = {bundle.label for bundle in bundles}
    required_classes = {
        "gradient_fitting": {bundle.label for bundle in bundles if bundle.role == "fitting_bank"},
        "inherited_fitting": {"trr0001_manifest", "trr0002_public_finance_records", "trr0002_public_pile_records", "trr_p09_nested_b1"},
        "checkpoint_selection": {"trr0002_configuration_winner", "trr0004_selection_plan", "trr0009_selection_v2", "pr20_source_selection_binding"},
        "calibration": {"trr0002_calibration", "trr0004_affine_validation", "trr0004_adapter_v2_validation", "trr0005_public_validation_selection", "trr_p09_public_validation_audit"},
        "previously_opened_development": {"trr0001_manifest", "trr0002_fresh_observation_index", "trr0002_public_finance_records", "trr0002_public_pile_records", "trr0007_selection_exclusions", "trr0008_selection_exclusions", "trr0009_selection_v2_exclusions"},
        "previously_opened_evaluation": {"trr0004_selection_plan", "trr0006_selection", "trr0006_panel", "trr0006_public_observation_panel", "trr0007_selection", "trr0007_eval_panel", "trr0008_selection", "trr0008_eval_panel", "trr0009_original_selection", "trr0009_selection_v2", "trr0009_eval_panel", "pr20_source_selection_binding", "pr20_source_selection", "pr20_source_panel"},
    }
    class_results = {
        name: {
            "required_sources": sorted(source_labels),
            "present_sources": sorted(source_labels & labels),
            "missing_sources": sorted(source_labels - labels),
            "status": "PASS" if source_labels <= labels else "MISSING_REQUIRED_SOURCE",
        }
        for name, source_labels in required_classes.items()
    }
    alias_rows = [item for item in alias_reconciliation.get("per_source", ()) if isinstance(item, Mapping)]
    residual_sources = [
        {
            "label": item.get("label"),
            "rows_with_identity": int(item.get("rows_with_identity", 0)),
            "rows_with_verified_canonical_anchor": int(item.get("rows_with_verified_canonical_anchor", 0)),
            "rows_without_verified_canonical_anchor": int(item.get("rows_without_verified_canonical_anchor", 0)),
            "verified_short_rows_h128_inapplicable": int(item.get("verified_short_rows_h128_inapplicable", 0)),
            "verified_h40_rows": int(item.get("verified_h40_rows", 0)),
            "eligible_rows_without_verified_canonical_anchor": int(item.get("eligible_rows_without_verified_canonical_anchor", 0)),
            "unresolved_rows_without_verified_canonical_anchor": int(item.get("unresolved_rows_without_verified_canonical_anchor", 0)),
        }
        for item in alias_rows
        if int(item.get("unresolved_rows_without_verified_canonical_anchor", 0)) or int(item.get("eligible_rows_without_verified_canonical_anchor", 0))
    ]
    residual_unresolved = int(alias_reconciliation.get("unresolved_rows_without_verified_canonical_anchor", 0))
    residual_eligible = int(alias_reconciliation.get("eligible_rows_without_verified_canonical_anchor", 0))
    residual_without_anchor = int(alias_reconciliation.get("rows_without_verified_canonical_anchor", 0))
    unknown_row_fields = _unknown_identity_row_fields([*bundles, *recovered_bundles])
    required_labels = {spec.label for spec in p10.source_specs()}
    explicit_inventory = labels == required_labels and len(bundles) == len(required_labels)
    recovered_by_label = {str(item.get("label")): item for item in recovered_identity_proofs}
    p05 = recovered_by_label.get("trr0005_enriched_fit_public_token_identity", {})
    target = recovered_by_label.get("trr0006_p04_targetfit_public_identity", {})
    trr0003 = recovered_by_label.get("trr0003_fit_records_h40_public_identity", {})
    original_fit = recovered_by_label.get("trr0004_original_fit_public_token_identity", {})
    original_validation = recovered_by_label.get("trr0004_original_validation_public_token_identity", {})
    trr0001_h40 = recovered_by_label.get("trr0001_public_h40_identity", {})
    trr0002_h40 = recovered_by_label.get("trr0002_public_pile_h40_recovery_identity", {})
    target_plan = p04.proof.get("targetfit_plan_proof", {})
    p04_recovery = (
        p04.proof.get("producer_source_bytes_available") is True
        and p04.proof.get("top_level_exchange_digest_match") is True
        and p04.proof.get("h128_individual_hashes_available") is True
        and all(item.get("available") is True for item in p04.proof.get("recovered_ledger_proof", {}).values())
        and target_plan.get("status") == "PASS_TARGET_RULE_COUNTS_ONLY"
    )
    canonical_recovery = (
        len(recovered_identity_proofs) == 7
        and int(p05.get("h128_rows", 0)) == 350
        and int(p05.get("record_count", 0)) == 1200
        and int(target.get("h128_rows", 0)) == 212
        and int(target.get("record_count", 0)) == 256
        and int(trr0003.get("h40_rows", 0)) == 128
        and int(trr0003.get("record_count", 0)) == 128
        and int(trr0003.get("h128_rows", 0)) == 0
        and int(trr0003.get("h129_rows", 0)) == 0
        and int(original_fit.get("h128_rows", 0)) == 350
        and int(original_fit.get("h129_rows", 0)) == 343
        and int(original_fit.get("record_count", 0)) == 1200
        and len(original_fit.get("covered_source_labels", ())) == 4
        and int(original_validation.get("h128_rows", 0)) == 6
        and int(original_validation.get("h129_rows", 0)) == 5
        and int(original_validation.get("record_count", 0)) == 48
        and len(original_validation.get("covered_source_labels", ())) == 2
        and int(trr0001_h40.get("h40_rows", 0)) == 352
        and int(trr0001_h40.get("record_count", 0)) == 352
        and int(trr0001_h40.get("h128_rows", 0)) == 0
        and int(trr0001_h40.get("h129_rows", 0)) == 0
        and int(trr0002_h40.get("h40_rows", 0)) == 96
        and int(trr0002_h40.get("record_count", 0)) == 96
        and int(trr0002_h40.get("h128_rows", 0)) == 0
        and int(trr0002_h40.get("h129_rows", 0)) == 0
        and replication_proof.get("tensor_payload_opened") is False
    )
    descriptor_proven = set(descriptor_pointer_proof.get("proven_descriptor_labels", ()))
    aggregate_allowed = {item.get("label") for item in AGGREGATE_BINDINGS if isinstance(item, Mapping)} if aggregate.get("status") == "PASS_ALL_SIX" else set()
    if aggregate_allowed:
        aggregate_allowed.add("pr20_source_selection_binding")
    pointer_only_allowed = descriptor_proven | aggregate_allowed
    unresolved_identity_gaps = [
        dict(item) for item in identity_gaps if item.get("label") not in pointer_only_allowed
    ]
    all_eligible_anchored = residual_eligible == 0
    no_unresolved_identity_rows = residual_unresolved == 0
    tests = {
        "explicit_source_inventory": explicit_inventory,
        "required_source_classes": all(item["status"] == "PASS" for item in class_results.values()),
        "aggregate_panel_bindings": aggregate.get("status") == "PASS_ALL_SIX" and aggregate.get("count") == 6,
        "trr0009_selection_manifest": "trr0009_selection_v2" in labels,
        "p04_producer_order_and_h128": p04_recovery,
        "canonical_p05_and_targetfit_recovery": canonical_recovery,
        "replication_inputs_exact": bool(replication_inputs) and replication_proof.get("tensor_payload_opened") is False,
        "descriptor_pointer_bindings": descriptor_pointer_proof.get("source_identity_pointer_bindings_complete") is True,
        "all_eligible_rows_have_verified_canonical_anchor": all_eligible_anchored,
        "no_unresolved_identity_rows": no_unresolved_identity_rows,
        # Kept as an explicit compatibility label, but now uses all row-level
        # residuals rather than the prior unique-key collapse.
        "legacy_unique_rows_have_verified_canonical_anchor": all_eligible_anchored and no_unresolved_identity_rows,
        "no_row_level_unknown_identity_keys": not unknown_row_fields,
        "no_unresolved_identity_pointer_gaps": not unresolved_identity_gaps,
        "p03_contract_exclusion": True,
    }
    blockers: list[dict[str, Any]] = []
    if residual_unresolved or residual_eligible:
        blockers.append({
            "reason": "rows without a verified canonical row-level H128/H40/short proof",
            "rows_without_verified_canonical_anchor": residual_without_anchor,
            "eligible_rows_without_verified_canonical_anchor": residual_eligible,
            "unresolved_rows_without_verified_canonical_anchor": residual_unresolved,
            "per_source": residual_sources,
        })
    if unknown_row_fields:
        blockers.append({"reason": "row-level unknown identity fields", "fields": unknown_row_fields})
    if unresolved_identity_gaps:
        blockers.append({"reason": "unresolved individual identity source gaps", "sources": unresolved_identity_gaps})
    return {
        "status": "PASS_COMPLETE_ACCESSIBLE_SOURCE_COVERAGE" if all(tests.values()) else "PARTIAL_EXACT_PREFIX_COVERAGE",
        "coverage_complete": all(tests.values()),
        "selection_release": False,
        "tests": tests,
        "blockers": blockers,
        "required_source_classes": class_results,
        "legacy_alias_summary": {
            "prior_unique_identity_keys": int(alias_reconciliation.get("unique_uncovered_identity_keys_across_sources", 0)),
            "prior_unique_source_row_keys": int(alias_reconciliation.get("unique_uncovered_source_row_keys", 0)),
            "rows_with_identity": int(alias_reconciliation.get("rows_with_identity_across_sources", 0)),
            "rows_without_verified_canonical_anchor": residual_without_anchor,
            "rows_covered_by_own_bound_identity": sum(int(item.get("unique_uncovered_rows_with_own_bound_identity", 0)) for item in alias_rows),
            "rows_covered_by_verified_canonical_anchor": int(alias_reconciliation.get("rows_with_verified_canonical_anchor", 0)),
            "verified_short_rows_h128_inapplicable": int(alias_reconciliation.get("verified_short_rows_h128_inapplicable", 0)),
            "verified_h40_rows": int(alias_reconciliation.get("verified_h40_rows", 0)),
            "eligible_rows_without_verified_canonical_anchor": residual_eligible,
            "unresolved_rows_without_verified_canonical_anchor": residual_unresolved,
            "residual_unique_rows_without_verified_canonical_anchor": int(alias_reconciliation.get("unique_uncovered_identity_keys_without_verified_canonical_anchor", 0)),
            "sources_with_residual_rows": residual_sources,
        },
        "unknown_identity_row_fields": unknown_row_fields,
        "descriptor_pointer_proof": descriptor_pointer_proof,
        "descriptor_pointer_only_allowed": sorted(pointer_only_allowed),
        "unresolved_identity_gaps": unresolved_identity_gaps,
        "p03": {"status": "INTENTIONALLY_EXCLUDED_AND_UNOPENED", "coverage_complete_definition": "accessible explicit source inventory only"},
    }

def _build_closure_table(
    *,
    bundles: Sequence[p10.IdentityBundle],
    recovered_bundles: Sequence[p10.IdentityBundle],
    alias_reconciliation: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    alias_by_label = {
        str(item.get("label")): item
        for item in alias_reconciliation.get("per_source", ())
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bundle in [*bundles, *recovered_bundles]:
        if bundle.label in seen:
            continue
        seen.add(bundle.label)
        alias = alias_by_label.get(bundle.label, {})
        counts = bundle.counts()
        direct_rows = int(alias.get("rows_with_verified_canonical_anchor", 0))
        no_anchor_rows = int(alias.get("rows_without_verified_canonical_anchor", 0))
        anchor_labels = alias.get("canonical_anchor_labels", [])
        row_totals = {
            "rows_with_identity": int(alias.get("rows_with_identity", 0)),
            "rows_with_verified_canonical_anchor": direct_rows,
            "rows_without_verified_canonical_anchor": no_anchor_rows,
            "verified_short_rows_h128_inapplicable": int(alias.get("verified_short_rows_h128_inapplicable", 0)),
            "verified_h40_rows": int(alias.get("verified_h40_rows", 0)),
            "eligible_rows_without_verified_canonical_anchor": int(alias.get("eligible_rows_without_verified_canonical_anchor", 0)),
            "unresolved_rows_without_verified_canonical_anchor": int(alias.get("unresolved_rows_without_verified_canonical_anchor", 0)),
        }
        rows.append({
            "label": bundle.label,
            "role": bundle.role,
            "identity_counts": counts,
            "prior_unique_keys": int(alias.get("unique_uncovered_row_keys", 0)),
            "exact_evidence_coverage": {
                "path": str(bundle.path),
                "sha256": bundle.sha256,
                "h128_count": counts.get("h128_sequence_sha256", 0),
                "h129_count": counts.get("h129_sequence_sha256", 0),
                "verified_canonical_anchor_labels": anchor_labels,
                "row_totals": row_totals,
            },
            "row_totals": row_totals,
            "residual_unique_keys": int(alias.get("unique_uncovered_row_keys", 0)),
            "residual_unresolved_rows": row_totals["unresolved_rows_without_verified_canonical_anchor"],
        })
    return {
        "schema": "token-reconstruction.trr-p11-exclusion-closure-checkpoint.v1",
        "task_id": TASK_ID,
        "status": completion["status"],
        "coverage_complete": bool(completion["coverage_complete"]),
        "selection_release": False,
        "required_source_classes": completion["required_source_classes"],
        "prior_unique_keys": completion["legacy_alias_summary"]["prior_unique_identity_keys"],
        "residual_unique_keys": completion["legacy_alias_summary"]["residual_unique_rows_without_verified_canonical_anchor"],
        "rows_without_verified_canonical_anchor": completion["legacy_alias_summary"]["rows_without_verified_canonical_anchor"],
        "eligible_rows_without_verified_canonical_anchor": completion["legacy_alias_summary"]["eligible_rows_without_verified_canonical_anchor"],
        "unresolved_rows_without_verified_canonical_anchor": completion["legacy_alias_summary"]["unresolved_rows_without_verified_canonical_anchor"],
        "per_source": rows,
        "blockers": completion["blockers"],
        "access_boundary": {"payload_opened": False, "truth_opened": False, "p03_holdout_accessed": False, "new_selection_started": False},
    }


def write_closure_checkpoint(path: Path, table: Mapping[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or path.exists():
        raise ExclusionAuditError(f"refusing to overwrite closure checkpoint: {path}")
    payload = canonical_json(dict(table))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ExclusionAuditError(f"refusing to overwrite closure checkpoint: {path}") from exc
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "status": table.get("status")}


def _identity_union_path_string(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _identity_union_document(
    union: p10.IdentityBundle,
    *,
    status: str,
    coverage_complete: bool,
    selection_release: bool,
) -> dict[str, Any]:
    """Build the payload-free canonical identity export.

    The export intentionally contains only opaque identities and namespace
    scoped indices.  It carries no source paths, rendered text, token values,
    labels, predictions, or truth.  The audit receipt binds the resulting
    file path, bytes, and SHA-256 separately.
    """
    if status not in {IDENTITY_UNION_PARTIAL_STATUS, IDENTITY_UNION_COMPLETE_STATUS}:
        raise ExclusionAuditError(f"unsupported identity-union status: {status}")
    if status == IDENTITY_UNION_COMPLETE_STATUS and not coverage_complete:
        raise ExclusionAuditError("complete identity-union status requires complete coverage")
    if status == IDENTITY_UNION_PARTIAL_STATUS and (coverage_complete or selection_release):
        raise ExclusionAuditError("partial identity-union status cannot claim complete coverage or release")
    fields: dict[str, dict[str, list[str | int]]] = {}
    for field_name, by_namespace in sorted(union.values.items()):
        if field_name not in IDENTITY_UNION_FIELDS:
            raise ExclusionAuditError(f"union contains an unsupported identity field: {field_name}")
        fields[field_name] = {}
        for namespace, values in sorted(by_namespace.items(), key=lambda item: item[0].as_string()):
            encoded: list[str | int] = []
            for value in sorted(values, key=lambda item: (isinstance(item, str), str(item))):
                if field_name == "source_index":
                    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                        raise ExclusionAuditError("identity-union source_index is malformed")
                    encoded.append(value)
                elif field_name == "record_id":
                    if not isinstance(value, str) or not value:
                        raise ExclusionAuditError("identity-union record_id is malformed")
                    encoded.append(value)
                else:
                    if not _is_sha256(value):
                        raise ExclusionAuditError(f"identity-union {field_name} is not an opaque SHA-256")
                    encoded.append(value.casefold())
            fields[field_name][namespace.as_string()] = encoded
    return {
        "schema": IDENTITY_UNION_SCHEMA,
        "task_id": TASK_ID,
        "status": status,
        "coverage_complete": coverage_complete,
        "selection_release": selection_release,
        "fields": fields,
        "counts": union.counts(),
        "namespace_counts": union.namespace_counts(),
        "contains_source_text": False,
        "contains_token_values": False,
        "contains_labels": False,
        "contains_predictions": False,
        "contains_truth": False,
        "access_boundary": {
            "source_text_read": False,
            "source_text_serialized": False,
            "source_tokens_serialized": False,
            "token_values_emitted": False,
            "labels_read": False,
            "predictions_read": False,
            "truth_or_scores_read": False,
            "p03_holdout_accessed": False,
            "new_selection_started": False,
        },
    }


def write_identity_union_export(
    path: Path,
    union: p10.IdentityBundle,
    *,
    root: Path,
    status: str = IDENTITY_UNION_PARTIAL_STATUS,
    coverage_complete: bool = False,
    selection_release: bool = False,
) -> dict[str, Any]:
    """Write an immutable sanitized identity union and return its descriptor."""
    path = path.resolve()
    if path.is_symlink() or path.exists():
        raise ExclusionAuditError(f"refusing to overwrite existing identity-union export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _identity_union_document(
        union,
        status=status,
        coverage_complete=coverage_complete,
        selection_release=selection_release,
    )
    payload = canonical_json(document)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ExclusionAuditError(f"refusing to overwrite existing identity-union export: {path}") from exc
    return {
        "schema": IDENTITY_UNION_SCHEMA,
        "path": _identity_union_path_string(path, root=root),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "status": status,
        "coverage_complete": coverage_complete,
        "selection_release": selection_release,
        "union_identity_counts": union.counts(),
    }


def _parse_identity_union_namespace(raw: Any) -> p10.Namespace:
    if not isinstance(raw, str):
        raise ExclusionAuditError("identity-union namespace key is not a string")
    parts = raw.split("|")
    if len(parts) != 4 or any(not isinstance(part, str) or not part for part in parts):
        raise ExclusionAuditError("identity-union namespace key is malformed")
    return p10.Namespace(*parts)


def load_identity_union_export(
    path: Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> p10.IdentityBundle:
    """Reconstruct an exclusion bundle from the sanitized union export."""
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ExclusionAuditError(f"identity-union export is unavailable: {path}")
    value = _read_json(path, label="identity-union export")
    if value.get("schema") != IDENTITY_UNION_SCHEMA or value.get("task_id") != TASK_ID:
        raise ExclusionAuditError("identity-union export schema changed")
    status = value.get("status")
    complete = value.get("coverage_complete")
    release = value.get("selection_release")
    if status not in {IDENTITY_UNION_PARTIAL_STATUS, IDENTITY_UNION_COMPLETE_STATUS} or not isinstance(complete, bool) or not isinstance(release, bool):
        raise ExclusionAuditError("identity-union export status fields are malformed")
    if status == IDENTITY_UNION_COMPLETE_STATUS and not complete:
        raise ExclusionAuditError("complete identity-union export has incomplete coverage flag")
    if status == IDENTITY_UNION_PARTIAL_STATUS and (complete or release):
        raise ExclusionAuditError("partial identity-union export has unsafe release flags")
    forbidden = {"source_text", "source_tokens", "token_ids", "input_ids", "labels", "predictions", "truth", "oracle", "activations", "model_weights"}
    if forbidden.intersection(value):
        raise ExclusionAuditError("identity-union export contains forbidden payload fields")
    access = value.get("access_boundary")
    if not isinstance(access, Mapping) or any(access.get(key) is True for key in ("source_text_read", "source_text_serialized", "source_tokens_serialized", "token_values_emitted", "labels_read", "predictions_read", "truth_or_scores_read", "p03_holdout_accessed", "new_selection_started")):
        raise ExclusionAuditError("identity-union export access boundary is unsafe")
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise ExclusionAuditError("identity-union export fields are absent")
    bundle = p10.IdentityBundle(
        label="sanitized_identity_union_export",
        role="union",
        path=path,
        sha256=p10.sha256_file(path),
        bytes=path.stat().st_size,
        schema=IDENTITY_UNION_SCHEMA,
        status=status,
    )
    for field_name, by_namespace in fields.items():
        if field_name not in IDENTITY_UNION_FIELDS or not isinstance(by_namespace, Mapping):
            raise ExclusionAuditError(f"identity-union field mapping is malformed: {field_name}")
        for raw_namespace, values in by_namespace.items():
            namespace = _parse_identity_union_namespace(raw_namespace)
            if not isinstance(values, list):
                raise ExclusionAuditError(f"identity-union values are not a list: {field_name}")
            for item in values:
                if field_name == "source_index":
                    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                        raise ExclusionAuditError("identity-union source_index value is malformed")
                    bundle.add(field_name, item, namespace)
                elif field_name == "record_id":
                    if not isinstance(item, str) or not item:
                        raise ExclusionAuditError("identity-union record_id value is malformed")
                    bundle.add(field_name, item, namespace)
                else:
                    if not _is_sha256(item):
                        raise ExclusionAuditError(f"identity-union {field_name} value is malformed")
                    bundle.add(field_name, item.casefold(), namespace)
    counts = value.get("counts")
    if not isinstance(counts, Mapping) or dict(counts) != bundle.counts():
        raise ExclusionAuditError("identity-union counts do not match values")
    namespaces = value.get("namespace_counts")
    if namespaces is not None and namespaces != bundle.namespace_counts():
        raise ExclusionAuditError("identity-union namespace counts do not match values")
    if expected_counts is not None and dict(expected_counts) != bundle.counts():
        raise ExclusionAuditError("identity-union counts do not match audit counts")
    return bundle

def build_audit(
    *,
    root: Path,
    pr20_root: Path | None = None,
    public_payload: Path | None = None,
    public_payload_metadata: Path | None = None,
    identity_union_output: Path | None = None,
    closure_output: Path | None = None,
    include_recovered_identity_exports: bool = False,
) -> dict[str, Any]:
    bundles, p04 = _load_specs(root, pr20_root)
    replication_inputs, replication_input_proof = _load_replication_input_binding(root)
    descriptor_pointer_proof = _load_descriptor_pointer_proof(root)
    recovered_bundles: list[p10.IdentityBundle] = []
    recovered_identity_proofs: list[dict[str, Any]] = []
    producer_lineage_proofs: list[dict[str, Any]] = []
    verified_lineage: dict[str, dict[str, set[tuple[str, tuple[str, ...]]]]] = {}
    if include_recovered_identity_exports:
        binding = _load_canonical_pass_binding(root)
        p05_bundle, p05_proof = _load_public_identity_export(
            root,
            identity_path=P11_P05_IDENTITY_PATH,
            identity_sha256=P11_P05_IDENTITY_SHA256,
            recovery_path=P11_P05_RECOVERY_PATH,
            recovery_sha256=P11_P05_RECOVERY_SHA256,
            expected_identity_schema="token-reconstruction.trr-p11-public-token-identity-rows.v1",
            expected_recovery_schema="token-reconstruction.trr-p11-public-token-identity-recovery.v1",
            expected_identity_status="PASS_PUBLIC_TOKEN_IDENTITY_RECOVERY",
            label="trr0005_enriched_fit_public_token_identity",
            role="inherited_fitting",
            binding=binding,
        )
        target_bundle, target_proof = _load_public_identity_export(
            root,
            identity_path=P11_P04_TARGET_IDENTITY_PATH,
            identity_sha256=P11_P04_TARGET_IDENTITY_SHA256,
            recovery_path=P11_P04_TARGET_RECOVERY_PATH,
            recovery_sha256=P11_P04_TARGET_RECOVERY_SHA256,
            expected_identity_schema="token-reconstruction.trr-p11-p04-targetfit-identity-rows.v1",
            expected_recovery_schema="token-reconstruction.trr-p11-p04-targetfit-identity-recovery.v1",
            expected_identity_status="PASS_P04_TARGETFIT_EXACT_RENDERED_H129_H128_RECOVERY",
            label="trr0006_p04_targetfit_public_identity",
            role="evaluator_target_fit",
            binding=binding,
        )
        trr0003_bundle, trr0003_proof = _load_trr0003_h40_identity_export(root)
        original_bundles, original_proofs, original_lineage = _load_original_recovery_lineage(root)
        h40_bundles, h40_proofs, h40_lineage, h40_lineage_proofs = _load_h40_recovery_exports(root)
        winner_lineage, winner_lineage_proof = _load_trr0002_winner_lineage(root, bundles, h40_bundles)
        recovered_bundles.extend((p05_bundle, target_bundle, trr0003_bundle, *original_bundles, *h40_bundles))
        recovered_identity_proofs.extend((p05_proof, target_proof, trr0003_proof, *original_proofs, *h40_proofs))
        verified_lineage.update(original_lineage)
        verified_lineage.update(h40_lineage)
        verified_lineage.update(winner_lineage)
        producer_lineage_proofs.extend((*h40_lineage_proofs, winner_lineage_proof))
    rehash_bundle: p10.IdentityBundle | None = None
    rehash_receipt: dict[str, Any]
    if public_payload is not None or public_payload_metadata is not None:
        if public_payload is None or public_payload_metadata is None:
            raise ExclusionAuditError("public payload and metadata must be supplied together")
        rehash_bundle, rehash_receipt = rehash_public_token_payload(payload_path=public_payload, metadata_path=public_payload_metadata)
        bundles_for_union = [*bundles, *recovered_bundles, rehash_bundle]
    else:
        bundles_for_union = [*bundles, *recovered_bundles]
        rehash_receipt = {
            "status": "PENDING_ROOT_LEASE",
            "rows_seen": 0,
            "h128_rows": 0,
            "short_rows_h128_inapplicable": 0,
            "payload_not_opened": True,
            "planned_payload": "TRR-0007/support/broader_capture_v2/enriched_fit_cut4.safetensors",
            "planned_metadata": "TRR-0007/support/broader_capture_v2/enriched_fit_records.json",
            "planned_payload_sha256": "a55814759dfa9d2567587935063fc49e44d8bff949c50014793deb982ebdf35d",
            "planned_payload_bytes": 947176760,
            "resource_plan": {
                "rows": 1200,
                "tensors_read": ["token_ids", "attention_mask"],
                "tensor_bytes_estimate": 1200 * 192 * 4 + 1200 * 192,
                "tensors_not_read": ["activations", "position_ids", "post_bos_selector_large", "post_bos_selector_small"],
                "model_loaded": False,
                "expected_peak_tensor_bytes": 1200 * 192 * 4 + 1200 * 192,
                "safety_margin": "payload is 947176760 bytes on disk; safetensors selective reads avoid activation/model allocation",
            },
        }
    union = p10.merge_bundles(bundles_for_union)
    by_label = {bundle.label: bundle for bundle in bundles}
    aggregate = validate_aggregate_bindings(root=root, pr20_root=pr20_root)
    sequence_report = _sequence_gap_report([*bundles, *recovered_bundles], root=root, p04=p04)
    alias_reconciliation = _reconcile_legacy_aliases([*bundles, *recovered_bundles], root=root, verified_lineage=verified_lineage)
    unknown = [
        {
            "label": bundle.label,
            "role": bundle.role,
            "keys": dict(sorted(bundle.unknown_identity_keys.items())),
            "reason": "unrecognized identity/hash fields are reported but excluded without producer proof",
        }
        for bundle in bundles
        if bundle.unknown_identity_keys
    ]
    probe = _synthetic_probe()
    identity_gaps = [
        {
            "label": bundle.label,
            "role": bundle.role,
            "reason": "loaded metadata contains no individual comparable identity fields",
        }
        for bundle in bundles
        if not bundle.values
    ]
    descriptor_only_labels = {
        "trr0002_calibration",
        "trr0005_public_validation_selection",
        "trr0007_selection_exclusions",
        "trr0008_selection_exclusions",
        "trr0009_selection_v2_exclusions",
        "trr_p09_public_validation_audit",
        "pr20_source_selection_binding",
    }
    completion = _build_closure_assessment(
        root=root,
        bundles=bundles,
        recovered_bundles=recovered_bundles,
        alias_reconciliation=alias_reconciliation,
        aggregate=aggregate,
        p04=p04,
        recovered_identity_proofs=recovered_identity_proofs,
        replication_inputs=replication_inputs,
        replication_proof=replication_input_proof,
        descriptor_pointer_proof=descriptor_pointer_proof,
        identity_gaps=identity_gaps,
        descriptor_only_labels=descriptor_only_labels,
    )
    closure_table = _build_closure_table(
        bundles=bundles,
        recovered_bundles=recovered_bundles,
        alias_reconciliation=alias_reconciliation,
        completion=completion,
    )
    targetfit_recovered = any(
        proof.get("label") == "trr0006_p04_targetfit_public_identity"
        for proof in recovered_identity_proofs
    )
    if targetfit_recovered:
        for item in sequence_report:
            if item.get("label") == "trr0006_p04_targetfit":
                item.update({
                    "role": "evaluator_target_fit",
                    "status": "PASS_EXACT_RENDERED_H129_H128_RECOVERY",
                    "record_rows_seen": 256,
                    "direct_h128_count": 212,
                    "verified_short_rows_h128_inapplicable": 44,
                    "eligible_rows_by_declared_length": 212,
                    "eligible_rows_without_h128": 0,
                })
    p04_result_proof = dict(p04.proof)
    p04_result_proof["targetfit_individual_hashes_available"] = targetfit_recovered
    p04_result_proof["targetfit_recovery_overlay"] = next(
        (dict(proof) for proof in recovered_identity_proofs if proof.get("label") == "trr0006_p04_targetfit_public_identity"),
        {"status": "NOT_REQUESTED"},
    )
    p04_result_proof["status"] = (
        "PASS_PRODUCER_CONVENTION_VERIFIED_H128_AND_TARGETFIT_EXACT_RECOVERY"
        if targetfit_recovered else p04.proof.get("status")
    )
    p04_result_gaps = [
        gap for gap in p04.gaps
        if not (targetfit_recovered and str(gap.get("field", "")).startswith("targetfit."))
    ]
    source_inventory = []
    for bundle in bundles_for_union:
        source_inventory.append(
            {
                "label": bundle.label,
                "role": bundle.role,
                "path": str(bundle.path),
                "bytes": bundle.bytes,
                "sha256": bundle.sha256,
                "schema": bundle.schema,
                "status": bundle.status,
                "identity_counts": bundle.counts(),
                "namespace_counts": bundle.namespace_counts(),
                "unrecognized_identity_key_counts": dict(sorted(bundle.unknown_identity_keys.items())),
                "metadata_notes": bundle.metadata_notes,
            }
        )
    result = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "status": "COMPLETE_CANONICAL_SEQUENCE_EXCLUSION_AUDIT" if completion["coverage_complete"] else "PARTIAL_CANONICAL_SEQUENCE_EXCLUSION_AUDIT",
        "coverage_complete": completion["coverage_complete"],
        "selection_release": False,
        "source_count": len(bundles),
        "source_inventory": source_inventory,
        "required_source_classes": ["gradient_fitting", "inherited_fitting", "checkpoint_selection", "calibration", "previously_opened_development", "previously_opened_evaluation"],
        "union_identity_counts": union.counts(),
        "union_namespace_counts": union.namespace_counts(),
        "required_sources": [spec.label for spec in p10.source_specs()],
        "source_class_coverage": {
            "gradient_fitting": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0003_fit_records", "trr0004_affine_fit", "trr0004_adapter_v2_fit", "trr0005_original_fit", "trr0005_enriched_fit", "trr0007_original_fit", "trr0007_enriched_fit", "trr_p09_nested_b1"],
            },
            "inherited_fitting": {
                "status": "INCLUDED_WHEN_EXPLICIT_SOURCE_OR_PUBLIC_PAYLOAD_IS_BOUND",
                "sources": ["trr0001_manifest", "trr0001_plan", "trr0001_r1_selection_reveal", "trr0002_public_finance_records", "trr0002_public_pile_records", "trr_p09_nested_b1"],
            },
            "checkpoint_selection": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0002_configuration_winner", "trr0004_selection_plan", "trr0009_selection_v2", "pr20_source_selection_binding"],
            },
            "calibration": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0002_calibration", "trr0004_affine_validation", "trr0004_adapter_v2_validation", "trr0005_public_validation_selection", "trr_p09_public_validation_audit"],
            },
            "previously_opened_development": {
                "status": "INCLUDED_METADATA_LEDGER_UNION",
                "sources": ["trr0001_manifest", "trr0001_plan", "trr0001_r1_selection_reveal", "trr0002_fresh_observation_index", "trr0002_public_finance_records", "trr0002_public_pile_records", "trr0007_selection_exclusions", "trr0008_selection_exclusions", "trr0009_selection_v2_exclusions"],
            },
            "previously_opened_evaluation": {
                "status": "INCLUDED_AGGREGATE_BINDINGS_AND_SELECTION_LEDGER",
                "sources": ["trr0004_selection_plan", "trr0006_selection", "trr0006_panel", "trr0006_public_observation_panel", "trr0007_selection", "trr0007_eval_panel", "trr0008_selection", "trr0008_eval_panel", "trr0009_original_selection", "trr0009_selection_v2", "trr0009_eval_panel", "pr20_source_selection_binding", "pr20_source_selection", "pr20_source_panel", "trr0003_opened_panel_metadata", "trr0004_opened_panel_metadata", "trr0005_opened_panel_metadata"],
            },
        },
        "agent1_replication_assets": {
            "status": "PUBLIC_METADATA_AND_EXACT_REPLICATION_INPUTS_BOUND",
            "included": True,
            "note": "B0/B1 manifest and ordered-identity bytes are hash-bound; no activation tensor or selected weight was opened by this audit.",
            "metadata_sources": alias_reconciliation["replication_metadata"],
            "canonical_identity_exports": [proof.get("label") for proof in recovered_identity_proofs],
        },
        "replication_inputs": replication_inputs,
        "replication_input_proof": replication_input_proof,
        "descriptor_pointer_proof": descriptor_pointer_proof,
        "producer_lineage_proofs": producer_lineage_proofs,
        "completion_assessment": completion,
        "closure_table": closure_table,
        "trr0009_selection_manifest_required_and_loaded": "trr0009_selection_v2" in by_label,
        "aggregate_panel_binding": aggregate,
        "p04_convention_proof": p04_result_proof,
        "canonical_sequence_audit": {
            "status": "COMPLETE_ACCESSIBLE_EXACT_PREFIX_AUDIT" if completion["coverage_complete"] else "PARTIAL_EXACT_PREFIX_AUDIT",
            "hash_convention": "SHA-256 of first 128 active BOS-inclusive IDs encoded as little-endian signed int32 bytes",
            "h128_and_h129_are_distinct_namespaces": True,
            "per_source": sequence_report,
            "public_payload_rehash": rehash_receipt,
            "recovered_identity_exports": recovered_identity_proofs,
            "legacy_alias_reconciliation": alias_reconciliation,
        },
        "coverage": {
            "coverage_complete": completion["coverage_complete"],
            "accessible_explicit_sources_loaded": completion["tests"]["explicit_source_inventory"],
            "accessible_explicit_source_count": len(bundles),
            "explicit_source_spec_count": len(p10.source_specs()),
            "descriptor_only_sources": sorted(descriptor_only_labels),
            "identity_gaps": identity_gaps,
            "unrecognized_identity_key_gaps": unknown,
            "aggregate_panel_binding_status": aggregate["status"],
            "p04_targetfit_individual_hashes_available": targetfit_recovered,
            "p04_h128_individual_hashes_available": p04.proof.get("h128_individual_hashes_available") is True,
            "public_payload_h128_rehash_status": rehash_receipt["status"],
            "recovered_identity_exports_status": "PASS_BOUND_P05_P04_TARGETFIT_AND_TRR0003_H40" if recovered_identity_proofs else "NOT_REQUESTED",
        },
        "coverage_gaps": (
            [
                "P03 sealed holdout is intentionally unopened and absent from this inventory; coverage_complete means complete over the explicit accessible inventory.",
                "TRR-0002 Pile remains H40-only because every opened row is exactly 40 tokens; TRR-0002 Finance H128 is derived only for exact active rows of length at least 128, with shorter rows marked inapplicable. Historical active/H40 fields remain candidate rejection keys.",
                "No new candidate rows were scanned and no selection/capture/prediction/truth operation was performed.",
                "PR20 final sources are explicitly development/selection-overlapping and cannot be reused as an independent final panel.",
            ]
            + (["P04 targetfit identities are bound by the exact rendered/H129/H128 recovery export; the retained P04 exchange itself remains counts-only for targetfit."] if targetfit_recovered else ["P04 targetfit public_record_sha256 and truncated_sequence_sha256 arrays remain counts-only until an exact per-record recovery is bound."])
            + (["All eligible identity rows are anchored to a verified canonical row proof; residual eligible/unresolved rows: 0."] if completion["coverage_complete"] else [f"Residual row-level canonical proof gaps remain: {completion['legacy_alias_summary']['eligible_rows_without_verified_canonical_anchor']} eligible rows and {completion['legacy_alias_summary']['unresolved_rows_without_verified_canonical_anchor']} unresolved rows; per-source totals are in completion_assessment.blockers and closure_table."])
            + ([f"{completion['legacy_alias_summary']['rows_without_verified_canonical_anchor']} rows have no cross-row anchor but are verified short (<128 active tokens), so H128 is inapplicable; they are excluded from eligible/unresolved residuals."] if completion['legacy_alias_summary']['rows_without_verified_canonical_anchor'] else [])
            + p04_result_gaps
        ),
        "access_boundary": {
            "source_text_read": False,
            "source_or_token_payload_emitted": False,
            "historical_panel_json_parsed": True,
            "historical_panel_identity_fields_inspected": True,
            "historical_panel_payload_fields_skipped": True,
            "activation_payload_read": False,
            "model_loaded": False,
            "truth_or_scores_read": False,
            "new_panel_selected": False,
            "p03_holdout_accessed": False,
            "public_token_ids_hashed_only_when_root_lease_supplied": True,
        },
        "hash_conventions": {
            "record_id": "global exact string identity",
            "rendered_sha256": "global only when producer field is explicitly bound to rendered/public record bytes",
            "h128_sequence_sha256": "global SHA-256 of exactly first 128 active BOS-inclusive IDs as little-endian signed-int32 bytes",
            "h129_sequence_sha256": "separate SHA-256 of first 129 active BOS-inclusive IDs; never compared with H128",
            "h40_sequence_sha256": "TRR-0001 raw H40: SHA-256 of first 40 BOS-inclusive IDs as little-endian signed-int32 bytes; distinct from TRR2 H40",
            "source_index": "dataset/style/split/revision namespace scoped",
            "trr0002_active_token_ids_sha256": "TRR2 Finance active IDs, little-endian signed int32 bytes; exact active rows of length at least 128 additionally prove canonical H128",
            "trr0002_h40_token_ids_sha256": "TRR2 Pile H40 tensor-header plus little-endian int64 bytes; H128 is inapplicable to its exact 40-token opened rows",
        },
        "regression": probe,
        "notes": [
            "This is a prior-identity union and canonical-prefix audit, not source selection or capacity certification.",
            "Descriptor-only pointer/count receipts are not interpreted as zero overlap.",
            "The P10 receipts and code remain unchanged; r8 and r10 are superseded by the verified TRR-0004 alias correction and TRR-0003 H40 overlay recorded below.",
        ],
        "superseded_artifacts": [
            {"path": "experiments/TRR-P11/exclusions/recovery_identity_audit_r7.json", "sha256": "b0357082d50082231b0b53e61a77a8e9159e72351689993fbe63502d7a7491e3", "reason": "rejected: unique-key alias collapse could report complete while row-level canonical anchors were absent"},
            {"path": "experiments/TRR-P11/exclusions/identity_union_export_r4.json", "sha256": "449ddfb1f0752be5ebdbc5eba04fff831c28d0bbc1601ad281b21ca1e93e72f3", "reason": "superseded by strong-commitment row-level closure"},
            {"path": "experiments/TRR-P11/exclusions/closure_checkpoint_r3.json", "sha256": "98ace5624574aad6584bf7d297180441ac8c480e478c9afcbc7957c6fea982e9", "reason": "superseded by fail-closed all-row closure"},
            {"path": "experiments/TRR-P11/exclusions/recovery_identity_audit_r8.json", "sha256": "a41d78ce545dbc16b36ca4e169965f8029dfc52340dbbfb72b4df002ad61e791", "reason": "superseded by verified TRR-0004 producer-specific H40/H128 mapping; r8 left 48 rows falsely unresolved"},
            {"path": "experiments/TRR-P11/exclusions/recovery_identity_audit_r10.json", "sha256": "334f0f08589193203df5a058d581886f4e376c03149d8d2d3838f4aca3425149", "reason": "superseded by the verified TRR-0003 public H40 overlay; r10 retained 72 TRR-0003 rows without their recovered H40 proof"},
            {"path": "experiments/TRR-P11/exclusions/recovery_identity_audit_r11.json", "sha256": "74b6b6a60bb4247875784a7296a4f4532944f9776f6e8225b19b8288e726deee", "reason": "superseded by the H40 row-field validator fix; r11 reported the verified H40 field as an unknown identity key"},
            {"path": "experiments/TRR-P11/exclusions/recovery_identity_audit_r13.json", "sha256": "b9f0b990f434f3eb41cc5860a898df1d238514a90e04e44dd11f35a0b6b2d214", "reason": "superseded by integration of the completed original-fit/validation identities and TRR7/8/9 descriptor-ledger proofs; r13 retained 1,412 already-recovered eligible rows as unresolved"},
        ],
    }
    if identity_union_output is not None:
        export = write_identity_union_export(
            identity_union_output,
            union,
            root=root,
            status=IDENTITY_UNION_COMPLETE_STATUS if completion["coverage_complete"] else IDENTITY_UNION_PARTIAL_STATUS,
            coverage_complete=completion["coverage_complete"],
            selection_release=False,
        )
        if export["union_identity_counts"] != result["union_identity_counts"]:
            raise ExclusionAuditError("identity-union export counts do not bind audit counts")
        result["identity_union_export"] = export
    else:
        result["identity_union_export"] = {
            "schema": IDENTITY_UNION_SCHEMA,
            "status": "NOT_REQUESTED",
            "union_identity_counts": result["union_identity_counts"],
        }
    if closure_output is not None:
        result["closure_checkpoint"] = write_closure_checkpoint(closure_output, closure_table)
    else:
        result["closure_checkpoint"] = {"schema": closure_table["schema"], "status": "NOT_REQUESTED"}
    return result


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--pr20-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-token-payload", type=Path)
    parser.add_argument("--public-token-metadata", type=Path)
    parser.add_argument("--identity-union-output", type=Path)
    parser.add_argument("--closure-output", type=Path)
    parser.add_argument("--include-recovered-identity-exports", action="store_true")
    args = parser.parse_args(argv)
    result = build_audit(
        root=args.root.resolve(),
        pr20_root=args.pr20_root.resolve(),
        public_payload=args.public_token_payload.resolve() if args.public_token_payload else None,
        public_payload_metadata=args.public_token_metadata.resolve() if args.public_token_metadata else None,
        identity_union_output=args.identity_union_output.resolve() if args.identity_union_output else None,
        closure_output=args.closure_output.resolve() if args.closure_output else None,
        include_recovered_identity_exports=args.include_recovered_identity_exports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("xb") as handle:
            handle.write(canonical_json(result))
    except FileExistsError as exc:
        raise ExclusionAuditError(f"refusing to overwrite existing P11 audit receipt: {args.output}") from exc
    print(json.dumps({"output": str(args.output), "status": result["status"], "source_count": result["source_count"], "aggregate": result["aggregate_panel_binding"]["status"], "rehash": result["canonical_sequence_audit"]["public_payload_rehash"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
