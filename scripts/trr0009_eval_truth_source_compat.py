"""Truth-boundary source-evidence adapter for the TRR-0009 public snapshot.

The frozen tokenizer binding is a directory containing three immutable
Hugging Face snapshot entries.  The generic truth boundary previously passed
that directory to its regular-file record helper after labels were materialized
in memory.  This adapter preserves the frozen descriptor convention by using
the trusted public descriptor constructors: the directory path is checked as a
directory, each component's resolved blob bytes/hash and snapshot symlink are
checked, and every public Arrow binding is checked.  It does not load source
records itself and does not alter the frozen selection or truth sidecar.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator

import torch
from safetensors.torch import save_file as _safetensors_save_file

from scripts import trr0005_produce_confirmation as trusted
from scripts import trr0009_eval_contract as contract
from scripts import trr0009_eval_gate_compat as gate_compat
from scripts import trr0009_eval_truth as truth


SOURCE_EVIDENCE_SCHEMA = "token-reconstruction.trr0009-truth-source-evidence-compat.v1"
ADAPTER_PATH = Path(__file__).resolve()


class TruthSourceCompatibilityError(truth.TruthGateError):
    """Raised when frozen public source descriptors do not match inputs."""


def _root(value: Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise TruthSourceCompatibilityError(f"repository root unavailable: {root}")
    return root


def _resolve(value: Path | str, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _record(path: Path, *, root: Path, description: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    try:
        return contract.validate_file_record(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": contract.sha256_file(path),
            },
            repository_root=root,
            description=description,
            verify=True,
        )
    except (OSError, contract.ContractError) as exc:
        raise TruthSourceCompatibilityError(str(exc)) from exc


def _same_record(actual: Mapping[str, Any], expected: Mapping[str, Any], *, description: str) -> None:
    for key in ("path", "bytes", "sha256"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise TruthSourceCompatibilityError(f"{description} {key} binding changed")


def _public_sources(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    sources = selection.get("public_sources_frozen")
    if not isinstance(sources, Mapping):
        raise TruthSourceCompatibilityError("selection public source descriptors are absent")
    for key in ("tokenizer", "pile", "finance"):
        if not isinstance(sources.get(key), Mapping):
            raise TruthSourceCompatibilityError(f"selection {key} descriptor is absent")
    return sources


def _validate_arrow_inputs(
    selection: Mapping[str, Any],
    *,
    root: Path,
    style: str,
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    sources = _public_sources(selection)
    descriptor = sources[style]
    frozen_files = descriptor.get("arrow_files")
    if not isinstance(frozen_files, list) or len(frozen_files) != len(paths):
        raise TruthSourceCompatibilityError(f"{style} Arrow file count differs from frozen selection")
    # The trusted descriptor constructor records each public Arrow file.  The
    # equality check below binds dataset identity, revision, and reserved range
    # as well as the exact file records without reading rows here.
    try:
        actual_descriptor = trusted._dataset_descriptor(paths, style=style)
    except Exception as exc:
        raise TruthSourceCompatibilityError(f"{style} public Arrow descriptor is unavailable") from exc
    if dict(actual_descriptor) != dict(descriptor):
        raise TruthSourceCompatibilityError(f"{style} public source differs from frozen selection")
    checked: list[dict[str, Any]] = []
    for index, (actual, frozen) in enumerate(zip(actual_descriptor["arrow_files"], frozen_files)):
        if not isinstance(frozen, Mapping):
            raise TruthSourceCompatibilityError(f"{style} Arrow descriptor is malformed: {index}")
        checked_record = _record(paths[index], root=root, description=f"truth {style} Arrow {index}")
        _same_record(checked_record, frozen, description=f"truth {style} Arrow {index}")
        _same_record(checked_record, actual, description=f"truth {style} Arrow {index} descriptor")
        checked.append(checked_record)
    return checked


def _validate_tokenizer_input(
    selection: Mapping[str, Any],
    *,
    root: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    sources = _public_sources(selection)
    frozen = sources["tokenizer"]
    expected_path = _resolve(str(frozen.get("path")), root=root)
    actual_path = Path(tokenizer_path).expanduser().resolve()
    if actual_path != expected_path:
        raise TruthSourceCompatibilityError("truth tokenizer directory differs from frozen selection")
    if actual_path.is_symlink() or not actual_path.is_dir():
        raise TruthSourceCompatibilityError(f"truth tokenizer snapshot directory is unavailable: {actual_path}")
    try:
        actual = trusted._tokenizer_descriptor(actual_path)
    except Exception as exc:
        raise TruthSourceCompatibilityError("truth tokenizer descriptor is unavailable") from exc
    if dict(actual) != dict(frozen):
        raise TruthSourceCompatibilityError("truth tokenizer directory/components differ from frozen selection")
    frozen_files = frozen.get("files")
    actual_files = actual.get("files")
    if not isinstance(frozen_files, Mapping) or not isinstance(actual_files, Mapping) or set(frozen_files) != set(actual_files):
        raise TruthSourceCompatibilityError("truth tokenizer component set differs from frozen selection")
    checked: dict[str, Any] = {}
    for name in sorted(frozen_files):
        frozen_file = frozen_files[name]
        actual_file = actual_files[name]
        if not isinstance(frozen_file, Mapping) or not isinstance(actual_file, Mapping):
            raise TruthSourceCompatibilityError(f"truth tokenizer component is malformed: {name}")
        component = Path(str(actual_file["path"])).expanduser().resolve()
        checked_record = _record(component, root=root, description=f"truth tokenizer component {name}")
        _same_record(checked_record, frozen_file, description=f"truth tokenizer component {name}")
        _same_record(checked_record, actual_file, description=f"truth tokenizer component {name} descriptor")
        snapshot = Path(str(actual_file.get("snapshot_path", actual_path / name))).expanduser()
        if not snapshot.is_file() or snapshot.resolve() != component:
            raise TruthSourceCompatibilityError(f"truth tokenizer snapshot entry changed: {name}")
        if bool(actual_file.get("symlink")) != snapshot.is_symlink():
            raise TruthSourceCompatibilityError(f"truth tokenizer snapshot symlink flag changed: {name}")
        checked[name] = dict(checked_record) | {
            "snapshot_path": str(snapshot),
            "symlink": snapshot.is_symlink(),
        }
    return {"path": str(actual_path), "files": checked}


def _adapter_source_record(*, root: Path) -> dict[str, Any]:
    return _record(ADAPTER_PATH, root=root, description="truth source compatibility adapter")


def source_evidence_compat(
    selection: Mapping[str, Any],
    *,
    root: Path,
    tokenizer_path: Path,
    pile_paths: Sequence[Path],
    finance_paths: Sequence[Path],
) -> dict[str, Any]:
    """Return stable, hash-bound evidence for the frozen public inputs."""
    root = _root(root)
    tokenizer = _validate_tokenizer_input(selection, root=root, tokenizer_path=tokenizer_path)
    pile = _validate_arrow_inputs(selection, root=root, style="pile", paths=tuple(pile_paths))
    finance = _validate_arrow_inputs(selection, root=root, style="finance", paths=tuple(finance_paths))
    counts = selection.get("records_by_domain")
    if not isinstance(counts, Mapping):
        raise TruthSourceCompatibilityError("selection record counts are absent")
    counts_checked = {style: int(counts[style]) for style in contract.DOMAIN_ORDER}
    return {
        "schema": SOURCE_EVIDENCE_SCHEMA,
        "adapter_source": _adapter_source_record(root=root),
        "tokenizer": tokenizer,
        "pile_arrow": pile,
        "finance_arrow": finance,
        "records_by_domain": counts_checked,
    }


@contextmanager
def _patched_source_evidence() -> Iterator[None]:
    previous = truth._source_evidence
    truth._source_evidence = source_evidence_compat  # type: ignore[assignment]
    try:
        yield
    finally:
        truth._source_evidence = previous  # type: ignore[assignment]


def _save_truth_tensors_alias_safe(
    tensors: Mapping[str, torch.Tensor],
    filename: str | Path,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Clone the paired tensors before delegating to the trusted writer.

    ``_materialize_truth`` intentionally reuses each domain tensor for its two
    target cells.  ``safetensors.save_file`` rejects that shared storage even
    though the values and geometry are valid.  Cloning here changes neither;
    it only gives the writer four independent storage regions.
    """
    if not isinstance(tensors, Mapping) or set(tensors) != set(truth.TRUTH_SIDECAR_KEYS):
        raise TruthSourceCompatibilityError("truth tensor matrix is incomplete before serialization")
    independent: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TruthSourceCompatibilityError(f"truth tensor is not a tensor: {key}")
        independent[str(key)] = value.clone()
    _safetensors_save_file(independent, str(filename), metadata=dict(metadata) if metadata is not None else None)


@contextmanager
def _patched_truth_serialization() -> Iterator[None]:
    """Scope the paired-tensor compatibility write to truth preparation only."""
    previous = truth.save_file
    truth.save_file = _save_truth_tensors_alias_safe  # type: ignore[assignment]
    try:
        yield
    finally:
        truth.save_file = previous  # type: ignore[assignment]


def prepare_truth_compat(**kwargs: Any) -> dict[str, Any]:
    """Prepare truth through the reviewed gate adapter and source adapter."""
    with _patched_source_evidence(), _patched_truth_serialization():
        return gate_compat.prepare_truth_compat(**kwargs)


def score_truth_sidecar_compat(**kwargs: Any) -> dict[str, Any]:
    """Score only after both reviewed public and source-evidence revalidation."""
    with _patched_source_evidence():
        return gate_compat.score_truth_sidecar_compat(**kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-truth")
    prepare.add_argument("--freeze", type=Path, required=True)
    prepare.add_argument("--selection", type=Path, required=True)
    prepare.add_argument("--truth-output", type=Path, required=True)
    prepare.add_argument("--truth-binding", type=Path, required=True)
    prepare.add_argument("--execute", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("--freeze", type=Path, required=True)
    score.add_argument("--truth-binding", type=Path, required=True)
    score.add_argument("--truth-sidecar", type=Path)
    score.add_argument("--frequency-reference", type=Path)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--bootstrap-seed", type=int, default=9009)
    score.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-truth":
            result = prepare_truth_compat(
                freeze_path=args.freeze,
                registration_path=None,
                selection_path=args.selection,
                truth_output=args.truth_output,
                truth_binding_path=args.truth_binding,
                repository_root=args.repository_root,
                execute=args.execute,
            )
        else:
            result = score_truth_sidecar_compat(
                freeze_path=args.freeze,
                truth_binding_path=args.truth_binding,
                truth_sidecar_path=args.truth_sidecar,
                frequency_reference_path=args.frequency_reference,
                output_path=args.output,
                repository_root=args.repository_root,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_draws=args.bootstrap_draws,
            )
    except Exception as exc:
        print(f"TRR-0009 truth source compatibility boundary failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "truth_opened": result.get("truth_opened"), "artifact": result.get("truth_binding", result.get("score_artifact"))}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
