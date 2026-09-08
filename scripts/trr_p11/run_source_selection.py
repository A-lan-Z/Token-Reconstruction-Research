"""CLI wrapper for the gated P11 trusted-curator source selector.

The command is intentionally explicit: it can create a selection only after
the manifest binds the B0/B1 banks and the exclusion audit is complete and
released.  No source scan is performed by ``--help`` or import.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.trr_p11.source_selector import select_sources


def _resolve_path(value: Path, *, root: Path, description: str, require_exists: bool) -> Path:
    """Resolve command paths relative to the declared P11 checkout.

    The selector command is recorded for execution from the P11 worktree, but
    callers may launch it from any current directory. Resolving inputs from
    ``--repository-root`` keeps the sibling TRR-0010 dependency unambiguous
    and prevents an accidental path lookup in the main checkout.
    """
    raw = Path(value).expanduser()
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    if require_exists and not resolved.exists():
        raise SystemExit(f"{description} is unavailable: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pr20-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.expanduser().resolve()
    manifest = _resolve_path(args.manifest, root=repository_root, description="manifest", require_exists=True)
    audit = _resolve_path(args.audit, root=repository_root, description="exclusion audit", require_exists=True)
    source_inputs = _resolve_path(args.source_inputs, root=repository_root, description="source inputs", require_exists=True)
    output = _resolve_path(args.output, root=repository_root, description="selection output", require_exists=False)
    pr20_root = None
    if args.pr20_root is not None:
        pr20_root = _resolve_path(args.pr20_root, root=repository_root, description="TRR-0010 dependency root", require_exists=True)
        if not pr20_root.is_dir():
            raise SystemExit(f"TRR-0010 dependency root is not a directory: {pr20_root}")
    result = select_sources(
        manifest_path=manifest,
        audit_path=audit,
        source_inputs=source_inputs,
        output_path=output,
        repository_root=repository_root,
        pr20_root=pr20_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
