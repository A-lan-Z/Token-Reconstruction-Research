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
    result = select_sources(
        manifest_path=args.manifest,
        audit_path=args.audit,
        source_inputs=args.source_inputs,
        output_path=args.output,
        repository_root=args.repository_root,
        pr20_root=args.pr20_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
