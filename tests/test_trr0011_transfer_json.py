from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from trr0011_transfer import TransferDiagnosticError, _write_transfer_json_create


def test_transfer_json_writer_roundtrips_with_real_newline(tmp_path: Path) -> None:
    root = tmp_path
    path = root / "experiments" / "TRR-0011" / "roundtrip.json"
    value = {"schema": "synthetic", "truth_opened": False, "items": [1, 2, 3]}

    record = _write_transfer_json_create(path, value, root=root)

    assert json.loads(path.read_text(encoding="utf-8")) == value
    assert path.read_bytes().endswith(b"\n")
    assert record["bytes"] == path.stat().st_size
    try:
        _write_transfer_json_create(path, value, root=root)
    except TransferDiagnosticError as exc:
        assert "create-only" in str(exc)
    else:
        raise AssertionError("writer permitted a second create-only write")
