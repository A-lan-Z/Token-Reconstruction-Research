from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trr0000_smoke.py"


def run_smoke(seed: str, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            seed,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_output_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    first_run = run_smoke("1729", first)
    second_run = run_smoke("1729", second)

    assert first_run.returncode == 0, first_run.stderr
    assert second_run.returncode == 0, second_run.stderr
    assert first.read_bytes() == second.read_bytes()

    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["format_version"] == "trr.bootstrap-smoke.v1"
    assert payload["algorithm"] == "splitmix64-v1"
    assert payload["seed"] == 1729
    assert payload["sample_count"] == 8
    assert [sample["index"] for sample in payload["samples"]] == list(range(8))
    assert all(
        sample["hex"] == f"0x{sample['uint64']:016x}"
        for sample in payload["samples"]
    )


def test_invalid_seed_exits_nonzero_without_output(tmp_path: Path) -> None:
    output = tmp_path / "invalid.json"

    completed = run_smoke("-1", output)

    assert completed.returncode != 0
    assert "seed must be between" in completed.stderr
    assert not output.exists()


def test_write_failure_exits_nonzero(tmp_path: Path) -> None:
    completed = run_smoke("1729", tmp_path)

    assert completed.returncode != 0
    assert "unable to write" in completed.stderr


def test_missing_required_arguments_exits_nonzero(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--seed" in completed.stderr
    assert "--output" in completed.stderr
