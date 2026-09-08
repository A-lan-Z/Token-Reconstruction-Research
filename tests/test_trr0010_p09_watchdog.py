from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_watchdog_enforces_output_bytes_cap_before_success(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "watchdog-run"
    child = [
        sys.executable,
        "-c",
        "import sys,time; sys.stdout.write('x' * 100000); sys.stdout.flush(); time.sleep(2)",
    ]
    command = [
        sys.executable,
        str(repository / "scripts/trr0010_p09_watchdog.py"),
        "--output-root",
        str(output),
        "--timeout-seconds",
        "10",
        "--poll-seconds",
        "0.01",
        "--max-rss-bytes",
        str(8 * 2**30),
        "--min-available-bytes",
        "1",
        "--max-output-bytes",
        str(16 * 1024),
        "--cwd",
        str(repository),
        "--label",
        "TRR-0010-test-output-cap",
        "--",
        *child,
    ]
    result = subprocess.run(command, cwd=repository, check=False, capture_output=True, text=True)
    assert result.returncode == 125
    guard = json.loads((output / "resource_guard.json").read_text())
    assert guard["status"] == "FAIL_CLOSED"
    assert guard["termination_reason"] == "output_bytes_limit_exceeded"
    assert guard["max_output_bytes"] == 16 * 1024
    assert guard["peak_output_bytes"] > 16 * 1024
    assert (output / "stdout.txt").stat().st_size >= 100000
    assert (output / "stderr.txt").is_file()
    assert (output / "resource_samples.jsonl").is_file()
