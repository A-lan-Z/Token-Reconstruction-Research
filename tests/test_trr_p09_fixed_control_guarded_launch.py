from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "trr_p09" / "fixed_control_guarded_launch.py"
WATCHDOG_SOURCE = ROOT / "scripts" / "trr_p09" / "resource_watchdog.py"


def _invoke(tmp_path: Path, child_source: str, *, max_output_bytes: int) -> tuple[subprocess.CompletedProcess[str], Path]:
    child_script = tmp_path / "child.py"
    child_script.write_text(child_source, encoding="utf-8")
    guard_root = tmp_path / "guard"
    child_root = tmp_path / "child-output"
    command = [
        sys.executable,
        str(LAUNCHER),
        "--output-root",
        str(guard_root),
        "--child-output-root",
        str(child_root),
        "--timeout-seconds",
        "5",
        "--poll-seconds",
        "0.01",
        "--max-rss-bytes",
        str(1024**3),
        "--min-available-bytes",
        "1",
        "--min-disk-free-bytes",
        "1",
        "--max-output-bytes",
        str(max_output_bytes),
        "--kill-grace-seconds",
        "0.2",
        "--cwd",
        str(ROOT),
        "--watchdog-source",
        str(WATCHDOG_SOURCE),
        "--label",
        "synthetic-fixed-control",
        "--",
        sys.executable,
        str(child_script),
        str(child_root),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=15)
    return result, guard_root


def test_guarded_launcher_normal_exit_records_disk_and_output_samples(tmp_path: Path) -> None:
    result, guard_root = _invoke(
        tmp_path,
        """
from pathlib import Path
import sys
import time
root = Path(sys.argv[1])
root.mkdir(parents=True)
(root / 'result.bin').write_bytes(b'ok')
time.sleep(0.08)
""",
        max_output_bytes=1_000_000,
    )
    assert result.returncode == 0, result.stderr
    guard = json.loads((guard_root / "resource_guard.json").read_text(encoding="utf-8"))
    finish = json.loads((guard_root / "finish.json").read_text(encoding="utf-8"))
    assert guard["status"] == "PASS"
    assert guard["termination_reason"] is None
    assert guard["minimum_sampled_disk_free_bytes"] > 0
    assert guard["peak_sampled_output_bytes"] >= 2
    assert guard["disk_and_output_enforced_during_live_process"] is True
    assert finish["status"] == "PASS"


def test_guarded_launcher_kills_process_group_on_output_breach(tmp_path: Path) -> None:
    result, guard_root = _invoke(
        tmp_path,
        """
from pathlib import Path
import sys
import time
root = Path(sys.argv[1])
root.mkdir(parents=True)
(root / 'too-large.bin').write_bytes(b'x' * 8192)
time.sleep(30)
""",
        max_output_bytes=1024,
    )
    assert result.returncode == 125, result.stderr
    guard = json.loads((guard_root / "resource_guard.json").read_text(encoding="utf-8"))
    time_receipt = json.loads((guard_root / "time.json").read_text(encoding="utf-8"))
    assert guard["status"] == "FAIL_CLOSED"
    assert guard["termination_reason"] == "output_bytes_limit_exceeded"
    assert guard["peak_sampled_output_bytes"] > 1024
    assert guard["termination_actions"]
    assert time_receipt["wrapper_exit_code"] == 125
