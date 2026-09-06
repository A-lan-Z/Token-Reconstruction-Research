"""Focused watchdog regressions for the post-exit /proc race."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.trr_p06 import resource_watchdog as watchdog


class _Leader:
    pid = 4242

    def __init__(self, return_code: int | None) -> None:
        self.return_code = return_code
        self._polls = 0
        self.wait_calls = 0

    def poll(self):
        self._polls += 1
        # The leader is live at the initial loop check, then exits before the
        # resource sample raises the missing-RSS error.
        return None if self._polls == 1 else self.return_code

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self.return_code


@pytest.mark.parametrize("child_code", [0, 1])
def test_missing_rss_after_leader_exit_preserves_child_status(
    monkeypatch, tmp_path: Path, child_code: int
) -> None:
    leader = _Leader(child_code)
    calls: list[bool] = []

    def fake_sample(_pgid: int, *, require_member: bool, elapsed_seconds: float):
        calls.append(require_member)
        raise watchdog.ResourceReadError("live process RSS is missing: /proc/4242/status")

    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: leader)
    monkeypatch.setattr(watchdog.os, "getpgid", lambda _pid: 4242)
    monkeypatch.setattr(watchdog, "_read_mem_available_bytes", lambda: 20 * 1024**3)
    monkeypatch.setattr(watchdog, "_sample", fake_sample)
    monkeypatch.setattr(watchdog, "_has_live_process_group_member", lambda _pgid: False)

    output = tmp_path / f"watchdog-{child_code}"
    code = watchdog.main(
        [
            "--output-root",
            str(output),
            "--poll-seconds",
            "0.001",
            "--timeout-seconds",
            "5",
            "--",
            "/bin/true",
        ]
    )

    receipt = json.loads((output / "finish.json").read_text(encoding="utf-8"))
    assert calls == [True, False]
    assert leader.wait_calls == 1
    assert receipt["child_return_code"] == child_code
    assert receipt["termination_reason"] is None
    assert receipt["status"] == ("PASS" if child_code == 0 else "CHILD_EXITED_NONZERO")
    assert code == child_code
    guard = json.loads((output / "resource_guard.json").read_text(encoding="utf-8"))
    assert any("post_exit_resource_sample_ignored" in value for value in guard["errors"])


def test_missing_rss_while_leader_is_live_fails_closed(monkeypatch, tmp_path: Path) -> None:
    leader = _Leader(None)

    def fake_sample(_pgid: int, *, require_member: bool, elapsed_seconds: float):
        raise watchdog.ResourceReadError("live process RSS is missing: /proc/4242/status")

    def fake_wait(timeout=None):
        leader.wait_calls += 1
        raise watchdog.subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    leader.wait = fake_wait
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: leader)
    monkeypatch.setattr(watchdog.os, "getpgid", lambda _pid: 4242)
    monkeypatch.setattr(watchdog, "_read_mem_available_bytes", lambda: 20 * 1024**3)
    monkeypatch.setattr(watchdog, "_sample", fake_sample)
    monkeypatch.setattr(watchdog, "_terminate_group", lambda *args, **kwargs: ["SIGTERM_process_group"])

    output = tmp_path / "watchdog-live-failure"
    code = watchdog.main(
        [
            "--output-root",
            str(output),
            "--poll-seconds",
            "0.001",
            "--timeout-seconds",
            "5",
            "--",
            "/bin/true",
        ]
    )

    receipt = json.loads((output / "finish.json").read_text(encoding="utf-8"))
    assert code == watchdog.WRAPPER_FAILURE_EXIT
    assert receipt["status"] == "FAIL_CLOSED"
    assert receipt["termination_reason"] == "live_resource_data_unreadable"
