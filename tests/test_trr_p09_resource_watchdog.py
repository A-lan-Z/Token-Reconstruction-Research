"""Focused P09 watchdog regressions for the post-exit /proc race."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from scripts.trr_p09 import resource_watchdog as watchdog


class _TransitionLeader:
    pid = 5252

    def __init__(self) -> None:
        self.exited = False
        self.wait_calls = 0

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise watchdog.subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        self.exited = True
        return 0


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


def test_missing_rss_after_reaped_leader_is_ignored_but_child_status_preserved(
    monkeypatch, tmp_path: Path,
) -> None:
    leader = _Leader(0)
    calls: list[bool] = []
    ignored: list[set[int]] = []

    def fake_sample(_pgid: int, *, require_member: bool, elapsed_seconds: float):
        calls.append(require_member)
        raise watchdog.ResourceReadError("live process RSS is missing: /proc/4242/status")

    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: leader)
    monkeypatch.setattr(watchdog.os, "getpgid", lambda _pid: 4242)
    monkeypatch.setattr(watchdog, "_read_mem_available_bytes", lambda: 20 * 1024**3)
    monkeypatch.setattr(watchdog, "_sample", fake_sample)
    monkeypatch.setattr(
        watchdog,
        "_has_live_process_group_member",
        lambda _pgid, *, ignored_pids: ignored.append(set(ignored_pids)) or False,
    )

    output = tmp_path / "watchdog-race"
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
    assert ignored == [{4242}]
    assert leader.wait_calls == 1
    assert receipt["child_return_code"] == 0
    assert receipt["termination_reason"] is None
    assert receipt["status"] == "PASS"
    assert code == 0
    guard = json.loads((output / "resource_guard.json").read_text(encoding="utf-8"))
    assert any("post_exit_resource_sample_ignored" in value for value in guard["errors"])


def test_missing_rss_during_exit_transition_rechecks_until_reaped(monkeypatch, tmp_path: Path) -> None:
    leader = _TransitionLeader()
    calls: list[bool] = []
    ignored: list[set[int]] = []

    def fake_sample(_pgid: int, *, require_member: bool, elapsed_seconds: float):
        calls.append(require_member)
        raise watchdog.ResourceReadError("live process RSS is missing: /proc/5252/status")

    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: leader)
    monkeypatch.setattr(watchdog.os, "getpgid", lambda _pid: 5252)
    monkeypatch.setattr(watchdog, "_read_mem_available_bytes", lambda: 20 * 1024**3)
    monkeypatch.setattr(watchdog, "_sample", fake_sample)
    monkeypatch.setattr(
        watchdog,
        "_has_live_process_group_member",
        lambda _pgid, *, ignored_pids: ignored.append(set(ignored_pids)) or False,
    )

    output = tmp_path / "watchdog-transition"
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
    assert code == 0
    assert receipt["status"] == "PASS"
    assert receipt["child_return_code"] == 0
    assert calls == [True, False]
    assert ignored == [{5252}]
    assert leader.wait_calls >= 2


def test_real_rapid_child_exit_repeated(tmp_path: Path) -> None:
    for index in range(12):
        output = tmp_path / f"watchdog-real-{index}"
        code = watchdog.main(
            [
                "--output-root",
                str(output),
                "--poll-seconds",
                "0.001",
                "--timeout-seconds",
                "5",
                "--min-available-bytes",
                "1",
                "--",
                sys.executable,
                "-c",
                "pass",
            ]
        )
        receipt = json.loads((output / "finish.json").read_text(encoding="utf-8"))
        assert code == 0
        assert receipt["status"] == "PASS"
        assert receipt["child_return_code"] == 0


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
    guard = json.loads((output / "resource_guard.json").read_text(encoding="utf-8"))
    assert any("leader_state_before_recheck=" in value for value in guard["errors"])
