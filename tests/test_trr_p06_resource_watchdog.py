"""Focused watchdog regression tests for the post-exit /proc race."""

from __future__ import annotations

from pathlib import Path

from scripts.trr_p06 import resource_watchdog as watchdog


class _ExitedLeader:
    pid = 4242

    def __init__(self) -> None:
        self._polls = 0
        self.wait_calls = 0

    def poll(self):
        self._polls += 1
        # The leader is live at the initial loop check, then exits before the
        # resource sample raises the missing-RSS error.
        return None if self._polls == 1 else 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        return 0


def test_missing_rss_after_leader_exit_is_rechecked_and_passes(monkeypatch, tmp_path: Path) -> None:
    leader = _ExitedLeader()
    calls: list[bool] = []

    def fake_sample(_pgid: int, *, require_member: bool, elapsed_seconds: float):
        calls.append(require_member)
        if len(calls) == 1:
            raise watchdog.ResourceReadError("live process RSS is missing: /proc/4242/status")
        return {
            "timestamp_utc": "2026-09-06T00:00:00Z",
            "elapsed_seconds": elapsed_seconds,
            "host_mem_available_bytes": 20 * 1024**3,
            "group_rss_bytes": 0,
            "group_pids": [],
            "group_member_rss_bytes": {},
        }

    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *args, **kwargs: leader)
    monkeypatch.setattr(watchdog.os, "getpgid", lambda _pid: 4242)
    monkeypatch.setattr(watchdog, "_read_mem_available_bytes", lambda: 20 * 1024**3)
    monkeypatch.setattr(watchdog, "_sample", fake_sample)

    code = watchdog.main(
        [
            "--output-root",
            str(tmp_path / "watchdog"),
            "--poll-seconds",
            "0.001",
            "--timeout-seconds",
            "5",
            "--",
            "/bin/true",
        ]
    )

    assert code == 0
    assert calls == [True, False]
    assert leader.wait_calls == 1
    assert (tmp_path / "watchdog" / "finish.json").exists()
