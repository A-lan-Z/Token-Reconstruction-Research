#!/usr/bin/env python3
"""External read-only resource guard attached to an already running P04 fit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

PID = 85284
OUTPUT = Path('/tmp/trr-p04/experiments/TRR-P04/runtime/training-r1')
EXPECTED = '/tmp/trr-p04/scripts/trr0004_p04_train.py'
MIN_FREE = 8 * 1024**3
MAX_RSS = 16 * 1024**3
MAX_TEMP_C = 90.0
INTERVAL = 5.0


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def cmdline(pid: int) -> str:
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()


def rss(pid: int) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            values[key] = int(value.strip().split()[0]) * 1024
    return values.get('VmRSS', 0), values.get('VmHWM', 0)


def gpu() -> dict[str, float | int]:
    query = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free,temperature.gpu', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True,
    )
    line = query.stdout.strip().splitlines()[0]
    index, total, used, free, temp = [part.strip() for part in line.split(',')]
    return {
        'index': int(index),
        'total_bytes': int(float(total) * 1024**2),
        'used_bytes': int(float(used) * 1024**2),
        'free_bytes': int(float(free) * 1024**2),
        'temperature_c': float(temp),
    }


def stop(reason: str) -> None:
    try:
        os.kill(PID, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not Path(f'/proc/{PID}').exists():
            return
        time.sleep(0.25)
    try:
        os.kill(PID, signal.SIGKILL)
    except ProcessLookupError:
        pass


OUTPUT.mkdir(parents=True, exist_ok=True)
log = OUTPUT / 'external_resource_watchdog.jsonl'
receipt = OUTPUT / 'external_resource_watchdog_receipt.json'
if receipt.exists() or receipt.is_symlink():
    raise SystemExit(f'create-only receipt exists: {receipt}')
if not Path(f'/proc/{PID}').is_dir():
    raise SystemExit(f'target PID is not live: {PID}')
command = cmdline(PID)
if EXPECTED not in command:
    raise SystemExit(f'target PID command mismatch: {command}')
command_sha = hashlib.sha256(command.encode()).hexdigest()
started = utc()
started_mono = time.monotonic()
status = 'PASS'
stop_reason = None
samples = 0
max_rss = 0
max_used = 0
min_free = None
max_temp = 0.0
errors = []
with log.open('w', encoding='utf-8') as stream:
    stream.write(json.dumps({'event': 'start', 'utc': started, 'pid': PID, 'command': command, 'command_sha256': command_sha, 'thresholds': {'minimum_free_gpu_bytes': MIN_FREE, 'maximum_host_rss_bytes': MAX_RSS, 'maximum_temperature_c': MAX_TEMP_C, 'interval_seconds': INTERVAL}}) + '\n')
    stream.flush()
    while Path(f'/proc/{PID}').is_dir():
        try:
            current_rss, hwm = rss(PID)
            current_gpu = gpu()
            max_rss = max(max_rss, hwm, current_rss)
            max_used = max(max_used, int(current_gpu['used_bytes']))
            free = int(current_gpu['free_bytes'])
            min_free = free if min_free is None else min(min_free, free)
            max_temp = max(max_temp, float(current_gpu['temperature_c']))
            sample = {'event': 'sample', 'utc': utc(), 'pid': PID, 'rss_bytes': current_rss, 'hwm_bytes': hwm, 'gpu': current_gpu}
            stream.write(json.dumps(sample) + '\n')
            stream.flush()
            samples += 1
            if free < MIN_FREE:
                status, stop_reason = 'FAIL_CLOSED', f'GPU free below {MIN_FREE}: {free}'
                stop(stop_reason)
                break
            if max_rss > MAX_RSS:
                status, stop_reason = 'FAIL_CLOSED', f'host RSS above {MAX_RSS}: {max_rss}'
                stop(stop_reason)
                break
            if float(current_gpu['temperature_c']) >= MAX_TEMP_C:
                status, stop_reason = 'FAIL_CLOSED', f'GPU temperature at or above {MAX_TEMP_C}: {current_gpu["temperature_c"]}'
                stop(stop_reason)
                break
        except Exception as exc:
            status, stop_reason = 'FAIL_CLOSED', f'resource sample failed: {exc}'
            errors.append(str(exc))
            stop(stop_reason)
            break
        time.sleep(INTERVAL)
    ended = utc()
    stream.write(json.dumps({'event': 'end', 'utc': ended, 'status': status, 'stop_reason': stop_reason}) + '\n')
    stream.flush()
if stop_reason is None and status == 'PASS':
    stop_reason = 'target process exited; no guard violation observed'
receipt_payload = {
    'task_id': 'TRR-P04',
    'schema': 'token-reconstruction.trr-p04-training-external-watchdog.v1',
    'status': status,
    'target_pid': PID,
    'target_command': command,
    'target_command_sha256': command_sha,
    'sampling': {'samples': samples, 'interval_seconds': INTERVAL, 'log_path': str(log), 'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()},
    'thresholds': {'minimum_free_gpu_bytes': MIN_FREE, 'maximum_host_rss_bytes': MAX_RSS, 'maximum_temperature_c': MAX_TEMP_C},
    'observed': {'minimum_external_free_gpu_bytes': min_free, 'maximum_external_device_used_bytes': max_used, 'maximum_host_rss_bytes': max_rss, 'maximum_temperature_c': max_temp},
    'measurement_note': 'nvidia-smi device-wide memory used/free is an external whole-device measure; it is not PyTorch max_memory_reserved. The qualified teacher allocator limit of 6 GiB is retained in teacher receipts and is not inferred here.',
    'started_utc': started,
    'ended_utc': ended,
    'wall_seconds': time.monotonic() - started_mono,
    'stop_reason': stop_reason,
    'errors': errors,
}
receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps({'status': status, 'receipt': str(receipt), 'samples': samples, 'stop_reason': stop_reason}, sort_keys=True))
if status != 'PASS':
    raise SystemExit(2)
