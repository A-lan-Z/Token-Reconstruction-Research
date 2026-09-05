#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


argv_path = Path(sys.argv[1]).resolve()
data = json.loads(argv_path.read_text(encoding='utf-8'))
order = list(data['launcher_contract']['invocation_order'])
entries = {entry['method_id']: entry for entry in data['methods']}
assert order == [entry['method_id'] for entry in (entries[mid] for mid in order)]
assert len(order) == 5 and len(entries) == 5
for method_id in order:
    entry = entries[method_id]
    command = list(entry['command'])
    output_root = Path(entry['output_root_absolute'])
    assert not output_root.exists() and not output_root.is_symlink(), output_root
    print(json.dumps({'event': 'launching', 'method': method_id, 'utc': utc_now(), 'cwd': entry['working_directory'], 'argv': command}, sort_keys=True), flush=True)
    process = subprocess.Popen(command, cwd=entry['working_directory'])
    print(json.dumps({'event': 'pid', 'method': method_id, 'pid': process.pid, 'utc': utc_now()}, sort_keys=True), flush=True)
    returncode = process.wait()
    print(json.dumps({'event': 'completed', 'method': method_id, 'pid': process.pid, 'returncode': returncode, 'utc': utc_now()}, sort_keys=True), flush=True)
    if returncode != 0:
        raise SystemExit(returncode if returncode > 0 else 1)
print(json.dumps({'event': 'all_completed', 'methods': order, 'utc': utc_now()}, sort_keys=True), flush=True)
