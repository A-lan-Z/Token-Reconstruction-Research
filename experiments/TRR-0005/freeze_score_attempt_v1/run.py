from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

REPO = Path('/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005').resolve()
OUT = Path('/tmp/trr5/trr0005_freeze_score_v1').resolve()
FRESH = REPO / 'experiments/TRR-0005/fresh_confirmation_v1'
PRED_ROOT = FRESH / 'predictions_v1'
PANEL = FRESH / 'panel_capture_v2/panel.json'
OBSERVATIONS = FRESH / 'panel_capture_v2/observations.json'
REGISTRATION = FRESH / 'method_registration.json'
PLAN = FRESH / 'selection_plan.json'
PREDICTIONS = PRED_ROOT / 'predictions.json'
TIMINGS = PRED_ROOT / 'timings.json'
FREEZE_RECEIPT = FRESH / 'freeze_receipt.json'
TRUTH_MANIFEST = Path('/tmp/trr5/fresh_confirmation_v1.truth.manifest.json')
TRUTH = Path('/tmp/trr5/fresh_confirmation_v1.truth.safetensors')
BINDING = PRED_ROOT / 'evaluator_binding.json'
FREQUENCIES = REPO / 'experiments/TRR-0005/frequency_references_v1.json'
RESULT = FRESH / 'result.json'

ENV_OVERRIDES = {
    'PYTHONPATH': '.:src:scripts',
    'OMP_NUM_THREADS': '8',
    'MKL_NUM_THREADS': '8',
    'OPENBLAS_NUM_THREADS': '8',
}
ENV = os.environ.copy()
ENV.update(ENV_OVERRIDES)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head() -> str | None:
    proc = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=REPO,
        env=ENV,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def resource_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        'captured_utc': utc_now(),
        'child_rusage': {
            'user_seconds': resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
            'system_seconds': resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            'max_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        },
    }
    try:
        values: dict[str, int] = {}
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].rstrip(':') in {'MemTotal', 'MemAvailable', 'MemFree'}:
                values[fields[0].rstrip(':')] = int(fields[1]) * 1024
        result['host_memory_bytes'] = values
    except (OSError, UnicodeError, ValueError):
        result['host_memory_bytes'] = None
    try:
        gpu = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            cwd=REPO,
            env=ENV,
            text=True,
            capture_output=True,
            check=False,
        )
        result['nvidia_smi'] = {
            'returncode': gpu.returncode,
            'stdout': gpu.stdout.strip(),
            'stderr': gpu.stderr.strip(),
        }
    except OSError as exc:
        result['nvidia_smi'] = {'error': repr(exc)}
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def run_phase(name: str, argv: list[str]) -> dict[str, Any]:
    stdout_path = OUT / f'{name}.stdout.log'
    stderr_path = OUT / f'{name}.stderr.log'
    resource_path = OUT / f'{name}.resource.json'
    started = utc_now()
    before = resource_snapshot()
    with stdout_path.open('w', encoding='utf-8') as stdout, stderr_path.open('w', encoding='utf-8') as stderr:
        proc = subprocess.run(argv, cwd=REPO, env=ENV, stdout=stdout, stderr=stderr, check=False)
    ended = utc_now()
    after = resource_snapshot()
    record = {
        'name': name,
        'argv': argv,
        'cwd': str(REPO),
        'environment_overrides': dict(ENV_OVERRIDES),
        'started_utc': started,
        'ended_utc': ended,
        'returncode': proc.returncode,
        'stdout_path': str(stdout_path),
        'stderr_path': str(stderr_path),
        'resource_snapshot_before': before,
        'resource_snapshot_after': after,
    }
    write_json(resource_path, record)
    record['resource_record_path'] = str(resource_path)
    write_json(OUT / f'{name}.record.json', record)
    return record


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    head_before = git_head()
    execution: dict[str, Any] = {
        'schema': 'token-reconstruction.trr0005-freeze-score-execution.v1',
        'task_id': 'TRR-0005',
        'status': 'RUNNING',
        'repository_root': str(REPO),
        'head_before': head_before,
        'started_utc': utc_now(),
        'truth_path_passed_to_scorer': str(TRUTH),
        'truth_manifest_source': str(TRUTH_MANIFEST),
        'frozen_prediction_root': str(PRED_ROOT),
        'freeze_receipt': str(FREEZE_RECEIPT),
        'phases': [],
    }
    write_json(OUT / 'execution_receipt.json', execution)

    if BINDING.exists() or BINDING.is_symlink():
        execution['status'] = 'ABORTED_BINDING_ALREADY_EXISTS'
        execution['ended_utc'] = utc_now()
        execution['head_after'] = git_head()
        write_json(OUT / 'execution_receipt.json', execution)
        return 2
    if not TRUTH_MANIFEST.is_file() or TRUTH_MANIFEST.is_symlink():
        execution['status'] = 'ABORTED_LABEL_FREE_BINDING_UNAVAILABLE'
        execution['ended_utc'] = utc_now()
        execution['head_after'] = git_head()
        write_json(OUT / 'execution_receipt.json', execution)
        return 3

    copy_test = run_phase('binding_absence_check', ['test', '!', '-e', str(BINDING)])
    execution['phases'].append(copy_test)
    if copy_test['returncode'] != 0:
        execution['status'] = 'ABORTED_BINDING_ABSENCE_CHECK'
        execution['ended_utc'] = utc_now()
        execution['head_after'] = git_head()
        write_json(OUT / 'execution_receipt.json', execution)
        return copy_test['returncode'] or 4
    copy_phase = run_phase('copy_label_free_binding', ['cp', '--', str(TRUTH_MANIFEST), str(BINDING)])
    execution['phases'].append(copy_phase)
    if copy_phase['returncode'] != 0:
        execution['status'] = 'ABORTED_BINDING_COPY'
        execution['ended_utc'] = utc_now()
        execution['head_after'] = git_head()
        write_json(OUT / 'execution_receipt.json', execution)
        return copy_phase['returncode'] or 5

    freeze_argv = [
        str(REPO / '.venv-trr0005/bin/python'),
        str(REPO / 'scripts/trr0005_freeze_confirmation.py'),
        '--repository-root', str(REPO),
        '--panel', str(PANEL),
        '--registration', str(REGISTRATION),
        '--output-root', str(PRED_ROOT),
        '--plan', str(PLAN),
        '--receipt', str(FREEZE_RECEIPT),
    ]
    freeze = run_phase('freeze_public_matrix', freeze_argv)
    execution['phases'].append(freeze)
    write_json(OUT / 'execution_receipt.json', execution)
    if freeze['returncode'] != 0:
        execution['status'] = 'FAILED_FREEZE_STOPPED_BEFORE_SCORE'
        execution['ended_utc'] = utc_now()
        execution['head_after'] = git_head()
        write_json(OUT / 'execution_receipt.json', execution)
        return freeze['returncode'] or 6

    score_argv = [
        str(REPO / '.venv-trr0005/bin/python'),
        str(REPO / 'scripts/trr0005_score_confirmation.py'),
        '--repository-root', str(REPO),
        '--panel', str(PANEL),
        '--registration', str(REGISTRATION),
        '--selection-plan', str(PLAN),
        '--observations', str(OBSERVATIONS),
        '--predictions', str(PREDICTIONS),
        '--timings', str(TIMINGS),
        '--receipt', str(FREEZE_RECEIPT),
        '--truth', str(TRUTH),
        '--truth-binding', str(BINDING),
        '--output-root', str(PRED_ROOT),
        '--result', str(RESULT),
        '--frequency-manifest', str(FREQUENCIES),
        '--bootstrap-draws', '10000',
        '--bootstrap-seed', '5005',
    ]
    score = run_phase('score_truth_gated_matrix', score_argv)
    execution['phases'].append(score)
    execution['ended_utc'] = utc_now()
    execution['head_after'] = git_head()
    execution['status'] = 'COMPLETE' if score['returncode'] == 0 else 'FAILED_SCORE'
    execution['head_unchanged'] = execution['head_before'] == execution['head_after']
    write_json(OUT / 'execution_receipt.json', execution)
    return score['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
