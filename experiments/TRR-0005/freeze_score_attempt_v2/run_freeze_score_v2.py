from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
from datetime import datetime, timezone
from typing import Any

REPO = Path('/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0005').resolve()
OUT = REPO / 'experiments/TRR-0005/freeze_score_attempt_v2'
FRESH = REPO / 'experiments/TRR-0005/fresh_confirmation_v1'
PRED_ROOT = FRESH / 'predictions_v2_contract_export'
PANEL = FRESH / 'panel_capture_v2/panel.json'
OBSERVATIONS = FRESH / 'panel_capture_v2/observations.json'
REGISTRATION = FRESH / 'method_registration.json'
PLAN = FRESH / 'selection_plan.json'
PREDICTIONS = PRED_ROOT / 'predictions.json'
TIMINGS = PRED_ROOT / 'timings.json'
FREEZE_RECEIPT = FRESH / 'freeze_receipt_v2.json'
TRUTH_MANIFEST_V1 = FRESH / 'predictions_v1/evaluator_binding.json'
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
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    result: dict[str, Any] = {
        'captured_utc': utc_now(),
        'child_rusage': {
            'user_seconds': children.ru_utime,
            'system_seconds': children.ru_stime,
            'max_rss_kib': children.ru_maxrss,
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )


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


def persist(execution: dict[str, Any]) -> None:
    write_json(OUT / 'execution_receipt.json', execution)


def abort(execution: dict[str, Any], status: str, code: int) -> int:
    execution['status'] = status
    execution['ended_utc'] = utc_now()
    execution['head_after'] = git_head()
    execution['head_unchanged'] = execution['head_before'] == execution['head_after']
    persist(execution)
    return code


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    execution: dict[str, Any] = {
        'schema': 'token-reconstruction.trr0005-freeze-score-execution.v2',
        'task_id': 'TRR-0005',
        'status': 'RUNNING',
        'repository_root': str(REPO),
        'head_before': git_head(),
        'started_utc': utc_now(),
        'frozen_prediction_root': str(PRED_ROOT),
        'freeze_receipt': str(FREEZE_RECEIPT),
        'truth_manifest_source': str(TRUTH_MANIFEST_V1),
        'truth_path_passed_to_scorer': str(TRUTH),
        'phases': [],
    }
    persist(execution)

    if not PRED_ROOT.is_dir() or PRED_ROOT.is_symlink():
        return abort(execution, 'ABORTED_V2_ROOT_UNAVAILABLE', 2)
    if BINDING.exists() or BINDING.is_symlink():
        return abort(execution, 'ABORTED_V2_BINDING_ALREADY_EXISTS', 3)
    if not TRUTH_MANIFEST_V1.is_file() or TRUTH_MANIFEST_V1.is_symlink():
        return abort(execution, 'ABORTED_V1_LABEL_FREE_BINDING_UNAVAILABLE', 4)
    if FREEZE_RECEIPT.exists() or FREEZE_RECEIPT.is_symlink():
        return abort(execution, 'ABORTED_V2_FREEZE_RECEIPT_ALREADY_EXISTS', 5)
    if RESULT.exists() or RESULT.is_symlink():
        return abort(execution, 'ABORTED_RESULT_ALREADY_EXISTS', 6)

    copy_test = run_phase('v2_binding_absence_check', ['test', '!', '-e', str(BINDING)])
    execution['phases'].append(copy_test)
    if copy_test['returncode'] != 0:
        return abort(execution, 'ABORTED_BINDING_ABSENCE_CHECK', copy_test['returncode'] or 7)
    copy_phase = run_phase('v2_copy_label_free_binding', ['cp', '--', str(TRUTH_MANIFEST_V1), str(BINDING)])
    execution['phases'].append(copy_phase)
    if copy_phase['returncode'] != 0:
        return abort(execution, 'ABORTED_BINDING_COPY', copy_phase['returncode'] or 8)

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
    freeze = run_phase('v2_freeze_public_matrix', freeze_argv)
    execution['phases'].append(freeze)
    persist(execution)
    if freeze['returncode'] != 0:
        return abort(execution, 'FAILED_V2_FREEZE_STOPPED_BEFORE_SCORE', freeze['returncode'] or 9)

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
    score = run_phase('v2_score_truth_gated_matrix', score_argv)
    execution['phases'].append(score)
    execution['ended_utc'] = utc_now()
    execution['head_after'] = git_head()
    execution['head_unchanged'] = execution['head_before'] == execution['head_after']
    execution['status'] = 'COMPLETE' if score['returncode'] == 0 else 'FAILED_V2_SCORE'
    persist(execution)
    return score['returncode']


if __name__ == '__main__':
    raise SystemExit(main())
