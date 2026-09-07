from __future__ import annotations
import hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

ROOT = Path('/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0010')
OUT = Path(sys.argv[1]).expanduser().resolve()
if OUT.exists() or OUT.is_symlink():
    raise RuntimeError(f'create-only recovery output exists: {OUT}')
OUT.mkdir(parents=True)

def sha_file(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {'path': str(path), 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
def write_exclusive(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f'create-only receipt exists: {path}')
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def utc(): return datetime.now(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')

raw_path = ROOT / 'experiments/TRR-0010/training/directional_fit/current_directional_watchdog_r2/fit/raw_runner_result.json'
raw = json.loads(raw_path.read_text(encoding='utf-8'))
if raw.get('status') != 'COMPLETED' or int(raw.get('selected_step', -1)) != 0:
    raise RuntimeError('raw runner receipt is not the frozen selected-step-0 result')
values = raw.get('checkpoint_state_bindings')
if not isinstance(values, list): raise RuntimeError('raw runner checkpoint bindings missing')
selected = []
for value in values:
    if not isinstance(value, dict) or not isinstance(value.get('checkpoint'), dict): continue
    cp = value['checkpoint']; md = cp.get('metadata')
    if isinstance(md, dict) and int(md.get('selected_step', -1)) == 0: selected.append(cp)
if len(selected) != 1: raise RuntimeError(f'step0 checkpoint binding count {len(selected)}')
checkpoint = selected[0]
if checkpoint.get('metadata', {}).get('runner_point_state_sha256') != raw.get('selected_state_sha256'):
    raise RuntimeError('step0 runner-point state digest differs from raw selected state')
checkpoint_path = Path(str(checkpoint['path'])).expanduser().resolve()
actual = sha_file(checkpoint_path)
if actual['bytes'] != int(checkpoint['bytes']) or actual['sha256'] != checkpoint['sha256']:
    raise RuntimeError('immutable step0 checkpoint changed')

# Import only the existing production provider and restore/export helpers.
import sys as _sys
_sys.path.insert(0, str(ROOT / 'scripts'))
from trr0010_directional_fit import restore_selected_and_export, export_selected_base_decoder_state
from trr0010_production_provider import build_inputs
from trr0010_directional_fit_cli import _load_diagnostic_binding

binding_path = ROOT / 'experiments/TRR-0010/setup/production_arm_binding_current_stage3_v3.json'
binding = json.loads(binding_path.read_text(encoding='utf-8'))
lease_path = ROOT / 'experiments/TRR-0010/training/directional_fit/current_directional_lease_r2.json'
lease = json.loads(lease_path.read_text(encoding='utf-8'))
# The recovery is deliberately CPU-only. Preserve the signed production caps,
# but use a CPU device so it performs no new CUDA allocation.
lease_caps = dict(lease)
lease_caps['device'] = 'cpu'
diag = _load_diagnostic_binding(Path('/tmp/trr-p09/experiments/TRR-P09/setup/fixed-diagnostic-binding-r1.json'))
stages=[]
def guard(stage: str) -> None:
    stages.append({'stage': str(stage), 'timestamp_utc': utc()})
started=time.perf_counter()
inputs = build_inputs(binding, lease_caps, torch.device('cpu'), guard, arm_name='current_directional', bank_role='B0', checkpoint_steps=(0,1000,2000,4000,8000,12000,13000), output_root=OUT, diagnostic_binding=diag)
provider_seconds=time.perf_counter()-started
runtime=inputs['runtime']
restore_started=time.perf_counter()
restore=restore_selected_and_export(checkpoint_path=checkpoint_path, expected_checkpoint=checkpoint, export_path=OUT/'effective_readout_w.safetensors', runtime=runtime, public_embedding=inputs['public_embedding'], selected_step=0)
restore_seconds=time.perf_counter()-restore_started
base_started=time.perf_counter()
base=export_selected_base_decoder_state(path=OUT/'base_decoder_state.safetensors', runtime=runtime, selected_receipt=restore, selected_step=0, metadata={'arm_name':'current_directional','bank_role':'B0','recovery':'export_only_from_immutable_selected_step_0'})
base_seconds=time.perf_counter()-base_started
receipt={
 'schema':'token-reconstruction.trr0010-directional-export-recovery.v1','task_id':'TRR-0010','status':'EXPORT_RECOVERY_COMPLETE','arm_name':'current_directional','bank_role':'B0','device':'cpu','truth_opened':False,'model_updates':False,'fit_loop_called':False,'selection_changed':False,
 'recovery_scope':'Export only from the immutable selected step-0 checkpoint retained by the completed current-arm fit; no refit, optimizer update, checkpoint reselection, capture, labels, or evaluation truth.',
 'source_failure_preserved':str(ROOT/'experiments/TRR-0010/training/directional_fit/current_directional_watchdog_r2/fit/failure.json'),
 'raw_runner_result':sha_file(raw_path),'checkpoint_binding':checkpoint,'checkpoint_actual':actual,'raw_selected_state_sha256':raw['selected_state_sha256'],
 'binding':sha_file(binding_path),'lease':sha_file(lease_path),'diagnostic_binding':{'path':diag['path'],'bytes':diag['bytes'],'sha256':diag['sha256']},
 'provider_source':sha_file(ROOT/'scripts/trr0010_production_provider.py'),'fit_source':sha_file(ROOT/'scripts/trr0010_directional_fit.py'),'caller_source':sha_file(ROOT/'scripts/trr0010_p09_caller.py'),
 'outputs':{'effective_readout':restore['export'],'base_decoder_state':base},
 'timing':{'provider_preparation_seconds':provider_seconds,'restore_export_seconds':restore_seconds,'base_decoder_export_seconds':base_seconds,'total_seconds':time.perf_counter()-started},
 'guard_stages':stages,'command':list(_sys.argv),'started_utc':stages[0]['timestamp_utc'] if stages else utc(),'finished_utc':utc(),
}
write_exclusive(OUT/'recovery_receipt.json',receipt)
print(json.dumps(receipt,sort_keys=True,indent=2))
