from __future__ import annotations
import hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
P11=Path('/home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-P11').resolve(); A1=P11.parent/'TRR-0012'; CALLBACK=A1/'scripts/trr0012_transfer_analysis.py'
MATRIX=A1/'experiments/TRR-0012/transfer/production_expanded_fixed_coordinator_r1/prediction_matrix_expanded_fixed.json'; FREEZE=A1/'experiments/TRR-0012/transfer/production_expanded_fixed_coordinator_r1/freeze_receipt.json'; TRUTH=P11/'outputs/TRR-P11/private-evaluation/transfer-analysis/truth_manifest.json'; OUT=P11/'outputs/TRR-P11/private-evaluation/transfer-analysis/analysis_result_r2.json'; RECEIPT=P11/'outputs/TRR-P11/private-evaluation/transfer-analysis/analysis_execution_receipt_r2.json'; CONTRACT=A1/'experiments/TRR-0012/transfer/analysis/analysis_contract_v1.json'
if OUT.exists() or OUT.is_symlink() or RECEIPT.exists() or RECEIPT.is_symlink(): raise RuntimeError('create-only analysis output already exists')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),bytes()): h.update(chunk)
 return h.hexdigest()
def rss_bytes(): return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
t0=time.monotonic(); cpu0=os.times(); rss0=rss_bytes()
sys.path[:0]=[str(A1),str(A1/'scripts')]
spec=importlib.util.spec_from_file_location('trr0012_transfer_analysis_repaired',CALLBACK); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
report=mod.analyze_transfer_matrix(matrix_path=MATRIX,freeze_receipt_path=FREEZE,truth_manifest_path=TRUTH,repository_root=A1,output_path=OUT)
t1=time.monotonic(); cpu1=os.times(); rss1=rss_bytes()
execution={'wrapper':'/tmp/run_trr_p11_transfer_analysis_r2.py','status':'COMPLETED','started_monotonic':t0,'ended_monotonic':t1,'wall_seconds':t1-t0,'user_cpu_seconds':cpu1.user-cpu0.user,'system_cpu_seconds':cpu1.system-cpu0.system,'wrapper_peak_rss_bytes':max(rss0,rss1),'resource_mode':'CPU one thread; external RSS cap 2147483648; external wall watchdog 120s; no GPU','matrix_sha256':sha(MATRIX),'freeze_sha256':sha(FREEZE),'truth_manifest_sha256':sha(TRUTH),'truth_opened':True,'source_text_loaded':False,'raw_token_values_exported':False}
report['analysis_code']={'path':str(CALLBACK),'bytes':CALLBACK.stat().st_size,'sha256':sha(CALLBACK),'commit':'ca831bf726c77823a11fb27594cebfafa43e8daa'}
report['analysis_contract']={'path':str(CONTRACT),'bytes':CONTRACT.stat().st_size,'sha256':sha(CONTRACT)}
report['execution']=execution
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
receipt={'schema':'token-reconstruction.trr-p11-transfer-analysis-execution-receipt.v2','task_id':'TRR-P11','status':'COMPLETED_PRIVATE_TRUTH_GATED_ANALYSIS','supersedes_failure_receipt':'outputs/TRR-P11/private-evaluation/transfer-analysis/analysis_failure_r1.json','callback':{'path':str(CALLBACK),'bytes':CALLBACK.stat().st_size,'sha256':sha(CALLBACK),'commit':'ca831bf726c77823a11fb27594cebfafa43e8daa'},'contract':{'path':str(CONTRACT),'bytes':CONTRACT.stat().st_size,'sha256':sha(CONTRACT)},'result':{'path':str(OUT),'bytes':OUT.stat().st_size,'sha256':sha(OUT)},'inputs':{'matrix':{'path':str(MATRIX),'sha256':sha(MATRIX)},'freeze':{'path':str(FREEZE),'sha256':sha(FREEZE)},'truth_manifest':{'path':str(TRUTH),'sha256':sha(TRUTH)}},'execution':execution,'output_boundary':{'result_in_p11_private':True,'raw_source_or_token_arrays_exported':False,'truth_accessed_only_after_public_gates':True,'truth_opened':True}}
RECEIPT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report.get('status'),'result_path':str(OUT),'result_sha256':sha(OUT),'receipt_path':str(RECEIPT),'receipt_sha256':sha(RECEIPT),'wall_seconds':execution['wall_seconds'],'user_cpu_seconds':execution['user_cpu_seconds'],'system_cpu_seconds':execution['system_cpu_seconds'],'peak_rss_bytes':execution['wrapper_peak_rss_bytes'],'truth_opened':report.get('truth_opened'),'result_cells':len(report.get('results',{}))},sort_keys=True))
