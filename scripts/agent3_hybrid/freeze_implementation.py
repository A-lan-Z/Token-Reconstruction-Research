"""Bind qualified code and assets before selecting the fresh source panel."""
import json,shutil
from pathlib import Path
from scripts.agent3_hybrid.common import *
from scripts.agent3_hybrid.runtime import CODE_SHA,ASSETS,SNAPSHOT

def main():
    q=OUT/'qualification-r2/qualification.json';r=json.loads(q.read_text())
    if r['status']!='PASS_DEVELOPMENT_INTEGRATION_QUALIFICATION':raise ValueError('qualification failed')
    code=[binding(p) for p in sorted((ROOT/'scripts/agent3_hybrid').glob('*.py'))]
    code += [binding(ROOT/p) for p in CODE_SHA]
    code += [binding(ROOT/'scripts/agent3_shortlists'/p) for p in ('core.py','predict.py')]
    for p in ('capture.py','select_panel.py','qualify_capture.py'):
        code.append(binding(SHARED/'scripts/trr_p12'/p))
    assets=[binding(SNAPSHOT/'model.safetensors'),binding(SNAPSHOT/'config.json'),binding(SNAPSHOT/'tokenizer.json'),binding(ASSETS/'public_a1_lens.pt'),binding(ASSETS/'package/readout/public_normalized_embeddings.safetensors')]
    assert assets[0]['sha256']=='1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f'
    backup=Path('/mnt/c/Users/alanz/Token-Reconstruction-Backups/agent3-static-prefix-hybrid/public-model')
    paths=[p for p in SNAPSHOT.iterdir() if p.is_file() and (p.suffix=='.json' or p.name=='model.safetensors')]
    needed=sum(p.stat().st_size for p in paths)
    if shutil.disk_usage(backup.parent.parent).free-needed-1024**3<10*2**30:raise RuntimeError('backup free-space margin insufficient')
    backup.mkdir(parents=True,exist_ok=False);copies=[]
    for p in paths:
        target=backup/p.name;shutil.copy2(p,target);a,b=binding(p),binding(target)
        if a['sha256']!=b['sha256']:raise ValueError('actual model backup mismatch')
        copies.append({'source':a,'backup':b})
    write_json(EV/'public-model-backup.json',{'status':'VERIFIED_ACTUAL_PUBLIC_MODEL_BYTES','files':copies,'free_bytes_after':shutil.disk_usage(backup).free,'verified_utc':utc(),'b1_package_actual_backup':str(ROOT.parent/'agent3-b1-small-budget-a2/experiments/agent3-b1-small-budget-a2/backup-verification.json')})
    frozen={'status':'FROZEN_AFTER_DEVELOPMENT_BEFORE_CONFIRMATION','created_utc':utc(),'code_commit':commit(),'code':code,'assets':assets,'qualification':binding(q),'plan':binding(EV/'plan.md'),'source_selection':{'seed':5103,'ranges':{'pile':[0,10000],'finance':[50000,52000]},'records_per_domain':32},'numerical_port':'Pinned B1 weights/readout; CUDA FP32 consistent across arms. Development CPU/GPU top1 differed1/508 positions; no cross-device equivalence claimed.','resource_release':{'qualification_peak_reserved_bytes':r['peak_resources']['peak_gpu_reserved_bytes'],'additional_32_record_retention_host_bound_bytes':150*2**20,'estimated_matrix_seconds':4600,'cell_deadline_seconds':1200,'static_public_prefix':True},'source_truth_opened':False}
    write_json(EV/'implementation-freeze.json',frozen)
    print(json.dumps({'status':frozen['status'],'freeze':binding(EV/'implementation-freeze.json')}))
if __name__=='__main__':main()
