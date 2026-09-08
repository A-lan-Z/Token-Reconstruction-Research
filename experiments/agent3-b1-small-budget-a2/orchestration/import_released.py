from pathlib import Path
import json,shutil,subprocess,datetime
import torch
from safetensors.torch import load_file
from scripts.agent3_shortlists.core import binding,verify,write_json,sha
root=Path('../TRR-P12').resolve();out=Path('outputs/agent3-b1-small-budget-a2');ev=Path('experiments/agent3-b1-small-budget-a2')
review=json.loads((root/'experiments/TRR-P12/capture-equivalence-review-r1.json').read_text())
assert review['status']=='EXACT_H_EQUIVALENCE_PASS'
for condition in review['conditions']:
 a=load_file(verify(condition['qualified_prefix_observation']))['activations'];b=load_file(verify(condition['full_model_observation']))['activations']
 assert torch.equal(a,b) and a.shape==(8,128,2048)
profile=root/'experiments/TRR-P12/capture-profile-clarification-r1.json';assert sha(profile)=='0c16fe368b4f6f9b6d9980b4a6fcabb5625e7c82d9ceb1fc3143736bfa301a99'
execution=root/'experiments/TRR-P12/capture-execution-r1.json';e=json.loads(execution.read_text());assert e['status']=='COMPLETE_NO_TRUTH' and len(e['events'])==8
order=json.loads((ev/'source-order-binding.json').read_text())['order_first32_by_domain']
releases=out/'releases';releases.mkdir(exist_ok=False)
records=[]
for event in e['events']:
 assert event['returncode']==0
 emitted=json.loads(event['stdout']);capture=verify(emitted['capture']);c=json.loads(capture.read_text());obs=verify(c['observation']);o=json.loads(verify(c['order']).read_text())
 d,s=event['domain'],event['stage']
 assert c['status']=='PUBLIC_OBSERVATION_CELL_COMPLETE_NO_TRUTH'
 assert c['observation']['domain']==d and c['observation']['stage_updates']==s and c['observation']['shape']==[128,128,2048]
 assert o['ordered_record_ids'][:32]==[r['record_id'] for r in order[d]]
 assert o['ordered_h128_sequence_sha256'][:32]==[r['h128_sequence_sha256'] for r in order[d]]
 release=releases/f'{d}-{s}.json'
 write_json(release,{'domain':d,'stage':s,'record_ids':o['ordered_record_ids'],'observations':binding(obs),'capture_receipt':binding(capture),'order_receipt':c['order']})
 dest=out/'inputs'/f'{d}-{s}'
 subprocess.run(['python3','-m','scripts.agent3_shortlists.import_observations','--release',str(release),'--output',str(dest)],check=True)
 records.append({'domain':d,'stage':s,'release':binding(release),'contract':binding(dest/'contract.json'),'import_receipt':binding(dest/'import-receipt.json'),'capture_receipt':binding(capture),'order_receipt':binding(verify(c['order']))})
reviewdir=ev/'shared-capture';reviewdir.mkdir(exist_ok=False)
for name in ('capture-equivalence-review-r1.json','capture-profile-clarification-r1.json','capture-execution-r1.json','target-execution-r1.json','target-preflight-r1.json','target-source-disjointness-r1.json'):
 shutil.copy2(root/'experiments/TRR-P12'/name,reviewdir/name)
write_json(ev/'input-matrix.json',{'schema':'agent3-input-matrix-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cells':records,'shared_capture_execution':binding(execution),'capture_profile':binding(profile),'exact_equivalence_independently_rechecked':True,'source_labels_opened':False,'decision':'RELEASE_8_CELL_CPU_MATRIX','scientific_compute_estimate_seconds':8*64.06,'available_meminfo':Path('/proc/meminfo').read_text()})
print('All8paired observation cells verified and imported; exact-H equivalence independently rechecked; no labels loaded')
