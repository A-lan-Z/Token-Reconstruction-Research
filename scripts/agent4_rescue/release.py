"""Create a reviewable sanitized package; never publishes anything."""
from shared import *
import shutil,zipfile
dst=OUT/'sanitized_release_r2';dst.mkdir(exist_ok=False)
paths=[ROOT/'coordination/results/agent4-prefix-only-rescue.md',X/'REPRODUCE.md',X/'final_sources.json',X/'development_sources.json',X/'freeze.json',ROOT/'tests/test_agent4_rescue.py']
paths+=[p for p in (ROOT/'scripts/agent4_rescue').glob('*.py') if p.name!='release.py']
for p in paths:
    q=dst/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
# Public package has aggregate results and selected numerical checks, no runtime paths.
summary=json.loads((E/'summary.json').read_text())
pub={'task_id':'agent4-prefix-only-rescue','status':'SANITIZED_REVIEW_COPY_NOT_PUBLISHED','summary':summary,'model_revision':'9213176726f574b556790deb65791e0c5aa438b6','runtime':{'python':'3.12.3','torch':'2.10.0+cu128','transformers':'5.3.0','gpu':'RTX 5080 16 GiB','cpu_threads':2},'raw_evidence':'Retained locally; excluded from this release package pending separate authorization.'}
q=dst/'experiments/agent4-prefix-only-rescue/manifest.json';q.write_text(json.dumps(pub,indent=2)+'\n')
for name in ['executor_qualification.json','inverse_preflight.json','bottleneck_diagnostic.json','post_freeze_failure_diagnostic.json']:
    v=json.loads((E/name).read_text());v.pop('environment',None)
    q=dst/'experiments/agent4-prefix-only-rescue/evidence'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(v,indent=2)+'\n')
# Background documents, coordination state and full traces were never copied.
allowed=[{'path':str(p.relative_to(dst)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(dst.rglob('*')) if p.is_file()]
for item in allowed:
    data=(dst/item['path']).read_text()
    assert '/home/alanz/' not in data and 'C:\\Users\\alanz' not in data
archive=OUT/'agent4_prefix_only_rescue_sanitized_release.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for item in allowed:z.write(dst/item['path'],item['path'])
write(X/'release_list.json',{'status':'PREPARED_NOT_PUBLISHED','new_approval_required':True,'package_path':str(archive),'package_sha256':digest(archive),'package_bytes':archive.stat().st_size,'allowlist':allowed,'excluded':['user-supplied archive/brief/research documents','coordination/STATE.json and task messages','full runtime metadata with absolute machine paths','raw predictions/candidate traces','evaluator truth and observation tensors','actual model/tokenizer/lens/dependency backup assets'],'publication_note':'Do not push the full local rescue branch: it contains excluded incoming documents. If approved, publish only this allowlist on a separately sanitized branch.'})
print('Sanitized release package:',len(allowed),'files',archive.stat().st_size,'bytes')
