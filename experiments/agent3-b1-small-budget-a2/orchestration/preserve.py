from pathlib import Path
import json,shutil,hashlib,datetime,subprocess,time
start=time.monotonic();root=Path.cwd();task='agent3-b1-small-budget-a2';ev=root/'experiments'/task;out=root/'outputs'/task
backup=Path('/mnt/c/Users/alanz/Token-Reconstruction-Backups')/task/'stage1-r1'
free=shutil.disk_usage(backup.parent.parent).free
files=[p for folder in ('inputs','predictions','releases') for p in (out/folder).rglob('*') if p.is_file()]
files += [out/'assets/public_a1_lens.pt']
need=sum(p.stat().st_size for p in files)
assert free-need>10*2**30, 'backup safety margin below10GiB'
backup.mkdir(parents=True,exist_ok=False)
def bind(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
inventory=[]
for source in files:
 target=backup/source.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 a,b=bind(source),bind(target);assert a['sha256']==b['sha256'];inventory.append({'source':a,'backup':b})
# Compact local copies make receipt evidence available from Git as well.
for folder in ('inputs','predictions','releases'):
 for source in (out/folder).rglob('*.json'):
  target=ev/'frozen-receipts'/source.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
for name in ('freeze.json','input-matrix.json','source-order-binding.json','plan.md','shared-target-verification.json'):
 shutil.copy2(ev/name,backup/name)
shutil.copytree(root/'scripts/agent3_shortlists',backup/'code',ignore=shutil.ignore_patterns('__pycache__'))
r={'schema':'agent3-scientific-byte-backup-v1','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'free_bytes_before':free,'free_bytes_after':shutil.disk_usage(backup).free,'required_bytes':need,'files':inventory,'elapsed_seconds':time.monotonic()-start,'model_package_actual_backup':'/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012','model_package_verification':bind(ev/'backup-verification.json'),'truth_opened':False}
(ev/'scientific-byte-backup.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':'ACTUAL_BYTES_BACKED_UP_AND_VERIFIED','files':len(inventory),'bytes':need,'seconds':r['elapsed_seconds']}))
