"""Actual new backup/restore and isolated dependency execution before final."""
from shared import *
import shutil,tarfile,os
torch.set_num_threads(2);start=time.perf_counter();backup=OUT/'backup';restore=OUT/'restore'
backup.mkdir();restore.mkdir();records=[]
for item in json.loads((ROOT/'experiments/agent4-prefix-only-inversion/evidence/backup.json').read_text())['files']:
    name=item['name'];src=OLD/'backup'/name
    if not src.is_file():continue
    shutil.copyfile(src,backup/name);shutil.copyfile(backup/name,restore/name)
    assert digest(backup/name)==digest(restore/name)==item['sha256']
    records.append({'file':name,'backup':str(backup/name),'restore':str(restore/name),'sha256':item['sha256'],'bytes':src.stat().st_size})
dep=json.loads((ROOT/'experiments/agent4-prefix-only-inversion/evidence/required_dependency_restore.json').read_text())
packages=OUT/'restored-dependencies';packages.mkdir()
for key,hkey in [('archive','sha256'),('supplement_archive','supplement_sha256')]:
    arc=Path(dep[key]);assert digest(arc)==dep[hkey]
    with tarfile.open(arc) as tf:tf.extractall(packages,filter='data')
    records.append({'archive':str(arc),'sha256':dep[hkey],'bytes':arc.stat().st_size,'fresh_restore':str(packages)})
prefix=load_prefix(torch.bfloat16,device='cpu',asset_root=backup)
ids=torch.tensor([[128000,2028,374,264,1296,13,220,16]])
expected=hashlib.sha256(prefix.forward_full(ids).float().numpy().tobytes()).hexdigest()
del prefix
script='''import sys,json,hashlib,torch,transformers
from shared import load_prefix
torch.set_num_threads(2)
p=load_prefix(torch.bfloat16,device="cpu",asset_root=sys.argv[1])
ids=torch.tensor([[128000,2028,374,264,1296,13,220,16]])
print(json.dumps({"hash":hashlib.sha256(p.forward_full(ids).float().numpy().tobytes()).hexdigest(),"torch_path":torch.__file__,"transformers_path":transformers.__file__,"sys_path":sys.path}))
'''
env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(packages),str(ROOT/'src'),str(ROOT/'scripts/agent4_rescue')]);env['PYTHONNOUSERSITE']='1'
result=subprocess.run([sys.executable,'-S','-c',script,str(restore)],env=env,text=True,capture_output=True)
write(E/'restore_attempt.json',{'command':'python3 -S isolated dependency and independent prefix restore','returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
if result.returncode:raise RuntimeError(result.stderr)
value=json.loads(result.stdout);assert value['hash']==expected
write(E/'backup_restore.json',{'environment':environment(),'assets':records,'isolated':value,'output_exact':True,'seconds':time.perf_counter()-start,'external_requirements':'Python interpreter/stdlib, system libraries and NVIDIA driver; source code retained in task git history'})
print('Actual independent model and fresh dependency restore passed')
