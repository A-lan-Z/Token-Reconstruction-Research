"""Archive installed required distributions, then restore without global site-packages."""
from common import *
import importlib.metadata as md,tarfile,sys
from packaging.requirements import Requirement
roots=['torch','transformers','numpy','safetensors','psutil']
queue=list(roots);seen={};missing=[]
while queue:
    name=queue.pop();key=name.lower().replace('_','-')
    if key in seen:continue
    try:dist=md.distribution(name)
    except md.PackageNotFoundError:missing.append(name);continue
    seen[key]=dist
    for dep in dist.requires or []:
        req=Requirement(dep)
        if req.marker is None or req.marker.evaluate({'extra':''}):queue.append(req.name)
if missing:raise RuntimeError(f'missing required installed distributions: {missing}')
archive=OUT/'backup/required-distributions.tar';start=time.perf_counter();members={}
with tarfile.open(archive,'x') as tar:
    for name,dist in seen.items():
        for rel in dist.files or []:
            path=Path(dist.locate_file(rel)).resolve()
            if not path.is_file() or '__pycache__' in str(rel):continue
            parts=Path(str(rel)).parts
            dest=str(rel) if '..' not in parts else f'external-bin/{path.name}'
            if dest in members:continue
            tar.add(path,arcname=dest,recursive=False);members[dest]=path.stat().st_size
restore=OUT/'restored-required-distributions';restore.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:tar.extractall(restore,filter='data')
code="""import sys,json,torch,transformers,safetensors,numpy
from common import load_prefix
import hashlib
p=load_prefix(torch.bfloat16,device='cpu')
torch.set_num_threads(2)
y=p.forward_full(torch.tensor([[128000,2028]])).view(torch.uint8).numpy().tobytes()
print(json.dumps({'packages':{m.__name__:m.__file__ for m in [torch,transformers,safetensors,numpy]},'output_sha256':hashlib.sha256(y).hexdigest(),'sys_path':sys.path}))
"""
env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(restore),str(ROOT/'src'),str(ROOT/'scripts/agent4')])
result=json.loads(subprocess.check_output([sys.executable,'-S','-c',code],env=env,cwd='/tmp',text=True))
assert all(str(restore) in p for p in result['packages'].values())
assert not any('site-packages' in p or 'dist-packages' in p for p in result['sys_path'])
torch.set_num_threads(2);p=load_prefix(torch.bfloat16,device='cpu')
expected=hashlib.sha256(p.forward_full(torch.tensor([[128000,2028]])).view(torch.uint8).numpy().tobytes()).hexdigest()
assert result['output_sha256']==expected
write(EVID/'required_dependency_restore.json',{'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'files':len(members),'distributions':{n:d.version for n,d in seen.items()},'seconds':time.perf_counter()-start,'restore':str(restore),'isolated_no_global_site_packages':True,'restored_prefix_output_exact':True,'result':result,'external_requirements':'system Python/stdlib, glibc and OS GPU driver; all active declared required Python distributions archived'})
print('Full required-distribution restore and prefix execution passed')
