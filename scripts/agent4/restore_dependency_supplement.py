"""Repair absent OS-package dependency metadata using actual installed modules."""
from common import *
import importlib.util,importlib.metadata as md,shutil,tarfile,re,sys
restore=OUT/'restored-required-distributions';added=[];failures=[{'module':'idna','reason':'OS-distributed httpx metadata omitted transitive requirement; first isolated import failed'}]
code="""import sys,json,torch,transformers,safetensors,numpy
from common import load_prefix
import hashlib
torch.set_num_threads(2)
p=load_prefix(torch.bfloat16,device='cpu')
y=p.forward_full(torch.tensor([[128000,2028]])).view(torch.uint8).numpy().tobytes()
print(json.dumps({'packages':{m.__name__:m.__file__ for m in [torch,transformers,safetensors,numpy]},'output_sha256':hashlib.sha256(y).hexdigest(),'sys_path':sys.path}))
"""
env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(restore),str(ROOT/'src'),str(ROOT/'scripts/agent4')])
module='idna';start=time.perf_counter()
for attempt in range(12):
    spec=importlib.util.find_spec(module)
    if spec is None:raise RuntimeError(f'installed module unavailable: {module}')
    source=Path(spec.origin)
    if spec.submodule_search_locations:
        source=source.parent;target=restore/module
        shutil.copytree(source,target,dirs_exist_ok=True)
    else:
        target=restore/source.name;shutil.copyfile(source,target)
    added.append(target)
    for dist_name in md.packages_distributions().get(module,[]):
        metadata=Path(md.distribution(dist_name)._path)
        dest=restore/metadata.name
        if metadata.is_dir():shutil.copytree(metadata,dest,dirs_exist_ok=True)
        elif metadata.is_file():shutil.copyfile(metadata,dest)
        added.append(dest)
    run=subprocess.run([sys.executable,'-S','-c',code],env=env,cwd='/tmp',text=True,capture_output=True)
    if run.returncode==0:
        result=json.loads(run.stdout);break
    failures.append({'attempt':attempt+1,'stderr':run.stderr})
    match=re.search("No module named '([A-Za-z0-9_]+)(?:\\.[A-Za-z0-9_.]+)?'",run.stderr)
    if not match:raise RuntimeError(run.stderr)
    module=match.group(1)
else:raise RuntimeError('bounded runtime supplement attempts exhausted')
assert all(str(restore) in p for p in result['packages'].values())
assert not any('site-packages' in p or 'dist-packages' in p for p in result['sys_path'])
torch.set_num_threads(2);p=load_prefix(torch.bfloat16,device='cpu')
expected=hashlib.sha256(p.forward_full(torch.tensor([[128000,2028]])).view(torch.uint8).numpy().tobytes()).hexdigest()
assert expected==result['output_sha256']
archive=OUT/'backup/runtime-dependency-supplement.tar'
with tarfile.open(archive,'x') as tar:
    for path in dict.fromkeys(added):tar.add(path,arcname=path.name)
main=OUT/'backup/required-distributions.tar'
write(EVID/'required_dependency_restore.json',{'archive':str(main),'sha256':digest(main),'bytes':main.stat().st_size,'supplement_archive':str(archive),'supplement_sha256':digest(archive),'supplement_bytes':archive.stat().st_size,'restored_supplements':[str(p) for p in added],'failed_attempts':failures,'isolated_no_global_site_packages':True,'restored_prefix_output_exact':True,'result':result,'seconds_for_supplement_restore':time.perf_counter()-start,'scope':'active declared required-distribution closure plus actual runtime modules missing from OS metadata; isolated prefix execution passed; system interpreter/stdlib/glibc/OS driver external'})
print('Complete isolated dependency restore and prefix execution passed')
