from common import *
import tarfile,gc,sys
start=time.perf_counter();torch.set_num_threads(2)
# An actual independently copied prefix is deserialized and executed on CPU.
p=load_prefix(torch.bfloat16,device='cpu',asset_root=OUT/'restore')
ids=torch.tensor([[128000,2028]])
x=p.forward_full(ids).float().clone();del p;gc.collect()
p=load_prefix(torch.bfloat16,device='cpu',asset_root=OUT/'backup')
y=p.forward_full(ids).float();assert torch.equal(x,y)
del p;gc.collect()
restore=OUT/'restored-packages';restore.mkdir(exist_ok=False)
with tarfile.open(OUT/'backup/python-packages.tar') as tar:tar.extractall(restore,filter='data')
# Verify isolated package imports resolve into the restored package directory.
code="import torch,transformers,safetensors,numpy,json; print(json.dumps({m.__name__:m.__file__ for m in [torch,transformers,safetensors,numpy]}))"
env=os.environ.copy();env['PYTHONPATH']=str(restore)
loaded=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,cwd='/tmp',text=True))
assert all(str(restore) in path for path in loaded.values())
write(EVID/'clean_restore.json',{'public_prefix_cpu_output_exact':True,'restored_package_imports':loaded,'seconds':time.perf_counter()-start,'scope':'actual independent prefix copy and restored Python package imports; system interpreter, system dependencies and GPU driver external','environment':environment()})
print('Actual prefix and dependency restore passed')
