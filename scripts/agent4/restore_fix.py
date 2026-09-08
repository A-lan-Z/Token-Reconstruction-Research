from common import *
import tarfile,shutil,numpy,sys
src=Path(numpy.__file__).parent.parent/'numpy.libs'
archive=OUT/'backup/numpy-libs.tar'
with tarfile.open(archive,'x') as t:t.add(src,arcname='numpy.libs')
with tarfile.open(archive) as t:t.extractall(OUT/'restored-packages',filter='data')
restore=OUT/'restored-packages'
code="import torch,transformers,safetensors,numpy,json; print(json.dumps({m.__name__:m.__file__ for m in [torch,transformers,safetensors,numpy]}))"
env=os.environ.copy();env['PYTHONPATH']=str(restore)
loaded=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,cwd='/tmp',text=True))
assert all(str(restore) in path for path in loaded.values())
write(EVID/'clean_restore.json',{'public_prefix_cpu_output_exact':True,'restored_package_imports':loaded,'numpy_libs_archive_sha256':digest(archive),'numpy_libs_archive_bytes':archive.stat().st_size,'failed_attempt':'restore_check.py dependency import initially failed because numpy.libs is a sibling directory; original archive retained, added companion archive and verified fresh imports','scope':'actual independent prefix copy and restored Python package imports; system interpreter, system dependencies and GPU driver external','environment':environment()})
print('Restored imports passed after numpy.libs companion backup')
