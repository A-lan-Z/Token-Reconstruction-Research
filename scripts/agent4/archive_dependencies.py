from common import *
import importlib,importlib.metadata,tarfile
names=['torch','transformers','safetensors','numpy','tokenizers','huggingface_hub','requests','packaging','regex','filelock','tqdm','yaml','sympy','networkx','jinja2','fsspec','mpmath','charset_normalizer','idna','urllib3','certifi']
paths=[]
for name in names:
    module=importlib.import_module(name);p=Path(module.__file__).parent
    if p not in paths:paths.append(p)
archive=OUT/'backup/python-packages.tar'
with tarfile.open(archive,'x') as tar:
    for p in paths:tar.add(p,arcname=p.name,recursive=True,filter=lambda info: None if '__pycache__' in info.name else info)
freeze=subprocess.check_output(['python3','-m','pip','freeze'],text=True)
(OUT/'backup/pip-freeze.txt').write_text(freeze)
write(EVID/'dependency_backup.json',{'archive':str(archive),'sha256':digest(archive),'bytes':archive.stat().st_size,'packages':[str(p) for p in paths],'pip_freeze':freeze,'scope':'installed Python packages and bundled libraries; host driver/CUDA OS runtime external','source_archive':'upstream/sipit-820683156b7257313046a4fb3c492e52519525b7.tar.gz'})
print(archive.stat().st_size)
