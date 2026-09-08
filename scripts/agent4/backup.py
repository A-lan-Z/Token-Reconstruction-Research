"""Only separately obtained public checkpoint assets; no target state access."""
from common import *
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
base=Path('/home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct')
revision=(base/'refs/main').read_text().strip(); snapshot=base/'snapshots'/revision
assets=OUT/'backup';assets.mkdir(exist_ok=False)
state={}
with safe_open(str(snapshot/'model.safetensors'),framework='pt',device='cpu') as f:
    for k in f.keys():
        if k=='model.embed_tokens.weight' or any(k.startswith(f'model.layers.{i}.') for i in range(4)):
            state[k.removeprefix('model.')]=f.get_tensor(k)
save_file(state,str(assets/'prefix.safetensors'));del state
for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
    shutil.copyfile(snapshot/name,assets/name)
lens=Path('/home/alanz/spartan/punim2939/Token-Reconstruction-Research/outputs/TRR-0002/strict-surrogate-heavy/control-assets/lens_alpaca.pt')
shutil.copyfile(lens,assets/'lens_alpaca.pt')
restore=OUT/'restore';shutil.copytree(assets,restore)
files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p),'restore_sha256':digest(restore/p.name)} for p in assets.iterdir()]
assert all(f['sha256']==f['restore_sha256'] for f in files)
write(EVID/'backup.json',{'revision':revision,'source':str(snapshot),'files':files,'actual_copies':2,'restore':str(restore),'environment':environment()})
print(json.dumps({'revision':revision,'files':len(files),'bytes':sum(f['bytes'] for f in files)}))
