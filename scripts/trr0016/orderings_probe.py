"""Candidate-only fixed exploratory family; freeze all proposals, then score."""
from pathlib import Path
import json,time,hashlib
from collections import defaultdict
import torch
from safetensors.torch import load_file,save_file
from token_reconstruction.prefix_fragment_orderings import VARIANTS,ordered_fragments
R=Path(__file__).resolve().parents[2];S=R.parent;P14=S/'TRR-0014';P15=S/'TRR-0015';X=R/'experiments/TRR-0016';OUT=R/'outputs/TRR-0016/orderings'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=False)
cache=json.loads((P14/'outputs/TRR-0014/tokenizer_suffix_cache.json').read_text())
rows=json.loads((P14/'outputs/TRR-0014/fresh_r2/metadata.json').read_text());data={v:{} for v in VARIANTS};timing={v:0. for v in VARIANTS};inputs={}
for row in rows:
    key=row['id'];p=P14/'outputs/TRR-0014/fresh_r2_predictions'/f'{key}__union128.safetensors';inputs[str(p)]=sha(p)
    base=load_file(str(p))['candidates'].tolist()
    for v in VARIANTS:
        t=time.perf_counter();data[v][key]=torch.tensor([ordered_fragments(b,cache,v) for b in base]);timing[v]+=time.perf_counter()-t
hashes={}
for v in VARIANTS:
    p=OUT/(v+'.safetensors');save_file(data[v],str(p));hashes[v]=sha(p)
write(X/'orderings_freeze.json',{'variants':list(VARIANTS),'candidates_sha256':hashes,'inputs_sha256':inputs,'proposal_seconds':timing,'implementation_sha256':sha(R/'src/token_reconstruction/prefix_fragment_orderings.py'),'probe_sha256':sha(Path(__file__)),'truth_read_before_freeze':False})
truth=json.loads((P14/'outputs/TRR-0014/fresh_r2/evaluator_truth.json').read_text());results={}
for v in VARIANTS:
    groups=defaultdict(lambda:dict(positions=0,old_hits=0,new_hits=0,rescued=0,new_misses=0))
    for row in rows:
        key=row['id'];t=torch.tensor(truth[key][1:]);new=data[v][key][1:]
        old=load_file(str(P15/'outputs/TRR-0015/predictions'/f'r2__{key}__fragment256.safetensors'))['candidates'][1:]
        a=(old==t[:,None]).any(-1);b=(new==t[:,None]).any(-1);r=groups[row['condition']+'__'+row['group']]
        for k,x in [('positions',len(t)),('old_hits',int(a.sum())),('new_hits',int(b.sum())),('rescued',int((~a&b).sum())),('new_misses',int((a&~b).sum()))]:r[k]+=x
    results[v]=dict(groups)
write(X/'orderings_score.json',{'results':results,'freeze_sha256':sha(X/'orderings_freeze.json'),'status':'development-only candidate recall; not full reconstruction'})
print(json.dumps({'results':results,'seconds':timing},indent=2),flush=True)