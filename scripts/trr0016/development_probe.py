"""Retrospective candidate-only probe; old predictions supply context, never truth."""
from pathlib import Path
import json, time, hashlib
from collections import defaultdict
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from token_reconstruction.prefix_context_fragments import build_context_cache, contextual_candidates
from token_reconstruction.prefix_fragments import expand_candidates
ROOT=Path(__file__).resolve().parents[2]; SIB=ROOT.parent
X=ROOT/'experiments/TRR-0016'; OUT=ROOT/'outputs/TRR-0016/development'
P14=SIB/'TRR-0014'; P15=SIB/'TRR-0015'
ASSET=SIB/'agent4-prefix-only-inversion/outputs/agent4-prefix-only-inversion/backup'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2)
torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=False)
tok=AutoTokenizer.from_pretrained(ASSET,local_files_only=True)
t=time.perf_counter(); suffix,context=build_context_cache(tok,128256); setup=time.perf_counter()-t
assert [list(v) for v in suffix]==json.loads((P14/'outputs/TRR-0014/tokenizer_suffix_cache.json').read_text())
rows=json.loads((P14/'outputs/TRR-0014/fresh_r2/metadata.json').read_text())
outputs={}; inputs={}; start=time.perf_counter()
for row in rows:
    key=row['id']
    basepath=P14/'outputs/TRR-0014/fresh_r2_predictions'/f'{key}__union128.safetensors'
    predpath=P15/'outputs/TRR-0015/predictions'/f'r2__{key}__fragment256.safetensors'
    base=load_file(str(basepath))['candidates'].tolist(); old=load_file(str(predpath)); prediction=old['tokens'].tolist()
    new=[]
    for pos,b in enumerate(base):
        fallback=expand_candidates(b,suffix,256)
        assert fallback==old['candidates'][pos].tolist()
        text=tok.decode(prediction[1:pos],skip_special_tokens=False) if pos else ''
        new.append(contextual_candidates(b,suffix,context,text))
    outputs[key]=torch.tensor(new,dtype=torch.long)
    inputs[str(basepath)]=sha(basepath); inputs[str(predpath)]=sha(predpath)
save_file(outputs,str(OUT/'candidates.safetensors'))
write(X/'development_freeze.json',{'status':'candidate-only; old frozen predictions provide context; retrospective',
    'candidate_sha256':sha(OUT/'candidates.safetensors'),'input_sha256':inputs,
    'implementation':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/token_reconstruction/prefix_context_fragments.py']},
    'setup_seconds':setup,'proposal_seconds':time.perf_counter()-start,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'truth_read_before_freeze':False})
# Only score after the entire diagnostic shortlist artifact is immutable.
truth=json.loads((P14/'outputs/TRR-0014/fresh_r2/evaluator_truth.json').read_text())
groups=defaultdict(lambda:dict(positions=0,old_hits=0,new_hits=0,rescued=0,new_misses=0)); details=[]
for row in rows:
    key=row['id']; t=torch.tensor(truth[key][1:]); new=outputs[key][1:]
    old=load_file(str(P15/'outputs/TRR-0015/predictions'/f'r2__{key}__fragment256.safetensors'))['candidates'][1:]
    a=(old==t[:,None]).any(-1); b=(new==t[:,None]).any(-1)
    r=groups[row['condition']+'__'+row['group']]
    for k,v in [('positions',len(t)),('old_hits',int(a.sum())),('new_hits',int(b.sum())),('rescued',int((~a&b).sum())),('new_misses',int((a&~b).sum()))]:r[k]+=v
    details.append({'id':key,'rescued_positions':(torch.where(~a&b)[0]+1).tolist(),'newly_missing_positions':(torch.where(a&~b)[0]+1).tolist()})
write(X/'development_score.json',{'summary':dict(groups),'details':details,'caveat':'candidate recall under old reconstructed histories; not causal end-to-end accuracy','freeze_sha256':sha(X/'development_freeze.json')})
print(json.dumps({'summary':dict(groups),'setup_seconds':setup},indent=2),flush=True)