"""Tokenizer-only suffix expansion of frozen prefix-derived proposals."""
from native import *
from transformers import AutoTokenizer
from functools import lru_cache
tok=AutoTokenizer.from_pretrained(ASSETS/'backup',local_files_only=True)
env=environment()
@lru_cache(None)
def fragments(v):
    s=tok.decode([int(v)],skip_special_tokens=False)
    if '\ufffd' in s:return ()
    found=[]
    # Every proper Unicode-character suffix; longest first. No learned word list.
    for j in range(1,len(s)):
        ids=tok.encode(s[j:],add_special_tokens=False)
        if ids and ids[0]!=v and ids[0] not in found:found.append(ids[0])
    return tuple(found)
def expand(base,k):
    found=list(dict.fromkeys(base));seen=set(found);ff=[fragments(v) for v in base]
    for j in range(max([len(v) for v in ff],default=0)):
        for f in ff:
            if j<len(f) and f[j] not in seen:
                found.append(f[j]);seen.add(f[j])
                if len(found)>=k:return found[:k]
    return found[:k]+[found[0]]*max(0,k-len(found))
meta=json.loads((OUT/'fresh_r1/metadata.json').read_text())
dst=OUT/'fragment_dev_r1';dst.mkdir(exist_ok=False);entries=[]
for row in meta:
    old=load_file(str(OUT/'fresh_r1_predictions'/(row['id']+'__metric64.safetensors')))['candidates'][1:]
    intrinsic=load_file(str(OUT/'intrinsic_dev_r1'/(row['id']+'__mlp_only.safetensors')))['candidates'][:,:64]
    for base_kind in ['embedding','union']:
        base=old if base_kind=='embedding' else torch.cat([old,intrinsic],dim=-1)
        ids=torch.tensor([expand(b.tolist(),512) for b in base]);tokens=torch.cat([torch.tensor([128000]),ids[:,0]])
        method=base_kind;path=dst/(row['id']+'__'+method+'.safetensors')
        save_file({'tokens':tokens,'candidates':ids},str(path))
        entries.append({**row,'method':method,'path':str(path.relative_to(ROOT)),'sha256':digest(path),'seconds':0.0})
write(X/'fragment_dev_r1_freeze.json',{'environment':env,'entries':entries,'truth_read':False,'scope':'opened-panel candidate recall; no reconstruction claim','fragment_cache_entries':fragments.cache_info().currsize,'rule':'round-robin all proper character suffixes, first tokenizer piece; longest suffix first; preserve base candidates; pad first ID to512'})
print('All96 token-fragment proposals frozen',flush=True)
