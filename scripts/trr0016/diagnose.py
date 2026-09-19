from pathlib import Path
import json, torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
R=Path(__file__).resolve().parents[2]; S=R.parent; P14=S/'TRR-0014'; P15=S/'TRR-0015'
tok=AutoTokenizer.from_pretrained(S/'agent4-prefix-only-inversion/outputs/agent4-prefix-only-inversion/backup',local_files_only=True)
suffix=json.loads((P14/'outputs/TRR-0014/tokenizer_suffix_cache.json').read_text())
truth=json.loads((P14/'outputs/TRR-0014/fresh_r2/evaluator_truth.json').read_text()); records=[]
for key,tt in truth.items():
    if not key.startswith('matched__book'):continue
    old=load_file(str(P15/'outputs/TRR-0015/predictions'/f'r2__{key}__fragment256.safetensors'))
    base=load_file(str(P14/'outputs/TRR-0014/fresh_r2_predictions'/f'{key}__union128.safetensors'))['candidates'].tolist()
    for pos in range(1,len(tt)):
        if tt[pos] in old['candidates'][pos]:continue
        parents=[]
        for rank,b in enumerate(base[pos]):
            if tt[pos] in suffix[b]:parents.append({'rank':rank,'text':tok.decode([b]),'suffix_depth':suffix[b].index(tt[pos]),'suffix_count':len(suffix[b])})
        records.append({'id':key,'position':pos,'true':tok.decode([tt[pos]]),'reconstructed_context':tok.decode(old['tokens'][max(1,pos-6):pos].tolist()),'parents':parents[:12]})
with (R/'experiments/TRR-0016/opened_misses.json').open('x') as f:json.dump(records,f,indent=2)
print(json.dumps(records,indent=2))