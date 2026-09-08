"""Evaluator-side unused-source selection; execute only after settings freeze."""
from common import *
import urllib.request,re,random
freeze=EVID/'final_settings_freeze.json'
assert freeze.is_file(), 'settings must be frozen before new sources are selected'
assert digest(ROOT/'experiments/agent4-prefix-only-inversion/adam_settings.json')==json.loads(freeze.read_text())['settings_sha256']
rng=random.Random(4402);records=[];source_receipts=[]
for book in [11,84]:
    url=f'https://www.gutenberg.org/ebooks/{book}.txt.utf-8'
    data=urllib.request.urlopen(url,timeout=45).read();text=data.decode('utf-8-sig')
    path=OUT/'backup'/f'gutenberg-{book}.txt';path.write_bytes(data)
    # Reserve one deterministic interior paragraph pair, separate from openings
    # and from the task's two development works. No target-specific labels used.
    paragraphs=[x.strip() for x in re.split(r'\n\s*\n',text) if len(x)>300]
    usable=list(range(len(paragraphs)//4,3*len(paragraphs)//4))
    choices=rng.sample(usable,2)
    for j,index in enumerate(choices):
        records.append({'id':f'final_book{book}_{j}','group':'ordinary','positions':16,'source':url,'paragraph_index':index,'text':paragraphs[index],'text_sha256':hashlib.sha256(paragraphs[index].encode()).hexdigest()})
    source_receipts.append({'url':url,'sha256':digest(path),'bytes':len(data),'paragraph_indices':choices})
for i in range(2):
    value=''.join(rng.choice('abcdef0123456789') for _ in range(24))
    text=f'parse_{value}::λ_{rng.randrange(1000)}(0x{value[:8]}) != null;'
    records.append({'id':f'final_stress_{i}','group':'stress','positions':16,'source':'task-public synthetic identifier generator seed4402','text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
write(ROOT/'experiments/agent4-prefix-only-inversion/final_sources.json',records)
write(EVID/'final_source_selection.json',{'seed':4402,'sources':source_receipts,'settings_freeze_sha256':digest(freeze),'source_file_sha256':digest(ROOT/'experiments/agent4-prefix-only-inversion/final_sources.json'),'records':[{k:v for k,v in r.items() if k!='text'} for r in records],'source_scope':'unused within this task; not a canonical blind benchmark or repository-wide freshness claim','selected_utc':environment()['utc']})
print('Selected',len(records),'unused task-local records without displaying text')
