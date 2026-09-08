"""Preserve verified frozen candidate bytes as deterministic compressed files."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from scripts.agent3_shortlists.core import binding, verify, write_json
from scripts.agent3_shortlists.evaluate import load_frozen


def main(a):
    freeze=load_frozen(a.freeze)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inventory=[]
    for cell in freeze['cells']:
        receipt=json.loads(verify(cell['receipt']).read_text())
        for method in receipt['methods']:
            original=verify(method['artifact'])
            target=out/f"{cell['domain']}-stage{cell['stage']}-{method['method']}.safetensors.gz"
            with original.open('rb') as source,target.open('xb') as dest:
                with gzip.GzipFile(filename='',mode='wb',fileobj=dest,compresslevel=6,mtime=0) as zipped:
                    shutil.copyfileobj(source,zipped,1<<20)
            digest=hashlib.sha256()
            with gzip.open(target,'rb') as source:
                for block in iter(lambda:source.read(1<<20),b''):digest.update(block)
            if digest.hexdigest()!=method['artifact']['sha256']:raise ValueError('compressed roundtrip mismatch')
            inventory.append({'domain':cell['domain'],'stage':cell['stage'],'method':method['method'],'original':method['artifact'],'compressed':binding(target),'roundtrip_verified':True})
    write_json(out/'inventory.json',{'schema':'agent3-frozen-candidate-archive-v1','freeze':binding(a.freeze),'files':inventory,
                                  'compressed_bytes':sum(x['compressed']['bytes'] for x in inventory),'truth_opened_by_archiver':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--output',required=True);main(p.parse_args())
