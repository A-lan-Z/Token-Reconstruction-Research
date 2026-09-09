"""Back up actual frozen outputs and create tracked, lossless candidate archives."""
import argparse,gzip,hashlib,json,shutil,time
from pathlib import Path
from scripts.agent3_hybrid.common import *
from scripts.agent3_hybrid.evaluate import load_freeze

def main(a):
    start=time.monotonic();f=load_freeze(a.freeze);archive=Path(a.output);archive.mkdir(parents=True,exist_ok=False)
    prior=ROOT/'experiments/agent3-b1-small-budget-a2';package=json.loads((prior/'package-files.json').read_text());model_backups=[]
    source_root=ROOT.parent/'agent3-b1-small-budget-a2/outputs/agent3-b1-small-budget-a2/assets/package'
    secondary=Path('/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012')
    for rel,meta in package['files'].items():
        source={'path':str(source_root/rel),**meta};target={'path':str(secondary/rel),**meta}
        verify(source);verify(target);model_backups.append({'source':source,'backup':target})
    old=json.loads((prior/'scientific-byte-backup.json').read_text())
    lens=next(x for x in old['files'] if x['source']['path'].endswith('/assets/public_a1_lens.pt'))
    verify(lens['source']);verify(lens['backup']);model_backups.append(lens)
    write_json(EV/'reverified-model-backups.json',{'verified_utc':utc(),'package_manifest':binding(prior/'package-files.json'),'actual_bytes_verified':model_backups})
    backup=Path('/mnt/c/Users/alanz/Token-Reconstruction-Backups/agent3-static-prefix-hybrid/confirmation-r1')
    files=[p for folder in ('inputs','predictions','sources-r1') for p in (OUT/folder).rglob('*') if p.is_file()]
    needed=sum(p.stat().st_size for p in files)
    if shutil.disk_usage(backup.parent).free-needed<10*2**30:raise RuntimeError('backup safety margin below10GiB')
    backup.mkdir(parents=True,exist_ok=False);backups=[]
    for source in files:
        target=backup/source.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        x,y=binding(source),binding(target)
        if x['sha256']!=y['sha256']:raise ValueError('backup differs')
        backups.append({'source':x,'backup':y})
    inventory=[]
    for c in f['cells']:
        r=json.loads(verify(c['receipt']).read_text())
        dest=EV/'frozen-receipts'/f"{c['domain']}-{c['stage']}.json";dest.parent.mkdir(exist_ok=True);shutil.copy2(c['receipt']['path'],dest)
        for m in r['methods']:
            source=verify(m['artifact']);target=archive/f"{c['domain']}-{c['stage']}-{m['method']}.safetensors.gz"
            with source.open('rb') as input_,target.open('xb') as output:
                with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0,compresslevel=6) as zipped:shutil.copyfileobj(input_,zipped)
            h=hashlib.sha256()
            with gzip.open(target,'rb') as input_:
                for block in iter(lambda:input_.read(1<<20),b''):h.update(block)
            if h.hexdigest()!=m['artifact']['sha256']:raise ValueError('archive roundtrip differs')
            inventory.append({'domain':c['domain'],'stage':c['stage'],'method':m['method'],'original':m['artifact'],'compressed':binding(target),'roundtrip_verified':True})
    for name in ('implementation-freeze.json','freeze.json','source-panel.json','plan.md'):shutil.copy2(EV/name,backup/name)
    shutil.copytree(ROOT/'scripts/agent3_hybrid',backup/'code',ignore=shutil.ignore_patterns('__pycache__'))
    write_json(archive/'inventory.json',{'freeze':binding(a.freeze),'files':inventory,'compressed_bytes':sum(x['compressed']['bytes'] for x in inventory),'actual_byte_backups':backups,'elapsed_seconds':time.monotonic()-start,'truth_opened':False,'completed_utc':utc()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',required=True);p.add_argument('--output',required=True);main(p.parse_args())
