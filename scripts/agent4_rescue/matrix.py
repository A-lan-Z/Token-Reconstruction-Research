"""Restart-safe bounded sequential matrix; every arm freezes before scoring."""
from shared import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--methods',nargs='+',default=['original','cached','discrete','inverse','a1a2']);a=p.parse_args()
for m in a.methods:
    name=a.panel+'_'+m
    if (E/name).exists():raise RuntimeError('immutable output exists; do not overwrite')
    cmd=[sys.executable,str(ROOT/'scripts/agent4_rescue/run.py'),'--panel',a.panel,'--method',m,'--name',name]
    print('Starting',a.panel,m,flush=True);subprocess.run(cmd,check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/agent4_rescue/score.py'),'--panel',a.panel,'--runs',*[a.panel+'_'+m for m in a.methods],'--name',a.panel+'_score'],check=True,cwd=ROOT)
