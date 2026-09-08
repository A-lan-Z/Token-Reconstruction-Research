"""Independent scalar verification of frozen shortlist ranks and reported counts."""
from datetime import datetime, timezone
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
from safetensors.torch import load_file
from scripts.agent3_shortlists.core import binding, verify, write_json
from scripts.agent3_shortlists.evaluate import load_frozen


def main(a):
    start=time.monotonic();f=load_frozen(a.freeze)
    r=json.loads(Path(a.results).read_text());truth=json.loads(verify(r['truth_manifest']).read_text())
    labels={d:load_file(verify(x['artifact']))['token_ids'][:,1:].tolist() for d,x in truth['domains'].items()}
    checks=[];histograms=[]
    for cell in f['cells']:
        receipt=json.loads(verify(cell['receipt']).read_text())
        for method in receipt['methods']:
            d,s,m=cell['domain'],cell['stage'],method['method']
            candidates=load_file(verify(method['artifact']))['candidates'].tolist()
            rows=[]
            for candidate_row,truth_row in zip(candidates,labels[d],strict=True):
                values=[]
                for ranked,y in zip(candidate_row,truth_row,strict=True):
                    try:rank=ranked.index(y)+1
                    except ValueError:rank=257
                    values.append(rank)
                rows.append(values)
            c=next(c for c in r['cells'] if (c['domain'],c['stage'],c['method'])==(d,s,m))
            assert rows==c['ranks_through256_else257']
            for metric in c['budgets']:
                k=metric['k'];omitted=sum(x>k for row in rows for x in row)
                complete=sum(all(x<=k for x in row) for row in rows)
                assert omitted==metric['omitted_tokens'] and complete==metric['complete_clips']
            bins=[(1,1),(2,8),(9,16),(17,32),(33,64),(65,256),(257,257)]
            histogram={f'{lo}-{hi}':sum(lo<=x<=hi for row in rows for x in row) for lo,hi in bins}
            assert sum(histogram.values())==4064
            histograms.append({'domain':d,'stage':s,'method':m,'rank_bins':histogram,'rank257_means_beyond256':True})
            checks.append({'domain':d,'stage':s,'method':m,'all4064_ranks_equal':True,'all_budget_counts_equal':True})
    write_json(a.output,{'status':'PASS_INDEPENDENT_SCALAR_AUDIT','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code':binding(__file__),'freeze':binding(a.freeze),'results':binding(a.results),'checks':checks,'rank_histograms':histograms,'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'reconstruction_outputs_changed':False})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('freeze','results','output'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
