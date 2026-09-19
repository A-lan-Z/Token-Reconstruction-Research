"""Assemble old sanitized and freshly captured observations; never read labels."""
from support import *

def main():
    old=ROOT.parent/'TRR-0016/outputs/TRR-0016';fresh=OUT/'fresh_r4'
    meta=json.loads((old/'metadata.json').read_text());observations=load_file(str(old/'observations.safetensors'))
    fresh_meta=json.loads((fresh/'metadata.json').read_text());fresh_obs=load_file(str(fresh/'observations.safetensors'))
    for row in fresh_meta:
        key='r4__'+row['id'];meta.append({**row,'id':key,'original_id':row['id'],'setup_id':'fresh_r4'})
        observations[key]=fresh_obs[row['id']]
    assert len(meta)==432 and len(observations)==432
    save_file(observations,str(OUT/'observations.safetensors'),metadata={'truth_included':'false','task_id':'TRR-0018'})
    write(OUT/'metadata.json',meta)
    paths=[old/'metadata.json',old/'observations.safetensors',fresh/'metadata.json',fresh/'observations.safetensors']
    write(X/'preparation.json',{'utc':utc(),'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'input_hashes':{str(p):digest(p) for p in paths},'observations':432,'prediction_cells':1296,'truth_read':False,
        'output_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [OUT/'observations.safetensors',OUT/'metadata.json']},
        'port':'Same single-record packing as TRR-0015; no candidate microbatching; native intrinsic-table batch256',
        'live_gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,temperature.gpu','--format=csv'],text=True),
        'host_meminfo':Path('/proc/meminfo').read_text(),'planned_resources':'PLAN.md; largest expected5.574GiB reserved, guard6GiB; timeout5400s'})
    print('Prepared432 truthless observations;60 canonical cells registered',flush=True)
if __name__=='__main__':main()