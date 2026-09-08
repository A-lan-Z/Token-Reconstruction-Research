import subprocess,json,time
from pathlib import Path
root=Path.cwd(); task='agent3-b1-small-budget-a2'; start=time.time()
for domain in ('pile','finance'):
 for stage in (0,64,128,256):
  cell=f'{domain}-{stage}'
  command=['python3','-m','scripts.agent3_shortlists.watchdog','--output',f'experiments/{task}/execution/{cell}','--','python3','-m','scripts.agent3_shortlists.predict','--contract',f'outputs/{task}/inputs/{cell}/contract.json','--package',f'outputs/{task}/assets/package','--lens',f'outputs/{task}/assets/public_a1_lens.pt','--reference','reference/strict_bos/round001_teacher.py','--output',f'outputs/{task}/predictions/{cell}','--device','cpu']
  print('START '+cell,flush=True)
  subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
  r=json.loads((root/f'experiments/{task}/execution/{cell}/finish.json').read_text())
  print(json.dumps({'cell':cell,'status':r['status'],'seconds':r['elapsed_seconds'],'peak_rss_bytes':r['peak_rss_bytes']}),flush=True)
print('MATRIX COMPLETE seconds='+str(time.time()-start),flush=True)
