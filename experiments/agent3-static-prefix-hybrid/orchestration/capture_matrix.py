import subprocess,json,time
from pathlib import Path
from datetime import datetime,timezone
root=Path.cwd();ev=Path('experiments/agent3-static-prefix-hybrid');out=Path('outputs/agent3-static-prefix-hybrid');events=[]
for d in ('pile','finance'):
 for s in (0,64,128,256):
  cell=f'{d}-{s}';cmd=['python3','-m','scripts.agent3_hybrid.watchdog','--output',str(ev/'capture-execution'/cell),'--max-seconds','600','--','python3','-m','scripts.agent3_hybrid.curator','capture','--panel',str(out/'sources-r1/panel.json'),'--domain',d,'--stage',str(s),'--output',str(out/'inputs'/cell)]
  print('CAPTURE '+cell,flush=True);t=time.monotonic();subprocess.run(cmd,check=True);events.append({'domain':d,'stage':s,'command':cmd,'elapsed_seconds':time.monotonic()-t})
(ev/'capture-matrix.json').write_text(json.dumps({'status':'COMPLETE_EIGHT_SANITIZED_OBSERVATIONS','events':events,'end_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
