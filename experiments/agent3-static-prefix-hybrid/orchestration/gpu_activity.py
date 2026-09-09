import subprocess,time,json
from pathlib import Path
from datetime import datetime,timezone
out=Path('experiments/agent3-static-prefix-hybrid/gpu-activity.jsonl');stop=out.with_suffix('.stop')
with out.open('x') as stream:
 while not stop.exists():
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True,timeout=5).strip().splitlines()
  stream.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'compute_apps':apps})+'\n');stream.flush();time.sleep(2)
