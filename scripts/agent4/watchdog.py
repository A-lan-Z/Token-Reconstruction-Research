"""Fail-closed process-group resource guard for the bounded GPU pilot phases."""
import argparse,subprocess,time,os,signal,json,sys
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--cpu-only',action='store_true');p.add_argument('--receipt',required=True);p.add_argument('--timeout',type=float,default=1800);p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
if Path(args.receipt).exists():raise SystemExit('create-only receipt already exists')
command=args.command
if command[0]=='--':command=command[1:]
start=time.time();samples=[];failure=None
proc=subprocess.Popen(command,start_new_session=True)
try:
    while proc.poll() is None:
        parent=psutil.Process(proc.pid)
        children=parent.children(recursive=True)
        rss=sum(x.memory_info().rss for x in [parent]+children if x.is_running())
        available=psutil.virtual_memory().available
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
        free,temp=[int(x.strip()) for x in gpu]
        samples.append({'elapsed':time.time()-start,'rss_bytes':rss,'host_available_bytes':available,'gpu_free_mib':free,'temperature_c':temp})
        if rss>10*2**30 or available<8*2**30 or (not args.cpu_only and (free<2048 or temp>=80)) or time.time()-start>args.timeout:
            raise RuntimeError('resource or timeout limit exceeded')
        time.sleep(2)
except Exception as exc:
    failure=repr(exc)
    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
rc=proc.wait()
with open(args.receipt,'x') as f:json.dump({'command':command,'start_unix':start,'end_unix':time.time(),'returncode':rc,'failure':failure,'samples':samples,'max_rss_bytes':max((s['rss_bytes'] for s in samples),default=0)},f,indent=2)
sys.exit(rc or (1 if failure else 0))
