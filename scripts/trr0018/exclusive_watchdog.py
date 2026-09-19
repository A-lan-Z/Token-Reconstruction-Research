"""Wait for exclusive compute access, then monitor only this task's child."""
import argparse,json,os,signal,subprocess,time,resource
from pathlib import Path
import psutil
p=argparse.ArgumentParser()
p.add_argument("--receipt",required=True);p.add_argument("--timeout",type=float,default=1800)
p.add_argument("--wait-timeout",type=float,default=7200)
p.add_argument("command",nargs=argparse.REMAINDER)
a=p.parse_args();dest=Path(a.receipt)
if dest.exists():raise SystemExit("create-only receipt exists")
command=a.command[1:] if a.command[0]=="--" else a.command
def gpu():
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=memory.free,temperature.gpu","--format=csv,noheader,nounits"],text=True).strip().split(",")
    free,temp=map(lambda x:int(x.strip()),raw)
    pp=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip()
    pids={int(x.strip()) for x in pp.splitlines() if x.strip().isdigit()}
    return free,temp,pids
waiting=time.time();queue=[];safe=0
while safe<3:
    free,temp,pids=gpu();now=time.time()
    queue.append({"elapsed":now-waiting,"free_mib":free,"temperature":temp,"compute_pids":sorted(pids)})
    safe=safe+1 if not pids and free>=10000 and temp<70 else 0
    if len(queue)==1 or len(queue)%5==0:print("GPU queue:",round(now-waiting),"s; other compute processes",sorted(pids),"free",free,"MiB",flush=True)
    if now-waiting>a.wait_timeout:raise SystemExit("exclusive GPU wait timeout")
    if safe<3:time.sleep(10)
print("Exclusive GPU preflight passed; launching",command,flush=True)
start=time.time();samples=[];failure=None
proc=subprocess.Popen(command,start_new_session=True)
try:
    while proc.poll() is None:
        parent=psutil.Process(proc.pid);children=parent.children(recursive=True)
        allowed={proc.pid}|{c.pid for c in children}
        rss=sum(x.memory_info().rss for x in [parent]+children if x.is_running())
        available=psutil.virtual_memory().available
        free,temp,pids=gpu()
        samples.append({"elapsed":time.time()-start,"rss_bytes":rss,"host_available_bytes":available,
                        "gpu_free_mib":free,"temperature_c":temp,"compute_pids":sorted(pids)})
        if pids-allowed:raise RuntimeError("competing GPU compute process appeared: "+str(sorted(pids-allowed)))
        if rss>10*2**30 or available<8*2**30 or free<2048 or temp>=80 or time.time()-start>a.timeout:
            raise RuntimeError("resource or timeout limit exceeded")
        time.sleep(2)
except Exception as exc:
    failure=repr(exc)
    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
rc=proc.wait()
with dest.open("x") as f:json.dump({"command":command,"queued_unix":waiting,"queue":queue,
     "start_unix":start,"end_unix":time.time(),"returncode":rc,"failure":failure,"samples":samples,
     "max_rss_bytes":max((s["rss_bytes"] for s in samples),default=0),
     "child_peak_rss_bytes":resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024},f,indent=2)
raise SystemExit(rc or (1 if failure else 0))

