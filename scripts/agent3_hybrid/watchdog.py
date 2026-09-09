"""Isolated GPU job guard, with create-only commands and live resource samples."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,timezone

def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cmd=a.command[1:] if a.command[0]=='--' else a.command
    t=time.monotonic();record={'command':cmd,'cwd':str(Path.cwd()),'start_utc':datetime.now(timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'limits':{'gpu_free_min_mib':3072,'host_available_min_gib':6,'rss_max_gib':12,'temperature_max_c':83,'max_seconds':a.max_seconds}}
    (out/'command.json').write_text(json.dumps(record,indent=2)+'\n');failure=None
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',TOKENIZERS_PARALLELISM='false')
    with (out/'stdout.txt').open('x') as so,(out/'stderr.txt').open('x') as se,(out/'resources.jsonl').open('x') as logs:
        process=subprocess.Popen(cmd,env=env,stdout=so,stderr=se,start_new_session=True)
        while process.poll() is None:
            try:
                rss=next(int(x.split()[1])*1024 for x in Path(f'/proc/{process.pid}/status').read_text().splitlines() if x.startswith('VmRSS:'))
                available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                free,util,temp=[int(x.strip()) for x in gpu.split(',')]
                sample={'seconds':time.monotonic()-t,'rss_bytes':rss,'available_bytes':available,'gpu_free_mib':free,'gpu_utilization':util,'temperature_c':temp};logs.write(json.dumps(sample)+'\n');logs.flush()
                if rss>12*2**30 or available<6*2**30 or free<3072 or temp>=83 or sample['seconds']>a.max_seconds:failure='RESOURCE_GUARD'
            except Exception as e:
                if process.poll() is None:failure='RESOURCE_SAMPLING_'+type(e).__name__
            if failure:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code=process.wait()
    record.update(end_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-t,returncode=code,guard_failure=failure,status='PASS' if code==0 and failure is None else 'FAILED_EXCLUDED')
    (out/'finish.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    if code or failure:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--max-seconds',type=float,default=3600);p.add_argument('command',nargs=argparse.REMAINDER);main(p.parse_args())
