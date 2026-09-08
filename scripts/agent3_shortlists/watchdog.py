"""Isolated CPU job with fail-closed RSS/RAM/deadline sampling and receipts."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from scripts.agent3_shortlists.core import binding,write_json


def main(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cmd=a.command
    if cmd and cmd[0]=='--':cmd=cmd[1:]
    start=time.monotonic()
    receipt={'command':cmd,'cwd':str(Path.cwd()),'start_utc':datetime.now(timezone.utc).isoformat(),
             'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             'limits':{'rss_bytes':a.rss_gib*2**30,'min_available_bytes':a.min_available_gib*2**30,'seconds':a.seconds}}
    write_json(out/'command.json',receipt)
    peak=0;failure=None
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr,(out/'resources.jsonl').open('x') as samples:
        p=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        while p.poll() is None:
            try:
                fields=Path(f'/proc/{p.pid}/status').read_text().splitlines()
                rss=next(int(l.split()[1])*1024 for l in fields if l.startswith('VmRSS:'))
                mem=Path('/proc/meminfo').read_text().splitlines()
                available=next(int(l.split()[1])*1024 for l in mem if l.startswith('MemAvailable:'))
                elapsed=time.monotonic()-start;peak=max(peak,rss)
                samples.write(json.dumps({'elapsed_seconds':elapsed,'rss_bytes':rss,'available_bytes':available})+'\n');samples.flush()
                if rss>a.rss_gib*2**30:failure='RSS_LIMIT'
                if available<a.min_available_gib*2**30:failure='AVAILABLE_RAM_LIMIT'
                if elapsed>a.seconds:failure='DEADLINE'
            except (FileNotFoundError,StopIteration):
                # Exiting process may have no resident mapping; otherwise fail closed.
                if p.poll() is None:
                    time.sleep(.05)
                    if p.poll() is None:failure='RESOURCE_SAMPLE_UNAVAILABLE'
            if failure:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(.5)
        code=p.wait()
    receipt.update(end_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
                   peak_rss_bytes=peak,returncode=code,guard_failure=failure,status='PASS' if code==0 and failure is None else 'FAILED_EXCLUDED')
    receipt['logs']=[binding(out/'stdout.txt'),binding(out/'stderr.txt'),binding(out/'resources.jsonl')]
    write_json(out/'finish.json',receipt)
    print(json.dumps(receipt),flush=True)
    if code or failure:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--rss-gib',type=float,default=4)
    p.add_argument('--min-available-gib',type=float,default=8);p.add_argument('--seconds',type=float,default=1800)
    p.add_argument('command',nargs=argparse.REMAINDER);main(p.parse_args())
