"""Run the eight isolated confirmation cells; never opens evaluator truth."""
import subprocess,sys,json,time
from scripts.agent3_hybrid.common import *

def main():
    check_implementation();head=commit();start=time.monotonic()
    for domain in DOMAINS:
        for stage in STAGES:
            cell=f'{domain}-{stage}';out=OUT/'predictions'/cell;watch=EV/'execution'/cell
            if out.exists() or watch.exists():raise ValueError('create-only matrix output already exists: '+cell)
            cmd=[sys.executable,'-m','scripts.agent3_hybrid.watchdog','--output',str(watch),'--max-seconds','1200','--',sys.executable,'-m','scripts.agent3_hybrid.predict','--contract',str(OUT/'inputs'/cell/'contract.json'),'--output',str(out)]
            print('START '+cell,flush=True);subprocess.run(cmd,check=True);print('COMPLETE '+cell,flush=True)
            if commit()!=head:raise ValueError('execution commit changed')
    write_json(EV/'matrix-execution.json',{'status':'ALL_EIGHT_CELLS_COMPLETE_NO_TRUTH','code_commit':head,'end_utc':utc(),'wall_seconds':time.monotonic()-start,'source_truth_opened':False})
if __name__=='__main__':main()
