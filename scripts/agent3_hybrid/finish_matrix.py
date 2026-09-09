"""Run the predeclared evaluation phases only after the complete matrix succeeds."""
import json,os,subprocess,sys,time
from pathlib import Path
from scripts.agent3_hybrid.common import *

def main():
    matrix=EV/'matrix-execution.json'
    while not matrix.exists():time.sleep(5)
    m=json.loads(matrix.read_text())
    if m.get('status')!='ALL_EIGHT_CELLS_COMPLETE_NO_TRUTH':raise ValueError('matrix must finish successfully')
    (EV/'gpu-activity.stop').touch(exist_ok=True)
    phases=EV/'evaluation-execution';phases.mkdir(exist_ok=False)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def run(name,args):
        directory=phases/name;directory.mkdir();start=utc();t=time.monotonic();command=[sys.executable,*args]
        receipt={'phase':name,'start_utc':start,'command':command,'cwd':str(ROOT),'code_commit':commit(),'orchestrator':binding(__file__),'environment_threads':2}
        write_json(directory/'command.json',receipt)
        with (directory/'stdout.txt').open('x') as out,(directory/'stderr.txt').open('x') as err:
            p=subprocess.run(command,cwd=ROOT,env=env,stdout=out,stderr=err)
        write_json(directory/'finish.json',receipt|{'end_utc':utc(),'seconds':time.monotonic()-t,'returncode':p.returncode})
        if p.returncode:raise RuntimeError(f'{name} failed: inspect {directory}')
        print('PASS '+name,flush=True)
    freeze=EV/'freeze.json';result=EV/'results.json';truth=OUT/'private-evaluation/truth-r1'
    receipts=[str(OUT/'predictions'/f'{d}-{s}'/'receipt.json') for d in DOMAINS for s in STAGES]
    run('freeze',['-m','scripts.agent3_hybrid.evaluate','freeze','--receipts',*receipts,'--panel',str(OUT/'sources-r1/panel.json'),'--output',str(freeze)])
    run('archive',['-m','scripts.agent3_hybrid.archive','--freeze',str(freeze),'--output',str(EV/'frozen-candidates')])
    run('truth',['-m','scripts.agent3_hybrid.curator','truth','--freeze',str(freeze),'--output',str(truth)])
    run('score',['-m','scripts.agent3_hybrid.evaluate','score','--freeze',str(freeze),'--truth',str(truth/'truth.json'),'--output',str(result)])
    run('audit',['-m','scripts.agent3_hybrid.audit','--freeze',str(freeze),'--results',str(result),'--output',str(EV/'independent-audit.json')])
    run('report',['-m','scripts.agent3_hybrid.report','--results',str(result),'--output',str(ROOT/'coordination/results/agent3-static-prefix-hybrid.md')])
    run('plot',['-m','scripts.agent3_hybrid.plot','--results',str(result),'--output',str(EV/'hybrid-pilot')])
    run('resources',['-m','scripts.agent3_hybrid.resource_summary','--output',str(EV/'resource-summary.json')])
    run('tests',['-m','pytest','-q','tests/agent3_hybrid'])
    print('ALL DECLARED EVALUATION PHASES COMPLETE',flush=True)

if __name__=='__main__':main()
