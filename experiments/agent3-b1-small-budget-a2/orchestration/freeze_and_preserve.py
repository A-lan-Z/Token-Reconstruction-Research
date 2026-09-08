from pathlib import Path
import json,subprocess,datetime,time,os
root=Path.cwd();ev=Path('experiments/agent3-b1-small-budget-a2');out=Path('outputs/agent3-b1-small-budget-a2')
cells=[f'{d}-{s}' for d in ('pile','finance') for s in (0,64,128,256)]
for c in cells:
 r=json.loads((ev/'execution'/c/'finish.json').read_text());assert r['status']=='PASS'
start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic()
v=subprocess.run(['python3','-m','pytest','-q','tests/agent3_shortlists'],capture_output=True,text=True)
record={'command':['python3','-m','pytest','-q','tests/agent3_shortlists'],'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'returncode':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'scope':'includes evaluator refusal before source access; all synthetic fixtures'}
(ev/'validation-r3.json').write_text(json.dumps(record,indent=2)+'\n');print(v.stdout,flush=True);assert v.returncode==0
subprocess.run(['git','add','scripts/agent3_shortlists','tests/agent3_shortlists',str(ev)],check=True)
subprocess.run(['git','commit','-m','Complete guarded inference matrix and bind evaluator before truth'],check=True)
env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
commands=[
 ['python3','-m','scripts.agent3_shortlists.evaluate','freeze','--receipts',*[str(out/'predictions'/c/'receipt.json') for c in cells],'--output',str(ev/'freeze.json')],
 ['python3','-m','scripts.agent3_shortlists.drift','--contracts',*[str(out/'inputs'/c/'contract.json') for c in cells],'--output',str(ev/'boundary-drift.json')],
 ['python3','-m','scripts.agent3_shortlists.archive','--freeze',str(ev/'freeze.json'),'--output',str(ev/'frozen-candidates')],
 ['python3',str(ev/'orchestration/preserve.py')],
]
phases=[]
for cmd in commands:
 print('START '+cmd[2] if cmd[1]=='-m' else 'START backup',flush=True)
 t=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc).isoformat();subprocess.run(cmd,check=True,env=env)
 phases.append({'command':cmd,'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-t,'returncode':0})
 print('PASS '+str(phases[-1]['elapsed_seconds']),flush=True)
(ev/'freeze-preservation-execution.json').write_text(json.dumps({'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'phases':phases,'truth_opened':False},indent=2)+'\n')
print('ALL CANDIDATES FROZEN AND BACKED UP BEFORE TRUTH',flush=True)
