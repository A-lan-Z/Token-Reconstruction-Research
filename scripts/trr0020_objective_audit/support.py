from pathlib import Path
import sys,json,os,time
ROOT=Path(__file__).resolve().parents[2]
for path in ['src','scripts/trr0014']:sys.path.insert(0,str(ROOT/path))
import native as n
torch=n.torch;torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('wrong deterministic environment')
X=ROOT/'experiments/TRR-0020/objective_audit';OUT=ROOT/'outputs/TRR-0020/objective_audit'
INPUT=ROOT.parent/'TRR-0018/outputs/TRR-0018'
METHODS={'a1_native','gini128','halfhard256','warm128white003'}

def selected():
    result=[]
    for folder,methods in [('canonical',{'gini128','halfhard256'}),('canonical_refine',{'a1_native','warm128white003'})]:
        f=json.loads((ROOT/f'experiments/TRR-0020/{folder}/prediction_freeze.json').read_text())
        result.extend(e for e in f['entries'] if e['method'] in methods and e['rep']==0)
    expected={(row['id'],m) for row in json.loads((ROOT/'experiments/TRR-0020/canonical/metadata.json').read_text()) for m in METHODS}
    if len(result)!=768 or {(e['id'],e['method']) for e in result}!=expected:raise ValueError('incomplete original output matrix')
    return sorted(result,key=lambda e:(e['id'],e['method']))

def binding():
    paths=list((ROOT/'scripts/trr0020_objective_audit').glob('*.py'))+[X/'PLAN.md',X/'preflight.json']
    paths += [ROOT/f'experiments/TRR-0020/{folder}/prediction_freeze.json' for folder in ['canonical','canonical_refine']]
    paths += [ROOT/p for p in ['scripts/trr0014/native.py','scripts/agent4/common.py','src/token_reconstruction/public_prefix.py','scripts/trr0020_resource_guard.py','experiments/TRR-0020/canonical/metadata.json']]
    return {'sources':{str(p.relative_to(ROOT)):n.digest(p) for p in sorted(paths)},'observations_sha256':n.digest(INPUT/'observations.safetensors'),'prefix_sha256':n.digest(n.ASSETS/'backup/prefix.safetensors'),'config_sha256':n.digest(n.ASSETS/'backup/config.json')}

def original(e):
    if n.digest(ROOT/e['path'])!=e['sha256']:raise ValueError('changed original prediction')
    ids=n.load_file(str(ROOT/e['path']))['tokens']
    if ids.shape!=(e['positions'],) or ids.dtype!=torch.int64 or int(ids[0])!=128000:raise ValueError('invalid original tokens')
    return ids

def gate(f):
    if f['binding']!=binding():raise ValueError('changed binding')
    expected={(e['id'],e['method']):e for e in selected()}
    if len(f['entries'])!=768 or {(e['id'],e['method']) for e in f['entries']}!=set(expected):raise ValueError('incomplete diagnostic')
    if len(f['qualification'])!=6 or not all(e['equal'] for e in f['qualification']):raise ValueError('failed repeatability')
    if {(q['length'],q['rep']) for q in f['qualification']}!={(length,rep) for length in [40,128] for rep in range(3)}:raise ValueError('incomplete public geometries')
    outputs={}
    for q in f['qualification']:
        if n.digest(ROOT/q['path'])!=q['sha256']:raise ValueError('changed qualification')
        activation=n.load_file(str(ROOT/q['path']))['activation']
        if activation.shape!=(1,q['length'],2048) or not torch.isfinite(activation).all():raise ValueError('bad qualification geometry')
    for e in f['entries']:
        key=(e['id'],e['method']);old=expected[key]
        if e['original_path']!=old['path'] or e['original_sha256']!=old['sha256']:raise ValueError('changed original binding')
        if e['binding']!=f['binding'] or n.digest(ROOT/e['path'])!=e['sha256']:raise ValueError('changed diagnostic output')
        data=n.load_file(str(ROOT/e['path']))
        if set(data)!={'tokens','cosine_error','mse'} or not torch.equal(data['tokens'],original(old)):raise ValueError('changed emitted token')
        for name in ['cosine_error','mse']:
            if data[name].shape!=(old['positions']-1,) or not torch.isfinite(data[name]).all():raise ValueError('invalid diagnostic loss')
        outputs[key]=data
    return outputs
