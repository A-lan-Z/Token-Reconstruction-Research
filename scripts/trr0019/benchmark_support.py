from pathlib import Path
import sys,json,time,subprocess
ROOT=Path(__file__).resolve().parents[2]
X=ROOT/'experiments/TRR-0019';OUT=ROOT/'outputs/TRR-0019';INPUT=ROOT.parent/'TRR-0018/outputs/TRR-0018'
sys.path.insert(0,str(ROOT/'scripts/trr0014'));import native as n
sys.path.insert(0,str(ROOT/'scripts/trr0017'))
sys.path.insert(0,str(ROOT/'scripts/trr0019'))
torch=n.torch;write=n.write;digest=n.digest
METHODS=('a1_native','a1_graph','mixed_native','mixed_graph','mixed_fast')
CANONICAL=('clean-pile-lora-64x40','historical-finance-strict-bos-128x128')
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def binding():
    paths=sorted((ROOT/'scripts/trr0019').glob('*.py'))+[X/'registry.json',X/'BENCHMARK_PLAN.md',X/'metadata.json']
    paths += [ROOT/p for p in ['scripts/trr0014/native.py','scripts/trr0014/fragment_predict.py','scripts/trr0017/shared_context.py',
        'scripts/agent4/comparator.py','scripts/agent4/common.py','src/token_reconstruction/public_prefix.py',
        'src/token_reconstruction/prefix_fragments.py','src/token_reconstruction/prefix_weight_metric.py',
        'src/token_reconstruction/prefix_fragment_mixed.py','src/token_reconstruction/component_crossover.py',
        'src/token_reconstruction/a1a2_configuration_search.py']]
    return {'sources':{str(p.relative_to(ROOT)):digest(p) for p in paths},
      'observations_sha256':digest(INPUT/'observations.safetensors'),'prefix_sha256':digest(n.ASSETS/'backup/prefix.safetensors'),
      'lens_sha256':digest(n.ASSETS/'backup/lens_alpaca.pt')}
def verify(entry):
    p=ROOT/entry['path']
    if digest(p)!=entry['sha256']:raise ValueError('changed output')
    data=n.load_file(str(p));t=entry['positions']
    shapes={'tokens':(t,),'candidates':(t,256),'scores':(t-1,256),'mse':(t-1,256)}
    if set(data)!=set(shapes) or any(tuple(data[k].shape)!=v for k,v in shapes.items()):raise ValueError('wrong geometry')
    if data['tokens'].dtype!=torch.int64 or data['candidates'].dtype!=torch.int64:raise ValueError('wrong IDdtype')
    if int(data['tokens'][0])!=128000:raise ValueError('wrong BOS')
    if not all(bool(torch.isfinite(data[k]).all()) for k in ['scores','mse']):raise ValueError('nonfinite')
    choice=data['candidates'][1:].gather(1,data['scores'].argmax(-1,keepdim=True)).flatten()
    if not torch.equal(choice,data['tokens'][1:]):raise ValueError('wrong decision')
    if len(entry['phases'])!=3 or not entry['repetitions_identical']:raise ValueError('repeat evidence missing')
    return data
