"""Package only frozen/scored scientific artifacts; never discovers target truth."""
from native import *
import zipfile,collections
assert (X/'fresh_r2_score.json').exists(), 'score the complete frozen matrix first'
destination=X/'artifacts';destination.mkdir(exist_ok=True)
groups=collections.defaultdict(list)
# Public/generated panel evidence is released only after evaluation closes.
for panel in ['fresh_r1','fresh_r2']:
    for name in ['observations.safetensors','metadata.json','sources.json','evaluator_truth.json']:
        groups[panel+'_panel'].append(OUT/panel/name)
    for p in sorted((OUT/(panel+'_predictions')).glob('*.safetensors')):
        method=p.stem.split('__')[-1]
        groups[panel+'_'+method].append(p)
for name in ['correction_dev_r1','dictionary_metric_dev_r1','reverse_dev_r1','intrinsic_dev_r1','context_dev_r1','closure_dev_r1','geometry_dev_r1','nuisance_dev_r1','fragment_dev_r1']:
    groups['development_'+name].extend(sorted((OUT/name).glob('*.safetensors')))
entries=[]
for name,paths in sorted(groups.items()):
    target=destination/(name+'.zip')
    if target.exists():raise FileExistsError(target)
    start=time.perf_counter();members=[]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths:
            relative=str(p.relative_to(ROOT));archive.write(p,relative)
            members.append({'path':relative,'bytes':p.stat().st_size,'sha256':digest(p)})
    # ZIP CRC and exact source hash catalog are independently retained.
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist())=={v['path'] for v in members}
    if target.stat().st_size>=95_000_000:raise RuntimeError('archive too large for planned repository publication')
    entries.append({'path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':digest(target),'members':members,'seconds':time.perf_counter()-start})
write(X/'artifact_index.json',{'environment':environment(),'archives':entries,'truth_release':'only already-scored public-domain/generated R1/R2 clips','large_model_assets':'excluded; immutable IDs and hashes in manifest','derived_vocab_tables':'excluded; regenerate from supplied prefix and tokenizer'})
print('Packaged',len(entries),'archives',sum(v['bytes'] for v in entries),'bytes',flush=True)
