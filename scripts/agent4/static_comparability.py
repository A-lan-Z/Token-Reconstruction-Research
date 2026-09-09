from common import *
a=load_file(str(OUT/'final/bfloat16/observations.safetensors'))
b=load_file(str(OUT/'static/bfloat16/observations.safetensors'))
m=json.loads((EVID/'final_cpu_score.json').read_text())['results']
rows=[]
for key in b:
    diff=(a[key].float()-b[key].float())[1:]
    rows.append({'id':key,'boundary_mse':float(diff.square().mean()),'relative_boundary_rms':float(diff.norm()/a[key][1:].float().norm()),'nonidentical':not torch.equal(a[key],b[key])})
matched_subset={}
for name,result in m.items():
    selected=[r for r in result['records'] if r['record_id'] in b]
    matched_subset[name]={'correct':sum(r['correct'] for r in selected),'denominator':sum(r['denominator'] for r in selected),'exact_records':sum(r['exact'] for r in selected),'records':len(selected),'seconds':sum(r['seconds'] for r in selected)}
write(EVID/'static_comparability.json',{'matched_subset':matched_subset,'boundary_drift':rows,'same_source_tokens':'same frozen source text, tokenization and16position clip; tokenizer vocabulary checked by evaluator','scope':'only the identical two-record subset supports matched-versus-static comparisons'})
print(json.dumps({'matched_subset':matched_subset,'boundary_drift':rows},indent=2))
