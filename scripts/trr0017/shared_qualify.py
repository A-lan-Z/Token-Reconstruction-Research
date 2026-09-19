from pathlib import Path
import sys,json,time,statistics
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/trr0014"));import native as n
sys.path.insert(0,str(ROOT/"scripts/trr0017"));from shared_context import shared_candidates
torch=n.torch;torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
p=n.load_prefix();n.guard();g=torch.Generator().manual_seed(170023)
fixture=torch.randint(256,128000,(1,128),generator=g).to("cuda");fixture[0,0]=128000
cache=n.new_context(p);checks=[]
for pos in range(1,128):
    if pos in (1,2,39,63,127):
        ids=torch.randint(0,128256,(256,),generator=g).to("cuda")
        ids[-16:]=ids[0]
        before=[(ly.keys.clone(),ly.values.clone()) for ly in cache.backend.layers]
        timings={"native":[],"shared":[]};out={}
        for rep in range(7):
            for name,fn in ([("native",n.candidates),("shared",shared_candidates)] if rep%2==0 else [("shared",shared_candidates),("native",n.candidates)]):
                n.sync();start=time.perf_counter();value=fn(p,cache,ids,pos);n.sync()
                timings[name].append(time.perf_counter()-start)
                if name in out and not torch.equal(out[name],value):raise RuntimeError("nonrepeatable output")
                out[name]=value
        equal=torch.equal(out["native"],out["shared"])
        untouched=cache.length==pos and all(torch.equal(a,ly.keys) and torch.equal(b,ly.values) for (a,b),ly in zip(before,cache.backend.layers))
        checks.append({"position":pos,"K":256,"byte_equal":equal,"committed_cache_unchanged":untouched,
                       "different_values":int((out["native"]!=out["shared"]).sum()),
                       "max_absolute_delta":float((out["native"]-out["shared"]).abs().max()),
                       "timings":timings,"median_seconds":{k:statistics.median(v[1:]) for k,v in timings.items()}})
        if not equal or not untouched:raise RuntimeError("shared-context optimization excluded: equivalence failure")
        n.guard()
    p.run_cached(fixture[:,pos:pos+1],cache,pos)
n.write(ROOT/"experiments/TRR-0017/shared_qualification.json",{"environment":n.environment(),"checks":checks,
        "status":"QUALIFIED_BYTE_IDENTICAL","peak_allocated":torch.cuda.max_memory_allocated(),
        "peak_reserved":torch.cuda.max_memory_reserved(),
        "source_sha256":n.digest(ROOT/"scripts/trr0017/shared_context.py")})
print(json.dumps(checks,indent=2))

