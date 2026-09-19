from benchmark_support import *
import argparse,gc,resource
from transformers import AutoTokenizer
from token_reconstruction.prefix_fragments import PrefixFragmentProposer,build_suffix_cache
from sequence_inverse import SequenceInverse
from discrete_parallel import DiscreteParallel
from shared_context import shared_candidates
import fragment_predict as inherited
from comparator import prepare as prepare_baseline

class Budget256:
    def __init__(self,proposer):self.proposer=proposer
    def propose(self,h,fragments=True):return self.proposer.propose(h,fragments=fragments,budget=256)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--group",choices=["controls","continuous","discrete"],required=True);args=ap.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    bind=binding();env=n.environment();started=utc()
    methods=METHODS[:3] if args.group=="controls" else (("continuous96",) if args.group=="continuous" else ("discrete64",))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"predictions").mkdir(exist_ok=True);(OUT/"receipts").mkdir(exist_ok=True)
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();setup={"prefix_load":time.perf_counter()-t}
    if args.group=="controls":
        assert json.loads((X/"shared_qualification.json").read_text())["status"]=="QUALIFIED_BYTE_IDENTICAL"
        t=time.perf_counter();tok=AutoTokenizer.from_pretrained(n.ASSETS/"backup",local_files_only=True)
        suffix=build_suffix_cache(tok,len(prefix.embed_tokens.weight));setup["tokenizer_and_suffix"]=time.perf_counter()-t
        n.sync();t=time.perf_counter();proposer=PrefixFragmentProposer(prefix,suffix);cache_stats=proposer.build(256);n.sync()
        setup["prefix_cache_build"]=time.perf_counter()-t;setup["cache_stats"]=cache_stats
        t=time.perf_counter();lens,emb=prepare_baseline(prefix);n.sync();setup["a1_asset_load"]=time.perf_counter()-t
        def decode(h,method):
            original=inherited.candidates
            try:
                if method=="fragment_shared":inherited.candidates=shared_candidates
                return inherited.decode(prefix,h,"a1a2" if method=="a1a2" else "fragment512",Budget256(proposer),lens,emb)
            finally:inherited.candidates=original
    elif args.group=="continuous":
        n.sync();t=time.perf_counter();engine=SequenceInverse(prefix);n.sync();setup["inverse_cache"]=time.perf_counter()-t
        def decode(h,method):
            engine.check();n.sync();t=time.perf_counter()
            target=h.to("cuda").float().unsqueeze(0)
            z,trace=engine.joint(target,96,None,"lbfgs")
            n.sync();solved=time.perf_counter()
            out=engine.project(z,"cosine").cpu();n.sync();end=time.perf_counter()
            return {"tokens":out},{"total":end-t,"input_and_solve":solved-t,
                 "projection_and_output":end-solved,
                 "continuous_sequence_forward_backward_equivalents":trace["forward_backward_evaluations"],
                 "discrete_candidate_simulations":0,"trace":trace}
    else:
        n.sync();t=time.perf_counter();engine=DiscreteParallel(prefix);n.sync();setup["inverse_cache"]=time.perf_counter()-t
        def decode(h,method):return engine.decode(h,"cosine_diagonal_64_0.003")
    n.guard()
    obs=n.load_file(str(INPUT/"observations.safetensors"));rows=json.loads((INPUT/"metadata.json").read_text())
    # Maximum declared geometry on an already available truthless observation.
    q=next(r for r in rows if r["positions"]==128);qualification={}
    for m in methods:
        runs=[];first=None
        for rep in range(3):
            result,timing=decode(obs[q["id"]],m);n.guard();runs.append(timing["total"])
            if first is None:first=result
            elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError("nonrepeatable qualifier")
        qualification[m]={"seconds":runs,"three_repetitions_identical":True,"positions":128}
    if args.group=="controls":
        a,_=decode(obs[q["id"]],"fragment_native");b,_=decode(obs[q["id"]],"fragment_shared")
        if not all(torch.equal(a[k],b[k]) for k in a):raise RuntimeError("full-record optimization not equivalent")
    write(X/(args.group+"_qualification.json"),{"binding":bind,"setup":setup,"rows":qualification,"environment":env,
         "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
    entries=[]
    for i,row in enumerate(rows):
        order=methods[i%len(methods):]+methods[:i%len(methods)]
        for m in order:
            path=OUT/"predictions"/(row["id"]+"__"+m+".safetensors")
            receipt=OUT/"receipts"/(row["id"]+"__"+m+".json")
            if receipt.exists():
                e=json.loads(receipt.read_text());assert e["binding"]==bind;verify_cell(e);entries.append(e);continue
            if path.exists():raise RuntimeError("orphan output: "+str(path))
            n.guard();result,timing=decode(obs[row["id"]],m);n.guard()
            reproduction=None
            if m in ("a1a2","fragment_native"):
                oldm="a1a2" if m=="a1a2" else "fragment256"
                old=n.load_file(str(INPUT/"predictions"/(row["id"]+"__"+oldm+".safetensors")))
                reproduction={k:bool(torch.equal(result[k],old[k])) for k in result}
                if not all(reproduction.values()):raise RuntimeError("TRR-0015 output reproduction failure")
            t=time.perf_counter();n.save_file(result,str(path));sha=digest(path);io=time.perf_counter()-t
            e={**row,"method":m,"path":str(path.relative_to(ROOT)),"sha256":sha,"binding":bind,"timing":timing,
               "io_and_hash_seconds":io,"reproduces_TRR0015":reproduction,"frozen_utc":utc()}
            verify_cell(e);write(receipt,e);entries.append(e)
        if args.group=="controls":
            a=n.load_file(str(OUT/"predictions"/(row["id"]+"__fragment_native.safetensors")))
            b=n.load_file(str(OUT/"predictions"/(row["id"]+"__fragment_shared.safetensors")))
            if not all(torch.equal(a[k],b[k]) for k in a):raise RuntimeError("shared A2 not byte-identical")
        print(args.group,f"{i+1}/{len(rows)}",row["id"],round(entries[-1]["timing"]["total"],3),flush=True)
    if bind!=binding():raise RuntimeError("binding changed during execution")
    write(X/(args.group+"_freeze.json"),{"task_id":"TRR-0017","binding":bind,"environment":env,"setup":setup,
        "started_utc":started,"ended_utc":utc(),"entries":entries,"truth_read":False,"qualification":qualification,
        "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
        "peak_host_rss":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"command":sys.argv,
        "all_shared_outputs_byte_identical":True if args.group=="controls" else None})
if __name__=="__main__":main()

