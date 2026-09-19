from support import *
import argparse,os,resource,gc
from replay_a2 import ReplayA2
from discrete_soft import DiscreteSoftVocabulary
from comparator import prepare
from token_reconstruction.component_crossover import propose_public_a1
import fragment_predict as inherited

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--method",choices=METHODS,required=True);parser.add_argument("--rep",type=int,choices=range(3),required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    novel=args.method in METHOD_IDS
    torch.use_deterministic_algorithms(novel)
    wanted=":4096:8" if novel else None
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=wanted:raise RuntimeError("wrong arithmetic environment")
    bound=binding();phase=X/f"phase_{args.method}_{args.rep}.json"
    if phase.exists():
        phase_gate(json.loads(phase.read_text()),bound);print("VERIFIED_EXISTING_PHASE",args.method,args.rep,flush=True);return
    env=n.environment();started=utc();setup={}
    for p in [OUT/"predictions",OUT/"receipts",X/"qualifications"]:p.mkdir(parents=True,exist_ok=True)
    n.sync();begin=time.perf_counter();prefix=n.load_prefix();n.sync();setup["prefix_load_seconds"]=time.perf_counter()-begin
    if novel:
        n.sync();begin=time.perf_counter()
        from reuse_stream import CanonicalRefinement
        engine=CanonicalRefinement(prefix);selected="best_position_error"
        n.sync();setup["engine_and_prefix_table_seconds"]=time.perf_counter()-begin
        def decode(h):
            n.sync();begin=time.perf_counter();all_output,stats=engine.decode(h)
            stats["loss_trace"]={"soft":{k:stats["soft"][k] for k in ["loss_trace","observed_error_trace","mean_confidence_trace","mean_gini_trace"]},"direct":stats["direct"]["loss_trace"]}
            result={"tokens":all_output[selected]};n.sync();end=time.perf_counter()
            return result,{"total":end-begin,"prefix_table_initialization":stats["soft"]["input_initialization_seconds"]+stats["direct"]["input_initialization_seconds"],"capture":stats["capture_seconds"],"optimize_and_output":stats["soft"]["optimization_output_seconds"]+stats["direct"]["optimization_output_seconds"]},stats
    else:
        n.sync();begin=time.perf_counter();lens,emb=prepare(prefix);n.sync();setup["a1_asset_preparation_seconds"]=time.perf_counter()-begin
        engine=ReplayA2(prefix) if args.method=="a1_graph" else None
        if engine is not None:setup["replay_setup"]=engine.setup
        def decode(h):
            if engine is None:
                out,phases=inherited.decode(prefix,h,"a1a2",None,lens,emb)
                return out,phases,{"candidate_simulations":(len(h)-1)*256,"shortlist_size":256,"separate_candidate_verification_calls":len(h)-1}
            n.sync();begin=time.perf_counter();h=h.to("cuda").float()
            proposal=propose_public_a1(observations=h.unsqueeze(0),attention_mask=torch.ones((1,len(h)),dtype=torch.long),lens=lens,normalized_embeddings=emb)
            ids=proposal.candidates[0,:,:256].to("cuda");n.sync();proposed=time.perf_counter()
            result=engine.run(h,ids);n.sync();verified=time.perf_counter()
            out={k:v.cpu() for k,v in result.items()};n.sync();end=time.perf_counter()
            return out,{"total":end-begin,"proposal_and_input":proposed-begin,"verification_and_commit":verified-proposed,"output_transfer":end-verified},{"candidate_simulations":(len(h)-1)*256,"shortlist_size":256,"separate_candidate_verification_calls":len(h)-1}
    rows=json.loads((X/"metadata.json").read_text());data=n.load_file(str(INPUT/"observations.safetensors"))
    qrow=next(r for r in rows if r["positions"]==128);qualifications=[];reference=None;losses=None
    for rep in range(3):
        out,phases,stats=decode(data[qrow["id"]]);n.guard()
        p=X/"qualifications"/f"{args.method}_{args.rep}_{rep}_{time.time_ns()}.safetensors";n.save_file(out,str(p))
        qualifications.append({"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"phases":phases,"stats":stats})
        if reference is None:reference=out;losses=stats.get("loss_trace")
        elif not all(torch.equal(v,out[k]) for k,v in reference.items()) or losses!=stats.get("loss_trace"):raise RuntimeError("largest-case nonrepeatability; outputs retained")
    free,total=torch.cuda.mem_get_info()
    n.write(X/"qualifications"/f"receipt_{args.method}_{args.rep}_{time.time_ns()}.json",{"binding":bound,"environment":env,"row":qrow,"runs":qualifications,"setup":setup,"free_bytes":free,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated()})
    if free<2*2**30 or torch.cuda.max_memory_reserved()>6*2**30:raise RuntimeError("insufficient largest-case margin")
    print("QUALIFIED",args.method,args.rep,round(free/2**30,3),flush=True)
    entries=[]
    for i,row in enumerate(rows):
        rp=OUT/"receipts"/f'{row["id"]}__{args.method}__r{args.rep}.json'
        if rp.exists():
            e=json.loads(rp.read_text())
            if e["binding"]!=bound:raise ValueError("resume binding changed")
            verify(e);entries.append(e);continue
        n.guard();out,phases,stats=decode(data[row["id"]]);n.guard()
        archived_equal=None
        if not novel:
            prior=INPUT/"predictions"/(row["id"]+"__a1a2.safetensors")
            archived=n.load_file(str(prior));archived_equal={k:bool(torch.equal(v,archived[k])) for k,v in out.items()}
            if not all(archived_equal.values()):raise RuntimeError("baseline archived anchor changed")
        p=OUT/"predictions"/f'{row["id"]}__{args.method}.safetensors'
        io_start=time.perf_counter()
        hashes=tensor_hashes(out)
        if args.rep==0:
            if p.exists():raise RuntimeError("orphan output")
            n.save_file(out,str(p))
        else:
            first_rp=OUT/"receipts"/f'{row["id"]}__{args.method}__r0.json'
            first=json.loads(first_rp.read_text());verify(first)
            if first["tensor_sha256"]!=hashes:raise RuntimeError("replicate token/score difference")
            if novel and first["stats"]["loss_trace"]!=stats["loss_trace"]:raise RuntimeError("replicate loss difference")
        e={**row,"method":args.method,"rep":args.rep,"path":str(p.relative_to(ROOT)),"sha256":n.digest(p),"tensor_sha256":hashes,
           "binding":bound,"phases":phases,"stats":stats,"archived_equal":archived_equal,"io_hash_seconds":time.perf_counter()-io_start,"frozen_utc":utc()}
        verify(e);n.write(rp,e);entries.append(e)
        if (i+1)%8==0:print("FROZEN",args.method,args.rep,i+1,len(rows),flush=True)
    if binding()!=bound:raise RuntimeError("source changed")
    f={"task_id":"TRR-0020","method":args.method,"rep":args.rep,"binding":bound,"environment":env,"setup":setup,"entries":entries,
       "start_utc":started,"end_utc":utc(),"truth_read":False,"peak_reserved":torch.cuda.max_memory_reserved(),"peak_allocated":torch.cuda.max_memory_allocated(),
       "peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"command":sys.argv}
    phase_gate(f,bound);n.write(phase,f);print("PHASE_COMPLETE",args.method,args.rep,flush=True)
if __name__=="__main__":main()
