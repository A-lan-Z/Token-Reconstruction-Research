"""Freeze full K256/K512/baseline outputs without evaluator labels."""
from support import *
import argparse, resource, statistics
from transformers import AutoTokenizer
from token_reconstruction.prefix_fragments import PrefixFragmentProposer,build_suffix_cache
from token_reconstruction.prefix_fragment_frequency import FrequencyFragmentView
from token_reconstruction.prefix_fragment_mixed import MixedFragmentView
from comparator import prepare as prepare_baseline, decode as original_baseline
import fragment_predict as inherited

def decode(prefix,h,method,proposer,lens,emb):
    view=MixedFragmentView(proposer) if method=="mixed256" else FrequencyFragmentView(proposer)
    return inherited.decode(prefix,h,"a1a2" if method=="a1a2" else "fragment512",view,lens,emb)

def setup():
    started=utc();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=n.environment()
    n.sync();t=time.perf_counter();prefix=n.load_prefix();n.sync();load_seconds=time.perf_counter()-t
    tok=AutoTokenizer.from_pretrained(n.ASSETS/"backup",local_files_only=True)
    t=time.perf_counter();suffix=build_suffix_cache(tok,len(prefix.embed_tokens.weight));suffix_seconds=time.perf_counter()-t
    cache_times=[]
    for repeat in range(3):
        n.sync();t=time.perf_counter();proposer=PrefixFragmentProposer(prefix,suffix);stats=proposer.build(256);n.sync()
        cache_times.append(time.perf_counter()-t)
        if repeat<2:del proposer;torch.cuda.empty_cache()
    n.sync();t=time.perf_counter();lens,emb=prepare_baseline(prefix);n.sync();a1_seconds=time.perf_counter()-t;n.guard()
    return prefix,proposer,lens,emb,{
        "started_utc":started,"environment":env,"prefix_load_seconds":load_seconds,
        "tokenizer_suffix_prepare_seconds":suffix_seconds,"prefix_rebuild_seconds":cache_times,
        "a1_asset_setup_seconds":a1_seconds,"a1_training_cost":"not remeasured","cache":stats,
        "prefix_model":"meta-llama/Llama-3.2-1B-Instruct; first4 layers",
        "revision":"9213176726f574b556790deb65791e0c5aa438b6","dtype":"BF16 native weights; FP32 RoPE and proposal algebra",
    }

def qualify(prefix,proposer,lens,emb,setup_record,bind):
    fixture=torch.tensor([[128000]+list(range(1000,1127))],device="cuda")
    with torch.no_grad():h=prefix.forward_full(fixture)[0].cpu()
    largest={}
    for method in METHODS:
        first=None;phases=[]
        for repeat in range(3):
            result,timing=decode(prefix,h,method,proposer,lens,emb);n.guard();phases.append(timing)
            if first is None:first=result
            else:assert all(torch.equal(first[k],result[k]) for k in first)
        largest[method]={"phases":phases,"candidate_shape":list(result["candidates"].shape),"three_identical_repeats":True}
    newer,_=decode(prefix,h[:16],"a1a2",proposer,lens,emb);older=original_baseline(prefix,h[:16],lens,emb)
    assert newer["tokens"].tolist()==older["tokens"]
    assert newer["candidates"][1:].tolist()==[[v["token"] for v in rr["checks"]] for rr in older["trace"]]
    previous=ROOT.parent/"TRR-0016/outputs/TRR-0016"
    rows=json.loads((previous/"metadata.json").read_text());observations=load_file(str(previous/"observations.safetensors"))
    diagnostic=json.loads((X/"development_freeze.json").read_text())
    for row,entry in zip(rows,diagnostic['entries']):
        assert row['id']==entry['id'];path=ROOT/entry['path'];assert digest(path)==entry['sha256']
        expected=load_file(str(path))["mix64128"]
        actual=MixedFragmentView(proposer).propose(observations[row['id']]).cpu()
        assert torch.equal(actual,expected),row['id'];n.guard()
    write(X/"qualification.json",{"binding":bind,"setup":setup_record,"largest":largest,
          "all_352_selected_diagnostic_arrays_identical":True,"baseline_native_equivalence":True,
          "peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved(),
          "finished_utc":utc(),"truth_read":False})
    print("Largest128xK256 repeats and all352 production-array equivalence passed",flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--qualify",action="store_true");args=parser.parse_args()
    bind=binding();prefix,proposer,lens,emb,setup_record=setup()
    if args.qualify:
        qualify(prefix,proposer,lens,emb,setup_record,bind);return
    qualified=json.loads((X/"qualification.json").read_text())
    assert qualified["binding"]==bind
    rows=json.loads((OUT/"metadata.json").read_text());obs=load_file(str(OUT/"observations.safetensors"))
    old_root=ROOT.parent/"TRR-0016"
    old_receipt=json.loads((old_root/"experiments/TRR-0016/prediction_receipt.json").read_text())
    old_entries={(r["id"],r["method"]):r for r in old_receipt["entries"]}
    session_id=time.time_ns();session_path=X/f"sessions/predict_{session_id}.json"
    write(session_path,{"binding":bind,"setup":setup_record,"started_utc":utc(),"command":sys.argv})
    dst=OUT/"predictions";receipts=OUT/"receipts";dst.mkdir(exist_ok=True);receipts.mkdir(exist_ok=True)
    entries=[]
    for i,row in enumerate(rows):
        for method in METHODS[i%len(METHODS):]+METHODS[:i%len(METHODS)]:
            name=row["id"]+"__"+method
            receipt_path=receipts/(name+".json");path=dst/(name+".safetensors")
            if receipt_path.exists():
                entry=json.loads(receipt_path.read_text())
                assert entry["binding"]==bind
                verify_cell(entry,row);entries.append(entry);continue
            if path.exists():raise RuntimeError("orphan output; preserve and investigate "+str(path))
            n.guard();first=None;phases=[]
            for repeat in range(3):
                result,timing=decode(prefix,obs[row["id"]],method,proposer,lens,emb);phases.append(timing)
                if first is None:first=result
                elif not all(torch.equal(first[k],result[k]) for k in first):raise RuntimeError("repeated output mismatch")
            reproduces=None
            if row["setup_id"]!="fresh_r4" and method!="mixed256":
                old_entry=old_entries[(row["id"],method)]
                old_path=old_root/old_entry["path"]
                assert digest(old_path)==old_entry["sha256"]
                old=load_file(str(old_path))
                reproduces={k:bool(torch.equal(first[k],old[k])) for k in first}
                if not all(reproduces.values()):
                    raise RuntimeError("original-panel baseline/512 anchor changed")
            t=time.perf_counter();save_file(first,str(path));sha=digest(path);io_seconds=time.perf_counter()-t
            entry={**row,"method":method,"path":str(path.relative_to(ROOT)),"sha256":sha,
                   "binding":bind,"session_path":str(session_path.relative_to(ROOT)),
                   "phases":phases,"median_seconds":statistics.median(p["total"] for p in phases),
                   "io_and_hash_seconds":io_seconds,"logical_simulations":(row["positions"]-1)*BUDGETS[method],
                   "output_repetitions_byte_identical":True,"original_reproduction":reproduces,"frozen_utc":utc()}
            write(receipt_path,entry);entries.append(entry)
        print(f"{i+1}/{len(rows)} {row['id']} all three methods frozen",flush=True)
    check_binding(bind)
    write(X/"prediction_receipt.json",{
        "binding":bind,"setup":setup_record,"entries":entries,"completed_utc":utc(),
        "sessions":[str(p.relative_to(ROOT)) for p in sorted((X/"sessions").glob("*.json"))],
        "truth_read":False,"method_count":len(METHODS),"observations":len(rows),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
        "peak_host_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "command":sys.argv,
    })
    print("Frozen full1296-cell matrix",flush=True)
if __name__=="__main__":main()
