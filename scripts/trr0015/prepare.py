"""Prepare only already-sanitized observations; evaluator labels are not read."""
from support import *
import argparse

def main():
    old_registry = json.loads((ROOT/"experiments/TRR-0002/preregistration/dual_benchmark_registry.v2.json").read_text())
    methods = old_registry["methods"] + [
        {"id":"a1_scale_calibrated_adaptive_causal_k32_to64","candidate_budget":64,"can_abstain":False},
        {"id":"a1_a2_exhaustive_configuration_winner","candidate_budget":256,"can_abstain":False},
        {"id":"prefix_native_fragment256","candidate_budget":256,"can_abstain":False},
        {"id":"prefix_native_fragment512","candidate_budget":512,"can_abstain":False},
    ]
    write(X/"registry.json",{
        "task_id":"TRR-0015","registered_before_execution":True,"methods":methods,
        "setups":list(CANONICAL),"required_cells":[{"setup_id":s,"method_id":m["id"]} for s in CANONICAL for m in methods],
        "inherited_cells":48,"new_cells":4,"total_cells":52,
        "current_methods":list(METHODS),"new_decision_rule":"TRR-0014 fragment rule, fixed budget256 or512; default order unchanged",
        "current_baseline_execution":"benchmark-compatible batch1 port; frozen K256 direct cosine",
        "source_protocol_sha256":digest(ROOT/"research/DUAL_BENCHMARK_PROTOCOL.md"),
        "inherited_sources":{str(p.relative_to(ROOT)):digest(p) for p in [
            ROOT/"experiments/TRR-0002/crossover/result.json",
            ROOT/"experiments/TRR-0002/calibrated-dual/result.json",
            ROOT/"experiments/TRR-0002/configuration-search/canonical/result.json"]},
    })
    meta=[];observations={}
    old_panel=PREVIOUS/"outputs/TRR-0014/fresh_r2"
    old_meta=json.loads((old_panel/"metadata.json").read_text())
    old_obs=load_file(str(old_panel/"observations.safetensors"))
    for row in old_meta:
        key="r2__"+row["id"]
        meta.append({**row,"id":key,"original_id":row["id"],"setup_id":"original_r2"})
        observations[key]=old_obs[row["id"]]
    canonical=MAIN/"outputs/TRR-0002/strict-surrogate-heavy/canonical/reconstructor_input"
    assert digest(canonical/"observations.safetensors") == "42fe0e685eed54fd779d678febf646228cb929d7d1bd3f3515cee6e582b30d61"
    data=load_file(str(canonical/"observations.safetensors"))
    config=json.loads((canonical/"config.json").read_text())
    for setup,condition,prefix in zip(CANONICAL,["clean_pile_lora_cut4","historical_finance_cut4"],["clean","finance"]):
        h=data[condition+".activations"]; mask=data[condition+".attention_mask"];positions=data[condition+".position_ids"]
        for i in range(len(h)):
            length=int(mask[i].sum())
            assert 2 <= length <= 128
            assert bool((mask[i,:length]==1).all()) and bool((mask[i,length:]==0).all())
            assert torch.equal(positions[i,:length].long(),torch.arange(length))
            key=f"{prefix}__{i:03d}"
            observations[key]=h[i,:length].contiguous()
            meta.append({"id":key,"source_id":config["opaque_record_ids"][condition][i],
                         "setup_id":setup,"condition":condition,"group":"canonical","positions":length,"source_row":i})
    assert len(meta)==272 and len(observations)==272
    OUT.mkdir(parents=True,exist_ok=True)
    save_file(observations,str(OUT/"observations.safetensors"),metadata={"truth_included":"false","task_id":"TRR-0015"})
    write(OUT/"metadata.json",meta)
    inputs=[old_panel/"metadata.json",old_panel/"observations.safetensors",canonical/"config.json",canonical/"observations.safetensors"]
    write(X/"preparation.json",{
        "utc":utc(),"execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "input_hashes":{str(p):digest(p) for p in inputs},"observations":len(meta),"prediction_cells":816,
        "truth_read":False,"output_hashes":{str(p.relative_to(ROOT)):digest(p) for p in [OUT/"observations.safetensors",OUT/"metadata.json"]},
        "port":"Separate records; trim right padding after validating contiguous masks and native position IDs; no numerical microbatching.",
        "live_gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.free,temperature.gpu","--format=csv"],text=True),
        "host_meminfo":Path("/proc/meminfo").read_text(),
        "planned_resources":"PLAN.md; prior largest reserved5.574GiB, guard6GiB; watchdog3600s",
    })
    print("Prepared272 truthless observations; registered52 canonical cells",flush=True)
if __name__=="__main__":main()
