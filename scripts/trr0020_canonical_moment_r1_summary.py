"""Consolidate the fixed canonical comparison after its complete truth gate."""
from pathlib import Path
from collections import defaultdict
import json,time,statistics,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
X=ROOT/"experiments/TRR-0020/canonical_moment_r1"
def digest(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    freeze=json.loads((X/"prediction_freeze.json").read_text())
    score=json.loads((X/"score.json").read_text())
    gate=json.loads((X/"truth_gate.json").read_text())
    guard=json.loads((X/"matrix_guard.json").read_text())
    archive=json.loads((X/"archive.json").read_text())
    matrix=json.loads((X/"canonical_matrix.json").read_text())
    assert guard["returncode"]==0 and guard["failure"] is None
    assert score["freeze_sha256"]==digest(X/"prediction_freeze.json")
    assert gate["replicate_cells"]==1728 and len(gate["negative_cases_rejected"])==3
    assert matrix["complete"] and matrix["required_cells"]==70
    phases=[];preparation=defaultdict(list)
    for e in freeze["phases"]:
        assert digest(ROOT/e["path"])==e["sha256"]
        d=json.loads((ROOT/e["path"]).read_text())
        item={k:d[k] for k in ["method","rep","setup","start_utc","end_utc","peak_reserved","peak_allocated","peak_rss_bytes","code_commit","command","environment"]}
        phases.append(item);preparation[d["method"]].append(d["setup"])
    preparation_ranges={}
    for method,values in preparation.items():
        numeric={key for value in values for key,v in value.items() if isinstance(v,(int,float))}
        preparation_ranges[method]={key:{"minimum":min(v[key] for v in values),"maximum":max(v[key] for v in values)} for key in sorted(numeric)}
        replay=[v["replay_setup"] for v in values if "replay_setup" in v]
        if replay:
            preparation_ranges[method]["replay_setup"]={key:{"minimum":min(v[key] for v in replay),"maximum":max(v[key] for v in replay)} for key in replay[0]}
    costs=[]
    for summary in score["summary"]:
        items=[e for e in score["per_record"] if e["setup_id"]==summary["setup_id"] and e["method"]==summary["method"]]
        keys={key for e in items for key in e["phase_seconds"]}
        costs.append({"setup_id":summary["setup_id"],"method":summary["method"],
          "mean_median_record_phases":{key:statistics.mean(e["phase_seconds"].get(key,0.) for e in items) for key in sorted(keys)}})
    physical=defaultdict(lambda:{"inference_seconds":0.,"io_hash_seconds":0.,"records":0,"forward_count":0,"backward_count":0,"candidate_simulations":0})
    for e in freeze["entries"]:
        p=physical[e["method"]];p["inference_seconds"]+=e["phases"]["total"];p["io_hash_seconds"]+=e["io_hash_seconds"];p["records"]+=1
        p["forward_count"]+=e["stats"].get("whole_sequence_prefix_forwards",0)
        p["backward_count"]+=e["stats"].get("whole_sequence_prefix_backwards",0)
        p["candidate_simulations"]+=e["stats"].get("candidate_simulations",0)
    baseline_entries=[e for e in freeze["entries"] if e["method"].startswith("a1_")]
    novel_entries=[e for e in freeze["entries"] if e["method"]=="moment64"]
    assert all(all(e["archived_equal"].values()) for e in baseline_entries)
    assert all(e["stats"]["shortlist_size"] is None and e["stats"]["separate_candidate_verification_calls"]==0 and e["stats"]["model_parameter_updates"]==0 and e["stats"]["vocabulary_entries_per_sweep"]==128256 for e in novel_entries)
    result={"task_id":"TRR-0020","method_id":"no_shortlist_bennett64_positionbest",
      "scope":score["scope"],"summary":score["summary"],"paired":score["paired"],"median_record_phase_means":costs,
      "preparation_ranges_seconds":preparation_ranges,"phase_receipts":phases,
      "physical_current_matrix_work":dict(physical),"physical_work_note":"Three full passes. Whole-prefix baseline counters are absent, not zero computation. Novel graph-capture warmup is timed separately from recorded optimization forward/backward counts.",
      "current_replicate_cells":1728,"current_unique_predictions":576,"canonical_active_methods":35,"canonical_complete_cells":70,
      "baseline_archive_exact_array_checks":len(baseline_entries)*4,
      "novel_exact_repeated_output_and_loss_trace_comparisons":192*2,
      "qualification_count":52,"public_and_development_equivalences":14,"native_lengths":38,
      "truth_gate_negative_tests":gate["negative_cases_rejected"],
      "peak_reserved_GiB_by_method":{method:max(p["peak_reserved"] for p in phases if p["method"]==method)/2**30 for method in preparation},
      "max_host_rss_bytes":guard["max_rss_bytes"],"minimum_gpu_free_mib":min(v["gpu_free_mib"] for v in guard["samples"]),
      "maximum_temperature_c":max(v["temperature_c"] for v in guard["samples"]),
      "guard_execution_seconds":guard["end_unix"]-guard["start_unix"],
      "guard_queue_seconds":guard["start_unix"]-guard["queued_unix"],
      "archive_members":archive["logical_member_count"],"archive_unique_objects":archive["unique_objects"],"archive_bytes":archive["raw_archive_bytes"],
      "freeze_sha256":digest(X/"prediction_freeze.json"),"score_sha256":digest(X/"score.json"),"archive_sha256":digest(X/"archive.json"),
      "timing_boundary":"Input initialization, required geometry capture/eviction,64updates, diagnostics,synchronization and CPU output transfer. Prefix/table preparation and disk/hash I/O reported separately; original A1 fitting not remeasured.",
      "new_method_semantics":{"vocabulary":128256,"steps":64,"prefix_forwards_per_record":66,"prefix_backwards_per_record":64,"separate_verifications":0,"model_parameter_updates":0,"shortlist":None,"readout":"best_position_error"},
      "execution_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "command":[sys.executable,*sys.argv],"created_unix":time.time()}
    with (X/"summary.json").open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ["phase_receipts","median_record_phase_means"]},indent=2))
if __name__=="__main__":main()
