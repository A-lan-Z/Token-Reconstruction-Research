"""Package and describe completed TRR-0015 evidence; never changes predictions."""
from pathlib import Path
import sys, json, hashlib, time, zipfile, statistics, datetime, subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/trr0015"))
from support import *
sys.path.insert(0,str(ROOT/"scripts/trr0015"))
from score import evaluator_truth

def pack(paths,path):
    members=[]
    with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:
            rel=str(p.relative_to(ROOT))
            z.write(p,rel)
            members.append({"path":rel,"bytes":p.stat().st_size,"sha256":digest(p)})
    # Verify archived bytes, not just the uncompressed originals.
    with zipfile.ZipFile(path) as z:
        for member in members:
            if hashlib.sha256(z.read(member["path"])).hexdigest()!=member["sha256"]:
                raise RuntimeError("archive member mismatch")
    return {"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":digest(path),"members":members}

def main():
    score=json.loads((X/"score.json").read_text())
    matrix=json.loads((X/"canonical_matrix.json").read_text())
    receipt=json.loads((X/"prediction_receipt.json").read_text())
    guard=json.loads((X/"prediction_guard.json").read_text())
    qualification=json.loads((X/"qualification.json").read_text())
    if guard["returncode"] or guard["failure"] or len(receipt["entries"])!=816 or not matrix["complete"]:
        raise RuntimeError("incomplete run")
    check_binding(receipt["binding"])
    rows=json.loads((OUT/"metadata.json").read_text())
    truth,truth_hashes=evaluator_truth(rows)
    assert truth_hashes==score["truth_hashes"]
    truth_path=OUT/"evaluator_truth.safetensors"
    save_file(truth,str(truth_path),metadata={"truth_opened_after":"experiments/TRR-0015/truth_gate.json"})
    archive_dir=X/"evidence";archive_dir.mkdir(exist_ok=False)
    archives=[pack([OUT/"observations.safetensors",OUT/"metadata.json",truth_path],archive_dir/"inputs-and-evaluator-labels.zip")]
    for start in range(0,len(rows),16):
        part=rows[start:start+16]
        paths=[]
        for row in part:
            for method in METHODS:
                name=row["id"]+"__"+method
                paths.extend([OUT/"predictions"/(name+".safetensors"),OUT/"receipts"/(name+".json")])
        archives.append(pack(paths,archive_dir/f"predictions-{start:03d}-{start+len(part)-1:03d}.zip"))
    archive_index={"task_id":"TRR-0015","archives":archives,"total_archive_bytes":sum(a["bytes"] for a in archives),
                   "member_count":sum(len(a["members"]) for a in archives),"every_member_hash_verified":True}
    write(X/"artifact_index.json",archive_index)
    setups=default_order=[
        ("original_r2","matched","natural","Original prose / matched"),
        ("original_r2","lora256","natural","Original prose / LoRA"),
        ("original_r2","matched","stress","Original identifiers / matched"),
        ("original_r2","lora256","stress","Original identifiers / LoRA"),
        (CANONICAL[0],"clean_pile_lora_cut4","canonical","Canonical Pile / rank4 LoRA"),
        (CANONICAL[1],"historical_finance_cut4","canonical","Canonical Finance / generation300"),
    ]
    lookup={(r["setup_id"],r["condition"],r["group"],r["method"]):r for r in score["summary"]}
    quality=["| Setup | Fragment K256 | A1+A2 K256 | Fragment K512 |","|---|---:|---:|---:|"]
    costs=["| Setup | Fragment K256 (s/input) | A1+A2 K256 | Fragment K512 | K256 / baseline |","|---|---:|---:|---:|---:|"]
    for setup,condition,group,label in setups:
        rr=[lookup[(setup,condition,group,m)] for m in ("fragment256","a1a2","fragment512")]
        quality.append("| "+label+" | "+" | ".join(f"{100*r['accuracy']:.4f}%; {r['exact']}/{r['records']} exact" for r in rr)+" |")
        costs.append("| "+label+" | "+" | ".join(f"{r['mean_record_seconds']:.4f}" for r in rr)+f" | {rr[0]['seconds']/rr[1]['seconds']:.3f}x |")
    rationale=("K512 was selected during TRR-0014 on opened R1 candidate recall: K256 omitted12/2032 matched prose tokens and8/2032 shifted prose tokens, whereas K512 included all. It was a search for a promising operating point, not an equal-budget baseline comparison. This follow-up closes that specific gap with full causal decoding.")
    build=statistics.median(receipt["setup"]["prefix_rebuild_seconds"])
    cache_bytes=sum(receipt["setup"]["cache"][k] for k in ("transform_bytes","embedding_cache_bytes","intrinsic_cache_bytes"))
    report=f"""# TRR-0015: equal-budget prefix-fragment reconstruction

{rationale}

At K256, cached reconstruction time is within about2.4% of the baseline across these panels. Canonical token scores are close and slightly higher as point estimates, but their paired uncertainty intervals overlap zero. On each original prose condition, K256 makes13 errors versus1 for A1+A2, and exact inputs fall from31/32 to21/32. All13 prose errors are candidate omissions; A2 succeeds whenever the true token is included.

## Results
Every cell uses the same current-prefix A2, its own reconstructed history and the declared candidate order. Fragment K256 is exactly the first256 proposal slots of the unchanged K512 mechanism, including padding. Only the cap changes; no new fit or retuning.
All816 predictions (272 inputs x3 methods) were frozen before the scorer loaded retrospective labels. Three byte-identical repeats per cell; original-panel A1 and K512 anchors reproduced.

{"\n".join(quality)}

Percentages are post-BOS token accuracy; exact counts require every scored token in the input to match. Detailed denominators, proposal recall, conditional selector errors, decoded-text exactness and per-record results are in score.json. Source-string recovery is not equated to decoded-token recovery.

## Runtime
{"\n".join(costs)}

These are synchronized medians over three runs, averaged across inputs. Ratios compare reruns on the same RTX5080, with tables already built. Prefix-dependent rebuild median: {build:.4f}s; derived-cache storage: {cache_bytes/1e9:.3f}GB. Add a rebuild whenever the supplied prefix changes. Static tokenizer preparation: {receipt['setup']['tokenizer_suffix_prepare_seconds']:.4f}s. Historical A1 training cost was not remeasured.
Phase breakdown and paired record bootstrap intervals are in score.json; a nonsignificant difference is not an equivalence claim.

## Comparability and limitations
All data in this follow-up were already opened in earlier research, so these are retrospective results. No current labels were used to alter candidates, decisions, timing or the method.
Both canonical setups were run for each compared method. canonical_matrix.json reports all52 active method/setup cells:48 prior completed cells carried forward with source hashes, plus4 new fragment cells. Current baseline ports are shown separately. Prior execution timings are not compared against current timings.
Canonical executions are benchmark-compatible record-batch1 ports with unchanged decision rules, removal only of right padding, and adapted serialization. They are not relabelled as exact historical batch8 executions. Native compatible-fixture checks passed. A post-run audit also confirmed every canonical baseline returned token matches the historical reference. Its candidate arrays are numerically different under the declared record-batch1 geometry:61 Pile positions have reordered candidates (3 different sets), and989 Finance positions have reordered candidates (208 different sets). These are disclosed ports, not exact native candidate executions; no prediction was revised. See baseline_canonical_reproduction.json and canonical_candidate_geometry_audit.json.
All methods used a supplied public four-layer Llama3.2-1B-Instruct prefix. This does not demonstrate an actual prefix-recovery trajectory.

## Verification and resources
Seven existing focused tests passed. Largest128-position/K512 qualification passed, with byte-identical native K512 and baseline compatible-fixture outputs. Candidate-subset identity was checked on every input. Freeze gates checked matrix completeness, code/input/asset/output hashes, geometries, ID ranges, finite scores and actual argmax decisions; missing-cell, altered-output and altered-code negative cases were rejected.
Full-run GPU peak allocated: {receipt['peak_allocated_bytes']/2**30:.3f}GiB; reserved: {receipt['peak_reserved_bytes']/2**30:.3f}GiB. Watchdog exit0, minimum sampled freeGPU: {min(s['gpu_free_mib'] for s in guard['samples'])}MiB, maximum temperature: {max(s['temperature_c'] for s in guard['samples'])}C.
Scientific execution commit: {receipt['setup']['environment']['commit']}.
Exact command: env PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0015/prediction_guard.json --timeout 3600 -- python3 scripts/trr0015/predict.py
No scientific execution failed or was excluded in this follow-up.

## Artifacts and publication
- experiments/TRR-0015/PLAN.md and registry.json
- experiments/TRR-0015/score.json and canonical_matrix.json
- experiments/TRR-0015/prediction_receipt.json and truth_gate.json
- experiments/TRR-0015/manifest.json and artifact_index.json
- {len(archives)} local evidence ZIPs; {archive_index['total_archive_bytes']:,}bytes; every archived member hash verified.
- Raw inputs, evaluator labels and predictions remain under outputs/TRR-0015.

The earlier automatic-approval public-publication block remains in force. This task's request authorized testing, not public release of the prior or new evidence payload. No push or PR creation was attempted.
"""
    result_path=ROOT/"coordination/results/TRR-0015.md"
    result_path.write_text(report)
    manifest={
        "task_id":"TRR-0015","branch":"task/TRR-0015","status":"COMPLETE_LOCAL_RETROSPECTIVE_EQUAL_BUDGET_STUDY",
        "request_path":"coordination/requests/TRR-0015.md","result_path":str(result_path.relative_to(ROOT)),
        "scientific_execution_commit":receipt["setup"]["environment"]["commit"],
        "methods":list(METHODS),"candidate_budgets":BUDGETS,"observations":272,"prediction_cells":816,"repetitions":3,
        "canonical_comparison_complete":True,"canonical_required_cells":52,"canonical_inherited_cells":48,"canonical_new_cells":4,
        "summary":score["summary"],"paired":score["paired"],"preparation":receipt["setup"],
        "binding":receipt["binding"],"truth_hashes":truth_hashes,
        "scoring_seed":1515256,"bootstrap_draws":10000,"model_fitting_steps":0,"adaptation_seconds":0,
        "actual_recovered_prefix_used":False,"truth_status":"RETROSPECTIVE_ALL_CURRENT_OUTPUTS_FROZEN_BEFORE_SCORING",
        "hardware_and_dependencies":receipt["setup"]["environment"],"peak_allocated_bytes":receipt["peak_allocated_bytes"],
        "peak_reserved_bytes":receipt["peak_reserved_bytes"],"peak_host_rss_bytes":receipt["peak_host_rss_bytes"],
        "prediction_start_unix":guard["start_unix"],"prediction_end_unix":guard["end_unix"],
        "prediction_wall_seconds":guard["end_unix"]-guard["start_unix"],
        "guard_command":guard["command"],"guard_failure":guard["failure"],"failed_or_excluded_scientific_runs":[],
        "canonical_ports":"unchanged algorithms, record batch1, right-padding removal, serialization only",
        "source_string_recovery":"unavailable/not defined; decoded-token exactness reported separately",
        "evidence_archive_count":len(archives),"evidence_archive_bytes":archive_index["total_archive_bytes"],
        "artifact_index_sha256":digest(X/"artifact_index.json"),"all_archive_members_verified":True,
        "publication_status":"BLOCKED_BY_PRIOR_AUTOMATIC_APPROVAL_REVIEW; NO_PUSH_ATTEMPT",
        "delivery_script_sha256":digest(Path(__file__)),
    }
    manifest["receipts"]={str(p.relative_to(ROOT)):digest(p) for p in sorted(X.glob("*.json"))}
    write(X/"manifest.json",manifest)
    state={
        "protocol":"TRR-RELAY/1.0","active_task":"TRR-0015","branch":"task/TRR-0015",
        "status":"RESEARCH_COMPLETE_LOCAL_PUBLICATION_AWAITING_EXPLICIT_APPROVAL",
        "updated_utc":utc(),"request_path":"coordination/requests/TRR-0015.md",
        "result_path":str(result_path.relative_to(ROOT)),"manifest_path":"experiments/TRR-0015/manifest.json",
        "execution_commit":manifest["scientific_execution_commit"],"canonical_comparison_complete":True,
        "canonical_matrix_path":"experiments/TRR-0015/canonical_matrix.json",
        "canonical_registry_path":"experiments/TRR-0015/registry.json","canonical_required_cells":52,
        "score_status":"RETROSPECTIVE_FULL816_CELL_MATRIX_FROZEN_BEFORE_SCORING",
        "recovered_prefix_integration":"NOT_RUN_SUPPLIED_PUBLIC_PREFIX",
        "publication_status":"PRIOR_AUTOMATIC_APPROVAL_REVIEW_BLOCK_PERSISTS",
    }
    (ROOT/"coordination/STATE.json").write_text(json.dumps(state,indent=2)+"\n")
    draft=f"""Compare prefix-fragment reconstruction at the baseline's256-candidate budget

The previous no-A1 result used512 candidates while the baseline used256. This change exposes the candidate budget without changing the original default, then tests fragment256, fragment512 and the fixed A1+A2 K256 comparator on the same272 inputs, including both canonical setups.

The816-cell matrix freezes before scorer access to retrospective labels; every cell has three byte-identical repetitions. Original-panel control outputs reproduce, candidate ordering is unchanged, native-fixture checks and seven focused tests pass. All52 registered canonical cells are reported, with48 inherited frozen results distinguished from new executions.

Result: coordination/results/TRR-0015.md
Manifest: experiments/TRR-0015/manifest.json
Local evidence: experiments/TRR-0015/artifact_index.json

Public release remains subject to the prior explicit-approval block. No PR has been created.
"""
    (X/"DRAFT_PR.md").write_text(draft)
    print(json.dumps({"archives":len(archives),"bytes":archive_index["total_archive_bytes"],
                      "result":str(result_path),"manifest":str(X/"manifest.json")},indent=2),flush=True)
if __name__=="__main__":main()
