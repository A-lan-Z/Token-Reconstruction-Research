"""Local evidence packaging and handoff after the full TRR-0017 score exists."""
from pathlib import Path
import sys,json,time,hashlib,zipfile,subprocess,copy
ROOT=Path(__file__).resolve().parents[1];X=ROOT/"experiments/TRR-0017";OUT=ROOT/"outputs/TRR-0017"
def digest(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()
def write(p,value):
    with Path(p).open("x") as f:json.dump(value,f,indent=2)
def main():
    score=json.loads((X/"score.json").read_text());matrix=json.loads((X/"canonical_matrix.json").read_text())
    assert matrix["complete"] and matrix["required_cells"]==56
    gate=json.loads((X/"truth_gate.json").read_text());assert gate["verified_cells"]==1360
    freezes={g:json.loads((X/(g+"_freeze.json")).read_text()) for g in ("controls","continuous","discrete")}
    assert all(f["binding"]==freezes["controls"]["binding"] for f in freezes.values())
    for rel,sha in freezes["controls"]["binding"]["source_hashes"].items():assert digest(ROOT/rel)==sha
    for f in freezes.values():
        for e in f["entries"]:assert digest(ROOT/e["path"])==e["sha256"]
    # All primary raw files plus compact logs are archived in bounded chunks.
    files=sorted(p for p in OUT.rglob("*") if p.is_file())
    files+=sorted(X.glob("*.log"))
    artifact_dir=X/"artifacts";artifact_dir.mkdir(exist_ok=False)
    chunks=[];batch=[];size=0
    for p in files:
        n=p.stat().st_size
        if batch and size+n>35*2**20:chunks.append(batch);batch=[];size=0
        batch.append(p);size+=n
    if batch:chunks.append(batch)
    archives=[];members=[]
    for i,batch in enumerate(chunks):
        path=artifact_dir/f"raw-{i:02d}.zip"
        with zipfile.ZipFile(path,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in batch:
                rel=str(p.relative_to(ROOT));z.write(p,rel)
                members.append({"path":rel,"sha256":digest(p),"bytes":p.stat().st_size,
                                "archive":str(path.relative_to(ROOT))})
        archives.append({"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":digest(path)})
    bymember={m["path"]:m for m in members}
    for a in archives:
        with zipfile.ZipFile(ROOT/a["path"]) as z:
            assert z.testzip() is None
            for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==bymember[name]["sha256"]
    write(X/"artifact_index.json",{"archives":archives,"members":members,"all_member_hashes_verified":True})
    names={"a1a2":"A1+A2 K256","fragment_native":"Prefix fragment256, native A2",
           "fragment_shared":"Prefix fragment256, shared context","continuous96":"Whole-sequence continuous inverse",
           "discrete64":"Whole-sequence discrete refinement"}
    lines=["# TRR-0017: remove candidate trials or reduce A2 cost","",
           "The direct-inversion experiments did not establish a near-baseline replacement. A separate execution improvement shares the committed context across A2 candidates and preserves every tested output byte. It still uses the original candidate list.","",
           "## Completed comparison","",
           "All 1,360 outputs (five methods on 272 observations) were frozen before the retrospective scorer opened labels. The active registry contains 28 methods and 56 canonical cells: 52 frozen cells carried forward with source hashes and four new cells. Current controls reproduce every TRR-0015 token, candidate and score tensor. No scores are pooled across setups.","",
           "| Setup / panel | Method | Correct tokens | Exact inputs | Mean seconds/input |",
           "|---|---|---:|---:|---:|"]
    for r in sorted(score["summary"],key=lambda r:(r["setup_id"],r["condition"],r["group"],r["method"])):
        label=r["setup_id"] if r["group"]=="canonical" else r["condition"]+"/"+r["group"]
        lines.append(f"| {label} | {names[r['method']]} | {r['correct']}/{r['scored']} ({100*r['accuracy']:.3f}%) | {r['exact']}/{r['records']} | {r['mean_seconds']:.4f} |")
    lines+=["","## What the candidate-free mechanisms do","",
      "The continuous inverse starts with the observed vectors, rescales them to the typical embedding magnitude and adjusts the whole sequence together. It runs the same prefix with weights converted to FP32, minimizing activation mismatch using L-BFGS, then scans every vocabulary embedding once to choose the final tokens. No A1, shortlist, fitted inverse or initial token lookup is used; initialization uses only a global embedding-scale statistic. The benchmark excludes the unused Euclidean diagnostic projection from development.",
      "",
      "The discrete variant starts from a deterministic same-prefix geometric lookup, then repeatedly adjusts one complete sequence estimate. Every forward pass uses actual vocabulary embeddings; a straight-through gradient updates the underlying continuous vectors. The diagonal gradient stops earlier positions from being changed merely to explain later-position errors. There is no proposed K-sized list or separate A2 verifier, but it still runs 64 forward/backward passes through the prefix and performs full-vocabulary projection each step. Its initialization needs the prefix-derived metric cache.",
      "",
      "Neither trains model weights or an independent predictor. They optimize input vectors online. This is different from eliminating the prefix calculation altogether. Sixteen fixed development variants across layerwise quasi-Newton inversion, joint continuous optimization, periodic projection, straight-through discrete refinement and diagonal context gradients are preserved. The full development matrices were frozen before their opened-panel scores. More accurate development configurations can still be too slow; none supports a replacement claim.",
      "",
      "## Exact A2 execution improvement","",
      "Previously, each token's candidate batch deep-copied the committed cache and replicated its context row 256 times across all four layers. The new adapter references that committed context, expands it as a read-only view and constructs only the exact tensors needed by the current layer. Candidate order, BF16 arithmetic, rotary inputs, native attention/MLP calls, direct-cosine choice and the separate winning-token commit all remain unchanged.",
      "",
      "Every candidate hidden value matched during qualification. Across all 272 full observations, final tokens, all candidates, cosine scores and MSE arrays match byte for byte between native and shared execution. This reduces copying and memory movement; it does not remove 256 candidate simulations, matrix multiplications or prefix-table preparation. Other candidate widths have not been qualified by this task.",
      "",
      "| Panel | Shared/native time ratio | Paired record bootstrap 95% |",
      "|---|---:|---:|"]
    for p in score["paired"]:
        if p["method"]=="fragment_shared" and p["control"]=="fragment_native":
            label=p["setup_id"] if p["group"]=="canonical" else p["condition"]+"/"+p["group"]
            lo,hi=p["time_ratio_bootstrap95"]
            lines.append(f"| {label} | {p['time_ratio']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines+=["","## Timing and scope","",
       "Final comparison processes had exclusive CUDA compute access. Each method/input has one synchronized timed invocation; control order rotates. Each method passed three identical repeated outputs at the largest 128-token geometry. The intervals describe variation across these records and do not replace a dedicated repeated latency study. Inference includes transfer, proposal/optimization, verification and commit when used, final projection and output transfer. Asset loading, deterministic cache builds, tokenizer work and disk I/O are separate in the freeze receipts. Historical A1 training cost was not remeasured; the table compares inference with prepared assets.",
       "",
       "These are supplied-public-prefix, retrospective results. They do not test an actually recovering prefix, establish robustness to arbitrary prefix error, or prove that candidate-free reconstruction is impossible. The continuous method is an explicitly different FP32 inverse; the diagonal variant uses eager BF16 attention. Neither is mislabeled as an exact native A2 execution. Current canonical controls retain TRR-0015's disclosed record-batch1 port; that prior port differs numerically from the historical batch8 candidate arrays.",
       "",
       "The useful demonstrated result is the exact cache optimization. A cheap, accurate direct inverse remains unestablished. A further candidate-free study should address staying on the discrete embedding set and respecting causal coupling, rather than simply increasing unconstrained optimization steps.",
       "",
       "## Failures, resources and reproduction","",
       "INTERRUPTIONS.md preserves the unpopulated-cache-layer harness failure, the cross-shell launch failure and the resource-guard stop caused by a competing GPU task. That stopped attempt produced no comparison outputs. The competitor was not modified. Final groups wait for exclusive compute access and fail closed if a competing process appears. Development timing may have been contended and is not used for final performance claims.",
       "",
       "Nine focused tests passed, including independent checks that shared caches preserve native materialization and that diagonal gradients agree with the same-position native Jacobian within the declared BF16 test tolerances on a small model. Full guards, environments, source commits, phase costs, memory samples, output hashes and raw archives are included in the manifest.",
       "",
       "- Request: coordination/requests/TRR-0017.md",
       "- Frozen benchmark plan: experiments/TRR-0017/BENCHMARK_PLAN.md",
       "- Registry/matrix: experiments/TRR-0017/registry.json and canonical_matrix.json",
       "- Metrics: experiments/TRR-0017/score.json",
       "- Manifest: experiments/TRR-0017/manifest.json",
       "- Raw evidence hashes: experiments/TRR-0017/artifact_index.json",
       "- Reproduction: bash scripts/trr0017/run_benchmark.sh (create-only output paths; use a clean task-output directory)",
       "",
       "Publication remains local. The earlier automatic approval-review block on publishing the research payload to the public repository remains unresolved in this conversation. No public push or PR was attempted."]
    report=ROOT/"coordination/results/TRR-0017.md";report.write_text("\n".join(lines)+"\n")
    receipts=sorted(X.glob("*.json"))
    manifest={"task_id":"TRR-0017","completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
       "delivery_source_sha256":digest(Path(__file__)),
       "execution_commits":{g:f["environment"]["commit"] for g,f in freezes.items()},
       "command":"bash scripts/trr0017/run_benchmark.sh","scientific_binding":freezes["controls"]["binding"],
       "model":{"id":"meta-llama/Llama-3.2-1B-Instruct","revision":"9213176726f574b556790deb65791e0c5aa438b6","layers":4,
         "prefix_sha256":freezes["controls"]["binding"]["prefix_sha256"],"prefix_state":"supplied public; not recovered trajectory"},
       "input":{"source":"../TRR-0015/outputs/TRR-0015","observations":272,"prediction_cells":1360,
                "truth_access":"only after full matrix freeze; retrospective"},
       "methods":{"new":["prefix_parallel_continuous96","prefix_parallel_discrete64"],
                  "cache_optimization":"byte-identical native fragment256 execution;256 trials remain","new_method_fit_weight_updates":0,"baseline_A1":"previously fitted public asset; historical training cost not remeasured"},
       "setup_costs":{g:f["setup"] for g,f in freezes.items()},
       "memory":{g:{k:f[k] for k in ("peak_allocated","peak_reserved","peak_host_rss")} for g,f in freezes.items()},
       "receipts":{str(p.relative_to(ROOT)):digest(p) for p in receipts},
       "report":{"path":str(report.relative_to(ROOT)),"sha256":digest(report)},
       "tests":{"path":"experiments/TRR-0017/tests.log","sha256":digest(X/"tests.log"),"passed":9},
       "artifacts":{"index":"experiments/TRR-0017/artifact_index.json","sha256":digest(X/"artifact_index.json"),
                    "archives":len(archives),"bytes":sum(a["bytes"] for a in archives),"members":len(members),"verified":True},
       "canonical_comparison_complete":True,"canonical_required_cells":56,
       "publication":"LOCAL_ONLY_PRIOR_AUTOMATIC_APPROVAL_REVIEW_BLOCK"}
    write(X/"manifest.json",manifest)
    state=json.loads((ROOT/"coordination/STATE.json").read_text())
    state.update(active_task="TRR-0017",branch="task/TRR-0017",status="RESEARCH_COMPLETE_LOCAL_PUBLICATION_AWAITING_EXPLICIT_APPROVAL",
         updated_utc=manifest["completed_utc"],request_path="coordination/requests/TRR-0017.md",
         result_path="coordination/results/TRR-0017.md",manifest_path="experiments/TRR-0017/manifest.json",
         execution_commit=freezes["controls"]["environment"]["commit"],canonical_comparison_complete=True,
         canonical_matrix_path="experiments/TRR-0017/canonical_matrix.json",
         canonical_registry_path="experiments/TRR-0017/registry.json",canonical_required_cells=56,
         score_status="RETROSPECTIVE_FULL1360_MATRIX_FROZEN_BEFORE_SCORING",
         recovered_prefix_integration="NOT_RUN_SUPPLIED_PUBLIC_PREFIX",publication_status="PRIOR_AUTOMATIC_APPROVAL_REVIEW_BLOCK_PERSISTS")
    (ROOT/"coordination/STATE.json").write_text(json.dumps(state,indent=2)+"\n")
    with (ROOT/"research/DUAL_BENCHMARK_PROTOCOL.md").open("a") as f:
        f.write("\n## TRR-0017: candidate-free inversion and exact A2 cache reuse\n\n"
          "Two active methods (prefix_parallel_continuous96 and prefix_parallel_discrete64) add four cells to the 52-cell registry. Both were evaluated in both canonical setups and the paired original R2 panel; the complete 1,360 current output matrix was frozen before retrospective scoring. The56-cell matrix retains all prior provenance. Neither candidate-free variant establishes a replacement. The shared-context fragment256 execution preserves all candidates, scores and emitted tokens byte for byte on all 272 observations; it changes execution cost, not the decision rule. Details: coordination/results/TRR-0017.md and experiments/TRR-0017/manifest.json.\n")
    with (ROOT/"research/ROADMAP.md").open("a") as f:
        f.write("\n## TRR-0017: reduce A2 work\n\n"
          "Whole-sequence continuous and discrete reconstruction now have a complete dual-benchmark comparison. Tested candidate-free variants remain below the baseline and do not demonstrate a cheap direct inverse. Preserve their negative evidence without claiming impossibility. Shared committed-context execution removes redundant A2 cache copies with byte-identical outputs across 272 inputs; it still evaluates 256 candidates. Prioritize this qualified execution improvement separately from further discrete/causal inverse research. Actual recovered-prefix integration remains untested. Result: coordination/results/TRR-0017.md.\n")
    (X/"DRAFT_PR.md").write_text("# TRR-0017: quantify candidate-free inversion and reuse A2 context\n\n"
        "The existing decoder repeats its committed context for every candidate. This change supplies a shared-context execution adapter, checks every candidate output for equivalence and measures the complete reconstruction cost. It also evaluates two new whole-sequence candidate-free inversion methods and preserves unsuccessful development variants.\n\n"
        "Validation: nine numerical tests; full 1,360-cell freeze; both canonical setups;56-cell active-method matrix; byte-identical native/shared tokens, candidate IDs, cosine and MSE arrays for 272 observations; archive hashes verified.\n\n"
        "Result: coordination/results/TRR-0017.md\nManifest: experiments/TRR-0017/manifest.json\n\nLocal draft only; publication approval remains unresolved.\n")
    print(json.dumps({"report":str(report),"manifest":str(X/"manifest.json"),"artifact_bytes":manifest["artifacts"]["bytes"],
                      "archives":len(archives),"verified_cells":gate["verified_cells"]},indent=2))
if __name__=="__main__":main()
