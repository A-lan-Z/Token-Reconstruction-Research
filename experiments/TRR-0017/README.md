# TRR-0017 reproduction

Run from a fresh worktree at the scientific execution commit reported in manifest.json. Output and guard receipts are create-only. Retain existing evidence; use a fresh worktree for a complete rerun.

Required sibling assets:

- ../TRR-0015/outputs/TRR-0015/{observations.safetensors,metadata.json,evaluator_truth.safetensors,predictions/}; hashes and original archives are recorded by TRR-0015.
- ../agent4-prefix-only-inversion/outputs/agent4-prefix-only-inversion/backup/: pinned public four-layer Llama prefix, tokenizer and public A1 comparator asset. The exact prefix hash and model revision are in manifest.json.
- Development-only reproduction additionally needs ../TRR-0014/outputs/TRR-0014/fresh_r1/.

The evaluator truth file is used only by benchmark_score.py after the entire matrix passes its freeze gate. Prediction programs never import or read it. These existing records were already opened historically, so results remain retrospective.

Dependencies and GPU driver versions are in environment_versions.json. The implementation was tested on Python3.12.3, PyTorch2.10.0+cu128, Transformers5.3.0 and RTX5080. Scientific comparison retains BF16 native A2 and FP32 RoPE; the continuous inverse explicitly uses an FP32 copy of the same weights.

Commands, from the worktree root:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src python3 -m pytest -q tests/test_trr0017_numerics.py tests/test_prefix_fragments.py tests/test_prefix_weight_metric.py
bash scripts/trr0017/run_benchmark.sh
python3 scripts/trr0017_delivery.py
```

The first group waits for exclusive CUDA compute access and >=10000MiB free memory. Every group qualifies the largest128-position record with three identical outputs, then writes independent prediction/receipt pairs. The exclusive watchdog stops only its child if another GPU workload appears. Resume an interrupted group with the same frozen source and a new guard-receipt filename; completed cell receipts must retain the same scientific binding. Unpaired orphan outputs fail closed and must be preserved and investigated before retrying.

The shared-context qualification has a separate script, shared_qualify.py. Its initial failed harness receipt and the corrected successful receipt are both retained. Numerical changes to batch size, cache semantics or candidate order require new equivalence qualification; this task qualified K256 only.

Raw outputs and logs are packaged in bounded ZIP parts under artifacts/ by the delivery script. artifact_index.json contains each archive hash and every member hash; delivery verifies all members before generating the final report and manifest. Publication was not attempted because the earlier automatic approval-review block remains unresolved.
