# Reproducing TRR-0019

Run from this worktree in Ubuntu with the pinned environment in the manifest. The frozen source expects sibling TRR-0018 observations and archived predictions, sibling public-prefix assets under agent4-prefix-only-inversion, and the task's selectedmetadata.json. Exact SHA256bindings are in prediction_freeze.json. No live unavailable target prefix is loaded by the reconstructor.

Qualification commands (create-only receipts; use a new output location for a rerun):

```bash
PYTHONPATH=src python3 scripts/trr0019/graph_probe.py
PYTHONPATH=src python3 scripts/trr0019/attention_probe.py
PYTHONPATH=src python3 scripts/trr0019/replay_probe.py
PYTHONPATH=src python3 scripts/trr0019/replay_probe.py --fast
```

Main comparison, executed under exclusive fresh resource checks:

```bash
PYTHONPATH=src python3 scripts/trr0017/exclusive_watchdog.py --receipt experiments/TRR-0019/benchmark_guard.json --timeout 7200 -- python3 scripts/trr0019/benchmark_predict.py
PYTHONPATH=src python3 scripts/trr0019/benchmark_score.py
```

Do not rerun the scorer against changed methods or incomplete predictions. It checks the complete1,360-cell freeze, code/input bindings,three-repeat evidence and output hashes before reading the existing retrospective labels. Both canonical setups are mandatory; R4 is additional previously opened paired evidence. Existing-cell resumes validate hashes and bindings, while orphan predictions fail closed. Qualification receipts are unique on each process start.

The exact replay engine is scripts/trr0019/replay_a2.py; its numerical attention variant is scripts/trr0019/shared_attention.py. Shared model storage is read on every replay. Graphs run sequentially, beginning with BOS, and must never be replayed concurrently. Returned GPU outputs are reusable buffers and must be consumed/copied before the next run, as the benchmark does.

Preparation and inference are separate phases. Report cold Triton compilation from the first probe as well as warm graph setup in the main comparison; compiled kernels can survive weight value changes, whereas lookup tables must be rebuilt. Timings require an otherwise idle GPU and the same resident assets.

For a fresh full-comparison rerun, use a new worktree at execution commit b32367c170dc19afc5880e026d537a040885713c and provide the hash-bound sibling assets/input bundle. That pre-run commit has no main prediction freeze or score files. Historical probe commands require their own recorded pre-run commits because their receipts are create-only.
