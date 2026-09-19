# TRR-0016 reproduction

The original result is published at https://github.com/A-lan-Z/Token-Reconstruction-Research/pull/30. This task leaves the overarching research goal unfinished.

Scientific code commit: 38419c0 (resolve the full hash from the manifest). The selected production implementation is `src/token_reconstruction/prefix_fragment_frequency.py`; exploratory context and six-ordering scripts are retained as development evidence only.

## Environment and fixed assets

Use the exact Python, PyTorch, transformers, tokenizer, public four-layer prefix, Alpaca A1 comparator and evaluator-only LoRA adapter hashes in the manifest and run receipts. The execution environment is WSL Ubuntu on an RTX5080. No model weights are included in the new evidence archives. Public-prefix and comparator assets reside in the sibling `agent4-prefix-only-inversion` worktree. The reconstructor never loads the evaluator target adapter.

The harness expects the worktree layout recorded in support.py: current TRR-0016, siblings TRR-0014 and TRR-0015, and the project root two levels above the worktree. Restore archived members to their relative `outputs/...` paths. The prior272 observations come from the already-published TRR-0015 input archive. The new fresh-panel archive supplies80 more observations. The combined observation file is deterministic and reconstructed by prepare.py; its SHA256 must match prediction_receipt.json.

## Commands

Run from this worktree with `PYTHONPATH=src`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. Use new empty output directories for a fresh replication; create-only files deliberately reject overwrites. Existing per-cell prediction receipts can resume only with an identical code/input/asset binding.

```sh
python3 scripts/trr0016/prepare.py
python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0016/qualification_guard.json --timeout 240 -- python3 scripts/trr0016/predict.py --qualify
python3 scripts/agent4/watchdog.py --receipt experiments/TRR-0016/prediction_guard.json --timeout 5400 -- python3 scripts/trr0016/predict.py
python3 scripts/trr0016_audit.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/TRR-0016/scoring_guard.json --timeout 600 -- python3 scripts/trr0016/score.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/trr0016_delivery.py
```

The archived qualification and prediction receipts are immutable evidence, not destinations for a second run. A genuine new replication requires a new task identity/output root and preregistered hashes; a replay of these inputs is retrospective. Fresh evaluator source capture was performed separately with capture.py after the method was committed. Its sources and labels were not read by the predictor. Scoring starts only after all1408 cells are frozen and verified.

The scorer's historical canonical Finance adapter expects the sibling `backdoor_lora/ersoy2026` research checkout and its cached dataset. Archived evaluator truth is also included for independent metric verification. Complete returned tokens, ordered candidates, cosine scores, MSE diagnostics and timing receipts are stored for every current cell.

## Cost interpretation

Compare inference timings only among the four current methods within each setup. Preparation and prefix-table rebuild costs are reported separately and must be added at the appropriate amortization interval. The inherited52 canonical cells establish matrix completeness; their historical timings are not compared with this run. Record batching is one, native A2 candidate batch is256 or512, intrinsic catalog construction batch is256, and its numerical geometry is unchanged.