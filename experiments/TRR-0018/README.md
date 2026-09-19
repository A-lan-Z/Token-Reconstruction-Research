# Reproduce TRR-0018

Use Ubuntu Python3.12.3, torch2.10.0+cu128, transformers5.3.0 and the recorded RTX5080 environment. Run from this task worktree with PYTHONPATH=src, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2. Scientific execution commit is recorded in committed_source_audit.json. Do not overwrite create-only receipts; use a fresh task-output directory for a new run.

Assets: supplied public Llama-3.2-1B-Instruct first4 layers and tokenizer at revision9213176726f574b556790deb65791e0c5aa438b6; prefix and comparator-lens hashes appear in each binding. The original asset worktree is agent4-prefix-only-inversion. Previous352 sanitized inputs and controls are from TRR-0016 (published PR31). Evaluation-only LoRA stage256 is from TRR-P12, SHA a6f13af1835743556ca80c8fed133832fb010bea2957014499740e4b0fe58b0a. The reconstructor never loads it. Target adapter configuration is rank8,alpha16,seed6012.

Development:
```
python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/development_qualification_guard.json --timeout 180 -- python3 scripts/trr0018/probe.py --qualify
python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/development_guard.json --timeout 300 -- python3 scripts/trr0018/probe.py
python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/TRR-0018/development_scoring_guard.json --timeout 300 -- python3 scripts/trr0018/score_probe.py
```

Full reconstruction:
```
python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/capture_guard.json --timeout 300 -- python3 scripts/trr0018/capture.py
python3 scripts/trr0018/prepare.py
python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/qualification_guard.json --timeout 300 -- python3 scripts/trr0018/predict.py --qualify
python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/prediction_guard.json --timeout 5400 -- python3 scripts/trr0018/predict.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/agent4/watchdog.py --cpu-only --receipt experiments/TRR-0018/scoring_guard.json --timeout 300 -- python3 scripts/trr0018/score.py
```

The executable commands use a space after --timeout. Every run is restart-safe at cell receipts: completed arrays must retain their hashes and the complete source/input binding; orphan outputs fail closed. A competing CUDA process causes a guarded stop, not an unrecorded timing comparison.

The fresh sources are public-domain [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342) and [The Adventures of Sherlock Holmes](https://www.gutenberg.org/ebooks/1661), plus generated identifier strings. public_source_downloads.json pins exact fetched text hashes. capture.py samples the registered seed1818256, checks paragraph and token-prefix disjointness, then captures paired targets. No fresh label read is allowed in the prediction process. Scoring verifies the whole1296-cell freeze and rejects missing or changed artifacts before reading labels.

Canonical matrix:58 inherited cells from TRR-0016 and the independent TRR-0017 study, with source hashes, plus2 current mixed-method cells. Inherited runtime is not compared with this run. All current canonical methods use the same disclosed record-batch1 port as TRR-0015; the historical batch8 candidate geometry caveat remains.

Current scientific scripts are immutable after qualification. Delivery and report generation live outside scripts/trr0018, so packaging cannot silently change a bound method.

Inherited TRR-0017 metadata is copied with source hashes and full commit02a2ead7348cd68027024fe1464a148e5ca4da65. Its original code and raw evidence remain in the separate TRR-0017 worktree, whose publication is managed separately. These four canonical cells are inherited evidence, not new executions or a claim that this task published that separate study.

Post-confirmation engineering study (no new candidate-rule selection):
```
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/engineering_qualification_guard_r2.json --timeout 300 -- python3 scripts/trr0018_engineering/benchmark.py --qualify
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 scripts/trr0018/exclusive_watchdog.py --receipt experiments/TRR-0018/engineering_guard.json --timeout 3600 -- python3 scripts/trr0018_engineering/benchmark.py
python3 scripts/trr0018_engineering_delivery.py
```
Each engineering timing cell independently reconstructs its input and only then compares every output tensor with the archived reference. Byte-identical arrays are referenced rather than duplicated; the new per-tensor hashes are archived. This study uses one timed invocation per arm/input and three repeated largest-cell qualification outputs. It occurs after R4 truth opening and supplies execution equivalence and paired timing evidence, not another fresh accuracy result. The first cache-check harness failure remains in attempts.json and its original guard receipt.

All 432 evaluator token arrays are included in this task's fresh-inputs-and-evaluator-labels archive. Previous-task observations are referenced by exact hashes from the published TRR-0016 evidence; the captured 80 R4 observation tensors are included here. No model weights or credentials are included in any archive.

Engineering timing retains the first invocation as well as every later invocation; no separate untimed A2 warm-up is added. The first original-R2 native mixed cell may include first-call startup cost. Fresh-R4 and canonical timing groups occur later in the same process. No rows are removed to claim speed.
