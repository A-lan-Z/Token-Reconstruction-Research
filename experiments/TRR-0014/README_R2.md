# Reproducing the selected prefix-native fragment mechanism

Read FRAGMENT_CONFIRMATION_PLAN.md, ACCESS_AND_COST.md and the final coordination/results/TRR-0014.md before interpreting scores. The original README.md documents R1 and remains valid for that earlier metric study.

Environment: Ubuntu/WSL, Python3.12.3, torch2.10.0+cu128, transformers5.3.0, RTX5080, two CPU threads. Prefix Llama3.2-1B-Instruct revision9213176726f574b556790deb65791e0c5aa438b6, cut4, BF16 and native FP32 rotary constants. Explicit sibling asset paths and checkpoint SHA are in native.py and capture receipts. Never substitute missing assets silently.

From a fresh reproduction worktree with the declared assets available:
1. Set PYTHONPATH=src, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2.
2. Run python3 -m pytest -q tests/test_prefix_weight_metric.py tests/test_prefix_fragments.py.
3. Read the preflight geometry; run python3 scripts/agent4/watchdog.py --receipt <new-receipt> --timeout180 -- python3 scripts/trr0014/fragment_predict.py --qualify. Keep argument and number separate when entering the shell.
4. The original failed qualifier is preserved. The final qualifier compares raw intrinsic256/4096 batches in bounded buffers, excludes4096 because outputs differ, verifies production versus frozen development candidates, and qualifies all three full128-position arms throughK512.
5. After qualification, evaluator runs capture_r2.py under the watchdog. This creates80 paired observations from40 task-local new sources, with R1 excluded by hash. All destinations are create-only; never overwrite original evidence.
6. Prediction runs fragment_predict.py under the watchdog with timeout900. It freezes240 distinct cells and three deterministic timing repetitions percell. Neither capture_r2.py nor its target adapter is imported by prediction.
7. Only after the prediction receipt exists, run score_fragment_confirmation.py. It checks the full matrix, hashes, ID ranges, finite values, declared argmax decisions, timing repetitions and negative gate cases before opening labels.
8. package_evidence.py can run only after the final score exists. Extract its ZIPs at repository root to restore raw artifacts under their original outputs/TRR-0014 paths. artifact_index.json records every archive and member hash. Check hashes before rescoring.

Exact commands, code commits, runtime/dependency IDs, hardware, timestamps, resource guards, costs, output paths and hashes are collected in manifest.json and its referenced receipts. Model weights, the evaluator-only target adapter and large generated vocabulary tables are not duplicated in the evidence ZIPs. Their identities and deterministic construction are explicit. Post-score panel labels are public-domain/generated material; they are now opened evidence and cannot be reused as fresh confirmation.

The selected policy is frozen at512 logical candidate slots. A follow-up that deduplicates native verification, changes numerical batch geometry, changes suffix ordering/budget, fits a decoder, or updates the prefix mid-record is a different method and needs its own equivalence/access checks and evaluation.
