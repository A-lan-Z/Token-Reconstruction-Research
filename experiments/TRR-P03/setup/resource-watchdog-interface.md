# TRR-P03 resource watchdog interface

`scripts/trr_p03/resource_watchdog.py` wraps one bounded CPU command. Wrapper options come before a literal `--`; the child command is copied verbatim after it:

```text
python3 scripts/trr_p03/resource_watchdog.py \
  --output-root experiments/TRR-P03/runtime/watchdog-run \
  --timeout-seconds 1800 \
  --poll-seconds 0.5 \
  --max-rss-bytes 8589934592 \
  --min-available-bytes 10737418240 \
  -- python3 scripts/trr_p03/generate_observations.py ...
```

The wrapper starts the child with `start_new_session=True`, captures `stdout.txt` and `stderr.txt`, and samples the whole process group. It terminates the group with `SIGTERM` followed by `SIGKILL` when group RSS exceeds 8 GiB, host `MemAvailable` falls below 10 GiB, live `/proc` resource data become unreadable, or the declared wall timeout expires. It waits for descendants to leave the process group before returning. A clean child exit returns the child code; a timeout returns 124; a fail-closed guard event returns 125.

Each invocation uses a create-only output directory and writes `command.json`, `resource_samples.jsonl`, `resource_guard.json`, `time.json`, `stdout.txt`, `stderr.txt`, and `finish.json`. The command receipt preserves the exact child argument vector and cwd plus an explicit safe environment allowlist (`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM`, `PYTHONPATH`, `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, and `TRANSFORMERS_OFFLINE`). The child still inherits the full execution environment, but unrelated variables are redacted from receipts. The guard and time receipts preserve thresholds, periodic RSS/availability samples, start/end timestamps, child/wrapper exit codes, and any termination reason. No model or truth is accessed by the wrapper.

Smoke evidence: `watchdog-smoke-positive-v2` passed with one tiny child command; `watchdog-smoke-limit-v2` failed closed on the forced 1-byte RSS limit; and `watchdog-smoke-host-limit` failed closed before launch on an intentionally impossible host-availability floor; `watchdog-smoke-timeout` returned the declared-timeout code 124. The initial smoke command receipts briefly captured the inherited environment; they were sanitized in place and are retained only as untracked development evidence. These are utility checks only and are not scientific runs.
