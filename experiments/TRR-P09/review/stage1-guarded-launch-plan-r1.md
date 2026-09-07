# TRR-P09 guarded launch plan

This is an execution template only. It contains no lease timestamp, root
authorization, model run, or capture result.

The executable helper is `scripts/trr_p09/stage1_guarded_launch.py`. Without
`--execute` it prints the exact child and external-watchdog argv, the approved
watchdog hash, and the enforcement map. With `--execute` it requires the
supplied watchdog receipt's command to match that argv exactly, requires both
receipt files to exist, and dispatches the existing P06 process-group watchdog.
The helper never creates or edits either receipt.

Qualification dry-run/launch shape:

```text
PYTHONPATH=src:. python3 scripts/trr_p09/stage1_guarded_launch.py \
  --mode qualify \
  --input-manifest /tmp/trr-p09-runtime/stage1-input-preparation-r1/preparation_manifest.json \
  --stage1-plan experiments/TRR-P09/planning/stage1-public-bank-plan.json \
  --output-root /tmp/trr-p09-runtime/stage1-qualification-r1 \
  --watchdog-output-root /tmp/trr-p09-runtime/watchdog-stage1-qualification-r1 \
  --model-snapshot /home/alanz/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --authorization-receipt /tmp/trr-p09-runtime/authorization-stage1-qualification-r1.json \
  --watchdog-receipt /tmp/trr-p09-runtime/watchdog-stage1-qualification-r1.json
```

Root supplies `--cuda-visible-devices` if a device-specific environment is
needed, reviews the printed command, fills the qualification template, and
adds `--execute` only after the live resource lease is signed. The child
performs the P09 prelaunch GPU/host/disk checks before model load.

Capture uses the same command with `--mode capture`, distinct create-only
output roots and receipts, and the 7,200-second cap. Capture authorization
must bind the passing qualification receipt and a finite production
extrapolation before the child may load the model.

The qualification receipt emits `elapsed_seconds` for its measured
qualification forwards and a bounded `batches` list. Each selected batch has
three forward calls: original, exact repeat, and future-padding variant. The
helper's estimate mode computes:

```text
qualification_forward_calls = 3 * len(qualification.batches)
scaled_forward_seconds = qualification.elapsed_seconds \
    * (1350 / qualification_forward_calls)
estimated_wall_seconds = scaled_forward_seconds \
    + root_supplied_fixed_overhead_seconds
```

The fixed overhead must be supplied from root evidence and may cover model
load, input validation, streamed serialization/hashing, synchronization, and
cleanup. The helper rejects non-finite values and reports whether the estimate
is within 7,200 seconds; it does not write the authorization receipt.

The external wrapper enforces only process-group wall timeout, aggregate child
RSS, host `MemAvailable`, and post-child race-safe evidence. It does **not**
measure GPU memory or retained output size. The P09 child `ResourceGuard`
enforces GPU reserved/free floors, prelaunch host floor, disk free, and the
20-GiB retained-output cap. The wrapper's runtime host floor is 6 GiB; the
child's prelaunch floor remains 10 GiB. This separation is recorded in the
launch metadata and both receipt templates.

Templates:

- `experiments/TRR-P09/review/stage1-watchdog-qualification-template.json`
- `experiments/TRR-P09/review/stage1-watchdog-capture-template.json`

Both templates are explicitly non-runnable until root replaces every
`<ROOT_FILL_...>` value and binds the exact helper-produced command.

The real-path dry-run argv evidence (no child launch) is preserved in:

- `experiments/TRR-P09/review/stage1-qualification-dry-run.json`, SHA-256 `d3816ec2cf31f0e5e22cb36d7ac14f63325569285ea45ccea9b9db4587a26a29`;
- `experiments/TRR-P09/review/stage1-capture-dry-run.json`, SHA-256 `ea37324624a1e8a5ff6bfa4d608219d09076864060f157501027255d059592db`.

The helper source SHA-256 is
`1b9f29e04b75e7147d448927f99fc60c8042b9374b16e8a37a59fe97efef2adc`, and
its focused test source SHA-256 is
`5e40a0aef0bbce093ba3482b260cf9475346ee5162b2b95441701aae29c5b3c5`.
