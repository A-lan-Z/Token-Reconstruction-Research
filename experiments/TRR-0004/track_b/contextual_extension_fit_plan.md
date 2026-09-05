# TRR-0004 contextual extension fit plan

This is a source-only registration and invocation plan. No contextual fit has run
from this worktree. It consumes the freshly selected Track-A historical-style
affine state and the footing public activation artifacts.

The preferred input interface is footing's combined safetensors format. Each
combined file contains `activations`, `token_ids`, `attention_mask`,
`position_ids`, and the two post-BOS selector tensors. The fit uses the registered
1200-record public split. Validation is two explicit, disjoint groups: 24
Alpaca records at native length 192 and 24 Pile records at native length 40.
The runner checks all record IDs against the supplied metadata-only combined
registration before reading `token_ids`; it preserves each native geometry and
right-pads the validation parts to one masked causal pass only when lengths
differ. The adapter's active-position output equivalence is covered by the
focused mixed-geometry test. The runner requires the fresh base state to contain
exactly `W`, `b`, and scalar `s`; the public vocabulary bias is excluded.

The planned invocation, with paths supplied by the Track-A/footing preparation,
is:

```text
PYTHONPATH=src:scripts .venv-trr0004/bin/python scripts/trr0004_fit_contextual_extensions.py \
  --base-state /home/alanz/spartan/punim2939/Token-Reconstruction-Research/.worktrees/TRR-0004/outputs/TRR-0004/fit_large_v1/historical_affine_ce_no_vocab_bias.safetensors \
  --fit-artifact <prepared-fit-combined.safetensors> \
  --fit-records <prepared-fit-records.json> \
  --validation-artifact <prepared-alpaca-validation-combined.safetensors> \
  --validation-artifact <prepared-pile-validation-combined.safetensors> \
  --validation-records <prepared-alpaca-validation-records.json> \
  --validation-records <prepared-pile-validation-records.json> \
  --validation-groups <prepared-validation-groups.json> \
  --embedding-table <prepared-normalized-embedding-table.safetensors> \
  --registration <combined-48-record-registration.json> \
  --output-root experiments/TRR-0004/track_b/contextual_fit_v2 \
  --steps 3000 --subset-steps 600 --subset-records 8 \
  --record-batch-size 8 --position-budget 512 \
  --validation-every 100 --learning-rate 1e-3 --weight-decay 0 \
  --gradient-clip-norm 1 --seed 1737 --device cuda \
  --minimum-free-gib 8 --maximum-gpu-reserved-gib 6 \
  --maximum-host-rss-gib 16 --max-seconds 1200
```

The runner records the common early checkpoints `0,25,50,75,100,150,200` and
the 100-step grid through the final step. Step 0 is retained as the exact-base
diagnostic. A selected deployment state must be the earliest best **nonzero**
checkpoint by style-balanced public validation token accuracy, so the contextual
arm cannot silently fall back to the frozen base. The eight-record, 600-step
runs are fit-subset overfit diagnostics and are not selection inputs.

Both methods use the same CPU-generated record/position schedule, batch size 8,
and at most 512 total valid post-BOS loss positions per batch (not 512 per
record). Causal attention sees
only `H_0...H_i` under a causal and padding mask; the MLP sees the same
fixed, nonlearned per-position layer normalization. Both added paths are
zero-output initialized around the identical frozen `W,b,s` base. Full-vocabulary
logits are projected only for selected rows during fitting and validation; no
public-prefix call, teacher input, candidate simulation, or A2 fallback exists.

For the planned geometry `(V,H,B,T,K)=(128256,2048,8,192,512)`, the fixed
embedding table is about 1,050,673,152 tensor bytes (0.979 GiB), one selected
FP32 vocabulary logits block is 262,668,288 bytes (0.245 GiB), and one FP32
activation batch is 12,582,912 bytes (0.012 GiB). The runner's conservative
preflight envelope is about 2.075 GiB before allocator overhead and requires 8
GiB of observed free device memory. This is a planning estimate, not measured
qualification evidence; the largest public geometry must pass a guarded
qualification before the 3,000-step arms. Existing TRR-0003 600-step direct
CE fits took approximately 21 seconds per arm at the smaller public panel (the
1,800-step arm took about 62 seconds); the contextual wall time is intentionally
left as an empirical qualification result because it adds the full sequence pass
and repeated validation.

Expected outputs are `position_schedule.safetensors`, one main curve and selected
state per method, one subset curve and final subset state per method, and
`run_evidence.json` with input/state/source hashes, per-phase timing, validation
projection counts, and peak memory. A failed output directory is preserved with
`failure.json` and is excluded from scientific scoring.
