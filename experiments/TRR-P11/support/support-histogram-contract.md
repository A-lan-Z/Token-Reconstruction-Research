# TRR-P11 public fitting support contract

This contract defines the support diagnostics for the rebuilt fixed-readout
pair. It is preparation evidence only. The producer reads public fitting
payloads and ordered public metadata; it does not open source text, hidden
states, reconstruction predictions, or evaluation truth.

## Bound inputs

The current contract binds the 1,200-row B0 payload and the single 12,000-row
B1 payload by path, byte count, and SHA-256. B1 contains the B0 rows in its
first 1,200 rows and then 10,800 additions. The producer verifies exact values
for `token_ids`, `attention_mask`, and `position_ids` in the prefix, checks the
record bindings (`b0_prefix_row` and `capture_local_row`), and counts the
single B1 file once. It reports B0, all B1 rows, and B1 additions as separate
views. It never appends B0 to all B1 rows.

The common frequency map is the frozen public B0 map from the P07/P09 support
binding. Every bank's position/frequency cell uses this same map, so an
expanded bank cannot make a token look newly supported merely by changing the
frequency reference. Each bank's own sparse frequency counts and TRR-0010
support digest are reported separately as a diagnostic.

The historical compact common payload is recorded but is not a
runtime dependency for recovery. `scripts/trr_p11/recover_common_frequency.py`
can derive the same map directly from the exact B0 payload, opening only
`token_ids` and `attention_mask`. It verifies the expected support digest and
the established plain tensor SHA-256 values for the sorted IDs, sorted counts,
and dense vocabulary vector, then writes all three tensors to a new persistent
P11 safetensors file and records the old path as unused (or missing). A later
histogram run binds the new file and its recovery receipt; a bank-local map is
never used as a fallback.

## Counting semantics

For each row, labels are `token_ids[:, 1:][attention_mask[:, 1:]]`. BOS and
padding are excluded. Positions are one-based offsets after BOS, with the
frozen bins `1-15`, `16-39`, `40-79`, `80-127`, and `128-191`. The producer
keeps all positions through 191 for fitting-support diagnostics. The first
four bins are the 128-token evaluation range; `128-191` is retained as a
post-evaluation diagnostic and is empty from the evaluation adapter's
position range.

The report includes aggregate counts by style/source type, position, common
frequency bin, and their joint cells. Correctness is always reported as
`not_computed` because this producer never sees predictions or labels.

## Deferred execution

Static review may compile the producer and parse this JSON contract. A real
run requires the explicit `--execute` flag and a create-only output. The
registered command is:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /usr/bin/python3 scripts/trr_p11/support_histogram.py --config experiments/TRR-P11/support/support-histogram-contract-r1.json --output experiments/TRR-P11/support/support-histogram-r1.json --execute
```

Before that histogram command, run the bounded recovery command from the
contract. It is public CPU preparation only and has no model, GPU, truth, or
source-text access:

```text
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /usr/bin/python3 scripts/trr_p11/recover_common_frequency.py --config experiments/TRR-P11/support/support-histogram-contract-r1.json --output experiments/TRR-P11/support/public_common_frequency_b0_seed4010.safetensors --receipt experiments/TRR-P11/support/public-frequency-recovery-r1.json --execute
```

The recovery output must pass its own create-only receipt and then be bound in
this contract by its new file hash. Run the histogram only after that binding
and its tensor identities pass. The output is a compact aggregate receipt and
does not replace the frozen fitting-bank manifests.
