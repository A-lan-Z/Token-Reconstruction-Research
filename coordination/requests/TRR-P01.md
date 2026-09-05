# TRR-P01 — Independent no-fit boundary-prototype pilot

## Objective

Investigate whether token reconstruction can dispense with both a fitted inverse/lens and candidate-by-candidate public-prefix simulation. Another agent is already working on standalone learned decoders, numerical inversion, and evaluation repairs under TRR-0003. Your assignment is a complementary mechanism study, not a continuation that depends on its outputs.

The owner values eliminating offline fitting most directly, while also valuing faster online reconstruction. Use A1+A2 as a comparator, not the architecture to optimise. This round should establish whether the prototype idea deserves further investment; it need not produce a successful replacement.

## Work independently and preserve concurrent work

Repository: `A-lan-Z/Token-Reconstruction-Research`.

Use a separate Git worktree or clone and a new branch `task/TRR-P01`, starting from the published research snapshot:

```text
6b618760f50055dc5c8a62e830ab7a9761190cfe
```

Read `RESEARCH_CHARTER.md`, `AGENTS.md`, and the relevant scientific records. Verify the starting point, but do not depend on unpublished TRR-0003 work. Do not switch branches in the other agent's checkout, modify its environment, overwrite its outputs, or merge/rebase its work. Treat shared model/data assets as read-only.

Save this brief as `coordination/requests/TRR-P01.md`. For this parallel assignment, record task-local state in `coordination/parallel/TRR-P01.json` instead of changing the global `coordination/STATE.json`. Keep changes in task-owned files where practical; defer changes to shared protocols and registries.

Check actual compute availability. With separate GPUs, experiments can run concurrently. On a shared GPU, do not overlap heavy jobs without explicit resource coordination, and take comparative timings without competing workloads. An apparently idle device or free memory is not a reservation. Do not stop the other agent's jobs or incur paid compute charges. CPU implementation and tests can proceed independently of GPU scheduling.

## Main hypothesis: compare tokens in boundary space

The previous checkpoint-only control compared intermediate activations with raw input embeddings. Test a different deterministic construction: pass each vocabulary token through the public prefix after BOS and store its activation at the observation boundary.

```text
b_v = public_prefix([BOS, v])[position 1]
prediction_i = nearest_boundary_prototype(observed_h_i, {b_v})
```

Start with the existing model family and cut 4 for comparability. Choose sensible distance measures and numerical settings; document choices. No lens, auxiliary-data training, fitted calibration, or A2 fallback is permitted in this arm. Table construction is preparation, not fitting: measure its time, storage, and memory rather than calling the method preparation-free.

A small vocabulary subset is acceptable for implementation checks, but meaningful unrestricted reconstruction must search the full declared vocabulary, not a truth-informed shortlist.

## Small experiments that distinguish the failure modes

First establish whether the table can identify tokens at the first post-BOS position under the matched public model. Then test ordinary multi-position public sequences. This separates a construction/implementation problem from sensitivity to context and position.

Next compare matched and shifted targets on identical records, with the target prefix unavailable to reconstruction. Use accessible historical observations or construct an evaluator-only target condition from permitted public resources. Neither obtaining missing historical assets nor waiting for TRR-0003 should block the matched-model pilot.

Use a modest, reproducible development panel spanning more than one input style. Compare with raw-embedding lookup and, where the frozen assets are available, historical A1 alone and a representative fixed A1+A2 comparator on those same inputs. Do not retrain historical baselines. Report unavailable comparisons rather than comparing scores from different datasets as though they were paired.

If static lookup fails mainly away from BOS, test whether a cheap context correction helps. One starting hypothesis is a shared offset estimated with a fixed public reference token:

```text
c_i = public_prefix(reconstructed_prefix_i + [reference_token])[-1] - b_reference
prediction_i = nearest_boundary_prototype(observed_h_i - c_i, {b_v})
```

Use only the reconstructed prefix, never the true prefix. The reference token is a public probe, not a known hidden source token. Count reference evaluations and cache commits. This arm retains some public-prefix computation but must not disguise per-candidate simulation as a correction. You may improve or reject this hypothesis based on evidence; do not build an elaborate system before testing the simple version.

## Evaluation and handoff

Preserve the charter's access and causal-state requirements. Use a small trustworthy runner that validates frozen prediction artifacts before scoring; do not rely on unrepaired evaluation claims or wait for a general infrastructure overhaul. These are exploratory development diagnostics, not fresh-confirmatory or comparison-complete replacement claims. Leave the active-method registry unchanged during screening; successful candidates can undergo the full canonical protocol later.

Record the development records used so later confirmation can exclude them. Distinguish matched-model, input-context, and target-weight effects. Report exact-token accuracy, completely correct records, runtime components, memory, and preparation costs. A low activation distance alone is not a successful reconstruction.

Deliver a first report after the minimum diagnostic panel; optional extensions should not delay it. Write findings to `coordination/results/TRR-P01.md` and structured evidence to `experiments/TRR-P01/manifest.json`, with code, tests, commands, input identities, hashes, and failed attempts. Commit and push only your branch and open a clearly labelled parallel-work PR against its actual parent branch, without merging anything.

End with a decision: does static lookup, context-corrected lookup, or neither warrant another round, and which uncertainty should the next experiment resolve? You have discretion over implementation and pilot scale. Do not duplicate the other agent's decoder-training, numerical-inversion, or broad audit work.
