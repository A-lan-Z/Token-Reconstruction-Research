# TRR-P11 — new-weight fixed-readout replication and independent evaluation plan

Status: planning only. Agent 2 is currently limited to preparation and
pre-outcome registration: no fitting, GPU execution, source selection,
activation capture, prediction, or evaluation-truth access has occurred in
this worktree. After the shared preflight and frozen-input checks, the
amendment authorizes Agent 1's single paired B0/B1 fit; root controls that
release. Agent 2 does not duplicate that fit.

TRR-P11 is a successor replication of the documented fixed-readout pair. The
exact PR20/TRR-0011 selected states remain unavailable after the bounded
recovery search and remain recorded as blocked/incomplete evidence under
TRR-P10. A rebuilt checkpoint is a new model unless its complete tensor
identity is independently verified; matching names, seeds, configurations,
old scores, or a few predictions do not establish identity.

The incoming amendment is coordination/requests/TRR-P11.md with SHA-256
70a2cf13b2795916749031bcfed3a55205cd52a44926c1d160bd83508706d9b6. This
successor starts at P10 commit cab7d455c305aa01c3d3bfac225fcffa3e66640b and
leaves P10 and PR22 records unchanged.

## Scientific question

After rebuilding one current-bank fixed-readout arm (B0) and one
expanded-bank fixed-readout arm (B1) with the documented fitting procedure,
does the expanded-bank advantage generalize to genuinely unused, source-paired
Pile and Finance records under the two existing public target conditions?

The result tests a new pair and one replication. It does not complete the
original exact-checkpoint confirmation, retroactively make PR20 independent,
or establish seed-independent training reliability.

## Ownership and method boundary

Agent 1 owns exactly one paired B0/B1 fit: one current-bank arm and one
expanded-bank arm from the same declared starting decoder, with the common
training budget, checkpoint grid, and validation-selection rule frozen before
fitting. Agent 1 records all preparation, fitting, selection, checkpoint,
deployment, timing, memory, and failure evidence and stores both selected
states durably. Agent 2 performs the complete exclusion audit, prospective
source-selection gate, independent evaluation preparation, prediction freeze,
restore gate, and post-freeze scoring handoff. Agent 2 does not fit either
arm or duplicate Agent 1's fit. Root coordinates Agent 1 and releases any
shared compute; no paid compute is authorized.

Only the two fixed public-readout arms are rebuilt. Directional-readout arms,
architecture or loss changes, additional data expansion, teacher/context/
solver experiments, and numerical-inversion campaigns are out of scope.
There is no second fit to chase the historical headline. A further fit would
need a separately authorized hypothesis after this replication.

## Required Agent 1 inventory before fitting

Agent 1 must bind, before fitting:

- the common starting decoder and its complete state identity;
- public model, tokenizer, and normalized fixed-readout resources;
- ordered B0 and B1 bank definitions, source revisions, and preprocessing;
- activation-generation cut, geometry, masks, positions, dtype, and numerical
  execution settings;
- the common optimizer/training budget, checkpoint grid, and validation
  selection rule;
- the exact seven-point checkpoint grid `[0, 1000, 2000, 4000, 8000,
  12000, 13000]`; any `validation_every_1000` field is non-grid metadata and
  is recorded as not executed rather than treated as a selection point;
- exact code/dependency identifiers and all persistent output locations.

If any prerequisite cannot be reproduced, the missing dependency and the exact
deviation are recorded. A substitute ancestor is not described as the same
recipe. Development validation and checkpoint selection remain separate from
the new independent evaluation.

## Frozen evaluation contract

The primary panel has 256 records per domain, 128 stored tokens including BOS,
and 127 scored post-BOS positions. Each of the four domain-target cells has
32,512 scored token positions and 256 complete-record units. The cells remain
separate:

- Pile × public_base
- Pile × public_lora_2601
- Finance × public_base
- Finance × public_lora_2601

The two public target conditions are source-paired within each domain. The
expanded and current fixed-readout predictions use the same prospective source
order and target pairing. A common A1+A2 K256 comparator remains in the plan
only on the first 128 records per domain in that same prospective order. Its
denominator is 16,256 scored token positions and 128 complete records per
cell. It is selected before observations, predictions, and truth; no
truth-driven subset is permitted. Agent 2 performs no A1+A2 fit. If the
comparator package is not independently supplied and restorable, mark that
comparator as a technical blocker and report a qualified partial matrix with
its diagnostics unavailable; do not drop the subset or replace the primary
B0/B1 replication based on any newly observed performance.

Agent 1 freezes the new B0/B1 state identities and deployment bindings after
fit and validation selection but before the restore gate. The restore gate
then verifies those exact identities. The old P10 state hashes are preserved
as unavailable historical identities and are never substituted into this new
evaluation.

The evaluator-only `public_lora_2601` resource, when available, is bound by
path and hash before evaluation preparation:
`outputs/TRR-0002/public-calibration/updates/public_lora_2601.safetensors`,
SHA-256 `eea7bb49f801b61df2e26a8f59af7c3096f6f3a2604404e16e589443bcfba595`,
generation SHA-256 `6bed8aca6668bd749bbad03c0c85bbace306085dac3141895dad7685f03d0682`.
It remains evaluator-only and is not a fitting or selection signal.

## Source reservation and exclusions

The registered candidate ranges and deterministic seed are inherited pending a
completed audit:

- seed 5011;
- Pile rows [0, 2,000);
- Finance rows [20,000, 28,000).

No fresh source selection has started. These ranges are reservations, not
capacity certification. Before selection, the exclusion audit must cover all
accessible gradient-fitting and inherited fitting banks, checkpoint-selection
and calibration sources, and every previously opened development/evaluation
source. It must explicitly include the TRR-0009 selection_v2 manifest because
it served as checkpoint-selection validation for PR20. PR20 final sources are
development/selection-overlapping and cannot be reused as an independent test.

The audit compares dataset revision/split/row identity, record identity,
rendered-record SHA-256, canonical H128 sequence SHA-256, and each producer's
published H129/truncated-sequence convention where available. Opaque
hash-only reservations remain wildcard namespaces until their producer
mapping is verified. The unresolved P10 aggregate opened-panel/pointer-link
coverage and unverified P04 fingerprint producer mappings must be resolved or
reported as unavailable before a source-selection release. A field that is unavailable is recorded as unavailable, never silently as zero.

Once the exclusion audit, frozen statistical contract, and B0/B1 bank
identities are complete, root may release metadata-only prospective source
selection while Agent 1's fit and package restoration proceed. Source
selection remains separate from prediction and truth.

A synthetic record copied from TRR-0009 selection_v2 must be rejected before
any new capture. If either registered range cannot yield 256 eligible records
after all exclusions, register a compatible extension before selecting a
replacement. Do not recycle validation examples or silently shrink the
primary panel. Only opaque canonical hashes and reservation metadata may
cross to Agent 1; no source text, labels, predictions, scores, or truth are
exchanged.

## Statistical contract frozen before outcomes

The rules below are fixed before any independent truth is opened. They retain
only the source-paired token bootstrap and the declared exact-record interval;
there is no automatic confidence-interval promotion or harm gate:

- resample complete source records with replacement, using seed 9009 and
  10,000 draws;
- use one shared source-record resample schedule within each domain for both
  target conditions and every compared method;
- report token accuracy and token contrasts with the descriptive central 95%
  percentile interval, using the frozen ascending-order percentile convention;
- report exact-record gain/loss intervals with
  `scripts/trr0010_analysis.py::paired_exact_cp`, route alpha 0.025 and
  component alpha 0.0125, with bootstrap substitution forbidden;
- compute fixed-width token contrasts with
  `scripts/trr0010_analysis.py::bootstrap_token_delta`, one-sided alpha
  0.025 (the central 95% percentile interval).

The statistical implementation binding is the byte-identified historical
TRR-0010 source `scripts/trr0010_analysis.py` from commit
`70c57db7643913eea97cc606775b3f1f3807967a`, SHA-256
`90078ac78bfcdcfb5f782a598c943b78cb0417f6895906a1e6d61056a3793cef`.
Agent 1/root must carry that file byte-identically into the task-local
execution package, or bind an equivalent implementation with a new hash,
before truth is opened. The P11 result will record the final task-local code
hash. Domains, target conditions, methods, and the A1+A2 subset are never
pooled. Intervals describe uncertainty only; no automatic CI or promotion
rule changes the method, source panel, fit, or submitted output.

## Curator, prediction, and truth boundary

The trusted curator may read public rows to construct identity metadata and
target observations. Prediction processes receive only sanitized activation
tensors, masks, positions, frozen new states, the fixed-readout deployment
resources, and approved public metadata. They receive no source payload,
source truth, target-prefix weights, labels, or answer-derived signal.

All source order, candidate generation, adaptation, stopping, timing, and
predictions are frozen before truth. The truth sidecar is materialized
separately and opened once only after every registered prediction and
diagnostic artifact is immutable. P03's sealed holdout remains untouched.

After scoring, report per cell and per method:

- token accuracy and complete-record accuracy;
- both-correct, only-B1-correct, only-A1+A2-correct, and both-wrong
  inventories where the comparator is available;
- position and fitting-support strata;
- true-token rank only when an immutable pre-truth trace preserves it;
- A1 proposal failure separately from A1+A2 direct-ranking failure;
- runtime, memory, preparation, fitting, inference, synchronization, and I/O
  costs.

If a trace is absent, report it unavailable. Do not selectively rerun after
truth to manufacture a missing diagnostic.

## Mandatory restore gate

Before evaluation release, Agent 1 must store the selected state and complete
deployment package in two durable copies on independent failure boundaries.
The proposed second boundary is the Windows backup directory
/mnt/c/Users/alanz/Token-Reconstruction-Backups/TRR-0012, with approximately
1.2 GB minimum package space and at least 10 GB free before copying. The first
copy must be a persistent training/output location; temporary directories and
symlinks do not qualify.

The gate is closed until all of the following pass:

1. both copies contain the actual selected files and required deployment
   resources;
2. each file has a recorded size and SHA-256, and each tensor has recorded
   names, shapes, dtypes, and content identities;
3. the two copies are independently retrievable across their failure
   boundaries;
4. from a clean working directory outside the training checkout and temporary
   runtime paths, Agent 2 retrieves the package and loads it successfully;
5. the exact retrieved tensors match the selected-file identities and the
   deployment configuration binds to the new state IDs;
6. a fixed, already-opened public smoke test runs with the retrieved package
   and reproduces its predeclared output identity. The smoke fixture is the
   first two already-opened TRR-0010 public_base records in each domain, in
   Finance-then-Pile order, evaluated for the two new fixed models; record
   IDs, order, H, mask, and position hashes are frozen before fitting. Smoke
   predictions are produced after normal checkpoint selection and are not a
   selection criterion.

A manifest, pointer, old replay, or successful training-directory load is not
a substitute for this gate. Metadata-only source inventory and prospective
source selection may proceed after the exclusion, frozen-plan, and bank-identity
requirements pass; prediction, scoring, and truth release remain blocked until
all restore-gate items pass.

## Interpretation and release dependencies

A new result is classified as a reproduced gain, smaller effect, regression,
or imprecise evidence. It is not automatically attributed to selection bias or
training variation. The old PR20 scores remain selection-overlapping
development evidence. The parallel transfer study resumes only after the new
package is frozen and passes this restore gate under its separate approved
plan.

Current release dependencies are:

- complete exclusion coverage and producer-fingerprint mappings;
- Agent 1's single paired B0/B1 fit and durable state identities;
- the frozen scoring/code manifest;
- the two-copy restore gate and clean-workspace public smoke test;
- root's explicit release for source selection, capture, prediction, and truth.

