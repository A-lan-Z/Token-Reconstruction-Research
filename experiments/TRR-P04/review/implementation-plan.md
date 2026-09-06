# TRR-P04 implementation plan

This implementation keeps the four registered arms on one public data path. It
loads the PR7 replay, the fixed 256-record public correction pool, and the fixed
192-record public validation pool. The pool loader validates activation and
right-padding geometry before labels, rejects record overlap, and records file
hashes and record-order hashes. No evaluator observation, target-update weight,
or private truth is in this path.

The model is a one-layer width-256 unidirectional GRU over fixed per-position
layer-normalized boundary activations, added to a trainable full hidden-space
affine path. The affine path starts from the public PR7 `W,b,s` checkpoint when
provided, but `W`, `b`, and `s` remain trainable in the affine reference and all
S/H/D arms. The GRU output projection is zero initialized, which makes the
initial recurrent output equal to the same affine function. All deployed arms
use only activation prefixes and a fixed normalized public embedding table for
unrestricted full-vocabulary inference.

For every seed, the schedule samples six replay and two correction records per
update, selects at most 512 post-BOS positions, and retains any qualified
teacher positions in the correction rows. The resulting record indices and
position masks are serialized before fitting and reused byte-for-byte by the
affine, S, H, and D arms. AdamW, learning rate `1e-3`, zero weight decay,
gradient clipping at 1, and 3,000 updates are fixed. Public validation is
checked at step zero and every configured interval; the best style-balanced
checkpoint uses strict improvement, so a regression retains step zero.

Candidate IDs are produced once by the preparation CLI from the frozen
PR7 public affine/table resource. It stores the deterministic A1 top-512 rows,
the K=32 prefix, and the pool/asset hashes in one create-only artifact. The
teacher consumes those exact rows by record ID, and the training runner
requires the same artifact for H/D; no component regenerates candidates.
H adds the fixed label-only
hard-confusion term with margin 1 and weight 0.25. D adds weighted adjacent
non-gold pairwise ranking from the frozen 384-row public teacher evidence with
weight 0.25. Equal and near-equal teacher score gaps are omitted; the robust
median nonzero gap and tie tolerance are recorded in the teacher artifact.
Teacher evidence is privileged public-prefix qualification only and is never
read by prediction/evaluation code.

The largest-cell qualifier exercises batch 8, sequence 192, hidden 2048, GRU
width 256, 512 selected rows, and the full 128,256-token output on forward and
backward passes with Adam state. The FP32 table is about 1.05 GB, a 512-row
logit tensor about 251 MiB, and a batch sequence activation about 12 MiB; the
conservative expected reserved-GPU range is 2--6 GiB, with a 6 GiB reserved and
16 GiB host-RSS guard. A qualifier receipt reports step-zero affine errors,
probe losses/accuracies, post-probe errors, exact records, timing, and peak
memory before the full matrix is released.

Source entry points:

- `scripts/trr0004_p04_prepare_candidates.py` freezes the shared PR7-affine
  candidate table before teacher qualification or H/D fitting.
- `scripts/trr0004_p04_qualify.py` performs the largest-cell public correction
  probe and capacity check.
- `scripts/trr0004_p04_teacher.py` freezes the bounded privileged teacher
  evidence and diagnostics.
- `scripts/trr0004_p04_train.py` creates the common candidate/schedule assets,
  trains paired seeds, and writes selected/final state receipts.

All outputs are create-only. A failure leaves its attempted receipt and does
not overwrite a prediction or state. The parent/root task supplies the exact
setup artifact paths, source commit, resource reservation, evaluator prediction
freeze, and final report.
